"""CPU schwa /ə/ detection: download, make-dataset, train, evaluate, predict, denoise."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter
from scipy.signal import istft, resample_poly, stft
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, roc_auc_score
from torch import nn

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'validation', 'test')
SPEAKERS = {'train': ['bdl', 'clb', 'rms'], 'validation': ['slt'], 'test': ['jmk']}
BASE_URL = 'http://festvox.org/cmu_arctic/cmu_arctic/'
AUDIO = {'sample_rate': 16000, 'window_seconds': 0.5}
FEATURES = {'sample_rate': 16000, 'n_mels': 32, 'n_fft': 512, 'win_length': 400,
            'hop_length': 160, 'fmin': 20, 'fmax': 8000, 'center': False, 'epsilon': 1e-10}
MODEL = {'channels': [16, 32, 64], 'hidden_size': 32, 'dropout': 0.2,
         'pooling': 'preserve_frequency', 'pooled_frequency_bins': 4}
DENOISE = {'strength': 0.6, 'minimum_gain': 0.35, 'quiet_fraction': 0.2,
           'maximum_quiet_to_loud_ratio': 0.25}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def seeded(seed, identity):
    number = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'little')
    return np.random.default_rng(number)


def pcm_hash(audio):
    pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype('<i2')
    return hashlib.sha256(b'mono-16000-pcm16-v1:' + pcm.tobytes()).hexdigest()


def load_audio(path, sample_rate=16000):
    values, rate = sf.read(path, dtype='float32', always_2d=True)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Empty or invalid audio: {path}')
    metadata = {'duration_seconds': len(values) / rate, 'original_sample_rate': rate,
                'original_channels': values.shape[1]}
    mono = values.mean(axis=1, dtype=np.float32)
    if rate != sample_rate:
        factor = math.gcd(rate, sample_rate)
        mono = resample_poly(mono, sample_rate // factor, rate // factor).astype(np.float32)
    if not np.isfinite(mono).all():
        raise ValueError(f'Invalid resampled audio: {path}')
    return np.ascontiguousarray(mono), metadata


def crop_window(audio, start, samples=8000):
    output = np.zeros(samples, dtype=np.float32)
    left, right = max(0, start), min(len(audio), start + samples)
    if right > left:
        output[left - start:right - start] = audio[left:right]
    return output


def denoise_audio(audio, config=None):
    config = config or DENOISE
    if len(audio) < 512 or not np.any(audio):
        return audio.copy(), 'skipped_short_or_silent'
    _, _, spectrum = stft(audio, fs=16000, nperseg=400, noverlap=240, nfft=512,
                          boundary='zeros', padded=True)
    magnitude = np.abs(spectrum)
    energy = np.sqrt(np.mean(magnitude ** 2, axis=0))
    count = max(3, round(len(energy) * config['quiet_fraction']))
    if len(energy) < count:
        return audio.copy(), 'skipped_insufficient_frames'
    quiet = np.argsort(energy)[:count]
    loud = float(np.percentile(energy, 90))
    # No reliable low-energy region: avoid treating weak speech as a noise profile.
    if loud <= 1e-10 or float(np.median(energy[quiet])) > loud * config['maximum_quiet_to_loud_ratio']:
        return audio.copy(), 'skipped_no_reliable_background'
    noise = np.median(magnitude[:, quiet], axis=1, keepdims=True)
    gain = 1 - config['strength'] * noise / np.maximum(magnitude, 1e-10)
    gain = uniform_filter(np.clip(gain, config['minimum_gain'], 1.0), size=(3, 3), mode='nearest')
    _, filtered = istft(spectrum * gain, fs=16000, nperseg=400, noverlap=240,
                        nfft=512, input_onesided=True, boundary=True)
    output = np.asarray(filtered[:len(audio)], dtype=np.float32)
    if len(output) < len(audio):
        output = np.pad(output, (0, len(audio) - len(output)))
    if not np.isfinite(output).all():
        raise ValueError('Denoising produced non-finite audio.')
    return output, 'filtered'


def parse_phones(text, duration):
    phones, previous, zero_duration = [], 0.0, 0
    for line in text.splitlines():
        tokens = line.split()
        if not tokens or tokens[0] == '#':
            continue
        end = float(tokens[0])
        if not math.isfinite(end) or end < previous or end > duration + 0.025:
            raise ValueError(f'Invalid phone boundary: {line!r}, duration={duration}')
        end = min(end, duration)
        if end > previous:
            phones.append({'start': previous, 'end': end, 'phone': tokens[-1].lower()})
        else:
            zero_duration += 1
        previous = end
    if not phones:
        raise ValueError('No valid phone annotations.')
    return phones, zero_duration


def fetch(url, destination, timeout=20, retries=3):
    destination = Path(destination)
    marker = destination.with_suffix(destination.suffix + '.sha256.json')
    if destination.is_file() and marker.is_file():
        recorded = read_json(marker)
        if digest(destination) != recorded['sha256'] or recorded['url'] != url:
            raise ValueError(f'Existing source checksum/URL differs: {destination}')
        return recorded
    if destination.exists():
        raise ValueError(f'Unverified source already exists: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'schwa-cpu-research/1.0'})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open('wb') as output:
                expected_length = response.headers.get('Content-Length')
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
            if temporary.stat().st_size == 0 or (expected_length and temporary.stat().st_size != int(expected_length)):
                raise ValueError('Empty/truncated download.')
            if destination.suffix == '.wav':
                sf.info(temporary)
            temporary.replace(destination)
            recorded = {'url': url, 'sha256': digest(destination), 'bytes': destination.stat().st_size,
                        'downloaded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
            write_json(marker, recorded)
            return recorded
        except Exception as caught:
            error = caught
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(min(2 ** attempt, 4))
    raise RuntimeError(f'Download failed: {url}: {error}')


def download(args):
    counts = {'train': args.train_prompts, 'validation': args.validation_prompts, 'test': args.test_prompts}
    if any(n < 1 for n in counts.values()) or sum(counts.values()) > 593:
        raise ValueError('Prompt counts must be positive and their sum <= 593.')
    selection = {'seed': args.seed, 'counts': counts, 'speakers': SPEAKERS}
    existing = ROOT / 'data/manifests/source_selection.json'
    if existing.exists() and read_json(existing) != selection:
        raise ValueError('Source selection differs from the existing dataset; use a separate project copy for another selection.')
    write_json(existing, selection)
    prompt_ids = [f'arctic_a{i:04d}' for i in range(1, 594)]
    random.Random(args.seed).shuffle(prompt_ids)
    planned, offset = [], 0
    for split in SPLITS:
        selected = prompt_ids[offset:offset + counts[split]]
        offset += counts[split]
        for speaker in SPEAKERS[split]:
            base = BASE_URL + f'cmu_us_{speaker}_arctic/'
            for name in ('COPYING', 'README'):
                fetch(base + name, ROOT / 'data/raw' / speaker / name, args.timeout, args.retries)
            for prompt in selected:
                planned.append({'split': split, 'speaker_id': speaker, 'prompt_id': prompt,
                                'recording_id': speaker + '_' + prompt,
                                'audio_path': f'data/raw/{speaker}/{prompt}.wav',
                                'lab_path': f'data/raw/{speaker}/{prompt}.lab', 'base_url': base})

    def get_recording(row):
        metadata = dict(row)
        for kind, folder in (('audio', 'wav'), ('lab', 'lab')):
            name = row['prompt_id'] + ('.wav' if kind == 'audio' else '.lab')
            result = fetch(row['base_url'] + folder + '/' + name, ROOT / row[kind + '_path'], args.timeout, args.retries)
            metadata[kind + '_sha256'] = result['sha256']
        return metadata

    ready, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(get_recording, row): row for row in planned}
        for index, future in enumerate(as_completed(pending), 1):
            try:
                ready.append(future.result())
            except Exception as error:
                failed.append({'recording_id': pending[future]['recording_id'], 'error': str(error)})
            if index % 25 == 0 or index == len(planned):
                print(f'Download {index}/{len(planned)}, failures={len(failed)}', flush=True)
    ready.sort(key=lambda row: row['recording_id'])
    write_csv(ROOT / 'data/manifests/recordings.csv', ready)
    write_json(ROOT / 'reports/download.json', {'planned': len(planned), 'ready': len(ready), 'failed': failed})
    if failed:
        raise RuntimeError(f'{len(failed)} recordings failed; rerun download to resume.')
    print(f'Sources ready: {len(ready)} real recordings from 5 speakers.', flush=True)


def balanced_negatives(rows, count, rng):
    pools = defaultdict(list)
    for row in rows:
        pools[row['center_phone']].append(row)
    for values in pools.values():
        rng.shuffle(values)
    selected = []
    keys = sorted(pools)
    while len(selected) < count and keys:
        for key in keys[:]:
            if pools[key]:
                selected.append(pools[key].pop())
                if len(selected) == count:
                    break
            else:
                keys.remove(key)
    return selected


def make_dataset(args):
    manifest = ROOT / 'data/manifests/dataset.csv'
    settings = {'seed': args.seed, 'audio': AUDIO, 'target_phones': ['ax'], 'positive_variants': 2,
                'selection': read_json(ROOT / 'data/manifests/source_selection.json')}
    if manifest.exists():
        metadata = read_json(ROOT / 'data/manifests/dataset_metadata.json')
        if metadata['settings'] != settings or metadata['source_manifest_sha256'] != digest(ROOT / 'data/manifests/recordings.csv'):
            raise ValueError('Dataset settings/source manifest changed; preserving frozen dataset.')
        audit_dataset()
        print('Existing dataset reused and audited.', flush=True)
        return
    recordings = read_csv(ROOT / 'data/manifests/recordings.csv')
    pools = {split: {0: [], 1: []} for split in SPLITS}
    excluded, excluded_recordings, source_rows = [], [], []
    sr, samples = 16000, 8000
    for rec_index, recording in enumerate(recordings, 1):
        for kind in ('audio', 'lab'):
            if digest(ROOT / recording[kind + '_path']) != recording[kind + '_sha256']:
                raise ValueError(f"Changed {kind} source: {recording['recording_id']}")
        candidates = []
        try:
            audio, meta = load_audio(ROOT / recording['audio_path'])
            phones, zero = parse_phones((ROOT / recording['lab_path']).read_text(), len(audio) / sr)
            targets = [phone for phone in phones if phone['phone'] == 'ax']
            unknown = [phone for phone in phones if phone['phone'] == 'unk']
            source_rows.append({**recording, 'source_pcm_sha256': pcm_hash(audio),
                                **meta, 'zero_duration_phones': zero,
                                'target_count': len(targets)})
            for index, phone in enumerate(phones):
                if phone['phone'] == 'unk':
                    continue
                if phone['end'] - phone['start'] < 0.02:
                    continue
                positive = phone['phone'] == 'ax'
                if positive and phone['end'] - phone['start'] > AUDIO['window_seconds'] - 0.04:
                    excluded.append({'recording_id': recording['recording_id'], 'phone_index': index, 'reason': 'target_too_long'})
                    continue
                for variant in range(2 if positive else 1):
                    identity = f"{recording['recording_id']}_{index:03d}_{variant}"
                    rng = seeded(args.seed, identity)
                    start = round((phone['start'] + phone['end']) / 2 * sr) - samples // 2
                    if positive and variant == 1:
                        low = math.ceil(phone['end'] * sr - samples + 320)
                        high = math.floor(phone['start'] * sr - 320)
                        start = int(rng.integers(low, high + 1))
                    end = start + samples
                    if any(start / sr < p['end'] and end / sr > p['start'] for p in unknown):
                        excluded.append({'recording_id': recording['recording_id'], 'phone_index': index,
                                         'reason': 'window_overlaps_unknown_annotation'})
                        continue
                    if not positive and any(start / sr < target['end'] and end / sr > target['start'] for target in targets):
                        continue
                    window = crop_window(audio, start, samples)
                    if float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) < 1e-5:
                        continue
                    candidates.append({
                        'sample_id': identity, 'recording_id': recording['recording_id'],
                        'speaker_id': recording['speaker_id'], 'prompt_id': recording['prompt_id'],
                        'split': recording['split'], 'label': int(positive), 'center_phone': phone['phone'],
                        'phone_start': phone['start'], 'phone_end': phone['end'],
                        'window_start_sample': start, 'window_end_sample': end,
                        'source_path': recording['audio_path'], 'source_sha256': recording['audio_sha256'],
                        'source_pcm_sha256': source_rows[-1]['source_pcm_sha256'], 'seed': args.seed})
            for candidate in candidates:
                pools[recording['split']][candidate['label']].append(candidate)
        except (OSError, ValueError, RuntimeError) as error:
            excluded_recordings.append({'recording_id': recording['recording_id'], 'reason': str(error)})
        if rec_index % 100 == 0 or rec_index == len(recordings):
            print(f'Phone QC {rec_index}/{len(recordings)}', flush=True)

    selected = []
    for split in SPLITS:
        positive, negative = pools[split][1], pools[split][0]
        if not positive or not negative:
            raise ValueError(f'{split} does not contain both labels.')
        selected.extend(positive)
        # Train balanced; evaluation retains all eligible speech negatives.
        selected.extend(balanced_negatives(negative, min(len(positive), len(negative)), seeded(args.seed, split)) if split == 'train' else negative)
    selected.sort(key=lambda row: row['sample_id'])
    cache_path, cache_audio = None, None
    for index, row in enumerate(selected, 1):
        if cache_path != row['source_path']:
            cache_path = row['source_path']
            cache_audio, _ = load_audio(ROOT / cache_path)
        audio = crop_window(cache_audio, int(row['window_start_sample']))
        output = ROOT / 'data/generated' / row['split'] / (row['sample_id'] + '.wav')
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, 16000, subtype='PCM_16')
        row['path'] = output.relative_to(ROOT).as_posix()
        row['audio_sha256'] = digest(output)
        row['pcm_sha256'] = pcm_hash(load_audio(output)[0])
        if index % 1000 == 0 or index == len(selected):
            print(f'WAV generation {index}/{len(selected)}', flush=True)
    write_csv(manifest, selected)
    write_csv(ROOT / 'data/manifests/source_qc.csv', source_rows)
    for split in SPLITS:
        write_csv(ROOT / f'data/manifests/{split}.csv', [row for row in selected if row['split'] == split])
    write_csv(ROOT / 'data/manifests/listening_checklist.csv',
              [{**row, 'review_status': 'unreviewed'} for row in selected[::max(1, len(selected) // 100)]])
    write_json(ROOT / 'data/manifests/dataset_metadata.json', {
        'settings': settings, 'source_manifest_sha256': digest(ROOT / 'data/manifests/recordings.csv'),
        'manifest_sha256': digest(manifest), 'candidate_counts': {s: {str(label): len(pools[s][label]) for label in (0, 1)} for s in SPLITS},
        'selected_counts': dict(Counter(row['split'] for row in selected)), 'excluded': excluded,
        'excluded_recordings': excluded_recordings,
        'labels': settings['selection'].get('annotation_origin',
            'Presence of auto-aligned CMU ARCTIC ax within 0.5-second windows; no human review completed.')})
    audit_dataset()


def audit_dataset(verify_wavs=True):
    manifest = ROOT / 'data/manifests/dataset.csv'
    metadata = read_json(ROOT / 'data/manifests/dataset_metadata.json')
    if digest(manifest) != metadata['manifest_sha256']:
        raise ValueError('Frozen dataset manifest changed.')
    if digest(ROOT / 'data/manifests/recordings.csv') != metadata['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed.')
    if not verify_wavs:
        if not (ROOT / 'reports/dataset_audit.json').exists():
            raise ValueError('A full WAV audit is required before using verified feature caches.')
        previous = read_json(ROOT / 'reports/dataset_audit.json')
        if not previous.get('passed') or previous.get('manifest_sha256') != metadata['manifest_sha256']:
            raise ValueError('A full WAV audit is required before using verified feature caches.')
    rows = read_csv(manifest)
    counts, overlaps, errors = {}, {}, []
    for field in ('speaker_id', 'prompt_id', 'recording_id', 'source_pcm_sha256', 'pcm_sha256'):
        seen = defaultdict(set)
        for row in rows:
            seen[row[field]].add(row['split'])
        collisions = [key for key, splits in seen.items() if len(splits) > 1]
        overlaps[field] = collisions[:10]
        if collisions:
            errors.append(f'{field}: {len(collisions)} cross-split collisions')
    target_intervals, unknown_intervals = {}, {}
    used_recordings = {row['recording_id'] for row in rows}
    quarantined = {r['recording_id'] for r in metadata.get('excluded_recordings', [])}
    if used_recordings & quarantined:
        errors.append('Quarantined recordings appear in the generated dataset.')
    for rec in read_csv(ROOT / 'data/manifests/recordings.csv'):
        if rec.get('manual_path') and digest(ROOT / rec['manual_path']) != rec['manual_sha256']:
            errors.append(f"Manual annotations changed: {rec['recording_id']}")
        if digest(ROOT / rec['lab_path']) != rec['lab_sha256'] or digest(ROOT / rec['audio_path']) != rec['audio_sha256']:
            errors.append(f"Source changed: {rec['recording_id']}")
        if rec['recording_id'] not in used_recordings:
            continue
        duration = sf.info(ROOT / rec['audio_path']).duration
        phones, _ = parse_phones((ROOT / rec['lab_path']).read_text(), duration)
        target_intervals[rec['recording_id']] = [p for p in phones if p['phone'] == 'ax']
        unknown_intervals[rec['recording_id']] = [p for p in phones if p['phone'] == 'unk']
    for row in rows:
        if verify_wavs:
            path = ROOT / row['path']
            info = sf.info(path)
            if info.frames != 8000 or info.samplerate != 16000 or info.channels != 1 or info.subtype != 'PCM_16':
                errors.append(f"Invalid format: {row['sample_id']}")
            audio, _ = load_audio(path)
            if digest(path) != row['audio_sha256'] or pcm_hash(audio) != row['pcm_sha256']:
                errors.append(f"Changed WAV: {row['sample_id']}")
        start, end = int(row['window_start_sample']) / 16000, int(row['window_end_sample']) / 16000
        targets = target_intervals[row['recording_id']]
        present = any(start < p['end'] and end > p['start'] for p in targets)
        if any(start < p['end'] and end > p['start'] for p in unknown_intervals[row['recording_id']]):
            errors.append(f"Uncertain target label: {row['sample_id']}")
        if int(row['label']) != int(present):
            errors.append(f"Target label mismatch: {row['sample_id']}")
        if row['label'] == '1' and not (start <= float(row['phone_start']) and end >= float(row['phone_end'])):
            errors.append(f"Target cut: {row['sample_id']}")
    for split in SPLITS:
        subset = [r for r in rows if r['split'] == split]
        counts[split] = {'files': len(subset), 'labels': dict(Counter(r['label'] for r in subset)),
                         'speakers': sorted({r['speaker_id'] for r in subset}),
                         'recordings': len({r['recording_id'] for r in subset}),
                         'prompts': len({r['prompt_id'] for r in subset}),
                         'negative_phones': dict(Counter(r['center_phone'] for r in subset if r['label'] == '0'))}
        if {r['label'] for r in subset} != {'0', '1'}:
            errors.append(f'{split}: both labels required')
    report = {'passed': not errors, 'counts': counts, 'cross_split_collisions': overlaps,
              'errors': errors, 'manifest_sha256': digest(manifest), 'human_reviewed_files': 0,
              'quarantined_recordings': metadata.get('excluded_recordings', [])}
    report['annotation_origin'] = metadata.get('labels')
    report['generated_wavs_rechecked'] = verify_wavs
    write_json(ROOT / ('reports/dataset_audit.json' if verify_wavs else 'reports/manifest_audit.json'), report)
    if errors:
        raise ValueError(f'Audit failed: {errors[:5]}')
    print('Dataset audit passed:', {s: counts[s]['files'] for s in SPLITS}, flush=True)
    return rows


class LogMel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or FEATURES
        self.register_buffer('window', torch.hann_window(self.config['win_length'], periodic=True))
        c = self.config
        mel = librosa.filters.mel(sr=c['sample_rate'], n_fft=c['n_fft'], n_mels=c['n_mels'],
                                 fmin=c['fmin'], fmax=c['fmax'], htk=False, norm='slaney')
        self.register_buffer('mel', torch.from_numpy(mel))

    def forward(self, audio):
        c = self.config
        spectrum = torch.stft(audio, n_fft=c['n_fft'], hop_length=c['hop_length'],
                              win_length=c['win_length'], window=self.window, center=False, return_complex=True)
        return torch.log(torch.matmul(self.mel, spectrum.abs().square()).clamp_min(c['epsilon']))


class SchwaCNN(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        config = config or MODEL
        blocks, previous = [], 1
        for channels in config['channels']:
            blocks.extend([nn.Conv2d(previous, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(), nn.MaxPool2d(2)])
            previous = channels
        self.convolutions = nn.Sequential(*blocks)
        self.pooling = config.get('pooling', 'global')
        multiplier = config.get('pooled_frequency_bins', 4) if self.pooling == 'preserve_frequency' else 1
        self.classifier = nn.Sequential(nn.Linear(previous * multiplier, config['hidden_size']), nn.ReLU(),
                                        nn.Dropout(config['dropout']), nn.Linear(config['hidden_size'], 1))

    def forward(self, values):
        encoded = self.convolutions(values)
        pooled = torch.amax(encoded, dim=3).flatten(1) if self.pooling == 'preserve_frequency' else torch.amax(encoded, dim=(2, 3))
        return self.classifier(pooled).squeeze(-1)


def feature_array(rows, denoise='off', noise_snr=None):
    key = object_digest({'audio': [r['audio_sha256'] for r in rows], 'features': FEATURES,
                         'denoise': denoise, 'denoise_config': DENOISE, 'white_noise_snr': noise_snr,
                         'implementation_sha256': digest(__file__)})
    path = ROOT / 'data/features' / (key + '.npy')
    marker = path.with_suffix('.sha256.json')
    if path.exists():
        if not marker.exists() or read_json(marker)['sha256'] != digest(path):
            raise ValueError('Changed or unverified feature cache.')
        cached = np.load(path, allow_pickle=False)
        if cached.shape != (len(rows), 32, 47) or not np.isfinite(cached).all():
            raise ValueError('Invalid feature cache.')
        return cached
    extractor = LogMel().eval()
    output, buffer = [], []
    for index, row in enumerate(rows, 1):
        path_audio = ROOT / row['path']
        if digest(path_audio) != row['audio_sha256']:
            raise ValueError(f'Changed input: {path_audio}')
        audio, _ = load_audio(path_audio)
        if noise_snr is not None:
            rng = seeded(42, row['sample_id'] + f':noise:{noise_snr}')
            noise = rng.normal(size=len(audio)).astype(np.float32)
            noise *= np.sqrt(np.mean(audio ** 2)) / max(1e-8, np.sqrt(np.mean(noise ** 2))) / 10 ** (noise_snr / 20)
            audio = audio + noise
        if denoise == 'auto':
            audio, _ = denoise_audio(audio)
        buffer.append(audio)
        if len(buffer) == 128 or index == len(rows):
            with torch.inference_mode():
                output.append(extractor(torch.from_numpy(np.stack(buffer))).numpy())
            buffer = []
        if index % 1000 == 0 or index == len(rows):
            print(f'Features {denoise}, SNR={noise_snr}: {index}/{len(rows)}', flush=True)
    values = np.concatenate(output).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite features.')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, values, allow_pickle=False)
    write_json(marker, {'sha256': digest(path), 'key': key, 'shape': list(values.shape)})
    return values


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    prediction = (probabilities >= threshold).astype(np.int64)
    if not len(labels) or len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError('Metrics require nonempty, finite, equally sized labels/scores.')
    matrix = confusion_matrix(labels, prediction, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    undefined = {}
    if not tp + fp:
        undefined['precision'] = 'No predicted positives.'
    if not tp + fn:
        undefined['recall'] = undefined['fnr'] = undefined['average_precision'] = 'No positive labels.'
    if not tn + fp:
        undefined['fpr'] = 'No negative labels.'
    if not 2 * tp + fp + fn:
        undefined['f1'] = 'No positive labels or predictions.'
    if len(set(labels)) != 2:
        undefined['auroc'] = 'Both label classes are required.'
    return {'files': len(labels), 'threshold': float(threshold), 'accuracy': float(accuracy_score(labels, prediction)),
            'precision': float(tp / (tp + fp)) if tp + fp else None,
            'recall': float(tp / (tp + fn)) if tp + fn else None,
            'f1': float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None,
            'auroc': float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None,
            'average_precision': float(average_precision_score(labels, probabilities)) if np.any(labels == 1) else None,
            'confusion_matrix': matrix.tolist(), 'fpr': float(fp / (fp + tn)) if fp + tn else None,
            'fnr': float(fn / (fn + tp)) if fn + tp else None, 'undefined_metrics': undefined}


def choose_threshold(labels, probabilities):
    if set(map(int, labels)) != {0, 1}:
        raise ValueError('Threshold selection requires both classes in validation.')
    trials = [binary_metrics(labels, probabilities, round(float(t), 2)) for t in np.arange(0.05, 0.951, 0.01)]
    return max(trials, key=lambda r: (r['f1'], -r['fpr'], r['threshold'])), trials


def probabilities(model, values, batch_size=64):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            result.extend(torch.sigmoid(model(torch.from_numpy(values[start:start + batch_size]).unsqueeze(1))).tolist())
    return np.asarray(result, dtype=np.float64)


def save_torch(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.pt')
    torch.save(state, temporary)
    temporary.replace(path)


def train(args):
    if args.device != 'cpu':
        raise ValueError('This baseline trains on CPU.')
    if min(args.threads, args.batch_size, args.epochs, args.patience) < 1 or args.lr <= 0:
        raise ValueError('Threads, batch size, epochs, patience and learning rate must be positive.')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rows = audit_dataset()
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'validation']
    train_raw = feature_array(train_rows, args.denoise)
    val_raw = feature_array(val_rows, args.denoise)
    mean = train_raw.mean(axis=(0, 2), dtype=np.float64).astype(np.float32)
    std = np.maximum(train_raw.std(axis=(0, 2), dtype=np.float64), 1e-4).astype(np.float32)
    train_x = ((train_raw - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    val_x = ((val_raw - mean[None, :, None]) / std[None, :, None]).astype(np.float32)
    train_y = torch.tensor([int(r['label']) for r in train_rows], dtype=torch.float32)
    val_y = np.array([int(r['label']) for r in val_rows])
    run_dir = args.run_dir.resolve()
    model_config = {**MODEL, 'pooling': getattr(args, 'pooling', MODEL['pooling'])}
    settings = {'seed': args.seed, 'epochs': args.epochs, 'patience': args.patience,
                'batch_size': args.batch_size, 'learning_rate': args.lr, 'denoise': args.denoise,
                'threads': args.threads, 'dataset_sha256': digest(ROOT / 'data/manifests/dataset.csv'),
                'code_sha256': digest(__file__), 'model_config': model_config}
    if (run_dir / 'best.pt').exists() and not args.resume:
        raise ValueError('Run already has weights. Use --resume or a new --run-dir.')
    model = SchwaCNN(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, bad_epochs, history, first_epoch = float('inf'), 0, [], 1
    if args.resume:
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=True)
        if state['settings'] != settings:
            raise ValueError('Resume settings/code/dataset differ from the frozen run.')
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        torch.set_rng_state(state['torch_rng_state'])
        best_loss, bad_epochs, history = state['best_loss'], state['bad_epochs'], state['history']
        first_epoch = state['epoch'] + 1
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_x).unsqueeze(1), train_y)
    started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs + 1):
        if bad_epochs >= args.patience:
            break
        epoch_start = time.perf_counter()
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=0, generator=generator)
        model.train()
        total_loss = 0.0
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(values), labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss.')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
        val_prob = probabilities(model, val_x)
        clipped = np.clip(val_prob, 1e-7, 1 - 1e-7)
        val_loss = float(-np.mean(val_y * np.log(clipped) + (1 - val_y) * np.log(1 - clipped)))
        val_metrics = binary_metrics(val_y, val_prob, 0.5)
        item = {'epoch': epoch, 'train_loss': total_loss / len(dataset), 'validation_loss': val_loss,
                'validation_f1_at_0_5': val_metrics['f1'], 'seconds': time.perf_counter() - epoch_start}
        history.append(item)
        if val_loss < best_loss:
            best_loss, bad_epochs = val_loss, 0
            save_torch(run_dir / 'best.pt', {'model_state_dict': model.state_dict(), 'model_config': model_config,
                       'feature_config': FEATURES, 'audio_config': AUDIO, 'denoise_config': DENOISE,
                       'denoise': args.denoise, 'normalization': {'mean': mean.tolist(), 'std': std.tolist(),
                                                               'split': 'train', 'files': len(train_rows)},
                       'target_symbol': '\u0259', 'target_phones': ['ax'], 'epoch': epoch,
                       'settings': settings, 'validation_loss': val_loss, 'cpu_threads': args.threads})
        else:
            bad_epochs += 1
        save_torch(run_dir / 'last.pt', {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                   'torch_rng_state': torch.get_rng_state(), 'settings': settings, 'epoch': epoch,
                   'best_loss': best_loss, 'bad_epochs': bad_epochs, 'history': history})
        write_csv(run_dir / 'history.csv', history)
        print(f"Epoch {epoch:02d}: train={item['train_loss']:.4f}, val={val_loss:.4f}, F1@0.5={val_metrics['f1']:.4f}, {item['seconds']:.1f}s", flush=True)
    best = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(best['model_state_dict'])
    val_prob = probabilities(model, val_x)
    selected, sweep = choose_threshold(val_y, val_prob)
    best['threshold'] = selected['threshold']
    best['threshold_selection_split'] = 'validation'
    save_torch(run_dir / 'best.pt', best)
    write_json(run_dir / 'threshold_sweep.json', sweep)
    write_json(run_dir / 'validation_metrics.json', selected)
    write_csv(run_dir / 'validation_predictions.csv',
              [{'sample_id': r['sample_id'], 'label': r['label'], 'probability': float(p)} for r, p in zip(val_rows, val_prob)])
    write_json(run_dir / 'training_summary.json', {'device': 'cpu', 'parameters': sum(p.numel() for p in model.parameters()),
               'selected_epoch': best['epoch'], 'threshold': selected['threshold'], 'epochs_completed': len(history),
               'best_validation_loss': best['validation_loss'], 'validation_metrics': selected,
               'training_loop_seconds_this_invocation': time.perf_counter() - started,
               'settings': settings, 'environment': {'python': platform.python_version(), 'torch': torch.__version__}})
    print(f"Training done. Best epoch={best['epoch']}, threshold={selected['threshold']:.2f}, weights={run_dir / 'best.pt'}", flush=True)


class SchwaDetector:
    def __init__(self, model_path=None):
        self.path = Path(model_path or (ROOT / 'best.pt' if (ROOT / 'best.pt').exists() else ROOT / 'runs/baseline/best.pt')).resolve()
        self.checkpoint = torch.load(self.path, map_location='cpu', weights_only=True)
        c = self.checkpoint
        if c.get('target_symbol') != '\u0259' or c.get('target_phones') != ['ax'] or 'threshold' not in c:
            raise ValueError('Not a trained schwa checkpoint with a frozen validation threshold.')
        torch.set_num_threads(c['cpu_threads'])
        self.model = SchwaCNN(c['model_config']).eval()
        self.model.load_state_dict(c['model_state_dict'])
        self.extractor = LogMel(c['feature_config']).eval()
        self.mean = np.array(c['normalization']['mean'], dtype=np.float32)[None, :, None]
        self.std = np.array(c['normalization']['std'], dtype=np.float32)[None, :, None]

    def describe(self):
        c = self.checkpoint
        return {'weights': str(self.path), 'target_symbol': c['target_symbol'], 'target_phones': c['target_phones'],
                'epoch': c['epoch'], 'threshold': c['threshold'], 'denoise': c['denoise'], 'device': 'cpu',
                'window_seconds': c['audio_config']['window_seconds'], 'sha256': digest(self.path)}

    def predict(self, audio_path, hop_seconds=0.05):
        if not 0 < hop_seconds <= self.checkpoint['audio_config']['window_seconds']:
            raise ValueError('Hop must be positive and no longer than the window.')
        audio, meta = load_audio(audio_path, self.checkpoint['audio_config']['sample_rate'])
        sr = self.checkpoint['audio_config']['sample_rate']
        samples = round(sr * self.checkpoint['audio_config']['window_seconds'])
        hop = max(1, round(sr * hop_seconds))
        starts = list(range(0, max(1, len(audio) - samples + 1), hop))
        last = max(0, len(audio) - samples)
        if starts[-1] != last:
            starts.append(last)
        detected, max_score, filtering = [], 0.0, Counter()
        for index in range(0, len(starts), 64):
            batch_starts = starts[index:index + 64]
            windows = []
            for start in batch_starts:
                values = crop_window(audio, start, samples)
                if self.checkpoint['denoise'] == 'auto':
                    values, status = denoise_audio(values, self.checkpoint['denoise_config'])
                    filtering[status] += 1
                windows.append(values)
            with torch.inference_mode():
                features = self.extractor(torch.from_numpy(np.stack(windows))).numpy()
            values = ((features - self.mean) / self.std).astype(np.float32)
            scores = probabilities(self.model, values)
            max_score = max(max_score, float(scores.max()))
            for start, score in zip(batch_starts, scores):
                if score >= self.checkpoint['threshold']:
                    detected.append({'start_seconds': start / sr, 'end_seconds': min(len(audio), start + samples) / sr,
                                     'score': float(score)})
        return {'audio': str(Path(audio_path).resolve()), 'target_symbol': '\u0259', 'score': max_score,
                'threshold': self.checkpoint['threshold'], 'detected': bool(detected), 'windows': detected,
                'duration_seconds': meta['duration_seconds'], 'denoise': self.checkpoint['denoise'],
                'filtering_status': dict(filtering), 'note': 'Windows indicate approximate regions, not exact phone boundaries.'}


def evaluate(args):
    rows = audit_dataset()
    rows = [row for row in rows if row['split'] == args.split]
    detector = SchwaDetector(args.model)
    frozen = detector.checkpoint
    if digest(ROOT / 'data/manifests/dataset.csv') != frozen['settings']['dataset_sha256']:
        raise ValueError('Evaluation dataset differs from trained manifest.')
    output = args.output.resolve()
    if (output / 'metrics.json').exists():
        raise ValueError('Evaluation report already exists; use a new --output directory.')
    raw = feature_array(rows, frozen['denoise'])
    values = ((raw - detector.mean) / detector.std).astype(np.float32)
    scores = probabilities(detector.model, values)
    labels = [int(row['label']) for row in rows]
    overall = binary_metrics(labels, scores, frozen['threshold'])
    groups = {}
    for field in ('speaker_id', 'center_phone'):
        groups[field] = {}
        for value in sorted({row[field] for row in rows}):
            indexes = [i for i, row in enumerate(rows) if row[field] == value]
            groups[field][value] = binary_metrics(np.asarray(labels)[indexes], scores[indexes], frozen['threshold'])
    predictions = [{**row, 'probability': float(score), 'prediction': int(score >= frozen['threshold'])} for row, score in zip(rows, scores)]
    write_csv(output / 'predictions.csv', predictions)
    fields = list(predictions[0])
    write_csv(output / 'false_positives.csv', [r for r in predictions if r['label'] == '0' and r['prediction'] == 1], fields)
    write_csv(output / 'false_negatives.csv', [r for r in predictions if r['label'] == '1' and r['prediction'] == 0], fields)
    noise_results = {}
    if args.noise_check:
        for snr in (0, 5, 10):
            noise_results[str(snr)] = {}
            for mode in ('off', 'auto'):
                noisy = feature_array(rows, mode, noise_snr=snr)
                values = ((noisy - detector.mean) / detector.std).astype(np.float32)
                noise_results[str(snr)][mode] = binary_metrics(labels, probabilities(detector.model, values), frozen['threshold'])
    report = {'split': args.split, 'overall': overall, 'groups': groups, 'checkpoint': detector.describe(),
              'dataset_sha256': frozen['settings']['dataset_sha256'], 'white_noise_exploratory': noise_results,
              'limitations': ['Ground truth comes from automatic CMU ARCTIC phone alignment, not human-confirmed pronunciation.',
                              'Few held-out speakers; no systematic evaluation on user recordings.',
                              'Overlapping windows and same-recording variants are correlated.',
                              'White-noise filtering comparisons are exploratory; preprocessing/threshold stay frozen.']}
    write_json(output / 'metrics.json', report)
    print(json.dumps(overall, indent=2), flush=True)


def predict_cli(args):
    detector = SchwaDetector(args.model)
    if args.inspect:
        print(json.dumps(detector.describe(), ensure_ascii=False, indent=2))
        return
    if not args.audio:
        raise ValueError('Provide --audio or --inspect.')
    results = [detector.predict(path, args.hop) for path in args.audio]
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
    else:
        for result in results:
            decision = 'PHÁT HIỆN' if result['detected'] else 'KHÔNG PHÁT HIỆN'
            print(f"{result['audio']}: {decision} /ə/ | score={result['score']:.6f} | ngưỡng={result['threshold']:.2f}")
            print(f"  Thời lượng={result['duration_seconds']:.3f}s; cửa sổ vượt ngưỡng={len(result['windows'])}")
            for window in result['windows'][:8]:
                print(f"  {window['start_seconds']:.2f}–{window['end_seconds']:.2f}s: score={window['score']:.4f}")
            if len(result['windows']) > 8:
                print('  Dùng --output result.json để lưu toàn bộ cửa sổ.')
    if args.output:
        write_json(args.output, results)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('download')
    p.add_argument('--train-prompts', type=int, default=200)
    p.add_argument('--validation-prompts', type=int, default=80)
    p.add_argument('--test-prompts', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--timeout', type=int, default=20)
    p.add_argument('--retries', type=int, default=3)
    p.set_defaults(function=download)
    for name in ('make-dataset', 'prepare'):
        p = commands.add_parser(name)
        p.add_argument('--seed', type=int, default=42)
        p.set_defaults(function=make_dataset)
    p = commands.add_parser('audit')
    p.set_defaults(function=lambda args: audit_dataset())
    p = commands.add_parser('train')
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--denoise', choices=['off', 'auto'], default='off')
    p.add_argument('--pooling', choices=['global', 'preserve_frequency'], default='preserve_frequency')
    p.add_argument('--run-dir', type=Path, default=ROOT / 'runs/baseline')
    p.add_argument('--resume', action='store_true')
    p.set_defaults(function=train)
    p = commands.add_parser('evaluate')
    p.add_argument('--model', type=Path, default=ROOT / 'runs/baseline/best.pt')
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--output', type=Path, default=ROOT / 'runs/baseline/evaluation/test')
    p.add_argument('--noise-check', action='store_true')
    p.set_defaults(function=evaluate)
    p = commands.add_parser('predict')
    p.add_argument('--model', type=Path)
    p.add_argument('--audio', type=Path, nargs='+')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--hop', type=float, default=0.05)
    p.add_argument('--json', action='store_true')
    p.add_argument('--output', type=Path)
    p.set_defaults(function=predict_cli)
    p = commands.add_parser('denoise')
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=lambda args: denoise_cli(args))
    args = parser.parse_args(argv)
    args.function(args)


def denoise_cli(args):
    audio, _ = load_audio(args.audio)
    filtered, status = denoise_audio(audio)
    if args.output.resolve() == args.audio.resolve():
        raise ValueError('Keep the original audio; choose a different output path.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, np.clip(filtered, -1, 1), 16000, subtype='PCM_16')
    print(f'Denoise status: {status}; saved {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)

"""CPU transfer-learning experiments for schwa, with frozen pretrained acoustic features."""
import argparse
import gc
import json
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import nn

import schwa as core
from frozen_phoneme import RECORDING, rebuild_acoustic, acoustic_features, recording_features, aligned_window_features, representation_state

ROOT = Path(__file__).resolve().parent
BACKBONE = ROOT / 'data/pretrained/wav2vec2_base_6layers.pt'
CONTEXT = 'window'
REPRESENTATION = 'stored'
V2 = ROOT / 'v2'
V3 = ROOT / 'v3'
V4 = ROOT / 'v4'
V5 = ROOT / 'v5'
SPECTRAL = {'sample_rate': 16000, 'n_fft': 400, 'hop_length': 320, 'n_mels': 40, 'fmin': 50., 'fmax': 8000.}
V2_SPEAKERS = {'train': ['bdl', 'clb', 'rms', 'jmk', 'slt'], 'validation': ['awb'], 'test': ['ksp']}


def download_v2(args):
    counts = {'train': 300, 'validation': 120, 'test': 120}
    selection = {'seed': 20261010, 'counts': counts, 'speakers': V2_SPEAKERS,
                 'purpose': 'Old test is retired. KSP is a new held-out test speaker; AWB is validation.'}
    selection_path = V2 / 'data/manifests/source_selection.json'
    if selection_path.exists() and core.read_json(selection_path) != selection:
        raise ValueError('Preserving existing v2 source selection.')
    core.write_json(selection_path, selection)
    prompts = [f'arctic_a{i:04d}' for i in range(1, 594)]
    random.Random(selection['seed']).shuffle(prompts)
    planned, offset = [], 0
    for split in core.SPLITS:
        selected = prompts[offset:offset + counts[split]]
        offset += counts[split]
        for speaker in V2_SPEAKERS[split]:
            base = core.BASE_URL + f'cmu_us_{speaker}_arctic/'
            for name in ('COPYING', 'README'):
                core.fetch(base + name, ROOT / 'data/raw' / speaker / name)
            for prompt in selected:
                planned.append({'split': split, 'speaker_id': speaker, 'prompt_id': prompt,
                                'recording_id': speaker + '_' + prompt,
                                'audio_path': f'../data/raw/{speaker}/{prompt}.wav',
                                'lab_path': f'../data/raw/{speaker}/{prompt}.lab', 'base_url': base})

    def fetch_recording(row):
        result = dict(row)
        for kind, folder, suffix in (('audio', 'wav', '.wav'), ('lab', 'lab', '.lab')):
            metadata = core.fetch(row['base_url'] + folder + '/' + row['prompt_id'] + suffix,
                                  V2 / row[kind + '_path'])
            result[kind + '_sha256'] = metadata['sha256']
        return result

    ready, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_recording, row): row for row in planned}
        for index, future in enumerate(as_completed(futures), 1):
            try:
                ready.append(future.result())
            except Exception as error:
                failures.append({'recording_id': futures[future]['recording_id'], 'error': str(error)})
            if index % 50 == 0 or index == len(planned):
                print(f'V2 sources {index}/{len(planned)}, failures={len(failures)}', flush=True)
    ready.sort(key=lambda r: r['recording_id'])
    core.write_csv(V2 / 'data/manifests/recordings.csv', ready)
    core.write_json(V2 / 'reports/download.json', {'planned': len(planned), 'ready': len(ready), 'failed': failures})
    if failures:
        raise RuntimeError('Incomplete download; rerun download-v2 to resume.')


def activate_dataset(version):
    directory = {'v1': ROOT, 'v2': V2, 'v3': V3, 'v4': V4, 'v5': V5}[version]
    core.ROOT = directory
    return directory


def select_training_rows(rows, limit=None, seed=42):
    """Stratify a CPU experiment by speaker and label without changing held-out data."""
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Training subset requires positive size and train-only rows.')
    groups = {}
    for row in rows:
        groups.setdefault((row['speaker_id'], row['label']), []).append(row)
    if limit < len(groups):
        raise ValueError('Training budget must include every speaker/class group.')
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = set()
    while len(selected) < limit:
        for key in sorted(groups):
            if groups[key] and len(selected) < limit:
                selected.add(groups[key].pop()['sample_id'])
    return [r for r in rows if r['sample_id'] in selected]


def backbone():
    state = torch.load(BACKBONE, map_location='cpu', weights_only=True, mmap=True)
    return rebuild_acoustic(representation_state(state, REPRESENTATION))


def normalized_waveforms(waveforms, enabled):
    # Normalize each input independently, so predictions do not depend on other files in a batch.
    return nn.functional.layer_norm(waveforms, (waveforms.shape[-1],)) if enabled else waveforms


def prepare_backbone(args):
    """Reproduce a frozen BASE encoder; feature caches bind its file hash and depth."""
    bundle_name = {'ssl': 'WAV2VEC2_BASE', 'asr': 'WAV2VEC2_ASR_BASE_960H', 'xlsr': 'WAV2VEC2_XLSR53'}[args.bundle]
    if BACKBONE.exists():
        state = torch.load(BACKBONE, map_location='cpu', weights_only=True)
        if state.get('source_bundle') != 'torchaudio.pipelines.' + bundle_name or state.get('layers') != args.layers:
            raise ValueError('Unexpected existing acoustic encoder.')
        print('Existing backbone preserved:', BACKBONE, core.digest(BACKBONE), flush=True)
        return
    torch.set_num_threads(args.threads)
    torch.hub.set_dir(str(ROOT / 'data/pretrained/torch_hub'))
    bundle = getattr(torchaudio.pipelines, bundle_name)
    config = dict(bundle._params)
    config.update(encoder_num_layers=args.layers, aux_num_out=None)
    checkpoint_path = ROOT / 'data/pretrained/torch_hub/checkpoints' / bundle._path
    source_url = 'https://download.pytorch.org/torchaudio/models/' + bundle._path
    if not checkpoint_path.exists():
        core.fetch(source_url, checkpoint_path, timeout=60)
    # Construct only the retained layers. Loading an entire LARGE model would waste RAM on this CPU machine.
    try:
        source_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError:
        source_state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = torchaudio.models.wav2vec2_model(**config).eval()
    retained = {}
    for key, tensor in source_state.items():
        if key.startswith('aux.'):
            continue
        if key.startswith('encoder.transformer.layers.') and int(key.split('.')[3]) >= args.layers:
            continue
        retained[key] = tensor
    model.load_state_dict(retained)
    core.save_torch(BACKBONE, {'model_config': config,
                    'model_state_dict': {k: v.half() for k, v in model.state_dict().items()},
                    'source_bundle': 'torchaudio.pipelines.' + bundle_name,
                    'source_url': source_url, 'source_checkpoint_sha256': core.digest(checkpoint_path),
                    'sample_rate': 16000, 'layers': args.layers, 'weight_storage': 'float16', 'computation': 'float32',
                    'normalize_waveform': bundle._normalize_waveform,
                    'parameters': sum(p.numel() for p in model.parameters())})
    print('Prepared frozen acoustic encoder:', BACKBONE, flush=True)


@lru_cache(maxsize=4)
def _backbone_geometry(path, fingerprint):
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    if state['layers'] != state['model_config']['encoder_num_layers']:
        raise ValueError('Encoder depth does not match its stored configuration.')
    attention = 'math' if state.get('feature_kind') in ('ctc_logits', 'ctc_embeddings_logits') else None
    return state['layers'], state.get('feature_dimension', state['model_config']['encoder_embed_dim']), attention, state['model_config']['encoder_embed_dim']


def _backbone_layers(path, fingerprint):
    return _backbone_geometry(path, fingerprint)[0]


def backbone_dimension():
    geometry = _backbone_geometry(str(BACKBONE), core.digest(BACKBONE))
    if REPRESENTATION == 'joint' and geometry[2] != 'math':
        raise ValueError('Joint representation requires a phoneme CTC encoder.')
    return geometry[1] + geometry[3] if REPRESENTATION == 'joint' else geometry[1]


def cache_paths(directory, rows, split):
    fingerprint = core.digest(BACKBONE)
    recipe = {'backbone_sha256': fingerprint,
                                   'audio_sha256': [r['audio_sha256'] for r in rows],
                                   'window_samples': 8000, 'storage': 'float16',
                                   'layer': _backbone_layers(str(BACKBONE), fingerprint)}
    if REPRESENTATION != 'stored':
        recipe['representation'] = REPRESENTATION
    if CONTEXT == 'recording':
        sources = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
        recipe.update(context=RECORDING, source_regions=[(sources[r['recording_id']]['audio_sha256'],
                                                        r['window_start_sample']) for r in rows])
    else:
        attention = _backbone_geometry(str(BACKBONE), fingerprint)[2]
        if attention:
            recipe['attention_backend'] = attention
    identity = core.object_digest(recipe)
    base = directory / 'data/transfer_features' / (split + '_' + identity[:20])
    return base.with_suffix('.npy'), base.with_suffix('.json')


def extract(args):
    directory = activate_dataset(args.dataset)
    torch.set_num_threads(args.threads)
    # A full audit was completed by make-dataset. Each consumed WAV is checked again below.
    all_rows = core.audit_dataset(verify_wavs=False)
    if CONTEXT == 'recording':
        if args.reuse_dataset:
            raise ValueError('Recording context uses per-source caches, rather than WAV-only donors.')
        return extract_recordings(args, directory, all_rows)
    reused_features = {}
    if args.reuse_dataset:
        donor = {'v1': ROOT, 'v2': V2, 'v3': V3, 'v4': V4, 'v5': V5}[args.reuse_dataset]
        donor_metadata = core.read_json(donor / 'data/manifests/dataset_metadata.json')
        if core.digest(donor / 'data/manifests/dataset.csv') != donor_metadata['manifest_sha256']:
            raise ValueError('Donor feature manifest changed.')
        donor_rows = core.read_csv(donor / 'data/manifests/dataset.csv')
        for donor_split in core.SPLITS:
            subset = [r for r in donor_rows if r['split'] == donor_split]
            donor_path, donor_marker = cache_paths(donor, subset, donor_split)
            if not donor_path.exists() or not donor_marker.exists():
                continue
            cached = feature_cache(donor, subset, donor_split)
            for index, row in enumerate(subset):
                reused_features[row['audio_sha256']] = (cached, index)
        print('Verified donor waveforms:', len(reused_features), flush=True)
    model = None
    dimension = backbone_dimension()
    for split in args.splits:
        rows = [r for r in all_rows if r['split'] == split]
        if split == 'train':
            rows = select_training_rows(rows, args.train_limit, args.seed)
        path, marker = cache_paths(directory, rows, split)
        if path.exists() and marker.exists():
            if core.digest(path) != core.read_json(marker)['sha256']:
                raise ValueError('Changed transfer feature cache.')
            print('Verified cache reused:', split, flush=True)
            continue
        if model is None:
            model = backbone()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial.npy')
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(len(rows), 24, dimension))
        started = time.perf_counter()
        reused = 0
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                subset = rows[start:start + args.batch_size]
                waveforms, positions = [], []
                for position, row in enumerate(subset):
                    audio_path = directory / row['path']
                    if core.digest(audio_path) != row['audio_sha256']:
                        raise ValueError('Changed input WAV.')
                    donor_feature = reused_features.get(row['audio_sha256'])
                    if donor_feature is not None:
                        cached, donor_index = donor_feature
                        output[start + position] = cached[donor_index]
                        reused += 1
                        continue
                    waveform, _ = core.load_audio(audio_path)
                    if len(waveform) != 8000:
                        raise ValueError('Expected exactly 0.5 second input.')
                    waveforms.append(waveform)
                    positions.append(position)
                if waveforms:
                    inputs = normalized_waveforms(torch.from_numpy(np.stack(waveforms)), model.normalize_waveform)
                    features = acoustic_features(model, inputs, len(model.encoder.transformer.layers))
                    values = features.numpy().astype(np.float16)
                    if not np.isfinite(values).all() or values.shape[1:] != (24, dimension):
                        raise ValueError('Invalid acoustic features.')
                    output[start + np.array(positions)] = values
                completed = start + len(subset)
                if completed % 128 == 0 or completed == len(rows):
                    print(f'Transfer features {args.dataset}/{split}: {completed}/{len(rows)}, {time.perf_counter()-started:.1f}s', flush=True)
        output.flush()
        del output
        temporary.replace(path)
        core.write_json(marker, {'sha256': core.digest(path), 'shape': [len(rows), 24, dimension],
                                 'representation': REPRESENTATION,
                                 'backbone_sha256': core.digest(BACKBONE), 'row_ids': [r['sample_id'] for r in rows],
                                 'layers': len(model.encoder.transformer.layers),
                                 'features_reused_from_dataset': args.reuse_dataset, 'reused_waveforms': reused,
                                 'seconds': time.perf_counter() - started})


def extract_recordings(args, directory, all_rows):
    records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
    encoder_sha = core.digest(BACKBONE)
    dimension = backbone_dimension()
    model = None
    for split in args.splits:
        rows = [r for r in all_rows if r['split'] == split]
        if split == 'train':
            rows = select_training_rows(rows, args.train_limit, args.seed)
        path, marker = cache_paths(directory, rows, split)
        if path.exists() and marker.exists():
            feature_cache(directory, rows, split)
            print('Verified recording-context cache reused:', split, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial.npy')
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(len(rows), 24, dimension))
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(row['recording_id'], []).append((index, row))
        started = time.perf_counter()
        completed = 0
        for number, (recording_id, group) in enumerate(sorted(groups.items()), 1):
            source = records[recording_id]
            audio_path = directory / source['audio_path']
            if core.digest(audio_path) != source['audio_sha256']:
                raise ValueError('Changed recording-context input WAV.')
            sequence_recipe = {'encoder': encoder_sha, 'source_audio': source['audio_sha256'], 'recipe': RECORDING}
            if REPRESENTATION != 'stored':
                sequence_recipe['representation'] = REPRESENTATION
            identity = core.object_digest(sequence_recipe)
            sequence_path = directory / 'data/recording_transfer_features' / (identity + '.npy')
            sequence_marker = sequence_path.with_suffix('.json')
            if sequence_path.exists() and sequence_marker.exists():
                metadata = core.read_json(sequence_marker)
                if (core.digest(sequence_path) != metadata['sha256']
                        or metadata['source_audio_sha256'] != source['audio_sha256']
                        or metadata['backbone_sha256'] != encoder_sha or metadata['recipe'] != RECORDING
                        or metadata.get('representation', 'stored') != REPRESENTATION):
                    raise ValueError('Changed recording feature cache.')
                sequence = torch.from_numpy(np.load(sequence_path, allow_pickle=False))
                if list(sequence.shape) != metadata['shape'] or sequence.shape[-1] != dimension or not torch.isfinite(sequence).all():
                    raise ValueError('Invalid cached recording features.')
            else:
                if model is None:
                    model = backbone()
                waveform, _ = core.load_audio(audio_path)
                with torch.inference_mode():
                    sequence = recording_features(model, torch.from_numpy(waveform), len(model.encoder.transformer.layers))
                if sequence.ndim != 2 or sequence.shape[-1] != dimension or not torch.isfinite(sequence).all():
                    raise ValueError('Invalid recording acoustic features.')
                sequence_path.parent.mkdir(parents=True, exist_ok=True)
                part = sequence_path.with_suffix('.partial.npy')
                np.save(part, sequence.numpy(), allow_pickle=False)
                part.replace(sequence_path)
                core.write_json(sequence_marker, {'sha256': core.digest(sequence_path), 'shape': list(sequence.shape),
                                                  'source_audio_sha256': source['audio_sha256'],
                                                  'backbone_sha256': encoder_sha, 'recipe': RECORDING,
                                                  'representation': REPRESENTATION})
            indices = [index for index, _ in group]
            starts = [int(row['window_start_sample']) for _, row in group]
            output[indices] = aligned_window_features(sequence, starts).numpy().astype(np.float16)
            completed += len(group)
            if number % 16 == 0 or number == len(groups):
                print(f'Recording features {split}: {number}/{len(groups)} sources, {completed}/{len(rows)} regions, {time.perf_counter()-started:.1f}s', flush=True)
        output.flush()
        del output
        temporary.replace(path)
        core.write_json(marker, {'sha256': core.digest(path), 'shape': [len(rows), 24, dimension],
                                 'backbone_sha256': encoder_sha, 'row_ids': [r['sample_id'] for r in rows],
                                 'context': 'recording', 'context_recipe': RECORDING, 'source_recordings': len(groups),
                                 'representation': REPRESENTATION,
                                 'layers': _backbone_layers(str(BACKBONE), encoder_sha),
                                 'seconds': time.perf_counter() - started})


def feature_cache(directory, rows, split):
    path, marker = cache_paths(directory, rows, split)
    if not path.exists() or not marker.exists() or core.digest(path) != core.read_json(marker)['sha256']:
        raise ValueError(f'Missing/unverified cache for {split}; run extract first.')
    metadata = core.read_json(marker)
    if metadata.get('representation', 'stored') != REPRESENTATION:
        raise ValueError('Feature cache representation differs from the requested encoder output.')
    if metadata.get('context', 'window') != CONTEXT or (CONTEXT == 'recording' and metadata.get('context_recipe') != RECORDING):
        raise ValueError('Feature cache input context differs from the requested pipeline.')
    if metadata['backbone_sha256'] != core.digest(BACKBONE) or metadata['row_ids'] != [r['sample_id'] for r in rows]:
        raise ValueError('Feature cache encoder or row order differs from the frozen manifest.')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != (len(rows), 24, backbone_dimension()):
        raise ValueError('Unexpected transfer feature cache shape.')
    return values


def frame_labels(directory, rows, vocabulary=None, boundary_margin=0.005):
    recordings = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
    annotations = {}
    for recording_id in sorted({r['recording_id'] for r in rows}):
        rec = recordings[recording_id]
        audio, _ = core.load_audio(directory / rec['audio_path'])
        phones, _ = core.parse_phones((directory / rec['lab_path']).read_text(), len(audio) / 16000)
        annotations[recording_id] = phones
    if vocabulary is None:
        vocabulary = sorted(({p['phone'] for phones in annotations.values() for p in phones} - {'unk'}) | {'pau'})
    indices = {phone: index for index, phone in enumerate(vocabulary)}
    result = np.full((len(rows), 24), -100, dtype=np.int64)
    for index, row in enumerate(rows):
        phones = annotations[row['recording_id']]
        centers = int(row['window_start_sample']) / 16000 + (199.5 + np.arange(24) * 320) / 16000
        for frame, center in enumerate(centers):
            matching = next((phone for phone in phones if phone['start'] <= center < phone['end']), None)
            if matching is None:
                result[index, frame] = indices['pau']
            elif min(center - matching['start'], matching['end'] - center) >= boundary_margin:
                result[index, frame] = indices.get(matching['phone'], -100)
    return result, vocabulary


class AlignedMel(nn.Module):
    """24 log-mel frames with centers matching the acoustic encoder's 400/320 receptive field."""
    def __init__(self, config=None):
        super().__init__()
        self.config = config or SPECTRAL
        c = self.config
        low, high = 2595 * np.log10(1 + c['fmin'] / 700), 2595 * np.log10(1 + c['fmax'] / 700)
        points = 700 * (10 ** (np.linspace(low, high, c['n_mels'] + 2) / 2595) - 1)
        frequencies = np.arange(c['n_fft'] // 2 + 1) * c['sample_rate'] / c['n_fft']
        rise = (frequencies[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
        fall = (points[2:, None] - frequencies[None]) / (points[2:] - points[1:-1])[:, None]
        filters = np.maximum(0, np.minimum(rise, fall)) * (2 / (points[2:] - points[:-2]))[:, None]
        self.register_buffer('filters', torch.from_numpy(filters.astype(np.float32)))
        self.register_buffer('window', torch.hann_window(c['n_fft']))

    @property
    def feature_dimension(self):
        return self.config['n_mels'] + (8 if self.config.get('prosody') else 0)

    def prosody(self, audio):
        """Label-free pitch, periodicity and energy on the same 25 ms / 20 ms timeline."""
        c, p = self.config, self.config['prosody']
        frames = audio.unfold(-1, c['n_fft'], c['hop_length'])
        rms = frames.square().mean(dim=-1).sqrt()
        log_rms = rms.clamp_min(1e-6).log()
        centered = frames - frames.mean(dim=-1, keepdim=True)
        fft = torch.fft.rfft(centered, n=2 * c['n_fft'])
        autocorrelation = torch.fft.irfft(fft.abs().square(), n=2 * c['n_fft'])[..., :c['n_fft']]
        energy = torch.nn.functional.pad(centered.square().cumsum(dim=-1), (1, 0))
        lo = int(math.ceil(c['sample_rate'] / p['maximum_f0']))
        hi = min(c['n_fft'] - 2, int(c['sample_rate'] / p['minimum_f0']))
        lags = torch.arange(lo, hi + 1, device=audio.device)
        denominator = (energy[..., c['n_fft'] - lags] *
                       (energy[..., -1:] - energy[..., lags])).clamp_min(1e-16).sqrt()
        correlations = (autocorrelation[..., lags] / denominator).clamp(-1, 1)
        # Prefer the first strong local maximum to avoid choosing a multiple of the period.
        padded = torch.nn.functional.pad(correlations, (1, 1), value=-2.)
        peaks = (correlations > padded[..., :-2]) & (correlations >= padded[..., 2:])
        strong = peaks & (correlations >= .95 * correlations.amax(dim=-1, keepdim=True))
        first = strong.to(torch.int64).argmax(dim=-1)
        index = torch.where(strong.any(dim=-1), first, correlations.argmax(dim=-1))
        confidence = correlations.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(0)
        confidence = torch.where(rms > 1e-4, confidence, torch.zeros_like(confidence))
        voiced = confidence >= p['voicing_threshold']
        log_f0 = (c['sample_rate'] / lags[index]).log()
        log_f0 = torch.where(voiced, log_f0, torch.zeros_like(log_f0))
        mean_f0 = log_f0.sum(dim=-1, keepdim=True) / voiced.sum(dim=-1, keepdim=True).clamp_min(1)
        relative_f0 = torch.where(voiced, log_f0 - mean_f0, torch.zeros_like(log_f0))
        relative_energy = log_rms - log_rms.mean(dim=-1, keepdim=True)
        delta_energy = torch.nn.functional.pad(log_rms[..., 1:] - log_rms[..., :-1], (1, 0))
        crossing = ((centered[..., 1:] >= 0) != (centered[..., :-1] >= 0)).float().mean(dim=-1)
        return torch.stack([log_f0, relative_f0, confidence, log_rms, relative_energy,
                            delta_energy, crossing, voiced.float()], dim=-1)

    def forward(self, audio):
        c = self.config
        spectrum = torch.stft(audio, n_fft=c['n_fft'], hop_length=c['hop_length'],
                              window=self.window, center=False, return_complex=True)
        mel = torch.log(torch.matmul(self.filters, spectrum.abs().square()).clamp_min(1e-10)).transpose(1, 2)
        return torch.cat([mel, self.prosody(audio)], dim=-1) if c.get('prosody') else mel


def spectral_paths(directory, rows, split, config=None):
    identity = core.object_digest({'recipe': 'aligned-mel-v1-float16', 'config': config or SPECTRAL,
                                   'audio_sha256': [r['audio_sha256'] for r in rows]})
    base = directory / 'data/spectral_features' / (split + '_' + identity[:20])
    return base.with_suffix('.npy'), base.with_suffix('.json')


def extract_spectral(args):
    directory = activate_dataset(args.dataset)
    torch.set_num_threads(args.threads)
    rows_all = core.audit_dataset(verify_wavs=False)
    config = spectral_config(args.prosody)
    model = AlignedMel(config).eval()
    dimension = model.feature_dimension
    for split in args.splits:
        rows = [r for r in rows_all if r['split'] == split]
        if split == 'train':
            rows = select_training_rows(rows, args.train_limit, args.seed)
        path, marker = spectral_paths(directory, rows, split, config)
        if path.exists() and marker.exists():
            spectral_cache(directory, rows, split, config)
            print('Verified spectral cache reused:', split, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial.npy')
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(len(rows), 24, dimension))
        with torch.inference_mode():
            for start in range(0, len(rows), args.batch_size):
                subset = rows[start:start + args.batch_size]
                waves = []
                for row in subset:
                    wav = directory / row['path']
                    if core.digest(wav) != row['audio_sha256']:
                        raise ValueError('Spectral input WAV changed.')
                    audio, _ = core.load_audio(wav)
                    if len(audio) != 8000:
                        raise ValueError('Spectral cache requires a 0.5-second window.')
                    waves.append(audio)
                values = model(torch.from_numpy(np.stack(waves))).numpy().astype(np.float16)
                if not np.isfinite(values).all() or values.shape[1:] != (24, dimension):
                    raise ValueError('Invalid spectral features.')
                output[start:start + len(subset)] = values
                completed = start + len(subset)
                if completed % 1024 == 0 or completed == len(rows):
                    print(f'Spectral features {split}: {completed}/{len(rows)}', flush=True)
        output.flush()
        del output
        temporary.replace(path)
        core.write_json(marker, {'sha256': core.digest(path), 'row_ids': [r['sample_id'] for r in rows],
                                 'shape': [len(rows), 24, dimension], 'config': config})


def spectral_config(prosody=False):
    config = dict(SPECTRAL)
    if prosody:
        config['prosody'] = {'recipe': 'normalized-autocorrelation-v1', 'minimum_f0': 70.,
                             'maximum_f0': 400., 'voicing_threshold': .7}
    return config


def spectral_cache(directory, rows, split, config=None):
    config = config or SPECTRAL
    path, marker = spectral_paths(directory, rows, split, config)
    if not path.exists() or not marker.exists():
        raise ValueError('Run extract-spectral for ' + split)
    metadata = core.read_json(marker)
    if (core.digest(path) != metadata['sha256'] or metadata['row_ids'] != [r['sample_id'] for r in rows]
            or metadata['config'] != config):
        raise ValueError('Unverified spectral cache.')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    dimension = config['n_mels'] + (8 if config.get('prosody') else 0)
    if values.shape != (len(rows), 24, dimension):
        raise ValueError('Unexpected spectral cache shape.')
    return values


class AugmentedFeatures:
    """Join small cached batches, without copying the full acoustic feature corpus into RAM."""
    def __init__(self, acoustic, spectral, metadata):
        self.acoustic, self.spectral = acoustic, spectral
        self.mean = np.array(metadata['mean'], dtype=np.float32)
        self.std = np.array(metadata['std'], dtype=np.float32)
        if len(acoustic) != len(spectral) or not np.all(self.std > 0):
            raise ValueError('Invalid joint feature normalization.')

    def __len__(self):
        return len(self.acoustic)

    def __getitem__(self, index):
        spectral = (np.array(self.spectral[index], dtype=np.float32) - self.mean) / self.std
        return np.concatenate([np.asarray(self.acoustic[index], dtype=np.float32), spectral], axis=-1)


class JoinedFeatures:
    """Stack verified memmaps and selected row indices, allocating only a requested batch."""
    def __init__(self, parts):
        self.parts = [(values, np.asarray(indices, dtype=np.int64)) for values, indices in parts if len(indices)]
        lengths = [len(indices) for _, indices in self.parts]
        self.offsets = np.cumsum([0, *lengths])
        self.shape = (int(self.offsets[-1]), *self.parts[0][0].shape[1:])
        if any(values.shape[1:] != self.shape[1:] for values, _ in self.parts):
            raise ValueError('Cannot join different frame geometries.')

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        scalar = np.isscalar(index)
        requested = np.arange(len(self))[index] if isinstance(index, slice) else np.atleast_1d(index)
        requested = np.asarray(requested, dtype=np.int64)
        if np.any(requested < 0) or np.any(requested >= len(self)):
            raise IndexError('Joint feature index out of range.')
        output = np.empty((len(requested), *self.shape[1:]), dtype=np.float16)
        for part, (values, indices) in enumerate(self.parts):
            mask = (requested >= self.offsets[part]) & (requested < self.offsets[part + 1])
            if np.any(mask):
                output[mask] = values[indices[requested[mask] - self.offsets[part]]]
        return output[0] if scalar else output


def native_augmentation(directory, foreign_rows, spectral=False, config=None):
    if directory not in (V3, V4, V5):
        raise ValueError('Native augmentation requires the fixed V3 speaker/prompt partition.')
    selection = core.read_json(directory / 'data/manifests/source_selection.json')
    reserved = set(selection['prompt_groups']['validation'] + selection['prompt_groups']['test'])
    previous_root = core.ROOT
    try:
        core.ROOT = V2
        native_all = core.audit_dataset(verify_wavs=False)
    finally:
        core.ROOT = previous_root
    foreign_speakers = {r['speaker_id'].lower() for r in foreign_rows}
    foreign_pcm = {r['pcm_sha256'] for r in foreign_rows}
    rows, acoustic_parts, spectral_parts = [], [], []
    for split in core.SPLITS:
        original = [r for r in native_all if r['split'] == split]
        indices = [i for i, row in enumerate(original) if row['prompt_id'] not in reserved]
        if not indices:
            continue
        acoustic_parts.append((feature_cache(V2, original, split), indices))
        if spectral:
            spectral_parts.append((spectral_cache(V2, original, split, config), indices))
        for index in indices:
            row = original[index]
            if row['speaker_id'].lower() in foreign_speakers or row['pcm_sha256'] in foreign_pcm:
                raise ValueError('Native augmentation overlaps current L2 evaluation audio/speakers.')
            rows.append(dict(row, split='train', historical_split=split, augmentation_source='CMU-ARCTIC V2 retired partitions'))
    metadata = {'dataset': 'CMU-ARCTIC V2', 'manifest_sha256': core.digest(V2 / 'data/manifests/dataset.csv'),
                'selected_rows_sha256': core.object_digest([(r['sample_id'], r['audio_sha256']) for r in rows]),
                'reserved_prompt_ids': sorted(reserved), 'files': len(rows),
                'speakers': sorted({r['speaker_id'] for r in rows}),
                'annotation_origin': 'Automatically aligned CMU ax; historical V2 validation/test are retired training data.'}
    return rows, acoustic_parts, spectral_parts, metadata


def spectral_statistics(values):
    dimension = values.shape[-1]
    total, squares, count = np.zeros(dimension, dtype=np.float64), np.zeros(dimension, dtype=np.float64), 0
    for start in range(0, len(values), 256):
        batch = np.asarray(values[start:start + 256], dtype=np.float64).reshape(-1, dimension)
        total += batch.sum(axis=0)
        squares += np.square(batch).sum(axis=0)
        count += len(batch)
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean ** 2, 1e-8))
    return mean.astype(np.float32).tolist(), np.maximum(std, 1e-4).astype(np.float32).tolist()


class GroupFeatureNorm(nn.Module):
    """Preserve both acoustic embedding and phoneme-logit variation before mixing."""
    def __init__(self, sizes):
        super().__init__()
        if not sizes or any(size <= 0 for size in sizes):
            raise ValueError('Feature groups must have positive dimensions.')
        self.sizes = tuple(sizes)
        self.norms = nn.ModuleList([nn.LayerNorm(size) for size in sizes])

    def forward(self, values):
        return torch.cat([norm(group) for norm, group in zip(self.norms, values.split(self.sizes, dim=-1))], dim=-1)


class PhoneHead(nn.Module):
    def __init__(self, classes, hidden=128, dropout=0.2, temporal='none', input_dim=768, group_sizes=None):
        super().__init__()
        self.temporal = temporal
        if group_sizes is not None and sum(group_sizes) != input_dim:
            raise ValueError('Feature group dimensions do not sum to the head input size.')
        normalizer = GroupFeatureNorm(group_sizes) if group_sizes else nn.LayerNorm(input_dim)
        if temporal != 'none':
            if hidden < 2 or hidden % 2:
                raise ValueError('Temporal head requires a positive even hidden size.')
            self.projection = nn.Sequential(normalizer, nn.Linear(input_dim, hidden), nn.GELU())
            if temporal == 'conv':
                self.context = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Dropout(dropout),
                                             nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
            elif temporal == 'gru':
                self.context = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
            else:
                raise ValueError('Unknown temporal head.')
            self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, classes))
        elif hidden:
            self.network = nn.Sequential(normalizer, nn.Linear(input_dim, hidden), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(hidden, classes))
        else:
            self.network = nn.Sequential(normalizer, nn.Dropout(dropout), nn.Linear(input_dim, classes))

    def forward(self, values):
        if self.temporal != 'none':
            projected = self.projection(values)
            if self.temporal == 'conv':
                context = self.context(projected.transpose(1, 2)).transpose(1, 2)
            else:
                context, _ = self.context(projected)
            return self.classifier(projected + context)
        return self.network(values)


def class_scores(model, features, target_index, pool_frames=1, batch_size=128):
    model.eval()
    result = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            values = torch.from_numpy(np.array(features[start:start + batch_size], dtype=np.float32))
            score = torch.softmax(model(values), dim=-1)[:, :, target_index]
            if pool_frames > 1:
                score = torch.nn.functional.avg_pool1d(score.unsqueeze(1), pool_frames, stride=1).squeeze(1)
            result.extend(score.amax(dim=1).tolist())
    return np.asarray(result)


def validation_selection(labels, scores, objective='f1'):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Validation requires both classes.')
    thresholds = np.round(np.arange(.01, .991, .01), 3)
    prediction = scores[:, None] >= thresholds[None, :]
    positives = labels[:, None] == 1
    tp = np.sum(prediction & positives, axis=0)
    fp = np.sum(prediction & ~positives, axis=0)
    fn = np.sum(~prediction & positives, axis=0)
    tn = np.sum(~prediction & ~positives, axis=0)
    f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
    accuracy = (tp + tn) / len(labels)
    fpr = fp / np.maximum(1, fp + tn)
    # F1 protects against reporting high accuracy by rejecting every positive.
    if objective not in ('f1', 'min_pr'):
        raise ValueError('Unsupported validation selection objective.')
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / np.maximum(1, tp + fn)
    primary = np.minimum(precision, recall) if objective == 'min_pr' else f1
    index = max(range(len(thresholds)), key=lambda i: (primary[i], f1[i], accuracy[i], -fpr[i], thresholds[i]))
    return core.binary_metrics(labels, scores, thresholds[index])


def initialize_ctc_head(model, state):
    """Start a linear presence classifier at the pretrained bare-schwa phoneme logit."""
    if state.get('feature_kind') != 'ctc_logits' or not hasattr(model, 'network'):
        raise ValueError('Phoneme initialization needs CTC features and a linear head.')
    linear = model.network[-1]
    if len(model.network) != 3 or linear.out_features != 2:
        raise ValueError('Phoneme initialization requires --hidden 0 and --task presence.')
    index = state['phoneme_vocabulary'].get('\u0259')
    if index is None or index >= linear.in_features:
        raise ValueError('Encoder has no bare schwa phoneme.')
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[1, index] = .25


def window_presence_loss(logits, target_index, labels):
    """Stable BCE on the most positive frame, including confidently wrong predictions."""
    others = torch.cat([logits[..., :target_index], logits[..., target_index + 1:]], dim=-1)
    log_odds = logits[..., target_index] - torch.logsumexp(others, dim=-1)
    return nn.functional.binary_cross_entropy_with_logits(log_odds.amax(dim=1), labels)


def train_head(args):
    directory = activate_dataset(args.dataset)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    training_code_sha256 = core.digest(__file__)
    if args.prosody:
        args.spectral = True
    spectral_recipe = spectral_config(args.prosody)
    # This stage consumes checksum-verified feature caches, rather than generated WAV files.
    rows = core.audit_dataset(verify_wavs=False)
    selection = core.read_json(directory / 'data/manifests/source_selection.json')
    source_metadata = {k: selection[k] for k in ('source_repo', 'source_revision', 'license', 'annotation_origin',
                                                'target_mapping', 'unknown_policy', 'initialization_policy', 'test_independence',
                                                'prior_test_manifest_sha256') if k in selection}
    train_rows = [r for r in rows if r['split'] == 'train']
    train_rows = select_training_rows(train_rows, args.train_limit, args.seed)
    val_rows = [r for r in rows if r['split'] == 'validation']
    train_features = feature_cache(directory, train_rows, 'train')
    val_features = feature_cache(directory, val_rows, 'validation')
    foreign_train_rows = train_rows
    targets, vocabulary = frame_labels(directory, train_rows)
    native_rows, native_spectral_parts, augmentation_metadata = [], [], None
    if args.native_augmentation:
        native_rows, native_acoustic_parts, native_spectral_parts, augmentation_metadata = native_augmentation(directory, rows, args.spectral, spectral_recipe)
        native_targets, native_vocabulary = frame_labels(V2, native_rows)
        combined_vocabulary = sorted(set(vocabulary) | set(native_vocabulary))
        def remap(labels, old_vocabulary):
            indices = np.array([combined_vocabulary.index(p) for p in old_vocabulary])
            result = np.full_like(labels, -100)
            valid = labels >= 0
            result[valid] = indices[labels[valid]]
            return result
        targets = np.concatenate([remap(targets, vocabulary), remap(native_targets, native_vocabulary)])
        vocabulary = combined_vocabulary
        train_features = JoinedFeatures([(train_features, np.arange(len(foreign_train_rows))), *native_acoustic_parts])
        train_rows = foreign_train_rows + native_rows
        source_metadata['native_augmentation'] = augmentation_metadata
        print('Native training augmentation:', len(native_rows), 'WAV; L2 validation/test unchanged.', flush=True)
    spectral_metadata = None
    if args.spectral:
        raw_train = spectral_cache(directory, foreign_train_rows, 'train', spectral_recipe)
        if native_rows:
            raw_train = JoinedFeatures([(raw_train, np.arange(len(foreign_train_rows))), *native_spectral_parts])
        raw_val = spectral_cache(directory, val_rows, 'validation', spectral_recipe)
        mean, std = spectral_statistics(raw_train)
        spectral_metadata = {'config': spectral_recipe, 'state_dict': AlignedMel(spectral_recipe).state_dict(),
                             'mean': mean, 'std': std, 'normalization_split': 'train'}
        train_features = AugmentedFeatures(train_features, raw_train, spectral_metadata)
        val_features = AugmentedFeatures(val_features, raw_val, spectral_metadata)
    ax = vocabulary.index('ax')
    if args.task in ('binary', 'presence'):
        targets = np.where(targets == -100, -100, (targets == ax).astype(np.int64))
        vocabulary, ax = ['non_ax', 'ax'], 1
    train_labels = np.array([int(r['label']) for r in train_rows], dtype=np.float32)
    val_labels = np.array([int(r['label']) for r in val_rows])
    config = {'classes': len(vocabulary), 'hidden': args.hidden, 'dropout': args.dropout,
              'temporal': args.temporal, 'input_dim': backbone_dimension() + (len(spectral_metadata['mean']) if args.spectral else 0)}
    if getattr(args, 'group_normalization', False):
        if REPRESENTATION != 'joint' or args.initialize_ctc:
            raise ValueError('Group normalization requires joint features and a fresh head.')
        embedding_dim = _backbone_geometry(str(BACKBONE), core.digest(BACKBONE))[3]
        config['group_sizes'] = [embedding_dim, backbone_dimension() - embedding_dim]
        if spectral_metadata:
            config['group_sizes'].append(len(spectral_metadata['mean']))
    model = PhoneHead(**config)
    if args.initialize_ctc:
        if args.task != 'presence':
            raise ValueError('--initialize-ctc requires --task presence.')
        initialize_ctc_head(model, torch.load(BACKBONE, map_location='cpu', weights_only=True))
    counts = np.bincount(targets[targets >= 0], minlength=len(vocabulary))
    class_weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    class_weights = np.clip(class_weights / np.median(class_weights), .25, 3)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(class_weights.astype(np.float32)), ignore_index=-100)
    if args.task == 'presence' and args.window_loss <= 0:
        raise ValueError('Presence training requires a positive --window-loss.')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    settings = {'dataset': args.dataset, 'batch_size': args.batch_size, 'epochs': args.epochs,
                'patience': args.patience, 'learning_rate': args.lr, 'optimizer': 'AdamW', 'weight_decay': .01,
                'threads': args.threads, 'seed': args.seed, 'head_config': config,
                'task': args.task, 'window_loss_weight': args.window_loss,
                'feature_context': CONTEXT,
                'acoustic_representation': REPRESENTATION,
                'spectral_features': args.spectral,
                'prosody_features': args.prosody,
                'training_subset': {'limit': args.train_limit, 'files': len(foreign_train_rows),
                                    'selected_rows_sha256': core.object_digest([r['sample_id'] for r in foreign_train_rows]),
                                    'selection': 'Train speaker/class round robin, seeded shuffle; validation/test unchanged.'},
                'frame_supervision': args.task != 'presence',
                'phoneme_initialization': args.initialize_ctc,
                'phoneme_initialization_gain': .25 if args.initialize_ctc else None,
                'window_loss_implementation': 'BCEWithLogits on maximum ax log odds',
                'native_augmentation': augmentation_metadata,
                'selection_objective': args.selection,
                'selection_rule': ('Validation min(precision,recall), then F1, accuracy, lower FPR; no test selection.'
                                   if args.selection == 'min_pr' else 'Validation F1, then accuracy, then lower FPR; no test selection.')}
    if (args.run_dir / 'best.pt').exists():
        raise ValueError('Preserving existing run; choose another run directory.')
    best_score, bad, history = (-1.0,), 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            values = torch.from_numpy(np.array(train_features[indices], dtype=np.float32))
            truth = torch.from_numpy(targets[indices])
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = criterion(logits.transpose(1, 2), truth) if args.task != 'presence' else logits.sum() * 0
            if args.window_loss:
                window_truth = torch.from_numpy(train_labels[indices])
                loss = loss + args.window_loss * window_presence_loss(logits, ax, window_truth)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite frame classification loss.')
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        selections = []
        for pool_frames in (1, 2, 3):
            scores = class_scores(model, val_features, ax, pool_frames)
            metric = validation_selection(val_labels, scores, args.selection)
            selections.append((metric, pool_frames))
        def selection_key(metric):
            if args.selection == 'min_pr':
                return (min(metric['precision'] or 0., metric['recall'] or 0.), metric['f1'], metric['accuracy'], -metric['fpr'])
            return (metric['f1'],)
        metric, pool_frames = max(selections, key=lambda r: (*selection_key(r[0]), r[0]['accuracy'], -r[0]['fpr']))
        item = {'epoch': epoch, 'train_frame_loss': float(np.mean(losses)), 'validation_accuracy': metric['accuracy'],
                'validation_precision': metric['precision'], 'validation_recall': metric['recall'], 'validation_f1': metric['f1'],
                'threshold': metric['threshold'], 'pool_frames': pool_frames, 'seconds': time.perf_counter() - started}
        history.append(item)
        if selection_key(metric) > best_score:
            best_score, bad = selection_key(metric), 0
            core.save_torch(args.run_dir / 'best.pt', {'head_state_dict': model.state_dict(), 'head_config': config,
                            'backend': 'wav2vec2_phone_head', 'settings': settings,
                            'phone_vocabulary': vocabulary, 'target_symbol': '\u0259', 'target_phones': ['ax'],
                            'threshold': metric['threshold'], 'pool_frames': pool_frames, 'epoch': epoch,
                            'validation_metrics': metric, 'threshold_selection_split': 'validation',
                            'dataset_version': args.dataset, 'dataset_sha256': core.digest(directory / 'data/manifests/dataset.csv'),
                            'backbone_sha256': core.digest(BACKBONE), 'code_sha256': training_code_sha256,
                            'device': 'cpu', 'cpu_threads': args.threads, 'label_boundary_margin': .005,
                            'seed': args.seed, 'frame_class_counts': counts.tolist(), 'class_weights': class_weights.tolist(),
                            'source_metadata': source_metadata, 'spectral': spectral_metadata})
        else:
            bad += 1
        core.write_csv(args.run_dir / 'history.csv', history)
        core.save_torch(args.run_dir / 'last.pt', {'head_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                         'epoch': epoch, 'history': history, 'torch_rng_state': torch.get_rng_state()})
        print(f"Head epoch {epoch:02d}: frame_loss={item['train_frame_loss']:.4f}, val_acc={metric['accuracy']:.4f}, P={metric['precision']:.4f}, R={metric['recall']:.4f}, F1={metric['f1']:.4f}, pool={pool_frames}, threshold={metric['threshold']:.2f}", flush=True)
        if bad >= args.patience:
            break
    best = torch.load(args.run_dir / 'best.pt', map_location='cpu', weights_only=True)
    core.write_json(args.run_dir / 'training_summary.json', {'best_epoch': best['epoch'], 'epochs_completed': len(history),
                    'seconds': time.perf_counter() - started, 'validation_metrics': best['validation_metrics'],
                    'dataset_version': args.dataset, 'parameters_head': sum(p.numel() for p in model.parameters()),
                    'phone_vocabulary': vocabulary, 'settings': settings, 'dataset_sha256': best['dataset_sha256'],
                    'backbone_sha256': best['backbone_sha256'], 'code_sha256': best['code_sha256']})


class TransferDetector:
    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        self.checkpoint = torch.load(self.path, map_location='cpu', weights_only=True, mmap=True)
        state = self.checkpoint
        if state.get('backend') != 'wav2vec2_phone_head' or state.get('target_phones') != ['ax']:
            raise ValueError('Not a schwa acoustic transfer checkpoint.')
        torch.set_num_threads(state['cpu_threads'])
        if 'backbone' in state:
            acoustic = state['backbone']
        else:
            if core.digest(BACKBONE) != state['backbone_sha256']:
                raise ValueError('The frozen acoustic backbone changed.')
            acoustic = torch.load(BACKBONE, map_location='cpu', weights_only=True, mmap=True)
        acoustic = representation_state(acoustic, state['settings'].get('acoustic_representation', 'stored'))
        self.acoustic = rebuild_acoustic(acoustic)
        self.layers = acoustic['layers']
        self.normalize_waveform = acoustic.get('normalize_waveform', False)
        self.feature_context = state['settings'].get('feature_context', 'window')
        self.head = PhoneHead(**state['head_config']).eval()
        self.head.load_state_dict(state['head_state_dict'])
        self.ax = state['phone_vocabulary'].index('ax')
        self.spectral = None
        if state.get('spectral'):
            metadata = state['spectral']
            self.spectral = AlignedMel(metadata['config']).eval()
            self.spectral.load_state_dict(metadata['state_dict'])
            self.spectral_mean = torch.tensor(metadata['mean'], dtype=torch.float32)
            self.spectral_std = torch.tensor(metadata['std'], dtype=torch.float32)

    def describe(self):
        state = self.checkpoint
        return {'weights': str(self.path), 'sha256': core.digest(self.path), 'backend': state['backend'],
                'target_symbol': state['target_symbol'], 'target_phones': state['target_phones'],
                'epoch': state['epoch'], 'threshold': state['threshold'], 'device': 'cpu', 'denoise': 'off',
                'window_seconds': .5, 'pool_frames': state['pool_frames'], 'backbone_layers': self.layers,
                'self_contained': 'backbone' in state, 'source_metadata': state.get('source_metadata', {}),
                'spectral_features': self.spectral is not None,
                'normalize_waveform_per_file': self.normalize_waveform,
                'feature_context': self.feature_context,
                'acoustic_feature_kind': self.acoustic.feature_kind,
                'encoder_quantization': self.acoustic.quantization,
                'encoder_attention_backend': ('math' if self.acoustic.feature_kind in ('ctc_logits', 'ctc_embeddings_logits')
                                              else 'default'),
                'parameters': (sum(p.numel() for p in self.acoustic.parameters()) + sum(p.numel() for p in self.head.parameters())
                               + sum(m.weight().numel() + (m.bias().numel() if m.bias() is not None else 0)
                                     for m in self.acoustic.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)))}

    def predict(self, path, hop_seconds=.05):
        if not 0 < hop_seconds <= .5:
            raise ValueError('Hop must be in (0, 0.5] seconds.')
        waveform, metadata = core.load_audio(path)
        hop = max(1, round(hop_seconds * 16000))
        starts = list(range(0, max(1, len(waveform) - 8000 + 1), hop))
        last = max(0, len(waveform) - 8000)
        if starts[-1] != last:
            starts.append(last)
        selected, maximum = [], 0.
        started = time.perf_counter()
        with torch.inference_mode():
            sequence = (recording_features(self.acoustic, torch.from_numpy(waveform), self.layers)
                        if self.feature_context == 'recording' else None)
            for index in range(0, len(starts), 8):
                subset = starts[index:index + 8]
                batch = np.stack([core.crop_window(waveform, start) for start in subset])
                if sequence is not None:
                    values = aligned_window_features(sequence, subset)
                else:
                    inputs = normalized_waveforms(torch.from_numpy(batch), self.normalize_waveform)
                    features = acoustic_features(self.acoustic, inputs, self.layers)
                    # Match the feature-cache precision used in training exactly.
                    values = features.to(torch.float16).to(torch.float32)
                if self.spectral is not None:
                    spectral = self.spectral(torch.from_numpy(batch)).to(torch.float16).to(torch.float32)
                    values = torch.cat([values, (spectral - self.spectral_mean) / self.spectral_std], dim=-1)
                scores = torch.softmax(self.head(values), dim=-1)[:, :, self.ax]
                if self.checkpoint['pool_frames'] > 1:
                    scores = torch.nn.functional.avg_pool1d(scores.unsqueeze(1), self.checkpoint['pool_frames'], stride=1).squeeze(1)
                scores = scores.amax(dim=1).tolist()
                maximum = max(maximum, max(scores))
                for start, score in zip(subset, scores):
                    if score >= self.checkpoint['threshold']:
                        selected.append({'start_seconds': start / 16000, 'end_seconds': min(len(waveform), start + 8000) / 16000,
                                         'score': float(score)})
        return {'audio': str(Path(path).resolve()), 'target_symbol': '\u0259', 'score': maximum,
                'threshold': self.checkpoint['threshold'], 'detected': bool(selected), 'windows': selected,
                'duration_seconds': metadata['duration_seconds'], 'denoise': 'off', 'scanned_windows': len(starts),
                'feature_context': self.feature_context,
                'processing_seconds': time.perf_counter() - started,
                'note': 'Window scores detect ax; regions are approximate, not exact phone boundaries.'}


def predict(args):
    detector = TransferDetector(args.model)
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
            print(f"  {result['duration_seconds']:.3f}s audio, {len(result['windows'])} cửa sổ vượt ngưỡng; CPU {result['processing_seconds']:.2f}s")
            for window in result['windows'][:8]:
                print(f"  {window['start_seconds']:.2f}–{window['end_seconds']:.2f}s: {window['score']:.4f}")
    if args.output:
        core.write_json(args.output, results)


def pack(args):
    state = torch.load(args.model, map_location='cpu', weights_only=True)
    if core.digest(BACKBONE) != state['backbone_sha256']:
        raise ValueError('Backbone fingerprint differs from training.')
    state['backbone'] = representation_state(torch.load(BACKBONE, map_location='cpu', weights_only=True), state['settings'].get('acoustic_representation', 'stored'))
    core.save_torch(args.output, state)
    print('Self-contained model:', args.output, args.output.stat().st_size, 'bytes', flush=True)


def evaluate(args):
    global CONTEXT, REPRESENTATION
    directory = activate_dataset(args.dataset)
    state = torch.load(args.model, map_location='cpu', weights_only=True)
    expected_context = state['settings'].get('feature_context', 'window')
    if args.context is not None and args.context != expected_context:
        raise ValueError('Evaluation input context differs from training.')
    CONTEXT = expected_context
    REPRESENTATION = state['settings'].get('acoustic_representation', 'stored')
    if state['backbone_sha256'] != core.digest(BACKBONE):
        raise ValueError('Use --backbone with the encoder used during training for cached-feature evaluation.')
    if state['dataset_sha256'] != core.digest(directory / 'data/manifests/dataset.csv'):
        raise ValueError('Evaluation dataset differs from training.')
    if (args.output / 'metrics.json').exists():
        raise ValueError('Evaluation already exists; preserving frozen results.')
    rows = [r for r in core.audit_dataset(verify_wavs=False) if r['split'] == args.split]
    features = feature_cache(directory, rows, args.split)
    if state.get('spectral'):
        features = AugmentedFeatures(features, spectral_cache(directory, rows, args.split, state['spectral']['config']), state['spectral'])
    model = PhoneHead(**state['head_config']).eval()
    model.load_state_dict(state['head_state_dict'])
    torch.set_num_threads(state['cpu_threads'])
    scores = class_scores(model, features, state['phone_vocabulary'].index('ax'), state['pool_frames'])
    labels = np.asarray([int(r['label']) for r in rows])
    metric = core.binary_metrics(labels, scores, state['threshold'])
    predictions = [{**row, 'score': float(score), 'prediction': int(score >= state['threshold'])} for row, score in zip(rows, scores)]
    core.write_csv(args.output / 'predictions.csv', predictions)
    fields = list(predictions[0])
    for name, truth, predicted in (('false_positives', '0', 1), ('false_negatives', '1', 0)):
        core.write_csv(args.output / (name + '.csv'), [r for r in predictions if r['label'] == truth and r['prediction'] == predicted], fields)
    groups = {}
    for field in ('speaker_id', 'center_phone'):
        groups[field] = {}
        for value in sorted({r[field] for r in rows}):
            indices = [i for i, row in enumerate(rows) if row[field] == value]
            groups[field][value] = core.binary_metrics(labels[indices], scores[indices], state['threshold'])
    report = {'split': args.split, 'dataset_version': args.dataset, 'overall': metric, 'groups': groups,
              'feature_context': CONTEXT, 'evaluation_input': ('Original recordings; unchanged 0.5-second focal regions.'
                                                              if CONTEXT == 'recording' else 'Independent 0.5-second WAV files.'),
              'source_recordings': len({r['recording_id'] for r in rows}),
              'acoustic_representation': REPRESENTATION,
              'acoustic_runtime_recipe': RECORDING if CONTEXT == 'recording' else
                                        {'attention_backend': _backbone_geometry(str(BACKBONE), core.digest(BACKBONE))[2]},
              'model_sha256': core.digest(args.model), 'dataset_sha256': state['dataset_sha256'],
              'inference_code_sha256': {'improve.py': core.digest(__file__), 'frozen_phoneme.py': core.digest(ROOT / 'frozen_phoneme.py')},
              'threshold_selection_split': state['threshold_selection_split'], 'threshold_frozen': True,
              'speakers': sorted({r['speaker_id'] for r in rows}), 'always_negative_accuracy': float(np.mean(labels == 0)),
              'limitations': [('L2-ARCTIC manually annotated perceived phones; generated windows have not been listened to individually.'
                               if args.dataset in ('v3', 'v4', 'v5') else 'Automatically aligned phone labels; no human pronunciation review.'),
                              'Overlapping windows are correlated; benchmark accuracy does not establish real recording accuracy.']}
    core.write_json(args.output / 'metrics.json', report)
    print(json.dumps(report['overall'], indent=2), flush=True)


def main():
    global BACKBONE, CONTEXT, REPRESENTATION
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare-backbone')
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.add_argument('--bundle', choices=['ssl', 'asr', 'xlsr'], default='ssl')
    p.add_argument('--layers', choices=[6, 9, 12], type=int, default=6)
    p.add_argument('--threads', type=int, default=2)
    p.set_defaults(function=prepare_backbone)
    p = commands.add_parser('download-v2')
    p.add_argument('--workers', type=int, default=4)
    p.set_defaults(function=download_v2)
    p = commands.add_parser('prepare-v2')
    p.set_defaults(function=lambda args: (activate_dataset('v2'), core.make_dataset(argparse.Namespace(seed=20261010))))
    p = commands.add_parser('extract-spectral')
    p.add_argument('--dataset', choices=['v1', 'v2', 'v3', 'v4', 'v5'], default='v3')
    p.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--prosody', action='store_true')
    p.add_argument('--train-limit', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(function=extract_spectral)
    p = commands.add_parser('extract')
    p.add_argument('--representation', choices=['stored', 'joint'], default='stored')
    p.add_argument('--context', choices=['window', 'recording'], default='window')
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.add_argument('--reuse-dataset', choices=['v1', 'v2', 'v3', 'v4', 'v5'])
    p.add_argument('--dataset', choices=['v1', 'v2', 'v3', 'v4', 'v5'], default='v1')
    p.add_argument('--splits', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(function=extract)
    p = commands.add_parser('train')
    p.add_argument('--representation', choices=['stored', 'joint'], default='stored')
    p.add_argument('--selection', choices=['f1', 'min_pr'], default='f1')
    p.add_argument('--context', choices=['window', 'recording'], default='window')
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.add_argument('--dataset', choices=['v1', 'v2', 'v3', 'v4', 'v5'], default='v1')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--temporal', choices=['none', 'conv', 'gru'], default='none')
    p.add_argument('--task', choices=['phones', 'binary', 'presence'], default='phones')
    p.add_argument('--window-loss', type=float, default=0)
    p.add_argument('--spectral', action='store_true')
    p.add_argument('--prosody', action='store_true', help='Append pitch, voicing and energy features; implies --spectral.')
    p.add_argument('--native-augmentation', action='store_true')
    p.add_argument('--dropout', type=float, default=.2)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--initialize-ctc', action='store_true')
    p.add_argument('--group-normalization', action='store_true', help='Normalize joint embedding/logit groups separately.')
    p.set_defaults(function=train_head)
    p = commands.add_parser('predict')
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--audio', type=Path, nargs='+')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--hop', type=float, default=.05)
    p.add_argument('--json', action='store_true')
    p.add_argument('--output', type=Path)
    p.set_defaults(function=predict)
    p = commands.add_parser('pack')
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=pack)
    p = commands.add_parser('evaluate')
    p.add_argument('--context', choices=['window', 'recording'])
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.add_argument('--dataset', choices=['v1', 'v2', 'v3', 'v4', 'v5'], required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--split', choices=['validation', 'test'], default='test')
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    if hasattr(args, 'backbone'):
        BACKBONE = args.backbone.resolve()
    if getattr(args, 'context', None) is not None:
        CONTEXT = args.context
    if hasattr(args, 'representation'):
        REPRESENTATION = args.representation
    args.function(args)


if __name__ == '__main__':
    main()

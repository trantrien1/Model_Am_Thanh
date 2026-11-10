"""Frozen public Whisper audio encoder features, aligned to original focal regions."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

import schwa as core
import whisper_cpu
from frozen_phoneme import RECORDING
from lexical_train_cpu import array_cache

ROOT = Path(__file__).resolve().parent
RECIPE = {'version': 1, 'dimension': 512, 'sample_rate': 16000,
          'mel_hop_samples': 160, 'encoder_hop_samples': 320,
          'encoder_center_sample': 0, 'maximum_samples': 480000,
          'core_samples': 320000, 'margin_samples': 80000,
          'short_recording': 'one zero-padded 30-second encoder input',
          'long_recording': '20-second cores with five-second context on either side',
          'normalization': 'official log_mel_spectrogram on each zero-padded encoder input',
          'output': 'final encoder LayerNorm; no decoder or reference inputs',
          'storage': 'float16', 'alignment': 'linear interpolation at original XLSR focal centers',
          'reference_transcript_consumed': False, 'manual_phone_timing_used_as_features': False}
SOURCES = ['whisper_acoustic_features.py', 'whisper_cpu.py', 'frozen_phoneme.py',
           'lexical_train_cpu.py', 'schwa.py']
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'


def metadata():
    if core.digest(whisper_cpu.MODEL) != whisper_cpu.MODEL_SHA:
        raise ValueError('Original public Whisper weight changed.')
    return {'model_sha256': whisper_cpu.MODEL_SHA,
            'package_version': whisper_cpu.VERSION,
            'package_files_sha256': whisper_cpu.package_identity(),
            'cpu_math': whisper_cpu.CPU_MATH, 'recipe': RECIPE,
            'focal_recipe': RECORDING,
            'source_sha256': {name: core.digest(ROOT / name) for name in SOURCES}}


def encode_piece(encoder, waveform):
    import whisper
    if not 0 < len(waveform) <= RECIPE['maximum_samples']:
        raise ValueError('Whisper encoder input must fit thirty seconds.')
    padded = whisper.pad_or_trim(waveform, length=RECIPE['maximum_samples'])
    mel = whisper.log_mel_spectrogram(padded, n_mels=80)
    with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
        values = encoder(mel.unsqueeze(0))[0]
    if values.shape != (1500, RECIPE['dimension']) or not torch.isfinite(values).all():
        raise ValueError('Unexpected public Whisper encoder output.')
    return values.to(torch.float16)


def recording_features(encoder, waveform):
    if waveform.ndim != 1 or not len(waveform) or not torch.isfinite(waveform).all():
        raise ValueError('Nonempty finite mono 16 kHz audio required.')
    hop = RECIPE['encoder_hop_samples']
    frame_count = (len(waveform) + hop - 1) // hop
    if len(waveform) <= RECIPE['maximum_samples']:
        return encode_piece(encoder, waveform)[:frame_count]
    result = torch.empty((frame_count, RECIPE['dimension']), dtype=torch.float16)
    for start in range(0, len(waveform), RECIPE['core_samples']):
        end = min(start + RECIPE['core_samples'], len(waveform))
        left = max(0, start - RECIPE['margin_samples'])
        right = min(len(waveform), end + RECIPE['margin_samples'])
        values = encode_piece(encoder, waveform[left:right])
        first, stop = start // hop, (end + hop - 1) // hop
        offset = (start - left) // hop
        count = stop - first
        if offset + count > len(values):
            raise ValueError('Whisper chunk timeline differs.')
        result[first:stop] = values[offset:offset + count]
    return result


def aligned_windows(sequence, starts):
    if sequence.ndim != 2 or not len(sequence) or not torch.isfinite(sequence).all():
        raise ValueError('Finite recording encoder sequence required.')
    positions = (torch.as_tensor(starts, dtype=torch.float64).reshape(-1, 1)
                 + RECORDING['center_sample']
                 + torch.arange(RECORDING['window_frames'], dtype=torch.float64) * RECORDING['hop_samples'])
    positions = (positions / RECIPE['encoder_hop_samples']).clamp(0, len(sequence) - 1)
    left = positions.floor().long()
    right = (left + 1).clamp(max=len(sequence) - 1)
    fraction = (positions - left).float().unsqueeze(-1)
    values = sequence.float()
    return (values[left] * (1 - fraction) + values[right] * fraction).half().float()


def window_path(directory, rows, split, recipe):
    if split not in ('train', 'validation') or any(r['split'] != split for r in rows):
        raise ValueError('Development extractor cannot consume test regions.')
    identity = {'recipe': recipe, 'row_ids': [r['sample_id'] for r in rows],
                'starts': [r['window_start_sample'] for r in rows]}
    return directory / 'data/whisper_acoustic_windows' / (split + '_' + core.object_digest(identity) + '.npy'), identity


def extract(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    directory = ROOT / 'v5'
    if core.digest(directory / 'data/manifests/dataset.csv') != DATASET_SHA:
        raise ValueError('Frozen V5 manifest changed.')
    recipe = metadata()
    all_rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
    encoder = None
    started = time.perf_counter()
    for split in args.splits:
        rows = [r for r in all_rows if r['split'] == split]
        if args.limit:
            allowed = set(sorted({r['recording_id'] for r in rows})[:args.limit])
            rows = [r for r in rows if r['recording_id'] in allowed]
        path, identity = window_path(directory, rows, split, recipe)
        if path.exists() and path.with_suffix('.json').exists():
            array_cache(path, identity)
            continue
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(row['recording_id'], []).append((index, row))
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix('.partial.npy')
        output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16,
                                           shape=(len(rows), 24, RECIPE['dimension']))
        for number, (rid, group) in enumerate(sorted(groups.items()), 1):
            record = records[rid]
            source = directory / record['audio_path']
            if record['split'] != split or core.digest(source) != record['audio_sha256']:
                raise ValueError('Source split or audio changed.')
            record_identity = {'recipe': recipe, 'source_audio_sha256': record['audio_sha256']}
            record_path = directory / 'data/whisper_acoustic_recordings' / (core.object_digest(record_identity) + '.npy')
            if record_path.exists() and record_path.with_suffix('.json').exists():
                cached = array_cache(record_path, record_identity)
                sequence = torch.from_numpy(np.array(cached, dtype=np.float32))
            else:
                if encoder is None:
                    model = whisper_cpu.load_model()
                    encoder = model.encoder.eval()
                    del model
                    gc.collect()
                audio, _ = core.load_audio(source)
                sequence = recording_features(encoder, torch.from_numpy(audio))
                from lexical_train_cpu import save_array
                save_array(record_path, sequence.numpy(), record_identity)
            if sequence.ndim != 2 or sequence.shape[-1] != RECIPE['dimension']:
                raise ValueError('Unexpected recording encoder geometry.')
            output[[index for index, _ in group]] = aligned_windows(sequence,
                [int(r['window_start_sample']) for _, r in group]).numpy()
            status = {'status': 'running', 'split': split, 'recordings_completed': number,
                      'recordings_total': len(groups), 'seconds': time.perf_counter() - started,
                      'reference_transcript_consumed': False, 'test_predictions_computed': False}
            core.write_json(ROOT / 'reports/whisper_acoustic_extraction_status.json', status)
            if number % 32 == 0 or number == len(groups):
                print(json.dumps(status), flush=True)
        output.flush()
        del output
        partial.replace(path)
        core.write_json(path.with_suffix('.json'), {'identity': identity, 'sha256': core.digest(path),
                                                   'shape': [len(rows), 24, RECIPE['dimension']]})
    core.write_json(ROOT / 'reports/whisper_acoustic_extraction_status.json',
        {'status': 'completed', 'splits': args.splits, 'limit': args.limit,
         'seconds': time.perf_counter() - started, 'reference_transcript_consumed': False,
         'test_predictions_computed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    if args.limit and args.splits != ['train']:
        parser.error('Limited feasibility extraction may use TRAIN only.')
    extract(args)

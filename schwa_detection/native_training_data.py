"""Prepare auxiliary native-speech TRAIN data disjoint from every V5 held-out prompt."""
import argparse
import hashlib
import io
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch

import improve
import schwa as core
from train_negative_pool import eligible_window

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'training_native'
V5 = ROOT / 'v5'
V5_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
SEED = 20261012


def source_selection():
    if core.digest(V5 / 'data/manifests/dataset.csv') != V5_SHA:
        raise ValueError('Original V5 evaluation dataset changed.')
    native = core.read_csv(ROOT / 'data/manifests/recordings.csv')
    original = core.read_csv(V5 / 'data/manifests/recordings.csv')
    heldout = [r for r in original if r['split'] != 'train']
    prompts = {r['prompt_id'] for r in heldout}
    speakers = {r['speaker_id'] for r in heldout}
    hashes = {r['audio_sha256'] for r in heldout}
    result = [r for r in native if r['prompt_id'] not in prompts and r['speaker_id'] not in speakers]
    if any(r['audio_sha256'] in hashes for r in result):
        raise ValueError('Native audio collides with V5 held-out audio.')
    return result


def prepare():
    records = source_selection()
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'v5_dataset_sha256': V5_SHA, 'native_source_manifest_sha256': core.digest(ROOT / 'data/manifests/recordings.csv'),
        'builder_sha256': core.digest(__file__), 'original_label_code_sha256': core.digest(ROOT / 'schwa.py'),
        'negative_QC_code_sha256': core.digest(ROOT / 'train_negative_pool.py'),
        'seed': SEED, 'native_recording_ids': sorted(r['recording_id'] for r in records),
        'annotation_origin': 'CMU ARCTIC automatically aligned phones; auxiliary lower-confidence TRAIN supervision',
        'historical_native_partitions': 'Retired native train/validation/test may be auxiliary TRAIN; V5 validation/test stay untouched',
        'target': 'Only native ax is positive; ah, er, axr and all other phones negative',
        'heldout_v5_prompt_speaker_audio_overlap': False, 'v5_test_audio_or_features_consumed': False}
    registered = DATA / 'registered_preparation.json'
    if registered.exists():
        saved = core.read_json(registered)
        if {k: v for k, v in saved.items() if k != 'created_utc'} != {
                k: v for k, v in protocol.items() if k != 'created_utc'}:
            raise ValueError('Preserve registered native preparation.')
    else:
        core.write_json(registered, protocol)
    positive, negative, converted, exclusions = [], [], [], []
    for number, record in enumerate(records, 1):
        for kind in ('audio', 'lab'):
            if core.digest(ROOT / record[kind + '_path']) != record[kind + '_sha256']:
                raise ValueError('Native source checksum changed.')
        audio, _ = core.load_audio(ROOT / record['audio_path'])
        phones, _ = core.parse_phones((ROOT / record['lab_path']).read_text(), len(audio) / 16000)
        converted.append({**record, 'historical_split': record['split'], 'split': 'train',
            'audio_path': '../' + record['audio_path'], 'lab_path': '../' + record['lab_path']})
        source_pcm = core.pcm_hash(audio)
        for index, phone in enumerate(phones):
            is_positive = phone['phone'] == 'ax'
            duration = phone['end'] - phone['start']
            if duration < .02 or phone['phone'] == 'unk':
                continue
            if is_positive and duration > .46:
                exclusions.append({'recording_id': record['recording_id'], 'phone_index': index, 'reason': 'target_too_long'})
                continue
            for variant in range(2 if is_positive else 1):
                identity = f"{record['recording_id']}_{index:03d}_{variant}"
                if is_positive:
                    start = round((phone['start'] + phone['end']) / 2 * 16000) - 4000
                    if variant:
                        low = math.ceil(phone['end'] * 16000 - 8000 + 320)
                        high = math.floor(phone['start'] * 16000 - 320)
                        start = int(core.seeded(SEED, identity).integers(low, high + 1))
                    end = start + 8000
                    if any(start / 16000 < p['end'] and end / 16000 > p['start'] for p in phones if p['phone'] == 'unk'):
                        continue
                else:
                    bounds = eligible_window(phone, phones)
                    if bounds is None:
                        continue
                    start, end = bounds
                window = core.crop_window(audio, start)
                if float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) < 1e-5:
                    continue
                # Fingerprint the exact PCM16 WAV bytes without writing thousands
                # of redundant files; recording-context extraction reads sources.
                encoded = io.BytesIO()
                sf.write(encoded, window, 16000, format='WAV', subtype='PCM_16')
                window_sha = hashlib.sha256(encoded.getvalue()).hexdigest()
                row = {'sample_id': identity, 'recording_id': record['recording_id'],
                    'speaker_id': record['speaker_id'], 'prompt_id': record['prompt_id'], 'split': 'train',
                    'label': str(int(is_positive)), 'center_phone': phone['phone'],
                    'phone_start': str(phone['start']), 'phone_end': str(phone['end']),
                    'window_start_sample': str(start), 'window_end_sample': str(end),
                    'source_path': '../' + record['audio_path'], 'source_sha256': record['audio_sha256'],
                    'source_pcm_sha256': source_pcm, 'seed': str(SEED),
                    'audio_sha256': window_sha,
                    'annotation_origin': 'CMU ARCTIC automatic phone alignment'}
                (positive if is_positive else negative).append(row)
        if number % 100 == 0 or number == len(records):
            print(f'Native TRAIN QC {number}/{len(records)} records: {len(positive)} positive regions', flush=True)
    selected = positive + core.balanced_negatives(negative, min(len(positive), len(negative)), core.seeded(SEED, 'train'))
    selected.sort(key=lambda r: r['sample_id'])
    for filename, values in [('recordings.csv', converted), ('dataset.csv', selected)]:
        path = DATA / 'data/manifests' / filename
        if path.exists():
            if core.read_csv(path) != values:
                raise ValueError('Preserve existing native auxiliary manifest.')
        else:
            core.write_csv(path, values)
    core.write_json(ROOT / 'reports/native_training_preparation_audit.json', {
        'passed': True, 'records': len(records), 'regions': len(selected),
        'labels': dict(Counter(r['label'] for r in selected)),
        'speakers': dict(Counter(r['speaker_id'] for r in records)),
        'native_historical_split_counts': dict(Counter(r['split'] for r in records)),
        'candidate_negative_regions': len(negative), 'exclusions': exclusions,
        'annotation_origin': protocol['annotation_origin'], 'heldout_v5_prompt_speaker_audio_overlap': False,
        'source_checksums_verified': True, 'v5_dataset_sha256': V5_SHA,
        'native_manifest_sha256': core.digest(DATA / 'data/manifests/dataset.csv'),
        'v5_test_audio_or_features_consumed': False, 'quality_evidence': False})
    print('Native auxiliary TRAIN manifest prepared.', flush=True)


def extract():
    audit = core.read_json(ROOT / 'reports/native_training_preparation_audit.json')
    protocol = core.read_json(DATA / 'registered_preparation.json')
    if (not audit['passed'] or core.digest(__file__) != protocol['builder_sha256']
            or core.digest(DATA / 'data/manifests/dataset.csv') != audit['native_manifest_sha256']):
        raise ValueError('Native preparation source/manifest changed.')
    source_selection()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = BACKBONE, 'recording', 'joint'
    rows = core.read_csv(DATA / 'data/manifests/dataset.csv')
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Native auxiliary features require TRAIN only.')
    identity = {'native_manifest_sha256': audit['native_manifest_sha256'],
        'backbone_sha256': core.digest(BACKBONE), 'context': 'recording', 'representation': 'joint',
        'code_sha256': {name: core.digest(ROOT / name) for name in
            ['native_training_data.py', 'improve.py', 'frozen_phoneme.py', 'schwa.py', 'train_negative_pool.py']},
        'cpu_threads': 2, 'mkldnn': False, 'v5_test_audio_or_features_consumed': False}
    path = DATA / 'registered_features.json'
    if path.exists() and core.read_json(path) != identity:
        raise ValueError('Native feature registration changed.')
    if not path.exists():
        core.write_json(path, identity)
    improve.extract_recordings(SimpleNamespace(splits=['train'], train_limit=None, seed=42), DATA, rows)
    values = improve.feature_cache(DATA, rows, 'train')
    cache, _ = improve.cache_paths(DATA, rows, 'train')
    core.write_json(ROOT / 'reports/native_training_feature_audit.json', {
        'passed': True, 'regions': len(rows), 'shape': list(values.shape), 'feature_sha256': core.digest(cache),
        'feature_path': cache.relative_to(ROOT).as_posix(), 'v5_test_audio_or_features_consumed': False,
        'quality_evidence': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'extract'])
    args = parser.parse_args()
    prepare() if args.command == 'prepare' else extract()

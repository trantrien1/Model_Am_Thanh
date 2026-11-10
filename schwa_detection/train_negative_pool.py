"""Recover unused, eligible negative windows from original TRAIN recordings only."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import schwa as core

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'v5'
OUTPUT = ROOT / 'training_all_negatives'
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'


def eligible_window(phone, phones):
    """Exactly the original negative duration/overlap rules; endpoints may touch."""
    if phone['phone'] in {'ax', 'unk'} or phone['end'] - phone['start'] < .02:
        return None
    start = round((phone['start'] + phone['end']) / 2 * 16000) - 4000
    end = start + 8000
    if any(start / 16000 < p['end'] and end / 16000 > p['start']
           for p in phones if p['phone'] in {'ax', 'unk'}):
        return None
    return start, end


def train_records(records):
    subset = [r for r in records if r['split'] == 'train']
    if not subset or any(r['split'] != 'train' for r in subset):
        raise ValueError('Original TRAIN recordings are required.')
    return subset


def prepare():
    manifest = DATA / 'data/manifests/dataset.csv'
    if core.digest(manifest) != DATASET_SHA:
        raise ValueError('Original V5 manifest changed.')
    all_rows = core.read_csv(manifest)
    original = [r for r in all_rows if r['split'] == 'train']
    original_ids = {r['sample_id'] for r in original}
    if len(original) != 9702 or len(original_ids) != 9702:
        raise ValueError('Every original TRAIN row must remain present.')
    records_path = DATA / 'data/manifests/recordings.csv'
    records = core.read_csv(records_path)
    training = train_records(records)
    heldout_speakers = {r['speaker_id'] for r in records if r['split'] != 'train'}
    heldout_prompts = {r['prompt_id'] for r in records if r['split'] != 'train'}
    if any(r['speaker_id'] in heldout_speakers or r['prompt_id'] in heldout_prompts for r in training):
        raise ValueError('TRAIN source identity overlaps held-out sources.')
    metadata = core.read_json(DATA / 'data/manifests/dataset_metadata.json')
    expected_negatives = metadata['candidate_counts']['train']['0']
    protocol = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Use every eligible original TRAIN negative instead of downsampling by center phone',
        'original_dataset_sha256': DATASET_SHA, 'original_recordings_sha256': core.digest(records_path),
        'builder_sha256': core.digest(__file__), 'original_label_code_sha256': core.digest(ROOT / 'schwa.py'),
        'train_recording_ids': sorted(r['recording_id'] for r in training),
        'expected_eligible_negatives': expected_negatives, 'original_train_rows_preserved': 9702,
        'window_samples': 8000, 'sample_rate': 16000, 'label': 'Strict AH0/AX; original negative QC unchanged',
        'selection_uses_model_scores': False, 'validation_or_test_audio_consumed': False,
        'output_is_recording_window_manifest': True, 'generated_wav_files_required': False}
    protocol_path = OUTPUT / 'registered_preparation.json'
    if protocol_path.exists():
        saved = core.read_json(protocol_path)
        if {k: v for k, v in saved.items() if k != 'created_utc'} != {
                k: v for k, v in protocol.items() if k != 'created_utc'}:
            raise ValueError('Preserve registered preparation protocol.')
    else:
        core.write_json(protocol_path, protocol)
    recovered, extra = {}, []
    for number, record in enumerate(training, 1):
        for kind in ('audio', 'lab'):
            if core.digest(DATA / record[kind + '_path']) != record[kind + '_sha256']:
                raise ValueError('Original TRAIN source checksum changed.')
        audio, _ = core.load_audio(DATA / record['audio_path'])
        phones, _ = core.parse_phones((DATA / record['lab_path']).read_text(), len(audio) / 16000)
        pcm_sha = core.pcm_hash(audio)
        for index, phone in enumerate(phones):
            bounds = eligible_window(phone, phones)
            if bounds is None:
                continue
            start, end = bounds
            window = core.crop_window(audio, start, 8000)
            if float(np.sqrt(np.mean(window.astype(np.float64) ** 2))) < 1e-5:
                continue
            identity = f"{record['recording_id']}_{index:03d}_0"
            row = {'sample_id': identity, 'recording_id': record['recording_id'],
                'speaker_id': record['speaker_id'], 'prompt_id': record['prompt_id'],
                'split': 'train', 'label': '0', 'center_phone': phone['phone'],
                'phone_start': str(phone['start']), 'phone_end': str(phone['end']),
                'window_start_sample': str(start), 'window_end_sample': str(end),
                'source_path': record['audio_path'], 'source_sha256': record['audio_sha256'],
                'source_pcm_sha256': pcm_sha, 'seed': str(metadata['settings']['seed']),
                'path': '', 'audio_sha256': '', 'pcm_sha256': ''}
            recovered[identity] = row
            if identity not in original_ids:
                extra.append(row)
        if number % 100 == 0 or number == len(training):
            print(f'TRAIN negative QC {number}/{len(training)}: {len(extra)} additional regions', flush=True)
    if len(recovered) != expected_negatives:
        raise ValueError(f'Negative pool differs from original generation: {len(recovered)} != {expected_negatives}')
    for row in original:
        if row['label'] == '0':
            reconstructed = recovered[row['sample_id']]
            if any(reconstructed[k] != row[k] for k in reconstructed if k not in {'path', 'audio_sha256', 'pcm_sha256'}):
                raise ValueError('Reconstructed original negative differs.')
    combined = sorted(original + extra, key=lambda r: r['sample_id'])
    if len({r['sample_id'] for r in combined}) != len(combined):
        raise ValueError('Repeated training region identity.')
    out = OUTPUT / 'data/manifests/dataset.csv'
    if out.exists():
        if core.read_csv(out) != combined:
            raise ValueError('Preserve existing expanded training manifest.')
    else:
        core.write_csv(out, combined)
    if core.digest(manifest) != DATASET_SHA:
        raise ValueError('Original frozen manifest changed during preparation.')
    audit = {'passed': True, 'original_train_regions_preserved': len(original),
        'additional_negative_regions': len(extra), 'expanded_train_regions': len(combined),
        'expanded_label_counts': dict(Counter(r['label'] for r in combined)),
        'additional_negative_phones': dict(Counter(r['center_phone'] for r in extra)),
        'train_recordings': len(training), 'source_audio_and_lab_checksums_verified': True,
        'heldout_speaker_and_prompt_overlap': False, 'original_manifest_sha256': DATASET_SHA,
        'expanded_manifest_sha256': core.digest(out), 'test_audio_or_features_consumed': False,
        'validation_regions_unchanged': 4884, 'test_regions_unchanged': 5157,
        'quality_evidence': False}
    core.write_json(ROOT / 'reports/all_train_negatives_preparation_audit.json', audit)
    print(audit, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare'])
    parser.parse_args()
    prepare()

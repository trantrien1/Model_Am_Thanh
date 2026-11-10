"""Align frozen recording caches to all eligible TRAIN regions; no encoder retraining."""
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import improve
import lexical_train_cpu as lexical
import schwa as core
from frozen_phoneme import RECORDING, aligned_window_features
from train_negative_pool import DATA, DATASET_SHA, OUTPUT, ROOT

BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'


def extract():
    torch.set_num_threads(2)
    torch.backends.mkldnn.enabled = False
    torch.use_deterministic_algorithms(True)
    original_path = DATA / 'data/manifests/dataset.csv'
    if core.digest(original_path) != DATASET_SHA:
        raise ValueError('Original V5 dataset changed.')
    audit = core.read_json(ROOT / 'reports/all_train_negatives_preparation_audit.json')
    manifest = OUTPUT / 'data/manifests/dataset.csv'
    if not audit['passed'] or core.digest(manifest) != audit['expanded_manifest_sha256']:
        raise ValueError('Expanded TRAIN manifest is not audited.')
    preparation = core.read_json(OUTPUT / 'registered_preparation.json')
    if core.digest(ROOT / 'train_negative_pool.py') != preparation['builder_sha256']:
        raise ValueError('Preparation implementation changed.')
    rows = core.read_csv(manifest)
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Expanded features may only consume TRAIN regions.')
    original = [r for r in core.read_csv(original_path) if r['split'] == 'train']
    original_index = {r['sample_id']: i for i, r in enumerate(original)}
    records = {r['recording_id']: r for r in core.read_csv(DATA / 'data/manifests/recordings.csv')}
    recipe = lexical.metadata(BACKBONE)
    identity = {'expanded_manifest_sha256': core.digest(manifest),
        'original_dataset_sha256': DATASET_SHA, 'recipe': recipe,
        'source_code_sha256': {name: core.digest(ROOT / name) for name in
            ['train_negative_features.py', 'train_negative_pool.py', 'frozen_phoneme.py',
             'lexical_train_cpu.py', 'lexical_context.py', 'improve.py', 'schwa.py']},
        'row_ids_sha256': core.object_digest([r['sample_id'] for r in rows]),
        'storage': 'float16', 'dimensions': 1420, 'frames': 24,
        'test_audio_or_features_consumed': False}
    registered_path = OUTPUT / 'registered_features.json'
    if registered_path.exists():
        if core.read_json(registered_path) != identity:
            raise ValueError('Preserve registered feature recipe.')
    else:
        core.write_json(registered_path, identity)
    path = OUTPUT / 'data/train_features.npy'
    marker = path.with_suffix('.json')
    if path.exists() and marker.exists():
        value = core.read_json(marker)
        if value['identity'] != identity or core.digest(path) != value['sha256']:
            raise ValueError('Existing expanded cache changed.')
        print('Verified complete expanded TRAIN cache.', flush=True)
        return
    improve.activate_dataset('v5')
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = BACKBONE, 'recording', 'joint'
    old_acoustic = improve.feature_cache(DATA, original, 'train')
    old_lexical = lexical.lexical_window_cache(DATA, original, 'train', recipe)
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        record = records[row['recording_id']]
        if record['split'] != 'train' or row['source_sha256'] != record['audio_sha256']:
            raise ValueError('Expanded source identity differs from original TRAIN.')
        groups[row['recording_id']].append((index, row))
    temporary = path.with_suffix('.partial.npy')
    output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
                                      shape=(len(rows), 24, 1420))
    source_caches, verified_original = [], 0
    for number, (recording_id, group) in enumerate(sorted(groups.items()), 1):
        record = records[recording_id]
        if core.digest(DATA / record['audio_path']) != record['audio_sha256']:
            raise ValueError('Original TRAIN audio changed.')
        teacher_recipe = {'encoder': recipe['main_encoder_sha256'],
            'source_audio': record['audio_sha256'], 'recipe': RECORDING, 'representation': 'joint'}
        teacher = DATA / 'data/recording_transfer_features' / (core.object_digest(teacher_recipe) + '.npy')
        teacher_marker = core.read_json(teacher.with_suffix('.json'))
        if (core.digest(teacher) != teacher_marker['sha256']
                or teacher_marker['source_audio_sha256'] != record['audio_sha256']
                or teacher_marker['backbone_sha256'] != recipe['main_encoder_sha256']
                or teacher_marker['recipe'] != RECORDING or teacher_marker['representation'] != 'joint'):
            raise ValueError('Original public phoneme cache differs.')
        local_recipe = {'recipe': recipe, 'source_audio_sha256': record['audio_sha256']}
        local = DATA / 'data/lexical_recording_features' / (core.object_digest(local_recipe) + '.npy')
        acoustic = torch.from_numpy(np.load(teacher, allow_pickle=False)).float()
        words = torch.from_numpy(np.array(lexical.array_cache(local, local_recipe))).float()
        if acoustic.ndim != 2 or acoustic.shape[-1] != 1416 or words.shape != (len(acoustic), 4):
            raise ValueError('Unexpected recording cache geometry.')
        starts = [int(row['window_start_sample']) for _, row in group]
        aligned_acoustic = aligned_window_features(acoustic, starts).numpy()
        aligned_words = aligned_window_features(words, starts).numpy()
        combined = np.concatenate([aligned_acoustic, aligned_words], axis=-1).astype(np.float16)
        if not np.isfinite(combined).all():
            raise ValueError('Expanded TRAIN features are not finite.')
        for j, (_, row) in enumerate(group):
            if row['sample_id'] in original_index:
                old = original_index[row['sample_id']]
                if (not np.array_equal(combined[j, :, :1416], old_acoustic[old])
                        or not np.array_equal(combined[j, :, 1416:], old_lexical[old])):
                    raise ValueError('Original TRAIN feature parity failed.')
                verified_original += 1
        output[[i for i, _ in group]] = combined
        source_caches.append({'recording_id': recording_id, 'acoustic_sha256': teacher_marker['sha256'],
                              'lexical_sha256': core.read_json(local.with_suffix('.json'))['sha256']})
        if number % 32 == 0 or number == len(groups):
            print(f'Expanded TRAIN features {number}/{len(groups)} recordings', flush=True)
    if verified_original != 9702:
        raise ValueError('Original TRAIN feature parity did not cover every region.')
    output.flush()
    del output
    temporary.replace(path)
    sha = core.digest(path)
    core.write_json(marker, {'identity': identity, 'sha256': sha, 'shape': [len(rows), 24, 1420],
        'source_caches': source_caches, 'original_train_feature_parity_regions': verified_original})
    core.write_json(ROOT / 'reports/all_train_negatives_feature_audit.json', {
        'passed': True, 'shape': [len(rows), 24, 1420], 'feature_sha256': sha,
        'original_train_feature_parity_regions': verified_original, 'source_recordings': len(groups),
        'new_encoder_forward_passes': 0, 'test_audio_or_features_consumed': False,
        'quality_evidence': False})
    print('Expanded TRAIN features complete; all 9702 original regions match exactly.', flush=True)


if __name__ == '__main__':
    extract()

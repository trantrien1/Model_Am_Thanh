"""Extract full audio-inferred pronunciation priors with exact legacy-feature checks."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import lexical_train_cpu as lexical
import schwa as core
from frozen_phoneme import RECORDING, aligned_window_features
from lexical_context import DICTIONARY_SOURCE, load_dictionary
from phonetic_prior import PRIOR_RECIPE, bounded_prior_features

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'v5'
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
SOURCES = ['phonetic_prior_features.py', 'phonetic_prior.py', 'lexical_context.py',
           'lexical_train_cpu.py', 'frozen_phoneme.py', 'schwa.py']


def metadata():
    if core.digest(DATA / 'data/manifests/dataset.csv') != DATASET_SHA:
        raise ValueError('Frozen V5 manifest changed.')
    return {'dataset_sha256': DATASET_SHA, 'legacy': lexical.metadata(BACKBONE),
        'prior_recipe': PRIOR_RECIPE, 'cpu_threads': 1, 'mkldnn': False,
        'source_code_sha256': {name: core.digest(ROOT / name) for name in SOURCES},
        'legacy_channels': 4, 'prior_channels': 392, 'storage': 'float16',
        'input': 'Automatic audio CTC hypotheses, public pronunciation dictionary and phoneme emissions only',
        'reference_text_consumed': False, 'manual_phone_features': False,
        'test_audio_or_features_consumed': False}


def register():
    value = metadata()
    path = ROOT / 'reports/precision_recall_88_phonetic_prior_feature_protocol.json'
    if path.exists():
        if core.read_json(path)['identity'] != value:
            raise ValueError('Preserve registered full-prior feature implementation.')
    else:
        core.write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'identity': value,
            'source_research_note': 'reports/vowel_reduction_research_notes.md',
            'original_train_regions': 9702, 'validation_regions': 4884, 'test_regions': 5157,
            'quality_evidence': False, 'new_head_not_trained': True})
    return value


def window_path(rows, split, value):
    if split not in {'train', 'validation'} or any(r['split'] != split for r in rows):
        raise ValueError('Full-prior development features cannot consume test regions.')
    identity = {'recipe': value, 'row_ids': [r['sample_id'] for r in rows],
                'starts': [r['window_start_sample'] for r in rows]}
    path = DATA / 'data/phonetic_prior_windows' / (split + '_' + core.object_digest(identity)[:20] + '.npy')
    return path, identity


def build_recording(record, value, dictionary, labels, vocabulary):
    if record['split'] not in {'train', 'validation'}:
        raise ValueError('Full-prior development code cannot consume test recordings.')
    recipe = value['legacy']
    source_sha = record['audio_sha256']
    if core.digest(DATA / record['audio_path']) != source_sha:
        raise ValueError('Audio source changed.')
    old_identity = {'recipe': recipe, 'source_audio_sha256': source_sha}
    old_path = DATA / 'data/lexical_recording_features' / (core.object_digest(old_identity) + '.npy')
    legacy = lexical.array_cache(old_path, old_identity)
    identity = {'recipe': value, 'source_audio_sha256': source_sha}
    path = DATA / 'data/phonetic_prior_recordings' / (core.object_digest(identity) + '.npy')
    if path.exists() and path.with_suffix('.json').exists():
        result = lexical.array_cache(path, identity)
    else:
        teacher_identity = {'encoder': recipe['main_encoder_sha256'], 'source_audio': source_sha,
                            'recipe': RECORDING, 'representation': 'joint'}
        teacher = DATA / 'data/recording_transfer_features' / (core.object_digest(teacher_identity) + '.npy')
        marker = core.read_json(teacher.with_suffix('.json'))
        if (core.digest(teacher) != marker['sha256'] or marker['source_audio_sha256'] != source_sha
                or marker['backbone_sha256'] != recipe['main_encoder_sha256']
                or marker['recipe'] != RECORDING or marker['representation'] != 'joint'):
            raise ValueError('Original public phoneme source cache differs.')
        teacher_values = np.load(teacher, allow_pickle=False)
        if teacher_values.ndim != 2 or teacher_values.shape[-1] != 1416:
            raise ValueError('Unexpected public phoneme cache geometry.')
        asr_identity = {'source_audio_sha256': source_sha,
                        'encoder_sha256': recipe['asr_encoder_sha256'], 'recipe': RECORDING}
        asr_path = DATA / 'data/word_asr_features' / (core.object_digest(asr_identity) + '.npy')
        asr = torch.from_numpy(np.array(lexical.array_cache(asr_path, asr_identity))).float()
        phones = torch.from_numpy(teacher_values[:, 1024:]).float()
        with torch.inference_mode():
            full, words = bounded_prior_features(asr, phones, dictionary, labels, vocabulary)
        result = full.half().numpy()
        if result.shape != (len(legacy), 396) or not np.isfinite(result).all():
            raise ValueError('Invalid full phoneme prior geometry or values.')
        if not np.array_equal(result[:, :4], legacy):
            raise ValueError('Original four recording lexical features do not match exactly.')
        lexical.save_array(path, result, identity)
        core.write_json(path.with_suffix('.words.json'), {'inferred_words': words,
            'reference_text_consumed': False, 'source_audio_sha256': source_sha})
    if result.shape != (len(legacy), 396) or not np.array_equal(result[:, :4], legacy):
        raise ValueError('Cached prior fails original recording-feature parity.')
    return result


def resources():
    dictionary = load_dictionary(DICTIONARY_SOURCE / 'cmudict.dict')
    state = torch.load(lexical.ASR_PATH, weights_only=True, mmap=True)
    labels = state['labels']
    del state
    vocabulary = core.read_json(ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json')
    return dictionary, labels, vocabulary


def prepare(probe=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    value = register()
    dictionary, labels, vocabulary = resources()
    rows = core.read_csv(DATA / 'data/manifests/dataset.csv')
    records = {r['recording_id']: r for r in core.read_csv(DATA / 'data/manifests/recordings.csv')}
    reports = []
    for split in (['train'] if probe else ['train', 'validation']):
        original = [r for r in rows if r['split'] == split]
        old_windows = lexical.lexical_window_cache(DATA, original, split, value['legacy'])
        selected_ids = set(sorted({r['recording_id'] for r in original})[:2]) if probe else None
        subset = [r for r in original if selected_ids is None or r['recording_id'] in selected_ids]
        old_indices = {r['sample_id']: i for i, r in enumerate(original)}
        path, identity = window_path(subset, split, value)
        if path.exists() and path.with_suffix('.json').exists():
            values = lexical.array_cache(path, identity)
            if values.shape != (len(subset), 24, 392):
                raise ValueError('Full-prior window geometry changed.')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix('.partial.npy')
            values = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16,
                                               shape=(len(subset), 24, 392))
            groups = defaultdict(list)
            for index, row in enumerate(subset):
                groups[row['recording_id']].append((index, row))
            for number, (recording_id, group) in enumerate(sorted(groups.items()), 1):
                record = records[recording_id]
                if record['split'] != split:
                    raise ValueError('Region split differs from source recording.')
                full = build_recording(record, value, dictionary, labels, vocabulary)
                aligned = aligned_window_features(torch.from_numpy(np.array(full)).float(),
                    [int(r['window_start_sample']) for _, r in group]).numpy().astype(np.float16)
                expected = old_windows[[old_indices[r['sample_id']] for _, r in group]]
                if not np.array_equal(aligned[:, :, :4], expected):
                    raise ValueError('Original focal lexical windows differ.')
                values[[i for i, _ in group]] = aligned[:, :, 4:]
                if number % 32 == 0 or number == len(groups):
                    print(f'Full phonetic prior {split}: {number}/{len(groups)} recordings', flush=True)
            values.flush()
            del values
            temporary.replace(path)
            core.write_json(path.with_suffix('.json'), {'identity': identity, 'sha256': core.digest(path),
                'shape': [len(subset), 24, 392], 'exact_original_lexical_parity_regions': len(subset)})
            values = lexical.array_cache(path, identity)
        reports.append({'split': split, 'regions': len(subset), 'shape': list(values.shape),
            'feature_sha256': core.digest(path), 'exact_original_lexical_parity_regions': len(subset),
            'recordings': len({r['recording_id'] for r in subset}),
            'nonzero_prior_frames': int(np.any(values != 0, axis=-1).sum())})
    if metadata() != value:
        raise ValueError('Registered feature implementation changed during extraction.')
    report = {'passed': True, 'probe_only': probe, 'splits': reports,
              'test_audio_or_features_consumed': False, 'quality_evidence': False}
    name = 'phonetic_prior_train_probe.json' if probe else 'phonetic_prior_feature_audit.json'
    core.write_json(ROOT / 'reports' / name, report)
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['probe', 'extract'])
    args = parser.parse_args()
    prepare(probe=args.command == 'probe')

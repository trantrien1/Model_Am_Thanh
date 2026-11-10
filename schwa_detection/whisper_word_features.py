"""Audio-inferred word IDs for the immutable V5 focal regions; TRAIN-only vocabulary."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

import schwa as core
import whisper_lexical as lexical
from lexical_train_cpu import array_cache, save_array

ROOT = Path(__file__).resolve().parent
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
VOCABULARY = ROOT / 'reports/train_audio_word_vocabulary.json'
DESIGN = ROOT / 'reports/precision_recall_88_word_identity_design.json'
RECIPE = {'version': 1, 'word_margin_seconds': .08, 'unknown_id': 0,
          'overlap': 'largest word confidence then nearest midpoint; stable input order for ties',
          'focal_center_sample': 199.5, 'hop_samples': 320, 'frames': 24,
          'storage': 'int32', 'input': 'Audio-inferred Whisper word spans only',
          'reference_transcript_consumed': False, 'manual_phone_timing_used_as_features': False}


def normalize_word(word):
    return re.sub(r"^[^A-Z']+|[^A-Z']+$", '', str(word).upper())


def metadata():
    design = core.read_json(DESIGN)
    if core.digest(VOCABULARY) != design['word_vocabulary']['sha256']:
        raise ValueError('TRAIN-only vocabulary changed after candidate design.')
    return {'vocabulary_sha256': core.digest(VOCABULARY), 'recipe': RECIPE,
            'whisper_audio_words': lexical.metadata(),
            'source_code_sha256': {'whisper_word_features.py': core.digest(Path(__file__))}}


def load_audio_words(records):
    """Match the registered audio-only identity; reference diagnostics are never read."""
    allowed = {r['recording_id']: r for r in records}
    expected_common = None
    found = {}
    for path in (ROOT / 'v5/data/whisper_word_features').glob('*.json'):
        cached = core.read_json(path)
        identity = cached['identity']
        rid = identity['recording_id']
        if rid not in allowed:
            continue
        if expected_common is None:
            expected_common = lexical.word_identity(allowed[rid])
        expected = {**expected_common, 'recording_id': rid,
                    'source_audio_sha256': allowed[rid]['audio_sha256']}
        matches = all(identity.get(key) == value for key, value in expected.items()
                      if key != 'package_files_sha256')
        matches &= all(identity.get('package_files_sha256', {}).get(key) == value
                       for key, value in expected['package_files_sha256'].items())
        if not matches:
            continue
        if cached['reference_transcript_consumed'] or core.object_digest(cached['audio_inferred_words']) != cached['words_sha256']:
            raise ValueError('Audio-only hypothesis cache changed.')
        if rid in found and found[rid] != cached['audio_inferred_words']:
            raise ValueError('Conflicting audio-only hypotheses.')
        found[rid] = cached['audio_inferred_words']
    if set(found) != set(allowed):
        raise ValueError('All registered audio-only word hypotheses must exist.')
    return found


def build_word_vocabulary(inferred):
    frequencies = Counter(normalize_word(w['word']) for rid in sorted(inferred) for w in inferred[rid])
    return {token: index + 1 for index, token in enumerate(
        sorted(token for token, count in frequencies.items() if token and count >= 2))}


def focal_word_ids(words, starts, word_to_id):
    if any(not np.isfinite([w['start'], w['end'], w.get('probability', 0.)]).all()
           or not 0 <= w['start'] <= w['end'] for w in words):
        raise ValueError('Finite valid audio-inferred word boundaries required.')
    starts = np.asarray(starts, dtype=np.float64)
    if starts.ndim != 1 or not np.isfinite(starts).all():
        raise ValueError('Finite one-dimensional window starts required.')
    centers = (starts[:, None] + RECIPE['focal_center_sample']
               + np.arange(RECIPE['frames'])[None, :] * RECIPE['hop_samples']) / 16000
    output = np.zeros((*centers.shape, 1), dtype=np.int32)
    for row in range(len(starts)):
        for frame, center in enumerate(centers[row]):
            if center < 0:
                continue
            candidates = [word for word in words
                          if word['end'] > word['start']
                          and max(0., word['start'] - RECIPE['word_margin_seconds']) <= center
                          <= word['end'] + RECIPE['word_margin_seconds']]
            if candidates:
                best = max(candidates, key=lambda w: (w.get('probability', 0.),
                           -abs(center - (w['start'] + w['end']) / 2)))
                output[row, frame, 0] = word_to_id.get(normalize_word(best['word']), 0)
    return output


def window_path(directory, rows, split, recipe):
    if split not in ('train', 'validation') or any(r['split'] != split for r in rows):
        raise ValueError('Word feature development cannot consume test regions.')
    identity = {'recipe': recipe, 'row_ids': [r['sample_id'] for r in rows],
                'starts': [r['window_start_sample'] for r in rows]}
    path = directory / 'data/whisper_word_id_windows' / (split + '_' + core.object_digest(identity) + '.npy')
    return path, identity


def extract(splits, limit=0):
    if limit and splits != ['train']:
        raise ValueError('Limited feasibility extraction may use TRAIN only.')
    directory = ROOT / 'v5'
    if core.digest(directory / 'data/manifests/dataset.csv') != DATASET_SHA:
        raise ValueError('Frozen manifest changed.')
    all_rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    records = core.read_csv(directory / 'data/manifests/recordings.csv')
    train_ids = {r['recording_id'] for r in all_rows if r['split'] == 'train'}
    train_records = [r for r in records if r['recording_id'] in train_ids]
    inferred = load_audio_words(train_records)
    vocabulary = core.read_json(VOCABULARY)
    if (vocabulary['recording_ids'] != sorted(train_ids)
            or build_word_vocabulary(inferred) != vocabulary['word_to_id']):
        raise ValueError('Vocabulary must derive solely from original TRAIN audio hypotheses.')
    recipe = metadata()
    reports = []
    for split in splits:
        rows = [r for r in all_rows if r['split'] == split]
        if limit:
            allowed = set(sorted({r['recording_id'] for r in rows})[:limit])
            rows = [r for r in rows if r['recording_id'] in allowed]
        path, identity = window_path(directory, rows, split, recipe)
        if path.exists() and path.with_suffix('.json').exists():
            values = array_cache(path, identity)
        else:
            ids = {r['recording_id'] for r in rows}
            sources = [r for r in records if r['recording_id'] in ids]
            if any(r['split'] != split or core.digest(directory / r['audio_path']) != r['audio_sha256'] for r in sources):
                raise ValueError('Original source audio or split changed.')
            inferred_split = inferred if split == 'train' else load_audio_words(sources)
            groups = {}
            for index, row in enumerate(rows):
                groups.setdefault(row['recording_id'], []).append((index, row))
            values = np.zeros((len(rows), 24, 1), dtype=np.int32)
            for rid, group in sorted(groups.items()):
                values[[index for index, _ in group]] = focal_word_ids(inferred_split[rid],
                    [int(r['window_start_sample']) for _, r in group], vocabulary['word_to_id'])
            save_array(path, values, identity)
        if values.shape != (len(rows), 24, 1) or values.dtype != np.int32:
            raise ValueError('Word identity shape/storage differs.')
        reports.append({'split': split, 'regions': len(rows), 'shape': list(values.shape),
                        'sha256': core.digest(path), 'known_frame_fraction': float((values > 0).mean())})
    report = {'status': 'completed', 'splits': splits, 'limit': limit, 'caches': reports,
              'train_vocabulary_rebuilt_from_audio_only': True, 'reference_transcript_consumed': False,
              'test_predictions_computed': False, 'quality_evidence': False}
    core.write_json(ROOT / 'reports/whisper_word_feature_extraction.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    extract(args.splits, args.limit)

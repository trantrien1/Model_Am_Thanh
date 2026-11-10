"""Audio-only Whisper word spans aligned to frozen public phone emissions."""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

import schwa as core
import whisper_cpu
from frozen_phoneme import RECORDING, aligned_window_features
from lexical_context import (DICTIONARY_SOURCE, ctc_state_posteriors,
                             load_dictionary, pronunciation_ids, verify_dictionary_source)
from lexical_train_cpu import array_cache, save_array

ROOT = Path(__file__).resolve().parent
BACKBONE_SHA = '4c15648baf544efa819f0798cd76664a131960e45345d9f57091c5443558bb0f'
RECIPE = {'version': 1, 'features': 8, 'word_margin_seconds': .08,
          'pronunciation': 'maximum_public_phone_CTC_evidence_per_Whisper_word',
          'alternative_limit': 16, 'overlapping_spans': 'arithmetic_mean',
          'unresolved_words': 'no_phone_prior; retain_word_confidence_and_coverage',
          'storage': 'float16', 'reference_transcript_consumed': False,
          'manual_phone_timing_used_as_features': False}


def word_identity(record):
    return {'model_sha256': whisper_cpu.MODEL_SHA, 'package_version': whisper_cpu.VERSION,
            'package_files_sha256': whisper_cpu.package_identity(),
            'decode_recipe': whisper_cpu.DECODE, 'cpu_math': whisper_cpu.CPU_MATH,
            'helper_code_sha256': core.digest(ROOT / 'whisper_cpu.py'),
            'source_audio_sha256': record['audio_sha256'], 'recording_id': record['recording_id']}


def word_features(words, phone_logits, dictionary, vocabulary):
    """No labels, transcripts or manual boundaries are accepted by this function."""
    if phone_logits.ndim != 2 or not len(phone_logits) or not torch.isfinite(phone_logits).all():
        raise ValueError('Finite public phone emissions are required.')
    output = phone_logits.new_zeros((len(phone_logits), RECIPE['features']))
    count = phone_logits.new_zeros(len(phone_logits))
    centers = (RECORDING['center_sample'] + torch.arange(len(phone_logits), dtype=torch.float64)
               * RECORDING['hop_samples']) / 16000
    for word in words:
        start, end = float(word['start']), float(word['end'])
        if not 0 <= start <= end or not np.isfinite([start, end]).all():
            raise ValueError('Invalid audio-inferred word span.')
        if end <= start:
            continue
        indices = torch.where((centers >= max(0, start - RECIPE['word_margin_seconds'])) &
                              (centers <= end + RECIPE['word_margin_seconds']))[0]
        if not len(indices):
            continue
        token = re.sub(r"^[^A-Z']+|[^A-Z']+$", '', str(word['word']).upper())
        values = output.new_zeros((len(indices), RECIPE['features']))
        best = None
        probabilities = phone_logits[indices].float().log_softmax(-1)
        alternatives = dictionary.get(token, [])[:RECIPE['alternative_limit']]
        for pronunciation in alternatives:
            try:
                targets = pronunciation_ids(pronunciation, vocabulary)
                posterior, symbols, evidence = ctc_state_posteriors(probabilities, targets)
            except ValueError:
                continue
            if best is None or float(evidence) > best[0]:
                best = (float(evidence), posterior, symbols, targets)
        if best is not None:
            evidence, posterior, symbols, targets = best
            values[:, 0] = posterior[:, symbols == vocabulary['\u0259']].sum(-1)
            values[:, 1] = posterior[:, symbols != 0].sum(-1)
            values[:, 2] = np.exp(evidence / len(indices))
            values[:, 3] = 1.
            values[:, 5] = float(vocabulary['\u0259'] in targets)
        values[:, 4] = float(np.clip(word.get('probability', 0.), 0., 1.))
        values[:, 6] = ((centers[indices] - start) / (end - start)).float().clamp(0., 1.)
        values[:, 7] = 1.
        output[indices] += values
        count[indices] += 1
    return output / count.clamp_min(1).unsqueeze(-1)


def metadata():
    verify_dictionary_source()
    protocol = core.read_json(ROOT / 'reports/whisper_validation_asr_protocol.json')
    package = whisper_cpu.package_identity()
    for name, sha in package.items():
        if protocol['package_files_sha256'][name] != sha:
            raise ValueError('Whisper package changed after paired benchmark.')
    if protocol['helper_code_sha256'] != core.digest(ROOT / 'whisper_cpu.py'):
        raise ValueError('Whisper decoding helper changed.')
    return {'backbone_sha256': BACKBONE_SHA, 'model_sha256': whisper_cpu.MODEL_SHA,
            'package_version': whisper_cpu.VERSION, 'package_files_sha256': package,
            'decode_recipe': whisper_cpu.DECODE, 'cpu_math': whisper_cpu.CPU_MATH,
            'dictionary_sha256': core.digest(DICTIONARY_SOURCE / 'cmudict.dict'),
            'vocabulary_sha256': core.digest(ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json'),
            'recording_recipe': RECORDING, 'lexical_recipe': RECIPE,
            'source_code_sha256': {name: core.digest(ROOT / name) for name in
                ['whisper_lexical.py', 'whisper_cpu.py', 'lexical_context.py',
                 'lexical_train_cpu.py', 'frozen_phoneme.py', 'schwa.py']}}


def cache_path(directory, rows, split, recipe):
    if split not in ('train', 'validation') or any(r['split'] != split for r in rows):
        raise ValueError('Test inputs are not allowed in development extraction.')
    identity = {'recipe': recipe, 'row_ids': [r['sample_id'] for r in rows],
                'starts': [r['window_start_sample'] for r in rows]}
    return directory / 'data/whisper_lexical_windows' / (split + '_' + core.object_digest(identity) + '.npy'), identity


def extract(args):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    directory = ROOT / 'v5'
    recipe = metadata()
    rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
    dictionary = load_dictionary(DICTIONARY_SOURCE / 'cmudict.dict')
    vocabulary = core.read_json(ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json')
    word_cache_index = {}
    for candidate in (directory / 'data/whisper_word_features').glob('*.json'):
        cached = core.read_json(candidate)
        word_cache_index.setdefault(cached['identity']['recording_id'], []).append(candidate)
    model = None
    started = time.perf_counter()
    for split in args.splits:
        subset = [r for r in rows if r['split'] == split]
        if args.limit:
            allowed = set(sorted({r['recording_id'] for r in subset})[:args.limit])
            subset = [r for r in subset if r['recording_id'] in allowed]
        output_path, output_identity = cache_path(directory, subset, split, recipe)
        if output_path.exists() and output_path.with_suffix('.json').exists():
            array_cache(output_path, output_identity)
            print('Verified Whisper lexical windows:', split, flush=True)
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_suffix('.partial.npy')
        output = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16,
                                           shape=(len(subset), 24, RECIPE['features']))
        groups = {}
        for index, row in enumerate(subset):
            groups.setdefault(row['recording_id'], []).append((index, row))
        for number, (rid, group) in enumerate(sorted(groups.items()), 1):
            record = records[rid]
            if record['split'] != split:
                raise ValueError('Source split differs.')
            source = directory / record['audio_path']
            if core.digest(source) != record['audio_sha256']:
                raise ValueError('Source audio changed.')
            # Audio-only benchmark word caches have additional normalizer hashes.
            # Match their decoder/model identity, never their reference diagnostics.
            identity = word_identity(record)
            candidates = word_cache_index.get(rid, [])
            inferred = None
            direct = directory / 'data/whisper_word_features' / (core.object_digest(identity) + '.json')
            if direct.exists():
                candidates = [direct]
            for candidate in candidates:
                cached = core.read_json(candidate)
                found = cached['identity']
                if found.get('recording_id') != rid:
                    continue
                matches = all(found.get(key) == value for key, value in identity.items()
                              if key != 'package_files_sha256')
                matches &= all(found['package_files_sha256'].get(key) == value
                               for key, value in identity['package_files_sha256'].items())
                if matches:
                    if (cached['reference_transcript_consumed'] or
                            cached['words_sha256'] != core.object_digest(cached['audio_inferred_words'])):
                        raise ValueError('Audio-only word cache changed.')
                    inferred = cached
                    break
            if inferred is None:
                if model is None:
                    model = whisper_cpu.load_model()
                audio, _ = core.load_audio(source)
                with torch.backends.mkldnn.flags(enabled=False):
                    decoded = model.transcribe(audio, **whisper_cpu.DECODE)
                words = [w for segment in decoded['segments'] for w in segment.get('words', [])]
                if any(not 0 <= w['start'] <= w['end'] <= len(audio)/16000 + .1 for w in words):
                    raise ValueError('Invalid inferred word timings.')
                inferred = {'identity': identity, 'audio_inferred_text': decoded['text'],
                    'audio_inferred_words': words, 'words_sha256': core.object_digest(words),
                    'reference_transcript_consumed': False}
                core.write_json(direct, inferred)
            phone_id = {'encoder': BACKBONE_SHA, 'source_audio': record['audio_sha256'],
                        'recipe': RECORDING, 'representation': 'joint'}
            phone_path = directory / 'data/recording_transfer_features' / (core.object_digest(phone_id) + '.npy')
            marker = core.read_json(phone_path.with_suffix('.json'))
            if (core.digest(phone_path) != marker['sha256'] or
                    marker['source_audio_sha256'] != record['audio_sha256'] or
                    marker['backbone_sha256'] != BACKBONE_SHA or marker['recipe'] != RECORDING or
                    marker['representation'] != 'joint'):
                raise ValueError('Frozen public phone cache changed.')
            phone_values = np.load(phone_path, allow_pickle=False)
            if phone_values.shape[1] != 1416 or not np.isfinite(phone_values).all():
                raise ValueError('Unexpected public phone feature geometry.')
            phone_logits = torch.from_numpy(phone_values[:, 1024:].astype(np.float32))
            with torch.inference_mode():
                lexical = word_features(inferred['audio_inferred_words'], phone_logits, dictionary, vocabulary).half().float()
            indices = [i for i, _ in group]
            output[indices] = aligned_window_features(lexical,
                [int(r['window_start_sample']) for _, r in group]).numpy()
            status = {'status': 'running', 'split': split, 'recordings_completed': number,
                      'recordings_total': len(groups), 'seconds': time.perf_counter()-started,
                      'reference_transcript_consumed': False, 'test_predictions_computed': False}
            core.write_json(ROOT / 'reports/whisper_lexical_extraction_status.json', status)
            if number % 16 == 0 or number == len(groups):
                print(json.dumps(status), flush=True)
        output.flush()
        del output
        partial.replace(output_path)
        core.write_json(output_path.with_suffix('.json'), {'identity': output_identity,
            'sha256': core.digest(output_path), 'shape': [len(subset), 24, RECIPE['features']]})
    core.write_json(ROOT / 'reports/whisper_lexical_extraction_status.json', {
        'status': 'completed', 'splits': args.splits, 'limit': args.limit,
        'seconds': time.perf_counter()-started, 'test_predictions_computed': False,
        'reference_transcript_consumed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--limit', type=int, default=0, help='TRAIN-only feasibility checks use a small positive limit.')
    args = parser.parse_args()
    if args.limit and args.splits != ['train']:
        parser.error('Partial feasibility extraction may use TRAIN only.')
    extract(args)

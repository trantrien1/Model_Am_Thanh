"""Frozen public Whisper BPE text embeddings for words inferred solely from audio."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import schwa as core
import whisper_cpu
import whisper_lexical
from lexical_train_cpu import array_cache, save_array
from whisper_word_features import load_audio_words, focal_word_ids, normalize_word

ROOT = Path(__file__).resolve().parent
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
RECIPE = {'version': 1, 'dimension': 512,
          'weights': 'Original public base.en decoder.token_embedding.weight; frozen',
          'tokenizer': 'official Whisper English GPT2 BPE; default 99 language special tokens',
          'word_text': 'normalized audio-inferred word lowercased with one leading space',
          'word_embedding': 'float32 mean of public token rows; float16 storage',
          'word_margin_seconds': .08, 'overlap': 'confidence then nearest word midpoint',
          'focal_center_sample': 199.5, 'hop_samples': 320, 'frames': 24,
          'outside_word': 'zero vector', 'unseen_words': 'public byte-pair tokenization',
          'reference_transcript_consumed': False, 'manual_phone_timing_used_as_features': False}


def metadata():
    if core.digest(whisper_cpu.MODEL) != whisper_cpu.MODEL_SHA:
        raise ValueError('Original public Whisper weight changed.')
    return {'public_model_sha256': whisper_cpu.MODEL_SHA, 'recipe': RECIPE,
            'audio_word_recipe': whisper_lexical.metadata(),
            'source_code_sha256': {name: core.digest(ROOT / name) for name in
                 ['whisper_text_features.py', 'whisper_word_features.py', 'whisper_cpu.py',
                  'frozen_phoneme.py', 'schwa.py']}}


def word_vector(word, tokenizer, embedding):
    text = normalize_word(word).lower()
    if not text:
        return embedding.new_zeros(embedding.shape[-1], dtype=torch.float16)
    tokens = tokenizer.encode(' ' + text)
    if not tokens or min(tokens) < 0 or max(tokens) >= len(embedding):
        raise ValueError('Public tokenizer/embedding vocabulary differs.')
    return embedding[tokens].float().mean(0).half()


def focal_vectors(words, starts, tokenizer, embedding):
    # This lookup only memoizes the fixed public function for current audio words;
    # no vocabulary is learned from validation and no labels are accepted.
    tokens = sorted({normalize_word(w['word']) for w in words} - {''})
    lookup = {token: index + 1 for index, token in enumerate(tokens)}
    matrix = torch.zeros((len(tokens) + 1, embedding.shape[-1]), dtype=torch.float16)
    for token, index in lookup.items():
        matrix[index] = word_vector(token, tokenizer, embedding)
    ids = focal_word_ids(words, starts, lookup)
    return matrix[torch.from_numpy(ids[..., 0]).long()].numpy()


def window_path(directory, rows, split, recipe):
    if split not in ('train', 'validation') or any(r['split'] != split for r in rows):
        raise ValueError('Development text extraction cannot consume test regions.')
    identity = {'recipe': recipe, 'row_ids': [r['sample_id'] for r in rows],
                'starts': [r['window_start_sample'] for r in rows]}
    path = directory / 'data/whisper_text_windows' / (split + '_' + core.object_digest(identity) + '.npy')
    return path, identity


def extract(splits, limit=0):
    if limit and splits != ['train']:
        raise ValueError('Limited feasibility extraction may use TRAIN only.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    directory = ROOT / 'v5'
    if core.digest(directory / 'data/manifests/dataset.csv') != DATASET_SHA:
        raise ValueError('Frozen V5 manifest changed.')
    recipe = metadata()
    rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    records = core.read_csv(directory / 'data/manifests/recordings.csv')
    state = tokenizer = embedding = None
    reports = []
    for split in splits:
        subset = [r for r in rows if r['split'] == split]
        if limit:
            allowed = set(sorted({r['recording_id'] for r in subset})[:limit])
            subset = [r for r in subset if r['recording_id'] in allowed]
        path, identity = window_path(directory, subset, split, recipe)
        if path.exists() and path.with_suffix('.json').exists():
            values = array_cache(path, identity)
        else:
            ids = {r['recording_id'] for r in subset}
            sources = [r for r in records if r['recording_id'] in ids]
            if any(r['split'] != split or core.digest(directory / r['audio_path']) != r['audio_sha256'] for r in sources):
                raise ValueError('Source audio or split changed.')
            inferred = load_audio_words(sources)
            if embedding is None:
                from whisper.tokenizer import get_tokenizer
                state = torch.load(whisper_cpu.MODEL, weights_only=True, mmap=True)
                embedding = state['model_state_dict']['decoder.token_embedding.weight']
                tokenizer = get_tokenizer(multilingual=False)
                if embedding.shape != (51864, 512) or tokenizer.encoding.n_vocab != 51864:
                    raise ValueError('Public English BPE shape differs.')
            groups = {}
            for index, row in enumerate(subset):
                groups.setdefault(row['recording_id'], []).append((index, row))
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix('.partial.npy')
            values = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16,
                                               shape=(len(subset), 24, RECIPE['dimension']))
            for rid, group in sorted(groups.items()):
                values[[index for index, _ in group]] = focal_vectors(inferred[rid],
                    [int(r['window_start_sample']) for _, r in group], tokenizer, embedding)
            values.flush()
            del values
            partial.replace(path)
            core.write_json(path.with_suffix('.json'), {'identity': identity,
                'sha256': core.digest(path), 'shape': [len(subset), 24, RECIPE['dimension']]})
            values = np.load(path, mmap_mode='r', allow_pickle=False)
        if values.shape != (len(subset), 24, 512) or values.dtype != np.float16:
            raise ValueError('Text feature geometry differs.')
        covered, total = 0, 0
        for start in range(0, len(values), 64):
            block = values[start:start + 64]
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite public text embeddings.')
            covered += int(np.any(block != 0, axis=-1).sum())
            total += block.shape[0] * block.shape[1]
        reports.append({'split': split, 'regions': len(subset), 'shape': list(values.shape),
                        'sha256': core.digest(path), 'covered_frame_fraction': covered / total})
    core.write_json(ROOT / 'reports/whisper_text_feature_extraction.json',
        {'status': 'completed', 'splits': splits, 'limit': limit, 'caches': reports,
         'public_embeddings_frozen': True, 'reference_transcript_consumed': False,
         'test_predictions_computed': False, 'quality_evidence': False})
    print(json.dumps(reports), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    extract(args.splits, args.limit)

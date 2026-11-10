"""CPU lexical context inferred from audio, never from reference transcripts/phones."""
import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch import nn

import schwa as core
from frozen_phoneme import RECORDING

ROOT = Path(__file__).resolve().parent
ASR_PATH = ROOT / 'data/pretrained/word_asr_base_fp16_linear.pt'
DICTIONARY_REVISION = '74790861f652b15e4ac49015a90074ad62a27690'
DICTIONARY_SOURCE = ROOT / 'data/pretrained/cmudict'
DICTIONARY_BLOBS = {'cmudict.dict': '2c0411740cce3e2026a80b90b650d5f6a7258164',
                    'LICENSE': '81a0ae8ec00962d2ccce9f93c1dcb17d3cb57abb'}
LEXICAL_RECIPE = {'version': 2, 'core_frames': 300, 'margin_frames': 100,
                  'maximum_frames': 500, 'pronunciation': 'per_ASR_word_span_acoustic_CTC_choice',
                  'unknown_words': 'one_spelling_edit_or_two_word_split; unresolved_span_has_no_prior',
                  'candidate_limit': 64, 'asr_evidence_weight': .25, 'edit_penalty': .5,
                  'recording_feature_storage': 'float16', 'aligned_window_storage': 'float16',
                  'input': 'audio_ASR_and_public_phoneme_emissions_only', 'features': 4}
ARPABET_IPA = {
    'AA': '\u0251\u02d0', 'AE': '\u00e6', 'AH': '\u028c', 'AO': '\u0254\u02d0',
    'AW': 'a\u028a', 'AY': 'a\u026a', 'B': 'b', 'CH': 't\u0283', 'D': 'd',
    'DH': '\u00f0', 'EH': '\u025b', 'ER': '\u025a', 'EY': 'e\u026a', 'F': 'f',
    'G': '\u0261', 'HH': 'h', 'IH': '\u026a', 'IY': 'i\u02d0', 'JH': 'd\u0292',
    'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': '\u014b', 'OW': 'o\u028a',
    'OY': '\u0254\u026a', 'P': 'p', 'R': '\u0279', 'S': 's', 'SH': '\u0283',
    'T': 't', 'TH': '\u03b8', 'UH': '\u028a', 'UW': 'u\u02d0', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': '\u0292',
}


def pronunciation_ids(phones, vocabulary):
    result = []
    for raw in phones:
        # Stress must survive until AH0 is mapped; ER0 remains a separate negative.
        symbol = '\u0259' if raw == 'AH0' else ARPABET_IPA.get(re.sub('[012]$', '', raw))
        if symbol not in vocabulary:
            raise ValueError('Unsupported dictionary phone: ' + raw)
        result.append(vocabulary[symbol])
    return result


def load_dictionary(path):
    result = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.partition('#')[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        word = re.sub(r'\(\d+\)$', '', word).upper()
        if phones:
            result.setdefault(word, []).append(phones)
    return result


def verify_dictionary_source():
    for name, expected in DICTIONARY_BLOBS.items():
        data = (DICTIONARY_SOURCE / name).read_bytes()
        blob_sha = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if blob_sha != expected:
            raise ValueError('Pinned dictionary Git blob changed: ' + name)


def prepare(args):
    if args.output.exists():
        raise ValueError('Preserving an existing ASR descriptor.')
    torch.set_num_threads(args.threads)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    path = ROOT / 'data/pretrained/torch_hub/checkpoints' / bundle._path
    if not path.is_file():
        core.fetch('https://download.pytorch.org/torchaudio/models/' + bundle._path, path)
    source_sha = core.digest(path)
    try:
        source = torch.load(path, weights_only=True, mmap=True)
    except RuntimeError:
        source = torch.load(path, weights_only=True)
    config = dict(bundle._params)
    if tuple(bundle._remove_aux_axis) != (1, 2, 3):
        raise ValueError('Unexpected official English CTC output mapping.')
    # Reproduce the official bundle's removal of fairseq BOS/PAD/EOS rows.
    keep = [i for i in range(source['aux.weight'].shape[0]) if i not in bundle._remove_aux_axis]
    if len(keep) != len(bundle.get_labels()):
        raise ValueError('English CTC labels do not match the output layer.')
    source['aux.weight'] = source['aux.weight'][keep]
    source['aux.bias'] = source['aux.bias'][keep]
    with torch.device('meta'):
        skeleton = torchaudio.models.wav2vec2_model(**config)
    linears = {name for name, module in skeleton.named_modules() if isinstance(module, nn.Linear)}
    state = {name: value.half() if name.rsplit('.', 1)[0] in linears else value
             for name, value in source.items()}
    del source, skeleton
    gc.collect()
    descriptor = {'model_config': config, 'model_state_dict': state,
        'source_bundle': 'torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H',
        'source_url': 'https://download.pytorch.org/torchaudio/models/' + bundle._path,
        'source_checkpoint_sha256': source_sha, 'sample_rate': 16000, 'layers': 12,
        'normalize_waveform': bundle._normalize_waveform, 'feature_kind': 'ctc_logits',
        'feature_dimension': len(bundle.get_labels()), 'quantization': 'float16_linear_storage',
        'labels': list(bundle.get_labels()), 'blank_id': 0,
        'weight_storage': 'float16 linear / float32 convolution and norm'}
    core.save_torch(args.output, descriptor)
    base = 'https://raw.githubusercontent.com/cmusphinx/cmudict/' + DICTIONARY_REVISION + '/'
    for name in ['cmudict.dict', 'LICENSE']:
        core.fetch(base + name, DICTIONARY_SOURCE / name)
    verify_dictionary_source()
    core.write_json(DICTIONARY_SOURCE / 'source.json', {
        'repo': 'cmusphinx/cmudict', 'revision': DICTIONARY_REVISION,
        'dictionary_sha256': core.digest(DICTIONARY_SOURCE / 'cmudict.dict'),
        'license_sha256': core.digest(DICTIONARY_SOURCE / 'LICENSE'),
        'license': 'BSD-style; retained source LICENSE'})
    core.write_json(args.output.with_suffix('.json'), {
        **{k: v for k, v in descriptor.items() if k != 'model_state_dict'},
        'prepared_sha256': core.digest(args.output), 'prepared_bytes': args.output.stat().st_size,
        'lexical_recipe': LEXICAL_RECIPE})
    print('Prepared audio ASR descriptor and pinned pronunciation dictionary.', flush=True)


def greedy_words(logits, labels, blank=0):
    return [word for word, _, _ in greedy_word_spans(logits, labels, blank)]


def greedy_word_spans(logits, labels, blank=0):
    if logits.ndim != 2 or logits.shape[1] != len(labels):
        raise ValueError('Invalid English ASR emission geometry.')
    characters, previous, start, words = [], None, 0, []
    for frame, token in enumerate(logits.argmax(dim=-1).tolist()):
        if token != previous and token != blank:
            if labels[token] == '|':
                if characters:
                    words.append((''.join(characters), start, frame))
                    characters = []
                start = frame
            else:
                characters.append(labels[token])
        previous = token
    if characters:
        words.append((''.join(characters), start, len(logits)))
    return words


def spelling_edits(word):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ'"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return ({left + right[1:] for left, right in splits if right}
            | {left + char + right[1:] for left, right in splits if right for char in alphabet}
            | {left + char + right for left, right in splits for char in alphabet}) - {word, ''}


def dictionary_lookup_words(words):
    """Small vocabulary subset for the one-recording smoke, equivalent to full lookup."""
    wanted = set(words)
    for word in words:
        wanted.update(spelling_edits(word))
        for i in range(2, len(word) - 1):
            for part in [word[:i], word[i:]]:
                wanted.add(part)
                wanted.update(spelling_edits(part))
    return wanted


def word_candidates(word, dictionary):
    if word in dictionary:
        return [([word], 0.)]
    candidates = [(tuple([candidate]), 1.) for candidate in spelling_edits(word) if candidate in dictionary]
    for i in range(2, len(word) - 1):
        left, right = word[:i], word[i:]
        if left in dictionary and right in dictionary:
            candidates.append(((left, right), .5))
        if right in dictionary:
            candidates.extend(((candidate, right), 1.5) for candidate in spelling_edits(left) if candidate in dictionary)
        if left in dictionary:
            candidates.extend(((left, candidate), 1.5) for candidate in spelling_edits(right) if candidate in dictionary)
    return [(list(phrase), cost) for phrase, cost in sorted(set(candidates), key=lambda item: (item[1], item[0]))
            [:LEXICAL_RECIPE['candidate_limit']]]


def ctc_state_posteriors(log_probs, targets, blank=0):
    """Exact CTC forward/backward, including repeated-label blank constraints."""
    if log_probs.ndim != 2 or len(log_probs) == 0 or not torch.isfinite(log_probs).all():
        raise ValueError('Invalid finite CTC emissions.')
    if any(t == blank or not 0 <= t < log_probs.shape[1] for t in targets):
        raise ValueError('Invalid CTC target.')
    minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
    if len(log_probs) < minimum:
        raise ValueError('Not enough frames for the CTC pronunciation.')
    symbols = torch.full((2 * len(targets) + 1,), blank, dtype=torch.long)
    symbols[1::2] = torch.tensor(targets, dtype=torch.long)
    skip = torch.zeros(len(symbols), dtype=torch.bool)
    skip[2:] = (symbols[2:] != blank) & (symbols[2:] != symbols[:-2])
    emissions = log_probs[:, symbols]
    forward = emissions.new_full(emissions.shape, -float('inf'))
    forward[0, 0] = emissions[0, 0]
    if targets:
        forward[0, 1] = emissions[0, 1]
    for t in range(1, len(emissions)):
        previous = forward[t - 1]
        value = previous.clone()
        value[1:] = torch.logaddexp(value[1:], previous[:-1])
        value[2:] = torch.logaddexp(value[2:], torch.where(skip[2:], previous[:-2], -float('inf')))
        forward[t] = value + emissions[t]
    terminals = forward[-1, -2:] if targets else forward[-1, -1:]
    normalizer = torch.logsumexp(terminals, dim=0)
    backward = emissions.new_full(emissions.shape, -float('inf'))
    backward[-1, -1] = 0.
    if targets:
        backward[-1, -2] = 0.
    for t in range(len(emissions) - 2, -1, -1):
        next_value = backward[t + 1] + emissions[t + 1]
        value = next_value.clone()
        value[:-1] = torch.logaddexp(value[:-1], next_value[1:])
        value[:-2] = torch.logaddexp(value[:-2], torch.where(skip[2:], next_value[2:], -float('inf')))
        backward[t] = value
    if not torch.isfinite(normalizer):
        raise ValueError('No finite CTC alignment path.')
    posterior = (forward + backward - normalizer).exp()
    posterior = posterior / posterior.sum(dim=-1, keepdim=True)
    return posterior, symbols, normalizer


def lexical_chunk(asr_logits, phone_logits, dictionary, asr_labels, vocabulary):
    if len(asr_logits) != len(phone_logits):
        raise ValueError('ASR and phoneme frame counts differ.')
    result = phone_logits.new_zeros((len(phone_logits), LEXICAL_RECIPE['features']))
    characters = {label: index for index, label in enumerate(asr_labels)}
    transcripts = []
    for word, start, end in greedy_word_spans(asr_logits, asr_labels):
        best = None
        phone_probabilities = phone_logits[start:end].float().log_softmax(-1)
        word_probabilities = asr_logits[start:end].float().log_softmax(-1)
        for phrase, edit_cost in word_candidates(word, dictionary):
            spelling = '|'.join(phrase)
            if any(char not in characters for char in spelling):
                continue
            try:
                _, _, char_evidence = ctc_state_posteriors(word_probabilities, [characters[c] for c in spelling])
            except ValueError:
                continue
            alternatives = [[]]
            for part in phrase:
                alternatives = [prefix + pronunciation_ids(pronunciation, vocabulary)
                                for prefix in alternatives for pronunciation in dictionary[part]][:16]
            for targets in alternatives:
                try:
                    posterior, symbols, evidence = ctc_state_posteriors(phone_probabilities, targets)
                except ValueError:
                    continue
                score = float(evidence + LEXICAL_RECIPE['asr_evidence_weight'] * char_evidence
                              - LEXICAL_RECIPE['edit_penalty'] * edit_cost)
                if best is None or score > best[0]:
                    best = (score, posterior, symbols, evidence, phrase, edit_cost)
        if best is None:
            transcripts.append(word)
            continue
        _, posterior, symbols, evidence, phrase, edit_cost = best
        result[start:end, 0] = posterior[:, symbols == vocabulary['\u0259']].sum(dim=-1)
        result[start:end, 1] = posterior[:, symbols != 0].sum(dim=-1)
        result[start:end, 2] = (evidence / (end - start)).exp()
        result[start:end, 3] = 1 / (1 + edit_cost)
        transcripts.extend(phrase)
    return result, transcripts


def bounded_lexical_features(asr_logits, phone_logits, dictionary, asr_labels, vocabulary):
    if len(asr_logits) != len(phone_logits):
        raise ValueError('ASR and phoneme timelines differ.')
    total = len(phone_logits)
    if total <= LEXICAL_RECIPE['maximum_frames']:
        features, words = lexical_chunk(asr_logits, phone_logits, dictionary, asr_labels, vocabulary)
        return features, [words]
    output = phone_logits.new_empty((total, LEXICAL_RECIPE['features']))
    transcripts = []
    core_frames, margin = LEXICAL_RECIPE['core_frames'], LEXICAL_RECIPE['margin_frames']
    for start in range(0, total, core_frames):
        end = min(total, start + core_frames)
        left, right = max(0, start - margin), min(total, end + margin)
        features, words = lexical_chunk(asr_logits[left:right], phone_logits[left:right], dictionary, asr_labels, vocabulary)
        output[start:end] = features[start-left:end-left]
        transcripts.append(words)
    return output, transcripts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ASR_PATH)
    parser.add_argument('--threads', type=int, default=1)
    prepare(parser.parse_args())

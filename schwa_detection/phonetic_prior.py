"""Complete pronunciation posterior inferred from audio, with exact legacy four-channel parity."""
import torch
from lexical_context import (LEXICAL_RECIPE, greedy_word_spans, word_candidates,
                             pronunciation_ids, ctc_state_posteriors)

PRIOR_RECIPE = {'version': 1, 'input': 'audio ASR and public phoneme emissions only',
    'legacy_channels': 4, 'additional_channels': 'Complete posterior over public phoneme vocabulary including blank',
    'reference_transcript_consumed': False, 'manual_phones_as_features': False,
    'selection_and_chunking': 'Exactly the existing lexical recipe; no new threshold or selection rule'}


def prior_chunk(asr_logits, phone_logits, dictionary, asr_labels, vocabulary):
    if len(asr_logits) != len(phone_logits):
        raise ValueError('ASR and phoneme frame counts differ.')
    result = phone_logits.new_zeros((len(phone_logits), LEXICAL_RECIPE['features'] + phone_logits.shape[-1]))
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
        for symbol in torch.unique(symbols).tolist():
            result[start:end, 4 + symbol] = posterior[:, symbols == symbol].sum(dim=-1)
        transcripts.extend(phrase)
    return result, transcripts


def bounded_prior_features(asr_logits, phone_logits, dictionary, asr_labels, vocabulary):
    if len(asr_logits) != len(phone_logits):
        raise ValueError('ASR and phoneme timelines differ.')
    total = len(phone_logits)
    if total <= LEXICAL_RECIPE['maximum_frames']:
        features, words = prior_chunk(asr_logits, phone_logits, dictionary, asr_labels, vocabulary)
        return features, [words]
    output = phone_logits.new_empty((total, LEXICAL_RECIPE['features'] + phone_logits.shape[-1]))
    transcripts = []
    core_frames, margin = LEXICAL_RECIPE['core_frames'], LEXICAL_RECIPE['margin_frames']
    for start in range(0, total, core_frames):
        end = min(total, start + core_frames)
        left, right = max(0, start - margin), min(total, end + margin)
        features, words = prior_chunk(asr_logits[left:right], phone_logits[left:right], dictionary, asr_labels, vocabulary)
        output[start:end] = features[start-left:end-left]
        transcripts.append(words)
    return output, transcripts


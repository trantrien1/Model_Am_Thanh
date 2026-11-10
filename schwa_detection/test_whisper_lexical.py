"""Check pronunciation stress, evidence choice and audio timeline independently."""
import itertools
import unittest

import numpy as np
import torch

from lexical_context import ctc_state_posteriors
from whisper_lexical import word_features


class WhisperLexicalChecks(unittest.TestCase):
    vocabulary = {'<pad>': 0, '\u0259': 1, '\u028c': 2, '\u025a': 3, 't': 4}

    def test_ctc_probability_matches_enumerated_paths(self):
        probabilities = torch.tensor([[.4, .5, .1], [.3, .6, .1], [.5, .4, .1]])
        _, _, evidence = ctc_state_posteriors(probabilities.log(), [1, 1])
        total = 0.
        for path in itertools.product(range(3), repeat=3):
            collapsed, previous = [], None
            for phone in path:
                if phone != previous and phone != 0:
                    collapsed.append(phone)
                previous = phone
            if collapsed == [1, 1]:
                total += float(np.prod([float(probabilities[t, p]) for t, p in enumerate(path)]))
        self.assertAlmostEqual(float(evidence.exp()), total, places=7)

    def test_stress_and_r_coloring_remain_distinct(self):
        emissions = torch.zeros(20, 5)
        word = [{'word': 'a', 'start': .1, 'end': .2, 'probability': .9}]
        ax = word_features(word, emissions, {'A': [['AH0']]}, self.vocabulary)
        ah = word_features(word, emissions, {'A': [['AH1']]}, self.vocabulary)
        er = word_features(word, emissions, {'A': [['ER0']]}, self.vocabulary)
        self.assertGreater(float(ax[:, 0].sum()), 0)
        self.assertEqual(float(ah[:, 0].sum()), 0)
        self.assertEqual(float(er[:, 0].sum()), 0)
        self.assertEqual(float(ax[:, 5].max()), 1)
        self.assertEqual(float(ah[:, 5].max()), 0)

    def test_alternative_selected_from_public_acoustic_evidence(self):
        emissions = torch.full((20, 5), -7.)
        emissions[:, 2] = 7.
        word = [{'word': 'a', 'start': .1, 'end': .2, 'probability': .9}]
        values = word_features(word, emissions, {'A': [['AH0'], ['AH1']]}, self.vocabulary)
        self.assertEqual(float(values[:, 0].sum()), 0)
        self.assertEqual(float(values[:, 5].max()), 0)
        self.assertGreater(float(values[:, 2].max()), .9)

    def test_unknown_timeline_overlap_and_finite_input(self):
        emissions = torch.zeros(25, 5)
        first = {'word': 'unknown', 'start': .1, 'end': .2, 'probability': .8}
        second = {**first, 'probability': .4}
        values = word_features([first, second], emissions, {}, self.vocabulary)
        # Frame center at 12.47 ms is before the padded start at 20 ms.
        self.assertTrue(torch.equal(values[0], torch.zeros(8)))
        self.assertAlmostEqual(float(values[5, 4]), .6, places=6)
        self.assertEqual(float(values[5, 7]), 1.)
        self.assertEqual(float(values[:, :4].sum()), 0.)
        self.assertEqual(float(values[24].sum()), 0.)
        with self.assertRaises(ValueError):
            word_features([first], emissions.fill_(float('nan')), {}, self.vocabulary)


if __name__ == '__main__':
    unittest.main()

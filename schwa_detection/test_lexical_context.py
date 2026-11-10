"""Independent CTC path enumeration and lexical input invariants."""
import itertools
import unittest
import argparse
import contextlib
import io
from unittest.mock import patch

import numpy as np
import torch

import lexical_train_cpu as training
import improve
import schwa as core
import test_schwa as fixtures

from lexical_context import (ctc_state_posteriors, pronunciation_ids, greedy_words,
                             lexical_chunk, bounded_lexical_features)


class LexicalChecks(unittest.TestCase):
    def test_one_epoch_lexical_head_save_and_reload_for_both_tasks(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fixture.fixture()
                rng = np.random.default_rng(17)
                parts = {}
                rows = core.read_csv(fixture.root / 'data/manifests/dataset.csv')
                for split in ['train', 'validation']:
                    subset = [r for r in rows if r['split'] == split]
                    parts[split] = rng.normal(size=(len(subset), 24, 1416)).astype(np.float16)
                recipe = {'main_encoder_sha256': 'fixture_encoder', 'asr_encoder_sha256': 'fixture_word_encoder',
                          'reference_transcript_consumed': False, 'manual_phones_consumed_as_features': False}
                with patch.object(improve, 'activate_dataset', return_value=fixture.root), \
                     patch.object(training, 'metadata', return_value=recipe), \
                     patch.object(improve, 'feature_cache', side_effect=lambda directory, rows, split: parts[split]), \
                     patch.object(training, 'lexical_window_cache', side_effect=lambda directory, rows, split, recipe: np.zeros((len(rows), 24, 4), np.float16)):
                    for task in ['phones', 'binary']:
                        run = fixture.root / ('lexical_' + task)
                        training.train(argparse.Namespace(run_dir=run, task=task, temporal='none', hidden=16,
                                       dropout=.2, lr=.0005, window_loss=.5, epochs=1, patience=1,
                                       batch_size=8, threads=1, seed=42))
                        checkpoint = torch.load(run / 'best.pt', weights_only=True)
                        self.assertEqual(checkpoint['backend'], 'wav2vec2_lexical_phone_head')
                        self.assertEqual(checkpoint['threshold_selection_split'], 'validation')
                        self.assertFalse(checkpoint['lexical']['reference_transcript_consumed'])
                        if task == 'binary':
                            self.assertEqual(checkpoint['phone_vocabulary'], ['non_ax', 'ax'])
                        head = improve.PhoneHead(**checkpoint['head_config']).eval()
                        head.load_state_dict(checkpoint['head_state_dict'])
                        features = training.CombinedFeatures(parts['validation'], np.zeros((len(parts['validation']), 24, 4), np.float16))
                        scores = improve.class_scores(head, features, checkpoint['phone_vocabulary'].index('ax'), checkpoint['pool_frames'])
                        labels = [int(r['label']) for r in rows if r['split'] == 'validation']
                        actual = core.binary_metrics(labels, scores, checkpoint['threshold'])
                        self.assertEqual(actual['confusion_matrix'], checkpoint['validation_metrics']['confusion_matrix'])
                        self.assertTrue((run / 'last.pt').is_file())
                        self.assertTrue((run / 'training_summary.json').is_file())
        finally:
            fixture.doCleanups()

    def test_ctc_evidence_matches_exhaustive_paths_including_repeated_labels(self):
        torch.manual_seed(9)
        probabilities = torch.randn(4, 3, dtype=torch.float64).softmax(-1)
        for targets in [[], [1], [1, 1], [1, 2]]:
            evidence = 0.
            for path in itertools.product(range(3), repeat=4):
                collapsed = [token for i, token in enumerate(path)
                             if token != 0 and (i == 0 or token != path[i-1])]
                if collapsed == targets:
                    evidence += probabilities[range(4), list(path)].prod()
            posterior, symbols, actual = ctc_state_posteriors(probabilities.log(), targets)
            torch.testing.assert_close(actual.exp(), torch.as_tensor(evidence), rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(posterior.sum(-1), torch.ones(4, dtype=torch.float64))
            self.assertTrue(torch.isfinite(posterior).all())

    def test_stress_and_r_colored_vowel_remain_separate(self):
        vocabulary = {'\u0259': 7, '\u028c': 33, '\u025a': 43}
        self.assertEqual(pronunciation_ids(['AH0', 'AH1', 'AH2', 'ER0'], vocabulary), [7, 33, 33, 43])

    def test_audio_ctc_words_and_unknown_span_preserves_known_span(self):
        labels = ['-', '|', 'A', 'H']
        tokens = [0, 2, 2, 0, 2, 1, 3, 0]
        emissions = torch.full((len(tokens), 4), -9.)
        emissions[range(len(tokens)), tokens] = 9.
        self.assertEqual(greedy_words(emissions, labels), ['AA', 'H'])
        features, words = lexical_chunk(emissions, torch.zeros(len(tokens), 8),
                                       {'AA': [['AH0']]}, labels, {'\u0259': 7})
        self.assertEqual(words, ['AA', 'H'])
        self.assertGreater(float(features[:5, 0].max()), 0.)
        self.assertTrue(torch.equal(features[5:, :3], torch.zeros(3, 3)))
        self.assertTrue(torch.equal(features[:5, 3], torch.ones(5)))
        self.assertTrue(torch.equal(features[5:, 3], torch.zeros(3)))

    def test_unknown_spelling_repair_uses_audio_and_public_dictionary_only(self):
        labels = ['-', '|', 'T', 'H', 'A', 'E']
        asr = torch.full((9, 6), -7.)
        asr[:, 0] = 7.
        for frame, token in [(1, 2), (3, 3), (5, 4)]:
            asr[frame, 0] = -7.
            asr[frame, token] = 7.
        phones = torch.full((9, 4), -7.)
        phones[:, 0] = 7.
        for frame, token in [(1, 2), (5, 3)]:
            phones[frame, 0] = -7.
            phones[frame, token] = 7.
        features, words = lexical_chunk(asr, phones, {'THE': [['DH', 'AH0']]}, labels,
                                       {'\u00f0': 2, '\u0259': 3})
        self.assertEqual(words, ['THE'])
        self.assertGreater(float(features[:, 0].max()), .9)

    def test_long_lexical_timeline_covers_every_frame(self):
        emissions = torch.zeros(1201, 3)
        emissions[:, 0] = 2.
        features, transcripts = bounded_lexical_features(emissions, emissions, {}, ['-', '|', 'A'], {'\u0259': 2})
        self.assertEqual(tuple(features.shape), (1201, 4))
        self.assertEqual(len(transcripts), 5)
        self.assertTrue(torch.equal(features, torch.zeros(1201, 4)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

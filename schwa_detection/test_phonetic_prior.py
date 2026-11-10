import unittest

import torch

from lexical_context import bounded_lexical_features
from phonetic_prior import bounded_prior_features
from phonetic_prior_features import window_path, build_recording


class PhoneticPriorTests(unittest.TestCase):
    def test_test_inputs_rejected_before_source_or_cache_access(self):
        with self.assertRaises(ValueError):
            window_path([{'split': 'test'}], 'test', {})
        with self.assertRaises(ValueError):
            build_recording({'split': 'test'}, {}, {}, [], {})

    def example(self, frames):
        asr = torch.full((frames, 3), -8.)
        asr[:, 2] = 8.
        phones = torch.full((frames, 4), -4.)
        phones[:, 1] = 4.
        return asr, phones, {'A': [['AH0']]}, ['<pad>', '|', 'A'], {'<pad>': 0, '\u0259': 1, '\u028c': 2, 'x': 3}

    def test_legacy_channels_and_full_phone_probability_mass(self):
        args = self.example(24)
        old, old_words = bounded_lexical_features(*args)
        new, new_words = bounded_prior_features(*args)
        torch.testing.assert_close(new[:, :4], old, rtol=0, atol=0)
        self.assertEqual(new_words, old_words)
        torch.testing.assert_close(new[:, 4:].sum(-1), torch.ones(24))
        torch.testing.assert_close(new[:, 5], old[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(new[:, 5:].sum(-1), old[:, 1])

    def test_bounded_long_audio_preserves_original_timeline(self):
        args = self.example(520)
        old, words = bounded_lexical_features(*args)
        new, new_words = bounded_prior_features(*args)
        self.assertEqual(tuple(new.shape), (520, 8))
        torch.testing.assert_close(new[:, :4], old, rtol=0, atol=0)
        self.assertEqual(new_words, words)

    def test_unresolved_word_has_no_invented_prior(self):
        args = list(self.example(24))
        args[2] = {}
        new, words = bounded_prior_features(*args)
        self.assertEqual(float(new.abs().sum()), 0.)
        self.assertEqual(words, [['A']])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()

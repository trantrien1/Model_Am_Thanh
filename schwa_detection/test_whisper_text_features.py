import unittest

import numpy as np
import torch

import whisper_text_features as features


class PublicTokenizerFixture:
    def encode(self, value):
        return {' of': [1], ' unfamiliar': [2, 3, 4]}[value]


class WhisperTextFeatureTests(unittest.TestCase):
    def test_unseen_word_uses_public_subwords_and_float32_mean(self):
        embedding = torch.tensor([[0., 0.], [2., 4.], [3., 6.], [6., 9.], [12., 15.]], dtype=torch.float16)
        expected = torch.tensor([7., 10.], dtype=torch.float16)
        actual = features.word_vector(' Unfamiliar,', PublicTokenizerFixture(), embedding)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(features.word_vector('OF', PublicTokenizerFixture(), embedding), embedding[1]))

    def test_focal_word_boundaries_and_outside_zero(self):
        embedding = torch.tensor([[0., 0.], [2., 4.], [3., 6.], [6., 9.], [12., 15.]], dtype=torch.float16)
        words = [{'word': ' of', 'start': .10, 'end': .20, 'probability': .9}]
        actual = features.focal_vectors(words, [-16000, 0, 16000], PublicTokenizerFixture(), embedding)
        self.assertEqual(actual.shape, (3, 24, 2))
        self.assertTrue(np.all(actual[0] == 0) and np.all(actual[2] == 0))
        self.assertTrue(np.all(actual[1, 0] == 0)) # .01246875, before the .02 margin edge
        np.testing.assert_array_equal(actual[1, 1], [2., 4.]) # .03246875
        self.assertTrue(np.all(actual[1, 14] == 0)) # .29246875, past .28

    def test_test_regions_rejected(self):
        with self.assertRaises(ValueError):
            features.window_path(features.ROOT / 'v5', [{'split': 'test'}], 'test', {})


if __name__ == '__main__':
    unittest.main()

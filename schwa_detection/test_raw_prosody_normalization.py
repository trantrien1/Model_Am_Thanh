import unittest

import numpy as np

import improve
import raw_prosody_train_cpu as training


class RawProsodyNormalizationTests(unittest.TestCase):
    def test_train_statistics_stay_fixed_on_validation(self):
        raw_train = np.stack([np.zeros((24, 48)), np.full((24, 48), 2.)]).astype(np.float16)
        mean, std = improve.spectral_statistics(raw_train)
        self.assertEqual(mean, [1.] * 48)
        self.assertEqual(std, [1.] * 48)
        validation = improve.AugmentedFeatures(np.zeros((1, 24, 1420), dtype=np.float32),
            np.full((1, 24, 48), 100., dtype=np.float16), {'mean': mean, 'std': std})
        actual = validation[:]
        self.assertEqual(actual.shape, (1, 24, 1468))
        np.testing.assert_array_equal(actual[..., -48:], 99.)
        np.testing.assert_array_equal(validation.mean, 1.)
        np.testing.assert_array_equal(validation.std, 1.)

    def test_test_features_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            training.load_features(training.ROOT / 'v5', [{'split': 'test'}], 'test', {})


if __name__ == '__main__':
    unittest.main()

import math
import unittest

import torch

from improve import AlignedMel, spectral_config


class RawProsodyFeatureTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.backends.mkldnn.enabled = False
        self.model = AlignedMel(spectral_config(True)).eval()

    def test_known_pitch_and_silence(self):
        t = torch.arange(8000, dtype=torch.float32) / 16000
        tone = .1 * torch.sin(2 * math.pi * 200 * t)
        values = self.model(torch.stack([tone, torch.zeros_like(tone)]))
        self.assertEqual(values.shape, (2, 24, 48))
        self.assertTrue(torch.isfinite(values).all())
        pitch = values[0, :, 40].exp()
        torch.testing.assert_close(pitch, torch.full_like(pitch, 200.), rtol=.02, atol=0)
        self.assertTrue(torch.all(values[0, :, 47] == 1))
        self.assertTrue(torch.all(values[1, :, 40] == 0) and torch.all(values[1, :, 47] == 0))

    def test_amplitude_changes_energy_but_not_pitch(self):
        t = torch.arange(8000, dtype=torch.float32) / 16000
        tone = .05 * torch.sin(2 * math.pi * 200 * t)
        values = self.model(torch.stack([tone, tone * 4]))
        torch.testing.assert_close(values[1, :, 43] - values[0, :, 43],
                                   torch.full((24,), math.log(4)), rtol=0, atol=2e-6)
        torch.testing.assert_close(values[0, :, 40], values[1, :, 40], rtol=0, atol=0)
        torch.testing.assert_close(values[0, :, 44], values[1, :, 44], rtol=0, atol=2e-6)


if __name__ == '__main__':
    unittest.main()

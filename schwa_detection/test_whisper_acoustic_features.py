import unittest
from unittest.mock import patch

import torch

import whisper_acoustic_features as features


class WhisperAcousticFeatureTests(unittest.TestCase):
    def test_physical_time_alignment_and_padding(self):
        # An encoder emitting physical time has an independently known answer
        # at the original XLSR frame centers (199.5 + 320*k samples).
        sequence = (torch.arange(100, dtype=torch.float32) * .02).reshape(-1, 1)
        starts = [-16000, 0, 10003, 999999]
        actual = features.aligned_windows(sequence, starts)
        expected = torch.tensor([
            [min(1.98, max(0, (start + 199.5 + 320 * index) / 16000)) for index in range(24)]
            for start in starts], dtype=torch.float32).half().float().unsqueeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_long_recording_has_no_frame_gap_or_offset(self):
        size = 16000 * 63 + 17
        audio = torch.arange(size, dtype=torch.float32) / 160000
        calls = []

        def time_encoder(encoder, piece):
            calls.append(len(piece))
            values = piece[0] + torch.arange(1500, dtype=torch.float32) * 320 / 160000
            return values.reshape(-1, 1).expand(-1, 512).half()

        with patch.object(features, 'encode_piece', side_effect=time_encoder):
            actual = features.recording_features(None, audio)
        expected = (torch.arange((size + 319) // 320, dtype=torch.float32) * .002).half()
        torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=.00390625)
        self.assertEqual(actual.shape, (3151, 512))
        self.assertTrue(all(length <= 480000 for length in calls))
        self.assertEqual(len(calls), 4)

    def test_test_inputs_and_invalid_audio_rejected(self):
        with self.assertRaises(ValueError):
            features.window_path(features.ROOT / 'v5', [{'split': 'test'}], 'test', {})
        for audio in [torch.empty(0), torch.zeros(2, 3), torch.tensor([float('nan')])]:
            with self.assertRaises(ValueError):
                features.recording_features(None, audio)


if __name__ == '__main__':
    unittest.main()

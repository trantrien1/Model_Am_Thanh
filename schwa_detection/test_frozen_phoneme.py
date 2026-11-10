"""Check CPU quantized encoder reconstruction and independence of input batches."""
import io
import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio

from frozen_phoneme import RECORDING, Float16StoredLinear, acoustic_features, rebuild_acoustic, recording_features, aligned_window_features, representation_state


class QuantizedEncoderChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = dict(extractor_mode='layer_norm', extractor_conv_layer_config=[(8, 10, 5), (8, 3, 2)],
                           extractor_conv_bias=True, encoder_embed_dim=32, encoder_projection_dropout=0.,
                           encoder_pos_conv_kernel=16, encoder_pos_conv_groups=4, encoder_num_layers=2,
                           encoder_num_heads=4, encoder_attention_dropout=0., encoder_ff_interm_features=64,
                           encoder_ff_interm_dropout=0., encoder_dropout=0., encoder_layer_norm_first=True,
                           encoder_layer_drop=0., aux_num_out=8)
        original = torchaudio.models.wav2vec2_model(**self.config).eval()
        self.reference = torch.ao.quantization.quantize_dynamic(original, {torch.nn.Linear}, dtype=torch.qint8)
        self.reference.feature_kind, self.reference.quantization = 'ctc_logits', 'dynamic_int8_linears'
        buffer = io.BytesIO()
        torch.save({'model_config': self.config, 'model_state_dict': self.reference.state_dict(),
                    'feature_kind': 'ctc_logits', 'quantization': 'dynamic_int8_linears'}, buffer)
        buffer.seek(0)
        self.state = torch.load(buffer, map_location='cpu', weights_only=True)

    def test_saved_encoder_reloads_with_identical_outputs(self):
        restored = rebuild_acoustic(self.state)
        audio = torch.randn(2, 640)
        with torch.inference_mode():
            expected = acoustic_features(self.reference, audio, 2)
            actual = acoustic_features(restored, audio, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(actual).all())

    def test_half_stored_linears_use_float_math_and_mapped_weights(self):
        reference = torchaudio.models.wav2vec2_model(**self.config).eval()
        linear_names = {name for name, module in reference.named_modules() if isinstance(module, torch.nn.Linear)}
        stored = {name: value.half() if name.rsplit('.', 1)[0] in linear_names else value
                  for name, value in reference.state_dict().items()}
        reference.load_state_dict({name: value.float() for name, value in stored.items()})
        reference.feature_kind = 'ctc_logits'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'encoder.pt'
            torch.save({'model_config': self.config, 'model_state_dict': stored,
                        'feature_kind': 'ctc_logits', 'quantization': 'float16_linear_storage'}, path)
            state = torch.load(path, weights_only=True, mmap=True)
            actual_model = rebuild_acoustic(state)
            for name, module in actual_model.named_modules():
                if name in linear_names:
                    self.assertIsInstance(module, Float16StoredLinear)
                    self.assertEqual(module.weight.dtype, torch.float16)
                    self.assertEqual(module.weight.data_ptr(), state['model_state_dict'][name + '.weight'].data_ptr())
            self.assertFalse(any(p.is_meta for p in actual_model.parameters()))
            audio = torch.randn(2, 640)
            with torch.inference_mode():
                actual = acoustic_features(actual_model, audio, 2)
                expected = acoustic_features(reference, audio, 2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            del actual_model, state

    def test_quantized_activation_scales_do_not_mix_files(self):
        model = rebuild_acoustic(self.state)
        quiet, loud = torch.randn(1, 640) * .01, torch.randn(1, 640) * 100
        with torch.inference_mode():
            alone = acoustic_features(model, quiet, 2)
            together = acoustic_features(model, torch.cat([quiet, loud]), 2)
        torch.testing.assert_close(alone, together[:1], rtol=0, atol=0)

    def test_joint_features_preserve_ctc_logits_and_do_not_mutate_source(self):
        original = rebuild_acoustic(self.state)
        joint_state = representation_state(self.state, 'joint')
        joint = rebuild_acoustic(joint_state)
        audio = torch.randn(2, 640)
        with torch.inference_mode():
            ctc = acoustic_features(original, audio, 2)
            features = acoustic_features(joint, audio, 2)
            alone = acoustic_features(joint, audio[:1], 2)
        self.assertEqual(features.shape[-1], 40)
        self.assertEqual(joint_state['feature_dimension'], 40)
        self.assertEqual(self.state['feature_kind'], 'ctc_logits')
        torch.testing.assert_close(features[..., 32:], ctc, rtol=0, atol=0)
        torch.testing.assert_close(alone, features[:1], rtol=0, atol=0)
        self.assertTrue(torch.isfinite(features).all())

    def test_focal_frames_keep_original_times_with_fractional_offsets(self):
        sequence = torch.arange(100, dtype=torch.float16).unsqueeze(-1)
        starts = [0, 160, 320, -8000, 32000]
        actual = aligned_window_features(sequence, starts)
        expected = (torch.tensor(starts).float().unsqueeze(-1) / 320 + torch.arange(24)).clamp(0, 99)
        torch.testing.assert_close(actual[..., 0], expected)
        self.assertEqual(tuple(actual.shape), (5, 24, 1))

    def test_long_recording_chunks_preserve_frame_timeline_and_bound_memory(self):
        class TimelineEncoder(torch.nn.Module):
            feature_kind, quantization, normalize_waveform = 'ctc_logits', None, False
            def __init__(self):
                super().__init__()
                self.sizes = []
            def forward(self, waves):
                self.sizes.append(waves.shape[-1])
                # A pointwise reference with the actual encoder's 400/320 geometry.
                frame_count = (waves.shape[-1] - 400) // 320 + 1
                return waves[:, torch.arange(frame_count) * 320, None], None
        model = TimelineEncoder()
        waveform = torch.arange(200321).float() / 320
        actual = recording_features(model, waveform, 2)
        expected = torch.arange((len(waveform) - 400) // 320 + 1).half().unsqueeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertLessEqual(max(model.sizes), RECORDING['maximum_samples'])
        self.assertGreater(len(model.sizes), 1)
        short = recording_features(model, torch.zeros(3200), 2)
        self.assertEqual(tuple(short.shape), (24, 1))

    def test_phoneme_elsewhere_does_not_turn_a_focal_region_positive(self):
        sequence = torch.zeros(120, 392, dtype=torch.float16)
        sequence[60, 7] = 10.
        focal = aligned_window_features(sequence, [0, 60 * 320])
        self.assertEqual(focal[0, :, 7].max().item(), 0.)
        self.assertEqual(focal[1, :, 7].max().item(), 10.)


if __name__ == '__main__':
    unittest.main(verbosity=2)

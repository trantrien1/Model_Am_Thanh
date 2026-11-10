"""Ensure cached-prefix adaptation and exported recording inference use identical math."""
import unittest
import argparse
import contextlib
import io
import numpy as np
from unittest.mock import patch
import torch
import torchaudio

from frozen_phoneme import encoder_prefix, recording_features, rebuild_acoustic
from finetune_cpu import chunk_plan, stitch_tail
import finetune_cpu as tune
import improve
import schwa as core
import test_schwa as fixtures


class PrefixAdaptationChecks(unittest.TestCase):
    def test_binary_supervision_preserves_ignored_frames_and_hard_negatives(self):
        vocabulary = ['pau', 'ax', 'ah', 'er', 'ih']
        labels = np.array([-100, 1, 0, 2, 3, 4], dtype=np.int64)
        np.testing.assert_array_equal(tune.supervision_targets(labels, vocabulary, 'binary'),
                                      [-100, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(tune.supervision_targets(labels, vocabulary, 'phones'), labels)

    def test_prepare_train_pack_and_inference_roundtrip(self):
        # This synthetic integration fixture verifies the pipeline, not quality.
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fixture.fixture()
                backbone = fixture.root / 'tiny_encoder.pt'
                state = {**self.state, 'layers': 2, 'feature_kind': 'ctc_logits', 'feature_dimension': 8}
                state.pop('prefix_rounding_layer')
                torch.save(state, backbone)
                with patch.object(improve, 'activate_dataset', return_value=fixture.root):
                    tune.prepare(argparse.Namespace(backbone=backbone, prefix_layers=1, splits=['train', 'validation']))
                    for task in ['phones', 'binary']:
                        with self.subTest(task=task):
                            run = fixture.root / ('run_' + task)
                            tune.train(argparse.Namespace(backbone=backbone, prefix_layers=1, run_dir=run, hidden=16,
                                       seed=42, threads=2, tail_lr=.0001, lr=.0005, epochs=1, warmup_epochs=0,
                                       accumulate=2, window_loss=.25, patience=1, task=task))
                            self.assertTrue((run / 'adapted_encoder.pt').is_file())
                            self.assertTrue((run / 'training_summary.json').is_file())
                            checkpoint = torch.load(run / 'best.pt', weights_only=True)
                            self.assertEqual(checkpoint['backbone_sha256'], core.digest(run / 'adapted_encoder.pt'))
                            self.assertEqual(checkpoint['settings']['task'], task)
                            self.assertEqual(checkpoint['threshold_selection_split'], 'validation')
                            if task == 'binary':
                                self.assertEqual(checkpoint['phone_vocabulary'], ['non_ax', 'ax'])
                            packed = run / 'packed.pt'
                            with patch.object(improve, 'BACKBONE', run / 'adapted_encoder.pt'):
                                improve.pack(argparse.Namespace(model=run / 'best.pt', output=packed))
                            detector = improve.TransferDetector(packed)
                            self.assertTrue(detector.describe()['self_contained'])
                            record = core.read_csv(fixture.root / 'data/manifests/recordings.csv')[0]
                            prediction = detector.predict(fixture.root / record['audio_path'])
                            self.assertTrue(torch.isfinite(torch.tensor(prediction['score'])))
                            self.assertEqual(detector.acoustic.prefix_rounding_layer, 1)
        finally:
            fixture.doCleanups()

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        self.config = dict(extractor_mode='layer_norm', extractor_conv_layer_config=[(8, 10, 5)] + [(8, 3, 2)] * 4 + [(8, 2, 2)] * 2,
                           extractor_conv_bias=True, encoder_embed_dim=32, encoder_projection_dropout=0.,
                           encoder_pos_conv_kernel=16, encoder_pos_conv_groups=4, encoder_num_layers=2,
                           encoder_num_heads=4, encoder_attention_dropout=0., encoder_ff_interm_features=64,
                           encoder_ff_interm_dropout=0., encoder_dropout=0., encoder_layer_norm_first=True,
                           encoder_layer_drop=0., aux_num_out=8)
        source = torchaudio.models.wav2vec2_model(**self.config).eval()
        linears = {name for name, module in source.named_modules() if isinstance(module, torch.nn.Linear)}
        self.state = {'model_config': self.config,
                      'model_state_dict': {name: value.half() if name.rsplit('.', 1)[0] in linears else value
                                           for name, value in source.state_dict().items()},
                      'feature_kind': 'ctc_embeddings_logits', 'quantization': 'float16_linear_storage',
                      'prefix_rounding_layer': 1}
        self.model = rebuild_acoustic(self.state)

    def cached_chunks(self, wave):
        if len(wave) < 8000:
            wave = torch.nn.functional.pad(wave, (0, 8000 - len(wave)))
        frames, plans = chunk_plan(len(wave))
        chunks = []
        with torch.no_grad():
            for plan in plans:
                inputs = wave[plan['input_start']:plan['input_end']].unsqueeze(0)
                chunks.append((plan, encoder_prefix(self.model, inputs, 1)[0].half()))
        return frames, chunks

    def test_cached_prefix_matches_bounded_recording_inference(self):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for samples in [3200, 55301, 200321]:
                wave = torch.randn(samples)
                frames, chunks = self.cached_chunks(wave)
                with torch.inference_mode():
                    actual = recording_features(self.model, wave, 2)
                    cached = stitch_tail(self.model, frames, chunks, 1)
                torch.testing.assert_close(cached, actual, rtol=0, atol=0)
                self.assertEqual(actual.shape[0], frames)
                self.assertLessEqual(max(p['input_end'] - p['input_start'] for p, _ in chunks), 160000)

    def test_only_tail_updates_and_rebuilt_inference_matches_updated_cache(self):
        self.model.requires_grad_(False)
        for parameter in self.model.encoder.transformer.layers[1].parameters():
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
        before = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        wave = torch.randn(20031)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            frames, chunks = self.cached_chunks(wave)
            optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad], lr=.01)
            values = stitch_tail(self.model, frames, chunks, 1)
            loss = values.float().square().mean()
            loss.backward()
            optimizer.step()
            changed = [name for name, value in self.model.state_dict().items() if not torch.equal(value, before[name])]
            self.assertTrue(changed)
            self.assertTrue(all(name.startswith('encoder.transformer.layers.1.') for name in changed))
            state = {**self.state, 'model_state_dict': self.model.state_dict()}
            restored = rebuild_acoustic(state)
            with torch.inference_mode():
                direct = recording_features(restored, wave, 2)
                from_cache = stitch_tail(self.model, frames, chunks, 1)
            torch.testing.assert_close(direct, from_cache, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)

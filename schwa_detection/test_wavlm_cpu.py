"""Check WavLM timing, input-only centering and resumable CPU-head integration."""
import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

import schwa as core
import improve
import frozen_wavlm as encoder
import wavlm_train_cpu as training
import test_schwa as fixtures


class TimedEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def extract_features(self, inputs, num_layers):
        self.calls.append(inputs.shape[-1])
        frames = (inputs.shape[-1] - 400) // 320 + 1
        base = inputs[:, 0].reshape(-1, 1, 1)
        values = base + torch.arange(frames).reshape(1, -1, 1)
        return [(values + layer).expand(-1, -1, 768) for layer in range(1, num_layers + 1)], None


class WavLMCPUChecks(unittest.TestCase):
    def test_layer_mean_bounded_chunks_and_short_padding_keep_global_timeline(self):
        torch.set_num_threads(2)
        for samples, frames in [(3200, 24), (55301, 172), (200321, 625)]:
            with self.subTest(samples=samples):
                model = TimedEncoder()
                waveform = torch.arange(samples).float() / 320
                actual = encoder.recording_embeddings(model, waveform)
                expected = (torch.arange(frames).float() + 9).half()
                torch.testing.assert_close(actual[:, 0], expected, rtol=0, atol=0)
                self.assertEqual(list(actual.shape), [frames, 768])
                self.assertLessEqual(max(model.calls), 160000)

    def test_centering_removes_unlabeled_channel_offsets(self):
        values = torch.arange(8).reshape(-1, 1).float().expand(-1, 768) * .5
        shifted = values + torch.arange(768).float().reshape(1, -1) * .125
        a, b = training.centered_embeddings(values), training.centered_embeddings(shifted)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(a.float().mean(0), torch.zeros(768), rtol=0, atol=0)

    def test_extract_two_head_types_and_exact_training_resume(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                fixture.fixture()
                backbone = fixture.root / 'synthetic_public.pt'
                torch.save({'recipe': encoder.WAVLM_RECIPE}, backbone)
                with patch.object(improve, 'activate_dataset', return_value=fixture.root), \
                     patch.object(training, 'rebuild_wavlm', return_value=TimedEncoder()):
                    training.extract(argparse.Namespace(backbone=backbone, splits=['train', 'validation']))
                    rows = core.read_csv(fixture.root / 'data/manifests/dataset.csv')
                    with self.assertRaisesRegex(ValueError, 'never consume test'):
                        training.FeatureWindows(fixture.root, [r for r in rows if r['split'] == 'test'], core.digest(backbone))
                    for temporal in ['none', 'gru']:
                        args = argparse.Namespace(backbone=backbone, run_dir=fixture.root / ('run_' + temporal),
                                                  resume=False, hidden=16, temporal=temporal, seed=42, threads=2,
                                                  epochs=2, patience=5, lr=.0005, batch_size=4)
                        training.train(args)
                        checkpoint = torch.load(args.run_dir / 'best.pt', weights_only=True)
                        self.assertEqual(checkpoint['backend'], 'wavlm_phone_head')
                        self.assertEqual(checkpoint['threshold_selection_split'], 'validation')
                        self.assertFalse(checkpoint['provenance']['test_used_for_selection'])
                        if temporal == 'none':
                            uninterrupted = torch.load(args.run_dir / 'last.pt', weights_only=True)
                            args.run_dir = fixture.root / 'interrupted'
                            original_save = core.save_torch
                            def interrupt_after_saved_epoch(path, values):
                                original_save(path, values)
                                if path.name == 'last.pt':
                                    raise InterruptedError('Simulated completed-epoch interruption')
                            with patch.object(core, 'save_torch', side_effect=interrupt_after_saved_epoch):
                                with self.assertRaises(InterruptedError):
                                    training.train(args)
                            args.resume = True
                            training.train(args)
                            resumed = torch.load(args.run_dir / 'last.pt', weights_only=True)
                            for name, values in uninterrupted['model_state_dict'].items():
                                torch.testing.assert_close(values, resumed['model_state_dict'][name], rtol=0, atol=0)
                            torch.testing.assert_close(uninterrupted['torch_random_state'], resumed['torch_random_state'], rtol=0, atol=0)
                            self.assertEqual(uninterrupted['numpy_random_state'], resumed['numpy_random_state'])
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main(verbosity=2)

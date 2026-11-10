"""Actual optimizer/RNG continuation and metadata rejection for Whisper heads."""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import improve
import schwa as core
from whisper_train_cpu import process_live, restore_training, step_epoch, training_snapshot


class WhisperTrainingChecks(unittest.TestCase):
    def test_interrupted_training_matches_three_uninterrupted_epochs(self):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.backends.mkldnn.enabled = False
        features = np.random.default_rng(77).normal(size=(9, 24, 14)).astype(np.float16)
        targets = np.random.default_rng(78).integers(0, 2, size=(9, 24), dtype=np.int64)
        targets[:, 2] = -100
        truth = torch.tensor([0., 1., 0., 1., 0., 1., 0., 1., 0.])
        criterion = torch.nn.CrossEntropyLoss(ignore_index=-100)
        provenance = {'seed': 42, 'feature_sha256': 'synthetic-engineering-only'}
        def new():
            torch.manual_seed(42)
            model = improve.PhoneHead(2, hidden=16, dropout=.3, temporal='gru', input_dim=14, group_sizes=[8, 6])
            return model, torch.optim.AdamW(model.parameters(), lr=.0005), np.random.default_rng(42)
        uninterrupted, full_optimizer, full_rng = new()
        for _ in range(3):
            step_epoch(uninterrupted, full_optimizer, features, targets, truth, criterion, full_rng, batch_size=4)
        expected_rng = torch.get_rng_state().clone()
        interrupted, optimizer, rng = new()
        step_epoch(interrupted, optimizer, features, targets, truth, criterion, rng, batch_size=4)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'last.pt'
            core.save_torch(path, training_snapshot(interrupted, optimizer, rng, [], (.5, .5, .5, -.1), 1, 0., provenance))
            restored, restored_optimizer, restored_rng = new()
            snapshot = torch.load(path, weights_only=True)
            restore_training(snapshot, restored, restored_optimizer, restored_rng, provenance)
            for _ in range(2):
                step_epoch(restored, restored_optimizer, features, targets, truth, criterion, restored_rng, batch_size=4)
            for key, value in uninterrupted.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[key]), key)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            self.assertEqual(full_rng.bit_generator.state, restored_rng.bit_generator.state)
            with self.assertRaisesRegex(ValueError, 'provenance'):
                restore_training(snapshot, restored, restored_optimizer, restored_rng, {'seed': 43})

    def test_process_check_and_invalid_training_loss(self):
        self.assertTrue(process_live(os.getpid()))
        self.assertFalse(process_live(2147483647))
        torch.manual_seed(42)
        model = improve.PhoneHead(2, hidden=16, input_dim=14)
        optimizer = torch.optim.AdamW(model.parameters())
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            step_epoch(model, optimizer, np.full((1, 24, 14), np.nan, dtype=np.float32),
                       np.zeros((1, 24), dtype=np.int64), torch.zeros(1),
                       torch.nn.CrossEntropyLoss(), np.random.default_rng(42))


if __name__ == '__main__':
    unittest.main()

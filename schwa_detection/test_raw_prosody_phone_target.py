import unittest

import numpy as np
import torch
from torch import nn

import raw_prosody_phone_train_cpu as training


class BiasHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(41))

    def forward(self, values):
        return self.bias.expand(values.shape[0], values.shape[1], -1)


class PhoneTargetTests(unittest.TestCase):
    def test_presence_gradient_uses_actual_phone_index(self):
        model = BiasHead()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        loss = training.step_epoch(model, optimizer, np.zeros((1, 24, 1), np.float32),
            np.zeros((1, 24), np.int64), torch.ones(1),
            lambda logits, targets: logits.sum() * 0,
            np.random.default_rng(42), 1, 1., target_index=5)
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(float(model.bias[5]), 0)
        self.assertLess(float(model.bias[1]), 0)
        self.assertEqual(int(model.bias.argmax()), 5)

    def test_test_features_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            training.load_features(training.ROOT / 'v5', [{'split': 'test'}], 'test', {})


if __name__ == '__main__':
    unittest.main()

import unittest

import numpy as np
import torch
from torch.nn import functional as F

import improve
import lexical_train_cpu as lexical
from native_aux_train_cpu import weighted_loss


class NativeAuxiliaryLossTests(unittest.TestCase):
    def test_manual_weight_matches_existing_objective_with_ignored_frames(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 3, 2)
        targets = torch.tensor([[0, 1, -100], [1, 0, 0]])
        truth = torch.tensor([1., 0.])
        weights = torch.tensor([.4, 1.6])
        expected = F.cross_entropy(logits.transpose(1, 2), targets, weight=weights, ignore_index=-100)
        expected += .5 * improve.window_presence_loss(logits, 1, truth)
        actual = weighted_loss(logits, targets, truth, torch.ones(2), weights)
        torch.testing.assert_close(actual, expected)

    def test_auxiliary_window_gradient_has_quarter_weight(self):
        logits = torch.zeros(2, 1, 2, requires_grad=True)
        loss = weighted_loss(logits, torch.full((2, 1), -100), torch.ones(2),
                             torch.tensor([1., .25]), torch.ones(2))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertAlmostEqual(float(logits.grad[0, 0, 1] / logits.grad[1, 0, 1]), 4.)

    def test_join_keeps_auxiliary_lexical_inputs_zero_and_row_order(self):
        manual = lexical.CombinedFeatures(np.ones((2, 24, 1416), np.float16),
                                          np.full((2, 24, 4), 3., np.float16))
        native = lexical.CombinedFeatures(np.full((2, 24, 1416), 2., np.float16),
                                          np.zeros((2, 24, 4), np.float16))
        joined = improve.JoinedFeatures([(manual, np.arange(2)), (native, np.arange(2))])
        batch = joined[np.array([3, 0, 2])]
        np.testing.assert_array_equal(batch[[0, 2], :, -4:], 0.)
        np.testing.assert_array_equal(batch[1, :, -4:], 3.)
        np.testing.assert_array_equal(batch[[0, 2], :, :1416], 2.)


if __name__ == '__main__':
    unittest.main()

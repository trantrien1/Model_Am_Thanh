"""Verify pooled feature semantics and portable tree inference independently."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from pooled_classifiers import pooled_features, summary_cache, tree_arrays, tree_scores


class PooledChecks(unittest.TestCase):
    def test_pooled_feature_order_probabilities_and_interaction(self):
        values = np.zeros((1, 24, 10), np.float32)
        values[:, :, 6:] = [.5, .75, .125, 1.]
        values[:, :12, 6] = .75
        actual = pooled_features(values, [1, 2, 3], embedding_dim=2, phone_dim=4)
        self.assertEqual(actual.shape, (1, 43))
        self.assertEqual(float(actual[0, 6]), .625)
        self.assertEqual(float(actual[0, 16]), .125)
        self.assertEqual(float(actual[0, 26]), .75)
        np.testing.assert_array_equal(actual[0, 30:39], np.full(9, .25))
        np.testing.assert_array_equal(actual[0, 39:], [.05859375, .0703125, .0703125, .5])

    def test_portable_scores_match_both_fitted_ensembles_after_reload(self):
        rng = np.random.default_rng(22)
        training = rng.normal(size=(90, 13)).astype(np.float32)
        labels = (training[:, 0] + .7*training[:, 1] > 0).astype(int)
        probes = rng.normal(size=(57, 13)).astype(np.float32)
        candidates = [('extra_trees', ExtraTreesClassifier(n_estimators=13, min_samples_leaf=2, random_state=5)),
                      ('hist_gradient_boosting', HistGradientBoostingClassifier(max_iter=9, min_samples_leaf=3,
                                                                             early_stopping=False, random_state=5))]
        with tempfile.TemporaryDirectory() as temporary, threadpool_limits(limits=2):
            for kind, model in candidates:
                with self.subTest(kind=kind):
                    model.fit(training, labels)
                    state = tree_arrays(model, kind)
                    path = Path(temporary) / (kind + '.pt')
                    torch.save(state, path)
                    restored = torch.load(path, weights_only=True)
                    inputs = np.concatenate([training, probes])
                    np.testing.assert_allclose(tree_scores(restored, inputs), model.predict_proba(inputs)[:, 1],
                                               rtol=0, atol=1e-12)
                    # Include probes lying at exported numeric split thresholds.
                    root = state['trees'][0]
                    if not root['leaf'][0]:
                        edge = np.zeros((1, 13), np.float32)
                        edge[0, int(root['feature'][0])] = float(root['threshold'][0])
                        np.testing.assert_allclose(tree_scores(restored, edge), model.predict_proba(edge)[:, 1],
                                                   rtol=0, atol=1e-12)

    def test_test_features_rejected_before_any_cache_is_read(self):
        with self.assertRaisesRegex(ValueError, 'must not consume test'):
            summary_cache(Path('unused'), [{'split': 'test'}], 'test', {}, [1, 2, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)

import numpy as np
import pytest
import torch

from src.common import load_config
from src.features import LogMel, training_statistics
from src.metrics import binary_metrics, select_threshold
from src.model import KeywordCNN


def test_feature_shape_finite_and_raw_logits(toy_project):
    cfg, _ = toy_project
    torch.set_num_threads(2)
    features = LogMel(cfg["features"])(torch.zeros(2, 80000))
    assert features.shape == (2, 64, 497)
    assert torch.isfinite(features).all()
    model = KeywordCNN(cfg["model"])
    logits = model(features.unsqueeze(1))
    loss = torch.nn.BCEWithLogitsLoss()(logits, torch.tensor([0., 1.]))
    loss.backward()
    assert logits.shape == (2,)
    assert torch.isfinite(loss)
    assert model.classifier[-1].__class__ is torch.nn.Linear


def test_normalization_train_only():
    features = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    rows = [{"split": "train"}] * 3
    statistics = training_statistics(features, rows)
    assert np.allclose(statistics["mean"], features.mean(axis=(0, 2)))
    assert np.allclose(statistics["std"], features.std(axis=(0, 2)))
    with pytest.raises(ValueError, match="exclusively"):
        training_statistics(features, [{"split": "test"}] * 3)


def test_metric_undefined_cases_and_threshold_ties():
    negative = binary_metrics([0, 0], [0.1, 0.8], 0.5)
    assert negative["fpr"] == 0.5
    assert negative["recall"] is None and negative["auroc"] is None
    assert negative["f1"] is None and negative["undefined_reasons"]
    positive = binary_metrics([1, 1], [0.2, 0.8], 0.5)
    assert positive["fnr"] == 0.5 and positive["fpr"] is None
    threshold, sweep = select_threshold([0, 1], [0.01, 0.8])
    assert threshold == 0.8  # Identical F1/FPR -> highest threshold.
    assert len(sweep) == 91

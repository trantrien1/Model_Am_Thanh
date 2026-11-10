import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


def binary_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid metric inputs.")
    if not set(labels).issubset({0, 1}):
        raise ValueError("Positive label must be 1, negative label 0.")
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel().tolist()
    reasons = {}

    def ratio(name, numerator, denominator, reason):
        if denominator == 0:
            reasons[name] = reason
            return None
        return numerator / denominator

    positives, negatives = tp + fn, tn + fp
    precision = ratio("precision", tp, tp + fp, "No predicted positives.") if positives else None
    if not positives:
        reasons["precision"] = "No positive ground-truth examples in this group."
    recall = ratio("recall", tp, positives, "No positive ground-truth examples.")
    f1 = ratio("f1", 2 * tp, 2 * tp + fp + fn, "No true or predicted positives.") if positives else None
    if not positives:
        reasons["f1"] = "No positive ground-truth examples in this group."
    result = {"files": len(labels), "positive_label": 1, "threshold": float(threshold),
              "accuracy": ratio("accuracy", tp + tn, len(labels), "Empty group."),
              "precision": precision, "recall": recall, "f1": f1,
              "fpr": ratio("fpr", fp, negatives, "No negative ground-truth examples."),
              "fnr": ratio("fnr", fn, positives, "No positive ground-truth examples."),
              "confusion_matrix": [[tn, fp], [fn, tp]], "confusion_matrix_order": [0, 1],
              "auroc": None, "average_precision": None, "undefined_reasons": reasons}
    if positives and negatives:
        result["auroc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(average_precision_score(labels, probabilities))
    else:
        reasons["auroc"] = "Both ground-truth labels are required."
        reasons["average_precision"] = "Both ground-truth labels are required for this comparative report."
    return result


def select_threshold(labels, probabilities):
    if set(labels) != {0, 1}:
        raise ValueError("Threshold selection requires both validation labels.")
    candidates = [binary_metrics(labels, probabilities, round(index / 100, 2)) for index in range(5, 96)]
    best = max(candidates, key=lambda row: (row["f1"], -row["fpr"], row["threshold"]))
    return best["threshold"], candidates

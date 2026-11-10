"""Evaluate frozen weights/preprocessing/validation threshold, with subgroup and error reports."""
import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from .artifacts import load_inference
from .audit_dataset import audit
from .common import directory, read_csv, read_json, sha256_file, timestamp, write_csv, write_json, project_path
from .dataset import FeatureDataset
from .features import cache_features
from .metrics import binary_metrics
from .train import epoch_pass, loader_for


def grouped_metrics(rows, probabilities, threshold):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups["background_category:" + row["background_category"]].append(index)
        if row["label"] == "0" and row["inserted_word"]:
            groups["negative_word:" + row["inserted_word"]].append(index)
        if row["mix_level_mode"] == "clean":
            groups["clean_background"].append(index)
        elif row["mix_level_mode"] == "fallback_dbfs":
            groups["fallback_dbfs"].append(index)
        else:
            snr = float(row["snr_db_requested"])
            group = "0-5" if 0 <= snr < 5 else "5-10" if 5 <= snr < 10 else "10-15" if 10 <= snr <= 15 else "below-0"
            groups["snr_db:" + group].append(index)
    return {name: binary_metrics([int(rows[index]["label"]) for index in indices],
                                 [probabilities[index] for index in indices], threshold)
            for name, indices in sorted(groups.items())}


def confusion_plot(metrics, output):
    matrix = np.array(metrics["confusion_matrix"])
    figure, axes = plt.subplots(figsize=(4, 4))
    axes.imshow(matrix, cmap="Blues")
    for (y, x), count in np.ndenumerate(matrix):
        axes.text(x, y, str(count), ha="center", va="center")
    axes.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted label", ylabel="True label", title="Positive label = 1")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def evaluate(run_dir, split="test", device="auto", config_path=None, supplemental=False, allow_smoke=False):
    run_dir = Path(run_dir).resolve()
    model, checkpoint, threshold_artifact, cfg, device = load_inference(run_dir, device, config_path)
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    if checkpoint["smoke_subset"] and not allow_smoke:
        raise ValueError("This is a smoke subset checkpoint. Use a full training run, or explicitly --allow-smoke for diagnostic results only.")
    if supplemental:
        audit(cfg, supplemental=True)
        manifests = directory(cfg, "manifests")
        metadata = read_json(manifests / "supplemental_metadata.json")
        if metadata["source_manifest_sha256"] != checkpoint["source_provenance"]["source_manifest_sha256"]:
            raise ValueError("Supplemental sources differ from the frozen run.")
        rows = read_csv(manifests / "supplemental.csv")
        output_name = "supplemental_test"
    else:
        manifest = run_dir / "manifests/dataset.csv"
        if sha256_file(manifest) != checkpoint["source_provenance"]["dataset_manifest_sha256"]:
            raise ValueError("Run manifest does not match frozen checkpoint provenance.")
        rows = [row for row in read_csv(manifest) if row["split"] == split]
        output_name = split
    if not rows:
        raise ValueError(f"No {output_name} rows to evaluate.")
    for row in rows:
        if sha256_file(project_path(cfg, row["path"])) != row["audio_sha256"]:
            raise ValueError(f"Evaluation audio changed after audit: {row['sample_id']}")
    feature_array, _ = cache_features(cfg, rows, output_name, directory(cfg, "features"), device)
    dataset = FeatureDataset(feature_array, rows, checkpoint["normalization"])
    with torch.no_grad():
        loss, labels, probabilities = epoch_pass(model, loader_for(dataset, cfg, device), device, nn.BCEWithLogitsLoss(), amp=False)
    threshold = threshold_artifact["threshold"]
    overall = binary_metrics(labels, probabilities, threshold)
    report = {"created_at": timestamp(), "split": output_name, "smoke_subset_model": checkpoint["smoke_subset"],
              "target_word": checkpoint["target_word"], "positive_label": 1, "loss": loss,
              "checkpoint_sha256": threshold_artifact["checkpoint_sha256"], "threshold_source": "validation",
              "selected_epoch": checkpoint["epoch"], "overall": overall,
              "at_threshold_0_5": binary_metrics(labels, probabilities, 0.5),
              "groups": grouped_metrics(rows, probabilities, threshold),
              "file_count": len(rows), "background_count": len({row["background_id"] for row in rows}),
              "independent_esc_source_count": len({row["background_source_id"] for row in rows}),
              "speaker_count": len({row["speaker_id"] for row in rows if row["speaker_id"]}),
              "limitations": ["Synthetic labels; not all ESC backgrounds have been verified by human listening.",
                              "Variants from one background or original ESC recording are correlated.",
                              "Probability is an uncalibrated model score.", "chưa đánh giá trên bản ghi thực tế"],
              "supplemental": supplemental}
    output = run_dir / "evaluation" / output_name
    output.mkdir(parents=True, exist_ok=True)
    predictions = [{**row, "probability": probability, "threshold": threshold, "prediction": int(probability >= threshold)}
                   for row, probability in zip(rows, probabilities)]
    fields = list(predictions[0])
    write_json(output / "metrics.json", report)
    write_csv(output / "predictions.csv", predictions)
    for kind, true, predicted in (("false_positives", "0", 1), ("false_negatives", "1", 0)):
        errors = [row for row in predictions if row["label"] == true and row["prediction"] == predicted]
        write_csv(output / f"{kind}.csv", sorted(errors, key=lambda row: row["probability"], reverse=kind == "false_positives"), fields)
    confusion_plot(overall, output / "confusion_matrix.png")
    text = [f"# Keyword spotting: {output_name}", "", f"Target: `{checkpoint['target_word']}`. Positive label: 1.", "",
            f"Checkpoint epoch: {checkpoint['epoch']}; threshold selected on validation: {threshold:.2f}.",
            f"Files: {len(rows)}; backgrounds: {report['background_count']}; independent original ESC recordings: {report['independent_esc_source_count']}; speakers: {report['speaker_count']}.", "",
            "| Metric | Value |", "| --- | --- |"]
    for name in ("accuracy", "precision", "recall", "f1", "auroc", "average_precision", "fpr", "fnr"):
        value = overall[name]
        text.append(f"| {name} | {value:.6f} |" if value is not None else f"| {name} | null: {overall['undefined_reasons'].get(name)} |")
    text.extend(["", "## Limitations", "", *[f"- {value}" for value in report["limitations"]]])
    if supplemental:
        text.extend(["", "Exploratory supplemental test only; excluded from main test and model tuning."])
    if checkpoint["smoke_subset"]:
        text.extend(["", "SMOKE SUBSET MODEL: diagnostic results, not a full baseline experiment."])
    (output / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    if not supplemental and split == "test":
        examples = run_dir / "examples"
        examples.mkdir(exist_ok=True)
        example_predictions = []
        for label, name in (("1", "positive"), ("0", "negative")):
            candidates = [row for row in predictions if row["label"] == label]
            if candidates:
                selected = max(candidates, key=lambda row: row["probability"]) if label == "1" else min(candidates, key=lambda row: row["probability"])
                shutil.copy2(project_path(cfg, selected["path"]), examples / f"test_{name}.wav")
                example_predictions.append({**selected, "example_file": f"examples/test_{name}.wav"})
        write_json(examples / "predictions.json", example_predictions)
    print(f"{output_name}: {overall}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen artifacts with the validation-selected threshold.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--config", help="Locate project data after moving artifacts; preprocessing is always read from checkpoint.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--supplemental", action="store_true")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    evaluate(args.run_dir, args.split, args.device, args.config, args.supplemental, args.allow_smoke)


if __name__ == "__main__":
    main()

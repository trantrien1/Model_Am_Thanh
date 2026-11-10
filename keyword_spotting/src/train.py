"""Full baseline training with audited data, resumable epoch checkpoints and validation threshold."""
import os
import random
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .artifacts import save_checkpoint, select_device
from .audit_dataset import audit
from .common import (code_hash, config_parser, directory, environment_info, load_config, object_hash,
                     public_config, read_csv, read_json, seed_everything, sha256_file,
                     timestamp, write_csv, write_json)
from .dataset import FeatureDataset
from .features import cache_features, training_statistics
from .metrics import binary_metrics, select_threshold
from .model import KeywordCNN


def epoch_pass(model, loader, device, criterion, optimizer=None, scaler=None, amp=False, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, examples, labels, probabilities = 0.0, 0, [], []
    for batch_index, (features, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(features)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Training/validation loss is non-finite.")
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        loss_sum += float(loss.detach()) * len(targets)
        examples += len(targets)
        labels.extend(targets.detach().cpu().int().tolist())
        probabilities.extend(torch.sigmoid(logits.detach().float()).cpu().tolist())
    if not examples:
        raise ValueError("No batches processed.")
    return loss_sum / examples, labels, probabilities


def loader_for(dataset, cfg, device, training=False, epoch=0, batch_size=None):
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    return DataLoader(dataset, batch_size=batch_size or cfg["train"]["batch_size"], shuffle=training,
                      num_workers=cfg["train"]["num_workers"], pin_memory=device.type == "cuda",
                      generator=generator)


def plot_history(history, output):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["validation_loss"] for row in history], label="validation")
    axes[0].set(xlabel="Epoch", ylabel="BCE loss")
    axes[0].legend()
    axes[1].plot(epochs, [row["validation_f1_at_0_5"] for row in history])
    axes[1].set(xlabel="Epoch", ylabel="Validation F1 at 0.5")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def train(cfg, run_dir=None, resume=False, smoke=False):
    if cfg["train"]["max_epochs"] < 1 or cfg["train"]["batch_size"] < 1 or cfg["train"]["patience"] < 1:
        raise ValueError("max_epochs, batch_size and patience must be positive.")
    torch.set_num_threads(cfg["train"]["cpu_threads"])
    seed_everything(cfg["seed"])
    device = select_device(cfg["train"]["device"])
    run_dir = Path(run_dir).resolve() if run_dir else directory(cfg, "run_dir")
    if smoke and run_dir == directory(cfg, "run_dir"):
        run_dir = run_dir.with_name(run_dir.name + "_smoke")
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "last.pt").exists() and not resume:
        raise ValueError(f"Run already has checkpoints: {run_dir}. Use --resume or another --run-dir.")
    if resume and (run_dir / "last.pt").exists():
        prior = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if prior["code_sha256"] != code_hash():
            raise ValueError("Training code changed since checkpoint. Use the original source snapshot to resume or start a new run.")
    # Always recheck audio bytes and identities before starting/resuming full training.
    audit_report = audit(cfg)
    manifests = directory(cfg, "manifests")
    all_rows = read_csv(manifests / "dataset.csv")
    train_rows = [row for row in all_rows if row["split"] == "train"]
    validation_rows = [row for row in all_rows if row["split"] == "validation"]
    if smoke:
        train_rows, validation_rows = train_rows[:min(64, len(train_rows))], validation_rows[:min(32, len(validation_rows))]
    config_hash = object_hash(public_config(cfg))
    if resume and (run_dir / "status.json").exists():
        status = read_json(run_dir / "status.json")
        if status.get("training_complete") and (run_dir / "threshold.json").is_file():
            checkpoint = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
            if checkpoint["config_sha256"] != config_hash or checkpoint["source_provenance"]["dataset_manifest_sha256"] != audit_report["manifest_sha256"]:
                raise ValueError("Completed run does not match current config/data.")
            print("Training and threshold selection already complete; proceed to evaluate.", flush=True)
            return run_dir
    amp = cfg["train"]["amp"] and device.type == "cuda"
    train_features, train_cache = cache_features(cfg, train_rows, "smoke_train" if smoke else "train", directory(cfg, "features"), device)
    validation_features, validation_cache = cache_features(cfg, validation_rows, "smoke_validation" if smoke else "validation", directory(cfg, "features"), device)
    statistics = training_statistics(train_features, train_rows)
    training_data = FeatureDataset(train_features, train_rows, statistics, cfg["train"]["augmentation"])
    validation_data = FeatureDataset(validation_features, validation_rows, statistics)
    model = KeywordCNN(cfg["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history, best_loss, wait, first_epoch = [], float("inf"), 0, 1
    effective_batch = cfg["train"]["batch_size"]
    provenance = {"dataset_manifest_sha256": audit_report["manifest_sha256"],
                  "source_manifest_sha256": audit_report["source_manifest_sha256"],
                  "sources_metadata": read_json(manifests / "sources_metadata.json"),
                  "dataset_metadata": read_json(manifests / "dataset_metadata.json"),
                  "audit_manifest_sha256": audit_report["manifest_sha256"],
                  "feature_cache_keys": {"train": train_cache["key"], "validation": validation_cache["key"]}}

    def base_checkpoint(epoch):
        return {"format_version": 1, "model_state_dict": model.state_dict(), "model_config": cfg["model"],
                "feature_config": cfg["features"], "audio_config": cfg["audio"], "normalization": statistics,
                "target_word": cfg["target_word"], "label_mapping": {"0": "target_absent", "1": "target_present"},
                "seed": cfg["seed"], "source_provenance": provenance, "epoch": epoch,
                "config": public_config(cfg), "config_sha256": config_hash, "code_sha256": code_hash(),
                "project_root_relative_to_run": os.path.relpath(cfg["_root"], run_dir), "smoke_subset": smoke}

    def save_last(epoch):
        checkpoint = base_checkpoint(epoch)
        checkpoint.update(optimizer_state_dict=optimizer.state_dict(), scaler_state_dict=scaler.state_dict(),
                          history=history, best_loss=best_loss, patience_wait=wait, effective_batch_size=effective_batch,
                          torch_rng_state=torch.get_rng_state(), python_rng_state=random.getstate(),
                          cuda_rng_states=torch.cuda.get_rng_state_all() if device.type == "cuda" else [])
        save_checkpoint(run_dir / "last.pt", checkpoint)

    def restore_last():
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if state["config_sha256"] != config_hash or state["source_provenance"]["dataset_manifest_sha256"] != provenance["dataset_manifest_sha256"] or state["smoke_subset"] != smoke:
            raise ValueError("Resume config/data/subset differs from original run. Start a separate run.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        torch.set_rng_state(state["torch_rng_state"])
        random.setstate(state["python_rng_state"])
        if device.type == "cuda" and state["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(state["cuda_rng_states"])
        return state

    if resume and (run_dir / "last.pt").exists():
        state = restore_last()
        history, best_loss, wait = state["history"], state["best_loss"], state["patience_wait"]
        first_epoch, effective_batch = state["epoch"] + 1, state["effective_batch_size"]
    else:
        # Frozen manifests live alongside weights, even when the run is stored on Drive.
        (run_dir / "manifests").mkdir(exist_ok=True)
        for name in ("dataset.csv", "background_sources.csv", "speech_sources.csv", "sources_metadata.json", "dataset_metadata.json", "download_provenance.json"):
            if (manifests / name).is_file():
                shutil.copy2(manifests / name, run_dir / "manifests" / name)
        write_json(run_dir / "dataset_audit.json", audit_report)
        write_json(run_dir / "normalization.json", statistics)
        execution = {**environment_info(), "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "amp": amp, "smoke_subset": smoke, "train_files": len(train_rows), "validation_files": len(validation_rows),
                     "started_at": timestamp(), "config": public_config(cfg), "code_sha256": code_hash()}
        write_json(run_dir / "environment.json", execution)
        (run_dir / "source_code").mkdir(exist_ok=True)
        for path in Path(__file__).parent.glob("*.py"):
            shutil.copy2(path, run_dir / "source_code" / path.name)
        save_last(0)
    max_epochs = 1 if smoke else cfg["train"]["max_epochs"]
    try:
        for epoch in range(first_epoch, max_epochs + 1):
            if wait >= cfg["train"]["patience"]:
                break
            while True:
                started = time.monotonic()
                try:
                    training_loader = loader_for(training_data, cfg, device, True, epoch, effective_batch)
                    validation_loader = loader_for(validation_data, cfg, device, batch_size=effective_batch)
                    train_loss, _, _ = epoch_pass(model, training_loader, device, criterion, optimizer, scaler, amp, 2 if smoke else None)
                    validation_loss, labels, probabilities = epoch_pass(model, validation_loader, device, criterion, amp=amp)
                    break
                except torch.cuda.OutOfMemoryError:
                    if effective_batch <= 1:
                        raise
                    effective_batch //= 2
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    restore_last()  # Roll back partial epoch, including BatchNorm and RNG.
                    write_json(run_dir / "batch_size_adjustment.json", {"effective_batch_size": effective_batch, "epoch": epoch, "reason": "CUDA OOM", "time": timestamp()})
                    print(f"CUDA OOM: restarting epoch {epoch} with batch size {effective_batch}", flush=True)
            score = binary_metrics(labels, probabilities, 0.5)
            history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                            "validation_f1_at_0_5": score["f1"], "seconds": time.monotonic() - started,
                            "effective_batch_size": effective_batch})
            if validation_loss < best_loss - cfg["train"]["min_delta"]:
                best_loss, wait = validation_loss, 0
                save_checkpoint(run_dir / "best.pt", base_checkpoint(epoch))
            else:
                wait += 1
            save_last(epoch)
            write_csv(run_dir / "history.csv", history)
            write_json(run_dir / "history.json", history)
            write_json(run_dir / "status.json", {"training_complete": False, "last_epoch": epoch, "best_validation_loss": best_loss,
                                                "smoke_subset": smoke, "time": timestamp()})
            print(f"Epoch {epoch}: train={train_loss:.5f}, val={validation_loss:.5f}, F1@0.5={score['f1']:.4f}, batch={effective_batch}", flush=True)
        selected = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(selected["model_state_dict"])
        _, labels, probabilities = epoch_pass(model, loader_for(validation_data, cfg, device, batch_size=effective_batch), device, criterion, amp=False)
        threshold, sweep = select_threshold(labels, probabilities)
        write_json(run_dir / "threshold.json", {"threshold": threshold, "selection_split": "validation", "selection_metric": "f1",
                                                "tie_break": "lower FPR, then higher threshold", "grid": [0.05, 0.95, 0.01],
                                                "target_word": cfg["target_word"], "checkpoint_sha256": sha256_file(run_dir / "best.pt"),
                                                "dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
                                                "selected_epoch": selected["epoch"], "smoke_subset": smoke})
        write_json(run_dir / "threshold_sweep.json", sweep)
        write_csv(run_dir / "validation_predictions.csv", [{**row, "probability": probability, "prediction": int(probability >= threshold)} for row, probability in zip(validation_rows, probabilities)])
        plot_history(history, run_dir / "loss_curve.png")
        write_json(run_dir / "status.json", {"training_complete": True, "last_epoch": history[-1]["epoch"], "selected_epoch": selected["epoch"],
                                            "best_validation_loss": best_loss, "threshold": threshold, "smoke_subset": smoke, "time": timestamp()})
        print(f"Training complete. Validation threshold={threshold:.2f}; artifacts: {run_dir}", flush=True)
    except Exception as error:
        write_json(run_dir / "failure.json", {"time": timestamp(), "error": f"{type(error).__name__}: {error}", "resume_command": f"python -m src.train --config configs/default.yaml --run-dir \"{run_dir}\" --resume"})
        raise
    return run_dir


def main():
    parser = config_parser("Train the audited log-mel CNN baseline; no test-set tuning.")
    parser.add_argument("--run-dir", help="Run artifact directory (may be on mounted Drive).")
    parser.add_argument("--resume", action="store_true", help="Restore model, Adam, scaler and RNG from last.pt.")
    parser.add_argument("--smoke", action="store_true", help="Clearly marked subset; two training batches, one epoch, separate run.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.device:
        cfg["train"]["device"] = args.device
    train(cfg, args.run_dir, args.resume, args.smoke)


if __name__ == "__main__":
    main()

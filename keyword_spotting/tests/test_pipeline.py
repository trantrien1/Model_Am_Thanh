from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audit_dataset import audit
from src.build_dataset import build
from src.common import read_csv, read_json, write_json
from src.evaluate import evaluate
from src.predict import predict
from src.prepare_sources import prepare
from src.train import train
import importlib


def test_end_to_end_mixing_audit_train_reload_predict(toy_project):
    cfg, config_path = toy_project
    root = Path(cfg["_root"])
    prepare(cfg, workers=2)
    rows = build(cfg)
    assert len(rows) == 48
    first_hashes = [row["audio_sha256"] for row in rows]
    assert [row["audio_sha256"] for row in build(cfg)] == first_hashes
    audit_report = audit(cfg)
    assert audit_report["passed"]
    for split in ("train", "validation", "test"):
        group = [row for row in rows if row["split"] == split]
        assert sum(row["label"] == 1 for row in group) == 8
        assert sum(row["variant_type"] == "N2_clean" for row in group) == 1
    run_dir = train(cfg, smoke=True)
    state = torch.load(run_dir / "best.pt", weights_only=True, map_location="cpu")
    assert state["smoke_subset"] is True
    assert state["normalization"]["files"] == 16
    assert state["target_word"] == "yes"
    with pytest.raises(ValueError, match="smoke subset"):
        evaluate(run_dir, config_path=config_path)
    report = evaluate(run_dir, config_path=config_path, allow_smoke=True, device="cpu")
    assert report["file_count"] == 16
    predictions = read_csv(run_dir / "evaluation/test/predictions.csv")
    selected = predictions[0]
    result = predict(run_dir, root / selected["path"], device="cpu")
    assert abs(result["probability"] - float(selected["probability"])) < 2e-5
    assert result["prediction"] == int(selected["prediction"])
    assert result["duration_seconds"] == 5
    short = root / "short.wav"
    sf.write(short, np.zeros(16000), 16000)
    assert predict(run_dir, short, "cpu")["duration_seconds"] == 1
    long_audio = root / "long.wav"
    sf.write(long_audio, np.zeros(80001), 16000)
    with pytest.raises(ValueError, match="exceeds 5 seconds"):
        predict(run_dir, long_audio, "cpu")
    assert train(cfg, resume=True, smoke=True) == run_dir
    supplemental = build(cfg, supplemental=True)
    assert len(supplemental) == 11  # Three remaining clean backgrounds plus 8 quieter mixtures.
    assert all(row["split"] == "test" for row in supplemental)
    assert audit(cfg, supplemental=True)["passed"]
    extra_report = evaluate(run_dir, config_path=config_path, supplemental=True, allow_smoke=True, device="cpu")
    assert extra_report["supplemental"] is True
    threshold_path = run_dir / "threshold.json"
    threshold = read_json(threshold_path)
    threshold["checkpoint_sha256"] = "invalid"
    write_json(threshold_path, threshold)
    with pytest.raises(ValueError, match="different checkpoint"):
        predict(run_dir, short, "cpu")


def test_audit_detects_tampered_wav_and_blocks_training(toy_project):
    cfg, _ = toy_project
    prepare(cfg, workers=2)
    rows = build(cfg)
    path = Path(cfg["_root"]) / rows[0]["path"]
    sf.write(path, np.zeros(80000), 16000)
    with pytest.raises(ValueError, match="audit FAILED"):
        train(cfg)
    assert not (Path(cfg["_root"]) / "runs/baseline/best.pt").exists()


def test_resume_interrupted_epoch_restores_adam_and_history(toy_project, monkeypatch):
    cfg, _ = toy_project
    prepare(cfg, workers=2)
    build(cfg)
    training_module = importlib.import_module("src.train")
    original = training_module.epoch_pass
    calls = 0

    def interrupt_epoch_two(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Simulated runtime interruption in epoch two")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_module, "epoch_pass", interrupt_epoch_two)
    with pytest.raises(RuntimeError, match="Simulated runtime"):
        train(cfg)
    run_dir = Path(cfg["_root"]) / "runs/baseline"
    state = torch.load(run_dir / "last.pt", weights_only=True, map_location="cpu")
    assert state["epoch"] == 1
    assert state["optimizer_state_dict"]["state"]
    monkeypatch.setattr(training_module, "epoch_pass", original)
    train(cfg, resume=True)
    history = read_json(run_dir / "history.json")
    assert [row["epoch"] for row in history] == [1, 2]
    assert read_json(run_dir / "status.json")["training_complete"] is True

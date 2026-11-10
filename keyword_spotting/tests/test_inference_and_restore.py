import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from inference import KeywordSpotter, main
from scripts.restore_dataset import restore_mixtures
from src.artifacts import save_checkpoint
from src.build_dataset import build
from src.common import read_json, sha256_file, write_json
from src.model import KeywordCNN
from src.predict import predict
from src.prepare_sources import prepare


def test_readable_inference_uses_same_artifacts_and_keeps_batch_errors(toy_project, capsys):
    cfg, _ = toy_project
    root = Path(cfg["_root"])
    run = root / "runs/unit_fixture"
    model = KeywordCNN(cfg["model"])
    for parameter in model.parameters():
        parameter.data.zero_()
    model.classifier[-1].bias.data.fill_(1)
    checkpoint = {"model_state_dict": model.state_dict(), "model_config": cfg["model"], "audio_config": cfg["audio"],
                  "feature_config": cfg["features"], "normalization": {"mean": [0.] * 64, "std": [1.] * 64},
                  "target_word": "yes", "epoch": 0, "smoke_subset": True,
                  "config": {key: value for key, value in cfg.items() if not key.startswith("_")},
                  "project_root_relative_to_run": "../.."}
    save_checkpoint(run / "best.pt", checkpoint)
    write_json(run / "threshold.json", {"target_word": "yes", "threshold": 0.5, "checkpoint_sha256": sha256_file(run / "best.pt")})
    folder = root / "inputs"
    folder.mkdir()
    sf.write(folder / "short.wav", np.zeros(16000), 16000)
    sf.write(folder / "long.wav", np.zeros(80001), 16000)
    spotter = KeywordSpotter(run)
    result = spotter.predict(folder / "short.wav")
    assert result["prediction"] == 1 and result["decision"] == "target_present"
    assert abs(result["probability"] - predict(run, folder / "short.wav", "cpu")["probability"]) < 1e-6
    assert spotter.describe()["weights"] == str(run / "best.pt")
    csv = root / "predictions.csv"
    assert main(["--run-dir", str(run), "--folder", str(folder), "--csv", str(csv), "--json"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert len(output) == 2
    assert sum(row["decision"] == "error" for row in output) == 1
    assert "exceeds 5 seconds" in next(row["error"] for row in output if row["error"])
    assert csv.exists()


def test_restoration_recovers_exact_hashes_without_overwriting_existing_data(toy_project):
    cfg, _ = toy_project
    root = Path(cfg["_root"])
    prepare(cfg, workers=2)
    rows = build(cfg)
    extras = build(cfg, supplemental=True)
    for row in rows[:3] + extras[:2]:
        (root / row["path"]).unlink()  # Only the disposable fixture files created by this test.
    report = restore_mixtures(cfg)
    assert report["passed"] and report["newly_restored"] == 5
    assert report["selected_files"] == len(rows) + len(extras)
    assert report["training_executed"] is False
    assert all(sha256_file(root / row["path"]) == row["audio_sha256"] for row in rows + extras)
    tampered = root / rows[0]["path"]
    sf.write(tampered, np.zeros(80000), 16000)
    bad_hash = sha256_file(tampered)
    with pytest.raises(ValueError, match="preserving it"):
        restore_mixtures(cfg)
    assert sha256_file(tampered) == bad_hash

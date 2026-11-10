"""Check the dataset CLI on artificial audio, without touching trained data."""
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src.common import read_csv, sha256_file

PROJECT = Path(__file__).resolve().parents[1]


def run_cli(config_path):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run([sys.executable, "-B", str(PROJECT / "data/make_dataset.py"),
                           "--config", str(config_path)], cwd=config_path.parent,
                          env=env, capture_output=True, text=True, timeout=180)


def test_cli_builds_splits_restores_missing_wav_and_rejects_changed_config(toy_project):
    cfg, config_path = toy_project
    root = Path(cfg["_root"])
    first = run_cli(config_path)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "build like Colab" in first.stdout
    manifests = root / "data/manifests"
    rows = read_csv(manifests / "dataset.csv")
    assert len(rows) == 48
    assert len(read_csv(manifests / "supplemental.csv")) == 11
    for split in ("train", "validation", "test"):
        split_rows = read_csv(manifests / f"{split}.csv")
        assert len(split_rows) == 16
        assert {row["split"] for row in split_rows} == {split}
    manifest_hash = sha256_file(manifests / "dataset.csv")
    missing = root / rows[0]["path"]
    missing.unlink()
    second = run_cli(config_path)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "reuse frozen manifests" in second.stdout
    assert sha256_file(missing) == rows[0]["audio_sha256"]
    assert sha256_file(manifests / "dataset.csv") == manifest_hash
    assert not list(config_path.parent.glob("make_dataset_*.yaml"))
    changed = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    changed["seed"] += 1
    config_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    third = run_cli(config_path)
    assert third.returncode == 1
    assert "Use --output-dir" in third.stderr
    assert sha256_file(manifests / "dataset.csv") == manifest_hash
    assert not (root / "runs/baseline/best.pt").exists()

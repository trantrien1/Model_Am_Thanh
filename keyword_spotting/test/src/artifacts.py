import os
from pathlib import Path

import torch

from .common import load_config, read_json, sha256_file
from .model import KeywordCNN


def select_device(requested="auto"):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Enable a Colab GPU or choose --device cpu.")
    return torch.device(requested)


def save_checkpoint(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_inference(run_dir, device="auto", config_path=None):
    run_dir = Path(run_dir).resolve()
    device = select_device(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    threshold = read_json(run_dir / "threshold.json")
    digest = sha256_file(run_dir / "best.pt")
    if threshold["checkpoint_sha256"] != digest:
        raise ValueError("Threshold was selected for a different checkpoint; rerun validation threshold selection.")
    if threshold["target_word"] != checkpoint["target_word"]:
        raise ValueError("Threshold/checkpoint target words differ.")
    cfg = checkpoint["config"]
    if config_path:
        # Only use the caller's config to locate data. Model preprocessing stays frozen in checkpoint.
        cfg["_root"] = load_config(config_path)["_root"]
    else:
        cfg["_root"] = str(Path(__file__).resolve().parents[1])
    cfg["features"] = checkpoint["feature_config"]
    cfg["audio"] = checkpoint["audio_config"]
    model = KeywordCNN(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint, threshold, cfg, device

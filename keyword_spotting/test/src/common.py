import argparse
import csv
import hashlib
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

SPLITS = ("train", "validation", "test")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def code_hash():
    return object_hash({p.name: sha256_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    # YAML 1.1 interprets unquoted "yes" as a boolean; fail explicitly instead of training "True".
    if not isinstance(cfg.get("target_word"), str):
        raise ValueError('target_word must be a string; quote "yes" in YAML.')
    if not cfg["target_word"].isalnum():
        raise ValueError("target_word must be a single alphanumeric directory name.")
    cfg["_root"] = str(path.parent.parent)
    cfg["_config_path"] = str(path)
    if cfg["audio"]["sample_rate"] != cfg["features"]["sample_rate"]:
        raise ValueError("Audio and feature sample rates differ.")
    if cfg["features"]["fmax"] > cfg["audio"]["sample_rate"] / 2:
        raise ValueError("fmax exceeds the Nyquist frequency.")
    if not 0 <= cfg["mix"]["clean_background_fraction"] <= 1:
        raise ValueError("clean_background_fraction must be in [0, 1].")
    if cfg["mix"]["snr_min_db"] > cfg["mix"]["snr_max_db"]:
        raise ValueError("Invalid SNR range.")
    return cfg


def public_config(cfg):
    return {key: value for key, value in cfg.items() if not key.startswith("_")}


def project_path(cfg, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(cfg["_root"]) / path).resolve()


def directory(cfg, key):
    return project_path(cfg, cfg["paths"][key])


def relative_path(cfg, path):
    try:
        return Path(os.path.relpath(Path(path).resolve(), cfg["_root"])).as_posix()
    except ValueError:  # Windows paths on different drives.
        return Path(path).resolve().as_posix()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def rng_for(seed, sample_id):
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def environment_info():
    from importlib.metadata import PackageNotFoundError, version
    packages = {}
    for package in ("torch", "numpy", "scipy", "soundfile", "librosa", "pandas", "scikit-learn", "PyYAML", "matplotlib"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def config_parser(description, sources=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    if sources:
        parser.add_argument("--esc-dir", help="Existing ESC directory containing audio/ and meta/.")
        parser.add_argument("--speech-dir", help="Existing extracted Speech Commands v0.02 directory.")
    return parser


def apply_source_overrides(cfg, args):
    for key in ("esc_dir", "speech_dir"):
        value = getattr(args, key, None)
        if value:
            cfg["paths"][key] = relative_path(cfg, Path(value).resolve())
    return cfg

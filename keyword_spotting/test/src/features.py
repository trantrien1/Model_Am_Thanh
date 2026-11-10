import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import nn

from .audio import fixed_length, load_audio
from .common import object_hash, project_path, read_json, write_json


class LogMel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        if config["power"] != 2 or config["mel_scale"] != "slaney":
            raise ValueError("Baseline implements power=2 and Slaney mel scale only.")
        self.register_buffer("window", torch.hann_window(config["win_length"], periodic=True))
        filters = librosa.filters.mel(sr=config["sample_rate"], n_fft=config["n_fft"], n_mels=config["n_mels"],
                                     fmin=config["fmin"], fmax=config["fmax"], htk=False, norm=config["mel_norm"])
        self.register_buffer("mel_filters", torch.from_numpy(filters))

    def forward(self, audio):
        config = self.config
        spectrum = torch.stft(audio, n_fft=config["n_fft"], hop_length=config["hop_length"],
                              win_length=config["win_length"], window=self.window, center=config["center"],
                              return_complex=True)
        power = spectrum.abs().square()
        return torch.log(torch.matmul(self.mel_filters, power).clamp_min(config["epsilon"]))


def preprocess_audio(path, audio_config, reject_long=True):
    audio, metadata = load_audio(path, audio_config["sample_rate"], allow_silence=True)
    # Use original duration as well, so resampling rounding cannot hide an overlong input.
    if reject_long and metadata["duration_seconds"] > audio_config["duration_seconds"]:
        raise ValueError("Audio exceeds 5 seconds; provide a file up to 5 seconds for this baseline.")
    audio, _ = fixed_length(audio, round(audio_config["sample_rate"] * audio_config["duration_seconds"]), reject_long)
    return audio, metadata


def cache_features(cfg, rows, split, cache_dir, device="cpu", batch_size=32):
    # Includes preprocessing and byte hashes from the audited manifest, never just filenames.
    key = object_hash({"implementation": "torch-stft-slaney-log-v1", "features": cfg["features"], "audio": cfg["audio"],
                       "sources": [(row["sample_id"], row["audio_sha256"]) for row in rows]})
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output, metadata_path = cache_dir / f"{split}-{key}.npy", cache_dir / f"{split}-{key}.json"
    if output.exists() and metadata_path.exists():
        metadata = read_json(metadata_path)
        cached = np.load(output, mmap_mode="r")
        if metadata["key"] == key and list(cached.shape) == metadata["shape"] and cached.shape[0] == len(rows):
            print(f"Feature cache reused: {split}, {len(rows)} files", flush=True)
            return cached, metadata
    extractor = LogMel(cfg["features"]).to(device).eval()
    total_samples = round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_seconds"])
    with torch.no_grad():
        shape = tuple(extractor(torch.zeros(1, total_samples, device=device)).shape[1:])
    temporary = output.with_suffix(".part.npy")
    array = np.lib.format.open_memmap(temporary, mode="w+", dtype="float32", shape=(len(rows), *shape))
    for start in range(0, len(rows), batch_size):
        waveforms = [preprocess_audio(project_path(cfg, row["path"]), cfg["audio"])[0] for row in rows[start:start + batch_size]]
        with torch.no_grad():
            batch = extractor(torch.from_numpy(np.stack(waveforms)).to(device)).cpu().numpy()
        if not np.isfinite(batch).all():
            raise ValueError("Non-finite log-mel features.")
        array[start:start + len(batch)] = batch
        if (start // batch_size) % 25 == 0:
            print(f"Features {split}: {min(start + len(batch), len(rows))}/{len(rows)}", flush=True)
    array.flush()
    del array
    temporary.replace(output)
    metadata = {"key": key, "shape": [len(rows), *shape], "feature_config": cfg["features"], "dtype": "float32"}
    write_json(metadata_path, metadata)
    return np.load(output, mmap_mode="r"), metadata


def training_statistics(features, rows):
    if len(features) != len(rows) or not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Normalization must use exclusively the complete training rows.")
    count = 0
    total = np.zeros(features.shape[1], dtype=np.float64)
    squares = np.zeros_like(total)
    for start in range(0, len(features), 64):
        values = np.asarray(features[start:start + 64], dtype=np.float64)
        total += values.sum(axis=(0, 2))
        squares += np.square(values).sum(axis=(0, 2))
        count += values.shape[0] * values.shape[2]
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 1e-8))
    return {"mean": mean.tolist(), "std": std.tolist(), "split": "train", "files": len(rows), "frames_per_bin": count}

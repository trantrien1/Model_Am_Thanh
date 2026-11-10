from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from src.common import load_config, write_csv


@pytest.fixture
def toy_project(tmp_path):
    """Artificial tones for software verification ONLY, never experimental datasets/metrics."""
    root = tmp_path / "fixture_project"
    (root / "configs").mkdir(parents=True)
    default = Path(__file__).resolve().parents[1] / "configs/default.yaml"
    config = yaml.safe_load(default.read_text(encoding="utf-8"))
    config["train"].update(device="cpu", cpu_threads=2, batch_size=4, max_epochs=2)
    config_path = root / "configs/default.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    esc = root / "data/raw/esc"
    speech = root / "data/raw/speech_commands"
    (esc / "audio").mkdir(parents=True)
    speech.mkdir(parents=True)
    rng = np.random.default_rng(12345)
    backgrounds, validation, test = [], [], []
    for split, fold in (("train", 1), ("validation", 4), ("test", 5)):
        for index in range(4):
            # Names do not encode the fold, testing that metadata defines it.
            name = f"fixture_{split}_{index}.wav"
            wave = rng.normal(0, 0.025, 220500).astype(np.float32)
            sf.write(esc / "audio" / name, np.column_stack((wave, wave * 0.8)), 44100)
            backgrounds.append({"filename": name, "fold": fold, "category": "laughing" if index == 0 else "rain",
                                "src_file": f"fixture_original_{split}_{index // 2}"})
        for word_index, word in enumerate(("yes", "no", "up", "stop")):
            (speech / word).mkdir(exist_ok=True)
            for index in range(4):
                speaker = f"fixture_{split}_speaker{index % 2}"
                name = f"{speaker}_nohash_{index}.wav"
                samples = np.zeros(16000, dtype=np.float32)
                time = np.arange(12000) / 16000
                wave = np.sin(2 * np.pi * (250 + word_index * 250 + index * 13) * time)
                samples[2000:14000] = 0.12 * wave * np.hanning(len(wave)) + rng.normal(0, 0.0001, len(wave))
                sf.write(speech / word / name, samples, 16000)
                if split == "validation":
                    validation.append(f"{word}/{name}")
                elif split == "test":
                    test.append(f"{word}/{name}")
    (speech / "_background_noise_").mkdir()
    sf.write(speech / "_background_noise_/ignored.wav", rng.normal(0, 0.01, 80000), 16000)
    write_csv(esc / "meta/esc50.csv", backgrounds)
    (speech / "validation_list.txt").write_text("\n".join(validation) + "\n", encoding="utf-8")
    (speech / "testing_list.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    return load_config(config_path), config_path

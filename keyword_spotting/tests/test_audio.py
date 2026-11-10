import numpy as np
import pytest
import soundfile as sf

from src.audio import fixed_length, load_audio, mix_audio, pcm_hash, rms, trim_speech
from src.common import rng_for


def test_resample_mono_duration_and_hash(tmp_path):
    time = np.arange(44100) / 44100
    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32) * 0.1
    path = tmp_path / "stereo.wav"
    sf.write(path, np.column_stack((audio, audio)), 44100)
    converted, meta = load_audio(path)
    assert converted.shape == (16000,)
    assert meta["original_channels"] == 2
    assert abs(rms(converted) - rms(audio)) < 0.001
    assert pcm_hash(converted) == pcm_hash(converted.copy())
    changed = converted.copy()
    changed[100] += 0.01
    assert pcm_hash(changed) != pcm_hash(converted)


def test_trim_preserves_entire_word_and_padding():
    speech = np.zeros(16000, dtype=np.float32)
    speech[3000:14000] = 0.1
    trimmed, start, end, mask = trim_speech(speech)
    assert start <= 3000 - 320
    assert end >= 14000 + 320
    assert len(trimmed) > 8000  # No forced 0.5-second crop.
    assert np.count_nonzero(trimmed) == np.count_nonzero(speech)
    assert len(mask) == len(trimmed)


def test_mixing_snr_full_length_and_no_clipping():
    background = np.full(80000, 0.25, dtype=np.float32)
    speech = np.full(12000, 0.2, dtype=np.float32)
    mixed, metadata = mix_audio(background, speech, 68000, 12)
    assert len(mixed) == 80000
    assert abs(metadata["snr_db_measured"] - 12) < 1e-5
    assert np.max(np.abs(mixed)) <= 0.980001
    assert metadata["output_gain"] < 1
    with pytest.raises(ValueError, match="no speech cropping"):
        mix_audio(background, speech, 68001, 12)


def test_fallback_and_clean_share_output_rules():
    speech = np.full(10000, 0.1, dtype=np.float32)
    mixed, metadata = mix_audio(np.zeros(80000, dtype=np.float32), speech, 0, 10)
    assert metadata["mix_level_mode"] == "fallback_dbfs"
    assert metadata["snr_db_measured"] == ""
    assert abs(20 * np.log10(rms(mixed[:10000])) + 25) < 0.001
    clean, clean_meta = mix_audio(np.full(80000, 1.5, dtype=np.float32), None, "", "")
    assert clean_meta["mix_level_mode"] == "clean"
    assert np.max(np.abs(clean)) <= 0.980001


def test_fixed_length_and_seed_are_repeatable():
    assert np.array_equal(rng_for(42, "sample").random(20), rng_for(42, "sample").random(20))
    assert not np.array_equal(rng_for(42, "sample").random(20), rng_for(42, "other").random(20))
    assert fixed_length(np.ones(16000), 80000)[0].shape == (80000,)
    with pytest.raises(ValueError, match="exceeds 5 seconds"):
        fixed_length(np.zeros(80001), 80000, reject_long=True)

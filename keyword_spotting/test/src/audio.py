import hashlib
import math

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def rms(audio):
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))) if len(audio) else 0.0


def load_audio(path, sample_rate=16000, allow_silence=False):
    audio, original_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f"Empty or non-finite audio: {path}")
    metadata = {"original_sample_rate": int(original_rate), "original_channels": int(audio.shape[1]),
                "original_samples": len(audio), "duration_seconds": len(audio) / original_rate}
    audio = audio.mean(axis=1, dtype=np.float32)
    if original_rate != sample_rate:
        factor = math.gcd(original_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // factor, original_rate // factor).astype(np.float32)
    if not np.isfinite(audio).all() or (not allow_silence and rms(audio) < 1e-8):
        raise ValueError(f"Silent or invalid mono audio: {path}")
    return np.ascontiguousarray(audio), metadata


def pcm_hash(audio, sample_rate=16000):
    # Untrimmed canonical mono resampled PCM16, explicit little-endian and length header.
    pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype("<i2")
    header = f"mono-pcm16le-v1:{sample_rate}:{len(pcm)}:".encode()
    return hashlib.sha256(header + pcm.tobytes()).hexdigest()


def trim_speech(audio, sample_rate=16000, trim_db=45, padding_ms=40):
    if len(audio) == 0 or rms(audio) < 1e-8:
        raise ValueError("Cannot trim empty or silent speech.")
    frame_length = max(1, round(sample_rate * 0.01))
    energies = np.array([rms(audio[i:i + frame_length]) for i in range(0, len(audio), frame_length)])
    active = np.flatnonzero(energies >= max(1e-8, energies.max() * 10 ** (-trim_db / 20)))
    padding = round(sample_rate * padding_ms / 1000)
    start = max(0, int(active[0]) * frame_length - padding)
    end = min(len(audio), (int(active[-1]) + 1) * frame_length + padding)
    # Active mask is used consistently for measuring speech and background RMS in mixing.
    mask = np.repeat(energies >= max(1e-8, energies.max() * 10 ** (-trim_db / 20)), frame_length)[:len(audio)]
    return audio[start:end].copy(), start, end, mask[start:end]


def fixed_length(audio, samples, reject_long=False):
    if len(audio) > samples:
        if reject_long:
            raise ValueError("Audio exceeds 5 seconds; baseline accepts files up to 5 seconds. Sliding windows are not enabled.")
        return audio[:samples].copy(), "crop"
    if len(audio) < samples:
        return np.pad(audio, (0, samples - len(audio))), "pad"
    return audio.copy(), "none"


def mix_audio(background, speech, start_sample, snr_db, active_mask=None,
              silence_rms=1e-5, fallback_dbfs=-25, peak_limit=0.98):
    mixed = background.copy()
    details = {"mix_level_mode": "clean", "speech_gain": "", "snr_db_measured": "",
               "background_rms": "", "speech_rms_output": "", "output_gain": 1.0}
    if speech is not None:
        end = start_sample + len(speech)
        if start_sample < 0 or end > len(background):
            raise ValueError("Speech insertion is outside the background; no speech cropping is allowed.")
        mask = np.ones(len(speech), dtype=bool) if active_mask is None else active_mask
        if len(mask) != len(speech) or not mask.any():
            raise ValueError("Invalid active speech mask.")
        speech_rms = rms(speech[mask])
        if speech_rms < 1e-8:
            raise ValueError("Speech has no useful energy.")
        background_rms = rms(background[start_sample:end][mask])
        if background_rms < silence_rms:
            alpha = 10 ** (fallback_dbfs / 20) / speech_rms
            mode = "fallback_dbfs"
            measured = ""  # Requested SNR is not meaningful on a near-silent region.
        else:
            alpha = background_rms * 10 ** (snr_db / 20) / speech_rms
            mode = "snr"
            measured = 20 * math.log10(rms(alpha * speech[mask]) / background_rms)
        mixed[start_sample:end] += alpha * speech
        details.update(mix_level_mode=mode, speech_gain=float(alpha), snr_db_measured=measured,
                       background_rms=background_rms, speech_rms_output=float(alpha * speech_rms))
    peak = float(np.max(np.abs(mixed)))
    gain = min(1.0, peak_limit / peak) if peak else 1.0
    mixed *= gain
    details["output_gain"] = gain
    if details["speech_rms_output"] != "":
        details["speech_rms_output"] *= gain
        details["background_rms"] *= gain
    if not np.isfinite(mixed).all():
        raise ValueError("Non-finite mixture.")
    return mixed, details

"""Standalone CPU schwa /ə/ inference. Model and threshold are embedded in best.pt."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import nn
from scipy.ndimage import uniform_filter
from scipy.signal import istft, resample_poly, stft

ROOT = Path(__file__).resolve().parent
DENOISE = {'strength': .6, 'minimum_gain': .35, 'quiet_fraction': .2, 'maximum_quiet_to_loud_ratio': .25}
SPECTRAL = {'sample_rate': 16000, 'n_fft': 400, 'hop_length': 320, 'n_mels': 40, 'fmin': 50., 'fmax': 8000.}



def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_audio(path, sample_rate=16000):
    (values, rate) = sf.read(path, dtype='float32', always_2d=True)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f'Empty or invalid audio: {path}')
    metadata = {'duration_seconds': len(values) / rate, 'original_sample_rate': rate, 'original_channels': values.shape[1]}
    mono = values.mean(axis=1, dtype=np.float32)
    if rate != sample_rate:
        factor = math.gcd(rate, sample_rate)
        mono = resample_poly(mono, sample_rate // factor, rate // factor).astype(np.float32)
    if not np.isfinite(mono).all():
        raise ValueError(f'Invalid resampled audio: {path}')
    return (np.ascontiguousarray(mono), metadata)


def crop_window(audio, start, samples=8000):
    output = np.zeros(samples, dtype=np.float32)
    (left, right) = (max(0, start), min(len(audio), start + samples))
    if right > left:
        output[left - start:right - start] = audio[left:right]
    return output


def denoise_audio(audio, config=None):
    config = config or DENOISE
    if len(audio) < 512 or not np.any(audio):
        return (audio.copy(), 'skipped_short_or_silent')
    (_, _, spectrum) = stft(audio, fs=16000, nperseg=400, noverlap=240, nfft=512, boundary='zeros', padded=True)
    magnitude = np.abs(spectrum)
    energy = np.sqrt(np.mean(magnitude ** 2, axis=0))
    count = max(3, round(len(energy) * config['quiet_fraction']))
    if len(energy) < count:
        return (audio.copy(), 'skipped_insufficient_frames')
    quiet = np.argsort(energy)[:count]
    loud = float(np.percentile(energy, 90))
    if loud <= 1e-10 or float(np.median(energy[quiet])) > loud * config['maximum_quiet_to_loud_ratio']:
        return (audio.copy(), 'skipped_no_reliable_background')
    noise = np.median(magnitude[:, quiet], axis=1, keepdims=True)
    gain = 1 - config['strength'] * noise / np.maximum(magnitude, 1e-10)
    gain = uniform_filter(np.clip(gain, config['minimum_gain'], 1.0), size=(3, 3), mode='nearest')
    (_, filtered) = istft(spectrum * gain, fs=16000, nperseg=400, noverlap=240, nfft=512, input_onesided=True, boundary=True)
    output = np.asarray(filtered[:len(audio)], dtype=np.float32)
    if len(output) < len(audio):
        output = np.pad(output, (0, len(audio) - len(output)))
    if not np.isfinite(output).all():
        raise ValueError('Denoising produced non-finite audio.')
    return (output, 'filtered')


def denoise_cli(args):
    (audio, _) = load_audio(args.audio)
    (filtered, status) = denoise_audio(audio)
    if args.output.resolve() == args.audio.resolve():
        raise ValueError('Keep the original audio; choose a different output path.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, np.clip(filtered, -1, 1), 16000, subtype='PCM_16')
    print(f'Denoise status: {status}; saved {args.output}')


class AlignedMel(nn.Module):
    """24 log-mel frames with centers matching the acoustic encoder's 400/320 receptive field."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or SPECTRAL
        c = self.config
        (low, high) = (2595 * np.log10(1 + c['fmin'] / 700), 2595 * np.log10(1 + c['fmax'] / 700))
        points = 700 * (10 ** (np.linspace(low, high, c['n_mels'] + 2) / 2595) - 1)
        frequencies = np.arange(c['n_fft'] // 2 + 1) * c['sample_rate'] / c['n_fft']
        rise = (frequencies[None] - points[:-2, None]) / (points[1:-1] - points[:-2])[:, None]
        fall = (points[2:, None] - frequencies[None]) / (points[2:] - points[1:-1])[:, None]
        filters = np.maximum(0, np.minimum(rise, fall)) * (2 / (points[2:] - points[:-2]))[:, None]
        self.register_buffer('filters', torch.from_numpy(filters.astype(np.float32)))
        self.register_buffer('window', torch.hann_window(c['n_fft']))

    def forward(self, audio):
        c = self.config
        spectrum = torch.stft(audio, n_fft=c['n_fft'], hop_length=c['hop_length'], window=self.window, center=False, return_complex=True)
        return torch.log(torch.matmul(self.filters, spectrum.abs().square()).clamp_min(1e-10)).transpose(1, 2)


class PhoneHead(nn.Module):

    def __init__(self, classes, hidden=128, dropout=0.2, temporal='none', input_dim=768):
        super().__init__()
        self.temporal = temporal
        if temporal != 'none':
            if hidden < 2 or hidden % 2:
                raise ValueError('Temporal head requires a positive even hidden size.')
            self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU())
            if temporal == 'conv':
                self.context = nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(), nn.Dropout(dropout), nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU())
            elif temporal == 'gru':
                self.context = nn.GRU(hidden, hidden // 2, batch_first=True, bidirectional=True)
            else:
                raise ValueError('Unknown temporal head.')
            self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, classes))
        elif hidden:
            self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, classes))
        else:
            self.network = nn.Sequential(nn.LayerNorm(input_dim), nn.Dropout(dropout), nn.Linear(input_dim, classes))

    def forward(self, values):
        if self.temporal != 'none':
            projected = self.projection(values)
            if self.temporal == 'conv':
                context = self.context(projected.transpose(1, 2)).transpose(1, 2)
            else:
                (context, _) = self.context(projected)
            return self.classifier(projected + context)
        return self.network(values)


class TransferDetector:

    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        self.checkpoint = torch.load(self.path, map_location='cpu', weights_only=True)
        state = self.checkpoint
        if state.get('backend') != 'wav2vec2_phone_head' or state.get('target_phones') != ['ax']:
            raise ValueError('Not a schwa acoustic transfer checkpoint.')
        torch.set_num_threads(state['cpu_threads'])
        if 'backbone' in state:
            acoustic = state['backbone']
        else:
            raise ValueError('Checkpoint is not self-contained; use the exported best.pt.')
        self.acoustic = torchaudio.models.wav2vec2_model(**acoustic['model_config']).eval()
        self.acoustic.load_state_dict(acoustic['model_state_dict'])
        self.layers = acoustic['layers']
        self.head = PhoneHead(**state['head_config']).eval()
        self.head.load_state_dict(state['head_state_dict'])
        self.ax = state['phone_vocabulary'].index('ax')
        self.spectral = None
        if state.get('spectral'):
            metadata = state['spectral']
            self.spectral = AlignedMel(metadata['config']).eval()
            self.spectral.load_state_dict(metadata['state_dict'])
            self.spectral_mean = torch.tensor(metadata['mean'], dtype=torch.float32)
            self.spectral_std = torch.tensor(metadata['std'], dtype=torch.float32)

    def describe(self):
        state = self.checkpoint
        return {'weights': str(self.path), 'sha256': digest(self.path), 'backend': state['backend'], 'target_symbol': state['target_symbol'], 'target_phones': state['target_phones'], 'epoch': state['epoch'], 'threshold': state['threshold'], 'device': 'cpu', 'denoise': 'off', 'window_seconds': 0.5, 'pool_frames': state['pool_frames'], 'backbone_layers': self.layers, 'self_contained': 'backbone' in state, 'source_metadata': state.get('source_metadata', {}), 'spectral_features': self.spectral is not None, 'parameters': sum((p.numel() for p in self.acoustic.parameters())) + sum((p.numel() for p in self.head.parameters()))}

    def predict(self, path, hop_seconds=0.05):
        if not 0 < hop_seconds <= 0.5:
            raise ValueError('Hop must be in (0, 0.5] seconds.')
        (waveform, metadata) = load_audio(path)
        hop = max(1, round(hop_seconds * 16000))
        starts = list(range(0, max(1, len(waveform) - 8000 + 1), hop))
        last = max(0, len(waveform) - 8000)
        if starts[-1] != last:
            starts.append(last)
        (selected, maximum) = ([], 0.0)
        started = time.perf_counter()
        with torch.inference_mode():
            for index in range(0, len(starts), 8):
                subset = starts[index:index + 8]
                batch = np.stack([crop_window(waveform, start) for start in subset])
                (features, _) = self.acoustic.extract_features(torch.from_numpy(batch), num_layers=self.layers)
                values = features[-1].to(torch.float16).to(torch.float32)
                if self.spectral is not None:
                    spectral = self.spectral(torch.from_numpy(batch)).to(torch.float16).to(torch.float32)
                    values = torch.cat([values, (spectral - self.spectral_mean) / self.spectral_std], dim=-1)
                scores = torch.softmax(self.head(values), dim=-1)[:, :, self.ax]
                if self.checkpoint['pool_frames'] > 1:
                    scores = torch.nn.functional.avg_pool1d(scores.unsqueeze(1), self.checkpoint['pool_frames'], stride=1).squeeze(1)
                scores = scores.amax(dim=1).tolist()
                maximum = max(maximum, max(scores))
                for (start, score) in zip(subset, scores):
                    if score >= self.checkpoint['threshold']:
                        selected.append({'start_seconds': start / 16000, 'end_seconds': min(len(waveform), start + 8000) / 16000, 'score': float(score)})
        return {'audio': str(Path(path).resolve()), 'target_symbol': 'ə', 'score': maximum, 'threshold': self.checkpoint['threshold'], 'detected': bool(selected), 'windows': selected, 'duration_seconds': metadata['duration_seconds'], 'denoise': 'off', 'scanned_windows': len(starts), 'processing_seconds': time.perf_counter() - started, 'note': 'Window scores detect ax; regions are approximate, not exact phone boundaries.'}


def predict(args):
    detector = TransferDetector(args.model)
    if args.inspect:
        print(json.dumps(detector.describe(), ensure_ascii=False, indent=2))
        return
    if not args.audio:
        raise ValueError('Provide --audio or --inspect.')
    results = [detector.predict(path, args.hop) for path in args.audio]
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False))
    else:
        for result in results:
            decision = 'PHÁT HIỆN' if result['detected'] else 'KHÔNG PHÁT HIỆN'
            print(f"{result['audio']}: {decision} /ə/ | score={result['score']:.6f} | ngưỡng={result['threshold']:.2f}")
            print(f"  {result['duration_seconds']:.3f}s audio, {len(result['windows'])} cửa sổ vượt ngưỡng; CPU {result['processing_seconds']:.2f}s")
            for window in result['windows'][:8]:
                print(f"  {window['start_seconds']:.2f}–{window['end_seconds']:.2f}s: {window['score']:.4f}")
    if args.output:
        write_json(args.output, results)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('predict', help='Detect schwa in one or more WAV files.')
    p.add_argument('--model', type=Path, default=ROOT / 'best.pt')
    p.add_argument('--audio', type=Path, nargs='+')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--hop', type=float, default=.05)
    p.add_argument('--json', action='store_true')
    p.add_argument('--output', type=Path)
    p.set_defaults(function=predict)
    p = commands.add_parser('denoise', help='Save filtered WAV and preserve input.')
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=denoise_cli)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)


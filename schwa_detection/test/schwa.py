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
RECORDING = {'version': 2, 'attention_backend': 'math', 'minimum_samples': 8000, 'maximum_samples': 160000, 'core_samples': 96000, 'margin_samples': 32000, 'hop_samples': 320, 'center_sample': 199.5, 'window_frames': 24}



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


def swap_dynamic_linears(model):
    """Construct quantized modules without allocating an entire float32 LARGE model."""
    for (name, module) in list(model.named_children()):
        if isinstance(module, nn.Linear):
            replacement = torch.ao.nn.quantized.dynamic.Linear(module.in_features, module.out_features, module.bias is not None, dtype=torch.qint8)
            setattr(model, name, replacement)
        else:
            swap_dynamic_linears(module)


def rebuild_acoustic(state):
    if state.get('quantization') == 'dynamic_int8_linears':
        with torch.device('meta'):
            model = torchaudio.models.wav2vec2_model(**state['model_config'])
        swap_dynamic_linears(model)
        model.to_empty(device='cpu')
    elif state.get('quantization'):
        raise ValueError('Unsupported encoder quantization.')
    else:
        model = torchaudio.models.wav2vec2_model(**state['model_config'])
    model.load_state_dict(state['model_state_dict'])
    model.normalize_waveform = state.get('normalize_waveform', False)
    model.feature_kind = state.get('feature_kind', 'embeddings')
    model.quantization = state.get('quantization')
    return model.eval()


def acoustic_features(model, waveforms, layers):
    if model.feature_kind == 'ctc_logits':
        if model.quantization:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                return torch.cat([model(waveform.unsqueeze(0))[0] for waveform in waveforms], dim=0)
        return model(waveforms)[0]
    if model.feature_kind != 'embeddings':
        raise ValueError('Unsupported acoustic feature representation.')
    return model.extract_features(waveforms, num_layers=layers)[0][-1]


def recording_features(model, waveform, layers):
    """Encode complete speech, using bounded chunks for recordings longer than ten seconds."""
    if waveform.ndim != 1 or len(waveform) == 0:
        raise ValueError('Recording features require nonempty mono audio.')
    if len(waveform) < RECORDING['minimum_samples']:
        waveform = nn.functional.pad(waveform, (0, RECORDING['minimum_samples'] - len(waveform)))

    def encode(audio):
        inputs = audio.unsqueeze(0)
        if model.normalize_waveform:
            inputs = nn.functional.layer_norm(inputs, (inputs.shape[-1],))
        return acoustic_features(model, inputs, layers)[0].to(torch.float16)
    if len(waveform) <= RECORDING['maximum_samples']:
        return encode(waveform)
    frame_count = (len(waveform) - 400) // RECORDING['hop_samples'] + 1
    centers = RECORDING['center_sample'] + torch.arange(frame_count) * RECORDING['hop_samples']
    result = None
    for start in range(0, len(waveform), RECORDING['core_samples']):
        end = min(len(waveform), start + RECORDING['core_samples'])
        indices = torch.nonzero((centers >= start) & (centers < end)).flatten()
        if not len(indices):
            continue
        input_start = max(0, start - RECORDING['margin_samples'])
        input_end = min(len(waveform), end + RECORDING['margin_samples'])
        values = encode(waveform[input_start:input_end])
        if result is None:
            result = torch.empty((frame_count, values.shape[-1]), dtype=torch.float16)
        local_indices = indices - input_start // RECORDING['hop_samples']
        if torch.any(local_indices < 0) or torch.any(local_indices >= len(values)):
            raise ValueError('Chunk frames do not align with the recording timeline.')
        result[indices] = values[local_indices]
    return result


def aligned_window_features(sequence, starts):
    """Interpolate the original 24 focal-frame centers; context changes neither labels nor regions."""
    if len(sequence) == 0:
        raise ValueError('Recording contains no acoustic frames.')
    positions = torch.as_tensor(starts, dtype=torch.float64).reshape(-1, 1) / RECORDING['hop_samples']
    positions = positions + torch.arange(RECORDING['window_frames'], dtype=torch.float64)
    positions = positions.clamp(0, len(sequence) - 1)
    left = positions.floor().long()
    right = (left + 1).clamp(max=len(sequence) - 1)
    fraction = (positions - left).float().unsqueeze(-1)
    values = sequence.float()
    interpolated = values[left] * (1 - fraction) + values[right] * fraction
    return interpolated.to(torch.float16).to(torch.float32)


def normalized_waveforms(waveforms, enabled):
    return nn.functional.layer_norm(waveforms, (waveforms.shape[-1],)) if enabled else waveforms


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

    @property
    def feature_dimension(self):
        return self.config['n_mels'] + (8 if self.config.get('prosody') else 0)

    def prosody(self, audio):
        """Label-free pitch, periodicity and energy on the same 25 ms / 20 ms timeline."""
        (c, p) = (self.config, self.config['prosody'])
        frames = audio.unfold(-1, c['n_fft'], c['hop_length'])
        rms = frames.square().mean(dim=-1).sqrt()
        log_rms = rms.clamp_min(1e-06).log()
        centered = frames - frames.mean(dim=-1, keepdim=True)
        fft = torch.fft.rfft(centered, n=2 * c['n_fft'])
        autocorrelation = torch.fft.irfft(fft.abs().square(), n=2 * c['n_fft'])[..., :c['n_fft']]
        energy = torch.nn.functional.pad(centered.square().cumsum(dim=-1), (1, 0))
        lo = int(math.ceil(c['sample_rate'] / p['maximum_f0']))
        hi = min(c['n_fft'] - 2, int(c['sample_rate'] / p['minimum_f0']))
        lags = torch.arange(lo, hi + 1, device=audio.device)
        denominator = (energy[..., c['n_fft'] - lags] * (energy[..., -1:] - energy[..., lags])).clamp_min(1e-16).sqrt()
        correlations = (autocorrelation[..., lags] / denominator).clamp(-1, 1)
        padded = torch.nn.functional.pad(correlations, (1, 1), value=-2.0)
        peaks = (correlations > padded[..., :-2]) & (correlations >= padded[..., 2:])
        strong = peaks & (correlations >= 0.95 * correlations.amax(dim=-1, keepdim=True))
        first = strong.to(torch.int64).argmax(dim=-1)
        index = torch.where(strong.any(dim=-1), first, correlations.argmax(dim=-1))
        confidence = correlations.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(0)
        confidence = torch.where(rms > 0.0001, confidence, torch.zeros_like(confidence))
        voiced = confidence >= p['voicing_threshold']
        log_f0 = (c['sample_rate'] / lags[index]).log()
        log_f0 = torch.where(voiced, log_f0, torch.zeros_like(log_f0))
        mean_f0 = log_f0.sum(dim=-1, keepdim=True) / voiced.sum(dim=-1, keepdim=True).clamp_min(1)
        relative_f0 = torch.where(voiced, log_f0 - mean_f0, torch.zeros_like(log_f0))
        relative_energy = log_rms - log_rms.mean(dim=-1, keepdim=True)
        delta_energy = torch.nn.functional.pad(log_rms[..., 1:] - log_rms[..., :-1], (1, 0))
        crossing = ((centered[..., 1:] >= 0) != (centered[..., :-1] >= 0)).float().mean(dim=-1)
        return torch.stack([log_f0, relative_f0, confidence, log_rms, relative_energy, delta_energy, crossing, voiced.float()], dim=-1)

    def forward(self, audio):
        c = self.config
        spectrum = torch.stft(audio, n_fft=c['n_fft'], hop_length=c['hop_length'], window=self.window, center=False, return_complex=True)
        mel = torch.log(torch.matmul(self.filters, spectrum.abs().square()).clamp_min(1e-10)).transpose(1, 2)
        return torch.cat([mel, self.prosody(audio)], dim=-1) if c.get('prosody') else mel


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
        self.acoustic = rebuild_acoustic(acoustic)
        self.layers = acoustic['layers']
        self.normalize_waveform = acoustic.get('normalize_waveform', False)
        self.feature_context = state['settings'].get('feature_context', 'window')
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
        return {'weights': str(self.path), 'sha256': digest(self.path), 'backend': state['backend'], 'target_symbol': state['target_symbol'], 'target_phones': state['target_phones'], 'epoch': state['epoch'], 'threshold': state['threshold'], 'device': 'cpu', 'denoise': 'off', 'window_seconds': 0.5, 'pool_frames': state['pool_frames'], 'backbone_layers': self.layers, 'self_contained': 'backbone' in state, 'source_metadata': state.get('source_metadata', {}), 'spectral_features': self.spectral is not None, 'normalize_waveform_per_file': self.normalize_waveform, 'feature_context': self.feature_context, 'acoustic_feature_kind': self.acoustic.feature_kind, 'encoder_quantization': self.acoustic.quantization, 'encoder_attention_backend': 'math' if self.acoustic.quantization and self.acoustic.feature_kind == 'ctc_logits' else 'default', 'parameters': sum((p.numel() for p in self.acoustic.parameters())) + sum((p.numel() for p in self.head.parameters())) + sum((m.weight().numel() + (m.bias().numel() if m.bias() is not None else 0) for m in self.acoustic.modules() if isinstance(m, torch.ao.nn.quantized.dynamic.Linear)))}

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
            sequence = recording_features(self.acoustic, torch.from_numpy(waveform), self.layers) if self.feature_context == 'recording' else None
            for index in range(0, len(starts), 8):
                subset = starts[index:index + 8]
                batch = np.stack([crop_window(waveform, start) for start in subset])
                if sequence is not None:
                    values = aligned_window_features(sequence, subset)
                else:
                    inputs = normalized_waveforms(torch.from_numpy(batch), self.normalize_waveform)
                    features = acoustic_features(self.acoustic, inputs, self.layers)
                    values = features.to(torch.float16).to(torch.float32)
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
        return {'audio': str(Path(path).resolve()), 'target_symbol': 'ə', 'score': maximum, 'threshold': self.checkpoint['threshold'], 'detected': bool(selected), 'windows': selected, 'duration_seconds': metadata['duration_seconds'], 'denoise': 'off', 'scanned_windows': len(starts), 'feature_context': self.feature_context, 'processing_seconds': time.perf_counter() - started, 'note': 'Window scores detect ax; regions are approximate, not exact phone boundaries.'}


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


"""Offline CPU inference for the phone GRU that reads the entire recording."""
import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

import schwa as core
from export_inference import definitions
from frozen_phoneme import RECORDING, rebuild_acoustic, recording_features, representation_state
from improve import PhoneHead
from recording_head_cpu import BACKBONE, region_scores

ROOT = Path(__file__).resolve().parent
RUNTIME_RECIPE = {'head_context': 'whole_recording', 'recording_recipe': RECORDING,
    'region_logit_interpolation': 'linear_then_float16_then_float32', 'mkldnn': False,
    'deterministic_algorithms': True, 'feature_dimension': 1416}
CODE_NAMES = ['lexical_inference.py', 'frozen_phoneme.py', 'improve.py', 'schwa.py',
              'recording_head_inference.py', 'recording_head_cpu.py']


def pack(model_path, output, backbone=BACKBONE):
    if output.exists():
        raise ValueError('Preserving existing packed weights.')
    head = torch.load(model_path, weights_only=True)
    if head.get('backend') != 'wav2vec2_recording_phone_head' or head.get('target_phones') != ['ax']:
        raise ValueError('Not a full-recording Schwa GRU checkpoint.')
    if core.digest(backbone) != head['backbone_sha256']:
        raise ValueError('Public encoder differs from the one used for training.')
    public = torch.load(backbone, weights_only=True, mmap=True)
    if public.get('fine_tuning'):
        raise ValueError('This head requires its original frozen public encoder.')
    head['backbone'] = representation_state(public, 'joint')
    head['licenses'] = {
        'TorchAudio': (ROOT / 'data/pretrained/LICENSE_WAV2VEC2.txt').read_text(encoding='utf-8'),
        'Phoneme XLSR weights Apache-2.0': (ROOT / 'data/pretrained/phoneme_xlsr53/LICENSE_APACHE_2.txt').read_text(encoding='utf-8')}
    core.save_torch(output, head)
    print('Packed full-recording Schwa GRU:', output, flush=True)


class RecordingHeadDetector:
    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        state = self.checkpoint = torch.load(self.path, weights_only=True, mmap=True)
        if (state.get('backend') != 'wav2vec2_recording_phone_head' or state.get('target_phones') != ['ax']
                or 'backbone' not in state or state['settings'].get('head_context') != 'whole_recording'
                or state['settings'].get('feature_context') != 'recording'
                or state['settings']['recording_recipe'] != RECORDING
                or state['settings']['region_logit_interpolation'] != RUNTIME_RECIPE['region_logit_interpolation']
                or state['head_config']['input_dim'] != 1416):
            raise ValueError('Use a packed full-recording Schwa GRU with the frozen runtime recipe.')
        if state['threshold_selection_split'] != 'validation' or state['provenance']['test_used_for_selection']:
            raise ValueError('Model and threshold must be selected using validation only.')
        torch.set_num_threads(state['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        self.acoustic = rebuild_acoustic(state['backbone'])
        self.head = PhoneHead(**state['head_config']).eval()
        self.head.load_state_dict(state['head_state_dict'])
        self.ax = state['phone_vocabulary'].index('ax')

    def sequence(self, audio):
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            return recording_features(self.acoustic, torch.from_numpy(audio), self.checkpoint['backbone']['layers'])

    def scores(self, sequence, starts):
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            logits = self.head(sequence.float().unsqueeze(0))[0]
            values = []
            for index in range(0, len(starts), 64):
                values.extend(region_scores(logits, starts[index:index + 64], self.ax, self.checkpoint['pool_frames']).tolist())
        return values

    def describe(self):
        return {'weights': str(self.path), 'sha256': core.digest(self.path), 'backend': self.checkpoint['backend'],
            'target_symbol': '\u0259', 'target_phones': ['ax'], 'epoch': self.checkpoint['epoch'],
            'threshold': self.checkpoint['threshold'], 'pool_frames': self.checkpoint['pool_frames'],
            'device': 'cpu', 'cpu_threads': self.checkpoint['cpu_threads'], 'self_contained': True,
            'feature_context': 'recording', 'head_context': 'whole_recording', 'window_seconds': .5,
            'runtime_recipe': RUNTIME_RECIPE, 'denoise': 'off'}

    def predict(self, path, hop_seconds=.05):
        if not 0 < hop_seconds <= .5:
            raise ValueError('Hop must be in (0, .5] seconds.')
        audio, metadata = core.load_audio(path)
        starts = list(range(0, max(1, len(audio) - 8000 + 1), max(1, round(hop_seconds * 16000))))
        last = max(0, len(audio) - 8000)
        if starts[-1] != last:
            starts.append(last)
        began = time.perf_counter()
        values = self.scores(self.sequence(audio), starts)
        threshold = self.checkpoint['threshold']
        windows = [{'start_seconds': start / 16000, 'end_seconds': min(len(audio), start + 8000) / 16000,
                    'score': value} for start, value in zip(starts, values) if value >= threshold]
        return {'audio': str(Path(path).resolve()), 'target_symbol': '\u0259', 'score': max(values),
            'threshold': threshold, 'detected': bool(windows), 'windows': windows,
            'duration_seconds': metadata['duration_seconds'], 'scanned_windows': len(starts),
            'feature_context': 'recording', 'head_context': 'whole_recording', 'denoise': 'off',
            'processing_seconds': time.perf_counter()-began,
            'note': 'Schwa presence in 0.5-second regions; boundaries are approximate.'}


def predict(args):
    detector = RecordingHeadDetector(args.model)
    if args.inspect:
        print(json.dumps(detector.describe(), ensure_ascii=False, indent=2))
        return
    if not args.audio:
        raise ValueError('Provide --audio or --inspect.')
    values = [detector.predict(path, args.hop) for path in args.audio]
    result = values[0] if len(values) == 1 else values
    if args.output:
        core.write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def standalone_source():
    header = '''"""Standalone offline CPU Schwa inference with a full-recording phone GRU."""
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
'''
    header += 'RECORDING = ' + repr(RECORDING) + '\nRUNTIME_RECIPE = ' + repr(RUNTIME_RECIPE) + '\nDENOISE = ' + repr(core.DENOISE) + '\n'
    body = definitions(ROOT / 'schwa.py', {'digest', 'write_json', 'load_audio', 'crop_window', 'denoise_audio', 'denoise_cli'})
    body += definitions(ROOT / 'frozen_phoneme.py', {'Float16StoredLinear', 'swap_stored_linears', 'swap_dynamic_linears',
        'rebuild_acoustic', 'encoder_prefix', 'rounded_encoder_embedding', 'acoustic_features', 'recording_features', 'aligned_window_features'})
    body += definitions(ROOT / 'improve.py', {'GroupFeatureNorm', 'PhoneHead'})
    body += definitions(ROOT / 'recording_head_cpu.py', {'region_logits', 'region_scores'})
    body += definitions(Path(__file__), {'RecordingHeadDetector', 'predict'})
    entry = '''def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('predict')
    p.add_argument('--model', type=Path, default=ROOT / 'best.pt')
    p.add_argument('--audio', type=Path, nargs='+')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--hop', type=float, default=.05)
    p.add_argument('--output', type=Path)
    p.set_defaults(function=predict)
    p = commands.add_parser('denoise')
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
'''
    return '\n\n\n'.join([header, *body, entry]) + '\n'


def export(model_path, output, metrics_path=None):
    archive_path = output.with_suffix('.zip')
    if (output.exists() and any(output.iterdir())) or archive_path.exists():
        raise ValueError('Preserving existing inference bundle.')
    state = torch.load(model_path, weights_only=True, mmap=True)
    if state.get('backend') != 'wav2vec2_recording_phone_head' or 'backbone' not in state or 'licenses' not in state:
        raise ValueError('Export a packed full-recording model.')
    model_sha = core.digest(model_path)
    evaluation = 'Gói thử nghiệm; chưa có kết quả test kèm theo.'
    if metrics_path:
        report = core.read_json(metrics_path)
        if (report['split'] != 'test' or report['model_sha256'] != model_sha
                or report['dataset_sha256'] != state['dataset_sha256'] or report['recording_head_recipe'] != RUNTIME_RECIPE
                or report['pool_frames'] != state['pool_frames'] or report['overall']['threshold'] != state['threshold']
                or report['threshold_selection_split'] != 'validation'):
            raise ValueError('Test report differs from exported weights, selection or runtime.')
        hashes = report['inference_code_sha256']
        if not set(CODE_NAMES).issubset(hashes) or any(core.digest(ROOT / name) != sha for name, sha in hashes.items()):
            raise ValueError('Inference source differs from the measured test runtime.')
        metric = report['overall']
        evaluation = (f"Test: {metric['files']} vùng 0,5 giây của {report['source_recordings']} bản ghi gốc; "
            f"precision {metric['precision']:.2%}, recall {metric['recall']:.2%}, accuracy {metric['accuracy']:.2%}. "
            'Model, ngưỡng và pooling được khóa bằng validation trước test.')
    source = standalone_source()
    compile(source, 'schwa.py', 'exec')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'schwa.py').write_text(source, encoding='utf-8')
    try:
        (output / 'best.pt').hardlink_to(model_path.resolve())
    except OSError:
        shutil.copy2(model_path, output / 'best.pt')
    (output / 'requirements.txt').write_text('torch==2.5.1\ntorchaudio==2.5.1\nnumpy==1.26.4\nscipy==1.14.1\nsoundfile==0.13.1\n', encoding='utf-8')
    licenses = '\n\n'.join(name + '\n\n' + text for name, text in state['licenses'].items())
    readme = f'''# Nhận diện Schwa /ə/ trên CPU

Gói đúng 4 file, chạy offline: `best.pt` có đủ encoder, head GRU và ngưỡng. Không cần src, dataset hoặc train.

{evaluation}

Python 3.10/3.11; chạy trong thư mục này:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe schwa.py predict --audio "E:\\audio\\recording.wav" --output result.json
.\\.venv\\Scripts\\python.exe schwa.py predict --inspect
```

WAV được đổi sang mono 16 kHz. Encoder dùng ngữ cảnh bản ghi, chia đoạn tối đa 10 giây có ngữ cảnh nếu cần;
GRU đọc toàn bộ chuỗi frame rồi chấm các vùng 0,5 giây, bước mặc định 50 ms. WAV ngắn được pad.
JSON có score, threshold, detected, windows; các vùng là vị trí dự đoán, chưa phải ranh giới âm vị chính xác.
Target AH0/AX; AH1/AH2 và ER/AXR là âm khác. Runtime CPU dùng MATH attention và tắt MKLDNN.
Kết quả trên corpus đọc câu chưa xác nhận chất lượng trên micro hoặc WAV đã lọc nhiễu.
Lọc nhiễu riêng: `python schwa.py denoise --audio input.wav --output cleaned.wav`.

Encoder công khai: facebook/wav2vec2-xlsr-53-espeak-cv-ft, revision 2c733782da5604684829819a5eb744c193fe9398,
weights Apache-2.0, linear lưu float16 và tính float32. Train dùng nhãn thủ công L2-ARCTIC v5.0,
bao gồm lời nói tự nhiên của các speaker train; nguồn 16 kHz chikingsley/l2-arctic-manual-v5.0-16k,
revision b7261545bc4e4a02910e9c85c659eb866e00db7a, CC BY-NC 4.0. Gói không kèm audio corpus; dùng phi thương mại.

SHA-256 best.pt: {model_sha}

## Giấy phép nguồn model

```text
{licenses}
```
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ['schwa.py', 'best.pt', 'requirements.txt', 'README.md']:
            archive.write(output / name, arcname=name)
    print('Offline full-recording GRU bundle:', archive_path, flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backbone', type=Path, default=BACKBONE)
    p.set_defaults(function=lambda args: pack(args.model, args.output, args.backbone))
    p = commands.add_parser('export')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metrics', type=Path)
    p.set_defaults(function=lambda args: export(args.model, args.output, args.metrics))
    p = commands.add_parser('predict')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--audio', type=Path, nargs='+')
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--hop', type=float, default=.05)
    p.add_argument('--output', type=Path)
    p.set_defaults(function=predict)
    args = parser.parse_args()
    args.function(args)


if __name__ == '__main__':
    main()

"""Offline CPU inference and four-file export for the frozen WavLM Schwa heads."""
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
from frozen_phoneme import RECORDING, aligned_window_features
from frozen_wavlm import OUTPUT, WAVLM_RECIPE, rebuild_wavlm, recording_embeddings
from wavlm_train_cpu import NORMALIZATION, centered_embeddings
from improve import PhoneHead

ROOT = Path(__file__).resolve().parent


def pack(model_path, output, backbone=OUTPUT):
    if output.exists():
        raise ValueError('Preserving an existing packed model.')
    state = torch.load(model_path, weights_only=True, mmap=True)
    if state.get('backend') != 'wavlm_phone_head' or state.get('target_phones') != ['ax']:
        raise ValueError('Not a WavLM Schwa phone head.')
    if core.digest(backbone) != state['backbone_sha256']:
        raise ValueError('Public encoder differs from the one used to train this head.')
    public = torch.load(backbone, weights_only=True, mmap=True)
    if public['recipe'] != state['settings']['wavlm_recipe']:
        raise ValueError('Encoder feature recipe differs from training.')
    state['backbone'] = public
    state['licenses'] = {'WavLM Microsoft MIT': (ROOT / 'data/pretrained/LICENSE_WAVLM_MICROSOFT.txt').read_text(encoding='utf-8'),
                         'TorchAudio': (ROOT / 'data/pretrained/LICENSE_WAV2VEC2.txt').read_text(encoding='utf-8')}
    core.save_torch(output, state)
    print('Packed offline WavLM model:', output, flush=True)


class WavLMDetector:
    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        self.checkpoint = torch.load(self.path, weights_only=True, mmap=True)
        state = self.checkpoint
        if (state.get('backend') != 'wavlm_phone_head' or state.get('target_phones') != ['ax']
                or 'backbone' not in state or state['settings'].get('feature_context') != 'recording'):
            raise ValueError('Use the packed recording-context WavLM model.')
        if (state['settings']['wavlm_recipe'] != WAVLM_RECIPE
                or state['settings']['recording_normalization'] != NORMALIZATION
                or state['backbone']['recipe'] != WAVLM_RECIPE):
            raise ValueError('WavLM inference feature recipe differs from training.')
        if state['threshold_selection_split'] != 'validation' or state['provenance']['test_used_for_selection']:
            raise ValueError('WavLM threshold must be selected using validation only.')
        torch.set_num_threads(state['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        self.acoustic = rebuild_wavlm(state['backbone'])
        self.head = PhoneHead(**state['head_config']).eval()
        self.head.load_state_dict(state['head_state_dict'])
        self.ax = state['phone_vocabulary'].index('ax')

    def sequence(self, audio):
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            return centered_embeddings(recording_embeddings(self.acoustic, torch.from_numpy(audio)))

    def scores(self, sequence, starts):
        values = []
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            for index in range(0, len(starts), 8):
                features = aligned_window_features(sequence, starts[index:index + 8])
                probabilities = self.head(features).softmax(-1)[:, :, self.ax]
                pool = self.checkpoint['pool_frames']
                if pool > 1:
                    probabilities = torch.nn.functional.avg_pool1d(probabilities.unsqueeze(1), pool, stride=1).squeeze(1)
                values.extend(probabilities.amax(-1).tolist())
        return values

    def describe(self):
        state = self.checkpoint
        return {'weights': str(self.path), 'sha256': core.digest(self.path), 'backend': state['backend'],
                'target_symbol': '\u0259', 'target_phones': ['ax'], 'epoch': state['epoch'],
                'threshold': state['threshold'], 'pool_frames': state['pool_frames'], 'device': 'cpu',
                'cpu_threads': state['cpu_threads'], 'self_contained': True, 'feature_context': 'recording',
                'window_seconds': .5, 'feature_dimension': 768, 'recipe': WAVLM_RECIPE,
                'normalization': NORMALIZATION, 'deterministic_algorithms': True,
                'mkldnn': False, 'denoise': 'off'}

    def predict(self, path, hop_seconds=.05):
        if not 0 < hop_seconds <= .5:
            raise ValueError('Hop must be in (0, .5] seconds.')
        audio, metadata = core.load_audio(path)
        hop = max(1, round(hop_seconds * 16000))
        starts = list(range(0, max(1, len(audio) - 8000 + 1), hop))
        last = max(0, len(audio) - 8000)
        if starts[-1] != last:
            starts.append(last)
        begun = time.perf_counter()
        scores = self.scores(self.sequence(audio), starts)
        threshold = self.checkpoint['threshold']
        windows = [{'start_seconds': start / 16000, 'end_seconds': min(len(audio), start + 8000) / 16000,
                    'score': score} for start, score in zip(starts, scores) if score >= threshold]
        return {'audio': str(Path(path).resolve()), 'target_symbol': '\u0259', 'score': max(scores),
                'threshold': threshold, 'detected': bool(windows), 'windows': windows,
                'duration_seconds': metadata['duration_seconds'], 'denoise': 'off',
                'scanned_windows': len(starts), 'feature_context': 'recording',
                'processing_seconds': time.perf_counter() - begun,
                'note': 'Window scores detect AX; regions are approximate, not exact phone boundaries.'}


def predict(args):
    detector = WavLMDetector(args.model)
    if args.inspect:
        print(json.dumps(detector.describe(), ensure_ascii=False, indent=2))
        return
    if not args.audio:
        raise ValueError('Provide --audio or --inspect.')
    results = [detector.predict(path, args.hop) for path in args.audio]
    value = results[0] if len(results) == 1 else results
    if args.output:
        core.write_json(args.output, value)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def standalone_source():
    header = '''"""Standalone offline CPU Schwa inference using WavLM."""
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
    header += 'RECORDING = ' + repr(RECORDING) + '\n'
    header += 'WAVLM_RECIPE = ' + repr(WAVLM_RECIPE) + '\n'
    header += 'NORMALIZATION = ' + repr(NORMALIZATION) + '\n'
    header += 'DENOISE = ' + repr(core.DENOISE) + '\n'
    body = definitions(ROOT / 'schwa.py', {'digest', 'write_json', 'load_audio', 'crop_window', 'denoise_audio', 'denoise_cli'})
    body += definitions(ROOT / 'frozen_phoneme.py', {'Float16StoredLinear', 'swap_stored_linears', 'aligned_window_features'})
    body += definitions(ROOT / 'frozen_wavlm.py', {'rebuild_wavlm', 'recording_embeddings'})
    body += definitions(ROOT / 'wavlm_train_cpu.py', {'centered_embeddings'})
    body += definitions(ROOT / 'improve.py', {'GroupFeatureNorm', 'PhoneHead'})
    body += definitions(Path(__file__), {'WavLMDetector', 'predict'})
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
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserving existing inference folder.')
    archive_path = output.with_suffix('.zip')
    if archive_path.exists():
        raise ValueError('Preserving existing archive.')
    state = torch.load(model_path, weights_only=True, mmap=True)
    if state.get('backend') != 'wavlm_phone_head' or 'backbone' not in state or 'licenses' not in state:
        raise ValueError('Export requires a packed WavLM model.')
    model_sha = core.digest(model_path)
    evaluation = 'Gói thử nghiệm, chưa có báo cáo test đi kèm; không coi kiểm tra kỹ thuật là bằng chứng chất lượng.'
    if metrics_path:
        report = core.read_json(metrics_path)
        if (report['split'] != 'test' or report['model_sha256'] != model_sha
                or report['dataset_sha256'] != state['dataset_sha256']
                or report['wavlm_recipe'] != WAVLM_RECIPE or report['recording_normalization'] != NORMALIZATION
                or report['pool_frames'] != state['pool_frames']
                or report['overall']['threshold'] != state['threshold']
                or report['threshold_selection_split'] != 'validation'):
            raise ValueError('Test report differs from the frozen exported model and runtime.')
        if any(core.digest(ROOT / name) != sha for name, sha in report['inference_code_sha256'].items()):
            raise ValueError('Inference source differs from the one used to measure test quality.')
        metric = report['overall']
        evaluation = (f"Test: {metric['files']} vùng 0,5 giây trên {report['source_recordings']} bản ghi gốc; "
                      f"precision {metric['precision']:.2%}, recall {metric['recall']:.2%}, accuracy {metric['accuracy']:.2%}. "
                      'Model và ngưỡng được khóa bằng validation trước test.')
    source = standalone_source()
    compile(source, 'schwa.py', 'exec')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'schwa.py').write_text(source, encoding='utf-8')
    shutil.copy2(model_path, output / 'best.pt')
    (output / 'requirements.txt').write_text('torch==2.5.1\ntorchaudio==2.5.1\nnumpy==1.26.4\nscipy==1.14.1\nsoundfile==0.13.1\n', encoding='utf-8')
    licenses = '\n\n'.join(name + '\n\n' + text for name, text in state['licenses'].items())
    readme = f'''# Nhận diện Schwa /ə/ trên CPU

Gói có đúng 4 file. Không cần src, dataset, notebook hoặc train. `best.pt` chứa encoder WavLM Base+,
lớp phân loại và ngưỡng; chạy offline, không tải model và không cần câu chép lại.

{evaluation}

Cài Python 3.10/3.11 rồi chạy trong thư mục này:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe schwa.py predict --audio "E:\\audio\\recording.wav" --output result.json
.\\.venv\\Scripts\\python.exe schwa.py predict --inspect
```

WAV được đổi sang mono 16 kHz, xử lý ngữ cảnh bản ghi rồi chấm vùng 0,5 giây, bước mặc định 50 ms.
WAV ngắn được pad; WAV dài chia đoạn tối đa 10 giây có ngữ cảnh. Đặc trưng trung bình lớp 6/9/12,
trừ trung bình từng kênh của chính bản ghi. JSON có `detected`, `score`, `threshold`, `windows`;
các vùng là vị trí dự đoán, chưa phải ranh giới âm vị chính xác. Target AH0/AX; AH1/AH2 và ER/AXR là âm khác.

Kết quả corpus đọc câu không bảo đảm cùng độ chính xác trên micro, tiếng ồn hoặc quyết định cho cả WAV.
Lọc nhiễu riêng bằng `python schwa.py denoise --audio input.wav --output cleaned.wav`; chưa đo chất lượng trên WAV đã lọc.

Encoder: [TorchAudio WavLM Base+](https://docs.pytorch.org/audio/2.5.0/pipelines.html), pretrained công khai,
linear float16 và convolution/norm/MHA in-projection float32; tính toán CPU float32.
Train: L2-ARCTIC manual v5.0, bản 16 kHz của chikingsley, revision b7261545bc4e4a02910e9c85c659eb866e00db7a,
CC BY-NC 4.0; nhãn perceived phone, split V5 tách người nói/câu đọc. Gói không kèm audio corpus; dùng phi thương mại.

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
    print('Offline WavLM bundle:', archive_path, flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backbone', type=Path, default=OUTPUT)
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

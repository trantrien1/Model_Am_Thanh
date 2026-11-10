"""Pack and run the audio-only Whisper lexical Schwa head offline on CPU."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import whisper

import schwa as core
from export_inference import definitions
from frozen_phoneme import RECORDING, aligned_window_features, rebuild_acoustic, recording_features, representation_state
from improve import PhoneHead
from lexical_context import ARPABET_IPA
from lexical_inference import dictionary_from_text
from whisper_cpu import MODEL, MODEL_SHA, VERSION, CPU_MATH, DECODE, package_identity
from whisper_lexical import RECIPE, word_features

ROOT = Path(__file__).resolve().parent
RUNTIME_RECIPE = {'acoustic': RECORDING, 'lexical': RECIPE, 'whisper_decode': DECODE,
                  'cpu_math': CPU_MATH, 'head_context': 'original_0.5_second_region',
                  'feature_dimension': 1424, 'storage': 'float16_then_float32'}
CODE_NAMES = ['lexical_inference.py', 'whisper_inference.py', 'whisper_lexical.py',
              'whisper_cpu.py', 'lexical_context.py', 'frozen_phoneme.py', 'improve.py', 'schwa.py']


def pack(model_path, output, backbone=None):
    from whisper_lexical import metadata
    from whisper_train_cpu import BACKBONE
    from lexical_context import DICTIONARY_SOURCE
    if output.exists():
        raise ValueError('Preserving existing packed weights.')
    state = torch.load(model_path, weights_only=True)
    if state.get('backend') != 'wav2vec2_whisper_lexical_head' or state.get('target_phones') != ['ax']:
        raise ValueError('Use a trained Whisper lexical Schwa head.')
    if state['lexical'] != metadata():
        raise ValueError('Whisper training resources or feature sources changed.')
    backbone = Path(backbone or BACKBONE)
    if core.digest(backbone) != state['backbone_sha256'] or core.digest(MODEL) != MODEL_SHA:
        raise ValueError('Frozen public weights differ from the trained recipe.')
    public = torch.load(backbone, weights_only=True, mmap=True)
    if public.get('fine_tuning'):
        raise ValueError('Use the original frozen public phoneme encoder.')
    state['backbone'] = representation_state(public, 'joint')
    state['whisper_resources'] = {
        'checkpoint': torch.load(MODEL, weights_only=True, mmap=True),
        'alignment_heads': whisper._ALIGNMENT_HEADS['base.en'],
        'dictionary_text': (DICTIONARY_SOURCE / 'cmudict.dict').read_text(encoding='utf-8'),
        'vocabulary_text': (ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json').read_text(encoding='utf-8')}
    state['licenses'] = {
        'TorchAudio': (ROOT / 'data/pretrained/LICENSE_WAV2VEC2.txt').read_text(encoding='utf-8'),
        'Phoneme XLSR Apache-2.0': (ROOT / 'data/pretrained/phoneme_xlsr53/LICENSE_APACHE_2.txt').read_text(encoding='utf-8'),
        'Whisper MIT': (MODEL.parent / 'LICENSE_WHISPER.txt').read_text(encoding='utf-8'),
        'CMU dictionary': (DICTIONARY_SOURCE / 'LICENSE').read_text(encoding='utf-8')}
    state['runtime_recipe'] = RUNTIME_RECIPE
    core.save_torch(output, state)
    print('Packed offline Whisper Schwa:', output, flush=True)


class WhisperLexicalDetector:
    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        state = self.checkpoint = torch.load(self.path, weights_only=True, mmap=True)
        if (state.get('backend') != 'wav2vec2_whisper_lexical_head' or state.get('target_phones') != ['ax']
                or 'backbone' not in state or 'whisper_resources' not in state
                or state.get('runtime_recipe') != RUNTIME_RECIPE):
            raise ValueError('Use a packed Whisper lexical Schwa model with the frozen runtime recipe.')
        if state['threshold_selection_split'] != 'validation' or state['provenance']['test_used_for_selection']:
            raise ValueError('Model and threshold must be selected on validation only.')
        recipe = state['lexical']
        if (recipe['model_sha256'] != MODEL_SHA or recipe['package_version'] != VERSION
                or recipe['package_files_sha256'] != package_identity()
                or recipe['recording_recipe'] != RECORDING or recipe['lexical_recipe'] != RECIPE
                or recipe['decode_recipe'] != DECODE or recipe['cpu_math'] != CPU_MATH
                or state['head_config']['input_dim'] != 1424):
            raise ValueError('Whisper package, decoder or feature recipe differs from training.')
        resources = state['whisper_resources']
        for name, sha in [('dictionary_text', 'dictionary_sha256'), ('vocabulary_text', 'vocabulary_sha256')]:
            if hashlib.sha256(resources[name].encode('utf-8')).hexdigest() != recipe[sha]:
                raise ValueError('Embedded lexical resource differs from training: ' + name)
        if resources['alignment_heads'] != whisper._ALIGNMENT_HEADS['base.en']:
            raise ValueError('Public Whisper word alignment heads changed.')
        torch.set_num_threads(CPU_MATH['threads'])
        torch.use_deterministic_algorithms(True)
        self.acoustic = rebuild_acoustic(state['backbone'])
        raw = resources['checkpoint']
        self.asr = whisper.Whisper(whisper.ModelDimensions(**raw['dims']))
        self.asr.load_state_dict(raw['model_state_dict'])
        self.asr.set_alignment_heads(resources['alignment_heads'])
        self.asr = self.asr.cpu().eval()
        self.dictionary = dictionary_from_text(resources['dictionary_text'])
        self.vocabulary = json.loads(resources['vocabulary_text'])
        self.head = PhoneHead(**state['head_config']).eval()
        self.head.load_state_dict(state['head_state_dict'])
        self.ax = state['phone_vocabulary'].index('ax')

    def sequence(self, waveform):
        audio = np.asarray(waveform, dtype=np.float32)
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            acoustic = recording_features(self.acoustic, torch.from_numpy(audio), self.checkpoint['backbone']['layers'])
            decoded = self.asr.transcribe(audio, **DECODE)
            words = [w for segment in decoded['segments'] for w in segment.get('words', [])]
            if any(not 0 <= w['start'] <= w['end'] <= len(audio)/16000 + .1 for w in words):
                raise ValueError('Invalid audio-inferred Whisper word timings.')
            lexical = word_features(words, acoustic[:, 1024:].float(), self.dictionary, self.vocabulary).half().float()
            return torch.cat([acoustic.float(), lexical], dim=-1)

    def scores(self, sequence, starts):
        result = []
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            for start in range(0, len(starts), 64):
                values = aligned_window_features(sequence, starts[start:start+64])
                scores = self.head(values).softmax(-1)[:, :, self.ax]
                if self.checkpoint['pool_frames'] > 1:
                    scores = torch.nn.functional.avg_pool1d(scores.unsqueeze(1), self.checkpoint['pool_frames'], stride=1).squeeze(1)
                result.extend(scores.amax(1).tolist())
        return result

    def describe(self):
        return {'weights': str(self.path), 'sha256': core.digest(self.path), 'backend': self.checkpoint['backend'],
                'target_symbol': '\u0259', 'target_phones': ['ax'], 'epoch': self.checkpoint['epoch'],
                'threshold': self.checkpoint['threshold'], 'pool_frames': self.checkpoint['pool_frames'],
                'device': 'cpu', 'cpu_threads': 2, 'self_contained': True, 'downloads_required': False,
                'feature_context': 'recording', 'window_seconds': .5, 'runtime_recipe': RUNTIME_RECIPE,
                'reference_text_used_as_features': False, 'denoise': 'off'}

    def predict(self, path, hop_seconds=.05):
        if not 0 < hop_seconds <= .5:
            raise ValueError('Hop must be in (0, .5] seconds.')
        audio, metadata = core.load_audio(path)
        starts = list(range(0, max(1, len(audio)-8000+1), max(1, round(hop_seconds*16000))))
        last = max(0, len(audio)-8000)
        if starts[-1] != last:
            starts.append(last)
        began = time.perf_counter()
        scores = self.scores(self.sequence(audio), starts)
        threshold = self.checkpoint['threshold']
        windows = [{'start_seconds': start/16000, 'end_seconds': min(len(audio), start+8000)/16000,
                    'score': score} for start, score in zip(starts, scores) if score >= threshold]
        return {'audio': str(Path(path).resolve()), 'target_symbol': '\u0259', 'score': max(scores),
                'threshold': threshold, 'detected': bool(windows), 'windows': windows,
                'duration_seconds': metadata['duration_seconds'], 'scanned_windows': len(starts),
                'processing_seconds': time.perf_counter()-began, 'feature_context': 'recording',
                'denoise': 'off', 'note': 'Schwa presence in 0.5-second regions; boundaries are approximate.'}


def predict(args):
    detector = WhisperLexicalDetector(args.model)
    if args.inspect:
        print(json.dumps(detector.describe(), ensure_ascii=False, indent=2))
        return
    if not args.audio:
        raise ValueError('Provide --audio or --inspect.')
    results = [detector.predict(path, args.hop) for path in args.audio]
    result = results[0] if len(results) == 1 else results
    if args.output:
        core.write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def standalone_source():
    header = '''"""Standalone offline CPU Schwa inference with Whisper lexical context."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import whisper
from torch import nn
from scipy.ndimage import uniform_filter
from scipy.signal import istft, resample_poly, stft
ROOT = Path(__file__).resolve().parent
'''
    header += '\n'.join(name + ' = ' + repr(value) for name, value in
        [('RECORDING', RECORDING), ('RECIPE', RECIPE), ('VERSION', VERSION), ('MODEL_SHA', MODEL_SHA),
         ('CPU_MATH', CPU_MATH), ('DECODE', DECODE), ('RUNTIME_RECIPE', RUNTIME_RECIPE),
         ('ARPABET_IPA', ARPABET_IPA), ('DENOISE', core.DENOISE)]) + '\n'
    body = definitions(ROOT / 'schwa.py', {'digest', 'write_json', 'load_audio', 'crop_window', 'denoise_audio', 'denoise_cli'})
    body += definitions(ROOT / 'frozen_phoneme.py', {'Float16StoredLinear', 'swap_stored_linears', 'swap_dynamic_linears',
        'rebuild_acoustic', 'encoder_prefix', 'rounded_encoder_embedding', 'acoustic_features', 'recording_features', 'aligned_window_features'})
    body += definitions(ROOT / 'improve.py', {'GroupFeatureNorm', 'PhoneHead'})
    body += definitions(ROOT / 'lexical_context.py', {'pronunciation_ids', 'ctc_state_posteriors'})
    body += definitions(ROOT / 'lexical_inference.py', {'dictionary_from_text'})
    body += definitions(ROOT / 'whisper_cpu.py', {'package_identity'})
    body += definitions(ROOT / 'whisper_lexical.py', {'word_features'})
    body += definitions(Path(__file__), {'WhisperLexicalDetector', 'predict'})
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
    if state.get('backend') != 'wav2vec2_whisper_lexical_head' or 'whisper_resources' not in state or 'licenses' not in state:
        raise ValueError('Export a packed Whisper Schwa model.')
    model_sha = core.digest(model_path)
    evaluation = 'Gói thử nghiệm; chưa có kết quả test kèm theo.'
    if metrics_path:
        report = core.read_json(metrics_path)
        if (report['split'] != 'test' or report['model_sha256'] != model_sha
                or report['dataset_sha256'] != state['dataset_sha256'] or report['whisper_runtime_recipe'] != RUNTIME_RECIPE
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
    (output / 'requirements.txt').write_text(
        'torch==2.5.1\ntorchaudio==2.5.1\nnumpy==1.26.4\nscipy==1.14.1\nsoundfile==0.13.1\n'
        'openai-whisper==20250625\ntiktoken==0.14.0\nmore-itertools==11.1.0\nregex==2026.9.29\nnumba==0.68.0\ntqdm==4.70.1\n', encoding='utf-8')
    licenses = '\n\n'.join(name + '\n\n' + text for name, text in state['licenses'].items())
    readme = f'''# Nhận diện Schwa /ə/ trên CPU

Gói đúng 4 file: `schwa.py`, `best.pt`, `requirements.txt`, `README.md`. Weight có đủ encoder âm vị,
Whisper Base English, từ điển, head GRU và ngưỡng. Không cần src, dataset hoặc train; inference không tải model.

{evaluation}

Python 3.10/3.11; cài dependency một lần rồi chạy offline trong thư mục này:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe schwa.py predict --audio "E:\\audio\\recording.wav" --output result.json
.\\.venv\\Scripts\\python.exe schwa.py predict --inspect
```

WAV được đổi sang mono 16 kHz. Model suy ra từ/mốc thời gian từ audio, căn với âm vị công khai rồi
chấm Schwa trong các vùng 0,5 giây, bước mặc định 50 ms. Encoder âm vị dùng đoạn tối đa 10 giây có ngữ cảnh,
Whisper tự xử lý audio dài; WAV ngắn được pad ở encoder âm vị. JSON có score, threshold, detected, windows.
Vùng dự đoán là vị trí gần đúng; không phải ranh giới âm vị chính xác. Target AH0/AX; AH1/AH2 và ER/AXR là âm khác.
CPU hai thread, MATH attention và MKLDNN tắt. Không nhập transcript chuẩn hay nhãn. Chất lượng trên corpus đọc câu
chưa xác nhận hiệu quả trên micro hoặc audio đã lọc nhiễu.
Lọc nhiễu riêng: `python schwa.py denoise --audio input.wav --output cleaned.wav`.

Nguồn encoder facebook/wav2vec2-xlsr-53-espeak-cv-ft, revision 2c733782da5604684829819a5eb744c193fe9398,
Apache-2.0; Whisper Base English công khai và code MIT; CMU dictionary kèm giấy phép bên dưới.
Train dùng nhãn thủ công L2-ARCTIC: chikingsley/l2-arctic-manual-v5.0-16k,
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
    print('Offline Whisper Schwa bundle:', archive_path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backbone', type=Path)
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

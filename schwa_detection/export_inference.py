"""Export four standalone inference files from a packed, evaluated schwa model."""
import argparse
import ast
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
import torch
from frozen_phoneme import RECORDING

ROOT = Path(__file__).resolve().parent


class LocalHelpers(ast.NodeTransformer):
    def visit_Attribute(self, node):
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == 'core':
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def definitions(path, names):
    source = Path(path).read_text(encoding='utf-8')
    result = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if node.name == 'TransferDetector':
                # A friend's bundle must contain its backbone, rather than depend on the training workspace.
                constructor = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
                for statement in constructor.body:
                    if isinstance(statement, ast.If) and ast.unparse(statement.test) == "'backbone' in state":
                        statement.orelse = ast.parse("raise ValueError('Checkpoint is not self-contained; use the exported best.pt.')").body
            node = LocalHelpers().visit(node)
            ast.fix_missing_locations(node)
            result.append(ast.unparse(node))
    if len(result) != len(names):
        raise ValueError('Missing an inference function in the source.')
    return result


def export(model_path, output, metrics_path=None):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preserving existing bundle; choose an empty output directory.')
    archive_path = output.with_suffix('.zip')
    if archive_path.exists():
        raise ValueError('Preserving existing archive; choose another output directory.')
    if not model_path.is_file():
        raise ValueError('Packed model does not exist.')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    if checkpoint.get('backend') != 'wav2vec2_phone_head' or 'backbone' not in checkpoint:
        raise ValueError('Export requires a self-contained checkpoint; run improve.py pack first.')
    layers = checkpoint['backbone']['layers']
    bundle_name = checkpoint['backbone']['source_bundle'].split('.')[-1]
    weights_mb = model_path.stat().st_size / 1024 ** 2
    classifier_text = ('bộ phân loại schwa theo khung âm' if checkpoint['settings'].get('task') in ('binary', 'presence')
                       else 'bộ phân loại âm vị')
    metrics = json.loads(metrics_path.read_text(encoding='utf-8')) if metrics_path else None
    if metrics and metrics.get('split') != 'test':
        raise ValueError('Only a held-out test report can be presented as test accuracy in an exported bundle.')
    feature_context = checkpoint['settings'].get('feature_context', 'window')
    if metrics and metrics.get('feature_context', 'window') != feature_context:
        raise ValueError('Metrics input context does not match the exported checkpoint.')
    if metrics and feature_context == 'recording' and metrics.get('acoustic_runtime_recipe') != RECORDING:
        raise ValueError('Test runtime recipe differs from exported inference; recheck the frozen model with the current runtime.')
    if metrics and metrics['model_sha256'] != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError('Metrics do not correspond to the exported checkpoint.')
    license_text = (ROOT / 'data/pretrained/LICENSE_WAV2VEC2.txt').read_text(encoding='utf-8')
    acoustic_source = checkpoint['backbone']
    if acoustic_source.get('license') == 'Apache-2.0':
        license_text += '\n\nModel weights: Apache License 2.0\n' + (ROOT / 'data/pretrained/phoneme_xlsr53/LICENSE_APACHE_2.txt').read_text(encoding='utf-8')
    if acoustic_source['source_bundle'].startswith('huggingface.'):
        backbone_text = ('[' + bundle_name + '](' + acoustic_source['source_url'] + '/tree/' + acoustic_source['source_revision'] + ')')
        if acoustic_source.get('fine_tuning'):
            weight_text = 'phần encoder cố định lưu linear float16; các lớp đã fine-tune lưu float32; tính toán float32'
        elif acoustic_source.get('quantization') == 'float16_linear_storage':
            weight_text = 'lưu linear float16, tính toán float32 từng lớp; convolution/chuẩn hóa float32'
        else:
            weight_text = 'lượng tử hóa int8 cho lớp linear, giữ convolution/chuẩn hóa float32'
    else:
        backbone_text = '[TorchAudio ' + bundle_name + '](https://docs.pytorch.org/audio/2.5.0/pipelines.html)'
        weight_text = 'lưu weights float16 và tính toán float32'
    output.mkdir(parents=True, exist_ok=True)
    header = '''"""Standalone CPU schwa /\u0259/ inference. Model and threshold are embedded in best.pt."""
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
'''
    header += 'RECORDING = ' + repr(RECORDING) + '\n'
    body = definitions(ROOT / 'schwa.py', {'digest', 'write_json', 'load_audio', 'crop_window', 'denoise_audio', 'denoise_cli'})
    body += definitions(ROOT / 'frozen_phoneme.py', {'Float16StoredLinear', 'swap_stored_linears', 'swap_dynamic_linears', 'rebuild_acoustic', 'representation_state', 'encoder_prefix', 'rounded_encoder_embedding', 'acoustic_features',
                                                   'recording_features', 'aligned_window_features'})
    body += definitions(ROOT / 'improve.py', {'normalized_waveforms', 'AlignedMel', 'GroupFeatureNorm', 'PhoneHead', 'TransferDetector', 'predict'})
    entry = '''def main():
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
'''
    (output / 'schwa.py').write_text('\n\n\n'.join([header, *body, entry]) + '\n', encoding='utf-8')
    requirements = 'torch==2.5.1\ntorchaudio==2.5.1\nnumpy==1.26.4\nscipy==1.14.1\nsoundfile==0.13.1\n'
    (output / 'requirements.txt').write_text(requirements, encoding='utf-8')
    shutil.copy2(model_path, output / 'best.pt')
    evaluation = 'Chưa có báo cáo test kèm gói này; đây là gói thử nghiệm.'
    context_text = ('Model phân tích WAV đầy đủ, chia đoạn tối đa 10 giây nếu cần, rồi chấm từng vùng 0,5 giây trên timeline. '
                    'Kết quả test có ngữ cảnh bản ghi gốc; không được hiểu là cùng accuracy trên các clip 0,5 giây đã cắt rời.'
                    if feature_context == 'recording' else
                    'Model trích đặc trưng riêng từng cửa sổ 0,5 giây.')
    annotation_text = 'Kết quả đo trên nhãn âm vị căn chỉnh tự động của CMU ARCTIC.'
    corpus_text = ('Nguồn corpus: [CMU ARCTIC](http://festvox.org/cmu_arctic/), John Kominek và Alan W. Black. '
                   'Train dùng BDL/CLB/RMS/JMK/SLT. Copyright Carnegie Mellon University: BDL/JMK/SLT 2003, RMS 2004, CLB 2005. '
                   'Gói không kèm WAV nguồn hoặc mã Festival; README/COPYING nguyên bản giữ tại dữ liệu dự án.')
    if metrics:
        sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if metrics['model_sha256'] != sha:
            raise ValueError('Metrics do not correspond to the exported checkpoint.')
        overall = metrics['overall']
        test_units = (f"{overall['files']:,} vùng 0,5 giây trên {metrics['source_recordings']} bản ghi gốc"
                      if feature_context == 'recording' else f"{overall['files']:,} WAV")
        evaluation = (f"Test: {test_units}; accuracy {overall['accuracy']:.2%}, "
                      f"precision {overall['precision']:.2%}, recall {overall['recall']:.2%}, F1 {overall['f1']:.4f}. "
                      f"Speaker test: {', '.join(metrics['speakers'])}. Ngưỡng {overall['threshold']:.2f} chọn trên validation.")
        if 'always_negative_accuracy' in metrics:
            evaluation += f" Accuracy nếu đoán tất cả âm tính: {metrics['always_negative_accuracy']:.2%}."
        if metrics.get('dataset_version') in ('v3', 'v4', 'v5'):
            annotation_text = ('Kết quả đo trên L2-ARCTIC có nhãn âm vị được người chú giải kiểm tra, '
                               'dùng âm thực sự phát ra (perceived phone). AH0/AX là schwa; AH1/AH2 và ER/AXR giữ riêng. '
                               'Loại cửa sổ chồng lên đoạn không rõ nhãn; chưa nghe lại từng WAV được cắt trong dự án này.')
            corpus_text = ('Nguồn: [L2-ARCTIC v5.0](https://psi.engr.tamu.edu/l2-arctic-corpus/), '
                           'Guanlong Zhao, Sinem Sonsaat, Alif Silpachai, Ivana Lucic, Evgeny Chukharev-Hudilainen, '
                           'John Levis và Ricardo Gutierrez-Osuna (Interspeech 2018, DOI 10.21437/Interspeech.2018-1110). '
                           'Dữ liệu theo [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/), chỉ dùng phi thương mại. '
                           'Bản 16 kHz: [chikingsley/l2-arctic-manual-v5.0-16k](https://huggingface.co/datasets/chikingsley/l2-arctic-manual-v5.0-16k), '
                           'revision b7261545bc4e4a02910e9c85c659eb866e00db7a. Thay đổi trong dự án: ánh xạ nhãn raw, '
                           'cắt WAV 0,5 giây, tách cả người nói và câu đọc giữa train/validation/test. Gói không kèm audio corpus.')
            native = checkpoint.get('source_metadata', {}).get('native_augmentation')
            if native:
                corpus_text += (' Train còn dùng ' + str(native['files']) + ' WAV của [CMU ARCTIC](http://festvox.org/cmu_arctic/), '
                                'John Kominek và Alan W. Black; speaker: ' + ', '.join(native['speakers']) + '. '
                                'Đã loại mọi prompt dành cho validation/test L2. README/COPYING gốc giữ cùng dữ liệu nguồn.')
    readme = f'''# Nhận diện âm schwa /ə/ trên CPU

Gói có đúng 4 file: `schwa.py`, `best.pt`, `requirements.txt`, `README.md`. Không cần `src`, dataset, notebook hay train. Checkpoint chứa cả bộ trích đặc trưng và bộ phân loại âm vị; không tải model khi chạy inference.

{evaluation}

{context_text}

{annotation_text} Các cửa sổ trong cùng bản ghi có tương quan. Chưa kiểm chứng độ chính xác trên micro/giọng nói của bạn; chỉ số trên corpus không bảo đảm cùng độ chính xác trong mọi môi trường.

## Cài một lần

Mở PowerShell trong thư mục đã giải nén. Dùng Python 3.10 hoặc 3.11:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```

## Inference

```powershell
.\\.venv\\Scripts\\python.exe schwa.py predict --audio "E:\\audio\\recording.wav"
.\\.venv\\Scripts\\python.exe schwa.py predict --audio "E:\\audio\\recording.wav" --output result.json
.\\.venv\\Scripts\\python.exe schwa.py predict --inspect
```

`best.pt` tự nạp từ cùng thư mục với `schwa.py`. WAV được đổi sang mono 16 kHz, quét cửa sổ 0,5 giây với bước 50 ms. WAV ngắn được pad; WAV dài được quét hết. Model trả `detected`, `score`, `threshold`, `windows`; score cao nhất vượt ngưỡng nghĩa là phát hiện schwa. Những cửa sổ trong JSON là vùng dự đoán, chưa phải ranh giới chính xác của âm vị. WAV dài có nhiều cơ hội báo nhầm hơn. Model nhận diện âm /ə/; không bắt từ “yes” hay từ viết “schwa”.

Checkpoint dùng {bundle_name} giữ {layers} lớp transformer và {classifier_text} đã train mới. Model chạy CPU với 2 luồng; weights {weights_mb:.1f} MiB. Chế độ tiền xử lý mặc định dùng WAV gốc.

Thử xuất WAV lọc nhiễu nhẹ:

```powershell
.\\.venv\\Scripts\\python.exe schwa.py denoise --audio input.wav --output cleaned.wav
```

Đây là soft spectral gating, giữ thời lượng. Lọc riêng thay đổi đầu vào; chưa có bảo đảm tăng độ chính xác của checkpoint.

## Nguồn model và dữ liệu

{corpus_text}

Backbone: {backbone_text}; nguồn checkpoint gốc được ghi trong metadata model. Thay đổi: dùng {layers} lớp transformer, {weight_text}; thêm bộ phân loại riêng và đặc trưng phổ/cao độ nếu checkpoint có cấu hình đó. Code inference này được viết mới, không phải sản phẩm chính thức của CMU hay Meta.

Giấy phép backbone được giữ trong README để không cần thư mục `licenses`:

```text
{license_text.strip()}
```
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for filename in ('schwa.py', 'best.pt', 'requirements.txt', 'README.md'):
            archive.write(output / filename, arcname=filename)
    print('Exported:', archive_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metrics', type=Path)
    args = parser.parse_args()
    export(args.model, args.output, args.metrics)

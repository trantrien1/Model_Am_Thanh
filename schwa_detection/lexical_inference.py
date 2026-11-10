"""Pack and run the audio-only lexical Schwa model on CPU, without downloads."""
import argparse
import hashlib
import json
import re
import shutil
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import torch

import schwa as core
from export_inference import definitions
from frozen_phoneme import (RECORDING, aligned_window_features, recording_features,
                            rebuild_acoustic, representation_state)
from improve import PhoneHead
from lexical_context import LEXICAL_RECIPE, ARPABET_IPA, bounded_lexical_features

ROOT = Path(__file__).resolve().parent


def dictionary_from_text(text):
    result = {}
    for line in text.splitlines():
        line = line.partition('#')[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        word = re.sub(r'\(\d+\)$', '', word).upper()
        if phones:
            result.setdefault(word, []).append(phones)
    return result


def pack(model_path, output, backbone=None):
    from lexical_train_cpu import metadata, MAIN_ENCODER
    from lexical_context import ASR_PATH, DICTIONARY_SOURCE
    if output.exists():
        raise ValueError('Preserving existing packed weights; choose a new output path.')
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    if state.get('backend') != 'wav2vec2_lexical_phone_head':
        raise ValueError('Expected a trained lexical Schwa checkpoint.')
    main_encoder = Path(backbone or MAIN_ENCODER)
    if backbone is None and core.digest(main_encoder) != state['lexical']['main_encoder_sha256']:
        main_encoder = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
    if state['lexical'] != metadata(main_encoder):
        raise ValueError('Training resources or feature recipes changed.')
    dictionary_text = (DICTIONARY_SOURCE / 'cmudict.dict').read_text(encoding='utf-8')
    vocabulary_text = (ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json').read_text(encoding='utf-8')
    state['backbone'] = representation_state(torch.load(main_encoder, weights_only=True, mmap=True), 'joint')
    state['lexical_resources'] = {
        'word_asr': torch.load(ASR_PATH, weights_only=True, mmap=True),
        'dictionary_text': dictionary_text, 'vocabulary_text': vocabulary_text,
        'licenses': {
            'wav2vec2': (ROOT / 'data/pretrained/LICENSE_WAV2VEC2.txt').read_text(encoding='utf-8'),
            'phoneme_model': (ROOT / 'data/pretrained/phoneme_xlsr53/LICENSE_APACHE_2.txt').read_text(encoding='utf-8'),
            'cmudict': (DICTIONARY_SOURCE / 'LICENSE').read_text(encoding='utf-8'),
        },
    }
    core.save_torch(output, state)
    print('Self-contained lexical model:', output, output.stat().st_size, 'bytes', flush=True)


class LexicalDetector:
    def __init__(self, model_path):
        self.path = Path(model_path).resolve()
        self.checkpoint = torch.load(self.path, map_location='cpu', weights_only=True, mmap=True)
        state = self.checkpoint
        if state.get('backend') != 'wav2vec2_lexical_phone_head' or state.get('target_phones') != ['ax']:
            raise ValueError('Expected a lexical Schwa checkpoint targeting AX.')
        if 'backbone' not in state or 'lexical_resources' not in state:
            raise ValueError('Checkpoint is not self-contained; pack it first.')
        recipe = state['lexical']
        if recipe['acoustic_recipe'] != RECORDING or recipe['lexical_recipe'] != LEXICAL_RECIPE:
            raise ValueError('Runtime feature recipes differ from training.')
        if recipe['reference_transcript_consumed'] or recipe['manual_phones_consumed_as_features']:
            raise ValueError('This runtime accepts audio-only model features.')
        if state['threshold_selection_split'] != 'validation':
            raise ValueError('Threshold must have been selected on validation.')
        if state['settings']['feature_context'] != 'recording' or state['settings']['acoustic_representation'] != 'joint':
            raise ValueError('Expected original recording context and joint acoustic features.')
        resources = state['lexical_resources']
        for field, expected in [('dictionary_text', 'dictionary_sha256'), ('vocabulary_text', 'vocabulary_sha256')]:
            if hashlib.sha256(resources[field].encode('utf-8')).hexdigest() != recipe[expected]:
                raise ValueError('Embedded lexical resource differs from training: ' + field)
        torch.set_num_threads(state['cpu_threads'])
        self.feature_threads = recipe.get('feature_compute_threads',
                              {'phoneme': state['cpu_threads'], 'word_asr': state['cpu_threads'], 'alignment': state['cpu_threads']})
        if set(self.feature_threads) != {'phoneme', 'word_asr', 'alignment'} or any(value <= 0 for value in self.feature_threads.values()):
            raise ValueError('Invalid feature thread recipe.')
        self.acoustic = rebuild_acoustic(representation_state(state['backbone'], 'joint'))
        self.asr = rebuild_acoustic(resources['word_asr'])
        self.layers = state['backbone']['layers']
        self.asr_layers = resources['word_asr']['layers']
        self.asr_labels = resources['word_asr']['labels']
        self.dictionary = dictionary_from_text(resources['dictionary_text'])
        self.vocabulary = json.loads(resources['vocabulary_text'])
        self.head = PhoneHead(**state['head_config']).eval()
        self.head.load_state_dict(state['head_state_dict'])
        self.ax = state['phone_vocabulary'].index('ax')
        self.embedding_dim = state['backbone']['model_config']['encoder_embed_dim']
        self.phone_dim = state['backbone']['model_config']['aux_num_out']
        if state['head_config']['input_dim'] != self.embedding_dim + self.phone_dim + LEXICAL_RECIPE['features']:
            raise ValueError('Classifier and encoder feature dimensions differ.')

    def describe(self):
        state = self.checkpoint
        return {'weights': str(self.path), 'sha256': core.digest(self.path), 'backend': state['backend'],
                'target_symbol': state['target_symbol'], 'target_phones': state['target_phones'],
                'epoch': state['epoch'], 'threshold': state['threshold'], 'pool_frames': state['pool_frames'],
                'device': 'cpu', 'self_contained': True, 'downloads_required': False,
                'feature_context': 'recording', 'window_seconds': .5, 'denoise': 'off',
                'feature_compute_threads': self.feature_threads,
                'reference_transcript_consumed': False, 'manual_phones_consumed_as_features': False}

    @torch.inference_mode()
    def sequence(self, waveform):
        audio = torch.as_tensor(waveform, dtype=torch.float32)
        torch.set_num_threads(self.feature_threads['phoneme'])
        acoustic = recording_features(self.acoustic, audio, self.layers)
        torch.set_num_threads(self.feature_threads['word_asr'])
        asr = recording_features(self.asr, audio, self.asr_layers).float()
        torch.set_num_threads(self.feature_threads['alignment'])
        lexical, _ = bounded_lexical_features(asr, acoustic[:, self.embedding_dim:].float(),
                                             self.dictionary, self.asr_labels, self.vocabulary)
        # Recording and window rounding must match both training caches exactly.
        lexical = lexical.half().float()
        torch.set_num_threads(self.checkpoint['cpu_threads'])
        return acoustic, lexical

    @torch.inference_mode()
    def scores(self, sequence, starts, batch_size=64):
        if batch_size <= 0:
            raise ValueError('Batch size must be positive.')
        torch.set_num_threads(self.checkpoint['cpu_threads'])
        acoustic, lexical = sequence
        output = []
        for index in range(0, len(starts), batch_size):
            subset = starts[index:index + batch_size]
            values = torch.cat([aligned_window_features(acoustic, subset),
                                aligned_window_features(lexical, subset)], dim=-1)
            scores = self.head(values).softmax(-1)[:, :, self.ax]
            pool = self.checkpoint['pool_frames']
            if pool > 1:
                scores = torch.nn.functional.avg_pool1d(scores.unsqueeze(1), pool, stride=1).squeeze(1)
            output.extend(scores.amax(-1).tolist())
        return output

    def predict(self, path, hop_seconds=.05):
        if not 0 < hop_seconds <= .5:
            raise ValueError('Hop must be in (0, 0.5] seconds.')
        audio, metadata = core.load_audio(path)
        hop = max(1, round(hop_seconds * 16000))
        starts = list(range(0, max(1, len(audio) - 8000 + 1), hop))
        last = max(0, len(audio) - 8000)
        if starts[-1] != last:
            starts.append(last)
        started = time.perf_counter()
        scores = self.scores(self.sequence(audio), starts)
        threshold = self.checkpoint['threshold']
        windows = [{'start_seconds': start / 16000,
                    'end_seconds': min(len(audio), start + 8000) / 16000, 'score': float(score)}
                   for start, score in zip(starts, scores) if score >= threshold]
        return {'audio': str(Path(path).resolve()), 'target_symbol': '\u0259', 'score': max(scores),
                'threshold': threshold, 'detected': bool(windows), 'windows': windows,
                'duration_seconds': metadata['duration_seconds'], 'denoise': 'off',
                'scanned_windows': len(starts), 'feature_context': 'recording',
                'processing_seconds': time.perf_counter() - started,
                'note': 'Window scores detect AX; regions are approximate, not exact phone boundaries.'}


def predict(args):
    detector = LexicalDetector(args.model)
    if args.inspect:
        print(json.dumps(detector.describe(), ensure_ascii=False, indent=2))
        return
    if not args.audio:
        raise ValueError('Provide --audio or --inspect.')
    results = [detector.predict(path, args.hop) for path in args.audio]
    if args.output:
        core.write_json(args.output, results[0] if len(results) == 1 else results)
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


def evaluation_identity(model_path, directory, state):
    dataset_sha = core.digest(directory / 'data/manifests/dataset.csv')
    if state['dataset_sha256'] != dataset_sha:
        raise ValueError('Evaluation dataset differs from training.')
    backend = state.get('backend')
    if backend not in ('wav2vec2_lexical_phone_head', 'wav2vec2_phone_head', 'wavlm_phone_head', 'wav2vec2_recording_phone_head', 'wav2vec2_whisper_lexical_head'):
        raise ValueError('Unsupported runtime evaluation backend.')
    names = ['lexical_inference.py', 'frozen_phoneme.py', 'improve.py', 'schwa.py']
    if backend == 'wav2vec2_lexical_phone_head':
        names.append('lexical_context.py')
    elif backend == 'wav2vec2_phone_head':
        names.append('recording_runtime.py')
    elif backend == 'wavlm_phone_head':
        names += ['wavlm_inference.py', 'frozen_wavlm.py', 'wavlm_train_cpu.py']
    elif backend == 'wav2vec2_whisper_lexical_head':
        names += ['whisper_inference.py', 'whisper_lexical.py', 'whisper_cpu.py', 'lexical_context.py']
    else:
        names += ['recording_head_inference.py', 'recording_head_cpu.py']
    result = {'model_sha256': core.digest(model_path), 'dataset_sha256': dataset_sha,
            'threshold': state['threshold'], 'pool_frames': state['pool_frames'],
            'threshold_selection_split': state['threshold_selection_split'],
            'acoustic_runtime_recipe': RECORDING,
            'lexical_recipe': LEXICAL_RECIPE if backend == 'wav2vec2_lexical_phone_head' else None,
            'inference_code_sha256': {name: core.digest(ROOT / name) for name in names}}
    if backend == 'wavlm_phone_head':
        result.update(wavlm_recipe=state['settings']['wavlm_recipe'],
                      recording_normalization=state['settings']['recording_normalization'])
    elif backend == 'wav2vec2_recording_phone_head':
        from recording_head_inference import RUNTIME_RECIPE
        result.update(recording_head_recipe=RUNTIME_RECIPE,
                      training_supplement_manifest_sha256=state['provenance']['supplement_manifest_sha256'])
    elif backend == 'wav2vec2_whisper_lexical_head':
        from whisper_inference import RUNTIME_RECIPE
        result.update(whisper_runtime_recipe=RUNTIME_RECIPE,
                      whisper_package_sha256=state['lexical']['package_files_sha256'])
    return result


def freeze(model_path, output):
    from improve import activate_dataset
    if output.exists():
        raise ValueError('Preserving the frozen selection report.')
    state = torch.load(model_path, weights_only=True, mmap=True)
    backend = state.get('backend')
    if backend == 'wav2vec2_lexical_phone_head' and 'lexical_resources' in state:
        provenance = state['provenance']
    elif backend == 'wav2vec2_phone_head' and 'backbone' in state and state['settings'].get('feature_context') == 'recording':
        provenance = state['backbone']['fine_tuning']
    elif backend in ('wavlm_phone_head', 'wav2vec2_recording_phone_head', 'wav2vec2_whisper_lexical_head') and 'backbone' in state and state['settings'].get('feature_context') == 'recording':
        provenance = state['provenance']
    else:
        raise ValueError('Freeze a packed recording-context model before evaluating test.')
    if state['threshold_selection_split'] != 'validation' or provenance['test_used_for_selection']:
        raise ValueError('Selection must use validation only.')
    directory = activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    test_rows = [row for row in rows if row['split'] == 'test']
    identity = evaluation_identity(model_path, directory, state)
    report = {**identity, 'dataset_version': 'v5', 'selected_epoch': state['epoch'],
              'validation_metrics': state['validation_metrics'],
              'test_region_ids_sha256': core.object_digest([row['sample_id'] for row in test_rows]),
              'test_regions': len(test_rows), 'test_recordings': len({row['recording_id'] for row in test_rows}),
              'test_used_for_selection': False,
              'selection_rule': 'Validation min(precision, recall), then F1, accuracy, lower FPR.',
              'created_utc': datetime.now(timezone.utc).isoformat()}
    core.write_json(output, report)
    print('Frozen model and threshold:', output, flush=True)


def evaluate(model_path, output, split='validation', frozen_path=None):
    import numpy as np
    from improve import activate_dataset
    if split not in ('train', 'validation', 'test'):
        raise ValueError('Unknown evaluation split.')
    if (output / 'metrics.json').exists():
        raise ValueError('Evaluation already completed; preserving frozen results.')
    directory = activate_dataset('v5')
    state = torch.load(model_path, weights_only=True, mmap=True)
    identity = evaluation_identity(model_path, directory, state)
    rows = [row for row in core.audit_dataset(verify_wavs=False) if row['split'] == split]
    if split == 'test':
        if frozen_path is None:
            raise ValueError('Provide the model selection report frozen before test.')
        frozen = core.read_json(frozen_path)
        if any(frozen[key] != value for key, value in identity.items()):
            raise ValueError('Model, threshold, feature recipe or inference source differs from the frozen selection.')
        if (frozen['test_region_ids_sha256'] != core.object_digest([row['sample_id'] for row in rows])
                or frozen['test_regions'] != len(rows) or frozen['test_used_for_selection']):
            raise ValueError('Frozen test region selection differs.')
    groups = {}
    for index, row in enumerate(rows):
        groups.setdefault(row['recording_id'], []).append((index, row))
    recordings = {row['recording_id']: row for row in core.read_csv(directory / 'data/manifests/recordings.csv')}
    if state['backend'] == 'wav2vec2_lexical_phone_head':
        detector = LexicalDetector(model_path)
    elif state['backend'] == 'wav2vec2_phone_head':
        from recording_runtime import RecordingPhoneDetector
        detector = RecordingPhoneDetector(model_path)
    elif state['backend'] == 'wavlm_phone_head':
        from wavlm_inference import WavLMDetector
        detector = WavLMDetector(model_path)
    elif state['backend'] == 'wav2vec2_whisper_lexical_head':
        from whisper_inference import WhisperLexicalDetector
        detector = WhisperLexicalDetector(model_path)
    else:
        from recording_head_inference import RecordingHeadDetector
        detector = RecordingHeadDetector(model_path)
    scores = np.full(len(rows), np.nan)
    started = time.perf_counter()
    for number, (record_id, regions) in enumerate(sorted(groups.items()), 1):
        record = recordings[record_id]
        path = directory / record['audio_path']
        if record['split'] != split or core.digest(path) != record['audio_sha256']:
            raise ValueError('Evaluation source recording differs from the audited split.')
        starts = [int(row['window_start_sample']) for _, row in regions]
        record_identity = {**identity, 'split': split, 'recording_id': record_id,
                           'audio_sha256': record['audio_sha256'], 'starts': starts,
                           'sample_ids': [row['sample_id'] for _, row in regions]}
        cached_path = output / 'recording_scores' / (core.object_digest(record_identity) + '.json')
        if cached_path.exists():
            cached = core.read_json(cached_path)
            if cached['identity'] != record_identity:
                raise ValueError('Cached recording prediction identity changed.')
            values = cached['scores']
            if cached['scores_sha256'] != core.object_digest(values):
                raise ValueError('Cached recording scores changed.')
        else:
            audio, _ = core.load_audio(path)
            values = detector.scores(detector.sequence(audio), starts)
            core.write_json(cached_path, {'identity': record_identity, 'scores': values,
                                         'scores_sha256': core.object_digest(values)})
        if len(values) != len(regions) or not np.isfinite(values).all():
            raise ValueError('Incomplete or nonfinite recording predictions.')
        scores[[index for index, _ in regions]] = values
        if number % 16 == 0 or number == len(groups):
            print(f'Runtime {split}: {number}/{len(groups)} recordings, {time.perf_counter()-started:.1f}s', flush=True)
    labels = np.asarray([int(row['label']) for row in rows])
    overall = core.binary_metrics(labels, scores, state['threshold'])
    predictions = [{**row, 'score': float(score), 'prediction': int(score >= state['threshold'])}
                   for row, score in zip(rows, scores)]
    core.write_csv(output / 'predictions.csv', predictions)
    for name, label, prediction in [('false_positives', '0', 1), ('false_negatives', '1', 0)]:
        core.write_csv(output / (name + '.csv'), [row for row in predictions if row['label'] == label
                       and row['prediction'] == prediction], list(predictions[0]))
    grouped_metrics = {}
    for field in ('speaker_id', 'center_phone'):
        grouped_metrics[field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = [index for index, row in enumerate(rows) if row[field] == value]
            grouped_metrics[field][value] = core.binary_metrics(labels[indices], scores[indices], state['threshold'])
    report = {**identity, 'split': split, 'dataset_version': 'v5', 'overall': overall, 'groups': grouped_metrics,
              'threshold_frozen': True, 'feature_context': 'recording', 'source_recordings': len(groups),
              'evaluation_input': 'Original recordings; unchanged 0.5-second focal regions.',
              'speakers': sorted({row['speaker_id'] for row in rows}),
              'always_negative_accuracy': float(np.mean(labels == 0)),
              'reference_transcript_consumed': False, 'manual_phones_consumed_as_features': False,
              'frozen_selection_sha256': core.digest(frozen_path) if frozen_path else None,
              'seconds': time.perf_counter()-started}
    core.write_json(output / 'metrics.json', report)
    print(json.dumps(overall, indent=2), flush=True)


def standalone_source():
    header = '''"""Offline CPU Schwa inference using audio-only lexical context."""
import argparse
import hashlib
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
from torch import nn
from scipy.ndimage import uniform_filter
from scipy.signal import istft, resample_poly, stft
ROOT = Path(__file__).resolve().parent
'''
    header += 'RECORDING = ' + repr(RECORDING) + '\n'
    header += 'LEXICAL_RECIPE = ' + repr(LEXICAL_RECIPE) + '\n'
    header += 'ARPABET_IPA = ' + repr(ARPABET_IPA) + '\n'
    header += 'DENOISE = ' + repr(core.DENOISE) + '\n'
    body = definitions(ROOT / 'schwa.py', {'digest', 'write_json', 'load_audio', 'crop_window', 'denoise_audio', 'denoise_cli'})
    body += definitions(ROOT / 'frozen_phoneme.py', {'Float16StoredLinear', 'swap_stored_linears', 'swap_dynamic_linears',
             'rebuild_acoustic', 'representation_state', 'encoder_prefix', 'rounded_encoder_embedding',
             'acoustic_features', 'recording_features', 'aligned_window_features'})
    body += definitions(ROOT / 'improve.py', {'GroupFeatureNorm', 'PhoneHead'})
    body += definitions(ROOT / 'lexical_context.py', {'pronunciation_ids', 'greedy_word_spans', 'spelling_edits',
             'word_candidates', 'ctc_state_posteriors', 'lexical_chunk', 'bounded_lexical_features'})
    body += definitions(Path(__file__), {'dictionary_from_text', 'LexicalDetector', 'predict'})
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
        raise ValueError('Preserving existing inference folder; choose an empty directory.')
    archive = output.with_suffix('.zip')
    if archive.exists():
        raise ValueError('Preserving existing archive; choose a new output path.')
    state = torch.load(model_path, map_location='cpu', weights_only=True, mmap=True)
    if state.get('backend') != 'wav2vec2_lexical_phone_head' or 'lexical_resources' not in state or 'backbone' not in state:
        raise ValueError('Export requires a packed lexical checkpoint.')
    model_sha = core.digest(model_path)
    evaluation = 'Gói thử nghiệm; chưa có báo cáo test đi kèm. Không dùng kết quả kỹ thuật để kết luận độ chính xác.'
    if metrics_path:
        report = core.read_json(metrics_path)
        if (report['split'] != 'test' or report['model_sha256'] != model_sha
                or report['dataset_sha256'] != state['dataset_sha256']
                or report['acoustic_runtime_recipe'] != RECORDING
                or report['lexical_recipe'] != LEXICAL_RECIPE
                or report['threshold_selection_split'] != 'validation'
                or report['overall']['threshold'] != state['threshold']):
            raise ValueError('Test report does not describe these frozen weights and runtime.')
        metric = report['overall']
        evaluation = (f"Test: {metric['files']} vùng 0,5 giây trên {report['source_recordings']} bản ghi gốc; "
                      f"precision {metric['precision']:.2%}, recall {metric['recall']:.2%}, accuracy {metric['accuracy']:.2%}. "
                      'Ngưỡng được chọn trên validation, giữ nguyên khi chấm test.')
    source = standalone_source()
    compile(source, 'schwa.py', 'exec')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'schwa.py').write_text(source, encoding='utf-8')
    shutil.copy2(model_path, output / 'best.pt')
    (output / 'requirements.txt').write_text('torch==2.5.1\ntorchaudio==2.5.1\nnumpy==1.26.4\nscipy==1.14.1\nsoundfile==0.13.1\n', encoding='utf-8')
    licenses = '\n\n'.join(name + '\n\n' + text for name, text in state['lexical_resources']['licenses'].items())
    readme = f'''# Nhận diện âm Schwa /ə/ trên CPU

Gói có đúng 4 file. Không cần src, dataset, notebook hay train. best.pt chứa hai encoder, bộ phân loại,
từ điển và ngưỡng; inference không tải model và không cần câu chép lại.

{evaluation}

Cài Python 3.10/3.11 và chạy trong thư mục này:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.\\.venv\\Scripts\\python.exe schwa.py predict --audio "E:\\audio\\recording.wav" --output result.json
.\\.venv\\Scripts\\python.exe schwa.py predict --inspect
```

Model phân tích bản ghi đầy đủ rồi chấm từng vùng 0,5 giây, mặc định bước 50 ms. WAV ngắn được pad,
WAV dài được chia đoạn có ngữ cảnh. WAV được đổi sang mono 16 kHz. JSON có score, threshold, detected và windows.
Các vùng vượt ngưỡng là vị trí dự đoán, chưa phải ranh giới âm vị chính xác. Target là AH0/AX;
AH1/AH2 và ER/AXR là âm khác. Không bắt từ "yes" hoặc từ viết "schwa".

Kết quả vùng trên corpus đọc câu không bảo đảm cùng độ chính xác trên micro, tiếng ồn hoặc cho cả WAV.
Lọc nhiễu riêng bằng `python schwa.py denoise --audio input.wav --output cleaned.wav`;
chưa đo chất lượng model trên WAV đã lọc. Hai encoder làm model chậm hơn phiên bản một encoder.

Nguồn: facebook/wav2vec2-xlsr-53-espeak-cv-ft revision 2c733782da5604684829819a5eb744c193fe9398;
TorchAudio WAV2VEC2_ASR_BASE_960H; CMUdict revision 74790861f652b15e4ac49015a90074ad62a27690.
Train dùng L2-ARCTIC manual v5.0 (CC BY-NC 4.0), bản 16 kHz của chikingsley,
revision b7261545bc4e4a02910e9c85c659eb866e00db7a; nhãn perceived phone và các split V5 của dự án.
Gói không kèm audio corpus. Dùng phi thương mại theo giấy phép nguồn dữ liệu.

SHA-256 best.pt: {model_sha}

## Giấy phép nguồn model và từ điển

```text
{licenses}
```
'''
    (output / 'README.md').write_text(readme, encoding='utf-8')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as handle:
        for name in ['schwa.py', 'best.pt', 'requirements.txt', 'README.md']:
            handle.write(output / name, arcname=name)
    print('Offline inference bundle:', archive, flush=True)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--backbone', type=Path)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=lambda args: pack(args.model, args.output, args.backbone))
    p = commands.add_parser('export')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metrics', type=Path)
    p.set_defaults(function=lambda args: export(args.model, args.output, args.metrics))
    p = commands.add_parser('freeze')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(function=lambda args: freeze(args.model, args.output))
    p = commands.add_parser('evaluate')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'validation', 'test'], default='validation')
    p.add_argument('--frozen-selection', type=Path)
    p.set_defaults(function=lambda args: evaluate(args.model, args.output, args.split, args.frozen_selection))
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

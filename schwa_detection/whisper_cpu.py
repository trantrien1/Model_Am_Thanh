"""Pinned public Whisper Base English CPU preparation and TRAIN-only smoke check."""
import argparse
import importlib.metadata
import time
from pathlib import Path

import torch

import schwa as core

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'data/pretrained/whisper/base.en.pt'
MODEL_SHA = '25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead'
MODEL_URL = 'https://openaipublic.azureedge.net/main/whisper/models/' + MODEL_SHA + '/base.en.pt'
VERSION = '20250625'
CPU_MATH = {'threads': 2, 'mkldnn': False, 'deterministic_algorithms': True}
DECODE = {'language': 'en', 'task': 'transcribe', 'fp16': False, 'temperature': 0.,
          'beam_size': 5, 'condition_on_previous_text': False, 'word_timestamps': True,
          'initial_prompt': None, 'verbose': False}


def package_identity():
    import whisper
    if importlib.metadata.version('openai-whisper') != VERSION:
        raise ValueError('Use the pinned Whisper package version.')
    package = Path(whisper.__file__).resolve().parent
    files = ['__init__.py', 'audio.py', 'decoding.py', 'model.py', 'timing.py', 'tokenizer.py', 'transcribe.py',
             'assets/gpt2.tiktoken', 'assets/mel_filters.npz']
    return {name: core.digest(package / name) for name in files}


def load_model():
    import whisper
    package_identity()
    if not MODEL.exists() or core.digest(MODEL) != MODEL_SHA:
        raise ValueError('Prepare the checksum-verified public Whisper model first.')
    if whisper._MODELS['base.en'] != MODEL_URL:
        raise ValueError('Pinned Whisper source URL differs.')
    return whisper.load_model('base.en', device='cpu', download_root=str(MODEL.parent))


def prepare(smoke):
    import whisper
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    identity = package_identity()
    core.fetch(MODEL_URL, MODEL, timeout=60)
    if core.digest(MODEL) != MODEL_SHA:
        raise ValueError('Pinned public Whisper checksum differs.')
    distribution = importlib.metadata.distribution('openai-whisper')
    license_file = next((distribution.locate_file(name) for name in distribution.files or [] if str(name).upper().endswith('/LICENSE')), None)
    if license_file is not None:
        license_text = Path(license_file).read_text(encoding='utf-8')
        (MODEL.parent / 'LICENSE_WHISPER.txt').write_text(license_text, encoding='utf-8')
    else:
        core.fetch('https://raw.githubusercontent.com/openai/whisper/main/LICENSE', MODEL.parent / 'LICENSE_WHISPER.txt', timeout=30)
    report = {'public_source_url': MODEL_URL, 'model_sha256': MODEL_SHA, 'bytes': MODEL.stat().st_size,
        'package': 'openai-whisper', 'package_version': VERSION, 'package_files_sha256': identity,
        'license': 'MIT', 'license_sha256': core.digest(MODEL.parent / 'LICENSE_WHISPER.txt'),
        'decode_recipe': DECODE, 'cpu_math': CPU_MATH, 'device': 'cpu', 'cpu_threads': 2,
        'new_schwa_classifier_trained': False, 'schwa_quality_evidence': False, 'test_predictions_computed': False}
    core.write_json(MODEL.with_suffix('.json'), report)
    print('Prepared checksum-verified Whisper Base English:', MODEL.stat().st_size, 'bytes', flush=True)
    if smoke:
        rows = [r for r in core.read_csv(ROOT / 'v5/data/manifests/dataset.csv') if r['split'] == 'train']
        ids = {r['recording_id'] for r in rows}
        record = next(r for r in core.read_csv(ROOT / 'v5/data/manifests/recordings.csv') if r['recording_id'] in ids)
        source = ROOT / 'v5' / record['audio_path']
        if core.digest(source) != record['audio_sha256']:
            raise ValueError('TRAIN smoke input changed.')
        audio, _ = core.load_audio(source)
        model = load_model()
        started = time.perf_counter()
        with torch.backends.mkldnn.flags(enabled=False):
            result = model.transcribe(audio, **DECODE)
        words = [word for segment in result['segments'] for word in segment.get('words', [])]
        if not result['text'].strip() or any(not 0 <= w['start'] <= w['end'] <= len(audio)/16000 + .1 for w in words):
            raise ValueError('Whisper CPU smoke returned empty text or invalid word timings.')
        report.update(passed=True, source_split='train', recording_id=record['recording_id'],
            audio_seconds=len(audio)/16000, inference_seconds=time.perf_counter()-started,
            recognized_text=result['text'], recognized_words=words, reference_transcript_consumed=False)
        core.write_json(ROOT / 'reports/whisper_base_en_cpu_smoke.json', report)
        print({'passed': True, 'seconds': report['inference_seconds'], 'recognized_text': result['text'],
               'word_count': len(words), 'schwa_quality_evidence': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    prepare(args.smoke)

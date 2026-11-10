"""Official WavLM Base+ CPU encoder, isolated from running XLSR experiments."""
import argparse
import gc
import time
from pathlib import Path

import torch
import torchaudio
from torch import nn

import schwa as core
from frozen_phoneme import RECORDING, swap_stored_linears

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'data/pretrained/wavlm_base_plus_fp16_linear.pt'
WAVLM_RECIPE = {**RECORDING, 'feature_layers': [6, 9, 12],
                'combination': 'arithmetic_mean_before_float16_recording_storage',
                'feature_dimension': 768, 'model_family': 'wavlm_base_plus',
                'normalize_waveform': False, 'compute_threads': 2}


def rebuild_wavlm(state):
    if state.get('model_family') != 'wavlm_base_plus' or state.get('quantization') != 'float16_linear_storage':
        raise ValueError('Not the prepared public WavLM Base+ encoder.')
    with torch.device('meta'):
        model = torchaudio.models.wavlm_model(**state['model_config'])
    swap_stored_linears(model)
    model.load_state_dict(state['model_state_dict'], assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Incomplete WavLM model weights.')
    return model.eval()


def recording_embeddings(model, waveform):
    if waveform.ndim != 1 or len(waveform) == 0:
        raise ValueError('WavLM requires nonempty mono audio.')
    if len(waveform) < RECORDING['minimum_samples']:
        waveform = nn.functional.pad(waveform, (0, RECORDING['minimum_samples'] - len(waveform)))
    def encode(audio):
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            layers = model.extract_features(audio.unsqueeze(0), num_layers=12)[0]
            values = torch.stack([layers[index - 1][0] for index in WAVLM_RECIPE['feature_layers']]).mean(0)
        return values.half()
    if len(waveform) <= RECORDING['maximum_samples']:
        return encode(waveform)
    frames = (len(waveform) - 400) // RECORDING['hop_samples'] + 1
    centers = RECORDING['center_sample'] + torch.arange(frames) * RECORDING['hop_samples']
    result = torch.empty(frames, WAVLM_RECIPE['feature_dimension'], dtype=torch.float16)
    for start in range(0, len(waveform), RECORDING['core_samples']):
        end = min(len(waveform), start + RECORDING['core_samples'])
        indices = torch.nonzero((centers >= start) & (centers < end)).flatten()
        if not len(indices):
            continue
        input_start, input_end = max(0, start - RECORDING['margin_samples']), min(len(waveform), end + RECORDING['margin_samples'])
        local = indices - input_start // RECORDING['hop_samples']
        values = encode(waveform[input_start:input_end])
        if torch.any(local < 0) or torch.any(local >= len(values)):
            raise ValueError('WavLM chunk frame indices do not match the recording.')
        result[indices] = values[local]
    return result


def prepare(args):
    if args.output.exists():
        raise ValueError('Preserving an existing prepared WavLM encoder.')
    bundle = torchaudio.pipelines.WAVLM_BASE_PLUS
    source = ROOT / 'data/pretrained/torch_hub/checkpoints' / bundle._path
    url = 'https://download.pytorch.org/torchaudio/models/' + bundle._path
    if not source.exists():
        core.fetch(url, source, timeout=60)
    source_sha = core.digest(source)
    original = torch.load(source, weights_only=True, mmap=True)
    with torch.device('meta'):
        skeleton = torchaudio.models.wavlm_model(**bundle._params)
    names = {name for name, module in skeleton.named_modules() if isinstance(module, nn.Linear)}
    weights = {name: values.half() if name.rsplit('.', 1)[0] in names else values
               for name, values in original.items()}
    # MHA in_proj_weight is used directly by functional.linear, so remains float32.
    state = {'model_family': 'wavlm_base_plus', 'model_config': dict(bundle._params),
             'model_state_dict': weights, 'source_bundle': 'torchaudio.pipelines.WAVLM_BASE_PLUS',
             'source_url': url, 'source_checkpoint_sha256': source_sha,
             'license': 'MIT', 'layers': 12, 'sample_rate': 16000,
             'quantization': 'float16_linear_storage', 'feature_kind': 'wavlm_layer_mean',
             'feature_dimension': 768, 'recipe': WAVLM_RECIPE,
             'normalize_waveform': bundle._normalize_waveform,
             'weight_storage': 'float16 nn.Linear; float32 convolution/norm/MHA in_proj',
             'computation': 'CPU float32 math; MATH scaled dot-product attention'}
    core.save_torch(args.output, state)
    core.write_json(args.output.with_suffix('.json'), {key: value for key, value in state.items() if key != 'model_state_dict'} | {
                    'prepared_sha256': core.digest(args.output), 'bytes': args.output.stat().st_size,
                    'source_bytes': source.stat().st_size})
    del skeleton, original, weights, state
    gc.collect()
    print('Prepared WavLM:', args.output, args.output.stat().st_size, 'bytes', flush=True)
    if args.smoke:
        state = torch.load(args.output, weights_only=True, mmap=True)
        model = rebuild_wavlm(state)
        record = next(r for r in core.read_csv(ROOT / 'v5/data/manifests/recordings.csv') if r['split'] == 'train')
        path = ROOT / 'v5' / record['audio_path']
        if core.digest(path) != record['audio_sha256']:
            raise ValueError('TRAIN source audio differs.')
        audio, _ = core.load_audio(path)
        started = time.perf_counter()
        with torch.inference_mode():
            values = recording_embeddings(model, torch.from_numpy(audio))
        if values.ndim != 2 or values.shape[1] != 768 or not torch.isfinite(values).all():
            raise ValueError('WavLM actual CPU smoke produced invalid embeddings.')
        report = {'passed': True, 'quality_evidence': False, 'source_split': 'train',
                  'recording_id': record['recording_id'], 'audio_seconds': len(audio) / 16000,
                  'shape': list(values.shape), 'encoder_seconds': time.perf_counter() - started,
                  'prepared_sha256': core.digest(args.output), 'test_predictions_computed': False}
        core.write_json(ROOT / 'reports/wavlm_base_plus_cpu_smoke.json', report)
        print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    prepare(args)

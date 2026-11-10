"""Prepare the official phoneme CTC encoder with bounded-memory CPU storage."""
import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torchaudio
from torch import nn

import schwa as core

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/pretrained/phoneme_xlsr53'
OUTPUT = ROOT / 'data/pretrained/phoneme_xlsr53_int8.pt'
RECORDING = {'version': 2, 'attention_backend': 'math', 'minimum_samples': 8000, 'maximum_samples': 160000,
             'core_samples': 96000, 'margin_samples': 32000, 'hop_samples': 320,
             'center_sample': 199.5, 'window_frames': 24}


def ensure_source(family='xlsr53'):
    repo = 'facebook/wav2vec2-xlsr-53-espeak-cv-ft'
    revision = '2c733782da5604684829819a5eb744c193fe9398'
    expected = {'config.json': '4609fb49b7e1d28aecb2840da1926c40bd915bc6f1120a940afacf7159bbfb13',
                'preprocessor_config.json': 'a2254a5b58f72cd4de3632f8eee64f3f098b7c1402128d2f419e7d00ae13e335',
                'vocab.json': 'd732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0',
                'README.md': '32a058c9a338807e9e237fad17a9f400f67c64ee2e18d71750c912ffbfb9e22b',
                'pytorch_model.bin': '04366b6c8d24099ef313cf02f0e58d26f5dddfda16edbfc8eb2c713d94a9f551'}
    source_directory = SOURCE
    if family == 'lv60':
        repo = 'facebook/wav2vec2-lv-60-espeak-cv-ft'
        revision = 'ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4'
        source_directory = ROOT / 'data/pretrained/phoneme_lv60'
        expected['README.md'] = '81a55816a87034bcf0828d6181fa88abad32a0cbfcebc2ebc8b336664b69ba60'
        expected['pytorch_model.bin'] = '3173bde9e9ce490fa0f989e413c42f25bc1820c020adc1e6b9b87025b3cfcc5e'
    elif family != 'xlsr53':
        raise ValueError('Unsupported public phoneme encoder family.')
    for name, fingerprint in expected.items():
        metadata = core.fetch('https://huggingface.co/' + repo + '/resolve/' + revision + '/' + name,
                              source_directory / name, timeout=60)
        if metadata['sha256'] != fingerprint:
            raise ValueError('Pinned phoneme source fingerprint differs: ' + name)
    description = {'repo': repo, 'revision': revision, 'license': 'Apache-2.0',
                   'model_sha256': expected['pytorch_model.bin'], 'sampling_rate': 16000}
    source_path = source_directory / 'source.json'
    if source_path.exists() and core.read_json(source_path) != description:
        raise ValueError('Existing pretrained source metadata differs.')
    core.write_json(source_path, description)
    core.fetch('https://www.apache.org/licenses/LICENSE-2.0.txt', source_directory / 'LICENSE_APACHE_2.txt')
    return source_directory


def swap_dynamic_linears(model):
    """Construct quantized modules without allocating an entire float32 LARGE model."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            replacement = torch.ao.nn.quantized.dynamic.Linear(
                module.in_features, module.out_features, module.bias is not None, dtype=torch.qint8)
            setattr(model, name, replacement)
        else:
            swap_dynamic_linears(module)


class Float16StoredLinear(nn.Linear):
    """Keep linear weights in half precision, casting one layer for float32 math."""
    def forward(self, inputs):
        bias = self.bias.float() if self.bias is not None else None
        return nn.functional.linear(inputs, self.weight.float(), bias)


def swap_stored_linears(model):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            with torch.device('meta'):
                replacement = Float16StoredLinear(module.in_features, module.out_features,
                                                  module.bias is not None, dtype=torch.float16)
            setattr(model, name, replacement)
        else:
            swap_stored_linears(module)


def rebuild_acoustic(state):
    assigned = False
    if state.get('quantization') == 'dynamic_int8_linears':
        with torch.device('meta'):
            model = torchaudio.models.wav2vec2_model(**state['model_config'])
        swap_dynamic_linears(model)
        model.to_empty(device='cpu')
    elif state.get('quantization') == 'float16_linear_storage':
        with torch.device('meta'):
            model = torchaudio.models.wav2vec2_model(**state['model_config'])
        swap_stored_linears(model)
        assigned = True
    elif state.get('quantization'):
        raise ValueError('Unsupported encoder quantization.')
    else:
        model = torchaudio.models.wav2vec2_model(**state['model_config'])
    model.load_state_dict(state['model_state_dict'], assign=assigned)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Incomplete acoustic encoder state.')
    model.normalize_waveform = state.get('normalize_waveform', False)
    model.feature_kind = state.get('feature_kind', 'embeddings')
    model.quantization = state.get('quantization')
    model.prefix_rounding_layer = state.get('prefix_rounding_layer', 0)
    return model.eval()


def representation_state(state, representation='stored'):
    if representation == 'stored':
        return state
    if representation != 'joint' or state.get('feature_kind') not in ('ctc_logits', 'ctc_embeddings_logits'):
        raise ValueError('Joint representation requires a phoneme CTC encoder.')
    result = dict(state)
    result['feature_kind'] = 'ctc_embeddings_logits'
    result['feature_dimension'] = state['model_config']['encoder_embed_dim'] + state['model_config']['aux_num_out']
    return result


def encoder_prefix(model, inputs, layers):
    """Frozen transformer prefix, without applying the final encoder norm."""
    if not 0 < layers <= len(model.encoder.transformer.layers):
        raise ValueError('Invalid transformer prefix depth.')
    features, lengths = model.feature_extractor(inputs, None)
    features, mask = model.encoder._preprocess(features, lengths)
    features = model.encoder.transformer._preprocess(features)
    for layer in model.encoder.transformer.layers[:layers]:
        features, _ = layer(features, attention_mask=mask)
    return features


def rounded_encoder_embedding(model, inputs):
    prefix_layers = model.prefix_rounding_layer
    features = encoder_prefix(model, inputs, prefix_layers).half().float()
    for layer in model.encoder.transformer.layers[prefix_layers:]:
        features, _ = layer(features)
    if not model.encoder.transformer.layer_norm_first:
        features = model.encoder.transformer.layer_norm(features)
    return features


def acoustic_features(model, waveforms, layers):
    if model.feature_kind in ('ctc_logits', 'ctc_embeddings_logits'):
        # Each waveform has independent activation scales; math attention is stable
        # across feature extraction and standalone inference on this CPU runtime.
        values = []
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for waveform in waveforms:
                inputs = waveform.unsqueeze(0)
                if getattr(model, 'prefix_rounding_layer', 0):
                    embedding = rounded_encoder_embedding(model, inputs)
                    logits = model.aux(embedding)
                    values.append(logits if model.feature_kind == 'ctc_logits' else torch.cat([embedding, logits], dim=-1))
                elif model.feature_kind == 'ctc_logits':
                    values.append(model(inputs)[0])
                else:
                    features, lengths = model.feature_extractor(inputs, None)
                    embedding = model.encoder(features, lengths)
                    values.append(torch.cat([embedding, model.aux(embedding)], dim=-1))
        return torch.cat(values, dim=0)
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


def prepare(args):
    if args.output.exists():
        raise ValueError('Preserving existing encoder; choose a new output path.')
    torch.set_num_threads(args.threads)
    source_directory = ensure_source(getattr(args, 'family', 'xlsr53'))
    source = core.read_json(source_directory / 'source.json')
    path = source_directory / 'pytorch_model.bin'
    if core.digest(path) != source['model_sha256']:
        raise ValueError('Source model fingerprint changed.')
    config = core.read_json(source_directory / 'config.json')
    processor = core.read_json(source_directory / 'preprocessor_config.json')
    vocabulary = core.read_json(source_directory / 'vocab.json')
    if processor['sampling_rate'] != 16000 or vocabulary.get('\u0259') != 7:
        raise ValueError('Unexpected phoneme model configuration.')
    # Match TorchAudio 2.5's official Hugging Face importer, retaining the full CTC head.
    import inspect
    from torchaudio.models.wav2vec2.utils import import_huggingface_model
    importer = inspect.getmodule(import_huggingface_model)
    acoustic_config = importer._get_config(SimpleNamespace(**config))
    acoustic_config['aux_num_out'] = config['vocab_size']
    original = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    converted = {}
    for key, value in original.items():
        if key == 'wav2vec2.masked_spec_embed':
            continue
        if key.startswith('wav2vec2.feature_extractor.'):
            name = key.removeprefix('wav2vec2.')
        elif key.startswith('wav2vec2.feature_projection.'):
            name = 'encoder.' + key.removeprefix('wav2vec2.')
        elif key.startswith('wav2vec2.encoder.'):
            name = 'encoder.transformer.' + key.removeprefix('wav2vec2.encoder.')
        elif key.startswith('lm_head.'):
            name = 'aux.' + key.removeprefix('lm_head.')
        else:
            raise ValueError('Unexpected Hugging Face state key: ' + key)
        converted[name] = value
    with torch.device('meta'):
        model = torchaudio.models.wav2vec2_model(**acoustic_config).eval()
    model.load_state_dict(converted, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise ValueError('Incomplete acoustic encoder import.')
    del original, converted
    started = time.perf_counter()
    storage = getattr(args, 'storage', 'int8')
    if storage == 'int8':
        model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=True)
        encoder_state = model.state_dict()
        quantization = 'dynamic_int8_linears'
        weight_storage = 'int8-linear/float32-convolution'
        computation = 'CPU dynamic int8 linear / float32 convolution and normalization'
    else:
        # Source tensors are memory mapped. Allocate only the half linear tensors,
        # then release them after saving; convolution and normalization stay float32.
        linears = {name for name, module in model.named_modules() if isinstance(module, nn.Linear)}
        encoder_state = {name: value.half() if name.rsplit('.', 1)[0] in linears else value
                         for name, value in model.state_dict().items()}
        quantization = 'float16_linear_storage'
        weight_storage = 'float16-linear/float32-convolution-and-normalization'
        computation = 'CPU float32 math; cast one stored float16 linear layer at a time'
    print(f'Prepared full phoneme encoder storage in {time.perf_counter()-started:.1f}s', flush=True)
    gc.collect()
    state = {'model_config': acoustic_config, 'model_state_dict': encoder_state,
             'source_bundle': 'huggingface.' + source['repo'],
             'source_url': 'https://huggingface.co/' + source['repo'],
             'source_revision': source['revision'], 'source_checkpoint_sha256': source['model_sha256'],
             'license': source['license'], 'sample_rate': 16000, 'layers': config['num_hidden_layers'],
             'normalize_waveform': processor['do_normalize'], 'feature_kind': 'ctc_logits',
             'feature_dimension': config['vocab_size'], 'phoneme_vocabulary': vocabulary,
             'quantization': quantization, 'weight_storage': weight_storage, 'computation': computation}
    core.save_torch(args.output, state)
    print('Prepared:', args.output, args.output.stat().st_size, 'bytes', core.digest(args.output), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--family', choices=['xlsr53', 'lv60'], default='xlsr53')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--storage', choices=['int8', 'fp16_linear'], default='int8')
    args = parser.parse_args()
    if args.output is None:
        suffix = '_fp16_linear.pt' if args.storage == 'fp16_linear' else '_int8.pt'
        args.output = ROOT / 'data/pretrained' / ('phoneme_' + args.family + suffix)
    prepare(args)

"""Synthetic true-WavLM CPU export/freeze checks; no model-quality claims."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch
import torchaudio

import improve
import schwa as core
import frozen_wavlm as frozen
import wavlm_inference as runtime
import lexical_inference as evaluation
import test_schwa as fixtures
from wavlm_train_cpu import NORMALIZATION, centered_embeddings
from frozen_phoneme import aligned_window_features


class WavLMOfflineChecks(unittest.TestCase):
    def test_frozen_evaluation_covers_all_rows_and_rejects_changed_weights(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = fixture.fixture()
                state = {'backend': 'wavlm_phone_head', 'backbone': {}, 'epoch': 1,
                         'settings': {'feature_context': 'recording', 'wavlm_recipe': frozen.WAVLM_RECIPE,
                                      'recording_normalization': NORMALIZATION},
                         'dataset_sha256': core.digest(fixture.root / 'data/manifests/dataset.csv'),
                         'threshold': .5, 'pool_frames': 2, 'threshold_selection_split': 'validation',
                         'provenance': {'test_used_for_selection': False}, 'validation_metrics': {}}
                model = fixture.root / 'fixture.pt'
                torch.save(state, model)
                frozen_path, output = fixture.root / 'frozen.json', fixture.root / 'evaluation'
                with patch.object(improve, 'activate_dataset', return_value=fixture.root):
                    evaluation.freeze(model, frozen_path)
                    with self.assertRaisesRegex(ValueError, 'frozen before test'):
                        evaluation.evaluate(model, output, 'test')
                    with patch.object(runtime, 'WavLMDetector') as detector:
                        detector.return_value.sequence.return_value = None
                        detector.return_value.scores.side_effect = lambda sequence, starts: [.9 for _ in starts]
                        evaluation.evaluate(model, output, 'test', frozen_path)
                    selected = [r for r in rows if r['split'] == 'test']
                    self.assertEqual(core.read_json(output / 'metrics.json')['overall']['files'], len(selected))
                    self.assertEqual([r['sample_id'] for r in core.read_csv(output / 'predictions.csv')],
                                     [r['sample_id'] for r in selected])
                    state['threshold'] = .6
                    torch.save(state, model)
                    with patch.object(runtime, 'WavLMDetector') as detector:
                        with self.assertRaisesRegex(ValueError, 'differs from the frozen selection'):
                            evaluation.evaluate(model, fixture.root / 'changed', 'test', frozen_path)
                        detector.assert_not_called()
        finally:
            fixture.doCleanups()

    def test_pack_four_files_and_isolated_inference_with_true_wavlm_attention(self):
        torch.set_num_threads(2)
        torch.manual_seed(14)
        config = dict(extractor_mode='group_norm', extractor_conv_layer_config=[(8, 10, 5)] + [(8, 3, 2)] * 4 + [(8, 2, 2)] * 2,
                      extractor_conv_bias=False, encoder_embed_dim=768, encoder_projection_dropout=0.,
                      encoder_pos_conv_kernel=16, encoder_pos_conv_groups=16, encoder_num_layers=12,
                      encoder_num_heads=12, encoder_max_distance=128, encoder_num_buckets=32,
                      encoder_attention_dropout=0., encoder_ff_interm_features=16,
                      encoder_ff_interm_dropout=0., encoder_dropout=0., encoder_layer_norm_first=False,
                      encoder_layer_drop=0., aux_num_out=None)
        source = torchaudio.models.wavlm_model(**config).eval()
        # Share random noninitial layer tensors to keep fixture files small. The
        # complete twelve-layer WavLM relative-position/gating math still runs.
        source.encoder.transformer.layers = torch.nn.ModuleList(
            [source.encoder.transformer.layers[0]] + [source.encoder.transformer.layers[1]] * 11)
        with torch.device('meta'):
            skeleton = torchaudio.models.wavlm_model(**config)
        linears = {name for name, module in skeleton.named_modules() if isinstance(module, torch.nn.Linear)}
        converted, shared = {}, {}
        for name, value in source.state_dict().items():
            half = name.rsplit('.', 1)[0] in linears
            identity = (value.data_ptr(), half)
            if identity not in shared:
                shared[identity] = value.half() if half else value
            converted[name] = shared[identity]
        acoustic = {'model_family': 'wavlm_base_plus', 'quantization': 'float16_linear_storage',
                    'model_config': config, 'model_state_dict': converted, 'recipe': frozen.WAVLM_RECIPE}
        with tempfile.TemporaryDirectory(prefix='wavlm_offline_') as temporary:
            root = Path(temporary)
            pretrained = root / 'data/pretrained'
            pretrained.mkdir(parents=True)
            for name in ['LICENSE_WAVLM_MICROSOFT.txt', 'LICENSE_WAV2VEC2.txt']:
                (pretrained / name).write_text('Synthetic test license', encoding='utf-8')
            backbone = root / 'public_fixture.pt'
            torch.save(acoustic, backbone)
            rng = np.random.default_rng(4)
            paths = []
            for index, (samples, rate, channels) in enumerate([(3200, 16000, 1), (22050, 44100, 2), (200321, 16000, 1)]):
                path = root / f'audio{index}.wav'
                sf.write(path, rng.normal(0., .08, (samples, channels)), rate, subtype='PCM_16')
                paths.append(path)
            for temporal in ['none', 'gru']:
                head_config = {'classes': 2, 'hidden': 16, 'dropout': .2, 'temporal': temporal, 'input_dim': 768}
                head = improve.PhoneHead(**head_config).eval()
                checkpoint = {'backend': 'wavlm_phone_head', 'target_phones': ['ax'], 'target_symbol': '\u0259',
                    'head_config': head_config, 'head_state_dict': head.state_dict(), 'phone_vocabulary': ['non_ax', 'ax'],
                    'epoch': 1, 'threshold': .5, 'pool_frames': 2, 'cpu_threads': 2,
                    'threshold_selection_split': 'validation', 'dataset_sha256': 'synthetic',
                    'backbone_sha256': core.digest(backbone), 'provenance': {'test_used_for_selection': False},
                    'settings': {'feature_context': 'recording', 'wavlm_recipe': frozen.WAVLM_RECIPE,
                                 'recording_normalization': NORMALIZATION}}
                learned, packed = root / (temporal + '_head.pt'), root / (temporal + '_packed.pt')
                torch.save(checkpoint, learned)
                with patch.object(runtime, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                    runtime.pack(learned, packed, backbone)
                detector = runtime.WavLMDetector(packed)
                mismatched = root / (temporal + '_mismatched_test.json')
                mismatched.write_text(json.dumps({'split': 'test', 'model_sha256': core.digest(packed),
                    'dataset_sha256': 'synthetic', 'wavlm_recipe': frozen.WAVLM_RECIPE,
                    'recording_normalization': NORMALIZATION, 'pool_frames': 2,
                    'overall': {'threshold': .5}, 'threshold_selection_split': 'validation',
                    'inference_code_sha256': {'frozen_wavlm.py': '0' * 64}}), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Inference source differs'):
                    runtime.export(packed, root / (temporal + '_mismatched_export'), mismatched)
                wrong_pool = json.loads(mismatched.read_text(encoding='utf-8'))
                wrong_pool['pool_frames'] = 3
                mismatched.write_text(json.dumps(wrong_pool), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'Test report differs'):
                    runtime.export(packed, root / (temporal + '_wrong_pool_export'), mismatched)
                bundle, isolated = root / (temporal + '_bundle'), root / (temporal + '_isolated')
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.export(packed, bundle)
                self.assertEqual({p.name for p in bundle.iterdir()}, {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
                with zipfile.ZipFile(bundle.with_suffix('.zip')) as archive:
                    self.assertEqual(set(archive.namelist()), {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
                    archive.extractall(isolated)
                for path in paths:
                    expected = detector.predict(path)
                    output = isolated / 'result.json'
                    process = subprocess.run([sys.executable, '-I', str(isolated / 'schwa.py'), 'predict',
                                              '--audio', str(path), '--output', str(output)], cwd=isolated,
                                             capture_output=True, text=True, encoding='utf-8', timeout=60)
                    self.assertEqual(process.returncode, 0, process.stderr)
                    actual = json.loads(output.read_text(encoding='utf-8'))
                    self.assertEqual(expected['detected'], actual['detected'])
                    self.assertEqual(expected['scanned_windows'], actual['scanned_windows'])
                    self.assertEqual(expected['windows'], actual['windows'])
                    self.assertAlmostEqual(expected['score'], actual['score'], delta=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

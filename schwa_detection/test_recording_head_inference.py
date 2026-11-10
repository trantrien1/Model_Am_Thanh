"""Full-recording offline export and frozen-evaluation checks with synthetic acoustic weights."""
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
import lexical_inference as evaluation
import recording_head_inference as runtime
import schwa as core
import test_schwa as fixtures
from frozen_phoneme import RECORDING


class RecordingRuntimeChecks(unittest.TestCase):
    def test_frozen_all_regions_and_changed_model_rejected_before_inference(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = fixture.fixture()
                state = {'backend': 'wav2vec2_recording_phone_head', 'backbone': {}, 'epoch': 1,
                    'settings': {'feature_context': 'recording'}, 'dataset_sha256': core.digest(fixture.root / 'data/manifests/dataset.csv'),
                    'threshold': .5, 'pool_frames': 2, 'threshold_selection_split': 'validation',
                    'provenance': {'test_used_for_selection': False, 'supplement_manifest_sha256': 'synthetic'}, 'validation_metrics': {}}
                model = fixture.root / 'fixture.pt'
                torch.save(state, model)
                frozen, output = fixture.root / 'frozen.json', fixture.root / 'evaluation'
                with patch.object(improve, 'activate_dataset', return_value=fixture.root):
                    evaluation.freeze(model, frozen)
                    with self.assertRaisesRegex(ValueError, 'frozen before test'):
                        evaluation.evaluate(model, output, 'test')
                    with patch.object(runtime, 'RecordingHeadDetector') as detector:
                        detector.return_value.sequence.return_value = None
                        detector.return_value.scores.side_effect = lambda sequence, starts: [.9] * len(starts)
                        evaluation.evaluate(model, output, 'test', frozen)
                    selected = [r for r in rows if r['split'] == 'test']
                    self.assertEqual(core.read_json(output / 'metrics.json')['overall']['files'], len(selected))
                    self.assertEqual([r['sample_id'] for r in core.read_csv(output / 'predictions.csv')], [r['sample_id'] for r in selected])
                    state['pool_frames'] = 3
                    torch.save(state, model)
                    with patch.object(runtime, 'RecordingHeadDetector') as detector:
                        with self.assertRaisesRegex(ValueError, 'differs from the frozen selection'):
                            evaluation.evaluate(model, fixture.root / 'changed', 'test', frozen)
                        detector.assert_not_called()
        finally:
            fixture.doCleanups()

    def test_four_file_zip_offline_parity_and_export_report_guards(self):
        torch.set_num_threads(2)
        torch.manual_seed(22)
        config = dict(extractor_mode='group_norm', extractor_conv_layer_config=[(8, 10, 5)] + [(8, 3, 2)] * 4 + [(8, 2, 2)] * 2,
            extractor_conv_bias=False, encoder_embed_dim=1024, encoder_projection_dropout=0.,
            encoder_pos_conv_kernel=16, encoder_pos_conv_groups=16, encoder_num_layers=2, encoder_num_heads=16,
            encoder_attention_dropout=0., encoder_ff_interm_features=16, encoder_ff_interm_dropout=0.,
            encoder_dropout=0., encoder_layer_norm_first=True, encoder_layer_drop=0., aux_num_out=392)
        source = torchaudio.models.wav2vec2_model(**config).eval()
        names = {name for name, module in source.named_modules() if isinstance(module, torch.nn.Linear)}
        state = {name: values.half() if name.rsplit('.', 1)[0] in names else values for name, values in source.state_dict().items()}
        acoustic = {'model_config': config, 'model_state_dict': state, 'layers': 2, 'feature_kind': 'ctc_logits',
            'feature_dimension': 392, 'quantization': 'float16_linear_storage', 'normalize_waveform': True}
        with tempfile.TemporaryDirectory(prefix='recording_offline_') as temporary:
            root = Path(temporary)
            for name in ['data/pretrained/LICENSE_WAV2VEC2.txt', 'data/pretrained/phoneme_xlsr53/LICENSE_APACHE_2.txt']:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Synthetic test license', encoding='utf-8')
            backbone = root / 'public_fixture.pt'
            torch.save(acoustic, backbone)
            head_config = {'classes': 2, 'hidden': 16, 'dropout': .3, 'temporal': 'gru', 'input_dim': 1416, 'group_sizes': [1024, 392]}
            head = improve.PhoneHead(**head_config).eval()
            learned = root / 'head.pt'
            torch.save({'backend': 'wav2vec2_recording_phone_head', 'target_phones': ['ax'], 'target_symbol': '\u0259',
                'epoch': 1, 'head_config': head_config, 'head_state_dict': head.state_dict(), 'phone_vocabulary': ['pau', 'ax'],
                'threshold': .1, 'pool_frames': 2, 'cpu_threads': 2, 'threshold_selection_split': 'validation',
                'provenance': {'test_used_for_selection': False, 'supplement_manifest_sha256': 'synthetic'},
                'dataset_sha256': 'synthetic', 'backbone_sha256': core.digest(backbone),
                'settings': {'head_context': 'whole_recording', 'feature_context': 'recording', 'recording_recipe': RECORDING,
                    'region_logit_interpolation': runtime.RUNTIME_RECIPE['region_logit_interpolation']}}, learned)
            packed, bundle, isolated = root / 'packed.pt', root / 'bundle', root / 'isolated'
            with patch.object(runtime, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                runtime.pack(learned, packed, backbone)
            report = {'split': 'test', 'model_sha256': core.digest(packed), 'dataset_sha256': 'synthetic',
                'recording_head_recipe': runtime.RUNTIME_RECIPE, 'pool_frames': 2, 'threshold_selection_split': 'validation',
                'overall': {'threshold': .1}, 'inference_code_sha256': {}}
            metrics = root / 'invalid_test.json'
            metrics.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Inference source differs'):
                runtime.export(packed, root / 'invalid_export', metrics)
            report['pool_frames'] = 3
            metrics.write_text(json.dumps(report), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Test report differs'):
                runtime.export(packed, root / 'wrong_pool_export', metrics)
            with contextlib.redirect_stdout(io.StringIO()):
                runtime.export(packed, bundle)
            self.assertEqual({p.name for p in bundle.iterdir()}, {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
            with zipfile.ZipFile(bundle.with_suffix('.zip')) as archive:
                self.assertEqual(set(archive.namelist()), {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
                archive.extractall(isolated)
            detector = runtime.RecordingHeadDetector(packed)
            rng = np.random.default_rng(7)
            for index, (samples, rate, channels) in enumerate([(14400, 16000, 1), (3200, 16000, 1), (22050, 44100, 2), (200321, 16000, 1)]):
                audio = root / f'audio{index}.wav'
                sf.write(audio, rng.normal(0, .08, (samples, channels)), rate, subtype='PCM_16')
                expected = detector.predict(audio)
                output = isolated / 'result.json'
                process = subprocess.run([sys.executable, '-I', str(isolated / 'schwa.py'), 'predict', '--audio', str(audio), '--output', str(output)],
                    cwd=isolated, capture_output=True, text=True, encoding='utf-8', timeout=60)
                self.assertEqual(process.returncode, 0, process.stderr)
                actual = core.read_json(output)
                self.assertEqual(expected['detected'], actual['detected'])
                self.assertEqual(expected['scanned_windows'], actual['scanned_windows'])
                self.assertEqual([(r['start_seconds'], r['end_seconds']) for r in expected['windows']],
                                 [(r['start_seconds'], r['end_seconds']) for r in actual['windows']])
                np.testing.assert_allclose([r['score'] for r in expected['windows']], [r['score'] for r in actual['windows']], atol=1e-6, rtol=0)
                self.assertAlmostEqual(expected['score'], actual['score'], delta=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)

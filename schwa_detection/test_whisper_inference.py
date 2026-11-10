"""Synthetic runtime wiring, full-region freeze and four-file export guards.

These checks do not establish public-encoder parity or Schwa quality.
"""
import contextlib
import io
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import improve
import lexical_inference as evaluation
import schwa as core
import test_schwa as fixtures
import whisper_inference as runtime
from frozen_phoneme import aligned_window_features


class WhisperRuntimeChecks(unittest.TestCase):
    def test_audio_only_sequence_and_region_scores(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)
        acoustic = torch.randn(31, 1416).half().float()
        dictionary = {'A': [['AH0']]}
        vocabulary = {'<pad>': 0, '\u0259': 1}
        words = [{'word': ' a', 'start': .1, 'end': .3, 'probability': .9}]
        calls = []
        class AudioASR:
            def transcribe(self, audio, **settings):
                calls.append((audio.copy(), settings))
                return {'segments': [{'words': words}]}
        detector = runtime.WhisperLexicalDetector.__new__(runtime.WhisperLexicalDetector)
        detector.acoustic, detector.asr = object(), AudioASR()
        detector.dictionary, detector.vocabulary = dictionary, vocabulary
        detector.checkpoint = {'backbone': {'layers': 24}, 'pool_frames': 2}
        detector.head = improve.PhoneHead(2, hidden=16, dropout=.3, temporal='gru', input_dim=1424, group_sizes=[1024, 392, 8]).eval()
        detector.ax = 1
        audio = np.zeros(10000, dtype=np.float32)
        with patch.object(runtime, 'recording_features', return_value=acoustic):
            sequence = detector.sequence(audio)
        self.assertEqual(sequence.shape, (31, 1424))
        self.assertEqual(len(calls), 1)
        self.assertTrue(np.array_equal(calls[0][0], audio))
        self.assertEqual(calls[0][1], runtime.DECODE)
        self.assertIsNone(calls[0][1]['initial_prompt'])
        self.assertTrue(torch.equal(sequence[:, :1416], acoustic))
        starts = [0, 731, 2000]
        with torch.inference_mode(), torch.backends.mkldnn.flags(enabled=False):
            frames = detector.head(aligned_window_features(sequence, starts)).softmax(-1)[:, :, 1]
            expected = ((frames[:, :-1] + frames[:, 1:]) / 2).amax(1)
        np.testing.assert_allclose(detector.scores(sequence, starts), expected.numpy(), rtol=0, atol=1e-7)

    def test_four_file_export_and_report_identity(self):
        with tempfile.TemporaryDirectory(prefix='whisper_export_engineering_') as temporary:
            root = Path(temporary)
            model = root / 'synthetic.pt'
            state = {'backend': 'wav2vec2_whisper_lexical_head', 'whisper_resources': {},
                'licenses': {'Synthetic': 'Engineering fixture license'}, 'dataset_sha256': 'fixture',
                'pool_frames': 2, 'threshold': .5}
            torch.save(state, model)
            report = {'split': 'test', 'model_sha256': core.digest(model), 'dataset_sha256': 'fixture',
                'whisper_runtime_recipe': runtime.RUNTIME_RECIPE, 'pool_frames': 2,
                'overall': {'threshold': .5}, 'threshold_selection_split': 'validation', 'inference_code_sha256': {}}
            metrics = root / 'metrics.json'
            core.write_json(metrics, report)
            with self.assertRaisesRegex(ValueError, 'Inference source differs'):
                runtime.export(model, root / 'invalid_sources', metrics)
            report['pool_frames'] = 3
            core.write_json(metrics, report)
            with self.assertRaisesRegex(ValueError, 'Test report differs'):
                runtime.export(model, root / 'invalid_pool', metrics)
            output = root / 'bundle'
            with contextlib.redirect_stdout(io.StringIO()):
                runtime.export(model, output)
            expected = {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'}
            self.assertEqual({p.name for p in output.iterdir()}, expected)
            with zipfile.ZipFile(output.with_suffix('.zip')) as archive:
                self.assertEqual(set(archive.namelist()), expected)
                self.assertEqual(archive.read('best.pt'), model.read_bytes())
            compile((output / 'schwa.py').read_text(encoding='utf-8'), 'schwa.py', 'exec')
            self.assertIn('openai-whisper==20250625', (output / 'requirements.txt').read_text())
            with self.assertRaisesRegex(ValueError, 'Preserving'):
                runtime.export(model, output)

    def test_freeze_preserves_all_regions_and_model_changes_rejected(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = fixture.fixture()
                model = fixture.root / 'fixture.pt'
                state = {'backend': 'wav2vec2_whisper_lexical_head', 'backbone': {}, 'epoch': 1,
                    'settings': {'feature_context': 'recording'}, 'lexical': {'package_files_sha256': {}},
                    'dataset_sha256': core.digest(fixture.root / 'data/manifests/dataset.csv'),
                    'threshold': .5, 'pool_frames': 2, 'threshold_selection_split': 'validation',
                    'provenance': {'test_used_for_selection': False}, 'validation_metrics': {}}
                torch.save(state, model)
                frozen, output = fixture.root / 'frozen.json', fixture.root / 'evaluation'
                with patch.object(improve, 'activate_dataset', return_value=fixture.root):
                    evaluation.freeze(model, frozen)
                    with self.assertRaisesRegex(ValueError, 'frozen before test'):
                        evaluation.evaluate(model, output, 'test')
                    with patch.object(runtime, 'WhisperLexicalDetector') as detector:
                        detector.return_value.sequence.return_value = None
                        detector.return_value.scores.side_effect = lambda sequence, starts: [.9] * len(starts)
                        evaluation.evaluate(model, output, 'test', frozen)
                    selected = [r for r in rows if r['split'] == 'test']
                    self.assertEqual([r['sample_id'] for r in core.read_csv(output / 'predictions.csv')], [r['sample_id'] for r in selected])
                    state['pool_frames'] = 3
                    torch.save(state, model)
                    with patch.object(runtime, 'WhisperLexicalDetector') as detector:
                        with self.assertRaisesRegex(ValueError, 'differs from the frozen selection'):
                            evaluation.evaluate(model, fixture.root / 'changed', 'test', frozen)
                        detector.assert_not_called()
        finally:
            fixture.doCleanups()


if __name__ == '__main__':
    unittest.main()

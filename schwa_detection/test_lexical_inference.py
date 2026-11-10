"""Verify offline lexical packing and cache/runtime parity with synthetic encoders."""
import argparse
import contextlib
import hashlib
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

import improve
import lexical_context as lexical
import lexical_inference as runtime
import lexical_train_cpu as training
import schwa as core
from frozen_phoneme import RECORDING, aligned_window_features, recording_features, rebuild_acoustic
import test_finetune_cpu as encoder_fixtures
import test_schwa as dataset_fixtures


class OfflineLexicalChecks(unittest.TestCase):
    def test_frozen_evaluation_covers_every_region_and_resumes_identical_scores(self):
        fixture = dataset_fixtures.SchwaChecks()
        fixture.setUp()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rows = fixture.fixture()
                selected = [r for r in rows if r['split'] == 'test']
                state = {'backend': 'wav2vec2_lexical_phone_head', 'lexical_resources': {},
                         'dataset_sha256': core.digest(fixture.root / 'data/manifests/dataset.csv'),
                         'threshold': .5, 'pool_frames': 2, 'threshold_selection_split': 'validation',
                         'epoch': 1, 'validation_metrics': {'precision': .9, 'recall': .9},
                         'provenance': {'test_used_for_selection': False}}
                model = fixture.root / 'fixture.pt'
                torch.save(state, model)
                frozen = fixture.root / 'frozen.json'
                output = fixture.root / 'evaluation'
                with patch.object(improve, 'activate_dataset', return_value=fixture.root):
                    runtime.freeze(model, frozen)
                    with self.assertRaisesRegex(ValueError, 'frozen before test'):
                        runtime.evaluate(model, output, 'test')
                    with patch.object(runtime, 'LexicalDetector') as detector:
                        detector.return_value.sequence.return_value = None
                        detector.return_value.scores.side_effect = lambda sequence, starts: [
                            .9 if start / 16000 <= .74 < start / 16000 + .5 else .1 for start in starts]
                        runtime.evaluate(model, output, 'test', frozen)
                        self.assertEqual(detector.return_value.sequence.call_count, len({r['recording_id'] for r in selected}))
                    report = core.read_json(output / 'metrics.json')
                    self.assertEqual(report['overall']['files'], len(selected))
                    positives = sum(int(r['label']) for r in selected)
                    self.assertEqual(report['overall']['confusion_matrix'], [[len(selected)-positives, 0], [0, positives]])
                    self.assertEqual([r['sample_id'] for r in core.read_csv(output / 'predictions.csv')], [r['sample_id'] for r in selected])
                    (output / 'metrics.json').unlink()
                    with patch.object(runtime, 'LexicalDetector') as detector:
                        detector.return_value.sequence.side_effect = AssertionError('Must reuse identical frozen scores')
                        runtime.evaluate(model, output, 'test', frozen)
                        detector.return_value.sequence.assert_not_called()
                    self.assertEqual(core.read_json(output / 'metrics.json')['overall'], report['overall'])
                    state['threshold'] = .6
                    torch.save(state, model)
                    with patch.object(runtime, 'LexicalDetector') as detector:
                        with self.assertRaisesRegex(ValueError, 'differs from the frozen selection'):
                            runtime.evaluate(model, fixture.root / 'changed', 'test', frozen)
                        detector.assert_not_called()
        finally:
            fixture.doCleanups()

    def test_pack_export_and_isolated_cpu_inference_match_training_windows(self):
        source = encoder_fixtures.PrefixAdaptationChecks()
        source.setUp()
        torch.manual_seed(25)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pre = root / 'data/pretrained'
            (pre / 'phoneme_xlsr53').mkdir(parents=True)
            (pre / 'cmudict').mkdir()
            (pre / 'LICENSE_WAV2VEC2.txt').write_text('synthetic encoder test license', encoding='utf-8')
            (pre / 'phoneme_xlsr53/LICENSE_APACHE_2.txt').write_text('synthetic model test license', encoding='utf-8')
            (pre / 'cmudict/LICENSE').write_text('synthetic dictionary test license', encoding='utf-8')
            dictionary_text = 'a AH0\na(2) AH1 # alternate\n'
            vocabulary_text = json.dumps({'\u0259': 7, '\u028c': 6})
            (pre / 'cmudict/cmudict.dict').write_text(dictionary_text, encoding='utf-8')
            (pre / 'phoneme_xlsr53/vocab.json').write_text(vocabulary_text, encoding='utf-8')
            dictionary = lexical.load_dictionary(pre / 'cmudict/cmudict.dict')
            self.assertEqual(runtime.dictionary_from_text(dictionary_text), dictionary)
            acoustic = {**source.state, 'layers': 2}
            acoustic.pop('prefix_rounding_layer')
            acoustic_path = pre / 'main.pt'
            torch.save(acoustic, acoustic_path)
            asr = {**acoustic, 'model_config': {**source.config, 'aux_num_out': 3},
                   'model_state_dict': dict(acoustic['model_state_dict']),
                   'feature_kind': 'ctc_logits', 'labels': ['-', '|', 'A']}
            asr['model_state_dict']['aux.weight'] = torch.zeros(3, 32, dtype=torch.float16)
            asr['model_state_dict']['aux.bias'] = torch.tensor([-2., -2., 2.], dtype=torch.float16)
            asr_path = pre / 'asr.pt'
            torch.save(asr, asr_path)
            recipe = {'main_encoder_sha256': core.digest(acoustic_path), 'asr_encoder_sha256': core.digest(asr_path),
                      'dictionary_sha256': hashlib.sha256(dictionary_text.encode()).hexdigest(),
                      'vocabulary_sha256': hashlib.sha256(vocabulary_text.encode()).hexdigest(),
                      'feature_compute_threads': {'phoneme': 2, 'word_asr': 1, 'alignment': 1},
                      'acoustic_recipe': RECORDING, 'lexical_recipe': lexical.LEXICAL_RECIPE,
                      'reference_transcript_consumed': False, 'manual_phones_consumed_as_features': False}
            rng = np.random.default_rng(12)
            cases = [(3200, 16000, 1), (13000, 44100, 2), (200321, 16000, 1)]
            audio_paths = []
            for index, (samples, rate, channels) in enumerate(cases):
                path = root / f'input{index}.wav'
                sf.write(path, rng.normal(0., .08, (samples, channels)), rate, subtype='PCM_16')
                audio_paths.append(path)
            for temporal in ['none', 'gru']:
                config = {'classes': 2, 'hidden': 16, 'dropout': .2, 'temporal': temporal,
                          'input_dim': 44, 'group_sizes': [32, 8, 4]}
                head = improve.PhoneHead(**config).eval()
                checkpoint = {'backend': 'wav2vec2_lexical_phone_head', 'target_symbol': '\u0259', 'target_phones': ['ax'],
                    'head_config': config, 'head_state_dict': head.state_dict(), 'phone_vocabulary': ['non_ax', 'ax'],
                    'epoch': 1, 'threshold': .5, 'pool_frames': 2, 'cpu_threads': 1,
                    'threshold_selection_split': 'validation', 'lexical': recipe, 'dataset_sha256': 'synthetic_dataset',
                    'settings': {'feature_context': 'recording', 'acoustic_representation': 'joint'}}
                learned = root / (temporal + '_head.pt')
                packed = root / (temporal + '_packed.pt')
                torch.save(checkpoint, learned)
                with patch.object(runtime, 'ROOT', root), patch.object(training, 'metadata', return_value=recipe), \
                     patch.object(training, 'MAIN_ENCODER', acoustic_path), patch.object(lexical, 'ASR_PATH', asr_path), \
                     patch.object(lexical, 'DICTIONARY_SOURCE', pre / 'cmudict'), contextlib.redirect_stdout(io.StringIO()):
                    runtime.pack(learned, packed)
                detector = runtime.LexicalDetector(packed)
                bundle = root / (temporal + '_bundle')
                with contextlib.redirect_stdout(io.StringIO()):
                    runtime.export(packed, bundle)
                self.assertEqual({p.name for p in bundle.iterdir()}, {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
                isolated = root / (temporal + '_isolated')
                with zipfile.ZipFile(bundle.with_suffix('.zip')) as archive:
                    self.assertEqual(set(archive.namelist()), {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
                    archive.extractall(isolated)
                for path in audio_paths:
                    audio, _ = core.load_audio(path)
                    acoustic_model = rebuild_acoustic(acoustic)
                    asr_model = rebuild_acoustic(asr)
                    with torch.inference_mode():
                        torch.set_num_threads(2)
                        teacher = recording_features(acoustic_model, torch.from_numpy(audio), 2)
                        torch.set_num_threads(1)
                        characters = recording_features(asr_model, torch.from_numpy(audio), 2).float()
                        prior, _ = lexical.bounded_lexical_features(characters, teacher[:, 32:].float(), dictionary,
                                                                  asr['labels'], json.loads(vocabulary_text))
                        prior = prior.half().float()
                        starts = [0, 73, max(0, len(audio) - 8000)]
                        cached = training.CombinedFeatures(aligned_window_features(teacher, starts).numpy().astype(np.float16),
                                                           aligned_window_features(prior, starts).numpy().astype(np.float16))
                        expected = improve.class_scores(head, cached, 1, 2)
                        actual = detector.scores(detector.sequence(audio), starts)
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
                    pipeline = detector.predict(path)
                    output = isolated / 'output.json'
                    # -I excludes the training workspace from import resolution.
                    result = subprocess.run([sys.executable, '-I', str(isolated / 'schwa.py'), 'predict',
                                             '--audio', str(path), '--output', str(output)],
                                            cwd=isolated, capture_output=True, text=True, encoding='utf-8', timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    standalone = json.loads(output.read_text(encoding='utf-8'))
                    self.assertEqual(pipeline['detected'], standalone['detected'])
                    self.assertEqual(pipeline['scanned_windows'], standalone['scanned_windows'])
                    self.assertEqual(pipeline['windows'], standalone['windows'])
                    self.assertAlmostEqual(pipeline['score'], standalone['score'], delta=1e-6)
                mutated = torch.load(packed, weights_only=True)
                mutated['lexical']['lexical_recipe'] = {**lexical.LEXICAL_RECIPE, 'version': -1}
                wrong = root / (temporal + '_wrong.pt')
                torch.save(mutated, wrong)
                with self.assertRaisesRegex(ValueError, 'recipes differ'):
                    runtime.LexicalDetector(wrong)


if __name__ == '__main__':
    unittest.main(verbosity=2)

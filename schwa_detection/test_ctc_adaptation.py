"""Verify CTC target semantics, unknown spans and genuine gradients independently."""
import itertools
import argparse
import contextlib
import io
import json
import subprocess
import sys
import unittest
import zipfile
from unittest.mock import patch

import numpy as np
import torch
import torchaudio
import soundfile as sf

import finetune_cpu as tune
import improve
import schwa as core
import test_schwa as fixtures
import lexical_inference as runtime
from recording_runtime import RecordingPhoneDetector
from export_inference import export


class CTCAdaptationChecks(unittest.TestCase):
    def test_joint_training_resume_matches_uninterrupted_updates(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            torch.set_num_threads(2)
            with contextlib.redirect_stdout(io.StringIO()):
                fixture.fixture()
                torch.manual_seed(123)
                config = dict(extractor_mode='layer_norm', extractor_conv_layer_config=[(8, 10, 5)] + [(8, 3, 2)] * 4 + [(8, 2, 2)] * 2,
                              extractor_conv_bias=True, encoder_embed_dim=32, encoder_projection_dropout=0.,
                              encoder_pos_conv_kernel=16, encoder_pos_conv_groups=4, encoder_num_layers=2,
                              encoder_num_heads=4, encoder_attention_dropout=0., encoder_ff_interm_features=64,
                              encoder_ff_interm_dropout=0., encoder_dropout=0., encoder_layer_norm_first=True,
                              encoder_layer_drop=0., aux_num_out=50)
                source = torchaudio.models.wav2vec2_model(**config).eval()
                linear = {name for name, module in source.named_modules() if isinstance(module, torch.nn.Linear)}
                state = {'model_config': config, 'layers': 2, 'feature_kind': 'ctc_logits', 'feature_dimension': 50,
                         'source_bundle': 'synthetic.cpu',
                         'quantization': 'float16_linear_storage',
                         'phoneme_vocabulary': {'\u0259': 7, '\u028c': 33, '\u025b': 11, 't': 12, 'i\u02d0': 13},
                         'model_state_dict': {name: value.half() if name.rsplit('.', 1)[0] in linear else value
                                              for name, value in source.state_dict().items()}}
                backbone = fixture.root / 'public_fixture.pt'
                torch.save(state, backbone)
                with patch.object(improve, 'activate_dataset', return_value=fixture.root):
                    tune.prepare(argparse.Namespace(backbone=backbone, prefix_layers=1, splits=['train', 'validation']))
                    args = argparse.Namespace(backbone=backbone, prefix_layers=1, run_dir=fixture.root / 'continuous',
                                              task='phones', hidden=16, seed=42, threads=2, tail_lr=.00003,
                                              lr=.0005, epochs=2, warmup_epochs=0, accumulate=2,
                                              window_loss=1., ctc_weight=1., patience=5, resume=False)
                    tune.train(args)
                    args.run_dir = fixture.root / 'interrupted'
                    original_save = core.save_torch
                    def interrupt_after_saved_epoch(path, values):
                        original_save(path, values)
                        if path.name == 'last.pt':
                            raise InterruptedError('Simulated interruption after complete checkpoint')
                    with patch.object(core, 'save_torch', side_effect=interrupt_after_saved_epoch):
                        with self.assertRaises(InterruptedError):
                            tune.train(args)
                    args.resume = True
                    tune.train(args)
                    a = torch.load(fixture.root / 'continuous/last.pt', weights_only=True)
                    b = torch.load(fixture.root / 'interrupted/last.pt', weights_only=True)
                    self.assertEqual(b['epoch'], 2)
                    self.assertEqual(a['best_epoch'], b['best_epoch'])
                    for category in ['adapted_state_dict', 'head_state_dict']:
                        for name in a[category]:
                            torch.testing.assert_close(a[category][name], b[category][name], rtol=0, atol=0)
                    self.assertEqual(a['numpy_random_state'], b['numpy_random_state'])
                    torch.testing.assert_close(a['torch_random_state'], b['torch_random_state'], rtol=0, atol=0)
                    aux_changed = any(not torch.equal(a['adapted_state_dict'][name], state['model_state_dict'][name])
                                      for name in a['adapted_state_dict'] if name.startswith('aux.'))
                    self.assertTrue(aux_changed)
                    run = fixture.root / 'continuous'
                    packed = run / 'packed.pt'
                    with patch.object(improve, 'BACKBONE', run / 'adapted_encoder.pt'):
                        improve.pack(argparse.Namespace(model=run / 'best.pt', output=packed))
                    frozen = run / 'frozen_selection.json'
                    evaluation = run / 'evaluation'
                    runtime.freeze(packed, frozen)
                    with self.assertRaisesRegex(ValueError, 'frozen before test'):
                        runtime.evaluate(packed, evaluation, 'test')
                    runtime.evaluate(packed, evaluation, 'test', frozen)
                    rows = [r for r in core.read_csv(fixture.root / 'data/manifests/dataset.csv') if r['split'] == 'test']
                    report = core.read_json(evaluation / 'metrics.json')
                    self.assertEqual(report['overall']['files'], len(rows))
                    self.assertIsNone(report['lexical_recipe'])
                    self.assertEqual([r['sample_id'] for r in core.read_csv(evaluation / 'predictions.csv')],
                                     [r['sample_id'] for r in rows])
                    detector = RecordingPhoneDetector(packed)
                    bundle = run / 'bundle'
                    export(packed, bundle, evaluation / 'metrics.json')
                    isolated = run / 'isolated'
                    with zipfile.ZipFile(bundle.with_suffix('.zip')) as archive:
                        self.assertEqual(set(archive.namelist()), {'schwa.py', 'best.pt', 'requirements.txt', 'README.md'})
                        archive.extractall(isolated)
                    record = next(r for r in core.read_csv(fixture.root / 'data/manifests/recordings.csv') if r['split'] == 'train')
                    original = fixture.root / record['audio_path']
                    audio, _ = core.load_audio(original)
                    audio_paths = [original]
                    cases = [(audio[:3200], 16000), (np.stack([audio[:13000], audio[:13000]], -1), 44100),
                             (np.tile(audio, 5), 16000)]
                    for index, (samples, rate) in enumerate(cases):
                        path = run / f'input_{index}.wav'
                        sf.write(path, samples, rate, subtype='PCM_16')
                        audio_paths.append(path)
                    for path in audio_paths:
                        expected = detector.predict(path)
                        output = isolated / 'result.json'
                        process = subprocess.run([sys.executable, '-I', str(isolated / 'schwa.py'), 'predict',
                                                  '--audio', str(path), '--output', str(output)], cwd=isolated,
                                                 capture_output=True, text=True, encoding='utf-8', timeout=60)
                        self.assertEqual(process.returncode, 0, process.stderr)
                        actual = json.loads(output.read_text(encoding='utf-8'))[0]
                        self.assertEqual(expected['detected'], actual['detected'])
                        self.assertEqual(expected['scanned_windows'], actual['scanned_windows'])
                        self.assertEqual(expected['windows'], actual['windows'])
                        self.assertAlmostEqual(expected['score'], actual['score'], delta=1e-6)
        finally:
            fixture.doCleanups()

    def test_unknown_splits_targets_and_ah_er_stay_negative(self):
        phones = [{'start': 0., 'end': .1, 'phone': 'pau'},
                  {'start': .1, 'end': .2, 'phone': 'ax'},
                  {'start': .2, 'end': .3, 'phone': 'ah'},
                  {'start': .3, 'end': .4, 'phone': 'unk'},
                  {'start': .4, 'end': .5, 'phone': 'er'},
                  {'start': .5, 'end': .6, 'phone': 'ax'}]
        result = tune.ctc_segments(phones, {'\u0259': 7, '\u028c': 33, '\u025a': 43}, 30)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][2].tolist(), [7, 33])
        self.assertEqual(result[1][2].tolist(), [43, 7])
        centers = (.01246875 + np.arange(30) * .02)
        self.assertTrue(np.all(centers[result[0][0]:result[0][1]] < .3))
        self.assertTrue(np.all(centers[result[1][0]:result[1][1]] >= .4))

    def test_ctc_matches_enumeration_and_backpropagates(self):
        logits = torch.tensor([[.1, .4, -.2], [.5, -.1, .3], [.2, .8, -.4]], requires_grad=True)
        target = [1, 1]
        expected = 0.
        probabilities = logits.detach().softmax(-1).numpy()
        for path in itertools.product(range(3), repeat=3):
            collapsed = [p for i, p in enumerate(path) if (i == 0 or p != path[i - 1]) and p != 0]
            if collapsed == target:
                expected += float(np.prod([probabilities[i, p] for i, p in enumerate(path)]))
        loss = tune.recording_ctc_loss(logits, [(0, 3, torch.tensor(target))])
        self.assertAlmostEqual(float(loss.detach()), -np.log(expected) / 2, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.)

    def test_repeats_require_blank_and_invalid_loss_is_rejected(self):
        phones = [{'start': 0., 'end': .02, 'phone': 'ax'},
                  {'start': .02, 'end': .04, 'phone': 'ax'}]
        with self.assertRaisesRegex(ValueError, 'too few frames'):
            tune.ctc_segments(phones, {'\u0259': 1}, 2)
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            tune.recording_ctc_loss(torch.zeros(2, 3), [(0, 2, torch.tensor([1, 1]))])


if __name__ == '__main__':
    unittest.main(verbosity=2)

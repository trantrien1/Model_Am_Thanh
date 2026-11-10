"""Full-recording timeline, context and exact-resume checks; synthetic data is not quality evidence."""
import argparse
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

import recording_head_cpu as train
import schwa as core
from frozen_phoneme import RECORDING


class RecordingHeadChecks(unittest.TestCase):
    def test_full_record_frame_centers_unknown_and_padding(self):
        phones = [{'start': 0., 'end': .021, 'phone': 'ax'},
                  {'start': .021, 'end': .04, 'phone': 'ah'},
                  {'start': .04, 'end': .1, 'phone': 'unk'}]
        actual = train.frame_targets(phones, 6, ['ah', 'ax', 'pau'], .1)
        np.testing.assert_array_equal(actual, [1, 0, -100, -100, -100, 2])

    def test_region_interpolation_and_pooling(self):
        logits = torch.tensor([[0., 2.], [2., 0.], [0., 1.]])
        starts = [160, -320, 640]
        actual = train.region_scores(logits, starts, 1, 2).numpy()
        expected = []
        for start in starts:
            positions = np.clip(start / 320 + np.arange(24), 0, 2)
            values = np.column_stack([np.interp(positions, np.arange(3), logits[:, i].numpy()) for i in range(2)])
            values = values.astype(np.float16).astype(np.float32)
            probabilities = np.exp(values - values.max(-1, keepdims=True))
            probabilities = probabilities[:, 1] / probabilities.sum(-1)
            expected.append(float(np.max((probabilities[:-1] + probabilities[1:]) / 2)))
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-7)

    def test_head_uses_surrounding_recording_context(self):
        torch.manual_seed(9)
        model = train.improve.PhoneHead(classes=2, hidden=8, dropout=0., temporal='gru', input_dim=6).eval()
        focal = torch.randn(24, 6)
        full = torch.cat([torch.ones(8, 6) * 5, focal, torch.ones(8, 6) * -5])
        with torch.inference_mode():
            contextual = model(full.unsqueeze(0))[0, 8:32]
            isolated = model(focal.unsqueeze(0))[0]
        self.assertGreater(float((contextual - isolated).abs().max()), 1e-4)

    def test_train_interruption_exact_resume_and_test_loader_guard(self):
        old_mkldnn = torch.backends.mkldnn.enabled
        old_deterministic = torch.are_deterministic_algorithms_enabled()
        try:
            with tempfile.TemporaryDirectory(prefix='recording_gru_') as temporary:
                root = Path(temporary)
                for name in train.HELPERS:
                    (root / name).write_bytes((train.ROOT / name).read_bytes())
                backbone = root / 'public_fixture.pt'
                torch.save({'synthetic': True}, backbone)
                encoder_sha = core.digest(backbone)
                rng = np.random.default_rng(19)
                base, supplement = root / 'v5', root / 'training_suitcase'
                for directory, plan in [(base, [('train', 't0'), ('train', 't1'), ('validation', 'v0'), ('validation', 'v1')]),
                                        (supplement, [('train', 's0')])]:
                    rows, records = [], []
                    for split, identity in plan:
                        wav, lab = directory / (identity + '.wav'), directory / (identity + '.lab')
                        directory.mkdir(exist_ok=True)
                        sf.write(wav, rng.normal(0, .1, 14400), 16000, subtype='PCM_16')
                        lab.write_text('#\n0.2 26 ax\n0.9 26 ah\n', encoding='utf-8')
                        record = {'recording_id': identity, 'split': split, 'audio_path': wav.name, 'audio_sha256': core.digest(wav),
                                  'lab_path': lab.name, 'lab_sha256': core.digest(lab)}
                        records.append(record)
                        for index, (start, label) in enumerate([(0, 1), (6400, 0)]):
                            rows.append({'recording_id': identity, 'sample_id': identity + str(index), 'split': split,
                                         'window_start_sample': start, 'label': label})
                        recipe = {'encoder': encoder_sha, 'source_audio': record['audio_sha256'], 'recipe': RECORDING, 'representation': 'joint'}
                        path = directory / 'data/recording_transfer_features' / (core.object_digest(recipe) + '.npy')
                        path.parent.mkdir(parents=True, exist_ok=True)
                        values = rng.normal(size=(44, 1416)).astype(np.float16)
                        np.save(path, values, allow_pickle=False)
                        core.write_json(path.with_suffix('.json'), {'sha256': core.digest(path), 'shape': list(values.shape),
                            'backbone_sha256': encoder_sha, 'source_audio_sha256': record['audio_sha256'], 'recipe': RECORDING, 'representation': 'joint'})
                    core.write_csv(directory / 'data/manifests/dataset.csv', rows)
                    core.write_csv(directory / 'data/manifests/recordings.csv', records)
                core.write_json(supplement / 'reports/v5_preservation_audit.json', {'passed': True,
                    'cross_dataset_held_out_collisions': {}, 'v5_manifests_sha256': {'dataset.csv': core.digest(base / 'data/manifests/dataset.csv')},
                    'source_sha256': 'synthetic'})
                def args(run, resume=False):
                    return argparse.Namespace(backbone=backbone, supplement=supplement, run_dir=root / run,
                        threads=2, seed=42, hidden=8, epochs=3, patience=8, lr=.0005, accumulate=2, window_loss=.5, resume=resume)
                with patch.object(train, 'ROOT', root), contextlib.redirect_stdout(io.StringIO()):
                    train.train(args('continuous'))
                    original_save = core.save_torch
                    def interrupted(path, state):
                        original_save(path, state)
                        if path.name == 'last.pt' and state['epoch'] == 1:
                            raise RuntimeError('Fixture interruption after completed epoch')
                    with patch.object(core, 'save_torch', side_effect=interrupted):
                        with self.assertRaisesRegex(RuntimeError, 'Fixture interruption'):
                            train.train(args('resumed'))
                    train.train(args('resumed', True))
                left = torch.load(root / 'continuous/last.pt', weights_only=True)
                right = torch.load(root / 'resumed/last.pt', weights_only=True)
                for name, values in left['model_state_dict'].items():
                    torch.testing.assert_close(values, right['model_state_dict'][name], atol=0, rtol=0)
                self.assertEqual(left['best_epoch'], right['best_epoch'])
                torch.testing.assert_close(left['torch_random_state'], right['torch_random_state'], atol=0, rtol=0)
                with self.assertRaisesRegex(ValueError, 'cannot consume test'):
                    train.RecordingStore(base, [{'split': 'test'}], encoder_sha)
        finally:
            torch.backends.mkldnn.enabled = old_mkldnn
            torch.use_deterministic_algorithms(old_deterministic)


if __name__ == '__main__':
    unittest.main(verbosity=2)

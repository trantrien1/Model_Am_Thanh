"""Small synthetic fixtures verify data integrity and inference; not model-quality evidence."""
import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
import torch

import schwa as s


class SchwaChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='schwa_check_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patch = mock.patch.object(s, 'ROOT', self.root)
        patch.start()
        self.addCleanup(patch.stop)
        torch.set_num_threads(2)

    def fixture(self, invalid_recording=False):
        recordings = []
        for split_index, split in enumerate(s.SPLITS):
            for speaker_index, speaker in enumerate(s.SPEAKERS[split]):
                identity = f'{speaker}_{split}_fixture'
                t = np.arange(38400, dtype=np.float32) / 16000
                frequency = 170 + split_index * 70 + speaker_index * 21
                audio = .2 * np.sin(2 * np.pi * frequency * t)
                audio += .03 * np.sin(2 * np.pi * (frequency * 2.73) * t)
                path = self.root / f'data/raw/{identity}.wav'
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio, 16000, subtype='PCM_16')
                lab = path.with_suffix('.lab')
                lab.write_text('0.10 125 pau\n0.70 125 ah\n0.78 125 ax\n1.30 125 eh\n1.50 125 t\n2.20 125 iy\n2.40 125 pau\n')
                recordings.append({'split': split, 'speaker_id': speaker, 'prompt_id': split + '_prompt',
                                   'recording_id': identity, 'audio_path': path.relative_to(self.root).as_posix(),
                                   'lab_path': lab.relative_to(self.root).as_posix(),
                                   'audio_sha256': s.digest(path), 'lab_sha256': s.digest(lab)})
        if invalid_recording:
            invalid = dict(recordings[0], recording_id='invalid', lab_path='data/raw/invalid.lab')
            (self.root / invalid['lab_path']).write_text('3.0 125 ax\n')
            invalid['lab_sha256'] = s.digest(self.root / invalid['lab_path'])
            recordings.append(invalid)
        s.write_csv(self.root / 'data/manifests/recordings.csv', recordings)
        s.write_json(self.root / 'data/manifests/source_selection.json', {'fixture': True})
        s.make_dataset(argparse.Namespace(seed=42))
        return s.read_csv(self.root / 'data/manifests/dataset.csv')

    def test_phone_boundaries_and_padding(self):
        phones, zero = s.parse_phones('0.1 125 ah\n0.1 125 ax\n0.3 125 eh\n', .3)
        self.assertEqual(zero, 1)
        self.assertEqual([p['phone'] for p in phones], ['ah', 'eh'])
        for text in ('0.2 125 ah\n0.1 125 ax\n', '0.4 125 ax\n', 'nan 125 ax\n'):
            with self.assertRaises(ValueError):
                s.parse_phones(text, .3)
        audio = np.arange(10, dtype=np.float32)
        np.testing.assert_array_equal(s.crop_window(audio, -3, 8), [0, 0, 0, 0, 1, 2, 3, 4])
        np.testing.assert_array_equal(s.crop_window(audio, 8, 8), [8, 9, 0, 0, 0, 0, 0, 0])

    def test_dataset_quarantine_and_integrity(self):
        rows = self.fixture(invalid_recording=True)
        report = s.read_json(self.root / 'reports/dataset_audit.json')
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['quarantined_recordings']), 1)
        self.assertFalse(any(r['recording_id'] == 'invalid' for r in rows))
        for row in rows:
            start, end = int(row['window_start_sample']) / 16000, int(row['window_end_sample']) / 16000
            if row['label'] == '1':
                self.assertLessEqual(start, .70)
                self.assertGreaterEqual(end, .78)
            else:
                self.assertFalse(start < .78 and end > .70)
        original = s.digest(self.root / 'data/manifests/dataset.csv')
        s.make_dataset(argparse.Namespace(seed=42))
        self.assertEqual(original, s.digest(self.root / 'data/manifests/dataset.csv'))
        path = self.root / rows[0]['path']
        values, _ = s.load_audio(path)
        sf.write(path, values * .5, 16000, subtype='PCM_16')
        with self.assertRaisesRegex(ValueError, 'Audit failed'):
            s.audit_dataset()

    def test_cross_split_leakage_is_rejected(self):
        rows = self.fixture()
        rows[-1]['prompt_id'] = rows[0]['prompt_id']
        manifest = self.root / 'data/manifests/dataset.csv'
        s.write_csv(manifest, rows)
        metadata_path = self.root / 'data/manifests/dataset_metadata.json'
        metadata = s.read_json(metadata_path)
        metadata['manifest_sha256'] = s.digest(manifest)
        s.write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, 'prompt_id'):
            s.audit_dataset()

    def test_cached_stage_keeps_full_audit_and_rechecks_labels(self):
        self.fixture()
        path = self.root / 'reports/dataset_audit.json'
        full = s.read_json(path)
        s.audit_dataset(verify_wavs=False)
        self.assertEqual(s.read_json(path), full)
        self.assertFalse(s.read_json(self.root / 'reports/manifest_audit.json')['generated_wavs_rechecked'])
        s.write_json(path, dict(full, passed=False))
        with self.assertRaisesRegex(ValueError, 'full WAV audit'):
            s.audit_dataset(verify_wavs=False)
        s.write_json(path, full)
        rec = s.read_csv(self.root / 'data/manifests/recordings.csv')[0]
        (self.root / rec['lab_path']).write_text('2.40 125 ah\n')
        with self.assertRaisesRegex(ValueError, 'Audit failed'):
            s.audit_dataset(verify_wavs=False)

    def test_denoise_preserves_timeline_and_weak_signal(self):
        rng = np.random.default_rng(42)
        audio = rng.normal(0, .003, 24000).astype(np.float32)
        time_axis = np.arange(8000) / 16000
        audio[8000:16000] += (.018 * np.sin(2 * np.pi * 480 * time_axis)).astype(np.float32)
        filtered, status = s.denoise_audio(audio)
        self.assertEqual(status, 'filtered')
        self.assertEqual(len(filtered), len(audio))
        self.assertTrue(np.isfinite(filtered).all())
        self.assertGreater(np.sqrt(np.mean(filtered[8500:15500] ** 2)), .006)
        self.assertLess(np.mean(filtered[:7500] ** 2), np.mean(audio[:7500] ** 2))
        values = (.1 * np.sin(2 * np.pi * 480 * np.arange(8000) / 16000)).astype(np.float32)
        stable, status = s.denoise_audio(values)
        self.assertEqual(status, 'skipped_no_reliable_background')
        np.testing.assert_array_equal(stable, values)

    def test_train_only_normalization_checkpoint_and_prediction(self):
        rows = self.fixture()
        run = self.root / 'runs/baseline'
        args = argparse.Namespace(device='cpu', threads=2, batch_size=4, epochs=1, patience=1,
                                  lr=.001, seed=42, denoise='off', run_dir=run, resume=False)
        with mock.patch.object(s, 'feature_array', wraps=s.feature_array) as features:
            s.train(args)
            self.assertEqual([{r['split'] for r in call.args[0]} for call in features.call_args_list],
                             [{'train'}, {'validation'}])
        checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
        train_rows = [r for r in rows if r['split'] == 'train']
        raw = s.feature_array(train_rows)
        np.testing.assert_allclose(checkpoint['normalization']['mean'], raw.mean(axis=(0, 2), dtype=np.float64), rtol=1e-6)
        self.assertEqual(checkpoint['threshold_selection_split'], 'validation')
        detector = s.SchwaDetector()
        test_row = next(r for r in rows if r['split'] == 'test')
        feature = s.feature_array([test_row])
        expected = s.probabilities(detector.model, ((feature - detector.mean) / detector.std).astype(np.float32))[0]
        result = detector.predict(self.root / test_row['path'])
        self.assertAlmostEqual(result['score'], expected, places=6)
        stereo = np.stack([np.sin(np.arange(48000) * .07), np.sin(np.arange(48000) * .09)], axis=1) * .05
        long_audio = self.root / 'long_stereo.wav'
        sf.write(long_audio, stereo, 8000)
        result = detector.predict(long_audio)
        self.assertEqual(result['duration_seconds'], 6)
        with self.assertRaisesRegex(ValueError, 'Changed or unverified feature cache'):
            cache = next(path for path in (self.root / 'data/features').glob('*.npy')
                         if np.array_equal(np.load(path, allow_pickle=False), raw))
            with cache.open('ab') as handle:
                handle.write(b'changed')
            # Select the cache's original row set by its stored key.
            s.feature_array(train_rows)

    def test_undefined_group_metrics_have_reasons(self):
        metrics = s.binary_metrics([0, 0], [.1, .2], .5)
        for name in ('recall', 'f1', 'precision', 'auroc', 'average_precision', 'fnr'):
            self.assertIsNone(metrics[name])
            self.assertIn(name, metrics['undefined_metrics'])
        with self.assertRaisesRegex(ValueError, 'both classes'):
            s.choose_threshold([0, 0], [.1, .2])

    def test_cli_prints_detection_and_window_times(self):
        result = {'audio': 'recording.wav', 'detected': True, 'score': .7, 'threshold': .5,
                  'duration_seconds': 1.0, 'windows': [{'start_seconds': .1, 'end_seconds': .6, 'score': .7}]}
        with mock.patch.object(s, 'SchwaDetector') as detector, mock.patch('sys.stdout', new_callable=io.StringIO) as output:
            detector.return_value.predict.return_value = result
            s.predict_cli(argparse.Namespace(model=None, inspect=False, audio=['recording.wav'], hop=.05, json=False, output=None))
            self.assertIn('0.10–0.60s', output.getvalue())
            self.assertIn('score=0.7000', output.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)

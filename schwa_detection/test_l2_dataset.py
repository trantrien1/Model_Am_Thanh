"""Guard the distinction between actual schwa, stressed vowels and uncertain speech."""
import unittest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch
import l2_dataset as data


class ManualAnnotationChecks(unittest.TestCase):
    def test_frozen_v5_reuses_without_deleted_legacy_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'v5'
            manifests = destination / 'data/manifests'
            manifests.mkdir(parents=True)
            (manifests / 'dataset.csv').write_text('frozen dataset')
            data.core.write_json(manifests / 'source_selection.json', {
                'source_revision': data.REVISION, 'source_sha256': data.SOURCE_SHA,
                'seed': 20261012, 'window_seed': data.SEED})
            with patch.object(data, 'DESTINATION', destination), patch.object(data.core, 'ROOT'), \
                    patch.object(data.core, 'make_dataset') as audit, \
                    patch.object(data, 'source_rows', side_effect=AssertionError('Frozen reuse must not read source parquet')):
                data.prepare(argparse.Namespace(version='v5'))
                self.assertEqual(audit.call_args.args[0].seed, data.SEED)

    def test_stress_and_rhotic_vowels_remain_distinct(self):
        for raw in ('AH0', 'AH0*', 'AX', 'AX0'):
            self.assertEqual(data.perceived_phone({'perceived_raw': raw}), 'ax')
        for raw in ('AH', 'AH1', 'AH2'):
            self.assertEqual(data.perceived_phone({'perceived_raw': raw}), 'ah')
        self.assertEqual(data.perceived_phone({'perceived_raw': 'ER0'}), 'er')
        self.assertEqual(data.perceived_phone({'perceived_raw': 'AXR'}), 'axr')

    def test_uses_perceived_phone_for_substitution_and_insertion(self):
        self.assertEqual(data.perceived_phone({'canonical_raw': 'ER0', 'perceived_raw': 'AX',
                                              'error_type': 'substitution'}), 'ax')
        self.assertEqual(data.perceived_phone({'canonical_raw': 'sil', 'perceived_raw': 'AX',
                                              'error_type': 'addition'}), 'ax')
        self.assertEqual(data.perceived_phone({'canonical_raw': 'AH0', 'perceived_raw': 'sil',
                                              'error_type': 'deletion'}), 'pau')

    def test_unknown_intervals_are_not_silent_negatives(self):
        self.assertEqual(data.perceived_phone({'perceived_raw': 'err'}), 'unk')
        phones = data.convert_events([{'start': .1, 'end': .2, 'perceived_raw': 'AH0'}], .3)
        self.assertEqual([p['phone'] for p in phones], ['unk', 'ax', 'unk'])
        self.assertAlmostEqual(phones[0]['end'], .1)
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            data.convert_events([{'start': 0, 'end': .2, 'perceived_raw': 'AH0'},
                                 {'start': .1, 'end': .3, 'perceived_raw': 'S'}], .3)

    def test_resampling_rounding_does_not_invent_unknown_speech(self):
        events = [{'start': 0, 'end': .1, 'perceived_raw': 'AH0'},
                  {'start': .1, 'end': .2, 'perceived_raw': 'sil'}]
        fixed = data.convert_events(events, .20004, sample_rounding=True)
        self.assertEqual([p['phone'] for p in fixed], ['ax', 'pau'])
        self.assertEqual(fixed[-1]['end'], .20004)
        self.assertEqual(data.convert_events(events, .20004)[-1]['phone'], 'unk')
        self.assertEqual(data.convert_events(events, .201, sample_rounding=True)[-1]['phone'], 'unk')
        self.assertEqual(data.convert_events(events[:1], .10004, sample_rounding=True)[-1]['phone'], 'unk')


if __name__ == '__main__':
    unittest.main(verbosity=2)

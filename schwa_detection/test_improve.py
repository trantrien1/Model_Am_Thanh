"""Verify phonetic timestamps, temporal pooling and class-imbalance safeguards."""
import argparse
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import improve as improved
import schwa as core
import test_schwa as fixtures


class TransferChecks(unittest.TestCase):
    def test_joint_group_normalization_preserves_embedding_scale(self):
        torch.manual_seed(42)
        values = torch.cat([torch.randn(3, 24, 32) * .2, torch.randn(3, 24, 8) * 3 - 5], dim=-1)
        norm = improved.GroupFeatureNorm([32, 8])
        normalized = norm(values)
        scaled = values.clone()
        scaled[..., 32:] = scaled[..., 32:] * 100 + 1000
        torch.testing.assert_close(norm(scaled)[..., :32], normalized[..., :32], rtol=0, atol=0)
        self.assertGreater(normalized[..., :32].std().item(), .99)
        self.assertGreater(normalized[..., 32:].std().item(), .99)
        for temporal in ['none', 'gru', 'conv']:
            model = improved.PhoneHead(classes=2, input_dim=40, hidden=16, temporal=temporal, group_sizes=[32, 8])
            logits = model(values)
            loss = improved.window_presence_loss(logits, 1, torch.tensor([1., 0., 1.]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            improved.PhoneHead(classes=2, input_dim=40, group_sizes=[32, 9])

    def test_balanced_precision_recall_selection_can_reject_unbalanced_f1_peak(self):
        labels = [1] * 12 + [0] * 38
        scores = [.9] * 8 + [.6] + [.1] * 3 + [.7] * 3 + [.2] * 15 + [.01] * 20
        f1 = improved.validation_selection(labels, scores)
        balanced = improved.validation_selection(labels, scores, 'min_pr')
        self.assertGreater(min(balanced['precision'], balanced['recall']), min(f1['precision'], f1['recall']))
        self.assertEqual(balanced['precision'], .75)
        self.assertEqual(balanced['recall'], .75)

    def test_joint_representation_cannot_reuse_logits_only_feature_cache(self):
        rows = [{'audio_sha256': 'same-audio'}]
        with mock.patch.object(improved, 'CONTEXT', 'window'), \
             mock.patch.object(core, 'digest', return_value='same-backbone'), \
             mock.patch.object(improved, '_backbone_geometry', return_value=(24, 392, 'math', 1024)):
            with mock.patch.object(improved, 'REPRESENTATION', 'stored'):
                old = improved.cache_paths(Path('dataset'), rows, 'test')
            with mock.patch.object(improved, 'REPRESENTATION', 'joint'):
                new = improved.cache_paths(Path('dataset'), rows, 'test')
                self.assertEqual(improved.backbone_dimension(), 1416)
        self.assertNotEqual(old, new)

    def test_attention_runtime_change_cannot_reuse_old_window_features(self):
        rows = [{'audio_sha256': 'fixed-audio'}]
        with mock.patch.object(improved, 'CONTEXT', 'window'), \
             mock.patch.object(core, 'digest', return_value='fixed-backbone'), \
             mock.patch.object(improved, '_backbone_geometry', return_value=(24, 392, None)):
            original = improved.cache_paths(Path('dataset'), rows, 'test')
        with mock.patch.object(improved, 'CONTEXT', 'window'), \
             mock.patch.object(core, 'digest', return_value='fixed-backbone'), \
             mock.patch.object(improved, '_backbone_geometry', return_value=(24, 392, 'math')):
            stable = improved.cache_paths(Path('dataset'), rows, 'test')
        self.assertNotEqual(original, stable)

    def test_frame_targets_preserve_real_timeline(self):
        fixture = fixtures.SchwaChecks()
        fixture.setUp()
        try:
            rows = fixture.fixture()
            targets, vocabulary = improved.frame_labels(fixture.root, rows)
            for index, row in enumerate(rows):
                start = int(row['window_start_sample']) / 16000
                centers = start + (199.5 + np.arange(24) * 320) / 16000
                expected_ax = (centers >= .705) & (centers <= .775)
                np.testing.assert_array_equal(targets[index] == vocabulary.index('ax'), expected_ax)
                if start < 0:
                    padded_frames = centers < 0
                    self.assertTrue(np.all(targets[index][padded_frames] == vocabulary.index('pau')))
        finally:
            fixture.doCleanups()

    def test_pooling_reduces_one_frame_spikes(self):
        class ControlledHead(torch.nn.Module):
            def forward(self, values):
                probabilities = torch.full((len(values), 24, 2), .1)
                probabilities[:, :, 0] = .9
                probabilities[:, 10, 0] = .1
                probabilities[:, 10, 1] = .9
                return probabilities.log()
        values = np.zeros((2, 24, 768), dtype=np.float16)
        one_frame = improved.class_scores(ControlledHead(), values, 1, pool_frames=1)
        two_frames = improved.class_scores(ControlledHead(), values, 1, pool_frames=2)
        np.testing.assert_allclose(one_frame, .9, atol=1e-6)
        np.testing.assert_allclose(two_frames, .5, atol=1e-6)

    def test_threshold_does_not_claim_majority_accuracy_as_success(self):
        labels = [0] * 95 + [1] * 5
        scores = np.concatenate([np.linspace(.01, .2, 95), np.linspace(.11, .16, 5)])
        selected = improved.validation_selection(labels, scores)
        self.assertGreater(selected['recall'], 0)
        self.assertGreater(selected['f1'], 0)
        self.assertLess(selected['accuracy'], .95)

    def test_phone_head_shape_and_finite_loss(self):
        torch.manual_seed(42)
        model = improved.PhoneHead(classes=41)
        inputs = torch.randn(4, 24, 768)
        targets = torch.randint(0, 41, (4, 24))
        targets[:, 0] = -100
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, ignore_index=-100)
        self.assertEqual(tuple(logits.shape), (4, 24, 41))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_linear_head_preserves_phone_probabilities(self):
        head = improved.PhoneHead(classes=41, hidden=0).eval()
        values = torch.randn(2, 24, 768)
        probabilities = torch.softmax(head(values), dim=-1)
        self.assertEqual(tuple(probabilities.shape), (2, 24, 41))
        torch.testing.assert_close(probabilities.sum(dim=-1), torch.ones(2, 24))

    def test_temporal_heads_learn_frame_and_presence_objectives(self):
        for temporal in ('conv', 'gru'):
            head = improved.PhoneHead(classes=2, hidden=32, temporal=temporal)
            logits = head(torch.randn(2, 24, 768))
            self.assertEqual(tuple(logits.shape), (2, 24, 2))
            truth = torch.randint(0, 2, (2, 24))
            truth[:, 0] = -100
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), truth)
            probability = torch.softmax(logits, dim=-1)[:, :, 1].amax(dim=1)
            loss += .5 * torch.nn.functional.binary_cross_entropy(probability, torch.tensor([0., 1.]))
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))

    def test_spectral_frames_and_joint_normalization(self):
        spectral = improved.AlignedMel()
        values = spectral(torch.randn(3, 8000))
        self.assertEqual(tuple(values.shape), (3, 24, 40))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(torch.all(spectral.filters.sum(dim=1) > 0))
        metadata = {'mean': [.5] * 40, 'std': [2.] * 40}
        acoustic = np.zeros((3, 24, 768), dtype=np.float16)
        mel = np.ones((3, 24, 40), dtype=np.float16)
        joint = improved.AugmentedFeatures(acoustic, mel, metadata)
        subset = joint[np.array([2, 0])]
        self.assertEqual(subset.shape, (2, 24, 808))
        np.testing.assert_allclose(subset[:, :, 768:], .25)
        head = improved.PhoneHead(classes=41, input_dim=808)
        self.assertEqual(tuple(head(torch.from_numpy(subset)).shape), (2, 24, 41))

    def test_joined_caches_preserve_shuffled_sample_order(self):
        a = np.arange(4 * 24 * 40, dtype=np.float16).reshape(4, 24, 40)
        b = np.full((3, 24, 40), -2, dtype=np.float16)
        joined = improved.JoinedFeatures([(a, [3, 1]), (b, [2, 0])])
        expected = np.stack([a[3], a[1], b[2], b[0]])
        np.testing.assert_array_equal(joined[np.array([3, 0, 2, 1])], expected[[3, 0, 2, 1]])
        np.testing.assert_array_equal(joined[1:3], expected[1:3])
        with self.assertRaises(IndexError):
            joined[4]
        mean, std = improved.spectral_statistics(joined)
        truth = expected.astype(np.float64)
        np.testing.assert_allclose(mean, truth.mean(axis=(0, 1)), atol=1e-3)
        np.testing.assert_allclose(std, truth.std(axis=(0, 1)), atol=1e-3)

    def test_prosody_pitch_silence_and_gain_behavior(self):
        extractor = improved.AlignedMel(improved.spectral_config(True))
        time = torch.arange(8000) / 16000
        waves = torch.stack([.2 * torch.sin(2 * torch.pi * 100 * time),
                             .1 * torch.sin(2 * torch.pi * 200 * time),
                             torch.zeros_like(time)])
        values = extractor(waves)
        self.assertEqual(tuple(values.shape), (3, 24, 48))
        self.assertTrue(torch.isfinite(values).all())
        pitch = values[:2, :, 40].exp()
        torch.testing.assert_close(pitch[0], torch.full((24,), 100.), atol=1., rtol=0)
        torch.testing.assert_close(pitch[1], torch.full((24,), 200.), atol=1., rtol=0)
        self.assertTrue(torch.all(values[:2, :, 47] == 1))
        self.assertTrue(torch.all(values[2, :, 42] == 0))
        self.assertTrue(torch.all(values[2, :, 47] == 0))
        quieter = extractor(waves / 2)
        torch.testing.assert_close(values[:2, :, [40, 41, 42, 44, 45, 46, 47]],
                                   quieter[:2, :, [40, 41, 42, 44, 45, 46, 47]], atol=1e-5, rtol=1e-5)
        self.assertEqual(len(improved.spectral_statistics(values.numpy().astype(np.float16))[0]), 48)

    def test_native_augmentation_excludes_reserved_prompts(self):
        selection = {'prompt_groups': {'validation': ['held'], 'test': ['held_test']}}
        native = [{'split': 'train', 'sample_id': 'n0', 'prompt_id': 'held', 'speaker_id': 'bdl',
                   'pcm_sha256': 'a', 'audio_sha256': 'h0'},
                  {'split': 'train', 'sample_id': 'n1', 'prompt_id': 'free', 'speaker_id': 'bdl',
                   'pcm_sha256': 'b', 'audio_sha256': 'h1'}]
        foreign = [{'speaker_id': 'THV', 'pcm_sha256': 'f'}]
        previous_root = core.ROOT
        with mock.patch.object(core, 'read_json', return_value=selection), \
             mock.patch.object(core, 'audit_dataset', return_value=native), \
             mock.patch.object(core, 'digest', return_value='verified'), \
             mock.patch.object(improved, 'feature_cache', return_value=np.zeros((2, 24, 768), dtype=np.float16)):
            rows, parts, _, metadata = improved.native_augmentation(improved.V3, foreign)
        self.assertEqual([r['sample_id'] for r in rows], ['n1'])
        self.assertEqual(parts[0][1], [1])
        self.assertEqual(metadata['files'], 1)
        self.assertEqual(core.ROOT, previous_root)

    def test_waveform_normalization_is_independent_of_batch_composition(self):
        a, b = torch.randn(1, 8000), 10 * torch.randn(1, 8000) + 3
        alone = improved.normalized_waveforms(a, True)
        together = improved.normalized_waveforms(torch.cat([a, b]), True)[0:1]
        torch.testing.assert_close(alone, together, rtol=0, atol=0)
        self.assertIs(improved.normalized_waveforms(a, False), a)
        head = improved.PhoneHead(classes=41, input_dim=1064)
        self.assertEqual(tuple(head(torch.randn(2, 24, 1064)).shape), (2, 24, 41))

    def test_cpu_training_subset_is_balanced_reproducible_and_train_only(self):
        rows = [{'sample_id': f'{speaker}_{label}_{i}', 'speaker_id': speaker, 'label': str(label), 'split': 'train'}
                for speaker in ('a', 'b', 'c') for label in (0, 1) for i in range(10)]
        first = improved.select_training_rows(rows, 12, 42)
        self.assertEqual(first, improved.select_training_rows(rows, 12, 42))
        self.assertNotEqual(first, improved.select_training_rows(rows, 12, 43))
        for speaker in ('a', 'b', 'c'):
            for label in ('0', '1'):
                self.assertEqual(sum(r['speaker_id'] == speaker and r['label'] == label for r in first), 2)
        with self.assertRaises(ValueError):
            improved.select_training_rows([dict(r, split='test') for r in rows], 12)
        self.assertIs(improved.select_training_rows(rows), rows)

    def test_pretrained_schwa_initialization_uses_only_bare_schwa_logit(self):
        head = improved.PhoneHead(classes=2, hidden=0, dropout=0, input_dim=392)
        state = {'feature_kind': 'ctc_logits', 'phoneme_vocabulary': {'\u0259': 7, '\u025a': 43}}
        improved.initialize_ctc_head(head, state)
        self.assertEqual(torch.count_nonzero(head.network[-1].weight).item(), 1)
        self.assertEqual(head.network[-1].weight[1, 7].item(), .25)
        background = torch.full((1, 24, 392), -10.)
        background[:, :, 0] = 10.
        schwa = background.clone()
        schwa[:, 12, 7], schwa[:, 12, 0] = 10., -10.
        self.assertGreater(improved.class_scores(head, schwa.numpy(), 1)[0],
                           improved.class_scores(head, background.numpy(), 1)[0] + .4)
        with self.assertRaises(ValueError):
            improved.initialize_ctc_head(head, dict(state, feature_kind='embeddings'))

    def test_presence_loss_matches_probabilities_and_keeps_extreme_gradients(self):
        torch.manual_seed(42)
        logits = torch.randn(3, 24, 5, requires_grad=True)
        labels = torch.tensor([0., 1., 0.])
        expected = torch.nn.functional.binary_cross_entropy(logits.softmax(-1)[..., 2].amax(1), labels)
        actual = improved.window_presence_loss(logits, 2, labels)
        torch.testing.assert_close(actual, expected)
        extreme = torch.tensor([[[-60., 60.]]], requires_grad=True)
        loss = improved.window_presence_loss(extreme, 1, torch.tensor([0.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(extreme.grad, torch.tensor([[[-1., 1.]]]))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)

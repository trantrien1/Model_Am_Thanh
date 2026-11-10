import unittest
import io

import numpy as np
import torch

import whisper_word_features as features
from whisper_word_head import WordAwareHead
from whisper_train_cpu import step_epoch, training_snapshot, restore_training


class WhisperWordFeatureTests(unittest.TestCase):
    def test_audio_vocabulary_has_no_validation_or_annotation_inputs(self):
        words = {'train_2': [{'word': ' The,'}, {'word': 'ONLYONCE'}],
                 'train_1': [{'word': 'the'}, {'word': 'A'}, {'word': 'a.'}]}
        self.assertEqual(features.build_word_vocabulary(words), {'A': 1, 'THE': 2})
        with self.assertRaises(ValueError):
            features.window_path(features.ROOT / 'v5', [{'split': 'test'}], 'test', {})

    def test_physical_focal_centers_word_overlap_and_unknown(self):
        words = [{'word': 'A', 'start': 0., 'end': .15, 'probability': .8},
                 {'word': 'THE', 'start': .20, 'end': .4, 'probability': .9},
                 {'word': 'unseen', 'start': .6, 'end': .8, 'probability': .99}]
        starts = [-16000, 0, 4800, 9600]
        actual = features.focal_word_ids(words, starts, {'A': 1, 'THE': 2})
        self.assertEqual(actual.dtype, np.int32)
        self.assertTrue(np.all(actual[0] == 0))
        self.assertEqual(actual[1, 0, 0], 1)   # .01246875 seconds, in first word
        self.assertEqual(actual[1, 6, 0], 2)   # .13246875 seconds, confidence wins overlap
        self.assertEqual(actual[2, 0, 0], 2)   # .31246875 seconds
        self.assertTrue(np.all(actual[3] == 0)) # unseen word remains unknown

    def test_word_ids_above_float16_range_and_embedding_gradients(self):
        torch.set_num_threads(2)
        config = {'classes': 2, 'hidden': 16, 'dropout': 0., 'temporal': 'gru',
                  'input_dim': 1456, 'group_sizes': [1024, 392, 8, 32]}
        model = WordAwareHead(3002, head_config=config)
        values = torch.randn(1, 24, 1425)
        values[..., -1] = 3001
        model(values).square().mean().backward()
        self.assertGreater(float(model.embedding.weight.grad[3001].abs().sum()), 0.)
        self.assertEqual(float(model.embedding.weight.grad[3000].abs().sum()), 0.)
        self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0.)
        for bad in [-1., 3002., 1.5, float('nan')]:
            values[..., -1] = bad
            with self.assertRaises(ValueError):
                model(values)

    def test_word_embedding_optimizer_and_rng_exact_resume(self):
        torch.set_num_threads(2)
        torch.backends.mkldnn.enabled = False
        torch.use_deterministic_algorithms(True)
        data = np.random.default_rng(8).normal(size=(4, 24, 1425)).astype(np.float32)
        data[..., -1] = np.array([0, 1, 2, 3])[:, None]
        targets = np.tile(np.arange(24, dtype=np.int64) % 2, (4, 1))
        truth = torch.ones(4)
        config = {'classes': 2, 'hidden': 16, 'dropout': .2, 'temporal': 'gru',
                  'input_dim': 1456, 'group_sizes': [1024, 392, 8, 32]}

        def fresh():
            torch.manual_seed(42)
            model = WordAwareHead(4, head_config=config)
            return model, torch.optim.AdamW(model.parameters(), lr=.0005), np.random.default_rng(42)

        def advance(model, optimizer, rng):
            step_epoch(model, optimizer, data, targets, truth, torch.nn.CrossEntropyLoss(),
                       rng, batch_size=2)

        direct, optimizer, rng = fresh()
        advance(direct, optimizer, rng)
        advance(direct, optimizer, rng)
        direct_rng = torch.get_rng_state().clone()
        direct_numpy = rng.bit_generator.state
        resumed, optimizer, rng = fresh()
        advance(resumed, optimizer, rng)
        buffer = io.BytesIO()
        torch.save(training_snapshot(resumed, optimizer, rng, [{}], (0.,), 1, 0., {'fixed': True}), buffer)
        buffer.seek(0)
        saved = torch.load(buffer, weights_only=True)
        resumed, optimizer, rng = fresh()
        restore_training(saved, resumed, optimizer, rng, {'fixed': True})
        advance(resumed, optimizer, rng)
        for expected, actual in zip(direct.parameters(), resumed.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), direct_rng))
        self.assertEqual(rng.bit_generator.state, direct_numpy)


if __name__ == '__main__':
    unittest.main()

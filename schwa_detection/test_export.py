"""Check that exported inference has no training-workspace dependency."""
import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

import export_inference as exporter


class ExportChecks(unittest.TestCase):
    def test_inference_class_has_no_external_backbone_dependency(self):
        definitions = exporter.definitions(exporter.ROOT / 'improve.py', {'PhoneHead', 'TransferDetector', 'predict'})
        definitions += exporter.definitions(exporter.ROOT / 'frozen_phoneme.py',
                                            {'swap_dynamic_linears', 'rebuild_acoustic', 'acoustic_features'})
        source = '\n\n'.join(definitions)
        compile(source, '<standalone inference>', 'exec')
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn('core', names)
        self.assertNotIn('BACKBONE', names)
        self.assertIn('load_audio', names)
        self.assertIn('wav2vec2_model', {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)})
        self.assertIn('Checkpoint is not self-contained', source)

    def test_all_audio_functions_are_exported(self):
        names = {'digest', 'write_json', 'load_audio', 'crop_window', 'denoise_audio', 'denoise_cli'}
        source = '\n\n'.join(exporter.definitions(exporter.ROOT / 'schwa.py', names))
        actual = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(names, actual)

    def test_validation_accuracy_cannot_be_exported_as_test_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'packed.pt'
            torch.save({'backend': 'wav2vec2_phone_head', 'backbone': {'layers': 6, 'source_bundle': 'torchaudio.WAV2VEC2_BASE'},
                        'settings': {'task': 'phones'}}, model)
            metrics = root / 'metrics.json'
            metrics.write_text(json.dumps({'split': 'validation', 'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}))
            output = root / 'bundle'
            with self.assertRaisesRegex(ValueError, 'held-out test'):
                exporter.export(model, output, metrics)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.zip').exists())

    def test_old_runtime_accuracy_cannot_be_attached_to_new_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'packed.pt'
            torch.save({'backend': 'wav2vec2_phone_head',
                        'backbone': {'layers': 24, 'source_bundle': 'huggingface.phoneme'},
                        'settings': {'task': 'phones', 'feature_context': 'recording'}}, model)
            metrics = root / 'metrics.json'
            metrics.write_text(json.dumps({'split': 'test', 'feature_context': 'recording',
                                          'acoustic_runtime_recipe': {'version': 1},
                                          'model_sha256': hashlib.sha256(model.read_bytes()).hexdigest()}))
            output = root / 'bundle'
            with self.assertRaisesRegex(ValueError, 'runtime recipe'):
                exporter.export(model, output, metrics)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.zip').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)

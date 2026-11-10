"""Fresh CPU Schwa GRU over frozen XLSR, Whisper words and Whisper public text embeddings."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

import improve
import schwa as core
import whisper_lexical as words
import whisper_text_features as audio_features
from lexical_train_cpu import CombinedFeatures, array_cache, selection_key
from whisper_train_cpu import step_epoch, restore_training, training_snapshot

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/pr88_v5_whisper_text_binary_gru'
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
SOURCES = ['whisper_text_train_cpu.py', 'whisper_text_features.py', 'whisper_word_features.py', 'whisper_train_cpu.py',
           'whisper_lexical.py', 'whisper_cpu.py', 'lexical_train_cpu.py', 'lexical_context.py',
           'improve.py', 'frozen_phoneme.py', 'schwa.py']
SETTINGS = {'threads': 2, 'seed': 42, 'epochs': 40, 'patience': 10,
            'lr': .0005, 'batch_size': 64, 'window_loss': .5, 'hidden': 128, 'dropout': .3}
CONFIG = {'classes': 2, 'hidden': 128, 'dropout': .3, 'temporal': 'gru',
          'input_dim': 1936, 'group_sizes': [1024, 392, 8, 512]}


def provenance():
    if core.digest(ROOT / 'v5/data/manifests/dataset.csv') != audio_features.DATASET_SHA:
        raise ValueError('Frozen V5 data changed.')
    return {'dataset': 'v5', 'dataset_sha256': audio_features.DATASET_SHA,
            'backbone_sha256': core.digest(BACKBONE), 'whisper_words': words.metadata(),
            'whisper_text': audio_features.metadata(), 'settings': SETTINGS, 'head_config': CONFIG,
            'source_code_sha256': {name: core.digest(ROOT / name) for name in SOURCES},
            'initialization': 'Original public frozen XLSR and Whisper; fresh random binary GRU',
            'hypothesis': 'Fixed public BPE token embeddings may generalize lexical conditioning to words not observed in TRAIN; all labels remain unchanged',
            'train_regions': 9702, 'validation_regions': 4884, 'test_regions': 5157,
            'threshold_selection_split': 'validation', 'selection_objective': 'min_precision_recall',
            'target': 'Strict AH0/AX; AH1/AH2 and ER/AXR negative',
            'frame_supervision': 'Original manual TRAIN labels only; 5ms phone-boundary margin',
            'head_context': 'Original 0.5-second focal region', 'training_mkldnn': False,
            'test_used_for_selection': False, 'test_features_computed': False}


def register():
    value = provenance()
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / 'registered_protocol.json'
    if path.exists():
        if core.read_json(path)['provenance'] != value:
            raise ValueError('Preserving registration with different recipe.')
    else:
        if (RUN / 'best.pt').exists():
            raise ValueError('Preserve existing unregistered weights.')
        core.write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
                              'provenance': value, 'validation_classifier_predictions_before_registration': False,
                              'objective': {'precision_above': .88, 'recall_above': .88, 'accuracy_at_least': .9},
                              'completion_requires': ['actual_runtime_validation', 'freeze_before_full_test',
                                    'all_original_test_regions', 'actual_final_checkpoint_four_file_bundle_parity']})
        for name in SOURCES:
            (RUN / ('registered_' + name)).write_bytes((ROOT / name).read_bytes())
    print('Registered fresh 1936-input Whisper public BPE text GRU', flush=True)


def load_features(directory, rows, split, value):
    acoustic = improve.feature_cache(directory, rows, split)
    path, identity = words.cache_path(directory, rows, split, value['whisper_words'])
    lexical = array_cache(path, identity)
    path, identity = audio_features.window_path(directory, rows, split, value['whisper_text'])
    encoded = array_cache(path, identity)
    if lexical.shape != (len(rows), 24, 8) or encoded.shape != (len(rows), 24, 512):
        raise ValueError('Full original public text/lexical feature geometry differs.')
    return CombinedFeatures(CombinedFeatures(acoustic, lexical), encoded)


def guard(value):
    if provenance() != value:
        raise ValueError('Registered source, data or feature recipe changed.')


def train(resume):
    value = core.read_json(RUN / 'registered_protocol.json')['provenance']
    guard(value)
    if not resume and ((RUN / 'best.pt').exists() or (RUN / 'last.pt').exists()):
        raise ValueError('Preserve trained state; use --resume.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    directory = improve.activate_dataset('v5')
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = BACKBONE, 'recording', 'joint'
    rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    train_rows = [r for r in rows if r['split'] == 'train']
    val_rows = [r for r in rows if r['split'] == 'validation']
    if len(train_rows) != 9702 or len(val_rows) != 4884:
        raise ValueError('Every original train/validation region is required.')
    audio_features.extract(['train', 'validation'])
    guard(value)
    training = load_features(directory, train_rows, 'train', value)
    validation = load_features(directory, val_rows, 'validation', value)
    dense, vocabulary = improve.frame_labels(directory, train_rows)
    ax = vocabulary.index('ax')
    targets = np.where(dense == -100, -100, (dense == ax).astype(np.int64))
    counts = np.bincount(targets[targets >= 0], minlength=2)
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    weights = np.clip(weights / np.median(weights), .25, 3).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), ignore_index=-100)
    truth = torch.tensor([int(r['label']) for r in train_rows], dtype=torch.float32)
    val_truth = [int(r['label']) for r in val_rows]
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    model = improve.PhoneHead(**CONFIG)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SETTINGS['lr'], weight_decay=.01)
    history, best_key, best_epoch, elapsed = [], None, 0, 0.
    if resume:
        saved = torch.load(RUN / 'last.pt', weights_only=True)
        history, best_key, best_epoch, elapsed = restore_training(saved, model, optimizer, rng, value)
        del saved
    core.write_json(RUN / 'training_protocol.json', value)
    started = time.perf_counter()
    for epoch in range(len(history) + 1, SETTINGS['epochs'] + 1):
        guard(value)
        loss = step_epoch(model, optimizer, training, targets, truth, criterion, rng,
                          SETTINGS['batch_size'], SETTINGS['window_loss'])
        choices = [(improve.validation_selection(val_truth,
                    improve.class_scores(model, validation, 1, pool), 'min_pr'), pool) for pool in (1, 2, 3)]
        metric, pool = max(choices, key=lambda choice: selection_key(choice[0]))
        key = selection_key(metric)
        history.append({'epoch': epoch, 'train_loss': loss, 'validation_precision': metric['precision'],
                        'validation_recall': metric['recall'], 'seconds': elapsed + time.perf_counter() - started})
        core.write_csv(RUN / 'history.csv', history)
        if best_key is None or key > tuple(best_key):
            best_key, best_epoch = key, epoch
            core.save_torch(RUN / 'best.pt', {'backend': 'wav2vec2_whisper_acoustic_head',
                'target_phones': ['ax'], 'target_symbol': '\u0259', 'head_config': CONFIG,
                'head_state_dict': model.state_dict(), 'phone_vocabulary': ['non_ax', 'ax'],
                'epoch': epoch, 'threshold': metric['threshold'], 'pool_frames': pool, 'cpu_threads': 2,
                'threshold_selection_split': 'validation', 'validation_metrics': metric,
                'dataset_sha256': value['dataset_sha256'], 'backbone_sha256': value['backbone_sha256'],
                'whisper_words': value['whisper_words'], 'whisper_text': value['whisper_text'],
                'settings': value, 'provenance': value})
            core.write_json(RUN / 'best_summary.json', {**value, 'best_epoch': epoch, 'validation_metrics': metric})
        core.save_torch(RUN / 'last.pt', training_snapshot(model, optimizer, rng, history, best_key,
                         best_epoch, elapsed + time.perf_counter() - started, value))
        core.write_json(RUN / 'status.json', {'status': 'training', 'epoch_completed': epoch,
                                            'best_epoch': best_epoch, 'test_predictions_computed': False})
        print(f'Whisper text GRU epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
        if epoch - best_epoch >= SETTINGS['patience']:
            break
    core.write_json(RUN / 'training_summary.json', {**value, 'best_epoch': best_epoch,
        'epochs_completed': len(history), 'seconds': elapsed + time.perf_counter() - started,
        'validation_metrics': core.read_json(RUN / 'best_summary.json')['validation_metrics']})
    core.write_json(RUN / 'status.json', {'status': 'completed', 'epochs_completed': len(history),
                                        'test_predictions_computed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'train'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    register() if args.command == 'register' else train(args.resume)

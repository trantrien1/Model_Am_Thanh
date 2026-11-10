"""Fresh CPU Schwa GRU with the complete automatic pronunciation posterior."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

import improve
import lexical_train_cpu as lexical
import phonetic_prior_features as encoded
import schwa as core
from whisper_train_cpu import step_epoch, restore_training, training_snapshot

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/pr88_v5_full_prior_binary_gru'
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
DESIGN = ROOT / 'reports/precision_recall_88_full_prior_classifier_design.json'
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
SOURCES = ['full_prior_train_cpu.py', 'improve.py', 'lexical_train_cpu.py',
           'lexical_context.py', 'frozen_phoneme.py', 'schwa.py', 'whisper_train_cpu.py', 'phonetic_prior.py', 'phonetic_prior_features.py']
SETTINGS = {'threads': 2, 'seed': 42, 'epochs': 40, 'patience': 10,
            'lr': .0005, 'batch_size': 64, 'window_loss': .5, 'hidden': 128, 'dropout': .3}
CONFIG = {'classes': 2, 'hidden': 128, 'dropout': .3, 'temporal': 'gru',
          'input_dim': 1812, 'group_sizes': [1024, 392, 4, 392]}


def source_hashes():
    return {name: core.digest(ROOT / name) for name in SOURCES}


def provenance():
    return {'dataset': 'v5', 'dataset_sha256': DATASET_SHA,
            'backbone_sha256': core.digest(BACKBONE), 'lexical': lexical.metadata(BACKBONE),
            'phonetic_prior': encoded.metadata(), 'settings': SETTINGS, 'head_config': CONFIG,
            'candidate_design_sha256': core.digest(DESIGN), 'source_code_sha256': source_hashes(),
            'initialization': 'Original public frozen XLSR and English CTC encoders; fresh binary GRU',
            'hypothesis': 'Complete audio-inferred pronunciation posterior may retain phonetic context lost by four-channel compression',
            'train_regions': 9702, 'validation_regions': 4884, 'test_regions': 5157,
            'target': 'Strict AH0/AX; AH1/AH2 and ER/AXR negative',
            'frame_supervision': 'Original manual TRAIN labels only; 5ms boundary margin',
            'head_context': 'Original 0.5-second focal region', 'feature_context': 'recording',
            'selection_objective': 'min_precision_recall', 'threshold_selection_split': 'validation',
            'training_mkldnn': False, 'test_used_for_selection': False, 'test_features_computed': False}


def register():
    design = core.read_json(DESIGN)
    if design['head_config'] != CONFIG or design['dataset_sha256'] != DATASET_SHA:
        raise ValueError('Classifier differs from the fixed design.')
    audit = core.read_json(ROOT / 'reports/phonetic_prior_feature_audit.json')
    if not audit['passed'] or audit['probe_only'] or {r['split']: r['regions'] for r in audit['splits']} != {'train': 9702, 'validation': 4884}:
        raise ValueError('Full original TRAIN and validation prior features must be audited first.')
    value = provenance()
    path = RUN / 'registered_protocol.json'
    if path.exists():
        if core.read_json(path)['provenance'] != value:
            raise ValueError('Preserve existing source/data/normalization registration.')
    else:
        if (RUN / 'best.pt').exists() or (RUN / 'last.pt').exists():
            raise ValueError('Preserve unregistered learned weights.')
        core.write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': value,
            'validation_classifier_predictions_before_registration': False,
            'objective': {'precision_above': .88, 'recall_above': .88, 'accuracy_at_least': .9},
            'completion_requires': design['completion_requires']})
        for name in SOURCES:
            (RUN / ('registered_' + name)).write_bytes((ROOT / name).read_bytes())
    print('Registered fresh full-prior GRU with the original audio-only lexical recipe.', flush=True)


def guard(value):
    if provenance() != value:
        raise ValueError('Registered source/data/normalization changed.')


def load_features(directory, rows, split, value):
    if split not in {'train', 'validation'} or any(r['split'] != split for r in rows):
        raise ValueError('Full-prior development head cannot consume test features.')
    acoustic = improve.feature_cache(directory, rows, split)
    words = lexical.lexical_window_cache(directory, rows, split, value['lexical'])
    path, identity = encoded.window_path(rows, split, value['phonetic_prior'])
    prior = lexical.array_cache(path, identity)
    if acoustic.shape != (len(rows), 24, 1416) or prior.shape != (len(rows), 24, 392):
        raise ValueError('Incomplete original acoustic/prior region geometry.')
    return lexical.CombinedFeatures(lexical.CombinedFeatures(acoustic, words), prior)


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
        raise ValueError('Every original train and validation region is required.')
    training = load_features(directory, train_rows, 'train', value)
    validation = load_features(directory, val_rows, 'validation', value)
    dense, vocabulary = improve.frame_labels(directory, train_rows)
    targets = np.where(dense == -100, -100, (dense == vocabulary.index('ax')).astype(np.int64))
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
        metric, pool = max(choices, key=lambda choice: lexical.selection_key(choice[0]))
        key = lexical.selection_key(metric)
        history.append({'epoch': epoch, 'train_loss': loss, 'validation_precision': metric['precision'],
                        'validation_recall': metric['recall'], 'seconds': elapsed + time.perf_counter() - started})
        core.write_csv(RUN / 'history.csv', history)
        if best_key is None or key > tuple(best_key):
            best_key, best_epoch = key, epoch
            core.save_torch(RUN / 'best.pt', {'backend': 'wav2vec2_lexical_full_prior_head',
                'target_phones': ['ax'], 'target_symbol': '\u0259', 'head_config': CONFIG,
                'head_state_dict': model.state_dict(), 'phone_vocabulary': ['non_ax', 'ax'],
                'epoch': epoch, 'threshold': metric['threshold'], 'pool_frames': pool, 'cpu_threads': 2,
                'threshold_selection_split': 'validation', 'validation_metrics': metric,
                'dataset_sha256': value['dataset_sha256'], 'backbone_sha256': value['backbone_sha256'],
                'lexical': value['lexical'], 'phonetic_prior': value['phonetic_prior'],
                'settings': value, 'provenance': value})
            core.write_json(RUN / 'best_summary.json', {**value, 'best_epoch': epoch, 'validation_metrics': metric})
        core.save_torch(RUN / 'last.pt', training_snapshot(model, optimizer, rng, history, best_key,
                         best_epoch, elapsed + time.perf_counter() - started, value))
        core.write_json(RUN / 'status.json', {'status': 'training', 'epoch_completed': epoch,
                                            'best_epoch': best_epoch, 'test_predictions_computed': False})
        print(f'Full phonetic prior GRU epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
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

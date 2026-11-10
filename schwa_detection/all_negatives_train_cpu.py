"""Fresh CPU Schwa GRU using every eligible original TRAIN negative region."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

import improve
import lexical_train_cpu as lexical
import schwa as core
from whisper_train_cpu import restore_training, step_epoch, training_snapshot

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'v5'
EXPANDED = ROOT / 'training_all_negatives'
RUN = ROOT / 'runs/pr88_v5_all_negatives_binary_gru'
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
DESIGN = ROOT / 'reports/precision_recall_88_all_negatives_design.json'
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
CONFIG = {'classes': 2, 'hidden': 128, 'dropout': .3, 'temporal': 'gru',
          'input_dim': 1420, 'group_sizes': [1024, 392, 4]}
SETTINGS = {'threads': 2, 'seed': 42, 'epochs': 40, 'patience': 10,
            'lr': .0005, 'batch_size': 64, 'window_loss': .5}
SOURCES = ['all_negatives_train_cpu.py', 'train_negative_features.py', 'train_negative_pool.py',
           'improve.py', 'lexical_train_cpu.py', 'lexical_context.py', 'frozen_phoneme.py',
           'schwa.py', 'whisper_train_cpu.py']


def provenance():
    if core.digest(DATA / 'data/manifests/dataset.csv') != DATASET_SHA:
        raise ValueError('Original evaluation dataset changed.')
    preparation = core.read_json(ROOT / 'reports/all_train_negatives_preparation_audit.json')
    features = core.read_json(ROOT / 'reports/all_train_negatives_feature_audit.json')
    if not preparation['passed'] or not features['passed'] or features['original_train_feature_parity_regions'] != 9702:
        raise ValueError('Expanded TRAIN data and original-region parity must pass.')
    marker = core.read_json(EXPANDED / 'data/train_features.json')
    identity = core.read_json(EXPANDED / 'registered_features.json')
    if marker['identity'] != identity or marker['sha256'] != features['feature_sha256']:
        raise ValueError('Expanded feature metadata changed.')
    if core.digest(EXPANDED / 'data/manifests/dataset.csv') != preparation['expanded_manifest_sha256']:
        raise ValueError('Expanded TRAIN manifest changed.')
    for name, sha in identity['source_code_sha256'].items():
        if core.digest(ROOT / name) != sha:
            raise ValueError('Registered feature source changed.')
    return {'dataset': 'v5', 'dataset_sha256': DATASET_SHA,
        'expanded_train_manifest_sha256': preparation['expanded_manifest_sha256'],
        'expanded_train_features_sha256': features['feature_sha256'],
        'backbone_sha256': core.digest(BACKBONE), 'lexical': lexical.metadata(BACKBONE),
        'head_config': CONFIG, 'settings': SETTINGS, 'candidate_design_sha256': core.digest(DESIGN),
        'source_code_sha256': {name: core.digest(ROOT / name) for name in SOURCES},
        'initialization': 'Fresh binary GRU; original public encoders only',
        'train_regions': 32078, 'original_train_regions_preserved': 9702,
        'validation_regions': 4884, 'test_regions': 5157,
        'target': 'Strict AH0/AX; AH1/AH2 and ER/AXR negative',
        'feature_context': 'recording', 'head_context': 'Original 0.5-second focal region',
        'frame_supervision': 'Original manual TRAIN phones, 5ms boundary margin',
        'selection_objective': 'min_precision_recall', 'threshold_selection_split': 'validation',
        'training_mkldnn': False, 'test_used_for_selection': False}


def guard(value):
    if provenance() != value:
        raise ValueError('Registered training recipe changed.')


def expanded_features(value):
    rows = core.read_csv(EXPANDED / 'data/manifests/dataset.csv')
    if len(rows) != 32078 or any(r['split'] != 'train' for r in rows):
        raise ValueError('Only every expanded TRAIN region is permitted.')
    path = EXPANDED / 'data/train_features.npy'
    if core.digest(path) != value['expanded_train_features_sha256']:
        raise ValueError('Expanded TRAIN feature checksum changed.')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.shape != (len(rows), 24, 1420) or values.dtype != np.float16:
        raise ValueError('Expanded TRAIN feature geometry changed.')
    return rows, values


def register():
    design = core.read_json(DESIGN)
    if design['head_config'] != CONFIG or design['settings'] != SETTINGS:
        raise ValueError('Candidate implementation differs from design.')
    value = provenance()
    expanded_features(value)
    path = RUN / 'registered_protocol.json'
    if path.exists():
        if core.read_json(path)['provenance'] != value:
            raise ValueError('Preserve registered candidate.')
    else:
        if (RUN / 'best.pt').exists() or (RUN / 'last.pt').exists():
            raise ValueError('Preserve existing learned weights.')
        core.write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': value,
            'validation_classifier_predictions_before_registration': False,
            'objective': {'precision_above': .88, 'recall_above': .88, 'accuracy_at_least': .9},
            'completion_requires': design['completion_requires']})
        for name in SOURCES:
            (RUN / ('registered_' + name)).write_bytes((ROOT / name).read_bytes())
    print('Registered fresh GRU on all 32078 original TRAIN regions.', flush=True)


def train(resume):
    value = core.read_json(RUN / 'registered_protocol.json')['provenance']
    guard(value)
    if not resume and ((RUN / 'best.pt').exists() or (RUN / 'last.pt').exists()):
        raise ValueError('Preserve checkpoints; use --resume.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    improve.activate_dataset('v5')
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = BACKBONE, 'recording', 'joint'
    train_rows, training = expanded_features(value)
    val_rows = [r for r in core.read_csv(DATA / 'data/manifests/dataset.csv') if r['split'] == 'validation']
    if len(val_rows) != 4884:
        raise ValueError('Every original validation region is required.')
    validation = lexical.CombinedFeatures(improve.feature_cache(DATA, val_rows, 'validation'),
        lexical.lexical_window_cache(DATA, val_rows, 'validation', value['lexical']))
    dense, vocabulary = improve.frame_labels(DATA, train_rows)
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
            core.save_torch(RUN / 'best.pt', {'backend': 'wav2vec2_lexical_phone_head',
                'target_phones': ['ax'], 'target_symbol': '\u0259', 'head_config': CONFIG,
                'head_state_dict': model.state_dict(), 'phone_vocabulary': ['non_ax', 'ax'],
                'epoch': epoch, 'threshold': metric['threshold'], 'pool_frames': pool, 'cpu_threads': 2,
                'threshold_selection_split': 'validation', 'validation_metrics': metric,
                'dataset_sha256': DATASET_SHA, 'backbone_sha256': value['backbone_sha256'],
                'lexical': value['lexical'], 'settings': {**SETTINGS, 'feature_context': 'recording',
                    'acoustic_representation': 'joint'}, 'provenance': value})
            core.write_json(RUN / 'best_summary.json', {**value, 'best_epoch': epoch, 'validation_metrics': metric})
        core.save_torch(RUN / 'last.pt', training_snapshot(model, optimizer, rng, history, best_key,
                         best_epoch, elapsed + time.perf_counter() - started, value))
        core.write_json(RUN / 'status.json', {'status': 'training', 'epoch_completed': epoch,
                                            'best_epoch': best_epoch, 'test_predictions_computed': False})
        print(f'All negatives GRU epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
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

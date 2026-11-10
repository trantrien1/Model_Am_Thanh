"""Fresh CPU Schwa GRU with lower-weight native auxiliary phone supervision."""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import improve
import lexical_train_cpu as lexical
import native_training_data as native
import schwa as core
from whisper_train_cpu import restore_training, training_snapshot

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'v5'
RUN = ROOT / 'runs/pr88_v5_native_aux_binary_gru'
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
DESIGN = ROOT / 'reports/precision_recall_88_native_aux_design.json'
CONFIG = {'classes': 2, 'hidden': 128, 'dropout': .3, 'temporal': 'gru',
          'input_dim': 1420, 'group_sizes': [1024, 392, 4]}
SETTINGS = {'threads': 2, 'seed': 42, 'epochs': 40, 'patience': 10, 'lr': .0005,
            'batch_size': 64, 'window_loss': .5, 'native_weight': .25}
SOURCES = ['native_aux_train_cpu.py', 'native_training_data.py', 'train_negative_pool.py',
           'improve.py', 'lexical_train_cpu.py', 'lexical_context.py', 'frozen_phoneme.py',
           'schwa.py', 'whisper_train_cpu.py']


def weighted_loss(logits, targets, truth, domain_weights, class_weights, window_weight=.5):
    frame = F.cross_entropy(logits.transpose(1, 2), targets, weight=class_weights,
                            ignore_index=-100, reduction='none')
    valid = targets >= 0
    denominator = (class_weights[targets.clamp_min(0)] * valid * domain_weights[:, None]).sum()
    dense_loss = (frame * domain_weights[:, None]).sum() / denominator.clamp_min(1e-12)
    odds = logits[..., 1] - logits[..., 0]
    window = F.binary_cross_entropy_with_logits(odds.amax(dim=1), truth, reduction='none')
    return dense_loss + window_weight * (window * domain_weights).sum() / domain_weights.sum()


def step_epoch(model, optimizer, features, targets, truth, domain_weights, class_weights, rng):
    model.train()
    losses = []
    order = rng.permutation(len(features))
    for start in range(0, len(order), SETTINGS['batch_size']):
        indices = order[start:start + SETTINGS['batch_size']]
        values = torch.from_numpy(np.array(features[indices], dtype=np.float32))
        logits = model(values)
        loss = weighted_loss(logits, torch.from_numpy(targets[indices]), truth[indices],
                             domain_weights[indices], class_weights, SETTINGS['window_loss'])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite native auxiliary training loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def provenance():
    native.source_selection()
    audit = core.read_json(ROOT / 'reports/native_training_feature_audit.json')
    identity = core.read_json(native.DATA / 'registered_features.json')
    for name, sha in identity['code_sha256'].items():
        if core.digest(ROOT / name) != sha:
            raise ValueError('Registered native feature source changed.')
    if not audit['passed'] or audit['regions'] != 5680:
        raise ValueError('Complete auxiliary native features required.')
    return {'dataset': 'v5', 'dataset_sha256': native.V5_SHA,
        'native_manifest_sha256': core.digest(native.DATA / 'data/manifests/dataset.csv'),
        'native_feature_sha256': audit['feature_sha256'], 'native_feature_path': audit['feature_path'],
        'backbone_sha256': core.digest(BACKBONE), 'lexical': lexical.metadata(BACKBONE),
        'head_config': CONFIG, 'settings': SETTINGS, 'design_sha256': core.digest(DESIGN),
        'source_code_sha256': {name: core.digest(ROOT / name) for name in SOURCES},
        'train_regions': 15382, 'manual_l2_train_regions': 9702, 'native_auxiliary_regions': 5680,
        'validation_regions': 4884, 'test_regions': 5157,
        'native_labels': 'Automatic aligned native ax; auxiliary loss weight 0.25',
        'native_lexical_inputs': 'Zero for all four lexical dimensions; no transcript used as features',
        'initialization': 'Fresh binary GRU; original public pretrained frozen encoders only',
        'target': 'Strict AH0/AX; AH1/AH2, ER/AXR negative',
        'feature_context': 'recording', 'head_context': 'Original 0.5-second focal region',
        'selection_objective': 'min_precision_recall', 'threshold_selection_split': 'validation',
        'training_mkldnn': False, 'test_used_for_selection': False}


def guard(value):
    if provenance() != value:
        raise ValueError('Registered native auxiliary recipe changed.')


def load_training(value):
    improve.activate_dataset('v5')
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = BACKBONE, 'recording', 'joint'
    manual_rows = [r for r in core.read_csv(DATA / 'data/manifests/dataset.csv') if r['split'] == 'train']
    native_rows = core.read_csv(native.DATA / 'data/manifests/dataset.csv')
    if len(manual_rows) != 9702 or len(native_rows) != 5680 or any(r['split'] != 'train' for r in native_rows):
        raise ValueError('All original manual and auxiliary TRAIN rows required.')
    manual_features = lexical.CombinedFeatures(improve.feature_cache(DATA, manual_rows, 'train'),
        lexical.lexical_window_cache(DATA, manual_rows, 'train', value['lexical']))
    acoustic_native = improve.feature_cache(native.DATA, native_rows, 'train')
    if core.digest(ROOT / value['native_feature_path']) != value['native_feature_sha256']:
        raise ValueError('Native feature checksum differs.')
    native_features = lexical.CombinedFeatures(acoustic_native, np.zeros((len(native_rows), 24, 4), np.float16))
    features = improve.JoinedFeatures([(manual_features, np.arange(len(manual_rows))),
                                       (native_features, np.arange(len(native_rows)))])
    targets = []
    for directory, rows in [(DATA, manual_rows), (native.DATA, native_rows)]:
        dense, vocabulary = improve.frame_labels(directory, rows)
        targets.append(np.where(dense == -100, -100, (dense == vocabulary.index('ax')).astype(np.int64)))
    targets = np.concatenate(targets)
    domains = torch.tensor([1.] * len(manual_rows) + [.25] * len(native_rows))
    truth = torch.tensor([int(r['label']) for r in manual_rows + native_rows], dtype=torch.float32)
    return manual_rows + native_rows, features, targets, truth, domains


def register():
    design = core.read_json(DESIGN)
    if design['head_config'] != CONFIG or design['settings'] != SETTINGS:
        raise ValueError('Candidate differs from fixed design.')
    value = provenance()
    path = RUN / 'registered_protocol.json'
    if path.exists():
        if core.read_json(path)['provenance'] != value:
            raise ValueError('Preserve registered recipe.')
    else:
        if (RUN / 'best.pt').exists() or (RUN / 'last.pt').exists():
            raise ValueError('Preserve learned weights.')
        core.write_json(path, {'created_utc': datetime.now(timezone.utc).isoformat(), 'provenance': value,
            'validation_predictions_before_registration': False, 'completion_requires': design['completion_requires']})
        for name in SOURCES:
            (RUN / ('registered_' + name)).write_bytes((ROOT / name).read_bytes())
    print('Registered fresh native auxiliary GRU.', flush=True)


def train(resume):
    value = core.read_json(RUN / 'registered_protocol.json')['provenance']
    guard(value)
    if not resume and ((RUN / 'best.pt').exists() or (RUN / 'last.pt').exists()):
        raise ValueError('Preserve existing training; use --resume.')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    train_rows, training, targets, truth, domains = load_training(value)
    valid = targets >= 0
    counts = np.bincount(targets[valid], weights=np.broadcast_to(domains.numpy()[:, None], targets.shape)[valid], minlength=2)
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    class_weights = torch.from_numpy(np.clip(weights / np.median(weights), .25, 3).astype(np.float32))
    val_rows = [r for r in core.read_csv(DATA / 'data/manifests/dataset.csv') if r['split'] == 'validation']
    if len(val_rows) != 4884:
        raise ValueError('All original validation regions required.')
    validation = lexical.CombinedFeatures(improve.feature_cache(DATA, val_rows, 'validation'),
        lexical.lexical_window_cache(DATA, val_rows, 'validation', value['lexical']))
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
        loss = step_epoch(model, optimizer, training, targets, truth, domains, class_weights, rng)
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
                'dataset_sha256': native.V5_SHA, 'backbone_sha256': value['backbone_sha256'],
                'lexical': value['lexical'], 'settings': {**SETTINGS, 'feature_context': 'recording',
                    'acoustic_representation': 'joint'}, 'provenance': value})
            core.write_json(RUN / 'best_summary.json', {**value, 'best_epoch': epoch, 'validation_metrics': metric})
        core.save_torch(RUN / 'last.pt', training_snapshot(model, optimizer, rng, history, best_key,
                         best_epoch, elapsed + time.perf_counter() - started, value))
        core.write_json(RUN / 'status.json', {'status': 'training', 'epoch_completed': epoch, 'best_epoch': best_epoch})
        print(f'Native auxiliary GRU epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
        if epoch - best_epoch >= SETTINGS['patience']:
            break
    core.write_json(RUN / 'training_summary.json', {**value, 'best_epoch': best_epoch,
        'epochs_completed': len(history), 'seconds': elapsed + time.perf_counter() - started,
        'validation_metrics': core.read_json(RUN / 'best_summary.json')['validation_metrics']})
    core.write_json(RUN / 'status.json', {'status': 'completed', 'epochs_completed': len(history)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['register', 'train'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    register() if args.command == 'register' else train(args.resume)

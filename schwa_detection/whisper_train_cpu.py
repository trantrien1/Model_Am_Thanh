"""Train a fresh Schwa GRU using audio-only Whisper lexical features on CPU."""
import argparse
import ctypes
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

import improve
import schwa as core
import whisper_lexical as lexical
from lexical_train_cpu import CombinedFeatures, array_cache, selection_key

ROOT = Path(__file__).resolve().parent
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
DATASET_SHA = '32d8dbd761693c3c06f94b57b99ce1dad44173aa227da54900c6b84d37788731'
SOURCES = ['whisper_train_cpu.py', 'whisper_lexical.py', 'whisper_cpu.py',
           'lexical_context.py', 'lexical_train_cpu.py', 'improve.py',
           'frozen_phoneme.py', 'schwa.py']


def source_hashes():
    return {name: core.digest(ROOT / name) for name in SOURCES}


def guard(provenance):
    if source_hashes() != provenance['source_code_sha256']:
        raise ValueError('Registered training or feature source changed.')
    if core.digest(ROOT / 'v5/data/manifests/dataset.csv') != provenance['dataset_sha256']:
        raise ValueError('Frozen dataset changed.')


def step_epoch(model, optimizer, features, targets, truth, criterion, rng,
               batch_size=64, window_loss=.5):
    model.train()
    order = rng.permutation(len(features))
    losses = []
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        values = torch.from_numpy(np.array(features[indices], dtype=np.float32))
        logits = model(values)
        dense = torch.from_numpy(targets[indices])
        loss = criterion(logits.transpose(1, 2), dense)
        loss = loss + window_loss * improve.window_presence_loss(logits, 1, truth[indices])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite Whisper Schwa training loss.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def process_live(pid):
    if not pid:
        return True
    if os.name == 'nt':
        kernel = ctypes.windll.kernel32
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            result = ctypes.c_ulong()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(result))) and result.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def restore_training(snapshot, model, optimizer, rng, provenance):
    if snapshot['provenance'] != provenance:
        raise ValueError('Resume provenance differs.')
    model.load_state_dict(snapshot['head_state_dict'])
    optimizer.load_state_dict(snapshot['optimizer_state_dict'])
    rng.bit_generator.state = snapshot['numpy_random_state']
    torch.set_rng_state(snapshot['torch_random_state'])
    return tuple(snapshot[key] for key in ['history', 'best_key', 'best_epoch', 'seconds'])


def training_snapshot(model, optimizer, rng, history, best_key, best_epoch, seconds, provenance):
    return {'head_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'numpy_random_state': rng.bit_generator.state, 'torch_random_state': torch.get_rng_state(),
            'history': history, 'best_key': list(best_key), 'best_epoch': best_epoch,
            'seconds': seconds, 'provenance': provenance}


def train(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    directory = improve.activate_dataset('v5')
    dataset_sha = core.digest(directory / 'data/manifests/dataset.csv')
    if dataset_sha != DATASET_SHA or core.digest(BACKBONE) != lexical.BACKBONE_SHA:
        raise ValueError('Frozen dataset or original public backbone differs.')
    rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    train_rows = [r for r in rows if r['split'] == 'train']
    validation_rows = [r for r in rows if r['split'] == 'validation']
    if len(train_rows) != 9702 or len(validation_rows) != 4884:
        raise ValueError('Use every original train and validation region.')
    recipe = lexical.metadata()
    settings = {key: getattr(args, key) for key in
                ['threads', 'seed', 'epochs', 'patience', 'lr', 'batch_size', 'window_loss', 'hidden', 'dropout']}
    config = {'classes': 2, 'hidden': args.hidden, 'dropout': args.dropout,
              'temporal': 'gru', 'input_dim': 1424, 'group_sizes': [1024, 392, 8]}
    provenance = {'dataset': 'v5', 'dataset_sha256': dataset_sha, 'lexical': recipe,
        'source_code_sha256': source_hashes(), 'settings': settings, 'head_config': config,
        'initialization': 'Original frozen public XLSR and Whisper Base English; fresh random binary GRU',
        'frame_supervision': 'Manual labels are TRAIN targets only; AH0/AX positive, AH1/AH2/ER/AXR negative',
        'frame_boundary_margin_seconds': .005, 'feature_context': 'recording', 'head_context': 'original_0.5_second_region',
        'threshold_selection_split': 'validation', 'selection_objective': 'min_precision_recall',
        'test_used_for_selection': False, 'test_features_computed': False, 'training_mkldnn': False}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    registered = args.run_dir / 'registered_protocol.json'
    if registered.exists():
        saved = core.read_json(registered)
        if any(saved[key] != value for key, value in provenance.items()):
            raise ValueError('Existing registration uses different training or feature settings.')
    else:
        if args.resume or (args.run_dir / 'best.pt').exists():
            raise ValueError('Preserve existing learned model without matching registration.')
        core.write_json(registered, {**provenance, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'new_validation_classifier_predictions_computed_before_registration': False,
            'objective': {'precision_minimum': .88, 'recall_minimum': .88, 'accuracy_minimum': .90},
            'completion_requires': ['actual_runtime_validation', 'freeze_before_full_test',
                'full_original_test_regions', 'actual_final_checkpoint_offline_four_file_bundle_parity']})
        for name in SOURCES:
            (args.run_dir / ('registered_' + name)).write_bytes((ROOT / name).read_bytes())
    required = [lexical.cache_path(directory, subset, split, recipe) for subset, split in
                [(train_rows, 'train'), (validation_rows, 'validation')]]
    while not all(path.exists() and path.with_suffix('.json').exists() for path, _ in required):
        guard(provenance)
        if not args.wait_features:
            raise ValueError('Full train and validation Whisper lexical caches must finish first.')
        if not process_live(args.feature_process_id):
            raise ValueError('Feature process is terminal and full caches are missing; preserve registration.')
        core.write_json(args.run_dir / 'status.json', {'status': 'waiting_for_full_feature_caches',
            'feature_process_id': args.feature_process_id, 'training_process_id': os.getpid(),
            'validation_classifier_predictions_computed': False, 'test_predictions_computed': False})
        time.sleep(30)
    guard(provenance)
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = BACKBONE, 'recording', 'joint'
    acoustics = [improve.feature_cache(directory, subset, split) for subset, split in
                 [(train_rows, 'train'), (validation_rows, 'validation')]]
    words = [array_cache(path, identity) for path, identity in required]
    if any(values.shape != (len(subset), 24, 8) for values, subset in zip(words, [train_rows, validation_rows])):
        raise ValueError('Whisper lexical cache geometry differs.')
    training, validation = [CombinedFeatures(a, w) for a, w in zip(acoustics, words)]
    frame_targets, vocabulary = improve.frame_labels(directory, train_rows)
    ax = vocabulary.index('ax')
    targets = np.where(frame_targets == -100, -100, (frame_targets == ax).astype(np.int64))
    counts = np.bincount(targets[targets >= 0], minlength=2)
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    weights = np.clip(weights / np.median(weights), .25, 3).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), ignore_index=-100)
    truth = torch.tensor([int(r['label']) for r in train_rows], dtype=torch.float32)
    validation_truth = [int(r['label']) for r in validation_rows]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = improve.PhoneHead(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    history, best_key, best_epoch, elapsed = [], None, 0, 0.
    if args.resume:
        snapshot = torch.load(args.run_dir / 'last.pt', weights_only=True)
        history, best_key, best_epoch, elapsed = restore_training(snapshot, model, optimizer, rng, provenance)
    elif (args.run_dir / 'best.pt').exists() or (args.run_dir / 'last.pt').exists():
        raise ValueError('Use --resume to preserve existing trained state.')
    started = time.perf_counter()
    core.write_json(args.run_dir / 'training_protocol.json', provenance)
    for epoch in range(len(history) + 1, args.epochs + 1):
        guard(provenance)
        loss = step_epoch(model, optimizer, training, targets, truth, criterion, rng,
                          args.batch_size, args.window_loss)
        choices = [(improve.validation_selection(validation_truth,
                    improve.class_scores(model, validation, 1, pool), 'min_pr'), pool) for pool in [1, 2, 3]]
        metric, pool = max(choices, key=lambda choice: selection_key(choice[0]))
        key = selection_key(metric)
        history.append({'epoch': epoch, 'train_loss': loss, 'validation_precision': metric['precision'],
            'validation_recall': metric['recall'], 'validation_f1': metric['f1'],
            'seconds': elapsed + time.perf_counter()-started})
        core.write_csv(args.run_dir / 'history.csv', history)
        if best_key is None or key > tuple(best_key):
            best_key, best_epoch = key, epoch
            checkpoint = {'backend': 'wav2vec2_whisper_lexical_head', 'target_phones': ['ax'], 'target_symbol': '\u0259',
                'head_state_dict': model.state_dict(), 'head_config': config, 'phone_vocabulary': ['non_ax', 'ax'],
                'epoch': epoch, 'threshold': metric['threshold'], 'pool_frames': pool, 'cpu_threads': args.threads,
                'validation_metrics': metric, 'threshold_selection_split': 'validation', 'settings': provenance,
                'dataset_sha256': dataset_sha, 'lexical': recipe, 'backbone_sha256': lexical.BACKBONE_SHA,
                'provenance': provenance}
            core.save_torch(args.run_dir / 'best.pt', checkpoint)
            core.write_json(args.run_dir / 'best_summary.json', {**provenance, 'best_epoch': epoch,
                'validation_metrics': metric})
        core.save_torch(args.run_dir / 'last.pt', training_snapshot(model, optimizer, rng,
            history, best_key, best_epoch, elapsed+time.perf_counter()-started, provenance))
        core.write_json(args.run_dir / 'status.json', {'status': 'training', 'epoch_completed': epoch,
            'training_process_id': os.getpid(), 'best_epoch': best_epoch, 'test_predictions_computed': False})
        print(f'Whisper Schwa epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    core.write_json(args.run_dir / 'training_summary.json', {**provenance, 'epochs_completed': len(history),
        'best_epoch': best_epoch, 'validation_metrics': core.read_json(args.run_dir / 'best_summary.json')['validation_metrics'],
        'seconds': elapsed+time.perf_counter()-started})
    core.write_json(args.run_dir / 'status.json', {'status': 'completed', 'epoch_completed': len(history),
        'best_epoch': best_epoch, 'test_predictions_computed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--window-loss', type=float, default=.5)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=.3)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--wait-features', action='store_true')
    parser.add_argument('--feature-process-id', type=int, default=0)
    train(parser.parse_args())

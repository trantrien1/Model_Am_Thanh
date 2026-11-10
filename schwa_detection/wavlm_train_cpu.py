"""CPU Schwa heads on frozen public WavLM embeddings; test remains untouched."""
import argparse
import gc
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch import nn

import schwa as core
import improve
from frozen_phoneme import aligned_window_features
from frozen_wavlm import OUTPUT, WAVLM_RECIPE, rebuild_wavlm, recording_embeddings
from lexical_train_cpu import array_cache, save_array

ROOT = Path(__file__).resolve().parent
NORMALIZATION = {'kind': 'whole_recording_channel_mean_center',
                 'statistics': 'unlabeled input recording only', 'storage': 'float16'}


def centered_embeddings(values):
    values = values.float()
    return (values - values.mean(0, keepdim=True)).half()


def cache_path(directory, record, encoder_sha):
    identity = {'encoder_sha256': encoder_sha, 'source_audio_sha256': record['audio_sha256'],
                'recipe': WAVLM_RECIPE, 'normalization': NORMALIZATION}
    return directory / 'data/wavlm_recording_features' / (core.object_digest(identity) + '.npy'), identity


class FeatureWindows:
    def __init__(self, directory, rows, encoder_sha):
        if any(r['split'] not in ('train', 'validation') for r in rows):
            raise ValueError('Development feature loader must never consume test rows.')
        self.directory, self.rows, self.encoder_sha = directory, rows, encoder_sha
        self.records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
        self.loaded, self.verified = OrderedDict(), set()

    def __len__(self):
        return len(self.rows)

    def record_features(self, identity):
        if identity not in self.loaded:
            path, recipe = cache_path(self.directory, self.records[identity], self.encoder_sha)
            if identity not in self.verified:
                values = array_cache(path, recipe)
                self.verified.add(identity)
            else:
                values = np.load(path, allow_pickle=False, mmap_mode='r')
            if values.ndim != 2 or values.shape[1] != 768:
                raise ValueError('WavLM recording has an invalid embedding dimension.')
            self.loaded[identity] = torch.from_numpy(np.array(values, dtype=np.float32))
            if len(self.loaded) > 128:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(identity)
        return self.loaded[identity]

    def __getitem__(self, item):
        if isinstance(item, (int, np.integer)):
            row = self.rows[int(item)]
            sequence = self.record_features(row['recording_id'])
            return aligned_window_features(sequence, [int(row['window_start_sample'])])[0].numpy()
        indices = range(*item.indices(len(self))) if isinstance(item, slice) else item
        return np.stack([self[int(index)] for index in indices])


def extract(args):
    if any(split not in ('train', 'validation') for split in args.splits):
        raise ValueError('Frozen development extraction must not consume test.')
    directory = improve.activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    selected = {r['recording_id'] for r in rows if r['split'] in args.splits}
    records = [r for r in core.read_csv(directory / 'data/manifests/recordings.csv') if r['recording_id'] in selected]
    encoder_sha = core.digest(args.backbone)
    state = torch.load(args.backbone, weights_only=True, mmap=True)
    if state['recipe'] != WAVLM_RECIPE or state.get('fine_tuning'):
        raise ValueError('Extract only the registered original public WavLM representation.')
    model = rebuild_wavlm(state)
    model.requires_grad_(False)
    started = time.perf_counter()
    with torch.inference_mode():
        for number, record in enumerate(records, 1):
            path, identity = cache_path(directory, record, encoder_sha)
            if path.exists():
                array_cache(path, identity)
            else:
                audio_path = directory / record['audio_path']
                if core.digest(audio_path) != record['audio_sha256']:
                    raise ValueError('WavLM source audio changed.')
                audio, _ = core.load_audio(audio_path)
                values = centered_embeddings(recording_embeddings(model, torch.from_numpy(audio)))
                save_array(path, values.numpy(), identity)
            if number % 16 == 0 or number == len(records):
                print(f'WavLM development features {number}/{len(records)}, {time.perf_counter()-started:.1f}s', flush=True)


def train(args):
    directory = improve.activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    train_rows = [r for r in rows if r['split'] == 'train']
    validation_rows = [r for r in rows if r['split'] == 'validation']
    if (args.run_dir / 'best.pt').exists() and not args.resume:
        raise ValueError('Preserving existing learned weights; use resume or a fresh run.')
    if args.resume and not (args.run_dir / 'last.pt').exists():
        raise ValueError('Resume requires a real completed-epoch checkpoint.')
    encoder_sha = core.digest(args.backbone)
    targets, vocabulary = improve.frame_labels(directory, train_rows)
    ax = vocabulary.index('ax')
    counts = np.bincount(targets[targets >= 0], minlength=len(vocabulary))
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    weights = np.clip(weights / weights.mean(), .25, 3).astype(np.float32)
    config = {'classes': len(vocabulary), 'hidden': args.hidden, 'dropout': .3,
              'temporal': args.temporal, 'input_dim': 768}
    settings = {'dataset': 'v5', 'feature_context': 'recording', 'seed': args.seed,
                'head_config': config, 'cpu_threads': args.threads, 'epochs': args.epochs,
                'patience': args.patience, 'learning_rate': args.lr, 'batch_size': args.batch_size,
                'frame_supervision': True, 'window_loss_weight': .5,
                'frame_class_counts': counts.tolist(), 'selection_objective': 'min_pr',
                'wavlm_recipe': WAVLM_RECIPE, 'recording_normalization': NORMALIZATION}
    provenance = {'dataset_sha256': core.digest(directory / 'data/manifests/dataset.csv'),
                  'backbone_sha256': encoder_sha, 'settings': settings,
                  'test_used_for_selection': False, 'initialization': 'Original public frozen WavLM; fresh random phone head',
                  'source_code_sha256': {name: core.digest(ROOT / name) for name in
                                        ['wavlm_train_cpu.py', 'frozen_wavlm.py', 'improve.py', 'frozen_phoneme.py']}}
    protocol_path = args.run_dir / 'training_protocol.json'
    if protocol_path.exists() and core.read_json(protocol_path) != provenance:
        raise ValueError('Training settings/code changed; preserving existing run.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(protocol_path, provenance)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    model = improve.PhoneHead(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), ignore_index=-100)
    training = FeatureWindows(directory, train_rows, encoder_sha)
    validation = FeatureWindows(directory, validation_rows, encoder_sha)
    truths = [int(r['label']) for r in validation_rows]
    history, best_key, best_epoch, start_epoch, elapsed = [], None, 0, 1, 0.
    if args.resume:
        saved = torch.load(args.run_dir / 'last.pt', weights_only=True)
        if saved['provenance'] != provenance:
            raise ValueError('Resume settings/code differ from the completed checkpoint.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        rng.bit_generator.state = saved['numpy_random_state']
        torch.set_rng_state(saved['torch_random_state'])
        history, best_key, best_epoch = saved['history'], tuple(saved['best_key']), saved['best_epoch']
        start_epoch, elapsed = saved['epoch'] + 1, saved['seconds']
        del saved
    started = time.perf_counter()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(train_rows))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            values = torch.from_numpy(training[indices])
            labels = torch.from_numpy(targets[indices])
            logits = model(values)
            loss = (criterion(logits.reshape(-1, len(vocabulary)), labels.reshape(-1))
                    if (labels >= 0).any() else logits.sum() * 0)
            loss = loss + .5 * improve.window_presence_loss(logits, ax,
                      torch.tensor([int(train_rows[int(i)]['label']) for i in indices], dtype=torch.float32))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite WavLM phone-head training loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        alternatives = []
        for pool in (1, 2, 3):
            scores = improve.class_scores(model, validation, ax, pool, batch_size=args.batch_size)
            metric = improve.validation_selection(truths, scores, 'min_pr')
            alternatives.append((metric, pool))
        metric, pool = max(alternatives, key=lambda a:
            (min(a[0]['precision'], a[0]['recall']), a[0]['f1'], a[0]['accuracy'], -a[0]['fpr']))
        key = (min(metric['precision'], metric['recall']), metric['f1'], metric['accuracy'], -metric['fpr'])
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)),
                        'validation_precision': metric['precision'], 'validation_recall': metric['recall'],
                        'validation_f1': metric['f1'], 'seconds': elapsed + time.perf_counter()-started})
        core.write_csv(args.run_dir / 'history.csv', history)
        print(f'WavLM {args.temporal} epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            checkpoint = {'backend': 'wavlm_phone_head', 'target_phones': ['ax'], 'target_symbol': '\u0259',
                          'epoch': epoch, 'head_config': config, 'head_state_dict': model.state_dict(),
                          'phone_vocabulary': vocabulary, 'threshold': metric['threshold'], 'pool_frames': pool,
                          'cpu_threads': args.threads, 'settings': settings, 'validation_metrics': metric,
                          'threshold_selection_split': 'validation', 'provenance': provenance,
                          'dataset_sha256': provenance['dataset_sha256'], 'backbone_sha256': encoder_sha,
                          'source_metadata': core.read_json(directory / 'data/manifests/source_selection.json')}
            core.save_torch(args.run_dir / 'best.pt', checkpoint)
            core.write_json(args.run_dir / 'best_summary.json', {**provenance, 'best_epoch': epoch,
                                                               'validation_metrics': metric})
        core.save_torch(args.run_dir / 'last.pt', {'provenance': provenance, 'epoch': epoch,
                        'best_epoch': best_epoch, 'best_key': best_key, 'history': history,
                        'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'numpy_random_state': rng.bit_generator.state, 'torch_random_state': torch.get_rng_state(),
                        'seconds': elapsed + time.perf_counter()-started})
        if epoch - best_epoch >= args.patience:
            break
    best = core.read_json(args.run_dir / 'best_summary.json')
    core.write_json(args.run_dir / 'training_summary.json', {**provenance, 'best_epoch': best_epoch,
                    'epochs_completed': len(history), 'seconds': elapsed + time.perf_counter()-started,
                    'validation_metrics': best['validation_metrics']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', type=Path, default=OUTPUT)
    parser.add_argument('--threads', type=int, default=2)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('extract')
    p.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    p.set_defaults(function=extract)
    p = commands.add_parser('train')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--temporal', choices=['none', 'gru'], default='none')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.set_defaults(function=train)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.function(args)

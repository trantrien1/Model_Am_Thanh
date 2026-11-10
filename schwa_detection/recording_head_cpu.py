"""Train a fresh full-recording phone GRU on public XLSR features and manual TRAIN speech."""
import argparse
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

import improve
import schwa as core
from frozen_phoneme import RECORDING, aligned_window_features

ROOT = Path(__file__).resolve().parent
BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
DIMENSION = 1416
HELPERS = ['recording_head_cpu.py', 'improve.py', 'frozen_phoneme.py', 'schwa.py', 'l2_dataset.py']


def frame_targets(phones, frames, vocabulary, duration, boundary_margin=.005):
    indices = {phone: index for index, phone in enumerate(vocabulary)}
    centers = (RECORDING['center_sample'] + np.arange(frames) * RECORDING['hop_samples']) / 16000
    result = np.full(frames, -100, dtype=np.int64)
    for index, center in enumerate(centers):
        if center >= duration:
            result[index] = indices['pau']
            continue
        phone = next((p for p in phones if p['start'] <= center < p['end']), None)
        if phone is not None and min(center - phone['start'], phone['end'] - center) >= boundary_margin:
            result[index] = indices.get(phone['phone'], -100)
    return result


def region_logits(logits, starts):
    return aligned_window_features(logits, starts)


def region_scores(logits, starts, ax, pool):
    probabilities = region_logits(logits, starts).softmax(-1)[:, :, ax]
    if pool > 1:
        probabilities = nn.functional.avg_pool1d(probabilities.unsqueeze(1), pool, stride=1).squeeze(1)
    return probabilities.amax(-1)


class RecordingStore:
    def __init__(self, directory, rows, encoder_sha):
        if any(r['split'] not in {'train', 'validation'} for r in rows):
            raise ValueError('Full-recording development loader cannot consume test rows.')
        self.directory, self.encoder_sha = directory, encoder_sha
        self.rows, self.records, self.paths, self.phones, self.durations = {}, {}, {}, {}, {}
        self.loaded = OrderedDict()
        records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
        for row in rows:
            self.rows.setdefault(row['recording_id'], []).append(row)
        for identity in sorted(self.rows):
            rec = records[identity]
            for kind in ['audio', 'lab', 'manual']:
                if rec.get(kind + '_path') and core.digest(directory / rec[kind + '_path']) != rec[kind + '_sha256']:
                    raise ValueError('Source changed: ' + identity)
            recipe = {'encoder': encoder_sha, 'source_audio': rec['audio_sha256'], 'recipe': RECORDING, 'representation': 'joint'}
            path = directory / 'data/recording_transfer_features' / (core.object_digest(recipe) + '.npy')
            metadata = core.read_json(path.with_suffix('.json'))
            if (metadata['backbone_sha256'] != encoder_sha or metadata['source_audio_sha256'] != rec['audio_sha256']
                    or metadata['recipe'] != RECORDING or metadata['representation'] != 'joint'
                    or metadata['sha256'] != core.digest(path)):
                raise ValueError('Recording cache changed: ' + identity)
            values = np.load(path, allow_pickle=False, mmap_mode='r')
            if values.ndim != 2 or values.shape[1] != DIMENSION or list(values.shape) != metadata['shape'] or not np.isfinite(values).all():
                raise ValueError('Invalid complete recording features.')
            duration = sf.info(directory / rec['audio_path']).duration
            self.phones[identity] = core.parse_phones((directory / rec['lab_path']).read_text(), duration)[0]
            self.durations[identity] = duration
            self.records[identity], self.paths[identity] = rec, path

    def sequence(self, identity):
        if identity not in self.loaded:
            self.loaded[identity] = torch.from_numpy(np.array(np.load(self.paths[identity], allow_pickle=False, mmap_mode='r'), dtype=np.float32))
            if len(self.loaded) > 32:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(identity)
        return self.loaded[identity]


def settings(args):
    return {'dataset': 'v5', 'feature_context': 'recording', 'head_context': 'whole_recording',
        'acoustic_representation': 'joint', 'cpu_threads': args.threads, 'seed': args.seed,
        'head_config': {'classes': args.classes, 'hidden': args.hidden, 'dropout': .3, 'temporal': 'gru',
                        'input_dim': DIMENSION, 'group_sizes': [1024, 392]},
        'epochs': args.epochs, 'patience': args.patience, 'learning_rate': args.lr,
        'gradient_accumulation': args.accumulate, 'frame_supervision': True,
        'window_loss_weight': args.window_loss, 'selection_objective': 'min_pr',
        'frame_boundary_margin_seconds': .005, 'region_logit_interpolation': 'linear_then_float16_then_float32',
        'training_mkldnn': False, 'recording_recipe': RECORDING}


def train(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False
    directory, supplement = ROOT / 'v5', args.supplement
    if (args.run_dir / 'best.pt').exists() and not args.resume:
        raise ValueError('Preserving existing learned weights; use a fresh run or exact resume.')
    if args.resume and not (args.run_dir / 'last.pt').exists():
        raise ValueError('Resume needs a completed epoch checkpoint.')
    base_rows = core.read_csv(directory / 'data/manifests/dataset.csv')
    extra_rows = core.read_csv(supplement / 'data/manifests/dataset.csv')
    audit = core.read_json(supplement / 'reports/v5_preservation_audit.json')
    if not audit['passed'] or any(audit['cross_dataset_held_out_collisions'].values()) or any(r['split'] != 'train' for r in extra_rows):
        raise ValueError('Audited training-only supplement required.')
    dataset_sha = core.digest(directory / 'data/manifests/dataset.csv')
    if dataset_sha != audit['v5_manifests_sha256']['dataset.csv']:
        raise ValueError('Frozen V5 dataset changed.')
    encoder_sha = core.digest(args.backbone)
    stores = [RecordingStore(directory, [r for r in base_rows if r['split'] == 'train'], encoder_sha),
              RecordingStore(supplement, extra_rows, encoder_sha)]
    validation = RecordingStore(directory, [r for r in base_rows if r['split'] == 'validation'], encoder_sha)
    examples = [(index, identity) for index, store in enumerate(stores) for identity in sorted(store.rows)]
    vocabulary = sorted(({p['phone'] for store in stores for phones in store.phones.values() for p in phones} - {'unk'}) | {'pau'})
    args.classes = len(vocabulary)
    ax = vocabulary.index('ax')
    targets, counts = {}, np.zeros(len(vocabulary), dtype=np.int64)
    for store_index, identity in examples:
        store = stores[store_index]
        labels = frame_targets(store.phones[identity], len(store.sequence(identity)), vocabulary, store.durations[identity])
        targets[(store_index, identity)] = torch.from_numpy(labels)
        counts += np.bincount(labels[labels >= 0], minlength=len(vocabulary))
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    weights = torch.from_numpy(np.clip(weights / weights.mean(), .25, 3).astype(np.float32))
    reference_frames = sum(float(weights[labels[labels >= 0]].sum()) for labels in targets.values()) / len(examples)
    reference_regions = sum(len(store.rows[identity]) for store in stores for identity in store.rows) / len(examples)
    config = settings(args)
    config['frame_class_counts'] = counts.tolist()
    provenance = {'dataset_sha256': dataset_sha, 'backbone_sha256': encoder_sha, 'settings': config,
        'supplement_manifest_sha256': core.digest(supplement / 'data/manifests/dataset.csv'),
        'supplement_source_sha256': audit['source_sha256'], 'test_used_for_selection': False,
        'initialization': 'Original public frozen XLSR; fresh full-recording phone GRU',
        'source_code_sha256': {name: core.digest(ROOT / name) for name in HELPERS}}
    args.run_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.run_dir / 'training_protocol.json'
    if protocol_path.exists() and core.read_json(protocol_path) != provenance:
        raise ValueError('Preserving run with different source, settings or data.')
    core.write_json(protocol_path, provenance)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = improve.PhoneHead(**config['head_config'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    history, best_key, best_epoch, first_epoch, elapsed = [], None, 0, 1, 0.
    if args.resume:
        saved = torch.load(args.run_dir / 'last.pt', weights_only=True)
        if saved['provenance'] != provenance:
            raise ValueError('Exact resume configuration differs.')
        model.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, best_key, best_epoch = saved['history'], tuple(saved['best_key']), saved['best_epoch']
        first_epoch, elapsed = saved['epoch'] + 1, saved['seconds']
        rng.bit_generator.state = saved['numpy_random_state']
        torch.set_rng_state(saved['torch_random_state'])
        del saved
    started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        order = rng.permutation(len(examples))
        losses = []
        for batch_start in range(0, len(order), args.accumulate):
            batch = order[batch_start:batch_start + args.accumulate]
            optimizer.zero_grad(set_to_none=True)
            for item in batch:
                key = examples[int(item)]
                store_index, identity = key
                store = stores[store_index]
                logits = model(store.sequence(identity).unsqueeze(0))[0]
                labels = targets[key]
                loss = nn.functional.cross_entropy(logits, labels, weight=weights, ignore_index=-100, reduction='sum') / reference_frames
                rows = store.rows[identity]
                regions = region_logits(logits, [int(r['window_start_sample']) for r in rows])
                loss += args.window_loss * improve.window_presence_loss(regions, ax, torch.tensor([int(r['label']) for r in rows], dtype=torch.float32)) * len(rows) / reference_regions
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite full-recording training loss.')
                (loss / len(batch)).backward()
                losses.append(float(loss.detach()))
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if (batch_start + len(batch)) % 128 == 0:
                print(f'Recording GRU epoch {epoch}: {batch_start+len(batch)}/{len(examples)}, loss={np.mean(losses):.4f}', flush=True)
        model.eval()
        alternatives = {pool: ([], []) for pool in [1, 2, 3]}
        with torch.inference_mode():
            for identity in sorted(validation.rows):
                logits = model(validation.sequence(identity).unsqueeze(0))[0]
                rows = validation.rows[identity]
                starts = [int(r['window_start_sample']) for r in rows]
                labels = [int(r['label']) for r in rows]
                for pool, (truths, scores) in alternatives.items():
                    truths.extend(labels)
                    scores.extend(region_scores(logits, starts, ax, pool).tolist())
        choices = [(improve.validation_selection(truths, scores, 'min_pr'), pool) for pool, (truths, scores) in alternatives.items()]
        metric, pool = max(choices, key=lambda choice: (min(choice[0]['precision'], choice[0]['recall']), choice[0]['f1'], choice[0]['accuracy'], -choice[0]['fpr']))
        key = (min(metric['precision'], metric['recall']), metric['f1'], metric['accuracy'], -metric['fpr'])
        history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_precision': metric['precision'],
            'validation_recall': metric['recall'], 'validation_f1': metric['f1'], 'seconds': elapsed + time.perf_counter()-started})
        core.write_csv(args.run_dir / 'history.csv', history)
        print(f'Recording GRU epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}', flush=True)
        if best_key is None or key > best_key:
            best_key, best_epoch = key, epoch
            checkpoint = {'backend': 'wav2vec2_recording_phone_head', 'target_phones': ['ax'], 'target_symbol': '\u0259',
                'epoch': epoch, 'head_config': config['head_config'], 'head_state_dict': model.state_dict(),
                'phone_vocabulary': vocabulary, 'threshold': metric['threshold'], 'pool_frames': pool,
                'cpu_threads': args.threads, 'settings': config, 'validation_metrics': metric,
                'threshold_selection_split': 'validation', 'provenance': provenance,
                'dataset_sha256': dataset_sha, 'backbone_sha256': encoder_sha}
            core.save_torch(args.run_dir / 'best.pt', checkpoint)
            core.write_json(args.run_dir / 'best_summary.json', {**provenance, 'best_epoch': epoch, 'validation_metrics': metric})
        core.save_torch(args.run_dir / 'last.pt', {'provenance': provenance, 'epoch': epoch, 'best_epoch': best_epoch,
            'best_key': best_key, 'history': history, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'numpy_random_state': rng.bit_generator.state, 'torch_random_state': torch.get_rng_state(),
            'seconds': elapsed + time.perf_counter()-started})
        if epoch - best_epoch >= args.patience:
            break
    core.write_json(args.run_dir / 'training_summary.json', {**provenance, 'best_epoch': best_epoch,
        'epochs_completed': len(history), 'seconds': elapsed + time.perf_counter()-started,
        'validation_metrics': core.read_json(args.run_dir / 'best_summary.json')['validation_metrics']})


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', type=Path, default=BACKBONE)
    parser.add_argument('--supplement', type=Path, default=ROOT / 'training_suitcase')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--window-loss', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    train(arguments())

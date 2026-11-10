"""Adapt the last transformer layers on CPU using cached, frozen recording prefixes."""
import argparse
import gc
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import schwa as core
import improve
from lexical_context import pronunciation_ids
from frozen_phoneme import RECORDING, rebuild_acoustic, encoder_prefix, aligned_window_features

ROOT = Path(__file__).resolve().parent


def chunk_plan(samples):
    samples = max(samples, RECORDING['minimum_samples'])
    count = (samples - 400) // RECORDING['hop_samples'] + 1
    if samples <= RECORDING['maximum_samples']:
        return count, [{'input_start': 0, 'input_end': samples, 'core_start': 0, 'core_end': samples}]
    chunks = []
    for start in range(0, samples, RECORDING['core_samples']):
        end = min(samples, start + RECORDING['core_samples'])
        centers = RECORDING['center_sample'] + np.arange(count) * RECORDING['hop_samples']
        if not np.any((centers >= start) & (centers < end)):
            continue
        chunks.append({'input_start': max(0, start - RECORDING['margin_samples']),
                       'input_end': min(samples, end + RECORDING['margin_samples']),
                       'core_start': start, 'core_end': end})
    return count, chunks


def prefix_paths(directory, record, encoder_sha, depth):
    recipe = {'encoder_sha256': encoder_sha, 'source_audio_sha256': record['audio_sha256'],
              'prefix_layers': depth, 'recipe': RECORDING, 'storage': 'float16'}
    base = directory / 'data/frozen_prefix_features' / core.object_digest(recipe)
    return base, recipe


def read_prefix(directory, record, encoder_sha, depth):
    base, recipe = prefix_paths(directory, record, encoder_sha, depth)
    marker = core.read_json(base.with_suffix('.json'))
    if marker['identity'] != recipe:
        raise ValueError('Frozen prefix identity differs.')
    chunks = []
    for chunk in marker['chunks']:
        path = base.parent / chunk['file']
        if core.digest(path) != chunk['sha256']:
            raise ValueError('Frozen prefix cache changed.')
        values = torch.load(path, weights_only=True, mmap=True)
        if list(values.shape) != chunk['shape'] or not torch.isfinite(values).all():
            raise ValueError('Invalid prefix tensor.')
        chunks.append((chunk, values))
    return marker['frames'], chunks


def prepare(args):
    directory = improve.activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    selected = {r['recording_id'] for r in rows if r['split'] in args.splits}
    records = [r for r in core.read_csv(directory / 'data/manifests/recordings.csv') if r['recording_id'] in selected]
    encoder_sha = core.digest(args.backbone)
    state = torch.load(args.backbone, weights_only=True, mmap=True)
    if state.get('prefix_rounding_layer') or state.get('fine_tuning'):
        raise ValueError('Prefix initialization must use the original public encoder.')
    model = rebuild_acoustic(state)
    model.requires_grad_(False)
    started = time.perf_counter()
    with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for index, record in enumerate(records, 1):
            base, recipe = prefix_paths(directory, record, encoder_sha, args.prefix_layers)
            audio_path = directory / record['audio_path']
            if core.digest(audio_path) != record['audio_sha256']:
                raise ValueError('Prefix source audio changed.')
            if base.with_suffix('.json').exists():
                read_prefix(directory, record, encoder_sha, args.prefix_layers)
            else:
                audio, _ = core.load_audio(audio_path)
                wave = torch.from_numpy(audio)
                if len(wave) < RECORDING['minimum_samples']:
                    wave = nn.functional.pad(wave, (0, RECORDING['minimum_samples'] - len(wave)))
                frames, chunks = chunk_plan(len(wave))
                saved = []
                for number, chunk in enumerate(chunks):
                    inputs = wave[chunk['input_start']:chunk['input_end']].unsqueeze(0)
                    if model.normalize_waveform:
                        inputs = nn.functional.layer_norm(inputs, (inputs.shape[-1],))
                    values = encoder_prefix(model, inputs, args.prefix_layers)[0].half().contiguous()
                    path = base.with_name(base.name + '_' + str(number)).with_suffix('.pt')
                    if path.exists():
                        existing = torch.load(path, weights_only=True, mmap=True)
                        if not torch.equal(existing, values):
                            raise ValueError('Preserving a differing unfinished prefix tensor.')
                    else:
                        core.save_torch(path, values)
                    saved.append({**chunk, 'file': path.name, 'sha256': core.digest(path), 'shape': list(values.shape)})
                core.write_json(base.with_suffix('.json'), {'identity': recipe, 'frames': frames, 'chunks': saved})
            if index % 16 == 0 or index == len(records):
                print(f'Frozen prefixes {index}/{len(records)} recordings, {time.perf_counter()-started:.1f}s', flush=True)


def tail_features(model, prefix, depth):
    values = prefix.float().unsqueeze(0)
    for layer in model.encoder.transformer.layers[depth:]:
        values, _ = layer(values)
    if not model.encoder.transformer.layer_norm_first:
        values = model.encoder.transformer.layer_norm(values)
    return torch.cat([values[0], model.aux(values)[0]], dim=-1).half()


def stitch_tail(model, frames, chunks, depth):
    result = None
    centers = RECORDING['center_sample'] + torch.arange(frames) * RECORDING['hop_samples']
    for chunk, prefix in chunks:
        values = tail_features(model, prefix, depth)
        if result is None:
            result = values.new_empty((frames, values.shape[-1]))
        indices = torch.nonzero((centers >= chunk['core_start']) & (centers < chunk['core_end'])).flatten()
        local = indices - chunk['input_start'] // RECORDING['hop_samples']
        if torch.any(local < 0) or torch.any(local >= len(values)):
            raise ValueError('Adapted encoder chunks do not match recording frames.')
        result[indices] = values[local]
    return result


def phone_labels(directory, record, vocabulary, frames):
    path = directory / record['lab_path']
    if core.digest(path) != record['lab_sha256']:
        raise ValueError('Manual phone annotations changed.')
    audio_path = directory / record['audio_path']
    if core.digest(audio_path) != record['audio_sha256']:
        raise ValueError('Manual annotation source audio changed.')
    audio, _ = core.load_audio(audio_path)
    phones, _ = core.parse_phones(path.read_text(), len(audio) / 16000)
    indices = {phone: index for index, phone in enumerate(vocabulary)}
    result = np.full(frames, -100, dtype=np.int64)
    for index, center in enumerate((RECORDING['center_sample'] + np.arange(frames) * 320) / 16000):
        phone = next((p for p in phones if p['start'] <= center < p['end']), None)
        if phone is None:
            result[index] = indices['pau']
        elif min(center - phone['start'], phone['end'] - center) >= .005:
            result[index] = indices.get(phone['phone'], -100)
    return result


def selection_key(metric):
    return min(metric['precision'], metric['recall']), metric['f1'], metric['accuracy'], -metric['fpr']


def ctc_segments(phones, vocabulary, frames):
    """Known annotation spans are training targets, never inference inputs."""
    centers = (RECORDING['center_sample'] + np.arange(frames) * RECORDING['hop_samples']) / 16000
    groups, current, start = [], [], 0.
    for phone in phones:
        if phone['phone'] == 'unk':
            groups.append((start, phone['start'], current))
            current, start = [], phone['end']
        elif phone['phone'] != 'pau':
            token = 'AH0' if phone['phone'] == 'ax' else phone['phone'].upper()
            current.extend(pronunciation_ids([token], vocabulary))
    groups.append((start, float('inf'), current))
    result = []
    for start, end, targets in groups:
        if not targets:
            continue
        indices = np.flatnonzero((centers >= start) & (centers < end))
        minimum = len(targets) + sum(a == b for a, b in zip(targets, targets[1:]))
        if len(indices) < minimum:
            raise ValueError('Annotated CTC segment has too few frames for phones/repeats.')
        if any(t == 0 for t in targets):
            raise ValueError('CTC target contains the blank token.')
        result.append((int(indices[0]), int(indices[-1]) + 1, torch.tensor(targets, dtype=torch.long)))
    if not result:
        raise ValueError('Recording has no known CTC phone targets.')
    return result


def recording_ctc_loss(logits, segments):
    criterion = nn.CTCLoss(blank=0, reduction='sum', zero_infinity=False)
    losses, count = [], 0
    for start, end, targets in segments:
        values = logits[start:end].float().log_softmax(-1).unsqueeze(1)
        loss = criterion(values, targets, torch.tensor([end - start]), torch.tensor([len(targets)]))
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite phoneme CTC loss; invalid target is not silently ignored.')
        losses.append(loss)
        count += len(targets)
    return torch.stack(losses).sum() / count


def supervision_targets(labels, vocabulary, task):
    """Keep unknown/boundary frames ignored when reducing phones to AX presence."""
    if task == 'phones':
        return labels
    if task != 'binary':
        raise ValueError('Unknown adaptation task.')
    return np.where(labels == -100, -100,
                    (labels == vocabulary.index('ax')).astype(np.int64)).astype(np.int64)


def train(args):
    directory = improve.activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    encoder_sha = core.digest(args.backbone)
    state = torch.load(args.backbone, weights_only=True, mmap=True)
    if state.get('fine_tuning') or state.get('prefix_rounding_layer'):
        raise ValueError('Initialize only from the original public encoder.')
    if not 0 < args.prefix_layers < state['layers']:
        raise ValueError('Fine tuning must leave a nonempty frozen prefix and tail.')
    if args.accumulate < 1 or args.epochs < 1 or args.warmup_epochs < 0:
        raise ValueError('Invalid fine tuning epoch/accumulation settings.')
    resume = getattr(args, 'resume', False)
    if args.run_dir.exists() and (args.run_dir / 'best.pt').exists() and not resume:
        raise ValueError('Existing learned weights preserved.')
    if resume and not (args.run_dir / 'last.pt').exists():
        raise ValueError('Resume requires an actual saved last-epoch checkpoint.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [r for r in rows if r['split'] == 'train']
    _, source_vocabulary = improve.frame_labels(directory, train_rows)
    task = getattr(args, 'task', 'phones')
    if task not in ('phones', 'binary'):
        raise ValueError('Unknown adaptation task.')
    vocabulary = ['non_ax', 'ax'] if task == 'binary' else source_vocabulary
    ax = vocabulary.index('ax')
    config = {'classes': len(vocabulary), 'hidden': args.hidden, 'dropout': .4,
              'temporal': 'none', 'input_dim': state['model_config']['encoder_embed_dim'] + state['model_config']['aux_num_out'],
              'group_sizes': [state['model_config']['encoder_embed_dim'], state['model_config']['aux_num_out']]}
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.use_deterministic_algorithms(True)
    model = rebuild_acoustic(state)
    model.requires_grad_(False)
    tail_parameters = []
    for layer in model.encoder.transformer.layers[args.prefix_layers:]:
        for parameter in layer.parameters():
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            tail_parameters.append(parameter)
    ctc_weight = getattr(args, 'ctc_weight', 0.)
    if ctc_weight < 0:
        raise ValueError('CTC loss weight must be nonnegative.')
    if ctc_weight:
        for parameter in model.aux.parameters():
            parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
            tail_parameters.append(parameter)
    model.prefix_rounding_layer = args.prefix_layers
    head = improve.PhoneHead(**config)
    groups = {}
    for row in rows:
        if row['split'] in ('train', 'validation'):
            groups.setdefault(row['recording_id'], []).append(row)
    records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv') if r['recording_id'] in groups}
    prefixes, targets, ctc_targets, counts = {}, {}, {}, np.zeros(len(vocabulary), dtype=np.int64)
    for identity, record in records.items():
        frames, _ = read_prefix(directory, record, encoder_sha, args.prefix_layers)
        prefixes[identity] = frames
        if record['split'] == 'train':
            source_labels = phone_labels(directory, record, source_vocabulary, frames)
            targets[identity] = supervision_targets(source_labels, source_vocabulary, task)
            counts += np.bincount(targets[identity][targets[identity] >= 0], minlength=len(vocabulary))
            if ctc_weight:
                audio, _ = core.load_audio(directory / record['audio_path'])
                phones, _ = core.parse_phones((directory / record['lab_path']).read_text(), len(audio) / 16000)
                ctc_targets[identity] = ctc_segments(phones, state['phoneme_vocabulary'], frames)
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.clip(weights, .25, 3).astype(np.float32)), ignore_index=-100)
    optimizer = torch.optim.AdamW([{'params': tail_parameters, 'lr': args.tail_lr},
                                   {'params': head.parameters(), 'lr': args.lr}], weight_decay=.01)
    settings = {'dataset': 'v5', 'task': task, 'feature_context': 'recording', 'acoustic_representation': 'joint',
                'head_config': config, 'selection_objective': 'min_pr', 'prefix_layers': args.prefix_layers,
                'tail_learning_rate': args.tail_lr, 'head_learning_rate': args.lr, 'seed': args.seed,
                'frame_supervision': True, 'window_loss_weight': args.window_loss, 'cpu_threads': args.threads,
                'warmup_epochs': args.warmup_epochs, 'gradient_accumulation': args.accumulate,
                'epochs': args.epochs, 'patience': args.patience,
                'frame_class_counts': counts.tolist(),
                'source_phone_vocabulary': source_vocabulary,
                'target_mapping': 'AX versus all other known phones; unknown/boundary frames ignored' if task == 'binary' else 'All manual phone classes'}
    settings.update(ctc_loss_weight=ctc_weight, ctc_output_layer_trainable=bool(ctc_weight),
                    ctc_unknown_policy='Split known spans at unknown annotations; all focal window labels remain unchanged',
                    ctc_target_mapping='AH0/AX to bare schwa; AH1/AH2 to open-mid central vowel; ER to rhotic vowel')
    provenance = {'dataset_sha256': core.digest(directory / 'data/manifests/dataset.csv'),
                  'base_encoder_sha256': encoder_sha, 'code_sha256': core.digest(__file__), 'settings': settings,
                  'helper_code_sha256': {name: core.digest(ROOT / name) for name in ['frozen_phoneme.py', 'improve.py', 'lexical_context.py']},
                  'test_used_for_selection': False, 'initialization': 'Original public encoder; fresh random head.'}
    protocol_path = args.run_dir / 'training_protocol.json'
    if protocol_path.exists() and core.read_json(protocol_path) != provenance:
        raise ValueError('Training provenance changed; preserving existing run.')
    core.write_json(protocol_path, provenance)
    history, best_key, best_epoch = [], None, 0
    start_epoch, elapsed = 1, 0.
    adapted_names = {name for name, _ in model.named_parameters()
                     if name.startswith(tuple('encoder.transformer.layers.' + str(i) + '.'
                                              for i in range(args.prefix_layers, state['layers'])))
                     or (ctc_weight and name.startswith('aux.'))}
    if resume:
        saved = torch.load(args.run_dir / 'last.pt', weights_only=True, mmap=True)
        if saved['provenance'] != provenance or set(saved['adapted_state_dict']) != adapted_names:
            raise ValueError('Resume provenance or adapted tensor names differ.')
        current = model.state_dict()
        with torch.no_grad():
            for name, values in saved['adapted_state_dict'].items():
                current[name].copy_(values)
        head.load_state_dict(saved['head_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        rng.bit_generator.state = saved['numpy_random_state']
        torch.set_rng_state(saved['torch_random_state'])
        history, best_epoch, best_key = saved['history'], saved['best_epoch'], tuple(saved['best_key'])
        start_epoch, elapsed = saved['epoch'] + 1, saved['seconds']
        del saved
        print(f'Resuming after completed epoch {start_epoch - 1}', flush=True)
    started = time.perf_counter()
    train_ids = sorted(i for i, r in records.items() if r['split'] == 'train')
    validation_ids = sorted(i for i, r in records.items() if r['split'] == 'validation')
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for epoch in range(start_epoch, args.epochs + 1):
            model.eval()
            head.train()
            for parameter in tail_parameters:
                parameter.requires_grad_(epoch > args.warmup_epochs)
            optimizer.zero_grad(set_to_none=True)
            losses = []
            order = rng.permutation(train_ids).tolist()
            for step, identity in enumerate(order, 1):
                frames, chunks = read_prefix(directory, records[identity], encoder_sha, args.prefix_layers)
                features = stitch_tail(model, frames, chunks, args.prefix_layers)
                labels = torch.from_numpy(targets[identity])
                dense_logits = head(features.float().unsqueeze(0))[0]
                valid = labels != -100
                loss = criterion(dense_logits[valid], labels[valid]) if valid.any() else dense_logits.sum() * 0
                regions = groups[identity]
                window_features = aligned_window_features(features, [int(r['window_start_sample']) for r in regions])
                logits = head(window_features)
                loss = loss + args.window_loss * improve.window_presence_loss(logits, ax, torch.tensor([int(r['label']) for r in regions], dtype=torch.float32))
                if ctc_weight:
                    loss = loss + ctc_weight * recording_ctc_loss(features[:, state['model_config']['encoder_embed_dim']:], ctc_targets[identity])
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite adapted-encoder training loss.')
                (loss / args.accumulate).backward()
                losses.append(float(loss.detach()))
                if step % args.accumulate == 0 or step == len(order):
                    nn.utils.clip_grad_norm_([*tail_parameters, *head.parameters()], 1.)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 64 == 0:
                    print(f'Adapt epoch {epoch:02d}: {step}/{len(order)} recordings, loss={np.mean(losses):.4f}', flush=True)
            model.eval()
            head.eval()
            all_scores = {p: [] for p in (1, 2, 3)}
            truths = []
            with torch.inference_mode():
                for identity in validation_ids:
                    frames, chunks = read_prefix(directory, records[identity], encoder_sha, args.prefix_layers)
                    features = stitch_tail(model, frames, chunks, args.prefix_layers)
                    regions = groups[identity]
                    windows = aligned_window_features(features, [int(r['window_start_sample']) for r in regions])
                    for pool in all_scores:
                        all_scores[pool].extend(improve.class_scores(head, windows.numpy(), ax, pool).tolist())
                    truths.extend(int(r['label']) for r in regions)
            selections = [(improve.validation_selection(truths, scores, 'min_pr'), pool) for pool, scores in all_scores.items()]
            metric, pool = max(selections, key=lambda pair: selection_key(pair[0]))
            history.append({'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation_precision': metric['precision'],
                            'validation_recall': metric['recall'], 'validation_f1': metric['f1'], 'seconds': elapsed + time.perf_counter() - started})
            core.write_csv(args.run_dir / 'history.csv', history)
            print(f'Adapt epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}, F1={metric["f1"]:.4f}', flush=True)
            if best_key is None or selection_key(metric) > best_key:
                best_key, best_epoch = selection_key(metric), epoch
                adapted = dict(state)
                adapted.update(model_state_dict=model.state_dict(), prefix_rounding_layer=args.prefix_layers,
                               fine_tuning=provenance, weight_storage='float16 frozen linear / float32 adapted tail and convolution')
                encoder_path = args.run_dir / 'adapted_encoder.pt'
                core.save_torch(encoder_path, adapted)
                selection = core.read_json(directory / 'data/manifests/source_selection.json')
                checkpoint = {'backend': 'wav2vec2_phone_head', 'target_phones': ['ax'], 'target_symbol': '\u0259',
                              'epoch': epoch, 'threshold': metric['threshold'], 'pool_frames': pool,
                              'head_config': config, 'head_state_dict': head.state_dict(), 'phone_vocabulary': vocabulary,
                              'cpu_threads': args.threads, 'settings': settings, 'validation_metrics': metric,
                              'threshold_selection_split': 'validation',
                              'dataset_sha256': provenance['dataset_sha256'], 'backbone_sha256': core.digest(encoder_path),
                              'source_metadata': selection, 'code_sha256': provenance['code_sha256']}
                core.save_torch(args.run_dir / 'best.pt', checkpoint)
                core.write_json(args.run_dir / 'best_summary.json', {**provenance, 'best_epoch': epoch,
                                'validation_metrics': metric, 'adapted_encoder_sha256': checkpoint['backbone_sha256']})
                gc.collect()
            if ctc_weight:
                core.save_torch(args.run_dir / 'last.pt', {
                    'provenance': provenance, 'epoch': epoch, 'best_epoch': best_epoch,
                    'best_key': best_key, 'history': history,
                    'adapted_state_dict': {name: value for name, value in model.state_dict().items() if name in adapted_names},
                    'head_state_dict': head.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                    'numpy_random_state': rng.bit_generator.state, 'torch_random_state': torch.get_rng_state(),
                    'seconds': elapsed + time.perf_counter() - started})
            if epoch - best_epoch >= args.patience:
                break
    best = core.read_json(args.run_dir / 'best_summary.json')
    core.write_json(args.run_dir / 'training_summary.json', {**provenance, 'best_epoch': best_epoch,
                    'epochs_completed': len(history), 'seconds': elapsed + time.perf_counter() - started,
                    'validation_metrics': best['validation_metrics']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', type=Path, required=True)
    parser.add_argument('--prefix-layers', type=int, default=22)
    parser.add_argument('--threads', type=int, default=2)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['train', 'validation'])
    p.set_defaults(function=prepare)
    p = commands.add_parser('train')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--task', choices=['phones', 'binary'], default='phones')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--warmup-epochs', type=int, default=2)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--tail-lr', type=float, default=.00001)
    p.add_argument('--window-loss', type=float, default=.25)
    p.add_argument('--ctc-weight', type=float, default=0.)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--accumulate', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(function=train)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    args.function(args)

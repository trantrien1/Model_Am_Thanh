"""Train fresh Schwa heads with lexical features inferred solely from audio on CPU."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import improve
import schwa as core
from frozen_phoneme import RECORDING, rebuild_acoustic, recording_features, aligned_window_features
from lexical_context import (ASR_PATH, DICTIONARY_SOURCE, LEXICAL_RECIPE, load_dictionary,
                             bounded_lexical_features, verify_dictionary_source)

ROOT = Path(__file__).resolve().parent
MAIN_ENCODER = ROOT / 'data/pretrained/phoneme_xlsr53_int8.pt'


class CombinedFeatures:
    def __init__(self, acoustic, lexical):
        if acoustic.shape[:2] != lexical.shape[:2]:
            raise ValueError('Acoustic/lexical window geometry differs.')
        self.acoustic, self.lexical = acoustic, lexical
        self.shape = (*acoustic.shape[:2], acoustic.shape[-1] + lexical.shape[-1])

    def __len__(self):
        return len(self.acoustic)

    def __getitem__(self, index):
        return np.concatenate([self.acoustic[index], self.lexical[index]], axis=-1)


def metadata(main_encoder=None):
    main_encoder = Path(main_encoder or MAIN_ENCODER)
    verify_dictionary_source()
    vocabulary_path = ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json'
    descriptor = core.read_json(ASR_PATH.with_suffix('.json'))
    asr_sha = core.digest(ASR_PATH)
    if asr_sha != descriptor['prepared_sha256'] or descriptor['source_checkpoint_sha256'] != '488fd4f16de84438ffc945334278c1b9fb9b7159a806c1080b16111a958c945d':
        raise ValueError('Pinned public English ASR descriptor changed.')
    if core.digest(vocabulary_path) != 'd732ab2456c0c017930001dc9af0b41b3b93d25b2eb9740bf9d925508d7d87d0':
        raise ValueError('Pinned public phoneme vocabulary changed.')
    main_sha = core.digest(main_encoder)
    if main_sha not in ('d5b0e5484d55cb701a556e7ef25c11a94c842815de24832b0541f4887e3a7cf0',
                        '4c15648baf544efa819f0798cd76664a131960e45345d9f57091c5443558bb0f'):
        raise ValueError('Unrecognized public phoneme encoder.')
    result = {'main_encoder_sha256': main_sha, 'asr_encoder_sha256': asr_sha,
            'dictionary_sha256': core.digest(DICTIONARY_SOURCE / 'cmudict.dict'),
            'vocabulary_sha256': core.digest(vocabulary_path), 'acoustic_recipe': RECORDING,
            'lexical_recipe': LEXICAL_RECIPE, 'reference_transcript_consumed': False,
            'manual_phones_consumed_as_features': False}
    if main_sha == '4c15648baf544efa819f0798cd76664a131960e45345d9f57091c5443558bb0f':
        result['feature_compute_threads'] = {'phoneme': 2, 'word_asr': 1, 'alignment': 1}
    return result


def window_paths(directory, rows, split, recipe):
    if split not in ('train', 'validation') or any(r['split'] != split for r in rows):
        raise ValueError('Development lexical caches must not consume test regions.')
    identity = core.object_digest({'recipe': recipe, 'region_ids': [r['sample_id'] for r in rows],
                                   'starts': [r['window_start_sample'] for r in rows]})
    path = directory / 'data/lexical_window_features' / (split + '_' + identity[:20] + '.npy')
    return path, path.with_suffix('.json')


def array_cache(path, expected):
    marker = core.read_json(path.with_suffix('.json'))
    if marker['identity'] != expected or core.digest(path) != marker['sha256']:
        raise ValueError('Lexical/ASR cache identity changed.')
    values = np.load(path, allow_pickle=False, mmap_mode='r')
    if list(values.shape) != marker['shape'] or not np.isfinite(values).all():
        raise ValueError('Invalid lexical/ASR cache values.')
    return values


def save_array(path, values, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.partial.npy')
    np.save(part, values, allow_pickle=False)
    part.replace(path)
    core.write_json(path.with_suffix('.json'), {'identity': identity, 'sha256': core.digest(path),
                                              'shape': list(values.shape)})


def lexical_window_cache(directory, rows, split, recipe):
    path, _ = window_paths(directory, rows, split, recipe)
    expected = {'recipe': recipe, 'row_ids': [r['sample_id'] for r in rows]}
    values = array_cache(path, expected)
    if values.shape != (len(rows), 24, 4):
        raise ValueError('Unexpected lexical window cache geometry.')
    return values


def extract(args):
    directory = improve.activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    torch.set_num_threads(args.threads)
    recipe = metadata()
    feature_threads = recipe.get('feature_compute_threads',
                                 {'phoneme': args.threads, 'word_asr': args.threads, 'alignment': args.threads})
    records = {r['recording_id']: r for r in core.read_csv(directory / 'data/manifests/recordings.csv')}
    dictionary = load_dictionary(DICTIONARY_SOURCE / 'cmudict.dict')
    vocabulary = core.read_json(ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json')
    state = torch.load(ASR_PATH, weights_only=True, mmap=True)
    labels, model = state['labels'], None
    for split in args.splits:
        subset = [r for r in rows if r['split'] == split]
        path, marker = window_paths(directory, subset, split, recipe)
        if path.exists() and marker.exists():
            lexical_window_cache(directory, subset, split, recipe)
            print('Verified lexical windows:', split, flush=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.partial.npy')
        output = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float16, shape=(len(subset), 24, 4))
        groups = {}
        for index, row in enumerate(subset):
            groups.setdefault(row['recording_id'], []).append((index, row))
        started = time.perf_counter()
        for number, (identity, group) in enumerate(sorted(groups.items()), 1):
            record = records[identity]
            if record['split'] != split:
                raise ValueError('Lexical source split differs.')
            source_path = directory / record['audio_path']
            if core.digest(source_path) != record['audio_sha256']:
                raise ValueError('Lexical source audio changed.')
            local_recipe = {'recipe': recipe, 'source_audio_sha256': record['audio_sha256']}
            local = directory / 'data/lexical_recording_features' / (core.object_digest(local_recipe) + '.npy')
            if local.exists() and local.with_suffix('.json').exists():
                lexical = torch.from_numpy(np.array(array_cache(local, local_recipe))).float()
            else:
                asr_recipe = {'source_audio_sha256': record['audio_sha256'],
                              'encoder_sha256': recipe['asr_encoder_sha256'], 'recipe': RECORDING}
                asr_path = directory / 'data/word_asr_features' / (core.object_digest(asr_recipe) + '.npy')
                if asr_path.exists() and asr_path.with_suffix('.json').exists():
                    asr = torch.from_numpy(np.array(array_cache(asr_path, asr_recipe))).float()
                else:
                    torch.set_num_threads(feature_threads['word_asr'])
                    if model is None:
                        model = rebuild_acoustic(state)
                    audio, _ = core.load_audio(source_path)
                    with torch.inference_mode():
                        asr = recording_features(model, torch.from_numpy(audio), state['layers'])
                    save_array(asr_path, asr.numpy(), asr_recipe)
                    asr = asr.float()
                teacher_recipe = {'encoder': recipe['main_encoder_sha256'],
                                  'source_audio': record['audio_sha256'], 'recipe': RECORDING, 'representation': 'joint'}
                teacher = directory / 'data/recording_transfer_features' / (core.object_digest(teacher_recipe) + '.npy')
                teacher_marker = core.read_json(teacher.with_suffix('.json'))
                if (core.digest(teacher) != teacher_marker['sha256']
                        or teacher_marker['source_audio_sha256'] != record['audio_sha256']
                        or teacher_marker['backbone_sha256'] != recipe['main_encoder_sha256']
                        or teacher_marker['recipe'] != RECORDING or teacher_marker['representation'] != 'joint'):
                    raise ValueError('Public phoneme recording cache changed.')
                raw_teacher = np.load(teacher, allow_pickle=False)
                if raw_teacher.shape[-1] != 1416:
                    raise ValueError('Unexpected public phoneme geometry.')
                phonemes = torch.from_numpy(raw_teacher[:, 1024:]).float()
                torch.set_num_threads(feature_threads['alignment'])
                with torch.inference_mode():
                    lexical, transcripts = bounded_lexical_features(asr, phonemes, dictionary, labels, vocabulary)
                if not torch.isfinite(lexical).all():
                    raise ValueError('Nonfinite lexical recording features.')
                save_array(local, lexical.half().numpy(), local_recipe)
                core.write_json(local.with_suffix('.words.json'), {'inferred_words': transcripts,
                                'reference_transcript_consumed': False, 'source_audio_sha256': record['audio_sha256']})
                lexical = lexical.half().float()
            indices = [i for i, _ in group]
            output[indices] = aligned_window_features(lexical, [int(r['window_start_sample']) for _, r in group]).numpy()
            if number % 16 == 0 or number == len(groups):
                print(f'Lexical {split}: {number}/{len(groups)} recordings, {time.perf_counter()-started:.1f}s', flush=True)
        output.flush()
        del output
        temporary.replace(path)
        core.write_json(marker, {'identity': {'recipe': recipe, 'row_ids': [r['sample_id'] for r in subset]},
                                'shape': [len(subset), 24, 4], 'sha256': core.digest(path)})


def selection_key(metric):
    return min(metric['precision'], metric['recall']), metric['f1'], metric['accuracy'], -metric['fpr']


def train(args):
    directory = improve.activate_dataset('v5')
    rows = core.audit_dataset(verify_wavs=False)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    if (args.run_dir / 'best.pt').exists():
        raise ValueError('Preserving learned lexical head.')
    args.run_dir.mkdir(parents=True, exist_ok=True)
    recipe = metadata()
    improve.BACKBONE, improve.CONTEXT, improve.REPRESENTATION = MAIN_ENCODER, 'recording', 'joint'
    train_rows = [r for r in rows if r['split'] == 'train']
    validation_rows = [r for r in rows if r['split'] == 'validation']
    training = CombinedFeatures(improve.feature_cache(directory, train_rows, 'train'),
                                lexical_window_cache(directory, train_rows, 'train', recipe))
    validation = CombinedFeatures(improve.feature_cache(directory, validation_rows, 'validation'),
                                  lexical_window_cache(directory, validation_rows, 'validation', recipe))
    targets, vocabulary = improve.frame_labels(directory, train_rows)
    ax = vocabulary.index('ax')
    if args.task == 'binary':
        targets = np.where(targets == -100, -100, (targets == ax).astype(np.int64))
        vocabulary, ax = ['non_ax', 'ax'], 1
    counts = np.bincount(targets[targets >= 0], minlength=len(vocabulary))
    weights = np.sqrt(np.maximum(counts, 1).sum() / np.maximum(counts, 1))
    weights = np.clip(weights / np.median(weights), .25, 3).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights), ignore_index=-100)
    config = {'classes': len(vocabulary), 'hidden': args.hidden, 'dropout': args.dropout,
              'temporal': args.temporal, 'input_dim': 1420, 'group_sizes': [1024, 392, 4]}
    model = improve.PhoneHead(**config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    truth = torch.tensor([int(r['label']) for r in train_rows], dtype=torch.float32)
    validation_truth = [int(r['label']) for r in validation_rows]
    provenance = {'dataset': 'v5', 'dataset_sha256': core.digest(directory / 'data/manifests/dataset.csv'),
        'lexical': recipe, 'test_used_for_selection': False, 'feature_context': 'recording',
        'initialization': 'Original public phoneme and English ASR encoders; fresh random head',
        'code_sha256': {name: core.digest(ROOT / name) for name in ['lexical_train_cpu.py', 'lexical_context.py', 'improve.py', 'frozen_phoneme.py']},
        'settings': {key: getattr(args, key) for key in ['task', 'seed', 'threads', 'epochs', 'patience', 'lr', 'window_loss', 'batch_size']}}
    core.write_json(args.run_dir / 'training_protocol.json', provenance)
    best, best_epoch, history = None, 0, []
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        order = np.random.default_rng(args.seed + epoch).permutation(len(train_rows))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            values = torch.from_numpy(np.array(training[indices], dtype=np.float32))
            logits = model(values)
            dense_truth = torch.from_numpy(targets[indices])
            loss = criterion(logits.transpose(1, 2), dense_truth)
            loss = loss + args.window_loss * improve.window_presence_loss(logits, ax, truth[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite lexical-head training loss.')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        selections = [(improve.validation_selection(validation_truth,
                       improve.class_scores(model, validation, ax, pool), 'min_pr'), pool) for pool in (1, 2, 3)]
        metric, pool = max(selections, key=lambda item: selection_key(item[0]))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses)), 'validation_precision': metric['precision'],
                        'validation_recall': metric['recall'], 'validation_f1': metric['f1'],
                        'seconds': time.perf_counter() - started})
        core.write_csv(args.run_dir / 'history.csv', history)
        if best is None or selection_key(metric) > best:
            best, best_epoch = selection_key(metric), epoch
            checkpoint = {'backend': 'wav2vec2_lexical_phone_head', 'target_symbol': '\u0259', 'target_phones': ['ax'],
                'head_state_dict': model.state_dict(), 'head_config': config, 'phone_vocabulary': vocabulary,
                'epoch': epoch, 'threshold': metric['threshold'], 'pool_frames': pool, 'cpu_threads': args.threads,
                'validation_metrics': metric, 'threshold_selection_split': 'validation',
                'settings': {**provenance['settings'], 'feature_context': 'recording', 'acoustic_representation': 'joint'},
                'dataset_sha256': provenance['dataset_sha256'], 'lexical': recipe,
                'backbone_sha256': recipe['main_encoder_sha256'], 'provenance': provenance}
            core.save_torch(args.run_dir / 'best.pt', checkpoint)
        core.save_torch(args.run_dir / 'last.pt', {'head_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(), 'epoch': epoch,
                         'torch_rng_state': torch.get_rng_state(), 'history': history, 'provenance': provenance})
        print(f'Lexical head epoch {epoch:02d}: P={metric["precision"]:.4f}, R={metric["recall"]:.4f}, F1={metric["f1"]:.4f}', flush=True)
        if epoch - best_epoch >= args.patience:
            break
    selected = torch.load(args.run_dir / 'best.pt', weights_only=True)
    core.write_json(args.run_dir / 'training_summary.json', {**provenance, 'best_epoch': best_epoch,
                    'epochs_completed': len(history), 'seconds': time.perf_counter()-started,
                    'validation_metrics': selected['validation_metrics']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('extract')
    p.add_argument('--backbone', type=Path, default=MAIN_ENCODER)
    p.add_argument('--splits', nargs='+', choices=['train', 'validation'], default=['validation', 'train'])
    p.add_argument('--threads', type=int, default=1)
    p.set_defaults(function=extract)
    p = commands.add_parser('train')
    p.add_argument('--backbone', type=Path, default=MAIN_ENCODER)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--task', choices=['phones', 'binary'], default='phones')
    p.add_argument('--temporal', choices=['none', 'gru'], default='none')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--lr', type=float, default=.0005)
    p.add_argument('--window-loss', type=float, default=.5)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=1)
    p.set_defaults(function=train)
    args = parser.parse_args()
    MAIN_ENCODER = args.backbone.resolve()
    args.function(args)

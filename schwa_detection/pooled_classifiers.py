"""CPU window classifiers over frozen acoustic and audio-only lexical statistics."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

import improve
import lexical_train_cpu as lexical
import schwa as core

ROOT = Path(__file__).resolve().parent
POOLED = {'version': 1, 'statistics': ['mean', 'population_std', 'maximum'],
          'phone_probabilities': ['\u0259', '\u028c', '\u025a'],
          'phone_statistics': ['mean', 'maximum', 'top_two_mean'],
          'lexical_interaction': 'AX posterior * nonblank posterior * pronunciation confidence * spelling confidence',
          'interaction_statistics': ['mean', 'maximum', 'top_two_mean'],
          'lexical_duration': 'fraction of 24 focal frames with AX posterior > .5',
          'feature_dtype': 'float32', 'region': 'unchanged 0.5-second focal region'}


def pooled_features(values, phone_ids, embedding_dim=1024, phone_dim=392):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 24 or values.shape[-1] != embedding_dim + phone_dim + 4:
        raise ValueError('Invalid focal acoustic/lexical geometry.')
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite focal features.')
    output = [values.mean(1), values.std(1), values.max(1)]
    logits = values[:, :, embedding_dim:embedding_dim + phone_dim]
    probabilities = np.exp(logits - logits.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    selected = probabilities[:, :, phone_ids]
    output.extend([selected.mean(1), selected.max(1), np.partition(selected, -2, axis=1)[:, -2:].mean(1)])
    prior = values[:, :, -4:]
    weighted = prior.prod(-1)
    output.extend([weighted.mean(1, keepdims=True), weighted.max(1, keepdims=True),
                   np.partition(weighted, -2, axis=1)[:, -2:].mean(1, keepdims=True),
                   (prior[:, :, 0] > .5).mean(1, keepdims=True).astype(np.float32)])
    return np.concatenate(output, axis=-1).astype(np.float32)


def tree_arrays(model, kind):
    def tensor(values):
        return torch.from_numpy(np.array(values, copy=True))
    trees = []
    if kind == 'extra_trees':
        for estimator in model.estimators_:
            tree = estimator.tree_
            counts = tree.value[:, 0, :]
            scores = counts[:, list(model.classes_).index(1)] / counts.sum(-1)
            trees.append({'feature': tensor(tree.feature), 'threshold': tensor(tree.threshold),
                          'left': tensor(tree.children_left), 'right': tensor(tree.children_right),
                          'leaf': tensor(tree.children_left == -1), 'value': tensor(scores)})
        baseline = 0.
    elif kind == 'hist_gradient_boosting':
        for iteration in model._predictors:
            if len(iteration) != 1:
                raise ValueError('Expected binary gradient boosting.')
            nodes = iteration[0].nodes
            if nodes['is_categorical'].any():
                raise ValueError('Only numeric statistics are supported.')
            trees.append({'feature': tensor(nodes['feature_idx']), 'threshold': tensor(nodes['num_threshold']),
                          'left': tensor(nodes['left'].astype(np.int64)), 'right': tensor(nodes['right'].astype(np.int64)),
                          'leaf': tensor(nodes['is_leaf'].astype(bool)), 'value': tensor(nodes['value'])})
        baseline = float(model._baseline_prediction[0, 0])
    else:
        raise ValueError('Unknown tree ensemble.')
    return {'kind': kind, 'trees': trees, 'baseline': baseline, 'input_dim': model.n_features_in_}


def tree_scores(state, features):
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != state['input_dim'] or not np.isfinite(features).all():
        raise ValueError('Invalid tree input features.')
    scores = np.full(len(features), state['baseline'], dtype=np.float64)
    for saved in state['trees']:
        tree = {key: value.numpy() if isinstance(value, torch.Tensor) else np.asarray(value) for key, value in saved.items()}
        nodes = np.zeros(len(features), dtype=np.int64)
        for _ in range(len(tree['leaf'])):
            active = np.flatnonzero(~tree['leaf'][nodes])
            if not len(active):
                break
            current = nodes[active]
            left = features[active, tree['feature'][current]].astype(np.float64) <= tree['threshold'][current]
            nodes[active] = np.where(left, tree['left'][current], tree['right'][current])
        else:
            raise ValueError('Tree traversal did not reach leaves.')
        scores += tree['value'][nodes]
    if state['kind'] == 'extra_trees':
        return scores / len(state['trees'])
    if state['kind'] == 'hist_gradient_boosting':
        return expit(scores)
    raise ValueError('Unknown tree score link.')


def summary_cache(directory, rows, split, recipe, phone_ids):
    if split not in ('train', 'validation') or any(row['split'] != split for row in rows):
        raise ValueError('Development statistics must not consume test regions.')
    identity = {'acoustic_lexical': recipe, 'pooling': POOLED, 'phone_ids': phone_ids,
                'row_ids': [row['sample_id'] for row in rows]}
    path = directory / 'data/pooled_lexical_features' / (split + '_' + core.object_digest(identity) + '.npy')
    if path.exists() and path.with_suffix('.json').exists():
        return lexical.array_cache(path, identity)
    raw = lexical.CombinedFeatures(improve.feature_cache(directory, rows, split),
                                   lexical.lexical_window_cache(directory, rows, split, recipe))
    pieces = []
    for index in range(0, len(rows), 128):
        pieces.append(pooled_features(raw[index:index + 128], phone_ids))
    values = np.concatenate(pieces)
    lexical.save_array(path, values, identity)
    return np.load(path, mmap_mode='r', allow_pickle=False)


def train(args):
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    import sklearn
    from threadpoolctl import threadpool_limits
    if (args.run_dir / 'best.pt').exists():
        raise ValueError('Preserving the learned ensemble.')
    torch.set_num_threads(2)
    directory = improve.activate_dataset('v5')
    improve.BACKBONE = ROOT / 'data/pretrained/phoneme_xlsr53_fp16_linear.pt'
    improve.CONTEXT, improve.REPRESENTATION = 'recording', 'joint'
    recipe = lexical.metadata(improve.BACKBONE)
    vocabulary = core.read_json(ROOT / 'data/pretrained/phoneme_xlsr53/vocab.json')
    phone_ids = [vocabulary[symbol] for symbol in POOLED['phone_probabilities']]
    rows = core.audit_dataset(verify_wavs=False)
    train_rows = [row for row in rows if row['split'] == 'train']
    validation_rows = [row for row in rows if row['split'] == 'validation']
    training = summary_cache(directory, train_rows, 'train', recipe, phone_ids)
    validation = summary_cache(directory, validation_rows, 'validation', recipe, phone_ids)
    labels = np.asarray([int(row['label']) for row in train_rows])
    validation_labels = [int(row['label']) for row in validation_rows]
    if args.kind == 'extra_trees':
        model = ExtraTreesClassifier(n_estimators=256, min_samples_leaf=2, max_features='sqrt',
                                     n_jobs=2, random_state=42)
    else:
        model = HistGradientBoostingClassifier(max_iter=256, learning_rate=.05, max_leaf_nodes=31,
                                               min_samples_leaf=20, l2_regularization=1.,
                                               early_stopping=False, random_state=42)
    provenance = {'dataset': 'v5', 'dataset_sha256': core.digest(directory / 'data/manifests/dataset.csv'),
                  'lexical': recipe, 'pooled_recipe': POOLED, 'train_regions': len(train_rows),
                  'validation_regions': len(validation_rows), 'kind': args.kind, 'cpu_threads': 2,
                  'sklearn_version': sklearn.__version__, 'parameters': model.get_params(),
                  'test_used_for_selection': False, 'initialization': 'Fresh classifier; pinned public encoders only.',
                  'code_sha256': {name: core.digest(ROOT / name) for name in
                                 ['pooled_classifiers.py', 'lexical_train_cpu.py', 'lexical_context.py', 'frozen_phoneme.py', 'improve.py']}}
    core.write_json(args.run_dir / 'training_protocol.json', provenance)
    started = time.perf_counter()
    print('Fitting', args.kind, len(train_rows), 'train regions,', training.shape[1], 'statistics', flush=True)
    with threadpool_limits(limits=2):
        model.fit(training, labels)
        scores = model.predict_proba(validation)[:, list(model.classes_).index(1)]
        saved = tree_arrays(model, args.kind)
        portable = tree_scores(saved, validation)
    difference = float(np.abs(portable-scores).max())
    if difference > 1e-12:
        raise ValueError('Portable ensemble differs from the fitted classifier.')
    metric = improve.validation_selection(validation_labels, scores, 'min_pr')
    checkpoint = {'backend': 'wav2vec2_lexical_window_trees', 'target_symbol': '\u0259', 'target_phones': ['ax'],
                  'tree_state': saved, 'pooled_recipe': POOLED, 'phone_ids': phone_ids, 'lexical': recipe,
                  'threshold': metric['threshold'], 'threshold_selection_split': 'validation',
                  'cpu_threads': 2, 'pool_frames': 1, 'epoch': 1,
                  'settings': {'feature_context': 'recording', 'acoustic_representation': 'joint', 'task': 'binary_window_statistics'},
                  'dataset_sha256': provenance['dataset_sha256'], 'backbone_sha256': recipe['main_encoder_sha256'],
                  'validation_metrics': metric, 'provenance': provenance}
    core.save_torch(args.run_dir / 'best.pt', checkpoint)
    core.write_csv(args.run_dir / 'validation_scores.csv', [{'sample_id': row['sample_id'], 'score': float(score),
                  'label': row['label']} for row, score in zip(validation_rows, scores)])
    report = {**provenance, 'best_epoch': 1, 'epochs_completed': 1, 'seconds': time.perf_counter()-started,
              'validation_metrics': metric, 'portable_score_maximum_difference': difference}
    core.write_json(args.run_dir / 'training_summary.json', report)
    print(args.kind, metric, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['extra_trees', 'hist_gradient_boosting'], required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    train(parser.parse_args())

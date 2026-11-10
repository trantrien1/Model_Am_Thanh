"""Prepare manually annotated L2-ARCTIC schwa windows, without speaker/prompt leakage."""
import argparse
import io
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

import schwa as core

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/l2_manual'
DESTINATION = ROOT / 'v3'
SEED = 20261011
REVISION = 'b7261545bc4e4a02910e9c85c659eb866e00db7a'
SOURCE_SHA = {
    'train': 'eba42243b45ca79eed72a853d6b69db20639e0ff78db4fb0ef8ba7f5b17c964c',
    'validation': 'd32d420a56b6631ce19a9172a8ee9678582fb4536714add3253f380cb3dd93ec',
    'test': '4fc5580acb1318793e9a929ce71279058bc2ef33dba8288517c423acb94b23dd',
}
PHONES = set('aa ae ah ao aw ax axr ay b ch d dh dx eh el em en eng er ey f g hh hv ih ix iy jh k l m n ng nx ow oy p q r s sh t th uh uw ux v w y z zh'.split())


def perceived_phone(event):
    """Stress is read BEFORE reduction: AH0=schwa; AH1/AH2=/ʌ/."""
    raw = (event.get('perceived_raw') or '').strip().upper()
    if event.get('error_type') == 'deletion' or raw in {'', 'SIL', 'SP', 'PAU', 'SSIL'}:
        return 'pau'
    if raw in {'AH0', 'AH0*', 'AX', 'AX0', 'AX*', 'AX0*'}:
        return 'ax'
    # * denotes an explicitly annotated non-native variant, not an exclusion.
    normalized = re.sub(r'[012]$', '', raw.replace('*', '')).lower()
    return normalized if normalized in PHONES else 'unk'


def convert_events(events, duration, sample_rounding=False):
    phones, previous = [], 0.
    for event in events:
        start, end = float(event['start']), float(event['end'])
        if not np.isfinite([start, end]).all() or start < -1e-6 or end < start or end > duration + .025:
            raise ValueError('Invalid manual phone boundary.')
        if start < previous - .001:
            raise ValueError('Overlapping manual intervals.')
        start, end = max(previous, start), min(end, duration)
        if start > previous + 1e-6:
            phones.append({'start': previous, 'end': start, 'phone': 'unk'})
        if end > start:
            phones.append({'start': start, 'end': end, 'phone': perceived_phone(event)})
        previous = max(previous, end)
    if previous < duration:
        # Resampling 44.1 kHz annotations to 16 kHz rounds the file length by < 1 sample.
        # The final interval is explicitly silence: this fraction is not unannotated speech.
        if sample_rounding and phones and phones[-1]['phone'] == 'pau' and duration - previous <= 1 / 16000 + 1e-9:
            phones[-1]['end'] = duration
        else:
            phones.append({'start': previous, 'end': duration, 'phone': 'unk'})
    if not phones:
        raise ValueError('Empty manual annotations.')
    return phones


def source_rows(split, audio=True):
    path = SOURCE / (split + '-00000-of-00001.parquet')
    if not path.exists():
        core.fetch('https://huggingface.co/datasets/chikingsley/l2-arctic-manual-v5.0-16k/resolve/'
                   + REVISION + '/data/' + path.name, path, timeout=60)
    if core.digest(path) != SOURCE_SHA[split]:
        raise ValueError('Pinned L2 source changed: ' + split)
    columns = ['utterance_id', 'speaker_id', 'native_language', 'transcript', 'manual_events']
    if audio:
        columns.append('audio')
    for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=columns):
        yield from batch.to_pylist()


def prepare(args):
    core.ROOT = DESTINATION
    # A frozen dataset is independently usable after older dataset versions are
    # removed. make_dataset checks its saved settings and source fingerprints.
    if (DESTINATION / 'data/manifests/dataset.csv').exists():
        selection = core.read_json(DESTINATION / 'data/manifests/source_selection.json')
        if selection['source_revision'] != REVISION or selection['source_sha256'] != SOURCE_SHA:
            raise ValueError('Frozen dataset uses a different manual source.')
        if args and args.version == 'v5' and (selection['seed'] != 20261012 or selection['window_seed'] != SEED):
            raise ValueError('Frozen V5 partition/window seed differs.')
        core.make_dataset(argparse.Namespace(seed=SEED))
        return
    prompt_ids, speakers = set(), {}
    for split in core.SPLITS:
        ids = set()
        for row in source_rows(split, audio=False):
            prompt_ids.add(row['utterance_id'])
            ids.add(row['speaker_id'])
        speakers[split] = sorted(ids)
    prompts = sorted(prompt_ids)
    random.Random(SEED).shuffle(prompts)
    counts = [int(len(prompts) * .6), int(len(prompts) * .2)]
    groups = {'train': prompts[:counts[0]], 'validation': prompts[counts[0]:sum(counts)],
              'test': prompts[sum(counts):]}
    previous_records = {}
    if args and args.version == 'v5':
        previous = core.read_json(ROOT / 'v4/data/manifests/source_selection.json')
        candidates = sorted(previous['prompt_groups']['train'])
        random.Random(20261012).shuffle(candidates)
        groups['test'] = sorted(candidates[:len(previous['prompt_groups']['test'])])
        groups['validation'] = previous['prompt_groups']['validation']
        groups['train'] = sorted(set(prompts) - set(groups['test']) - set(groups['validation']))
        if set(groups['test']) & set(previous['prompt_groups']['test']):
            raise ValueError('New held-out prompt groups overlap retired test.')
        previous_records = {r['recording_id']: r for r in core.read_csv(ROOT / 'v4/data/manifests/recordings.csv')}
    selection = {'seed': SEED, 'speakers': speakers, 'prompt_groups': groups,
                 'source_repo': 'chikingsley/l2-arctic-manual-v5.0-16k', 'source_revision': REVISION,
                 'source_sha256': SOURCE_SHA, 'license': 'CC BY-NC 4.0',
                 'annotation_origin': 'manual perceived phonemes; substitutions/additions/deletions preserved',
                 'target_mapping': {'AH0': 'ax', 'AX': 'ax', 'AX0': 'ax', 'AH1/AH2': 'ah', 'ER/AXR': 'excluded from target'},
                 'unknown_policy': 'Exclude any generated window overlapping an uncertain/unannotated interval.',
                 'partition': 'Published 12/6/6 speakers plus globally disjoint 60/20/20 percent prompt groups.'}
    if args and args.version in ('v4', 'v5'):
        selection['sample_rounding_policy'] = 'Extend final annotated silence by at most 1/16000 second; other unknown intervals unchanged.'
    if args and args.version == 'v5':
        selection.update(seed=20261012, window_seed=SEED,
                         partition='Same published 12/6/6 speakers; 180/60/60 disjoint prompts. New test prompts seeded from retired V4 train prompts, never from retired test.',
                         prior_test_manifest_sha256=core.digest(ROOT / 'v4/data/manifests/test.csv'),
                         test_independence='Previously held-out speakers remain excluded from train/validation; new test utterances were not part of V4. Speakers were evaluated previously.',
                         initialization_policy='New heads from random initialization and public pretrained encoder only; no V4-trained head or finetuned encoder initialization, because V4 train included new test prompts.')
    selection_path = DESTINATION / 'data/manifests/source_selection.json'
    if selection_path.exists() and core.read_json(selection_path) != selection:
        raise ValueError('Preserving frozen L2 selection.')
    core.write_json(selection_path, selection)
    if (DESTINATION / 'data/manifests/dataset.csv').exists():
        core.make_dataset(argparse.Namespace(seed=SEED))
        return
    records, quarantined, mapping_counts = [], [], Counter()
    for split in core.SPLITS:
        allowed = set(groups[split])
        for row in source_rows(split):
            if row['utterance_id'] not in allowed:
                continue
            speaker, prompt = row['speaker_id'], row['utterance_id']
            identity = speaker + '_' + prompt
            audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32', always_2d=True)
            if rate != 16000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
                raise ValueError('Expected valid 16 kHz mono L2 source.')
            try:
                phones = convert_events(row['manual_events'], len(audio) / rate,
                                        sample_rounding=bool(args and args.version in ('v4', 'v5')))
            except ValueError as error:
                quarantined.append({'recording_id': identity, 'reason': str(error)})
                continue
            mapping_counts.update(p['phone'] for p in phones)
            base = DESTINATION / 'data/raw' / speaker / prompt
            base.parent.mkdir(parents=True, exist_ok=True)
            old = previous_records.get(identity)
            if old:
                expected_lab = '#\n' + ''.join(f"{p['end']:.9f} 26 {p['phone']}\n" for p in phones)
                if (ROOT / 'v4' / old['lab_path']).read_text(encoding='utf-8') != expected_lab:
                    raise ValueError('Shared V4 annotation differs from the pinned manual source.')
                for kind, suffix in [('audio', '.wav'), ('lab', '.lab'), ('manual', '.json')]:
                    source_path = ROOT / 'v4' / old[kind + '_path']
                    if core.digest(source_path) != old[kind + '_sha256']:
                        raise ValueError('Shared V4 source changed.')
                    destination = base.with_suffix(suffix)
                    if destination.exists():
                        if core.digest(destination) != old[kind + '_sha256']:
                            raise ValueError('Preserving differing V5 source.')
                    else:
                        destination.hardlink_to(source_path)
            else:
                sf.write(base.with_suffix('.wav'), audio[:, 0], rate, subtype='PCM_16')
                base.with_suffix('.lab').write_text('#\n' + ''.join(f"{p['end']:.9f} 26 {p['phone']}\n" for p in phones), encoding='utf-8')
                core.write_json(base.with_suffix('.json'), {k: v for k, v in row.items() if k != 'audio'})
            rec = {'split': split, 'speaker_id': speaker, 'prompt_id': prompt, 'recording_id': identity,
                   'native_language': row['native_language'], 'annotation_origin': 'human perceived phonemes',
                   'source_parquet_sha256': SOURCE_SHA[split]}
            for kind, suffix in [('audio', '.wav'), ('lab', '.lab'), ('manual', '.json')]:
                path = base.with_suffix(suffix)
                rec[kind + '_path'] = path.relative_to(DESTINATION).as_posix()
                rec[kind + '_sha256'] = core.digest(path)
            records.append(rec)
            if len(records) % 50 == 0:
                print('Decoded manual recordings:', len(records), flush=True)
        print('Manual sources:', split, sum(r['split'] == split for r in records), flush=True)
    records.sort(key=lambda r: r['recording_id'])
    core.write_csv(DESTINATION / 'data/manifests/recordings.csv', records)
    core.write_json(DESTINATION / 'reports/manual_source_qc.json', {'quarantined': quarantined, 'mapping_counts': dict(mapping_counts),
                    'source_revision': REVISION, 'annotation_source': selection['annotation_origin'], 'label_code_sha256': core.digest(__file__)})
    core.make_dataset(argparse.Namespace(seed=SEED))


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=['v3', 'v4', 'v5'], default='v3')
    args = parser.parse_args()
    DESTINATION = ROOT / args.version
    prepare(args)

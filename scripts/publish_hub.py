"""Upload artifact shards without staging a second copy of the whole project."""
import argparse
import collections
import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from hub_layout import ROOT, SKIP_DIRS, PROFILES, artifact_group


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory():
    groups = collections.defaultdict(list)
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            path = Path(base) / name
            if path.is_symlink() or name.endswith(('.part', '.log', '.pyc')):
                continue
            relative = path.relative_to(ROOT).as_posix()
            group = artifact_group(relative)
            if group:
                stat = path.stat()
                groups[group].append((relative, stat.st_size, stat.st_mtime_ns))
    return groups


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', help='Defaults to authenticated HF account')
    parser.add_argument('--shard-mib', type=int, default=384)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--profile', choices=['all', 'inference'], default='all')
    args = parser.parse_args()
    groups = inventory()
    if args.profile == 'inference':
        groups = {key: items for key, items in groups.items() if key[1] in PROFILES['inference']}
    print(json.dumps({group: {'files': len(items), 'bytes': sum(x[1] for x in items)}
                      for (_, group), items in sorted(groups.items())}, indent=2), flush=True)
    if args.dry_run:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    namespace = args.namespace or api.whoami()['name']
    repos = {'model': namespace + '/Model_Am_Thanh-models', 'dataset': namespace + '/Model_Am_Thanh-data'}
    config = {'schema_version': 1, 'status': 'complete' if args.profile == 'all' else 'inference-ready', 'repositories': {}}
    for kind, repo in repos.items():
        if not any(key[0] == kind for key in groups):
            continue
        api.create_repo(repo, repo_type=kind, exist_ok=True, private=False)
        manifest = {'schema_version': 1, 'archives': []}
        state_path = ROOT / '.hf-publish' / (kind + '.json')
        state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
        remote_files = set(api.list_repo_files(repo, repo_type=kind))
        for (group_kind, group), items in sorted(groups.items()):
            if group_kind != kind:
                continue
            batches, batch, size = [], [], 0
            for item in items:
                if batch and size + item[1] > args.shard_mib * 1024**2:
                    batches.append(batch)
                    batch, size = [], 0
                batch.append(item)
                size += item[1]
            if batch:
                batches.append(batch)
            for index, batch in enumerate(batches):
                filename = f'archives/{group}-{index:04d}.tar.gz'
                source_key = hashlib.sha256(json.dumps(batch).encode()).hexdigest()
                previous = state.get(filename)
                if previous and previous['source_key'] == source_key and filename in remote_files:
                    entry = previous['entry']
                    print('Already uploaded:', filename, flush=True)
                else:
                    # The OS temporary directory is used for just one shard at a time.
                    with tempfile.TemporaryDirectory(prefix='model-am-thanh-') as temporary:
                        archive = Path(temporary) / 'shard.tar.gz'
                        with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0, filename='') as compressed, tarfile.open(fileobj=compressed, mode='w|') as tar:
                            for relative, expected_size, expected_mtime in batch:
                                path = ROOT / relative
                                before = path.stat()
                                if (before.st_size, before.st_mtime_ns) != (expected_size, expected_mtime):
                                    raise RuntimeError('Source changed during publication: ' + relative)
                                tar.add(path, arcname=relative, recursive=False)
                                after = path.stat()
                                if (after.st_size, after.st_mtime_ns) != (expected_size, expected_mtime):
                                    raise RuntimeError('Source changed during publication: ' + relative)
                        entry = {'path': filename, 'group': group, 'sha256': digest(archive),
                                 'bytes': archive.stat().st_size, 'files': len(batch),
                                 'uncompressed_bytes': sum(item[1] for item in batch)}
                        print(f'Uploading {repo}/{filename}: {entry["bytes"] / 1024**2:.1f} MiB', flush=True)
                        api.upload_file(path_or_fileobj=str(archive), path_in_repo=filename,
                                        repo_id=repo, repo_type=kind, commit_message='Add ' + filename)
                    state[filename] = {'source_key': source_key, 'entry': entry}
                    save(state_path, state)
                manifest['archives'].append(entry)
        card = '''---
language:
- en
tags:
- audio
- schwa
- keyword-spotting
---
# Model_Am_Thanh artifacts

Code and Vietnamese reproduction instructions: https://github.com/trantrien1/Model_Am_Thanh

Archives preserve paths relative to that Git repository. Use scripts/restore_hub.py
from the code repository; manifest.json records SHA-256 and sizes for every shard.

Sources: ESC-50 (CC BY-NC; ESC-10 exceptions per original metadata), Google Speech
Commands v0.02 (CC BY 4.0), CMU ARCTIC (see bundled source notices), L2-ARCTIC v5.0
(CC BY-NC 4.0; 16 kHz mirror chikingsley/l2-arctic-manual-v5.0-16k),
facebook/wav2vec2-xlsr-53-espeak-cv-ft (Apache 2.0 weights), plus the original
WavLM, wav2vec2 and Whisper notices included in the archived metadata and code README.
Derived audio retains source attribution and noncommercial restrictions where applicable.
This mixed collection does not grant a new blanket license to its contents.

Dataset repositories contain audio, labels, manifests, feature caches and experiment reports.
Model repositories contain pretrained weights, training checkpoints, registered source snapshots
and deployment packages. These are project archives, not a Transformers from_pretrained model.
The deployed Schwa package uses the previously verified V4 model; later V5 head checkpoints
are experimental, require their encoders and have not met the 88% precision/recall objective.
'''
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo='README.md', repo_id=repo,
                        repo_type=kind, commit_message='Document artifact provenance and restore procedure')
        result = api.upload_file(path_or_fileobj=(json.dumps(manifest, indent=2) + '\n').encode(),
                                 path_in_repo='manifest.json', repo_id=repo, repo_type=kind,
                                 commit_message='Finalize verified artifact manifest')
        config['repositories'][kind] = {'repo_id': repo, 'revision': result.oid,
                                       'archives': len(manifest['archives']),
                                       'bytes': sum(x['bytes'] for x in manifest['archives'])}
    save(ROOT / 'hub-artifacts.json', config)
    print('Publication complete; hub-artifacts.json pins both HF revisions.', flush=True)


if __name__ == '__main__':
    main()

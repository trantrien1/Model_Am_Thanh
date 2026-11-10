"""Restore selected HF artifacts to their original paths after git clone."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from hub_layout import ROOT, PROFILES, artifact_group


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack(archive, destination, group):
    destination = destination.resolve()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            relative = PurePosixPath(member.name)
            if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                    or '\\' in member.name or ':' in member.name):
                raise ValueError('Unsafe archive member: ' + member.name)
            classification = artifact_group(member.name)
            if not classification or classification[1] != group:
                raise ValueError('Unexpected artifact path: ' + member.name)
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError('Archive member escapes destination')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, tempfile.NamedTemporaryFile(
                    dir=target.parent, prefix='.hf-restore-', delete=False) as output:
                temporary = Path(output.name)
                try:
                    shutil.copyfileobj(source, output)
                except BaseException:
                    output.close()
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                if target.exists():
                    if digest(target) != digest(temporary):
                        raise FileExistsError(f'Local file differs: {target}. Move it aside before restoring.')
                else:
                    temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=[*PROFILES, 'all'], default='inference')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, default=ROOT / 'hub-artifacts.json')
    parser.add_argument('--list', action='store_true', help='List selected archives without downloading them')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    if config.get('status') != 'complete':
        raise SystemExit('HF publication is not complete; hub-artifacts.json must contain pinned revisions.')
    from huggingface_hub import hf_hub_download
    selected_groups = None if args.profile == 'all' else PROFILES[args.profile]
    for kind, repository in config['repositories'].items():
        with tempfile.TemporaryDirectory(prefix='model-am-thanh-download-') as temp:
            common = dict(repo_id=repository['repo_id'], repo_type=kind,
                          revision=repository['revision'], local_dir=temp)
            manifest_path = hf_hub_download(filename='manifest.json', **common)
            manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
            for entry in manifest['archives']:
                if selected_groups is not None and entry['group'] not in selected_groups:
                    continue
                print(f'{entry["group"]}: {entry["path"]} ({entry["bytes"] / 1024**2:.1f} MiB)', flush=True)
                if args.list:
                    continue
                archive = Path(hf_hub_download(filename=entry['path'], **common))
                if archive.stat().st_size != entry['bytes'] or digest(archive) != entry['sha256']:
                    raise ValueError('Archive failed integrity check: ' + entry['path'])
                unpack(archive, args.destination, entry['group'])
                archive.unlink()
    print('Listing complete.' if args.list else 'Restore complete.', flush=True)


if __name__ == '__main__':
    main()

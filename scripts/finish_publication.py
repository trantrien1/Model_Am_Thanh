"""Resume HF upload, verify a fresh clone, then push the pinned artifact revisions."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from hub_layout import ROOT


def status(stage, **details):
    value = {'stage': stage, 'updated_utc': datetime.now(timezone.utc).isoformat(), **details}
    path = ROOT / '.hf-publish' / 'status.json'
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
    print(json.dumps(value), flush=True)


def run(command, cwd=ROOT, **kwargs):
    subprocess.run([str(x) for x in command], cwd=cwd, check=True, **kwargs)


def main():
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    config = json.loads((ROOT / 'hub-artifacts.json').read_text(encoding='utf-8'))
    if config.get('status') == 'uploading':
        status('publishing_inference_first')
        run([sys.executable, ROOT / 'scripts/publish_hub.py', '--namespace', 'anonymos111', '--profile', 'inference'])
        run(['git', 'add', '--', 'hub-artifacts.json'])
        run(['git', 'commit', '-m', 'Enable downloading the published inference checkpoints', '--', 'hub-artifacts.json'])
        run(['git', '-c', 'credential.interactive=never', 'push', 'origin', 'main'])
    for attempt in range(1, 13):
        try:
            status('uploading', attempt=attempt)
            run([sys.executable, ROOT / 'scripts/publish_hub.py', '--namespace', 'anonymos111'])
            break
        except Exception:
            traceback.print_exc()
            if attempt == 12:
                status('failed', reason='Upload failed after 12 resumable attempts; rerun this script to continue.')
                raise
            status('retrying', attempt=attempt, retry_in_seconds=60)
            time.sleep(60)
    status('verifying_fresh_clone')
    runtime = ROOT / '.venv/Scripts/python.exe'
    if not runtime.exists():
        runtime = ROOT / '.venv/bin/python'
    if not runtime.exists():
        runtime = Path(sys.executable)
    with tempfile.TemporaryDirectory(prefix='model-am-thanh-clone-') as temporary:
        clone = Path(temporary) / 'project'
        run(['git', 'clone', '--depth', '1', 'https://github.com/trantrien1/Model_Am_Thanh.git', clone])
        shutil.copyfile(ROOT / 'hub-artifacts.json', clone / 'hub-artifacts.json')
        run([sys.executable, clone / 'scripts/restore_hub.py', '--profile', 'inference'], cwd=clone)
        run([runtime, clone / 'schwa_detection/test/schwa.py', 'predict', '--inspect'], cwd=clone)
        run([runtime, clone / 'keyword_spotting/test/inference.py', '--inspect'], cwd=clone)
        run([runtime, clone / 'keyword_spotting/test/inference.py', '--audio',
             clone / 'keyword_spotting/test/examples/test_positive.wav',
             clone / 'keyword_spotting/test/examples/test_negative.wav'], cwd=clone)
    status('pushing_verified_revisions')
    run(['git', 'add', '--', 'hub-artifacts.json'])
    changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', 'hub-artifacts.json'], cwd=ROOT)
    if changed.returncode == 1:
        run(['git', 'commit', '-m', 'Pin published HF artifacts verified from a fresh clone', '--', 'hub-artifacts.json'])
    elif changed.returncode != 0:
        raise RuntimeError('Cannot inspect staged artifact configuration')
    run(['git', '-c', 'credential.interactive=never', 'push', 'origin', 'main'])
    status('complete', github='https://github.com/trantrien1/Model_Am_Thanh',
           dataset='https://huggingface.co/datasets/anonymos111/Model_Am_Thanh-data',
           models='https://huggingface.co/anonymos111/Model_Am_Thanh-models')


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        status('failed', reason=str(error))
        raise

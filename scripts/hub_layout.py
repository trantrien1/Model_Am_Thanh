"""One shared artifact layout for publishing and restoring the workspace."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.pytest_cache', '.hf-publish', '.hf-download'}
PROFILES = {
    'schwa': {'schwa-inference'},
    'keyword': {'keyword-inference'},
    'inference': {'schwa-inference', 'keyword-inference'},
}


def artifact_group(path):
    p = Path(path).as_posix()
    for project in ('keyword_spotting', 'schwa_detection'):
        short = 'keyword' if project == 'keyword_spotting' else 'schwa'
        if p.startswith(project + '/runs/'):
            return 'model', short + '-runs'
        if p.startswith(project + '/reports/'):
            return 'dataset', short + '-reports'
        if p.startswith(project + '/data/'):
            if p in {'keyword_spotting/data/make_dataset.py', 'keyword_spotting/data/real_recordings/README.md'}:
                return None
            if p.startswith('schwa_detection/data/pretrained/'):
                return 'model', 'schwa-pretrained'
            return 'dataset', short + '-data'
        if p.startswith(project + '/test/model/') or p.startswith(project + '/test/examples/'):
            return 'model', short + '-inference'
        if p == project + '/test/best.pt':
            return 'model', short + '-inference'
        if p in {project + '/test.zip', project + '/test_improved.zip'}:
            return 'model', short + '-packages'
    if p.startswith('schwa_detection/v5/'):
        return 'dataset', 'schwa-v5'
    for name in ('training_all_negatives', 'training_native', 'training_suitcase'):
        if p.startswith('schwa_detection/' + name + '/'):
            return 'dataset', 'schwa-' + name.replace('_', '-')
    if p == 'schwa_detection/test_improved/best.pt':
        return 'model', 'schwa-legacy-inference'
    if Path(p).suffix.lower() in {'.wav', '.flac', '.mp3', '.npy', '.npz', '.mmap', '.parquet'}:
        return 'dataset', 'other-data'
    if Path(p).suffix.lower() in {'.pt', '.pth', '.bin', '.safetensors', '.zip'}:
        return 'model', 'other-models'
    return None

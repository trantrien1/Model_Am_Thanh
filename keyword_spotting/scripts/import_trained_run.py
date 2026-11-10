"""Import an existing results ZIP without replacing project code or retraining."""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import read_json, sha256_file, timestamp, write_json
from src.download import safe_extract, validate_archive


def copy_preserving(source, destination):
    destination = Path(destination)
    for path in sorted(Path(source).rglob("*")):
        if not path.is_file():
            continue
        output = destination / path.relative_to(source)
        if output.exists():
            if sha256_file(path) != sha256_file(output):
                raise ValueError(f"Existing file differs; preserving it: {output}. Choose a different destination.")
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, output)


def import_run(archive, project=ROOT, run_name="baseline"):
    archive, project = Path(archive).resolve(), Path(project).resolve()
    if not run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_name):
        raise ValueError("Invalid run_name.")
    digest = sha256_file(archive)
    staging = project / "data/imports/extracted" / digest
    marker = staging / ".extracted.json"
    if not marker.exists():
        validate_archive(archive)
        if staging.exists():
            raise ValueError("Import staging directory already exists without an extraction marker.")
        safe_extract(archive, staging)
        write_json(marker, {"archive_sha256": digest, "created_at": timestamp()})
    extracted = staging / "keyword_spotting"
    if not (extracted / "runs" / run_name / "best.pt").is_file():
        raise FileNotFoundError(f"No trained checkpoint for {run_name} in the results ZIP.")
    copy_preserving(extracted / "runs" / run_name, project / "runs" / run_name)
    if (extracted / "data/manifests").is_dir():
        copy_preserving(extracted / "data/manifests", project / "data/manifests")
    if (extracted / "reports").is_dir():
        copy_preserving(extracted / "reports", project / "reports" / "trained_run")
    report = {"created_at": timestamp(), "archive": str(archive), "archive_sha256": digest,
              "run_dir": str(project / "runs" / run_name), "source_snapshot": str(extracted / "src"),
              "project_code_replaced": False, "training_executed": False,
              "dataset_audio_in_archive": len(list((extracted / "data/generated").rglob("*.wav"))) if (extracted / "data/generated").exists() else 0}
    write_json(project / "reports" / "trained_run_import.json", report)
    print(f"Imported trained run: {report['run_dir']}. Dataset WAVs in archive: {report['dataset_audio_in_archive']}.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--project-dir", default=str(ROOT))
    parser.add_argument("--run-name", default="baseline")
    args = parser.parse_args()
    import_run(args.archive, args.project_dir, args.run_name)


if __name__ == "__main__":
    main()

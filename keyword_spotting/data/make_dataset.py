r"""Make the same WAV dataset as keyword_spotting_colab.ipynb, using CPU.

From keyword_spotting:
    ..\.venv\Scripts\python.exe data/make_dataset.py
    ..\.venv\Scripts\python.exe data/make_dataset.py --output-dir data/new_dataset
    ..\.venv\Scripts\python.exe data/make_dataset.py --dry-run

Fresh data: download/reuse sources, source QC, mix, audit, supplemental test.
Existing data: restore missing WAVs from the frozen manifest, then audit.
"""

import argparse
import csv
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/default.yaml")
    parser.add_argument("--output-dir", type=Path,
                        help="New data directory: manifests/, generated/, features/, reports/.")
    parser.add_argument("--esc-dir", type=Path, help="Extracted ESC-50 directory.")
    parser.add_argument("--speech-dir", type=Path, help="Extracted Speech Commands v0.02 directory.")
    parser.add_argument("--workers", type=int, default=2, help="Source QC threads; default: 2.")
    parser.add_argument("--no-supplemental", action="store_true", help="Only main train/validation/test.")
    parser.add_argument("--dry-run", action="store_true", help="Print steps without writing or downloading.")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def make_plan(cfg, args):
    from src.common import directory, project_path, read_json

    manifests = directory(cfg, "manifests")
    main_manifest = manifests / "dataset.csv"
    existing = main_manifest.is_file()
    if existing:
        metadata = read_json(manifests / "dataset_metadata.json")
        for key in ("audio", "mix", "seed", "target_word"):
            if cfg[key] != metadata["config"][key]:
                raise ValueError(f"Existing dataset differs in {key}. Use --output-dir data/new_dataset.")
        source_metadata = read_json(manifests / "sources_metadata.json")
        for key in ("esc_dir", "speech_dir"):
            frozen_path = project_path(cfg, source_metadata["config"]["paths"][key])
            if directory(cfg, key) != frozen_path:
                raise ValueError(f"Existing dataset uses a different {key}. Use a new --output-dir.")

    esc = directory(cfg, "esc_dir")
    speech = directory(cfg, "speech_dir")
    esc_ready = (esc / "audio").is_dir() and (esc / "meta/esc50.csv").is_file()
    speech_ready = all((speech / name).is_file() for name in ("validation_list.txt", "testing_list.txt"))
    if args.esc_dir and not esc_ready:
        raise ValueError(f"ESC directory needs audio/ and meta/esc50.csv: {esc}")
    if args.speech_dir and not speech_ready:
        raise ValueError(f"Speech Commands directory needs both split lists: {speech}")

    plan = []
    if existing:
        restore_args = []
        if esc_ready and speech_ready:
            restore_args.append("--skip-download")
        if args.no_supplemental:
            restore_args.append("--no-supplemental")
        plan.append(("script", "scripts/restore_dataset.py", restore_args))
    else:
        source_args = []
        if esc_ready:
            source_args.extend(("--esc-dir", str(esc)))
        if speech_ready:
            source_args.extend(("--speech-dir", str(speech)))
        plan.extend([
            ("module", "src.download", source_args),
            ("module", "src.prepare_sources", ["--workers", str(args.workers)]),
            ("module", "src.build_dataset", []),
        ])
    plan.append(("module", "src.audit_dataset", []))
    if not args.no_supplemental:
        if not existing or not (manifests / "supplemental.csv").is_file():
            plan.append(("module", "src.build_dataset", ["--supplemental"]))
        plan.append(("module", "src.audit_dataset", ["--supplemental"]))
    return plan, existing


def summarize(cfg):
    from src.common import directory, write_csv

    manifests = directory(cfg, "manifests")
    with (manifests / "dataset.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    counts = Counter((row["split"], row["label"]) for row in rows)
    print("\nSplit         WAVs  Positive  Negative", flush=True)
    for split in ("train", "validation", "test"):
        subset = [row for row in rows if row["split"] == split]
        write_csv(manifests / f"{split}.csv", subset, fields)
        print(f"{split:<12} {len(subset):>5} {counts[split, '1']:>9} {counts[split, '0']:>9}")
    print("WAVs:", directory(cfg, "generated"))
    print("Labels:", manifests)
    print("Audit reports:", directory(cfg, "reports"))


def main(argv=None):
    args = parse_args(argv)
    sys.path.insert(0, str(PROJECT))
    import yaml
    from src.common import directory, load_config, public_config

    cfg = load_config(args.config)
    root = Path(cfg["_root"])
    for key in ("esc_dir", "speech_dir"):
        override = getattr(args, key)
        if override is not None:
            cfg["paths"][key] = str(override.resolve())
    if args.output_dir is not None:
        output = args.output_dir.resolve()
        if root != output and root not in output.parents:
            raise ValueError("--output-dir must be inside the project containing the config.")
        for key, folder in (("manifests", "manifests"), ("generated", "generated"),
                            ("features", "features"), ("reports", "reports")):
            cfg["paths"][key] = str(output / folder)

    plan, existing = make_plan(cfg, args)
    print("Mode:", "reuse frozen manifests and restore missing WAVs" if existing else "build like Colab", flush=True)
    print("Target:", cfg["target_word"], "| Seed:", cfg["seed"], flush=True)
    print("Audio:", cfg["audio"]["sample_rate"], "Hz |", cfg["audio"]["duration_seconds"], "seconds", flush=True)
    print("Output:", directory(cfg, "generated"), flush=True)
    if args.dry_run:
        for kind, name, extra in plan:
            command = [sys.executable, "-B", "-u"]
            command += ["-m", name] if kind == "module" else [str(PROJECT / name)]
            print(shlex.join(command + ["--config", "<runtime config>", *extra]))
        return 0

    # Keep the config in its original configs/ folder: load_config derives the
    # project root from that folder. Close the handle before subprocesses on Windows.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="make_dataset_",
                                     dir=args.config.resolve().parent, encoding="utf-8", delete=False) as handle:
        yaml.safe_dump(public_config(cfg), handle, sort_keys=False)
        runtime_config = Path(handle.name)
    try:
        for index, (kind, name, extra) in enumerate(plan, 1):
            command = [sys.executable, "-B", "-u"]
            command += ["-m", name] if kind == "module" else [str(PROJECT / name)]
            command += ["--config", str(runtime_config), *extra]
            print(f"\n[{index}/{len(plan)}] {name}", flush=True)
            subprocess.run(command, cwd=PROJECT, check=True)
        summarize(cfg)
        if not args.no_supplemental:
            with (directory(cfg, "manifests") / "supplemental.csv").open(encoding="utf-8", newline="") as handle:
                extra_count = sum(1 for _ in csv.DictReader(handle))
            print("Supplemental test:", extra_count, "WAVs (separate from main test)")
        print("Dataset ready.", flush=True)
    finally:
        runtime_config.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Dataset failed: {error}", file=sys.stderr)
        raise SystemExit(1)

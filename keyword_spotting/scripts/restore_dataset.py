"""Restore exact mixed WAVs from a trained manifest, without running source QC or training again."""
import argparse
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.audio import fixed_length, load_audio, pcm_hash
from src.common import (directory, load_config, project_path, read_csv, read_json,
                        sha256_file, timestamp, write_json)
from src.download import extract_verified, fetch_archive


def restore_source(cfg, name):
    manifests = directory(cfg, "manifests")
    recorded = read_json(manifests / "download_provenance.json")[name]
    if not recorded.get("sha256") or not recorded.get("url"):
        raise ValueError(f"No original archive checksum/URL for {name}; provide the original extracted sources instead.")
    archive = Path(cfg["_root"]) / "data/raw/archives" / ("esc50.zip" if name == "esc" else "speech_commands_v0.02.tar.gz")
    metadata = fetch_archive(recorded["url"], archive, cfg["download"]["timeout_seconds"], cfg["download"]["retries"])
    if metadata["sha256"] != recorded["sha256"]:
        raise ValueError(f"{name} archive differs from the trained run. Expected {recorded['sha256']}, got {metadata['sha256']}. Refusing to restore with a different source version.")
    key = "esc_dir" if name == "esc" else "speech_dir"
    destination = directory(cfg, key)
    if not destination.exists() or (destination / ".extracted.json").exists():
        extract_verified(archive, destination, metadata, esc=name == "esc")
    else:
        print(f"Existing {name} extraction: each referenced source will be checked against its frozen PCM hash.", flush=True)
    write_json(directory(cfg, "reports") / f"restored_{name}_archive.json", {
        "created_at": timestamp(), "archive": str(archive), "url": recorded["url"],
        "sha256": metadata["sha256"], "matches_trained_run": True, "bytes": archive.stat().st_size})
    return metadata


def restore_mixtures(cfg, split="all", supplemental=True):
    manifests = directory(cfg, "manifests")
    source_metadata = read_json(manifests / "sources_metadata.json")
    for name, digest in source_metadata["manifest_sha256"].items():
        if sha256_file(manifests / name) != digest:
            raise ValueError(f"Frozen source manifest changed: {name}")
    backgrounds = {row["source_id"]: row for row in read_csv(manifests / "background_sources.csv")}
    speeches = {row["source_id"]: row for row in read_csv(manifests / "speech_sources.csv")}
    main_path = manifests / "dataset.csv"
    main_metadata = read_json(manifests / "dataset_metadata.json")
    if sha256_file(main_path) != main_metadata["manifest_sha256"]:
        raise ValueError("Dataset manifest differs from the frozen training version.")
    for key in ("audio", "mix", "seed", "target_word"):
        if cfg[key] != main_metadata["config"][key]:
            raise ValueError(f"Restoration configuration differs from generation: {key}")
    rows = read_csv(main_path)
    main_count = len(rows)
    if supplemental and (manifests / "supplemental.csv").is_file():
        additional = manifests / "supplemental.csv"
        extra_metadata = read_json(manifests / "supplemental_metadata.json")
        if sha256_file(additional) != extra_metadata["manifest_sha256"] or extra_metadata["source_manifest_sha256"] != main_metadata["source_manifest_sha256"]:
            raise ValueError("Supplemental manifest/source fingerprint differs from the trained run.")
        rows.extend(read_csv(additional))
    if split != "all":
        rows = [row for row in rows if row["split"] == split]
    samples = round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_seconds"])

    @lru_cache(maxsize=64)
    def background_audio(identity):
        row = backgrounds[identity]
        audio, _ = load_audio(project_path(cfg, row["path"]), cfg["audio"]["sample_rate"])
        if pcm_hash(audio, cfg["audio"]["sample_rate"]) != row["pcm_sha256"]:
            raise ValueError(f"Background PCM differs from frozen source: {identity}")
        return fixed_length(audio, samples)[0]

    @lru_cache(maxsize=256)
    def speech_audio(identity):
        row = speeches[identity]
        audio, _ = load_audio(project_path(cfg, row["path"]), cfg["audio"]["sample_rate"])
        if pcm_hash(audio, cfg["audio"]["sample_rate"]) != row["pcm_sha256"]:
            raise ValueError(f"Speech PCM differs from frozen source: {identity}")
        return audio[int(row["trim_start"]):int(row["trim_end"])].copy()

    restored, existing, mismatches, counts = 0, 0, [], Counter()
    for index, row in enumerate(rows, 1):
        output = project_path(cfg, row["path"])
        if Path(cfg["_root"]).resolve() not in output.parents:
            raise ValueError(f"Generated output escapes the project: {row['path']}")
        counts[row["split"]] += 1
        if output.exists():
            if sha256_file(output) != row["audio_sha256"]:
                raise ValueError(f"Existing WAV differs; preserving it rather than overwriting: {output}")
            existing += 1
            continue
        try:
            mixed = background_audio(row["background_id"]).copy()
            if row["speech_id"]:
                speech = speech_audio(row["speech_id"])
                start = int(row["start_sample"])
                if len(speech) != int(row["speech_samples"]) or start < 0 or start + len(speech) > samples:
                    raise ValueError("Frozen speech position/length is invalid.")
                mixed[start:start + len(speech)] += float(row["speech_gain"]) * speech
            mixed *= float(row["output_gain"])
            if not np.isfinite(mixed).all():
                raise ValueError("Non-finite restored mixture.")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".restore.tmp.wav")
            sf.write(temporary, mixed, cfg["audio"]["sample_rate"], subtype="PCM_16")
            digest = sha256_file(temporary)
            if digest != row["audio_sha256"]:
                temporary.replace(output.with_suffix(".restore_mismatch.wav"))
                raise ValueError(f"Output SHA-256 mismatch: expected {row['audio_sha256']}, got {digest}. Check source/DSP library versions.")
            temporary.replace(output)
            restored += 1
        except Exception as error:
            mismatches.append({"sample_id": row["sample_id"], "path": row["path"], "error": f"{type(error).__name__}: {error}"})
        if index % 200 == 0:
            print(f"Restored/verified {index}/{len(rows)} WAVs; mismatches={len(mismatches)}", flush=True)
    report = {"created_at": timestamp(), "passed": not mismatches, "split": split, "supplemental_included": supplemental,
              "frozen_main_dataset_files": main_count, "selected_files": len(rows), "newly_restored": restored,
              "already_matching": existing, "counts_by_split_including_supplemental": dict(counts),
              "manifest_sha256": sha256_file(main_path), "all_output_hashes_match": not mismatches,
              "training_executed": False, "mismatches": mismatches}
    write_json(directory(cfg, "reports") / f"dataset_restoration_{split}.json", report)
    if mismatches:
        raise ValueError(f"{len(mismatches)} WAVs could not be restored exactly. See dataset_restoration_{split}.json. No approximate WAVs were accepted as the original dataset.")
    print(f"Restoration PASSED: {restored} created, {existing} already matching; {len(rows)} exact WAV hashes.", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trained_run.yaml")
    parser.add_argument("--source", choices=("esc", "speech", "both"), default="both")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--skip-download", action="store_true", help="Use the already extracted sources, checking referenced PCM hashes.")
    parser.add_argument("--split", choices=("all", "train", "validation", "test"), default="all")
    parser.add_argument("--no-supplemental", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.skip_download:
        for name in (("esc", "speech") if args.source == "both" else (args.source,)):
            restore_source(cfg, name)
    if not args.download_only:
        restore_mixtures(cfg, args.split, not args.no_supplemental)


if __name__ == "__main__":
    main()

"""Freeze official splits, QC sources, and quarantine connected leakage groups."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import shutil

from .audio import load_audio, pcm_hash, rms, trim_speech
from .common import (SPLITS, apply_source_overrides, code_hash, config_parser, directory,
                     load_config, object_hash, public_config, read_csv, relative_path,
                     sha256_file, timestamp, write_csv, write_json)

SOURCE_FIELDS = ["source_kind", "source_id", "path", "split", "background_source_id", "category", "fold",
                 "word", "speaker_id", "pcm_sha256", "file_sha256", "original_sample_rate", "original_channels",
                 "original_samples", "duration_seconds", "canonical_samples", "rms", "trim_start", "trim_end"]


def split_overlap(rows, field):
    memberships = defaultdict(set)
    for row in rows:
        if row.get(field):
            memberships[row[field]].add(row["split"])
    return {key: sorted(value) for key, value in memberships.items() if len(value) > 1}


def quarantine_sources(rows):
    """Quarantine all linked speakers/ESC sources for duplicate PCM or conflicting splits.

    The policy removes the whole connected component from all splits; it never reassigns a split.
    Duplicate content in the same split is quarantined too, keeping source counts conservative.
    """
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        kind = row["source_kind"]
        groups[(kind, "source_id", row["source_id"])].append(index)
        # PCM hash is shared across BOTH corpora as a further cross-source leakage check.
        groups[("all", "pcm_sha256", row["pcm_sha256"])].append(index)
        if kind == "background":
            groups[(kind, "background_source_id", row["background_source_id"])].append(index)
        else:
            groups[(kind, "speaker_id", row["speaker_id"])].append(index)
    reasons = []
    offending = []
    for key, indices in sorted(groups.items()):
        for index in indices[1:]:
            union(indices[0], index)
        split_conflict = len({rows[index]["split"] for index in indices}) > 1
        duplicate = key[1] in ("source_id", "pcm_sha256") and len(indices) > 1
        if split_conflict or duplicate:
            reasons.append({"field": key[1], "value": key[2], "source_kind": key[0],
                            "reason": "cross_split" if split_conflict else "duplicate_content_or_id",
                            "records": len(indices)})
            offending.extend(indices)
    bad_roots = {find(index) for index in offending}
    kept, removed = [], []
    for index, row in enumerate(rows):
        if find(index) in bad_roots:
            removed.append({**row, "reason": "quarantined_connected_duplicate_or_split_conflict"})
        else:
            kept.append(row)
    return kept, removed, reasons


def inspect_source(job, cfg):
    row, path = job
    try:
        audio, metadata = load_audio(path, cfg["audio"]["sample_rate"])
        row.update(metadata)
        row["canonical_samples"] = len(audio)
        row["rms"] = rms(audio)
        row["pcm_sha256"] = pcm_hash(audio, cfg["audio"]["sample_rate"])
        row["file_sha256"] = sha256_file(path)
        if row["source_kind"] == "speech":
            trimmed, start, end, _ = trim_speech(audio, cfg["audio"]["sample_rate"],
                                                cfg["audio"]["trim_db"], cfg["audio"]["trim_padding_ms"])
            if len(trimmed) > round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_seconds"]):
                raise ValueError("Speech word is longer than the entire input; refusing to crop it.")
            row.update(trim_start=start, trim_end=end)
        return row, None
    except Exception as error:
        return None, {**row, "reason": f"{type(error).__name__}: {error}"}


def prepare(cfg, workers=4):
    esc, speech = directory(cfg, "esc_dir"), directory(cfg, "speech_dir")
    metadata_path = esc / "meta/esc50.csv"
    esc_rows = read_csv(metadata_path)
    required = {"filename", "fold", "category", "src_file"}
    if not esc_rows or not required.issubset(esc_rows[0]):
        raise ValueError("ESC metadata must contain filename, fold, category and src_file.")
    validation_path, test_path = speech / "validation_list.txt", speech / "testing_list.txt"
    validation = set(validation_path.read_text(encoding="utf-8").splitlines())
    test = set(test_path.read_text(encoding="utf-8").splitlines())
    if validation & test:
        raise ValueError("Official validation/testing lists overlap. Repair or replace the source archive.")
    jobs = []
    for entry in esc_rows:
        fold = int(entry["fold"])
        if fold not in range(1, 6) or not entry["src_file"]:
            raise ValueError(f"Invalid ESC metadata: {entry}")
        name = entry["filename"]
        if "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError(f"Unsafe ESC filename: {name}")
        row = dict.fromkeys(SOURCE_FIELDS, "")
        row.update(source_kind="background", source_id=name, path=relative_path(cfg, esc / "audio" / name),
                   split="train" if fold <= 3 else "validation" if fold == 4 else "test",
                   background_source_id=entry["src_file"], category=entry["category"], fold=fold)
        jobs.append((row, esc / "audio" / name))
    found_speech = set()
    for word_directory in sorted(speech.iterdir()):
        if not word_directory.is_dir() or word_directory.name.startswith("_"):
            continue
        for path in sorted(word_directory.glob("*.wav")):
            speech_id = f"{word_directory.name}/{path.name}"
            if "_nohash_" not in path.stem:
                raise ValueError(f"Cannot derive speaker ID from {speech_id}")
            row = dict.fromkeys(SOURCE_FIELDS, "")
            row.update(source_kind="speech", source_id=speech_id, path=relative_path(cfg, path),
                       split="validation" if speech_id in validation else "test" if speech_id in test else "train",
                       word=word_directory.name, speaker_id=path.stem.split("_nohash_", 1)[0])
            jobs.append((row, path))
            found_speech.add(speech_id)
    missing = sorted((validation | test) - found_speech)
    if missing:
        raise ValueError(f"Speech Commands extraction is incomplete: {len(missing)} listed recordings missing, e.g. {missing[:3]}")
    valid, rejected = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (row, error) in enumerate(pool.map(lambda job: inspect_source(job, cfg), jobs), 1):
            (rejected if error else valid).append(error or row)
            if index % 5000 == 0:
                print(f"QC {index}/{len(jobs)} sources", flush=True)
    kept, quarantined, conflicts = quarantine_sources(valid)
    backgrounds = [row for row in kept if row["source_kind"] == "background"]
    speeches = [row for row in kept if row["source_kind"] == "speech"]
    manifests = directory(cfg, "manifests")
    write_csv(manifests / "background_sources.csv", backgrounds, SOURCE_FIELDS)
    write_csv(manifests / "speech_sources.csv", speeches, SOURCE_FIELDS)
    write_csv(manifests / "quarantine.csv", rejected + quarantined, SOURCE_FIELDS + ["reason"])
    licenses = directory(cfg, "reports") / "source_licenses"
    licenses.mkdir(parents=True, exist_ok=True)
    for prefix, source_dir in (("esc50", esc), ("speech_commands_v0.02", speech)):
        for filename in ("LICENSE", "LICENSE.txt", "README.md", "README.txt"):
            if (source_dir / filename).is_file():
                shutil.copy2(source_dir / filename, licenses / (prefix + "_" + filename))
    # All backgrounds appear in a listening checklist, ordered by likely human content.
    human = {"crying_baby", "sneezing", "clapping", "breathing", "coughing", "laughing", "snoring", "footsteps", "drinking_sipping", "brushing_teeth"}
    existing_listening = {row["background_id"]: row for row in read_csv(manifests / "background_listening_checklist.csv")} if (manifests / "background_listening_checklist.csv").exists() else {}
    listening = [{"background_id": row["source_id"], "path": row["path"], "split": row["split"],
                  "category": row["category"], "priority": 1 if row["category"] in human else 2,
                  "human_review_status": existing_listening.get(row["source_id"], {}).get("human_review_status", "unreviewed"),
                  "target_word_heard": existing_listening.get(row["source_id"], {}).get("target_word_heard", ""),
                  "notes": existing_listening.get(row["source_id"], {}).get("notes", "")} for row in backgrounds]
    write_csv(manifests / "background_listening_checklist.csv", sorted(listening, key=lambda row: (row["priority"], row["background_id"])))
    audit = {}
    for field, rows in (("source_id", backgrounds), ("background_source_id", backgrounds),
                        ("source_id", speeches), ("speaker_id", speeches), ("pcm_sha256", kept)):
        overlap = split_overlap(rows, field)
        audit[f"{rows[0]['source_kind'] if rows else 'empty'}_{field}"] = {"overlap_count": len(overlap), "examples": list(overlap)[:10]}
        if overlap:
            raise ValueError(f"Leakage persists after quarantine: {field}")
    report = {"created_at": timestamp(), "config": public_config(cfg), "code_sha256": code_hash(),
              "quarantine_policy": "Remove all connected speakers/ESC source IDs involved in duplicate PCM or split conflicts from every split; no reassignment.",
              "inputs": {"esc_metadata_sha256": sha256_file(metadata_path),
                         "validation_list_sha256": sha256_file(validation_path), "testing_list_sha256": sha256_file(test_path)},
              "manifest_sha256": {name: sha256_file(manifests / name) for name in ("background_sources.csv", "speech_sources.csv")},
              "counts": {split: {"backgrounds": sum(row["split"] == split for row in backgrounds),
                                 "speech_files": sum(row["split"] == split for row in speeches),
                                 "speakers": len({row["speaker_id"] for row in speeches if row["split"] == split})} for split in SPLITS},
              "qc_rejected_count": len(rejected), "quarantined_count": len(quarantined),
              "conflicts": conflicts, "split_audit": audit,
              "labels_limitation": "Backgrounds have not all been verified by human listening. Synthetic negative labels may contain unrecognized target speech.",
              "real_recordings": "chưa đánh giá trên bản ghi thực tế"}
    write_json(manifests / "sources_metadata.json", report)
    for split in SPLITS:
        words = {row["word"] for row in speeches if row["split"] == split}
        if cfg["target_word"] not in words or len(words - {cfg["target_word"]}) == 0 or not report["counts"][split]["backgrounds"]:
            raise ValueError(f"{split} has insufficient usable sources after QC/quarantine; see sources_metadata.json.")
    print(report["counts"], flush=True)
    return report


def main():
    parser = config_parser("Freeze source splits and audit canonical PCM/speaker identities.", sources=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    prepare(apply_source_overrides(load_config(args.config), args), args.workers)


if __name__ == "__main__":
    main()

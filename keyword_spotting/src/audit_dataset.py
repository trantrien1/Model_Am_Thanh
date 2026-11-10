"""Gate training on structural, source, identity, hash and audio checks."""
from collections import Counter, defaultdict

import numpy as np
import soundfile as sf

from .audio import pcm_hash
from .build_dataset import verified_sources
from .common import (SPLITS, config_parser, directory, load_config, project_path, read_csv,
                     read_json, sha256_file, timestamp, write_csv, write_json)
from .prepare_sources import split_overlap


def distribution(values):
    values = [float(value) for value in values if value != ""]
    return {"count": len(values), "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def audit(cfg, supplemental=False):
    name = "supplemental" if supplemental else "dataset"
    manifests, reports = directory(cfg, "manifests"), directory(cfg, "reports")
    rows = read_csv(manifests / f"{name}.csv")
    metadata = read_json(manifests / f"{name}_metadata.json")
    backgrounds, speeches, source_metadata = verified_sources(cfg)
    errors = []
    if sha256_file(manifests / f"{name}.csv") != metadata["manifest_sha256"]:
        errors.append("Generated manifest changed since mixing.")
    if metadata["source_manifest_sha256"] != source_metadata["manifest_sha256"]:
        errors.append("Source manifests changed since mixing.")
    for key in ("audio", "mix", "target_word", "seed"):
        if metadata["config"][key] != cfg[key]:
            errors.append(f"Dataset generation config differs: {key}")
    bg_by_id = {row["source_id"]: row for row in backgrounds}
    speech_by_id = {row["source_id"]: row for row in speeches}
    overlaps = {}
    for field in ("background_id", "background_source_id", "speech_id", "speaker_id", "background_pcm_sha256", "speech_pcm_sha256"):
        overlap = split_overlap(rows, field)
        overlaps[field] = {"count": len(overlap), "examples": list(overlap)[:10]}
        if overlap:
            errors.append(f"Cross-split leakage in {field}: {len(overlap)}")
    # Catch cross-kind source hashes (a speech file matching a background from another split).
    combined_hashes = [{"split": row["split"], "hash": row[field]} for row in rows
                       for field in ("background_pcm_sha256", "speech_pcm_sha256") if row[field]]
    cross_kind = split_overlap(combined_hashes, "hash")
    if cross_kind:
        errors.append(f"Cross-kind canonical PCM leakage: {len(cross_kind)}")
    output_hashes, groups = [], defaultdict(list)
    seen_ids, seen_paths = set(), set()
    for index, row in enumerate(rows):
        identity = row["sample_id"]
        try:
            if identity in seen_ids or row["path"] in seen_paths:
                raise ValueError("Duplicate sample ID/path.")
            seen_ids.add(identity)
            seen_paths.add(row["path"])
            if row["split"] not in SPLITS or row["label"] not in ("0", "1"):
                raise ValueError("Invalid label/split.")
            if supplemental and row["split"] != "test":
                raise ValueError("Supplemental data must be test-only.")
            background = bg_by_id[row["background_id"]]
            for output_key, source_key in (("split", "split"), ("background_source_id", "background_source_id"),
                                           ("background_pcm_sha256", "pcm_sha256"), ("background_fold", "fold"),
                                           ("background_category", "category")):
                if row[output_key] != background[source_key]:
                    raise ValueError(f"Background provenance mismatch: {output_key}")
            expected_split = "train" if int(row["background_fold"]) <= 3 else "validation" if int(row["background_fold"]) == 4 else "test"
            if row["split"] != expected_split:
                raise ValueError("ESC official fold was changed.")
            if row["speech_id"]:
                speech = speech_by_id[row["speech_id"]]
                for output_key, source_key in (("split", "split"), ("speaker_id", "speaker_id"),
                                               ("speech_pcm_sha256", "pcm_sha256"), ("inserted_word", "word")):
                    if row[output_key] != speech[source_key]:
                        raise ValueError(f"Speech provenance mismatch: {output_key}")
                if int(row["label"]) != int(row["inserted_word"] == cfg["target_word"]):
                    raise ValueError("Label conflicts with inserted word.")
                start, length = int(row["start_sample"]), int(row["speech_samples"])
                expected_length = int(speech["trim_end"]) - int(speech["trim_start"])
                total = round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_seconds"])
                if start < 0 or length != expected_length or start + length > total:
                    raise ValueError("Insertion position/length invalid or word cropped.")
                if row["mix_level_mode"] == "snr":
                    if abs(float(row["snr_db_requested"]) - float(row["snr_db_measured"])) > 0.01:
                        raise ValueError("Measured component SNR differs from requested SNR.")
                elif row["mix_level_mode"] != "fallback_dbfs" or row["snr_db_measured"]:
                    raise ValueError("Fallback SNR must be blank.")
            else:
                speech_fields = ("speaker_id", "speech_pcm_sha256", "inserted_word", "start_sample", "speech_samples", "snr_db_requested", "snr_db_measured", "speech_gain")
                if row["label"] != "0" or row["mix_level_mode"] != "clean" or any(row[field] for field in speech_fields):
                    raise ValueError("Clean background label or empty speech metadata is invalid.")
            path = project_path(cfg, row["path"])
            info = sf.info(path)
            expected_samples = round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_seconds"])
            if info.samplerate != cfg["audio"]["sample_rate"] or info.channels != 1 or info.frames != expected_samples or info.subtype != "PCM_16":
                raise ValueError("Output must be exactly 5 s, mono, 16 kHz PCM_16.")
            if sha256_file(path) != row["audio_sha256"]:
                raise ValueError("Output WAV was changed after generation.")
            audio, _ = sf.read(path, dtype="float32")
            if not np.isfinite(audio).all() or np.max(np.abs(audio)) > cfg["mix"]["peak_limit"] + 1 / 32768:
                raise ValueError("Non-finite output or peak exceeds anti-clipping limit.")
            output_hashes.append({"split": row["split"], "hash": pcm_hash(audio, cfg["audio"]["sample_rate"])})
            groups[row["background_id"]].append(row)
        except Exception as error:
            errors.append(f"{identity}: {type(error).__name__}: {error}")
        if (index + 1) % 2000 == 0:
            print(f"Audited {index + 1}/{len(rows)} WAVs", flush=True)
    generated_overlap = split_overlap(output_hashes, "hash")
    overlaps["generated_pcm_sha256"] = {"count": len(generated_overlap), "examples": list(generated_overlap)[:10]}
    if generated_overlap:
        errors.append("Identical generated PCM appears in different splits.")
    if not supplemental:
        for background_id, variants in groups.items():
            names = {row["variant_type"] for row in variants}
            if len(variants) != 4 or sum(int(row["label"]) for row in variants) != 2 or names not in ({"P1", "P2", "N1", "N2"}, {"P1", "P2", "N1", "N2_clean"}):
                errors.append(f"Incomplete/unbalanced variants for {background_id}")
    counts = {}
    for split in SPLITS:
        subset = [row for row in rows if row["split"] == split]
        if not supplemental and (not subset or {row["label"] for row in subset} != {"0", "1"}):
            errors.append(f"{split}: both labels are required.")
        counts[split] = {"files": len(subset), "backgrounds": len({row["background_id"] for row in subset}),
                         "independent_esc_sources": len({row["background_source_id"] for row in subset}),
                         "speakers": len({row["speaker_id"] for row in subset if row["speaker_id"]}),
                         "labels": dict(Counter(row["label"] for row in subset)),
                         "variant_types": dict(Counter(row["variant_type"] for row in subset)),
                         "negative_words": dict(Counter(row["inserted_word"] for row in subset if row["label"] == "0" and row["inserted_word"])),
                         "by_label": {label: {"snr_requested": distribution(row["snr_db_requested"] for row in subset if row["label"] == label),
                                              "start_sample": distribution(row["start_sample"] for row in subset if row["label"] == label),
                                              "output_gain": distribution(row["output_gain"] for row in subset if row["label"] == label),
                                              "fallback_files": sum(row["mix_level_mode"] == "fallback_dbfs" for row in subset if row["label"] == label)} for label in ("0", "1")}}
    speaker_counts = Counter((row["split"], row["speaker_id"], row["label"]) for row in rows if row["speaker_id"])
    write_csv(reports / f"{name}_speaker_label_distribution.csv",
              [{"split": split, "speaker_id": speaker, "label": label, "count": count} for (split, speaker, label), count in sorted(speaker_counts.items())],
              ["split", "speaker_id", "label", "count"])
    report = {"passed": not errors, "created_at": timestamp(), "manifest_sha256": sha256_file(manifests / f"{name}.csv"),
              "source_manifest_sha256": source_metadata["manifest_sha256"], "checked_audio_files": len(rows),
              "split_overlap": overlaps, "counts": counts, "errors": errors,
              "snr_measurement": "Component speech/background RMS on the same active-speech mask before PCM16 quantization; global gain preserves this ratio.",
              "limitations": [source_metadata["labels_limitation"], "Variants sharing ESC backgrounds/source recordings are correlated.",
                              "chưa đánh giá trên bản ghi thực tế"]}
    write_json(reports / f"{name}_audit.json", report)
    if errors:
        raise ValueError(f"Dataset audit FAILED with {len(errors)} errors. Training prohibited. See {reports / (name + '_audit.json')}")
    print(f"Audit PASSED: {counts}", flush=True)
    return report


def main():
    parser = config_parser("Audit every WAV, label and source split before training.")
    parser.add_argument("--supplemental", action="store_true")
    args = parser.parse_args()
    audit(load_config(args.config), args.supplemental)


if __name__ == "__main__":
    main()

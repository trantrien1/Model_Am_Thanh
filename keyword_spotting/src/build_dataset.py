"""Reproducible P1/P2/N1/N2 mixtures, with complete source provenance."""
from collections import defaultdict
from functools import lru_cache

import numpy as np
import soundfile as sf

from .audio import fixed_length, load_audio, mix_audio, pcm_hash, trim_speech
from .common import (SPLITS, code_hash, config_parser, directory, load_config, object_hash,
                     project_path, public_config, read_csv, read_json, relative_path,
                     rng_for, sha256_file, timestamp, write_csv, write_json)

GENERATED_FIELDS = ["sample_id", "path", "label", "split", "variant_type", "background_id", "background_source_id",
                    "background_category", "background_fold", "background_pcm_sha256", "speech_id", "speaker_id",
                    "speech_pcm_sha256", "inserted_word", "start_sample", "speech_samples", "snr_db_requested",
                    "snr_db_measured", "mix_level_mode", "speech_gain", "output_gain", "background_rms",
                    "speech_rms_output", "background_length_action", "audio_sha256", "seed"]


def verified_sources(cfg):
    manifests = directory(cfg, "manifests")
    metadata = read_json(manifests / "sources_metadata.json")
    for name, expected in metadata["manifest_sha256"].items():
        if sha256_file(manifests / name) != expected:
            raise ValueError(f"Source manifest changed after QC: {name}. Run prepare_sources again.")
    original = metadata["config"]
    for key in ("audio", "target_word", "seed"):
        if original[key] != cfg[key]:
            raise ValueError(f"Source preprocessing configuration changed: {key}. Run prepare_sources again.")
    return read_csv(manifests / "background_sources.csv"), read_csv(manifests / "speech_sources.csv"), metadata


def build(cfg, supplemental=False):
    backgrounds, speeches, source_metadata = verified_sources(cfg)
    samples = round(cfg["audio"]["sample_rate"] * cfg["audio"]["duration_seconds"])
    pools = defaultdict(lambda: defaultdict(list))
    for row in speeches:
        pools[row["split"]][row["word"]].append(row)

    @lru_cache(maxsize=256)
    def speech_audio(source_id):
        row = speech_by_id[source_id]
        audio, _ = load_audio(project_path(cfg, row["path"]), cfg["audio"]["sample_rate"])
        if pcm_hash(audio, cfg["audio"]["sample_rate"]) != row["pcm_sha256"]:
            raise ValueError(f"Speech source changed: {source_id}")
        trimmed, start, end, mask = trim_speech(audio, cfg["audio"]["sample_rate"], cfg["audio"]["trim_db"], cfg["audio"]["trim_padding_ms"])
        if (start, end) != (int(row["trim_start"]), int(row["trim_end"])):
            raise ValueError("Source trim changed after source manifest was written.")
        return trimmed, mask

    speech_by_id = {row["source_id"]: row for row in speeches}
    rows, rejected = [], []
    for split in SPLITS:
        if supplemental and split != "test":
            continue
        split_backgrounds = [row for row in backgrounds if row["split"] == split]
        choice_rng = rng_for(cfg["seed"], split + ":clean_selection")
        clean_count = round(len(split_backgrounds) * cfg["mix"]["clean_background_fraction"])
        clean_ids = set(choice_rng.choice([row["source_id"] for row in split_backgrounds], clean_count, replace=False))
        negative_words = sorted(set(pools[split]) - {cfg["target_word"]})
        # Balanced shuffled cyclic schedule: each negative word differs in count by at most one.
        count = len(split_backgrounds) * 2 - clean_count
        schedule = np.resize(np.array(negative_words), count)
        rng_for(cfg["seed"], split + ":negative_word_schedule").shuffle(schedule)
        negative_index = 0
        for background_index, background in enumerate(split_backgrounds):
            stem = background["source_id"].rsplit(".", 1)[0]
            planned_words = {"N1": None, "N2": None}
            if not supplemental:
                planned_words["N1"] = str(schedule[negative_index])
                negative_index += 1
                if background["source_id"] not in clean_ids:
                    planned_words["N2"] = str(schedule[negative_index])
                    negative_index += 1
            group = []
            variants = ("clean_extra", "quiet_P1", "quiet_N1") if supplemental else ("P1", "P2", "N1", "N2")
            try:
                audio, _ = load_audio(project_path(cfg, background["path"]), cfg["audio"]["sample_rate"])
                if pcm_hash(audio, cfg["audio"]["sample_rate"]) != background["pcm_sha256"]:
                    raise ValueError("Background source changed since QC.")
                audio, action = fixed_length(audio, samples)
                positive_one = None
                for variant in variants:
                    clean = variant == "clean_extra" or (variant == "N2" and background["source_id"] in clean_ids)
                    if supplemental and clean and background["source_id"] in clean_ids:
                        continue  # Remaining clean test backgrounds, avoiding exact main-test copies.
                    positive = variant in ("P1", "P2", "quiet_P1")
                    sample_id = f"{split}_{stem}_{variant}"
                    rng = rng_for(cfg["seed"], sample_id)
                    word = cfg["target_word"] if positive else (negative_words[background_index % len(negative_words)] if supplemental else planned_words[variant])
                    source, speech, mask = None, None, None
                    start, requested = "", ""
                    if not clean:
                        candidates = pools[split][word]
                        if variant == "P2" and positive_one is not None:
                            alternate = [row for row in candidates if row["speaker_id"] != positive_one["speaker_id"]]
                            candidates = alternate or [row for row in candidates if row["source_id"] != positive_one["source_id"]]
                            if not candidates:
                                raise ValueError("P1 and P2 require two different recordings.")
                        source = candidates[int(rng.integers(len(candidates)))]
                        if variant == "P1":
                            positive_one = source
                        speech, mask = speech_audio(source["source_id"])
                        start = int(rng.integers(0, samples - len(speech) + 1))
                        requested = float(rng.uniform(-10, 0) if supplemental else rng.uniform(cfg["mix"]["snr_min_db"], cfg["mix"]["snr_max_db"]))
                    mixed, details = mix_audio(audio, speech, start, requested, mask,
                                               cfg["audio"]["silence_rms"], cfg["mix"]["fallback_speech_dbfs"], cfg["mix"]["peak_limit"])
                    output = directory(cfg, "generated") / ("supplemental" if supplemental else split) / (sample_id + ".wav")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(".tmp.wav")
                    sf.write(temporary, mixed, cfg["audio"]["sample_rate"], subtype="PCM_16")
                    temporary.replace(output)
                    row = dict.fromkeys(GENERATED_FIELDS, "")
                    row.update(sample_id=sample_id, path=relative_path(cfg, output), label=int(positive), split=split,
                               variant_type=variant + ("_clean" if clean and variant == "N2" else ""),
                               background_id=background["source_id"], background_source_id=background["background_source_id"],
                               background_category=background["category"], background_fold=background["fold"],
                               background_pcm_sha256=background["pcm_sha256"], start_sample=start,
                               speech_samples=len(speech) if speech is not None else "", snr_db_requested=requested,
                               background_length_action=action, audio_sha256=sha256_file(output), seed=cfg["seed"], **details)
                    if source:
                        row.update(speech_id=source["source_id"], speaker_id=source["speaker_id"], inserted_word=source["word"],
                                   speech_pcm_sha256=source["pcm_sha256"])
                    group.append(row)
                rows.extend(group)
            except Exception as error:
                # No incomplete group reaches the manifest; any orphan WAV is ignored by later stages.
                rejected.append({"background_id": background["source_id"], "split": split,
                                 "reason": f"{type(error).__name__}: {error}", "variants_excluded": len(variants)})
            if (background_index + 1) % 100 == 0:
                print(f"{split}: mixed {background_index + 1}/{len(split_backgrounds)} backgrounds", flush=True)
    name = "supplemental" if supplemental else "dataset"
    manifests = directory(cfg, "manifests")
    write_csv(manifests / f"{name}.csv", rows, GENERATED_FIELDS)
    write_csv(manifests / f"{name}_rejected.csv", rejected, ["background_id", "split", "reason", "variants_excluded"])
    actual = {split: sum(row["split"] == split for row in rows) for split in SPLITS}
    default = {"train": 4800, "validation": 1600, "test": 1600}
    report = {"created_at": timestamp(), "config": public_config(cfg), "code_sha256": code_hash(),
              "manifest_sha256": sha256_file(manifests / f"{name}.csv"),
              "source_manifest_sha256": source_metadata["manifest_sha256"], "counts": actual,
              "difference_from_full_default": {split: actual[split] - default[split] for split in SPLITS},
              "rejected_background_groups": len(rejected), "supplemental": supplemental,
              "note": "Supplemental test-only exploratory data, never used to tune the model." if supplemental else "Four variants per retained background; labels are synthetic."}
    write_json(manifests / f"{name}_metadata.json", report)
    if not rows:
        raise ValueError(f"No mixtures created; see {name}_rejected.csv.")
    print(f"Created {name}: {actual}; excluded groups: {len(rejected)}", flush=True)
    return rows


def main():
    parser = config_parser("Build seeded, source-disjoint 5-second keyword mixtures.")
    parser.add_argument("--supplemental", action="store_true", help="Test-only remaining clean backgrounds and quieter speech (-10 to 0 dB).")
    args = parser.parse_args()
    build(load_config(args.config), args.supplemental)


if __name__ == "__main__":
    main()

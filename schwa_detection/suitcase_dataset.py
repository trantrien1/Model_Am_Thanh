"""Make a training-only supplement from pinned manual spontaneous L2-ARCTIC speech."""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

import l2_dataset as source
import schwa as core

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "training_suitcase"
SHA = "b0f2ac6d03410b764a90ae656e69befe12565f3dd1d91ffe6251b6056d68877e"
SEED = 20261013


def prepare():
    path = source.SOURCE / "suitcase-00000-of-00001.parquet"
    if core.digest(path) != SHA:
        raise ValueError("Pinned suitcase source missing or changed; verify download before preparation.")
    base = ROOT / "v5/data/manifests"
    frozen_names = ["dataset.csv", "validation.csv", "test.csv", "source_selection.json"]
    before = {name: core.digest(base / name) for name in frozen_names}
    selection = core.read_json(base / "source_selection.json")
    allowed = sorted(selection["speakers"]["train"])
    held = set(selection["speakers"]["validation"] + selection["speakers"]["test"])
    if set(allowed) & held:
        raise ValueError("Fixed speaker partitions overlap.")
    settings = {"source_repo": selection["source_repo"], "source_revision": source.REVISION,
        "source_sha256": {"suitcase": SHA}, "license": "CC BY-NC 4.0", "seed": SEED,
        "speakers": {"train": allowed}, "partition": "Training only; original V5 validation/test unchanged",
        "annotation_origin": "Manual perceived phones in spontaneous suitcase narrative; substitutions/additions/deletions preserved",
        "target_mapping": selection["target_mapping"],
        "unknown_policy": "Exclude windows overlapping uncertain intervals; preserve explicit non-native variants",
        "v5_manifests_sha256": before, "no_held_out_speakers": True}
    target = OUTPUT / "data/manifests/source_selection.json"
    if target.exists() and core.read_json(target) != settings:
        raise ValueError("Preserving an existing supplement with different source/settings.")
    core.ROOT, core.SPLITS = OUTPUT, ("train",)
    if not (OUTPUT / "data/manifests/dataset.csv").exists():
        core.write_json(target, settings)
        table = pq.read_table(path, filters=[("speaker_id", "in", allowed)],
            columns=["speaker_id", "utterance_id", "native_language", "transcript", "manual_events", "audio"])
        records = []
        for row in table.to_pylist():
            if row["speaker_id"] not in allowed or row["speaker_id"] in held:
                raise ValueError("Held-out speaker found in training supplement.")
            audio, rate = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32", always_2d=True)
            if rate != 16000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
                raise ValueError("Supplement requires finite 16 kHz mono source audio.")
            phones = source.convert_events(row["manual_events"], len(audio) / rate, sample_rounding=True)
            identity = row["speaker_id"] + "_" + row["utterance_id"]
            base_path = OUTPUT / "data/raw" / row["speaker_id"] / row["utterance_id"]
            base_path.parent.mkdir(parents=True, exist_ok=True)
            wav, lab, manual = base_path.with_suffix(".wav"), base_path.with_suffix(".lab"), base_path.with_suffix(".json")
            sf.write(wav, audio[:, 0], rate, subtype="PCM_16")
            lab.write_text("#\n" + "".join(f"{p['end']:.9f} 26 {p['phone']}\n" for p in phones), encoding="utf-8")
            core.write_json(manual, {key: value for key, value in row.items() if key != "audio"})
            records.append({"split": "train", "speaker_id": row["speaker_id"], "prompt_id": row["utterance_id"],
                "recording_id": identity, "native_language": row["native_language"],
                "annotation_origin": "human perceived phonemes, spontaneous suitcase narrative",
                "source_parquet_sha256": SHA, "audio_path": wav.relative_to(OUTPUT).as_posix(),
                "audio_sha256": core.digest(wav), "lab_path": lab.relative_to(OUTPUT).as_posix(),
                "lab_sha256": core.digest(lab), "manual_path": manual.relative_to(OUTPUT).as_posix(),
                "manual_sha256": core.digest(manual)})
        core.write_csv(OUTPUT / "data/manifests/recordings.csv", sorted(records, key=lambda r: r["recording_id"]))
    core.make_dataset(argparse.Namespace(seed=SEED))
    original = core.read_csv(base / "dataset.csv")
    supplemental = core.read_csv(OUTPUT / "data/manifests/dataset.csv")
    held_rows = [r for r in original if r["split"] in {"validation", "test"}]
    overlaps = {}
    for field in ["speaker_id", "prompt_id", "recording_id", "source_pcm_sha256", "pcm_sha256"]:
        collision = set(r[field] for r in supplemental) & set(r[field] for r in held_rows)
        overlaps[field] = sorted(collision)
        if collision:
            raise ValueError("Supplement overlaps held-out V5 data: " + field)
    after = {name: core.digest(base / name) for name in frozen_names}
    if before != after:
        raise ValueError("Original V5 evaluation manifests changed.")
    report = {"passed": True, "purpose": "Additional training only", "v5_manifests_unchanged": True,
        "v5_manifests_sha256": after, "source_sha256": SHA, "source_revision": source.REVISION,
        "cross_dataset_held_out_collisions": overlaps,
        "test_predictions_computed": False, "held_out_audio_decoded": False}
    core.write_json(OUTPUT / "reports/v5_preservation_audit.json", report)
    print("TRAIN-only supplement prepared; original V5 manifests unchanged.", flush=True)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    prepare()

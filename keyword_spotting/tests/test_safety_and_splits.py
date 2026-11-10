import io
import tarfile
import zipfile

import pytest

from src.common import read_csv
from src.download import fetch_archive, safe_extract, safe_member_path
from src.prepare_sources import prepare, quarantine_sources


@pytest.mark.parametrize("name", ["../escape.wav", "/absolute.wav", "C:/escape.wav", "a\\..\\escape.wav", "nested/../../escape.wav"])
def test_archive_traversal_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        safe_member_path(tmp_path, name)


def test_zip_traversal_and_tar_symlink_rejected(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.wav", b"invalid")
    with pytest.raises(ValueError):
        safe_extract(archive, tmp_path / "out")
    tar_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(tar_path, "w:gz") as handle:
        entry = tarfile.TarInfo("link")
        entry.type = tarfile.SYMTYPE
        entry.linkname = "../../outside"
        handle.addfile(entry)
    with pytest.raises(ValueError, match="links/special"):
        safe_extract(tar_path, tmp_path / "out2")
    assert not (tmp_path / "escape.wav").exists()


def test_verified_archive_is_not_downloaded_again(tmp_path, monkeypatch):
    original = tmp_path / "original.zip"
    with zipfile.ZipFile(original, "w") as handle:
        handle.writestr("nested/data.txt", b"valid archive")
    cached = tmp_path / "cache.zip"
    metadata = fetch_archive(original.as_uri(), cached, retries=1)
    assert len(metadata["sha256"]) == 64
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Should reuse cache")))
    assert fetch_archive(original.as_uri(), cached)["sha256"] == metadata["sha256"]
    with pytest.raises(ValueError, match="different recorded source URL"):
        fetch_archive("https://example.invalid/different-source.zip", cached)


def test_quarantine_all_connected_speaker_records():
    def record(identity, split, speaker, content):
        return {"source_kind": "speech", "source_id": identity, "split": split, "speaker_id": speaker, "pcm_sha256": content}
    rows = [record("a", "train", "speaker_a", "duplicate"), record("b", "test", "speaker_b", "duplicate"),
            record("c", "train", "speaker_a", "other_a"), record("d", "test", "speaker_b", "other_b"),
            record("e", "validation", "speaker_ok", "unique")]
    kept, removed, reasons = quarantine_sources(rows)
    assert [row["source_id"] for row in kept] == ["e"]
    assert len(removed) == 4
    assert any(row["field"] == "pcm_sha256" for row in reasons)


def test_source_lists_and_metadata_define_splits(toy_project):
    cfg, _ = toy_project
    report = prepare(cfg, workers=2)
    backgrounds = read_csv(cfg["_root"] + "/data/manifests/background_sources.csv")
    speech = read_csv(cfg["_root"] + "/data/manifests/speech_sources.csv")
    assert len(backgrounds) == 12
    assert len(speech) == 48
    assert not any(row["word"] == "_background_noise_" for row in speech)
    assert all(row["split"] == "validation" for row in backgrounds if row["fold"] == "4")
    assert all(value["overlap_count"] == 0 for value in report["split_audit"].values())

"""Bounded, verified downloads and conservative ZIP/TAR extraction."""
import gzip
import shutil
import stat
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from .common import (apply_source_overrides, config_parser, directory, load_config,
                     read_json, sha256_file, timestamp, write_json)


def safe_member_path(destination, name):
    name = name.replace("\\", "/")
    parts = PurePosixPath(name)
    if parts.is_absolute() or ".." in parts.parts or any(":" in part for part in parts.parts):
        raise ValueError(f"Unsafe archive path: {name}")
    root = Path(destination).resolve()
    path = (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Archive path escapes destination: {name}")
    return path


def validate_archive(path):
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None:
                raise ValueError("ZIP CRC verification failed.")
            if not archive.namelist():
                raise ValueError("Empty ZIP archive.")
    else:
        # Read through the gzip trailer as well: TAR parsing can stop before a damaged trailer.
        with gzip.open(path, "rb") as compressed:
            for _ in iter(lambda: compressed.read(1024 * 1024), b""):
                pass
        count = 0
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                if member.isfile():
                    count += 1
                    with archive.extractfile(member) as handle:
                        for _ in iter(lambda: handle.read(1024 * 1024), b""):
                            pass
        if count == 0:
            raise ValueError("Empty TAR archive.")


def safe_extract(path, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".zip"):
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            # Validate all paths and types before writing any member.
            for member in members:
                safe_member_path(destination, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                    raise ValueError(f"Archive links/special files are forbidden: {member.filename}")
            for member in members:
                output = safe_member_path(destination, member.filename)
                if member.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(member) as source, output.open("wb") as target:
                        shutil.copyfileobj(source, target)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                safe_member_path(destination, member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Archive links/special files are forbidden: {member.name}")
            for member in archive.getmembers():
                output = safe_member_path(destination, member.name)
                if member.isdir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, output.open("wb") as target:
                        shutil.copyfileobj(source, target)


def fetch_archive(url, path, timeout=60, retries=3):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = path.with_name(path.name + ".verified.json")
    if path.exists():
        digest = sha256_file(path)
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata.get("url") != url:
                raise ValueError("Cached archive belongs to a different recorded source URL. Keep it and choose a different archive path.")
            if metadata.get("sha256") == digest and metadata.get("url") == url:
                print(f"Already verified: {path.name}", flush=True)
                return metadata
        validate_archive(path)
        metadata = {"url": url, "sha256": digest, "bytes": path.stat().st_size,
                    "verified_at": timestamp(), "downloaded_at": None, "existing_archive": True}
        write_json(metadata_path, metadata)
        return metadata
    temporary = path.with_name(path.name + ".part")
    errors = []
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "keyword-spotting-research/1.0"})
            with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
                expected = response.headers.get("Content-Length")
                total = 0
                last_report = time.monotonic()
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    total += len(block)
                    if time.monotonic() - last_report > 10:
                        print(f"{path.name}: {total / 1024**2:.1f} MiB downloaded", flush=True)
                        last_report = time.monotonic()
            if expected is not None and total != int(expected):
                raise ValueError(f"Incomplete download: {total} / {expected} bytes")
            temporary.replace(path)
            try:
                validate_archive(path)
            except Exception:
                # Move only this newly downloaded archive out of the verified-name location.
                path.replace(temporary)
                raise
            metadata = {"url": url, "sha256": sha256_file(path), "bytes": total,
                        "downloaded_at": timestamp(), "verified_at": timestamp(), "existing_archive": False}
            write_json(metadata_path, metadata)
            return metadata
        except Exception as error:
            errors.append(f"Attempt {attempt}: {type(error).__name__}: {error}")
            print(errors[-1], flush=True)
            if attempt < retries:
                time.sleep(min(attempt * 2, 6))
    write_json(path.with_name(path.name + ".failure.json"), {"url": url, "time": timestamp(), "errors": errors})
    raise RuntimeError("Download failed after bounded retries. " + errors[-1])


def extract_verified(archive, destination, metadata, esc=False):
    destination = Path(destination)
    marker = destination / ".extracted.json"
    if marker.exists() and read_json(marker).get("archive_sha256") == metadata["sha256"]:
        print(f"Already extracted: {destination}", flush=True)
        return
    if destination.exists():
        raise ValueError(f"Existing directory has no matching extraction marker: {destination}. Use --esc-dir/--speech-dir to keep existing data.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + "-extract-", dir=destination.parent))
    try:
        safe_extract(archive, staging)
        if esc:
            roots = [path.parent.parent for path in staging.glob("*/meta/esc50.csv")]
            if len(roots) != 1:
                raise ValueError("ESC archive must contain exactly one meta/esc50.csv.")
            roots[0].replace(destination)
        else:
            if not (staging / "validation_list.txt").is_file() or not (staging / "testing_list.txt").is_file():
                raise ValueError("Full Speech Commands v0.02 archive must include both split lists.")
            staging.replace(destination)
        write_json(marker, {"archive_sha256": metadata["sha256"], "extracted_at": timestamp()})
    finally:
        # Staging is generated by tempfile under the explicitly resolved dataset parent.
        if staging.exists() and staging.parent.resolve() == destination.parent.resolve():
            shutil.rmtree(staging)


def main():
    args = config_parser("Download ESC-50 and Speech Commands v0.02 safely.", sources=True).parse_args()
    cfg = apply_source_overrides(load_config(args.config), args)
    archives = Path(cfg["_root"]) / "data/raw/archives"
    provenance = {}
    for name, override, key, filename in (("esc", args.esc_dir, "esc_dir", "esc50.zip"),
                                          ("speech", args.speech_dir, "speech_dir", "speech_commands_v0.02.tar.gz")):
        if override:
            if not directory(cfg, key).is_dir():
                raise FileNotFoundError(directory(cfg, key))
            provenance[name] = {"external_directory": cfg["paths"][key], "archive_sha256": None,
                                "note": "User-supplied extraction; original archive not available for verification."}
            continue
        metadata = fetch_archive(cfg["download"][name + "_url"], archives / filename,
                                 cfg["download"]["timeout_seconds"], cfg["download"]["retries"])
        extract_verified(archives / filename, directory(cfg, key), metadata, esc=name == "esc")
        provenance[name] = metadata
    write_json(directory(cfg, "manifests") / "download_provenance.json", provenance)
    print("Sources ready. Next: python -m src.prepare_sources --config " + args.config)


if __name__ == "__main__":
    main()

"""Download and verify the complete BSCCMNIST archives from Dryad.

Extract the eligible cohort's four DPC intensity channels and metadata. The
archives retain every cell and imaging contrast.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import time
import zipfile

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

BASE = "https://datadryad.org"
DOI = "10.5061/dryad.sxksn038s"
VERSION = "362939"


def safe_target(destination: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if ".." in path.parts or path.is_absolute() or any(":" in part for part in path.parts):
        raise ValueError(f"Unsafe tar path: {name}")
    root = destination.resolve()
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Tar path escapes extraction directory: {name}")
    return target


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_files(version: str) -> list[dict]:
    url = f"{BASE}/api/v2/versions/{version}/files"
    records = []
    while url:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        body = response.json()
        records.extend(body["_embedded"]["stash:files"])
        url = body.get("_links", {}).get("next", {}).get("href")
        if url and url.startswith("/"):
            url = BASE + url
    selected = sorted(
        (x for x in records if re.fullmatch(r"BSCCMNIST\.tar\.gz_chunk\d+\.bin", x["path"])),
        key=lambda x: x["path"],
    )
    numbers = [int(x["path"].split("_chunk")[1].split(".")[0]) for x in selected]
    if numbers != list(range(len(selected))):
        raise ValueError("Archive chunk sequence has gaps or duplicates")
    if any(
        x.get("digestType", "").lower() != "sha-256"
        or not re.fullmatch(r"[0-9a-f]{64}", x["digest"])
        for x in selected
    ):
        raise ValueError("Official SHA-256 metadata missing or malformed")
    return selected


def ingest_public_zip(source: Path, data_root: Path, records: list[dict]) -> dict:
    """Copy an official website ZIP and verify its full-variant chunk contents.

    The browser's original download is preserved. Arbitrary extra ZIP members are
    never extracted, and no supplied archive path is used as an output path.
    """
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"The browser ZIP is missing or incomplete: {source}")
    saved = data_root / "downloads" / source.name
    saved.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != saved.resolve():
        if saved.exists():
            if saved.stat().st_size != source.stat().st_size or digest_file(saved) != digest_file(
                source
            ):
                raise RuntimeError(f"A different ZIP already exists at {saved}")
        else:
            shutil.copy2(source, saved)
    expected = {record["path"]: record for record in records}
    archive_root = data_root / "archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    found = set()
    with zipfile.ZipFile(saved) as archive:
        for member in archive.infolist():
            filename = PurePosixPath(member.filename).name
            if filename not in expected:
                continue
            if filename in found:
                raise ValueError(f"Duplicate requested chunk inside ZIP: {filename}")
            found.add(filename)
            record = expected[filename]
            if member.file_size != record["size"]:
                raise ValueError(f"Wrong uncompressed size for ZIP member {filename}")
            target = archive_root / filename
            if (
                target.exists()
                and target.stat().st_size == record["size"]
                and digest_file(target) == record["digest"]
            ):
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            digest = hashlib.sha256()
            with archive.open(member) as stream, partial.open("wb") as output:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != record["digest"]:
                raise ValueError(f"Official SHA-256 mismatch for ZIP member {filename}")
            partial.replace(target)
            print(f"Verified ZIP member {filename}", flush=True)
    if found != set(expected):
        raise ValueError(
            f"Official ZIP missing requested full-variant chunks: {set(expected) - found}"
        )
    return {
        "method": "official_public_selected_files_zip",
        "zip_path": str(saved.resolve()),
        "zip_sha256": digest_file(saved),
    }


def download_one(
    record: dict, output: Path, token: str | None = None, public_download: bool = False
) -> dict:
    target = output / record["path"]
    partial = target.with_suffix(target.suffix + ".partial")
    expected = record["digest"]
    if target.exists():
        if target.stat().st_size == record["size"] and digest_file(target) == expected:
            print(f"Verified existing {target.name}", flush=True)
            return record
        raise RuntimeError(f"Existing archive failed integrity check: {target}")
    for attempt in range(8):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {"Range": f"bytes={start}-"} if start else {}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            file_id = record["_links"]["self"]["href"].split("/")[-1]
            url = (
                f"{BASE}/downloads/file_stream/{file_id}"
                if public_download
                else BASE + record["_links"]["stash:download"]["href"]
            )
            with requests.get(url, headers=headers, stream=True, timeout=(60, 180)) as response:
                response.raise_for_status()
                if start and response.status_code != 206:
                    start = 0
                mode = "ab" if start else "wb"
                written = start
                logged = time.monotonic()
                with partial.open(mode) as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
                            written += len(block)
                        if time.monotonic() - logged > 20:
                            print(
                                f"{target.name}: {written:,}/{record['size']:,} bytes", flush=True
                            )
                            logged = time.monotonic()
            if partial.stat().st_size != record["size"]:
                raise RuntimeError(f"Wrong download size for {target.name}")
            if digest_file(partial) != expected:
                raise RuntimeError(
                    f"SHA-256 mismatch for {target.name}; remove partial file before retrying"
                )
            partial.rename(target)
            print(f"Verified {target.name}: {expected}", flush=True)
            return record
        except (requests.RequestException, OSError) as error:
            if isinstance(error, requests.HTTPError) and error.response.status_code in (401, 403):
                raise RuntimeError(
                    "Dryad denied anonymous download. Set the chosen token environment variable from your Dryad account, or download the two named full BSCCMNIST chunks through its official website into archives/. Credentials are never saved by this script."
                ) from error
            if attempt == 7:
                raise
            print(f"Retry {attempt + 1} for {target.name}: {error}", flush=True)
            time.sleep(min(30, 2**attempt))
    raise RuntimeError("Unreachable")


class ConcatenatedReader(io.RawIOBase):
    def __init__(self, paths: list[Path]):
        self.paths = iter(paths)
        self.current = None

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while True:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    return 0
            size = self.current.readinto(buffer)
            if size:
                return size
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract(paths: list[Path], destination: Path, all_contrasts: bool) -> dict:
    # First obtain the metadata, so the second pass can retain all eligible cells
    # without unpacking unrelated fluorescence/histology files or unlabelled cells.
    import pandas as pd
    from cellselect.data import CHANNELS, make_cohort

    metadata_count = 0
    scanned = 0
    metadata_log = time.monotonic()
    with io.BufferedReader(ConcatenatedReader(paths), buffer_size=8 * 1024 * 1024) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                # Streaming tar iteration otherwise retains every TarInfo object.
                # BSCCMNIST has millions of small entries; we need no member cache.
                archive.members.clear()
                scanned += 1
                if time.monotonic() - metadata_log > 20:
                    print(f"Reading metadata: scanned {scanned:,} archive entries", flush=True)
                    metadata_log = time.monotonic()
                parts = PurePosixPath(member.name).parts
                if (
                    len(parts) == 2
                    and member.isfile()
                    and parts[-1]
                    in {
                        "BSCCM_index.csv",
                        "BSCCM_surface_markers.csv",
                        "BSCCM_global_metadata.json",
                    }
                ):
                    target = safe_target(destination, member.name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as source, target.open("wb") as output:
                        output.write(source.read())
                    metadata_count += 1
                    if metadata_count == 3:
                        break
    root = destination / "BSCCMNIST"
    index = pd.read_csv(root / "BSCCM_index.csv", index_col="global_index", low_memory=False)
    markers = pd.read_csv(
        root / "BSCCM_surface_markers.csv", index_col="global_index", low_memory=False
    )
    cohort, cohort_qc = make_cohort(index, markers)
    eligible = set(cohort.index.astype(int))
    global_metadata = json.loads((root / "BSCCM_global_metadata.json").read_text())
    channels = {global_metadata["led_array"]["channel_names"].index(name) for name in CHANNELS}
    count, total = 0, 0
    last_log = time.monotonic()
    with io.BufferedReader(ConcatenatedReader(paths), buffer_size=8 * 1024 * 1024) as stream:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                archive.members.clear()
                total += 1
                parts = PurePosixPath(member.name).parts
                # Complete eligible cohort, raw four-direction intensity channels.
                # Full source archives stay intact, including every other cell.
                selected = all_contrasts or len(parts) <= 2 or member.name.endswith(".zgroup")
                cell_parts = [part for part in parts if part.startswith("cell_")]
                if (
                    "led_array" in parts
                    and cell_parts
                    and int(cell_parts[0].split("_")[-1]) in eligible
                ):
                    leaf = parts[-1]
                    selected = (
                        selected
                        or member.isdir()
                        or leaf.startswith(".")
                        or (
                            re.match(r"^\d+\.\d+\.\d+$", leaf)
                            and int(leaf.split(".")[0]) in channels
                        )
                    )
                if not selected:
                    continue
                if member.issym() or member.islnk():
                    raise ValueError(f"Links are not expected in the data archive: {member.name}")
                target = safe_target(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not (target.exists() and target.stat().st_size == member.size):
                        with archive.extractfile(member) as source, target.open("wb") as out:
                            while block := source.read(1024 * 1024):
                                out.write(block)
                    count += 1
                if time.monotonic() - last_log > 20:
                    print(
                        f"Extraction: examined {total:,} entries, retained {count:,} files",
                        flush=True,
                    )
                    last_log = time.monotonic()
    return {
        "examined_members": total,
        "retained_files": count,
        "all_contrasts": all_contrasts,
        "scope": "full archive retained; eligible cohort four raw DPC intensity channels unpacked"
        if not all_contrasts
        else "all archive contents",
        "cohort_qc": cohort_qc,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--version", default=VERSION)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--all-contrasts", action="store_true")
    parser.add_argument(
        "--token-env",
        default="DRYAD_TOKEN",
        help="Name of environment variable containing a Dryad API token (never printed/saved)",
    )
    parser.add_argument(
        "--public-download",
        action="store_true",
        help="Use the official browser download endpoint instead of the authenticated API",
    )
    parser.add_argument(
        "--from-zip",
        type=Path,
        help="Ingest the completed official website selected-files ZIP; the original is preserved",
    )
    args = parser.parse_args()
    archive_root = args.data_root / "archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    records = list_files(args.version)
    if not records:
        raise RuntimeError("No full BSCCMNIST archives found")
    provenance = {
        "doi": DOI,
        "version": args.version,
        "files": records,
        "variant": "BSCCMNIST",
        "tiny": False,
    }
    (args.data_root / "download_metadata.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    if args.from_zip:
        provenance["acquisition"] = ingest_public_zip(args.from_zip, args.data_root, records)
    token = os.environ.get(args.token_env)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda r: download_one(r, archive_root, token, args.public_download), records
            )
        )
    if not args.download_only:
        provenance["extraction"] = extract(
            [archive_root / r["path"] for r in records], args.data_root / "raw", args.all_contrasts
        )
    (args.data_root / "download_manifest.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"status": "complete", "data_root": str(args.data_root), "files": len(records)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

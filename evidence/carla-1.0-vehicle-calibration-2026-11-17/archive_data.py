"""Archive completed collector roots and verify every preserved file."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import BinaryIO


def sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    roots = [path.resolve(strict=True) for path in args.data]
    destination = args.out.resolve()
    if len({root.name.casefold() for root in roots}) != len(roots):
        raise ValueError("Collector roots must have unique basenames")
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Collector root is not a directory: {root}")
        if destination.is_relative_to(root):
            raise ValueError("Archive output must be outside each collector root")
        status = json.loads((root / "run.json").read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            raise ValueError(f"Collector root is incomplete: {root}")
    outputs = [destination / f"{root.name}.zip" for root in roots]
    manifest_path = destination / "manifest.json"
    if any(path.exists() or path.is_symlink() for path in [*outputs, manifest_path]):
        raise FileExistsError("Existing archive output must be preserved")
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "compression": "ZIP_DEFLATED", "level": 6}
    archives = {}
    for root, output in zip(roots, outputs):
        members = {}
        paths = sorted(root.rglob("*"))
        if any(path.is_symlink() for path in paths):
            raise ValueError(f"Collector root contains a symbolic link: {root}")
        with zipfile.ZipFile(
            output, "x", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in paths:
                if not path.is_file():
                    continue
                if not path.resolve().is_relative_to(root):
                    raise ValueError(
                        f"Collector file resolves outside its root: {path}"
                    )
                name = path.relative_to(root).as_posix()
                with path.open("rb") as stream:
                    digest = sha256(stream)
                members[name] = {
                    "original_path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": digest,
                }
                archive.write(path, name)
        with zipfile.ZipFile(output) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"Archive CRC failed: {output}: {bad_member}")
            if set(archive.namelist()) != set(members):
                raise ValueError(f"Archive member inventory differs: {output}")
            for name, record in members.items():
                with archive.open(name) as stream:
                    actual = sha256(stream)
                if (
                    archive.getinfo(name).file_size != record["bytes"]
                    or actual != record["sha256"]
                ):
                    raise ValueError(f"Archive member verification failed: {name}")
        with output.open("rb") as stream:
            archive_digest = sha256(stream)
        archives[root.name] = {
            "source_root": str(root),
            "file": output.name,
            "bytes": output.stat().st_size,
            "sha256": archive_digest,
            "members": members,
        }
        print(f"Verified {output.name}: {len(members)} files", flush=True)
    manifest["archives"] = archives
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(f"Saved {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

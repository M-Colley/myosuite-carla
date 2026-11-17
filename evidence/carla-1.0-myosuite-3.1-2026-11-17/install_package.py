"""Extract the official CARLA ZIP into an empty, explicitly named directory."""

import argparse
import hashlib
import json
import shutil
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--archive", type=Path, required=True)
parser.add_argument("--destination", type=Path, required=True)
parser.add_argument("--report", type=Path, required=True)
args = parser.parse_args()
archive_path = args.archive.resolve(strict=True)
destination = args.destination.resolve()
if destination.exists() and any(destination.iterdir()):
    raise RuntimeError(f"Installation directory is not empty: {destination}")
if archive_path.stat().st_size != 13_916_909_891:
    raise RuntimeError("The archive size differs from the official download")
started = time.time()
with zipfile.ZipFile(archive_path) as archive:
    entries = archive.infolist()
    launchers = [info for info in entries if PurePosixPath(info.filename).name == "CarlaUnreal.exe"]
    if len(launchers) != 1:
        raise RuntimeError("Expected one CARLA server launcher in the archive")
    prefix = PurePosixPath(launchers[0].filename).parent
    expanded = sum(info.file_size for info in entries)
    free = shutil.disk_usage(destination.parent).free
    preflight = {"archive": str(archive_path), "archive_bytes": archive_path.stat().st_size,
                 "destination": str(destination), "expanded_bytes": expanded,
                 "free_bytes_before_extraction": free, "files": len(entries),
                 "archive_prefix": str(prefix)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"status": "preflight", **preflight}, indent=2) + "\n")
    print(json.dumps(preflight), flush=True)
    if expanded + 2 * 1024**3 > free:
        raise RuntimeError("Insufficient free space for extraction with a 2 GiB reserve")
    checked = []
    for info in entries:
        name = PurePosixPath(info.filename)
        if name.is_absolute() or ".." in name.parts or "\\" in info.filename:
            raise RuntimeError(f"Unsafe archive path: {info.filename}")
        if stat.S_ISLNK(info.external_attr >> 16):
            raise RuntimeError(f"Archive contains a symbolic link: {info.filename}")
        relative = name.relative_to(prefix)
        target = (destination / str(relative)).resolve()
        if not target.is_relative_to(destination):
            raise RuntimeError(f"Archive entry leaves the installation directory: {info.filename}")
        checked.append((info, target))
    destination.mkdir(parents=True, exist_ok=True)
    extracted_bytes = 0
    for index, (info, target) in enumerate(checked, 1):
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("xb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024**2)
        extracted_bytes += info.file_size
        if index % 500 == 0:
            print(f"Extracted {index}/{len(checked)} entries, {extracted_bytes} bytes", flush=True)
with archive_path.open("rb") as stream:
    archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
report = {"status": "extracted_and_crc_verified", **preflight,
          "archive_sha256": archive_hash, "extracted_bytes": extracted_bytes,
          "elapsed_seconds": time.time() - started,
          "launcher": str(destination / "CarlaUnreal.exe"),
          "wheel_files": [str(p) for p in (destination / "PythonAPI/carla/dist").glob("*.whl")]}
args.report.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)

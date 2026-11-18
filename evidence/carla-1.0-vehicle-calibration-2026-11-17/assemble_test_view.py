"""Assemble complete fixed test seeds after a documented RPC interruption."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_FIELDS = (
    "client_version",
    "server_version",
    "map",
    "blueprint",
    "control_dt",
    "script_sha256",
    "counts",
    "spawn_points_used",
    "min_straight_m",
    "max_road_pitch_deg",
    "max_lateral_acceleration_mps2",
    "steer_input_exponent",
)
PROGRAM_FIELDS = (
    "segment",
    "kind",
    "straight_m",
    "road_pitch_deg",
    "level",
    "brake",
    "target_speed",
    "steer_amplitude",
    "steer_frequency",
    "steer_hold_s",
    "steer_input_exponent",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def select_fields(
    metadata: dict[str, Any], fields: tuple[str, ...], source: Path
) -> dict[str, Any]:
    missing = set(fields) - metadata.keys()
    if missing:
        raise ValueError(f"Missing fields {sorted(missing)} in {source}")
    return {field: metadata[field] for field in fields}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_manifest(directory: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not path.resolve().is_relative_to(directory):
            raise ValueError(f"Source contains a linked or external path: {path}")
        if path.is_file():
            files[path.relative_to(directory).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interrupted", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started_utc = datetime.now(timezone.utc).isoformat()
    interrupted = args.interrupted.resolve(strict=True)
    recovery = args.recovery.resolve(strict=True)
    output = args.out.resolve()
    if interrupted.is_relative_to(recovery) or recovery.is_relative_to(interrupted):
        raise ValueError("Interrupted and recovery roots must be separate")
    if any(output.is_relative_to(root) for root in (interrupted, recovery)):
        raise ValueError("The analysis view must be outside both source roots")
    if output.exists() or output.is_symlink():
        raise FileExistsError("The analysis view must be a new directory")

    original_run = read_json(interrupted / "run.json")
    recovery_run = read_json(recovery / "run.json")
    if original_run.get("status") != "failed":
        raise ValueError("The interrupted run must have failed status")
    if recovery_run.get("status") != "completed":
        raise ValueError("The recovery run must have completed status")
    protocol = select_fields(original_run, PROTOCOL_FIELDS, interrupted / "run.json")
    if protocol["client_version"] != protocol["server_version"]:
        raise ValueError("CARLA client and server versions must match")
    if select_fields(recovery_run, PROTOCOL_FIELDS, recovery / "run.json") != protocol:
        raise ValueError("Interrupted and recovery protocols differ")

    logs = {}
    for root, seed, status in (
        (interrupted, 30, "completed"),
        (interrupted, 31, "failed"),
        (recovery, 31, "completed"),
        (recovery, 32, "completed"),
    ):
        directory = root / f"seed{seed}"
        path = directory / "collection.json"
        log = read_json(path)
        if log.get("seed") != seed or log.get("status") != status:
            raise ValueError(f"Seed or completion status differs in {path}")
        if select_fields(log, PROTOCOL_FIELDS, path) != protocol:
            raise ValueError(f"Collection protocol differs in {path}")
        if not isinstance(log.get("segments"), list) or not log["segments"]:
            raise ValueError(f"Collection has no trial log: {path}")
        logs[root, seed] = log
        if status == "completed":
            descriptor = directory / "datapackage.json"
            metadata = read_json(descriptor)["myosuite_carla"]["metadata"]
            if (
                metadata.get("status") != "completed"
                or metadata.get("seed") != seed
                or metadata.get("vehicle") != protocol["blueprint"]
                or select_fields(metadata, PROTOCOL_FIELDS, descriptor) != protocol
            ):
                raise ValueError(f"Recording metadata differs in {descriptor}")

    partial = logs[interrupted, 31]["segments"]
    complete = logs[recovery, 31]["segments"]
    if len(partial) > len(complete):
        raise ValueError("The recovered seed 31 is shorter than its failed prefix")
    for index, (original, recovered) in enumerate(zip(partial, complete)):
        source = interrupted / "seed31" / "collection.json"
        target = recovery / "seed31" / "collection.json"
        if select_fields(original, PROGRAM_FIELDS, source) != select_fields(
            recovered, PROGRAM_FIELDS, target
        ):
            raise ValueError(f"Seed 31 program metadata differ at trial {index}")

    sources = {30: interrupted, 31: recovery, 32: recovery}
    inventories = {
        seed: file_manifest(root / f"seed{seed}") for seed, root in sources.items()
    }
    manifest = {
        "schema_version": 1,
        "started_utc": started_utc,
        "assembler_sha256": sha256(Path(__file__)),
        "protocol": protocol,
        "prefix_comparison": {
            "seed": 31,
            "trials": len(partial),
            "fields": PROGRAM_FIELDS,
            "comparison": "exact",
            "interrupted_collection_sha256": sha256(
                interrupted / "seed31" / "collection.json"
            ),
            "recovery_collection_sha256": sha256(
                recovery / "seed31" / "collection.json"
            ),
        },
        "source_runs": {
            label: {
                "path": str(root),
                "status": run["status"],
                "started_utc": run.get("started_utc"),
                "finished_utc": run.get("finished_utc"),
                "run_json_sha256": sha256(root / "run.json"),
            }
            for label, root, run in (
                ("interrupted", interrupted, original_run),
                ("recovery", recovery, recovery_run),
            )
        },
        "seeds": {},
    }
    output.mkdir(parents=True)
    for seed, root in sources.items():
        source = root / f"seed{seed}"
        destination = output / source.name
        shutil.copytree(source, destination)
        if (
            file_manifest(destination) != inventories[seed]
            or file_manifest(source) != inventories[seed]
        ):
            raise RuntimeError(f"Seed {seed} changed during assembly")
        manifest["seeds"][str(seed)] = {
            "source_directory": str(source),
            "files": inventories[seed],
        }
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    with (output / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Assembled and verified fixed test seeds in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

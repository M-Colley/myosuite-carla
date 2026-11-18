"""Describe excluded seed 21 after publishing the frozen validation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--design", type=Path, default=Path(__file__).with_name("design-v2.json")
    )
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Seed 21 directory containing the recording and collection.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("An existing diagnostic output must be preserved")

    design = read_json(args.design)
    digest = sha256(args.design)
    if (
        design.get("schema_version") != 2
        or design.get("amendment", {}).get("failed_validation_seed") != 21
        or any(
            21 in design[partition]
            for partition in ("train_seeds", "validation_seeds", "test_seeds")
        )
    ):
        raise ValueError("The amended design must exclude seed 21 from all partitions")
    fitted = read_json(args.fit_report)
    selected = fitted.get("selected")
    if (
        fitted.get("design_sha256") != digest
        or selected is None
        or selected not in fitted.get("candidates", {})
        or fitted.get("selection_data") != "training and validation only"
        or not fitted.get("finished_utc")
    ):
        raise ValueError(
            "A completed selection matching the amended design is required"
        )

    from myosuite_carla import Recording
    from myosuite_carla.vehicle_model import (
        VehicleModelParams,
        carla_nissan_patrol,
        evaluate_vehicle_model,
        segments_from_recording,
    )

    scoring_path = (
        Path(__file__).resolve().parents[2] / "examples/evaluate_vehicle_calibration.py"
    )
    scoring = runpy.run_path(str(scoring_path))
    kind, source_score = scoring["kind"], scoring["source_score"]
    parameters = VehicleModelParams.from_dict(
        fitted["candidates"][selected]["parameters"]
    )
    collection_path = args.data / "collection.json"
    log = read_json(collection_path)
    recording = Recording.load(args.data)
    expected = {
        "status": "completed",
        "seed": 21,
        "client_version": design["target"]["carla"],
        "server_version": design["target"]["carla"],
        "map": design["target"]["map"],
        "blueprint": design["target"]["vehicle"],
        "control_dt": design["target"]["dt_s"],
        "counts": design["counts"],
    }
    for source, metadata in (("collection", log), ("recording", recording.metadata)):
        for field, value in expected.items():
            if metadata.get(field) != value:
                raise ValueError(f"Seed 21 {source} {field} differs from the design")
    if recording.metadata.get("vehicle") != design["target"]["vehicle"]:
        raise ValueError("Seed 21 recording vehicle differs from the design")
    attempted_ids = [row["segment"] for row in log["segments"]]
    if len(attempted_ids) != len(set(attempted_ids)) or any(
        not row["segment"].startswith("s21-")
        or kind(row["segment"]) != row["kind"]
        or not isinstance(row["kept"], bool)
        for row in log["segments"]
    ):
        raise ValueError("Invalid or duplicate seed 21 collection identities")
    attempted = Counter(row["kind"] for row in log["segments"])
    if dict(attempted) != design["counts"]:
        raise ValueError("Seed 21 attempted counts differ from the design")
    accepted = [row for row in log["segments"] if row["kept"]]
    if not accepted or any(
        row.get("status") != "completed" or row.get("completed") is not True
        for row in accepted
    ):
        raise ValueError("Accepted seed 21 trials must be completed")
    segments = segments_from_recording(recording)
    if Counter(segment.name for segment in segments) != Counter(
        row["segment"] for row in accepted
    ):
        raise ValueError("Saved segments differ from the exact accepted seed 21 IDs")
    if any(
        not math.isclose(
            segment.dt, design["target"]["dt_s"], rel_tol=1e-3, abs_tol=1e-6
        )
        for segment in segments
    ):
        raise ValueError("Seed 21 segment timestep differs from the design")
    kept = Counter(row["kind"] for row in accepted)
    families = [family for family in design["counts"] if kept[family]]
    report: dict[str, Any] = {
        "role": "descriptive_excluded_seed_diagnostic",
        "seed": 21,
        "selected": selected,
        "design_sha256": digest,
        "fit_report_sha256": sha256(args.fit_report),
        "collection_sha256": sha256(collection_path),
        "scoring_source_sha256": sha256(scoring_path),
        "recording_path": str(args.data.resolve()),
        "use_pitch": False,
        "measured_speed": False,
        "selection_and_acceptance_unchanged": True,
        "coverage": {
            "attempted": dict(attempted),
            "accepted": dict(kept),
            "rejected": len(log["segments"]) - len(accepted),
            "accepted_segment_ids": sorted(row["segment"] for row in accepted),
            "observed_families": families,
            "missing_families": [name for name in design["counts"] if not kept[name]],
        },
    }
    for label, params in (("legacy", carla_nissan_patrol()), ("candidate", parameters)):
        evaluated = evaluate_vehicle_model(
            params, segments, group=kind, use_pitch=False
        )
        report[label] = {
            "parameters": params.as_dict(),
            "report": evaluated,
            "score": source_score(evaluated["segments"], families),
        }
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(f"Saved descriptive diagnostic {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

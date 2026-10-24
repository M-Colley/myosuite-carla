"""Fit the vehicle surrogate to identification drives and report held-out errors.

Training and held-out data are recordings saved by ``DriveRecorder`` (for
example by ``examples/identify_vehicle.py`` or ``validate_fullbody.py
--record``); each continuous run of one ``phase`` label is a segment. Errors
are open-loop rollouts from each segment's initial state with its logged
commands, compared with constant-velocity extrapolation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from myosuite_carla import Recording
from myosuite_carla.vehicle_model import (
    evaluate_vehicle_model,
    fit_vehicle_model,
    segments_from_recording,
)


def _segments(directories: list[Path]) -> list:
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
    ]


def _kind(name: str) -> str:
    return name.rsplit("-", 1)[-1]


def main() -> int:
    """Fit, evaluate and write the surrogate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--test-by-phase",
        type=Path,
        nargs="*",
        default=[],
        help="held-out recordings grouped by phase name instead of manoeuvre",
    )
    parser.add_argument("--wheelbase", type=float, default=3.075)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/vehicle-surrogate.json")
    )
    args = parser.parse_args()
    train = _segments(args.train)
    params, fit = fit_vehicle_model(
        train, wheelbase=args.wheelbase, smoothing=args.smoothing
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": "examples/fit_vehicle_model.py " + " ".join(sys.argv[1:]),
        "train_data": [str(path) for path in args.train],
        "test_data": [str(path) for path in args.test],
        "params": params.as_dict(),
        "fit": fit,
        "train": evaluate_vehicle_model(params, train, group=_kind),
    }
    if args.test:
        report["held_out"] = evaluate_vehicle_model(
            params, _segments(args.test), group=_kind
        )
    if args.test_by_phase:
        report["held_out_phases"] = evaluate_vehicle_model(
            params, _segments(args.test_by_phase), group=lambda name: name
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name in ("train", "held_out", "held_out_phases"):
        if name in report:
            print(name, json.dumps(report[name]["summary"]))
            for group, summary in report[name].get("by_group", {}).items():
                print(f"  {group}", json.dumps(summary))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Drive an actual CARLA vehicle with one muscle-actuated MyoSuite full body.

The seated ``myofullbody`` operates throttle and brake pedals with its feet,
steers with both hands, and uses finger switches for the parking brake,
reverse and sequential manual gears. Evidence is written on success and
failure. Use --target-ref 0.10.0 --build-ref 0.10.0 for the UE5.5 release.
--record DIR also exports every frame (CSV data package, OpenSim tables and an
arm_force_driving recording by default; see myosuite_carla.recording).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myosuite_carla.fullbody import FullBodyDriverConfig
from myosuite_carla.fullbody_validation import run_fullbody_validation
from myosuite_carla.recording import FORMATS, DriveRecorder
from myosuite_carla.validation import LiveValidationError


def main() -> int:
    """Run the full-body drive and write its JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000))
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--blueprint", default=os.environ.get("CARLA_BLUEPRINT"))
    parser.add_argument(
        "--target-ref", default=os.environ.get("CARLA_TARGET_REF", "ue58-dev")
    )
    parser.add_argument("--build-ref", default=os.environ.get("CARLA_BUILD_REF"))
    parser.add_argument("--build-commit", default=os.environ.get("CARLA_BUILD_COMMIT"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/carla-fullbody.json")
    )
    parser.add_argument("--record", type=Path, help="export every frame here")
    parser.add_argument(
        "--formats",
        default="csv,opensim,arm_force",
        help=f"comma-separated subset of {','.join(FORMATS)}",
    )
    parser.add_argument(
        "--fast", action="store_true", help="2 ms physics with pruned locked muscles"
    )
    args = parser.parse_args()
    recorder = DriveRecorder() if args.record else None
    config = FullBodyDriverConfig.fast() if args.fast else FullBodyDriverConfig()
    try:
        report = run_fullbody_validation(
            args.host,
            args.port,
            timeout=args.timeout,
            blueprint=args.blueprint,
            target_ref=args.target_ref,
            build_ref=args.build_ref,
            build_commit=args.build_commit,
            observer=recorder,
            config=config,
        )
    except LiveValidationError as error:
        report = error.report
    if recorder is not None and len(recorder):
        recorder.metadata.update(status=report["status"], target=args.target_ref)
        written = recorder.recording().save(args.record, args.formats.split(","))
        print(f"recorded {len(recorder)} frames: {sorted(written)} in {args.record}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"{report['status'].upper()}: evidence at {args.output.resolve()}")
    for check in report.get("checks", []):
        print(
            f"  [{'x' if check['passed'] else ' '}] {check['name']}: {check['detail']}"
        )
    if "error" in report:
        print(report["error"]["message"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

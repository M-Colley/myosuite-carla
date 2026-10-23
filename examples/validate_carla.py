"""Run actual CARLA acceptance checks and retain measured JSON evidence.

The default target is ue58-dev. For supplementary UE5.5 release testing, use
--target-ref 0.10.0 --build-ref 0.10.0 and keep that evidence separate.
Vehicle selection queries the running server; --blueprint overrides discovery.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myosuite_carla.validation import LiveValidationError, run_live_validation


def main() -> int:
    """Execute real-server checks, writing evidence on both success and failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000))
    )
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument(
        "--blueprint",
        default=os.environ.get("CARLA_BLUEPRINT"),
        help="Vehicle ID; default discovers a four-wheel vehicle on the server",
    )
    parser.add_argument(
        "--target-ref",
        default=os.environ.get("CARLA_TARGET_REF", "ue58-dev"),
        help="Actual validation target, e.g. ue58-dev or 0.10.0 (UE5.5)",
    )
    parser.add_argument("--build-ref", default=os.environ.get("CARLA_BUILD_REF"))
    parser.add_argument("--build-commit", default=os.environ.get("CARLA_BUILD_COMMIT"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/carla-live.json")
    )
    args = parser.parse_args()
    try:
        report = run_live_validation(
            args.host,
            args.port,
            timeout=args.timeout,
            blueprint=args.blueprint,
            target_ref=args.target_ref,
            build_ref=args.build_ref,
            build_commit=args.build_commit,
        )
    except LiveValidationError as error:
        report = error.report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"{report['status'].upper()}: evidence at {args.output.resolve()}")
    if "error" in report:
        print(report["error"]["message"])
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

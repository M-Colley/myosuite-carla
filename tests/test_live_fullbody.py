"""Opt-in full-body drive against an actual CARLA server."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.carla,
    pytest.mark.skipif(
        os.environ.get("CARLA_BRIDGE_LIVE") != "1",
        reason="Set CARLA_BRIDGE_LIVE=1 with a matching actual CARLA server",
    ),
]


def test_one_full_body_operates_every_control_of_a_real_vehicle() -> None:
    from myosuite_carla.fullbody_validation import run_fullbody_validation
    from myosuite_carla.validation import LiveValidationError

    output = Path(
        os.environ.get("CARLA_FULLBODY_OUTPUT", "artifacts/carla-fullbody.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run_fullbody_validation(
            os.environ.get("CARLA_HOST", "localhost"),
            int(os.environ.get("CARLA_PORT", 2000)),
            timeout=float(os.environ.get("CARLA_TIMEOUT", 30)),
            blueprint=os.environ.get("CARLA_BLUEPRINT"),
            target_ref=os.environ.get("CARLA_TARGET_REF", "ue58-dev"),
            build_ref=os.environ.get("CARLA_BUILD_REF"),
            build_commit=os.environ.get("CARLA_BUILD_COMMIT"),
        )
    except LiveValidationError as error:
        output.write_text(json.dumps(error.report), encoding="utf-8")
        raise
    output.write_text(json.dumps(report), encoding="utf-8")
    assert report["status"] == "passed"
    assert len(report["checks"]) == 12
    assert all(check["passed"] for check in report["checks"])
    assert report["settings_restored"] and report["actor_ids_restored"]

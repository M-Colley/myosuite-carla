"""Opt-in verification against the user's matching CARLA UE5.8 build."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from myosuite_carla.validation import LiveValidationError, run_live_validation

pytestmark = [
    pytest.mark.carla,
    pytest.mark.skipif(
        os.environ.get("CARLA_BRIDGE_LIVE") != "1",
        reason="Set CARLA_BRIDGE_LIVE=1 with a matching UE5.8 server",
    ),
]


def test_real_muscles_accelerate_brake_and_turn_a_real_carla_vehicle() -> None:
    output = Path(os.environ.get("CARLA_LIVE_OUTPUT", "artifacts/carla-live.json"))
    try:
        report = run_live_validation(
            os.environ.get("CARLA_HOST", "localhost"),
            int(os.environ.get("CARLA_PORT", 2000)),
            timeout=float(os.environ.get("CARLA_TIMEOUT", 15)),
            blueprint=os.environ.get("CARLA_BLUEPRINT"),
            target_ref=os.environ.get("CARLA_TARGET_REF", "ue58-dev"),
            build_ref=os.environ.get("CARLA_BUILD_REF"),
            build_commit=os.environ.get("CARLA_BUILD_COMMIT"),
        )
    except LiveValidationError as error:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(error.report, indent=2, allow_nan=False), encoding="utf-8"
        )
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    assert report["status"] == "passed"
    assert len(report["scenarios"]) == 3
    assert all(trial["settings_restored"] for trial in report["scenarios"])

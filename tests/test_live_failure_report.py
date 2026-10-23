"""Unavailable CARLA clients must produce failures rather than live-test passes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from myosuite_carla import validation


def test_missing_client_retains_failed_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_client(name: str) -> None:
        raise ModuleNotFoundError("Matching UE5.8 client is unavailable")

    monkeypatch.setattr(validation.importlib, "import_module", unavailable_client)
    with pytest.raises(validation.LiveValidationError) as failure:
        validation.run_live_validation(build_ref="ue58-dev")
    report = failure.value.report
    assert report["status"] == "failed"
    assert report["scenarios"] == []
    assert report["error"]["type"] == "ModuleNotFoundError"
    assert report["provenance"]["source_verified"] is False
    assert report["provenance"]["expected_ref"] == "ue58-dev"
    assert report["provenance"]["asserted_ref_matches_target"] is True
    assert report["finished_utc"] >= report["started_utc"]


def test_control_mismatch_preserves_failed_sample_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, simulation: Any
) -> None:
    driver = SimpleNamespace(
        model=SimpleNamespace(nu=0),
        config=SimpleNamespace(names=SimpleNamespace(throttle_joint="throttle")),
        data=SimpleNamespace(time=0.0),
        measurement=lambda: {"travel": 0.0, "contact_newtons": 0.0},
    )
    monkeypatch.setattr(validation, "build_muscle_driver", lambda control: driver)
    monkeypatch.setattr(
        validation, "_spawn", lambda world, blueprint: simulation.vehicle
    )
    simulation.vehicle.set_autopilot = lambda enabled: None
    simulation.vehicle.get_control = lambda: SimpleNamespace(
        throttle=0, brake=0, steer=0
    )
    simulation.world.wait_for_tick = lambda timeout: None
    evidence: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="frame 101.*requested=.*applied="):
        validation._trial(
            simulation.client,
            simulation.api,
            "throttle",
            "vehicle.test",
            10.0,
            evidence,
        )

    assert len(evidence["samples"]) == 1
    sample = evidence["samples"][0]
    assert sample["phase"] == "settle_scalar_setup"
    assert sample["command"]["brake"] == 1
    assert sample["applied"]["brake"] == 0
    assert sample["vehicle"]["frame"] == 101
    assert sample["physical"] == driver.measurement()
    assert sample["mujoco_time"] == 0.0
    assert evidence["vehicle_destroyed"] is True
    assert evidence["settings_restored"] is True

"""Offline selection and evidence checks; these do not exercise a live server."""

from __future__ import annotations

import json
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from myosuite_carla import validation


@dataclass
class Blueprint:
    id: str
    wheels: int | None = 4

    def has_attribute(self, name: str) -> bool:
        return name == "number_of_wheels" and self.wheels is not None

    def get_attribute(self, name: str) -> str:
        assert name == "number_of_wheels" and self.wheels is not None
        return str(self.wheels)


class Library:
    def __init__(self, blueprints: list[Blueprint]) -> None:
        self.blueprints = blueprints

    def filter(self, pattern: str) -> list[Blueprint]:
        assert pattern == "vehicle.*"
        return [item for item in self.blueprints if item.id.startswith("vehicle.")]

    def find(self, blueprint_id: str) -> Blueprint:
        for item in self.blueprints:
            if item.id == blueprint_id:
                return item
        raise KeyError(blueprint_id)


def world_with(blueprints: list[Blueprint]) -> Any:
    library = Library(blueprints)
    return SimpleNamespace(get_blueprint_library=lambda: library)


def test_release_catalogue_selects_available_patrol_without_tesla() -> None:
    world = world_with(
        [Blueprint("vehicle.dodge.charger"), Blueprint("vehicle.nissan.patrol")]
    )
    assert validation._select_blueprint(world, None) == "vehicle.nissan.patrol"


def test_fallback_is_deterministic_and_excludes_nonvehicle_or_wrong_wheels() -> None:
    candidates = [
        Blueprint("vehicle.z_car"),
        Blueprint("vehicle.bicycle", 2),
        Blueprint("vehicle.unknown", None),
        Blueprint("walker.test"),
        Blueprint("vehicle.a_car"),
    ]
    for order in (candidates, list(reversed(candidates))):
        assert validation._select_blueprint(world_with(order), None) == "vehicle.a_car"


def test_explicit_blueprint_is_honored_over_discovered_default() -> None:
    world = world_with(
        [Blueprint("vehicle.nissan.patrol"), Blueprint("vehicle.mini.cooper")]
    )
    assert (
        validation._select_blueprint(world, "vehicle.mini.cooper")
        == "vehicle.mini.cooper"
    )


@pytest.mark.parametrize("blueprint", ["vehicle.motorbike", "walker.test"])
def test_explicit_invalid_vehicle_fails_instead_of_falling_back(blueprint: str) -> None:
    world = world_with(
        [
            Blueprint("vehicle.nissan.patrol"),
            Blueprint("vehicle.motorbike", 2),
            Blueprint("walker.test"),
        ]
    )
    with pytest.raises(RuntimeError, match="four-wheeled vehicle"):
        validation._select_blueprint(world, blueprint)


def test_explicit_missing_vehicle_fails_instead_of_falling_back() -> None:
    world = world_with([Blueprint("vehicle.nissan.patrol")])
    with pytest.raises(KeyError, match="vehicle.tesla.model3"):
        validation._select_blueprint(world, "vehicle.tesla.model3")


def test_catalogue_without_four_wheels_reports_failure() -> None:
    world = world_with([Blueprint("vehicle.motorbike", 2)])
    with pytest.raises(RuntimeError, match="no four-wheeled"):
        validation._select_blueprint(world, None)


def test_release_report_records_actual_target_and_reuses_selected_vehicle(
    monkeypatch: pytest.MonkeyPatch, simulation: Any
) -> None:
    world = world_with([Blueprint("vehicle.nissan.patrol")])
    world.get_map = lambda: SimpleNamespace(name="Town10HD_Opt")
    simulation.client.world = world
    simulation.client.client_version = simulation.client.server_version = "0.10.0"
    api = SimpleNamespace(Client=lambda host, port: simulation.client)
    monkeypatch.setattr(validation.importlib, "import_module", lambda name: api)
    calls = []

    def record_trial(
        client: Any,
        carla: Any,
        control: str,
        blueprint: str,
        timeout: float,
        evidence: dict[str, Any],
    ) -> None:
        calls.append((control, blueprint))

    monkeypatch.setattr(validation, "_trial", record_trial)
    report = validation.run_live_validation(
        target_ref="0.10.0", build_ref="0.10.0+local"
    )
    assert report["requested_blueprint"] is None
    assert report["selected_blueprint"] == "vehicle.nissan.patrol"
    assert calls == [
        (control, "vehicle.nissan.patrol")
        for control in ("throttle", "brake", "steering")
    ]
    assert report["provenance"] == {
        "expected_ref": "0.10.0",
        "asserted_ref": "0.10.0+local",
        "asserted_ref_matches_target": False,
        "asserted_commit": None,
        "source_verified": False,
    }


def test_cli_release_target_and_discovery_are_retained_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "release.json"
    monkeypatch.delenv("CARLA_BLUEPRINT", raising=False)
    monkeypatch.setenv("CARLA_BUILD_REF", "ue58-dev")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_carla.py",
            "--target-ref",
            "0.10.0",
            "--build-ref",
            "0.10.0",
            "--output",
            str(output),
        ],
    )

    def unavailable_client(name: str) -> None:
        raise ModuleNotFoundError("Offline CLI failure check")

    script = Path(__file__).resolve().parents[1] / "examples" / "validate_carla.py"
    namespace = runpy.run_path(str(script))
    monkeypatch.setattr(validation.importlib, "import_module", unavailable_client)
    assert namespace["main"]() == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["requested_blueprint"] is None
    assert report["provenance"]["expected_ref"] == "0.10.0"
    assert report["provenance"]["asserted_ref"] == "0.10.0"
    assert report["provenance"]["asserted_ref_matches_target"] is True
    assert report["provenance"]["source_verified"] is False


@pytest.mark.parametrize("destroy_raises", [False, True])
def test_trial_retains_primary_error_and_settings_evidence_after_destroy_failure(
    monkeypatch: pytest.MonkeyPatch, destroy_raises: bool
) -> None:
    reads = []
    settings = {
        "synchronous_mode": False,
        "fixed_delta_seconds": None,
        "substepping": True,
        "max_substeps": 10,
        "max_substep_delta_time": 0.01,
        "no_rendering_mode": False,
    }
    world = SimpleNamespace(wait_for_tick=lambda timeout: None)

    def read_settings(current_world: Any) -> dict[str, Any]:
        assert current_world is world
        reads.append(True)
        return dict(settings)

    def destroy() -> bool:
        if destroy_raises:
            raise RuntimeError("destroy RPC failed")
        return False

    vehicle = SimpleNamespace(
        id=1,
        type_id="vehicle.nissan.patrol",
        set_autopilot=lambda value: None,
        destroy=destroy,
    )
    driver = SimpleNamespace(
        model=SimpleNamespace(nu=0),
        config=SimpleNamespace(names=SimpleNamespace(throttle_joint="pedal")),
    )

    class BrokenSession:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> None:
            raise RuntimeError("original trial RPC failed")

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr(validation, "_settings", read_settings)
    monkeypatch.setattr(validation, "_spawn", lambda world, blueprint: vehicle)
    monkeypatch.setattr(validation, "build_muscle_driver", lambda control: driver)
    monkeypatch.setattr(validation, "CarlaSession", BrokenSession)
    evidence: dict[str, Any] = {}
    with pytest.raises(RuntimeError, match="^original trial RPC failed$"):
        validation._trial(
            SimpleNamespace(get_world=lambda: world),
            object(),
            "throttle",
            vehicle.type_id,
            1,
            evidence,
        )
    assert len(reads) == 2
    assert evidence["error"]["message"] == "original trial RPC failed"
    assert evidence["vehicle_destroyed"] is False
    assert evidence["settings_restored"] is True
    assert evidence["settings_after"] == settings
    assert evidence["cleanup_errors"] == [
        "destroy RPC failed"
        if destroy_raises
        else "Could not destroy validation vehicle"
    ]
    assert "passed" not in evidence


def test_successful_measurements_do_not_pass_when_vehicle_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch, simulation: Any
) -> None:
    """A CARLA double supplies movement; real MuJoCo supplies muscle/contact data."""
    world, vehicle = simulation.world, simulation.vehicle
    original_snapshot = world.get_snapshot

    def moving_snapshot() -> Any:
        snapshot = original_snapshot()
        actor = snapshot.find(vehicle.id)
        actor.get_transform = lambda: SimpleNamespace(
            location=SimpleNamespace(x=world.frame * 0.02, y=0, z=0),
            rotation=SimpleNamespace(pitch=0, yaw=0, roll=0),
        )
        actor.get_velocity = lambda: SimpleNamespace(
            x=0
            if vehicle.controls and vehicle.controls[-1].hand_brake
            else world.frame * 0.01,
            y=0,
            z=0,
        )
        return snapshot

    monkeypatch.setattr(world, "get_snapshot", moving_snapshot)
    monkeypatch.setattr(world, "wait_for_tick", lambda timeout: None, raising=False)
    monkeypatch.setattr(vehicle, "set_autopilot", lambda value: None, raising=False)
    monkeypatch.setattr(
        vehicle, "get_control", lambda: vehicle.controls[-1], raising=False
    )
    monkeypatch.setattr(vehicle, "destroy", lambda: False)
    monkeypatch.setattr(validation, "_spawn", lambda world, blueprint: vehicle)
    evidence: dict[str, Any] = {}
    with pytest.raises(
        RuntimeError,
        match="^Validation cleanup failed: Could not destroy validation vehicle$",
    ):
        validation._trial(
            simulation.client,
            simulation.api,
            "throttle",
            vehicle.type_id,
            1,
            evidence,
        )
    assert evidence["end_speed_mps"] > evidence["start_speed_mps"] + 0.8
    assert evidence["displacement_metres"] > 1
    assert evidence["peaks"]["contact_newtons"] > 10
    assert evidence["peaks"]["muscle_newtons"] > 100
    assert evidence["vehicle_destroyed"] is False
    assert evidence["settings_restored"] is True
    assert evidence["cleanup_errors"] == ["Could not destroy validation vehicle"]
    assert "passed" not in evidence


@pytest.mark.parametrize(
    ("speeds", "expected_frames"),
    [
        ([0.0] * 35, 35),
        ([0.4] * 45 + [0.0] * 10, 55),
        ([0.0] * 30 + [0.1] + [0.0] * 10, 41),
    ],
    ids=["minimum_wait", "prolonged_bounce", "interrupted_rest"],
)
def test_settling_requires_consecutive_rest_after_minimum_wait(
    speeds: list[float], expected_frames: int
) -> None:
    remaining = iter(speeds)
    records = []

    def step(command: Any) -> Any:
        assert command.throttle == 0
        assert command.brake == 1
        assert command.hand_brake is True
        return SimpleNamespace(speed=next(remaining))

    frames = validation._settle_vehicle(
        SimpleNamespace(step=step),
        lambda phase, command, telemetry: records.append((phase, telemetry.speed)),
    )
    assert frames == expected_frames
    assert len(records) == expected_frames
    assert all(phase == "settle_scalar_setup" for phase, _ in records)
    assert all(speed < 0.05 for _, speed in records[-10:])


def test_settling_fails_within_five_seconds_if_vehicle_never_rests() -> None:
    records = []
    session = SimpleNamespace(step=lambda command: SimpleNamespace(speed=0.1))
    with pytest.raises(RuntimeError, match="did not settle.*5 s setup limit"):
        validation._settle_vehicle(
            session,
            lambda phase, command, telemetry: records.append(telemetry.speed),
        )
    assert records == [0.1] * 250

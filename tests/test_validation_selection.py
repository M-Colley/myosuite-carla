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

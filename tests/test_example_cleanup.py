"""Example setup failures release acquired simulator resources."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

import myosuite_carla
import myosuite_carla.carla_scenario as carla_scenario
from myosuite_carla.scenario import Lane


def _example(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Config:
    control_dt = 0.02

    @classmethod
    def fast(cls, **kwargs: Any) -> _Config:
        return cls()


def _carla_setup(monkeypatch: pytest.MonkeyPatch, stage: str) -> list[str]:
    events: list[str] = []

    def fail() -> None:
        raise RuntimeError(stage)

    vehicle = SimpleNamespace(destroy=lambda: events.append("vehicle.destroy"))
    sensor = SimpleNamespace(
        stop=lambda: events.append("sensor.stop"),
        destroy=lambda: events.append("sensor.destroy"),
        listen=lambda callback: fail() if stage == "listen" else None,
    )
    spawn = SimpleNamespace(location=object())
    world_map = SimpleNamespace(
        name="test-map",
        get_spawn_points=lambda: [spawn],
        get_waypoint=lambda location: SimpleNamespace(),
    )

    def spawn_actor(blueprint: str, *args: Any, **kwargs: Any) -> Any:
        if blueprint.startswith("sensor."):
            if stage == "sensor_spawn":
                fail()
            return sensor
        return vehicle

    world = SimpleNamespace(
        get_map=lambda: world_map,
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: name),
        try_spawn_actor=lambda *args: vehicle,
        spawn_actor=spawn_actor,
    )
    client = SimpleNamespace(
        set_timeout=lambda timeout: None,
        get_world=lambda: world,
        get_server_version=lambda: fail() if stage == "metadata" else "test",
    )
    monkeypatch.setitem(
        sys.modules,
        "carla",
        SimpleNamespace(Client=lambda *args: client, Transform=lambda: object()),
    )
    monkeypatch.setattr(
        carla_scenario, "lane_from_carla", lambda *args, **kwargs: Lane.straight()
    )

    class Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if stage == "session_init":
                fail()
            assert kwargs["own_vehicle"] is False

        def start(self) -> None:
            events.append("session.start")
            if stage == "session_start":
                fail()

        def close(self) -> None:
            events.append("session.close")

    monkeypatch.setattr(myosuite_carla, "CarlaSession", Session)
    return events


@pytest.mark.parametrize("stage", ["sensor_spawn", "listen", "session_init"])
def test_scenario_setup_failures_destroy_ego_and_sensor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stage: str
) -> None:
    module = _example("drive_scenario")
    events = _carla_setup(monkeypatch, stage)
    monkeypatch.setattr(module, "FullBodyDriverConfig", _Config)
    monkeypatch.setattr(module, "build_fullbody_driver", lambda config: object())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "drive_scenario.py",
            "--backend",
            "carla",
            "--scenario",
            "free",
            "--spawn",
            "0",
            "--output",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(RuntimeError, match=stage):
        module.main()
    expected = ["vehicle.destroy"]
    if stage != "sensor_spawn":
        expected = ["sensor.stop", "sensor.destroy", *expected]
    assert events == expected


@pytest.mark.parametrize(
    "stage", ["session_init", "session_start", "hook", "metadata", "environment"]
)
def test_interface_setup_failures_close_session_and_destroy_ego(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    # Route construction needs no embodied-ui implementation in this test.
    engine = ModuleType("embodied_ui.engine")
    engine.Route = lambda *args: object()  # type: ignore[attr-defined]
    engine.Routes = lambda **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "embodied_ui", ModuleType("embodied_ui"))
    monkeypatch.setitem(sys.modules, "embodied_ui.engine", engine)
    module = _example("drive_with_interface")
    events = _carla_setup(monkeypatch, stage)
    monkeypatch.setattr(module, "FullBodyDriverConfig", _Config)
    monkeypatch.setattr(module, "dashboard", lambda: object())
    monkeypatch.setattr(module, "build_fullbody_driver", lambda config: object())

    def hook(*args: Any) -> None:
        if stage == "hook":
            raise RuntimeError(stage)

    def environment(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("environment")

    monkeypatch.setattr(module, "interface_hook", hook)
    monkeypatch.setattr(module, "FullBodyDrivingEnv", environment)
    monkeypatch.setattr(
        sys, "argv", ["drive_with_interface.py", "--backend", "carla", "--spawn", "0"]
    )
    with pytest.raises(RuntimeError, match=stage):
        module.main()
    assert events[-1] == "vehicle.destroy"
    if stage == "session_init":
        assert events == ["vehicle.destroy"]
    else:
        assert events == ["session.start", "session.close", "vehicle.destroy"]


def test_unity_start_failure_closes_engine_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = ModuleType("embodied_ui.engine")
    engine.Route = lambda *args: object()  # type: ignore[attr-defined]
    engine.Routes = lambda **kwargs: object()  # type: ignore[attr-defined]
    backends = ModuleType("embodied_ui.backends")
    backends.RemoteEngine = lambda *args, **kwargs: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "embodied_ui", ModuleType("embodied_ui"))
    monkeypatch.setitem(sys.modules, "embodied_ui.engine", engine)
    monkeypatch.setitem(sys.modules, "embodied_ui.backends", backends)
    module = _example("drive_with_interface")
    monkeypatch.setattr(module, "FullBodyDriverConfig", _Config)
    monkeypatch.setattr(module, "dashboard", lambda: object())
    monkeypatch.setattr(module, "build_fullbody_driver", lambda config: object())
    events: list[str] = []

    class Session:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("unity start")

        def close(self) -> None:
            events.append("session.close")

    monkeypatch.setattr(module, "EngineVehicleSession", Session)
    monkeypatch.setattr(sys, "argv", ["drive_with_interface.py", "--backend", "unity"])
    with pytest.raises(RuntimeError, match="unity start"):
        module.main()
    assert events == ["session.close"]

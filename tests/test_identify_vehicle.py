"""Designed collection preserves failures and releases acquired CARLA resources."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite_carla.session import VehicleTelemetry


@pytest.fixture
def collector():
    path = Path(__file__).resolve().parents[1] / "examples/identify_vehicle.py"
    spec = importlib.util.spec_from_file_location("identify_vehicle_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("exponent", [1.0, 2.0])
@pytest.mark.parametrize(
    "kind", ["steer", "steer_fast", "steer_low", "steer_hold", "steer_doublet"]
)
def test_steering_programs_bound_requested_lateral_acceleration(
    collector, exponent, kind
):
    for seed in range(30):
        program = collector.Segment(
            kind, np.random.default_rng(seed), 200, steer_exponent=exponent
        )
        angle = 0.92 * abs(program.amplitude) ** exponent
        lateral = program.target**2 * math.tan(angle) / 3.075
        assert 0 < lateral <= 3.0 + 1e-12
        assert abs(program.amplitude) <= 1


def test_linear_doublet_uses_smaller_input_than_squared_reference(collector):
    linear = collector.Segment(
        "steer_doublet", np.random.default_rng(4), 200, steer_exponent=1
    )
    squared = collector.Segment(
        "steer_doublet", np.random.default_rng(4), 200, steer_exponent=2
    )
    assert linear.target == squared.target
    assert abs(linear.amplitude) < abs(squared.amplitude)


@pytest.mark.parametrize("target", [17.0, 19.5, 22.0])
@pytest.mark.parametrize("throttle", [0.75, 1.0])
@pytest.mark.parametrize("brake", [0.4, 0.8])
def test_targeted_launch_reaches_profile_and_stops_within_reserved_road(
    collector, target, throttle, brake
):
    from myosuite_carla.vehicle_model import (
        SurrogateVehicle,
        carla_1_0_nissan_patrol,
    )

    program = collector.Segment(
        "launch",
        np.random.default_rng(0),
        350,
        launch_profile=(target, throttle, brake),
    )
    vehicle = SurrogateVehicle(carla_1_0_nissan_patrol())
    vehicle.start()
    speed = 0.0
    peak = 0.0
    coast_frames = 0
    for _ in range(1500):
        command = program.command(speed, 0.02)
        if command is None:
            break
        coast_frames += program.stage == "coast"
        assert command.steer == 0
        speed = vehicle.step(command).velocity[0]
        peak = max(peak, speed)
    assert program.target_reached and peak >= target
    assert program.incomplete_reason is None
    assert coast_frames >= 75
    assert speed < 0.02
    assert program.frames < 1500 and program.travelled < 350


def test_targeted_launch_rejects_incomplete_speed_when_road_runs_out(collector):
    program = collector.Segment(
        "launch",
        np.random.default_rng(0),
        40,
        launch_profile=(22.0, 0.75, 0.4),
    )
    for _ in range(26):
        program.command(0, 0.02)
    program.command(10, 0.02)
    assert program.stage == "stop"
    assert program.incomplete_reason == "distance_limit"
    assert not program.target_reached


@pytest.mark.parametrize("speed", [17.0, 20.0])
@pytest.mark.parametrize("history", ["acceleration", "braking"])
@pytest.mark.parametrize("action", ["throttle", "brake"])
def test_moving_probe_preserves_origin_and_complete_forecast(
    collector, speed, history, action
):
    from myosuite_carla.vehicle_model import SurrogateVehicle, carla_1_0_nissan_patrol

    program = collector.MovingProbe(
        np.random.default_rng(3), 350, (speed, history, action)
    )
    vehicle = SurrogateVehicle(carla_1_0_nissan_patrol())
    vehicle.start()
    value = 0.0
    states, commands = [], []
    for _ in range(1500):
        command = program.command(value, 0.02)
        states.append(value)
        if command is None:
            break
        commands.append(command)
        value = vehicle.step(command).speed
    assert program.incomplete_reason is None
    assert program.target_reached and program.probe_origin_row is not None
    origin = program.probe_origin_row
    assert abs(states[origin] - speed) <= 0.25
    assert len(states) >= origin + 151
    assert commands[origin].throttle == (0.75 if action == "throttle" else 0)
    assert commands[origin].brake == (0.4 if action == "brake" else 0)
    assert value < 0.02 and program.frames < 1500


def _waypoint(pitch, following=()):
    return SimpleNamespace(
        transform=SimpleNamespace(rotation=SimpleNamespace(yaw=0.0, pitch=pitch)),
        next=lambda _: following,
    )


def test_straight_road_pitch_gate_checks_the_following_span(collector):
    hill = _waypoint(1.2)
    flat = _waypoint(0.0, [hill])
    start = _waypoint(0.0, [flat])
    town = SimpleNamespace(get_waypoint=lambda _: start)
    transform = SimpleNamespace(location=object())
    assert collector._straight(town, transform, max_road_pitch=1.0) == 2.0


@pytest.mark.parametrize("waypoint", [None, _waypoint(-1.1)])
def test_straight_rejects_missing_or_tilted_start(collector, waypoint):
    town = SimpleNamespace(get_waypoint=lambda _: waypoint)
    assert collector._straight(town, SimpleNamespace(location=0)) == 0


def _install_carla(monkeypatch, collector, *, failure=None, no_spawns=False):
    events = []
    vehicle = SimpleNamespace(destroy=lambda: events.append("vehicle_destroy"))

    def listen(_):
        if failure == "listen":
            raise RuntimeError("listen failed")

    sensor = SimpleNamespace(
        listen=listen,
        stop=lambda: events.append("sensor_stop"),
        destroy=lambda: events.append("sensor_destroy"),
    )

    def spawn_sensor(*_, **__):
        if failure == "sensor":
            raise RuntimeError("sensor failed")
        return sensor

    spawn = SimpleNamespace(location=0)
    town = SimpleNamespace(
        name="Town10HD_Opt",
        get_spawn_points=lambda: [] if no_spawns else [spawn],
        get_waypoint=lambda _: _waypoint(0.0),
    )
    world = SimpleNamespace(
        get_settings=lambda: SimpleNamespace(synchronous_mode=False),
        get_map=lambda: town,
        get_blueprint_library=lambda: SimpleNamespace(find=lambda _: object()),
        try_spawn_actor=lambda *_: vehicle,
        spawn_actor=spawn_sensor,
    )
    client = SimpleNamespace(
        set_timeout=lambda _: None,
        get_server_version=lambda: "1.0",
        get_client_version=lambda: "1.0",
        get_world=lambda: world,
    )
    monkeypatch.setitem(
        sys.modules,
        "carla",
        SimpleNamespace(Client=lambda *_: client, Transform=object),
    )
    monkeypatch.setattr(collector, "_straight", lambda *_, **__: 150.0)

    class Session:
        def __init__(self, *_, **kwargs):
            assert kwargs["own_vehicle"] is False

        def start(self):
            if failure == "session":
                raise RuntimeError("session failed")

        def close(self):
            events.append("session_close")

    monkeypatch.setattr(collector, "CarlaSession", Session)
    return events


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        ("sensor", ["vehicle_destroy"]),
        ("listen", ["sensor_destroy", "vehicle_destroy"]),
        (
            "session",
            ["session_close", "sensor_stop", "sensor_destroy", "vehicle_destroy"],
        ),
    ],
)
def test_setup_failures_release_every_acquired_resource(
    collector, monkeypatch, tmp_path, failure, expected
):
    events = _install_carla(monkeypatch, collector, failure=failure)
    destination = tmp_path / "data"
    monkeypatch.setattr(sys, "argv", ["identify_vehicle", "--data", str(destination)])
    with pytest.raises(RuntimeError, match=failure):
        collector.main()
    assert events == expected
    report = json.loads((destination / "run.json").read_text())
    assert report["status"] == "failed"
    assert report["error"]["message"] == f"{failure} failed"


def test_no_eligible_road_fails_before_acquiring_actors(
    collector, monkeypatch, tmp_path
):
    events = _install_carla(monkeypatch, collector, no_spawns=True)
    monkeypatch.setattr(sys, "argv", ["identify_vehicle", "--data", str(tmp_path)])
    with pytest.raises(RuntimeError, match="No spawn point"):
        collector.main()
    assert events == []


def test_existing_evidence_is_rejected_before_carla_import(
    collector, monkeypatch, tmp_path
):
    evidence = tmp_path / "evidence.json"
    evidence.write_text("keep this")
    monkeypatch.delitem(sys.modules, "carla", raising=False)
    monkeypatch.setattr(sys, "argv", ["identify_vehicle", "--data", str(tmp_path)])
    with pytest.raises(SystemExit):
        collector.main()
    assert evidence.read_text() == "keep this"
    assert "carla" not in sys.modules


@pytest.mark.parametrize("fail_step", [None, 76])
def test_rejected_and_failed_attempt_trajectories_are_saved(
    collector, tmp_path, fail_step
):
    class Session:
        steps = 0

        def step(self, _):
            self.steps += 1
            if self.steps == fail_step:
                raise RuntimeError("RPC failed")
            return self.snapshot()

        def snapshot(self):
            return VehicleTelemetry(
                frame=self.steps,
                elapsed_seconds=self.steps * 0.02,
                location=(0.0, 0.0, 0.0),
                rotation=(0.0, 0.0, 4.0 if fail_step is None else 0.0),
                velocity=(0.0, 0.0, 0.0),
                acceleration=(0.0, 0.0, 0.0),
                angular_velocity=(0.0, 0.0, 0.0),
            )

    vehicle = SimpleNamespace(
        set_transform=lambda _: None,
        set_target_velocity=lambda _: None,
        set_target_angular_velocity=lambda _: None,
    )
    arguments = dict(
        session=Session(),
        api=SimpleNamespace(Vector3D=lambda *_: object()),
        vehicle=vehicle,
        spawns=[(object(), 150.0, 0.0)],
        seed=0,
        counts={"launch": 1},
        collisions=[],
        steer_exponent=1.0,
        max_lateral_acceleration=3.0,
        directory=tmp_path,
        metadata={"seed": 0},
    )
    if fail_step is None:
        kept, _ = collector._run_seed(**arguments)
        assert kept == []
    else:
        with pytest.raises(RuntimeError, match="RPC failed"):
            collector._run_seed(**arguments)
    report = json.loads((tmp_path / "collection.json").read_text())
    entry = report["segments"][0]
    assert entry["kept"] is False
    assert entry["status"] == ("rejected" if fail_step is None else "failed")
    assert entry["frames"] == 1
    assert (tmp_path / entry["recording"] / "datapackage.json").exists()

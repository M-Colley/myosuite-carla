"""Opt-in long-duration and competing-client checks against an actual CARLA server."""

from __future__ import annotations

import importlib
import json
import math
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite_carla import (
    CarlaBridge,
    CarlaSession,
    SessionConfig,
    SynchronizationError,
    VehicleCommand,
)
from myosuite_carla.validation import (
    _select_blueprint,
    _settings,
    _settle_vehicle,
    _spawn,
    build_muscle_driver,
)

pytestmark = [
    pytest.mark.carla,
    pytest.mark.skipif(
        os.environ.get("CARLA_BRIDGE_LIVE") != "1",
        reason="Set CARLA_BRIDGE_LIVE=1 with a matching actual CARLA server",
    ),
]


def _timeout() -> float:
    return float(os.environ.get("CARLA_TIMEOUT", 15.0))


def _poll(predicate: Any, timeout: float | None = None) -> Any:
    if timeout is None:
        timeout = _timeout()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise TimeoutError("CARLA client cache did not reach the expected state")


@pytest.fixture(scope="module")
def live_runtime() -> Iterator[tuple[Any, Any, dict[str, Any]]]:
    report: dict[str, Any] = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "target_ref": os.environ.get("CARLA_TARGET_REF", "ue58-dev"),
        "asserted_build_ref": os.environ.get("CARLA_BUILD_REF"),
        "source_verified": False,
        "mujoco_version": importlib.import_module("mujoco").__version__,
        "tests": [],
    }
    try:
        carla = importlib.import_module("carla")
        client = carla.Client(
            os.environ.get("CARLA_HOST", "127.0.0.1"),
            int(os.environ.get("CARLA_PORT", 2000)),
        )
        client.set_timeout(_timeout())
        report.update(
            client_version=client.get_client_version(),
            server_version=client.get_server_version(),
            map=client.get_world().get_map().name,
        )
        assert report["client_version"] == report["server_version"]
        yield carla, client, report
        if report["tests"] and all(
            case["status"] == "passed" for case in report["tests"]
        ):
            report["status"] = "passed"
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        output = Path(
            os.environ.get("CARLA_SESSION_OUTPUT", "artifacts/deep-live-session.json")
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )


@pytest.fixture
def live_case(live_runtime: Any, request: Any) -> tuple[Any, Any, dict[str, Any]]:
    carla, client, report = live_runtime
    case: dict[str, Any] = {"name": request.node.name, "status": "failed"}
    report["tests"].append(case)
    return carla, client, case


@contextmanager
def _owned_session(
    carla: Any, client: Any, case: dict[str, Any]
) -> Iterator[CarlaSession]:
    world = client.get_world()
    original = _settings(world)
    assert not original["synchronous_mode"], "Stop any existing CARLA ticking owner"
    before = {actor.id for actor in world.get_actors()}
    case["settings_before"] = original
    vehicle = None
    session = None
    primary_error: BaseException | None = None
    try:
        blueprint = _select_blueprint(world, os.environ.get("CARLA_BLUEPRINT"))
        vehicle = _spawn(world, blueprint)
        case.update(blueprint=blueprint, vehicle_id=vehicle.id)
        vehicle.set_autopilot(False)
        _poll(lambda: world.get_snapshot().find(vehicle.id))
        session = CarlaSession(
            client,
            vehicle,
            SessionConfig(timeout_seconds=_timeout(), no_rendering_mode=True),
            api=carla,
            own_vehicle=True,
        )
        session.start()
        case["settling_frames"] = _settle_vehicle(session, lambda *args: None)
        yield session
    except BaseException as error:
        primary_error = error
        case["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        errors: list[str] = []
        try:
            if session is not None:
                session.close()
            elif vehicle is not None and vehicle.is_alive:
                assert vehicle.destroy(), "Failed to destroy setup vehicle"
        except Exception as error:
            errors.append(str(error))
        try:
            if vehicle is not None:
                _poll(lambda: world.get_snapshot().find(vehicle.id) is None)
                assert not vehicle.is_alive
            case["vehicle_destroyed"] = True
            case["settings_after"] = _settings(world)
            case["settings_restored"] = case["settings_after"] == original
            assert case["settings_restored"], "CARLA settings were not restored"
            after = {actor.id for actor in world.get_actors()}
            case["actor_ids_restored"] = before == after
            assert case["actor_ids_restored"], "CARLA actor IDs differ after cleanup"
        except Exception as error:
            errors.append(str(error))
        if errors:
            case["cleanup_errors"] = errors
            if primary_error is None:
                raise RuntimeError("Actual-server cleanup failed: " + "; ".join(errors))


def test_1200_real_muscle_frames_remain_in_lockstep(live_case: Any) -> None:
    carla, client, case = live_case
    driver = build_muscle_driver("throttle")
    total_frames = 1200
    case.update(
        frames=total_frames,
        excitation=0.015,
        control_dt=0.02,
        muscles=[driver.model.actuator(index).name for index in range(driver.model.nu)],
        sampled_frames=[],
    )
    with _owned_session(carla, client, case) as session:
        bridge = CarlaBridge(driver.mapper, session, 0.02)
        initial = bridge.reset(driver.data)
        last = initial.telemetry
        durations: list[float] = []
        max_drift = max_control_error = 0.0
        started = time.perf_counter()
        for index in range(total_frames):
            frame_started = time.perf_counter()
            driver.advance(0.015)
            frame = bridge.step(driver.data)
            telemetry, command = frame.telemetry, frame.sample.command
            assert telemetry.frame == last.frame + 1
            assert telemetry.frame == initial.telemetry.frame + index + 1
            assert math.isclose(
                telemetry.elapsed_seconds - last.elapsed_seconds,
                0.02,
                rel_tol=1e-6,
                abs_tol=2e-8,
            )
            drift = (
                telemetry.elapsed_seconds
                - initial.telemetry.elapsed_seconds
                - frame.simulation_time
                + initial.simulation_time
            )
            max_drift = max(max_drift, abs(drift))
            assert abs(drift) < 2e-6
            assert all(
                math.isfinite(value)
                for value in (
                    telemetry.elapsed_seconds,
                    telemetry.speed,
                    *telemetry.location,
                    *telemetry.rotation,
                    *telemetry.velocity,
                    *telemetry.acceleration,
                    *telemetry.angular_velocity,
                )
            )
            for state in (
                driver.data.qpos,
                driver.data.qvel,
                driver.data.act,
                driver.data.ctrl,
                driver.data.sensordata,
                driver.data.actuator_force,
            ):
                assert np.all(np.isfinite(state))
            applied = session.vehicle.get_control()
            for channel in ("throttle", "steer", "brake"):
                error = abs(
                    float(getattr(applied, channel)) - getattr(command, channel)
                )
                max_control_error = max(max_control_error, error)
                assert error < 1e-5
            for channel in ("hand_brake", "reverse", "manual_gear_shift"):
                assert getattr(applied, channel) == getattr(command, channel)
            # With automatic shifting, get_control().gear reports the gearbox state.
            if command.manual_gear_shift:
                assert applied.gear == command.gear
            durations.append(time.perf_counter() - frame_started)
            if index % 100 == 0 or index == total_frames - 1:
                case["sampled_frames"].append(
                    {
                        "frame": telemetry.frame,
                        "mujoco_time": frame.simulation_time,
                        "carla_time": telemetry.elapsed_seconds,
                        "requested": command.as_dict(),
                        "applied": {
                            name: getattr(applied, name) for name in command.as_dict()
                        },
                        "speed_mps": telemetry.speed,
                        "clock_drift_seconds": drift,
                    }
                )
            last = telemetry
        wall_seconds = time.perf_counter() - started
        assert driver.peaks["contact_newtons"] > 10
        assert driver.peaks["muscle_newtons"] > 100
        assert driver.peaks["travel"] > 0.045
        case.update(
            start_frame=initial.telemetry.frame,
            final_frame=last.frame,
            simulated_seconds=total_frames * 0.02,
            wall_seconds=wall_seconds,
            achieved_frames_per_wall_second=total_frames / wall_seconds,
            frame_wall_seconds={
                "median": float(np.median(durations)),
                "p95": float(np.percentile(durations, 95)),
                "max": max(durations),
            },
            max_clock_drift_seconds=max_drift,
            max_applied_control_error=max_control_error,
            physical_peaks=driver.peaks,
            displacement_metres=math.dist(initial.telemetry.location, last.location),
            final_speed_mps=last.speed,
        )
    case["status"] = "passed"


def test_second_client_tick_invalidates_public_snapshot_and_brakes(
    live_case: Any,
) -> None:
    carla, client, case = live_case
    second = carla.Client(
        os.environ.get("CARLA_HOST", "127.0.0.1"),
        int(os.environ.get("CARLA_PORT", 2000)),
    )
    second.set_timeout(_timeout())
    with _owned_session(carla, client, case) as session:
        before = session.step(VehicleCommand(throttle=0.4))
        assert session.vehicle.get_control().throttle == pytest.approx(0.4, abs=1e-6)
        external_frame = second.get_world().tick(_timeout())
        assert external_frame == before.frame + 1
        _poll(lambda: session.world.get_snapshot().frame == external_frame)
        with pytest.raises(SynchronizationError, match="Another client") as captured:
            session.snapshot()
        assert not session.active
        with pytest.raises(RuntimeError, match="healthy"):
            session.step(VehicleCommand(throttle=0.4))
        assert session.world.get_snapshot().frame == external_frame
        # A new snapshot is needed to observe controls cached by CARLA's client.
        verification_frame = second.get_world().tick(_timeout())
        assert verification_frame == external_frame + 1
        _poll(lambda: session.world.get_snapshot().frame == verification_frame)
        applied = session.vehicle.get_control()
        assert applied.throttle == 0.0
        assert applied.brake == 1.0
        assert applied.hand_brake
        case.update(
            original_frame=before.frame,
            external_frame=external_frame,
            verification_frame=verification_frame,
            detected_error=str(captured.value),
            session_invalidated=True,
            emergency_control={
                "throttle": applied.throttle,
                "brake": applied.brake,
                "hand_brake": applied.hand_brake,
            },
        )
    case["status"] = "passed"

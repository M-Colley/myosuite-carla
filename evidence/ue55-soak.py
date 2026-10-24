"""Long lockstep soak: real MyoSuite throttle muscles drive CARLA in bursts.

Supplementary evidence script. Each 6 s cycle applies 2 s of flexor excitation
to the throttle fixture, then releases it for 4 s while a scalar 0.5 brake
binding stops the car, so the vehicle stays near its spawn road. Flexors alone
leave the passive finger curled and drifting sideways after a few presses, so
both abductors are co-activated at 0.05 throughout and the extensor at 0.02
during release; this returns the finger to a repeatable press. Every frame
checks one-frame lockstep, the 0.02 s interval, clock drift, finite state and
requested/applied controls. Server and client private memory are sampled.

CARLA's elapsed time lags MuJoCo's by n * (0.02 - float32(0.02)) after n frames,
consistent with the server advancing by the single-precision step. The drift
check therefore bounds the residual after removing that systematic term.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import psutil

from myosuite_carla import CarlaBridge, CarlaSession, SessionConfig
from myosuite_carla.inputs import Binding, CallbackSource, ControlMapper
from myosuite_carla.validation import (
    _select_blueprint,
    _settings,
    _settle_vehicle,
    _spawn,
    build_muscle_driver,
)


def _advance(driver: Any, flexors: float, extensor: float, abductors: float) -> None:
    """Advance 20 ms with muscle excitation only, mirroring MuscleDriver.advance."""
    if np.any(driver.data.qfrc_applied) or np.any(driver.data.xfrc_applied):
        raise RuntimeError("External forces are forbidden in muscle/contact soak")
    excitation = {
        "mflx": flexors,
        "dflx": flexors,
        "extn": extensor,
        "adabR": abductors,
        "adabL": abductors,
    }
    driver.data.ctrl[:] = 0
    for name, value in excitation.items():
        driver.data.ctrl[driver.model.actuator(name).id] = value
    for _ in range(40):
        mujoco.mj_step(driver.model, driver.data)
        mujoco.mj_forward(driver.model, driver.data)
        for name, value in driver.measurement().items():
            driver.peaks[name] = max(driver.peaks[name], abs(value))


def _server_private_bytes() -> int | None:
    sizes = [
        process.memory_info().private
        for process in psutil.process_iter(["name"])
        if (process.info["name"] or "").startswith("CarlaUnreal-Win64")
    ]
    return max(sizes) if sizes else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--frames", type=int, default=15000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    carla = importlib.import_module("carla")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original = _settings(world)
    if original["synchronous_mode"]:
        raise RuntimeError("Stop any existing CARLA ticking owner")
    before = {actor.id for actor in world.get_actors()}
    report: dict[str, Any] = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "map": world.get_map().name,
        "mujoco_version": mujoco.__version__,
        "frames": args.frames,
        "cycle": {
            "excite_frames": 100,
            "release_frames": 200,
            "flexor_excitation": 0.015,
            "release_extensor_excitation": 0.02,
            "abductor_coactivation": 0.05,
        },
        "bursts_detail": [],
        "release_brake": 0.5,
        "memory_samples": [],
        "settings_before": original,
    }
    driver = build_muscle_driver("throttle")
    phase = {"release": False}
    driver.mapper = ControlMapper(
        driver.model,
        [
            *driver.mapper.bindings,
            Binding(
                "brake",
                CallbackSource(lambda model, data: 0.5 if phase["release"] else 0.0),
            ),
        ],
    )
    blueprint = _select_blueprint(world, None)
    vehicle = _spawn(world, blueprint)
    report.update(blueprint=blueprint, vehicle_id=vehicle.id)
    process = psutil.Process()
    session = CarlaSession(
        client,
        vehicle,
        SessionConfig(timeout_seconds=args.timeout),
        api=carla,
        own_vehicle=True,
    )
    try:
        vehicle.set_autopilot(False)
        session.start()
        report["settling_frames"] = _settle_vehicle(session, lambda *a: None)
        bridge = CarlaBridge(driver.mapper, session, 0.02)
        initial = bridge.reset(driver.data)
        last = initial.telemetry
        durations: list[float] = []
        float32_step_error = 0.02 - float(np.float32(0.02))
        max_drift = max_residual = max_error = max_speed = 0.0
        bursts_moving = 0
        burst_start_speed = 0.0
        started = time.perf_counter()
        for index in range(args.frames):
            position = index % 300
            phase["release"] = position >= 100
            frame_started = time.perf_counter()
            if phase["release"]:
                _advance(driver, 0.0, 0.02, 0.05)
            else:
                _advance(driver, 0.015, 0.0, 0.05)
            frame = bridge.step(driver.data)
            telemetry, command = frame.telemetry, frame.sample.command
            if position == 0:
                burst_throttle: list[float] = []
                burst_contact = 0.0
            if position < 100:
                burst_throttle.append(command.throttle)
                burst_contact = max(
                    burst_contact, driver.measurement()["contact_newtons"]
                )
            if telemetry.frame != last.frame + 1:
                raise RuntimeError(f"Frame gap at step {index}")
            if not math.isclose(
                telemetry.elapsed_seconds - last.elapsed_seconds,
                0.02,
                rel_tol=1e-6,
                abs_tol=2e-8,
            ):
                raise RuntimeError(f"Interval error at step {index}")
            drift = (
                telemetry.elapsed_seconds
                - initial.telemetry.elapsed_seconds
                - frame.simulation_time
                + initial.simulation_time
            )
            max_drift = max(max_drift, abs(drift))
            residual = drift + (index + 1) * float32_step_error
            max_residual = max(max_residual, abs(residual))
            if abs(residual) >= 1e-8:
                raise RuntimeError(f"Clock residual {residual} at step {index}")
            values = (
                telemetry.speed,
                *telemetry.location,
                *telemetry.velocity,
                *telemetry.angular_velocity,
            )
            if not all(math.isfinite(v) for v in values) or not np.all(
                np.isfinite(driver.data.qpos)
            ):
                raise RuntimeError(f"Nonfinite state at step {index}")
            applied = session.vehicle.get_control()
            for channel in ("throttle", "steer", "brake"):
                error = abs(
                    float(getattr(applied, channel)) - getattr(command, channel)
                )
                max_error = max(max_error, error)
                if error >= 1e-5:
                    raise RuntimeError(f"{channel} not applied at step {index}")
            max_speed = max(max_speed, telemetry.speed)
            if position == 0:
                burst_start_speed = telemetry.speed
            if position == 99:
                bursts_moving += telemetry.speed > burst_start_speed + 0.5
                report["bursts_detail"].append(
                    {
                        "burst": index // 300,
                        "mean_throttle": float(np.mean(burst_throttle)),
                        "max_contact_newtons": burst_contact,
                        "speed_gain_mps": telemetry.speed - burst_start_speed,
                    }
                )
            if position == 299 and command.throttle != 0:
                raise RuntimeError(f"Throttle not released before step {index + 1}")
            durations.append(time.perf_counter() - frame_started)
            if index % 1000 == 0 or index == args.frames - 1:
                report["memory_samples"].append(
                    {
                        "step": index,
                        "frame": telemetry.frame,
                        "server_private_bytes": _server_private_bytes(),
                        "client_private_bytes": process.memory_info().private,
                        "speed_mps": telemetry.speed,
                        "clock_drift_seconds": drift,
                    }
                )
            last = telemetry
        wall = time.perf_counter() - started
        report.update(
            start_frame=initial.telemetry.frame,
            final_frame=last.frame,
            simulated_seconds=args.frames * 0.02,
            wall_seconds=wall,
            achieved_frames_per_wall_second=args.frames / wall,
            frame_wall_seconds={
                "median": float(np.median(durations)),
                "p95": float(np.percentile(durations, 95)),
                "p99": float(np.percentile(durations, 99)),
                "max": max(durations),
            },
            max_clock_drift_seconds=max_drift,
            final_clock_drift_seconds=drift,
            float32_step_error_seconds=float32_step_error,
            max_clock_residual_seconds=max_residual,
            max_applied_control_error=max_error,
            max_speed_mps=max_speed,
            bursts=args.frames // 300,
            bursts_accelerating_over_half_mps=bursts_moving,
            physical_peaks=driver.peaks,
            displacement_metres=math.dist(initial.telemetry.location, last.location),
        )
        report["status"] = "passed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        session.close()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline and world.get_snapshot().find(vehicle.id):
            time.sleep(0.01)
        report["vehicle_destroyed"] = not vehicle.is_alive
        report["settings_after"] = _settings(world)
        report["settings_restored"] = report["settings_after"] == original
        report["actor_ids_restored"] = {a.id for a in world.get_actors()} == before
        if not (
            report["vehicle_destroyed"]
            and report["settings_restored"]
            and report["actor_ids_restored"]
        ):
            report["status"] = "failed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report["status"], report.get("error", ""))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

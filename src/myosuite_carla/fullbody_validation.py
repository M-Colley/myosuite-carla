"""Live CARLA drive in which one MyoSuite full body operates every control."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import mujoco

from .bridge import BridgeFrame, CarlaBridge
from .fullbody import (
    DriverIntent,
    FullBodyDriver,
    FullBodyDriverConfig,
    build_fullbody_driver,
    summary,
)
from .session import CarlaSession, SessionConfig
from .validation import (
    LiveValidationError,
    _require,
    _select_blueprint,
    _settings,
    _spawn,
    _yaw_delta,
)
from .vehicle_loads import VehicleMotion


@dataclass(frozen=True)
class DrivePhase:
    """A scripted motor intent held for a number of synchronous frames.

    Args:
        name: Phase label used in evidence and checks.
        intent: Muscle excitations and reflex offset for every frame.
        frames: Maximum number of 20 ms frames.
        until_stopped: End early once the car rests for 10 frames.
    """

    name: str
    intent: DriverIntent
    frames: int
    until_stopped: bool = False


def fullbody_scenario() -> tuple[DrivePhase, ...]:
    """Return the continuous muscle-driven drive used for live validation.

    Parking brake and reverse are latched switches: one brief press engages
    them and another releases them, so no digit holds a button for long.
    """
    hold = DriverIntent(left_foot=0.15)
    pull = 0.06
    return (
        DrivePhase("hold", hold, 250, until_stopped=True),
        DrivePhase(
            "engage_handbrake", DriverIntent(left_foot=0.15, park_finger=pull), 20
        ),
        DrivePhase("release_park_switch", hold, 15),
        DrivePhase("handbrake_hold_throttle", DriverIntent(right_foot=0.012), 100),
        DrivePhase("release_handbrake", DriverIntent(park_finger=pull), 25),
        DrivePhase("throttle", DriverIntent(right_foot=0.008), 150),
        DrivePhase("steer_left", DriverIntent(right_foot=0.008, steer=-0.14), 100),
        DrivePhase("steer_right", DriverIntent(right_foot=0.008, steer=0.14), 50),
        DrivePhase("straighten", DriverIntent(right_foot=0.004), 25),
        DrivePhase("brake", DriverIntent(left_foot=0.15), 250, until_stopped=True),
        DrivePhase(
            "select_reverse", DriverIntent(left_foot=0.15, reverse_finger=0.05), 25
        ),
        DrivePhase("reverse_throttle", DriverIntent(right_foot=0.012), 125),
        DrivePhase("reverse_brake", hold, 250, until_stopped=True),
        DrivePhase(
            "leave_reverse", DriverIntent(left_foot=0.15, reverse_finger=0.05), 25
        ),
        DrivePhase("shift_1", DriverIntent(left_foot=0.15, gear_finger=pull), 15),
        DrivePhase("release_paddle_1", DriverIntent(left_foot=0.15), 25),
        DrivePhase("gear_1_throttle", DriverIntent(right_foot=0.012), 75),
        DrivePhase("shift_2", DriverIntent(right_foot=0.012, gear_finger=pull), 15),
        DrivePhase("gear_2_throttle", DriverIntent(right_foot=0.012), 75),
        DrivePhase("shift_3", DriverIntent(right_foot=0.012, gear_finger=pull), 15),
        DrivePhase("gear_3_throttle", DriverIntent(right_foot=0.012), 75),
        DrivePhase("shift_automatic", DriverIntent(gear_finger=pull), 15),
        DrivePhase("coast", DriverIntent(), 25),
        DrivePhase("pull_handbrake", DriverIntent(park_finger=pull), 20),
        DrivePhase("handbrake_stop", DriverIntent(), 250, until_stopped=True),
        DrivePhase("park", hold, 25),
    )


class DriveObserver(Protocol):
    """Optional hooks, e.g. for recording synchronized camera images."""

    def start(self, session: CarlaSession, driver: FullBodyDriver) -> None:
        """Called after the session starts and the driver is anchored."""

    def frame(self, record: dict[str, Any], driver: FullBodyDriver) -> None:
        """Called after every verified frame."""

    def close(self) -> None:
        """Called before the session restores CARLA."""


_BOOLEANS = ("hand_brake", "reverse", "manual_gear_shift")
#: Frames CARLA's gearbox may take to report a commanded reverse or gear.
SHIFT_LAG_FRAMES = 25


def _heading(rotation: tuple[float, float, float]) -> tuple[float, float]:
    yaw = math.radians(rotation[1])
    return math.cos(yaw), math.sin(yaw)


def _phase_summary(phase: DrivePhase, frames: list[dict[str, Any]]) -> dict[str, Any]:
    start, end = frames[0]["vehicle"], frames[-1]["vehicle"]
    heading = _heading(frames[0]["start_rotation"])
    dx = end["location"][0] - frames[0]["start_location"][0]
    dy = end["location"][1] - frames[0]["start_location"][1]
    commands = [frame["command"] for frame in frames]
    applied = [frame["applied"] for frame in frames]
    return {
        "name": phase.name,
        "intent": vars(phase.intent),
        "frames": len(frames),
        "start_speed_mps": frames[0]["start_speed_mps"],
        "end_speed_mps": end["speed"],
        "signed_displacement_metres": dx * heading[0] + dy * heading[1],
        "yaw_change_degrees": _yaw_delta(
            frames[0]["start_rotation"][1], end["rotation"][1]
        ),
        "max_command": {
            name: max(abs(float(command[name])) for command in commands)
            for name in ("throttle", "brake", "steer")
        },
        "min_steer": min(float(command["steer"]) for command in commands),
        "max_steer": max(float(command["steer"]) for command in commands),
        "applied_true_frames": {
            name: sum(bool(control[name]) for control in applied) for name in _BOOLEANS
        },
        "applied_gears": sorted({int(control["gear"]) for control in applied}),
        "end_applied": applied[-1],
        "end_inputs": frames[-1]["inputs"],
        "peak_contact_newtons": {
            key: max(frame["mechanics"][key] for frame in frames)
            for key in frames[0]["mechanics"]
            if key.endswith("_contact_newtons")
        },
        "peak_muscle_newtons": max(
            frame["mechanics"]["largest_muscle_newtons"] for frame in frames
        ),
        "start_vehicle": start,
    }


def _checks(phases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    def check(name: str, passed: bool, detail: str) -> dict[str, Any]:
        return {"name": name, "passed": bool(passed), "detail": detail}

    throttle = phases["throttle"]
    held = phases["handbrake_hold_throttle"]
    left, right = phases["steer_left"], phases["steer_right"]
    reverse = phases["reverse_throttle"]
    pull, stop = phases["pull_handbrake"], phases["handbrake_stop"]
    released = phases["release_handbrake"]
    results = [
        check(
            "foot throttle accelerates",
            throttle["end_speed_mps"] > throttle["start_speed_mps"] + 0.8
            and throttle["signed_displacement_metres"] > 1.0,
            f"{throttle['start_speed_mps']:.2f} -> {throttle['end_speed_mps']:.2f} "
            f"m/s over {throttle['signed_displacement_metres']:.2f} m",
        ),
        check(
            "parking brake resists foot throttle",
            held["applied_true_frames"]["hand_brake"] == held["frames"]
            and held["max_command"]["throttle"] > 0.3
            and abs(held["signed_displacement_metres"])
            < 0.5 * throttle["signed_displacement_metres"],
            f"{held['signed_displacement_metres']:.2f} m with parking brake vs "
            f"{throttle['signed_displacement_metres']:.2f} m released",
        ),
        check(
            "second switch press releases the parking brake",
            not released["end_applied"]["hand_brake"]
            and released["applied_true_frames"]["hand_brake"] < released["frames"],
            "hand_brake after the second press",
        ),
        check(
            "push-pull hands steer left under foot throttle",
            left["yaw_change_degrees"] < -3
            and left["min_steer"] < -0.2
            and left["max_command"]["throttle"] > 0.2,
            f"{left['yaw_change_degrees']:.1f} deg, steer {left['min_steer']:.2f}",
        ),
        check(
            "push-pull hands steer right under foot throttle",
            right["yaw_change_degrees"] > 3
            and right["max_steer"] > 0.2
            and right["max_command"]["throttle"] > 0.2,
            f"{right['yaw_change_degrees']:.1f} deg, steer {right['max_steer']:.2f}",
        ),
        check(
            "foot brake stops the car",
            phases["brake"]["end_speed_mps"] < 0.1
            and phases["brake"]["max_command"]["brake"] > 0.5,
            f"{phases['brake']['start_speed_mps']:.2f} -> "
            f"{phases['brake']['end_speed_mps']:.3f} m/s",
        ),
        check(
            "reverse button press drives backwards",
            reverse["applied_true_frames"]["reverse"] == reverse["frames"]
            and reverse["signed_displacement_metres"] < -0.5,
            f"{reverse['signed_displacement_metres']:.2f} m along the heading",
        ),
        check(
            "second button press leaves reverse",
            not phases["leave_reverse"]["end_applied"]["reverse"],
            "reverse flag after the second press",
        ),
    ]
    for gear in (1, 2, 3):
        phase = phases[f"gear_{gear}_throttle"]
        results.append(
            check(
                f"paddle selects manual gear {gear}",
                phase["end_applied"]["manual_gear_shift"]
                and phase["end_applied"]["gear"] == gear,
                f"applied gears {phase['applied_gears']}",
            )
        )
    results += [
        check(
            "paddle returns to automatic",
            not phases["coast"]["end_applied"]["manual_gear_shift"],
            "manual_gear_shift after the fourth pull",
        ),
        check(
            "parking brake stops the rolling car",
            stop["end_applied"]["hand_brake"]
            and pull["max_command"]["brake"] == stop["max_command"]["brake"] == 0.0
            and pull["start_speed_mps"] > 1.0
            and stop["end_speed_mps"] < 0.35 * pull["start_speed_mps"],
            f"{pull['start_speed_mps']:.2f} -> {stop['end_speed_mps']:.2f} m/s "
            "without the brake pedal",
        ),
    ]
    return results


def run_fullbody_validation(
    host: str = "localhost",
    port: int = 2000,
    *,
    timeout: float = 30,
    blueprint: str | None = None,
    target_ref: str = "ue58-dev",
    build_ref: str | None = None,
    build_commit: str | None = None,
    observer: DriveObserver | None = None,
    config: FullBodyDriverConfig | None = None,
) -> dict[str, Any]:
    """Drive a real CARLA vehicle with one muscle-actuated MyoSuite full body.

    Every frame advances the body by one control period, measures the cockpit
    and finger mechanisms, applies the measured command and ticks CARLA once.
    Applied controls are verified each frame; phase outcomes are then checked
    against physical criteria. Build declarations are recorded, not verified.

    Args:
        observer: Optional hooks, e.g. a ``DriveRecorder``.
        config: Driver configuration; defaults to ``FullBodyDriverConfig()``.

    Raises:
        LiveValidationError: Validation failed; the exception retains evidence.
    """
    report: dict[str, Any] = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "port": port,
        "mujoco_version": mujoco.__version__,
        "provenance": {
            "expected_ref": target_ref,
            "asserted_ref": build_ref,
            "asserted_ref_matches_target": (
                None if build_ref is None else build_ref == target_ref
            ),
            "asserted_commit": build_commit,
            "source_verified": False,
        },
        "phases": [],
        "frames": [],
    }
    session = None
    try:
        carla = importlib.import_module("carla")
        try:
            report["carla_distribution"] = importlib.metadata.version("carla")
        except importlib.metadata.PackageNotFoundError:
            report["carla_distribution"] = None
        client = carla.Client(host, port)
        client.set_timeout(timeout)
        report["client_version"] = client.get_client_version()
        report["server_version"] = client.get_server_version()
        _require(
            report["client_version"] == report["server_version"],
            "CARLA client/server versions differ",
        )
        world = client.get_world()
        report["map"] = world.get_map().name
        original = _settings(world)
        _require(not original["synchronous_mode"], "Stop the existing ticking owner")
        report["settings_before"] = original
        before = {actor.id for actor in world.get_actors()}
        driver = build_fullbody_driver(config)
        report["driver"] = summary(driver)
        selected = _select_blueprint(world, blueprint)
        vehicle = _spawn(world, selected)
        report.update(selected_blueprint=selected, vehicle_id=vehicle.id)
        vehicle.set_autopilot(False)
        world.wait_for_tick(timeout)
        session = CarlaSession(
            client,
            vehicle,
            SessionConfig(
                fixed_delta_seconds=driver.config.control_dt, timeout_seconds=timeout
            ),
            api=carla,
            own_vehicle=True,
        )
        session.start()
        bridge = CarlaBridge(driver.mapper, session, driver.config.control_dt)
        driver.reset()
        anchor = bridge.reset(driver.data)
        if observer is not None:
            observer.start(session, driver)
        last = anchor.telemetry
        phases: dict[str, dict[str, Any]] = {}
        lag = {"reverse": 0, "gear": 0}
        report["max_shift_lag_frames"] = dict(lag)
        for phase in fullbody_scenario():
            records: list[dict[str, Any]] = []
            resting = 0
            phase_start = last
            for index in range(phase.frames):
                driver.advance(phase.intent)
                frame: BridgeFrame = bridge.step(driver.data)
                telemetry, command = frame.telemetry, frame.sample.command
                # The vehicle's measured motion loads the body for the next period.
                driver.set_vehicle_motion(
                    VehicleMotion.from_telemetry(
                        telemetry, last, driver.config.control_dt
                    )
                )
                _require(telemetry.frame == last.frame + 1, "CARLA frame gap")
                control = vehicle.get_control()
                applied = {
                    name: getattr(control, name) for name in (*command.as_dict(),)
                }
                for name in ("throttle", "steer", "brake"):
                    _require(
                        math.isclose(
                            float(applied[name]), getattr(command, name), abs_tol=1e-5
                        ),
                        f"CARLA did not apply measured {name} at frame "
                        f"{telemetry.frame}",
                    )
                for name in ("hand_brake", "manual_gear_shift"):
                    _require(
                        bool(applied[name]) == getattr(command, name),
                        f"CARLA did not apply measured {name} at frame "
                        f"{telemetry.frame}",
                    )
                # get_control() reports the gearbox, which completes reverse and
                # manual gear changes after a shift delay; bound that lag.
                for name, matches in (
                    ("reverse", bool(applied["reverse"]) == command.reverse),
                    (
                        "gear",
                        not command.manual_gear_shift
                        or int(applied["gear"]) == command.gear,
                    ),
                ):
                    lag[name] = 0 if matches else lag[name] + 1
                    report["max_shift_lag_frames"][name] = max(
                        report["max_shift_lag_frames"][name], lag[name]
                    )
                    _require(
                        lag[name] <= SHIFT_LAG_FRAMES,
                        f"CARLA did not apply measured {name} within "
                        f"{SHIFT_LAG_FRAMES} frames (frame {telemetry.frame})",
                    )
                record = {
                    "phase": phase.name,
                    "frame": telemetry.frame,
                    "simulation_time": frame.simulation_time,
                    "intent": vars(phase.intent),
                    "start_location": phase_start.location,
                    "start_rotation": phase_start.rotation,
                    "start_speed_mps": phase_start.speed,
                    "inputs": dict(frame.sample.values),
                    "command": command.as_dict(),
                    "applied": applied,
                    "vehicle": telemetry.as_dict(),
                    "mechanics": driver.measurement(),
                }
                records.append(record)
                report["frames"].append(
                    {
                        key: record[key]
                        for key in (
                            "phase",
                            "frame",
                            "simulation_time",
                            "command",
                            "applied",
                            "vehicle",
                            "mechanics",
                        )
                    }
                )
                if observer is not None:
                    observer.frame(record, driver)
                last = telemetry
                resting = resting + 1 if telemetry.speed < 0.05 else 0
                if phase.until_stopped and index >= 25 and resting >= 10:
                    break
            phases[phase.name] = _phase_summary(phase, records)
            report["phases"].append(phases[phase.name])
        report["checks"] = _checks(phases)
        failed = [check["name"] for check in report["checks"] if not check["passed"]]
        _require(not failed, "Failed checks: " + ", ".join(failed))
        report["status"] = "passed"
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise LiveValidationError(str(error), report) from error
    finally:
        if observer is not None:
            try:
                observer.close()
            except Exception as error:  # pragma: no cover - evidence only
                report.setdefault("cleanup_errors", []).append(str(error))
        if session is not None:
            try:
                session.close()
                world = client.get_world()
                report["settings_after"] = _settings(world)
                report["settings_restored"] = (
                    report["settings_after"] == report["settings_before"]
                )
                report["actor_ids_restored"] = {
                    actor.id for actor in world.get_actors()
                } == before
            except Exception as error:
                report.setdefault("cleanup_errors", []).append(str(error))
            if not (
                report.get("settings_restored") and report.get("actor_ids_restored")
            ):
                report["status"] = "failed"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()

"""Physical acceptance fixtures and measured tests against a real CARLA server."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import mujoco
import numpy as np

from .bridge import CarlaBridge
from .cockpit import CockpitConfig, PedalConfig, SteeringConfig, add_cockpit
from .controls import VehicleCommand
from .inputs import (
    Binding,
    BipolarCalibration,
    CallbackSource,
    ControlMapper,
    SensorValue,
    UnipolarCalibration,
)
from .session import CarlaSession, SessionConfig


@dataclass
class MuscleDriver:
    """Real MyoSuite finger, passive mechanism, and contact measurements."""

    model: mujoco.MjModel
    data: mujoco.MjData
    mapper: ControlMapper
    config: CockpitConfig
    control: str
    peaks: dict[str, float] = field(
        default_factory=lambda: {"travel": 0, "contact_newtons": 0, "muscle_newtons": 0}
    )

    def measurement(self) -> dict[str, float]:
        """Read mechanism travel and contact/muscle force magnitudes."""
        names = self.config.names
        return {
            "travel": float(
                self.data.joint(getattr(names, f"{self.control}_joint")).qpos[0]
            ),
            "contact_newtons": float(
                self.data.sensor(names.touch_sensor(self.control)).data[0]
            ),
            "muscle_newtons": float(np.max(np.abs(self.data.actuator_force))),
        }

    def advance(self, excitation: float = 0.05) -> None:
        """Excite flexors and advance 20 ms using muscle forces and contact only."""
        if not math.isfinite(excitation) or not 0 <= excitation <= 1:
            raise ValueError("excitation must be finite and in [0, 1]")
        _require(
            not np.any(self.data.qfrc_applied) and not np.any(self.data.xfrc_applied),
            "External forces are forbidden in muscle/contact validation",
        )
        self.data.ctrl[:] = 0
        for name in ("mflx", "dflx"):
            self.data.ctrl[self.model.actuator(name).id] = excitation
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            for name, value in self.measurement().items():
                self.peaks[name] = max(self.peaks[name], abs(value))
        _require(bool(np.all(np.isfinite(self.data.qpos))), "Nonfinite MuJoCo state")


def _require(condition: Any, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def build_muscle_driver(control: str, *, contacts_enabled: bool = True) -> MuscleDriver:
    """Build the real MyoSuite finger/contact fixture for one cockpit control."""
    from myosuite.core.model_builder import ModelBuilder

    if control not in ("throttle", "brake", "steering"):
        raise ValueError("control must be throttle, brake, or steering")
    source = mujoco.MjSpec()
    source.option.gravity[:] = 0
    pedal = PedalConfig(
        position=(0.235, 0, 0.245),
        quaternion=(math.sqrt(0.5), -math.sqrt(0.5), 0, 0),
        half_size=(0.06, 0.01, 0.045),
    )
    wheel = SteeringConfig(
        position=(0.14, 0, 0.12),
        radius=0.1,
        grip_radius=0.01,
        segments=16,
        angle_limit=1,
        stiffness=0,
        damping=0.1,
        mass=0.5,
    )
    config = CockpitConfig(
        throttle=pedal if control == "throttle" else PedalConfig(position=(2, 0, 1)),
        brake=pedal if control == "brake" else PedalConfig(position=(2, 0, 2)),
        steering=wheel if control == "steering" else SteeringConfig(position=(2, 0, 3)),
    )
    builder = add_cockpit(
        ModelBuilder.from_spec(source).attach_fragment("finger").set_timestep(0.0005),
        config,
    )
    if not contacts_enabled:

        def remove_contacts(spec: mujoco.MjSpec) -> mujoco.MjSpec:
            for geom in spec.geoms:
                if geom.name.startswith(f"{config.prefix}{control}_"):
                    geom.contype = geom.conaffinity = 0
            return spec

        builder.apply_transform(remove_contacts)
    model, _ = builder.build()
    _require(
        model.nu == 5 and np.all(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE),
        "Expected five real MyoSuite muscle actuators",
    )
    mapper = ControlMapper(
        model,
        [
            Binding(
                "steer" if control == "steering" else control,
                SensorValue(getattr(config.names, f"{control}_position")),
                BipolarCalibration(-1, 0, 1)
                if control == "steering"
                else UnipolarCalibration(0, pedal.travel),
            )
        ],
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return MuscleDriver(model, data, mapper, config, control)


class LiveValidationError(RuntimeError):
    """Failed actual-server validation with the collected evidence attached."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


def _settings(world: Any) -> dict[str, Any]:
    settings = world.get_settings()
    return {
        name: getattr(settings, name)
        for name in (
            "synchronous_mode",
            "fixed_delta_seconds",
            "substepping",
            "max_substeps",
            "max_substep_delta_time",
            "no_rendering_mode",
        )
    }


def _yaw_delta(start: float, end: float) -> float:
    return (end - start + 180) % 360 - 180


def _spawn(world: Any, blueprint_id: str) -> Any:
    blueprint = world.get_blueprint_library().find(blueprint_id)
    _require(
        blueprint.has_attribute("number_of_wheels")
        and int(blueprint.get_attribute("number_of_wheels")) == 4,
        "Use a four-wheeled passenger vehicle blueprint",
    )
    road_map = world.get_map()
    candidates = sorted(
        road_map.get_spawn_points(),
        key=lambda pose: (
            pose.location.x,
            pose.location.y,
            pose.location.z,
            pose.rotation.yaw,
        ),
    )
    for pose in candidates:
        waypoint = road_map.get_waypoint(pose.location)
        if waypoint is None or waypoint.is_junction:
            continue
        ahead = waypoint.next(35)
        if not ahead or any(
            abs(_yaw_delta(pose.rotation.yaw, point.transform.rotation.yaw)) > 5
            or abs(point.transform.location.z - pose.location.z) > 2
            for point in ahead
        ):
            continue
        vehicle = world.try_spawn_actor(blueprint, pose)
        if vehicle is not None:
            return vehicle
    raise RuntimeError("Map needs an unoccupied 35 m straight road spawn")


def _trial(
    client: Any,
    carla: Any,
    control: str,
    blueprint: str,
    timeout: float,
    evidence: dict[str, Any],
) -> None:
    world = client.get_world()
    original = _settings(world)
    evidence.update(control=control, settings_before=original, samples=[])
    _require(not original["synchronous_mode"], "Stop the existing CARLA ticking owner")
    driver = build_muscle_driver(control)
    evidence["muscles"] = [
        driver.model.actuator(i).name for i in range(driver.model.nu)
    ]
    evidence["mechanism"] = getattr(driver.config.names, f"{control}_joint")
    vehicle = None
    try:
        vehicle = _spawn(world, blueprint)
        evidence.update(vehicle_id=vehicle.id, blueprint=vehicle.type_id)
        vehicle.set_autopilot(False)
        world.wait_for_tick(timeout)
        with CarlaSession(
            client,
            vehicle,
            SessionConfig(timeout_seconds=timeout),
            api=carla,
        ) as session:

            def record(phase: str, command: VehicleCommand, telemetry: Any) -> None:
                applied = vehicle.get_control()
                actual = {
                    name: getattr(applied, name)
                    for name in (
                        "throttle",
                        "brake",
                        "steer",
                    )
                }
                _require(
                    all(
                        math.isclose(actual[name], getattr(command, name), abs_tol=1e-5)
                        for name in actual
                    ),
                    "CARLA did not apply the measured controls",
                )
                evidence["samples"].append(
                    {
                        "phase": phase,
                        "command": command.as_dict(),
                        "applied": actual,
                        "vehicle": telemetry.as_dict(),
                        "mujoco_time": float(driver.data.time),
                        "physical": driver.measurement(),
                    }
                )

            for _ in range(25):
                command = VehicleCommand(brake=1, hand_brake=True)
                record("settle_scalar_setup", command, session.step(command))
            if control != "throttle":
                for _ in range(150):
                    command = VehicleCommand(throttle=0.6)
                    record("accelerate_scalar_setup", command, session.step(command))
            excitation = 0.05 if control == "steering" else 0.01
            evidence.update(
                excitation=excitation,
                cruising_throttle=0.3 if control == "steering" else 0,
            )
            if control == "steering":
                driver.mapper = ControlMapper(
                    driver.model,
                    [
                        *driver.mapper.bindings,
                        Binding("throttle", CallbackSource(lambda model, data: 0.3)),
                    ],
                )
            bridge = CarlaBridge(driver.mapper, session, 0.02)
            start = bridge.reset(driver.data).telemetry
            if control != "throttle":
                _require(start.speed > 2, "Scalar setup did not reach driving speed")
            for _ in range(150 if control == "throttle" else 100):
                driver.advance(excitation)
                frame = bridge.step(driver.data)
                command, end = frame.sample.command, frame.telemetry
                record("muscle_contact", command, end)
            yaw = _yaw_delta(start.rotation[1], end.rotation[1])
            distance = math.dist(start.location[:2], end.location[:2])
            evidence.update(
                peaks=driver.peaks,
                start_speed_mps=start.speed,
                end_speed_mps=end.speed,
                displacement_metres=distance,
                yaw_change_degrees=yaw,
            )
            _require(
                driver.peaks["contact_newtons"] > 10, "No measured cockpit contact"
            )
            _require(driver.peaks["muscle_newtons"] > 100, "No measured muscle force")
            if control == "throttle":
                _require(
                    end.speed > start.speed + 0.8 and distance > 1,
                    "Physical throttle did not accelerate and move the CARLA car",
                )
            elif control == "brake":
                _require(
                    end.speed < start.speed - 1 and end.speed < start.speed * 0.7,
                    "Physical brake did not substantially reduce vehicle speed",
                )
            else:
                _require(
                    end.speed > 1 and yaw < -2,
                    "Physical negative steering did not turn the moving car left",
                )
            evidence["passed"] = True
    finally:
        if vehicle is not None:
            evidence["vehicle_destroyed"] = bool(vehicle.destroy())
            _require(
                evidence["vehicle_destroyed"], "Could not destroy validation vehicle"
            )
        evidence["settings_after"] = _settings(world)
        evidence["settings_restored"] = evidence["settings_after"] == original
        _require(evidence["settings_restored"], "CARLA settings were not restored")


def run_live_validation(
    host: str = "localhost",
    port: int = 2000,
    *,
    timeout: float = 15,
    blueprint: str = "vehicle.tesla.model3",
    build_ref: str | None = None,
    build_commit: str | None = None,
) -> dict[str, Any]:
    """Test actual CARLA vehicle movement, braking and steering; retain evidence.

    Build ref and commit are caller assertions. RPC version equality establishes
    a matching client/server pair, but cannot independently prove a Git branch.
    """
    report: dict[str, Any] = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "host": host,
        "port": port,
        "mujoco_version": mujoco.__version__,
        "provenance": {
            "expected_ref": "ue58-dev",
            "asserted_ref": build_ref,
            "asserted_commit": build_commit,
            "source_verified": False,
        },
        "scenarios": [],
    }
    try:
        carla = importlib.import_module("carla")
        report["carla_module"] = str(getattr(carla, "__file__", None))
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
        report["map"] = client.get_world().get_map().name
        for control in ("throttle", "brake", "steering"):
            evidence: dict[str, Any] = {}
            report["scenarios"].append(evidence)
            _trial(client, carla, control, blueprint, timeout, evidence)
        report["status"] = "passed"
        return report
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise LiveValidationError(str(error), report) from error
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()

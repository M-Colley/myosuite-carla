"""Mix and match with embodied-ui: devices, engines and vehicles.

- Compose ``embodied_ui.Interface`` objects into the full-body driver with
  ``FullBodyDriverConfig(interfaces=...)`` and forward their channels to
  CARLA with :func:`interface_hook` (lights) in a :class:`CarlaBridge`.
- :class:`EngineVehicleSession` drives a vehicle in any embodied-ui engine,
  e.g. a Unity car through ``embodied_ui.backends.RemoteEngine``, with the
  same session protocol as :class:`CarlaSession`, so ``CarlaBridge``,
  ``FullBodyDrivingEnv`` and the scenarios work unchanged.
- :class:`VehicleSessionEngine` exposes a myosuite-carla session (CARLA or
  the fitted :class:`SurrogateVehicle`) as an embodied-ui engine, so any
  MuJoCo body with embodied-ui controls can drive it.

embodied-ui is imported lazily; install it with ``pip install embodied-ui``.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .controls import VehicleCommand
from .inputs import ControlSample
from .session import SynchronizationError, VehicleTelemetry

#: Vehicle controls shared by CARLA, Unity and every embodied-ui engine.
VEHICLE_CONTROLS = (
    "throttle",
    "steer",
    "brake",
    "hand_brake",
    "reverse",
    "manual_gear_shift",
    "gear",
)


def interface_hook(driver: Any, routes: Any) -> Callable[[Any, ControlSample], None]:
    """Return a CarlaBridge hook that connects interface devices to CARLA.

    ``routes`` (``embodied_ui.engine.Routes``) maps channels to ``lights.*``
    targets, e.g. ``Route("hazard.state", "lights.hazard")``, and its
    ``feedback`` routes the vehicle's ``lights.*`` back to device outputs,
    e.g. ``Route("lights.left_blinker", "lamp.on")``. Vehicle controls stay
    with the driver's own pedals and wheel.
    """
    from embodied_ui.backends.carla import light_outputs, light_state

    def hook(session: Any, sample: ControlSample) -> None:
        reading = driver.interface_reading()
        if reading is None:
            return
        command = routes.command(reading)
        lights = {
            name.partition(".")[2]: float(value) >= 0.5
            for name, value in command.controls.items()
            if name.startswith("lights.") and not math.isnan(float(value))
        }
        carla = importlib.import_module("carla")
        if lights:
            session.vehicle.set_light_state(light_state(carla, lights))
        if routes.feedback:
            outputs = light_outputs(carla, session.vehicle.get_light_state())
            values = {
                route.target: route.value(outputs)
                for route in routes.feedback
                if route.source in outputs
            }
            for runtime in driver.interfaces:
                runtime.set_outputs(driver.data, values)

    return hook


@dataclass(frozen=True)
class EngineSessionConfig:
    """The part of :class:`SessionConfig` a vehicle session must expose."""

    fixed_delta_seconds: float


class EngineVehicleSession:
    """A myosuite-carla vehicle session backed by an embodied-ui engine.

    The engine must report ``vehicle.pose``, ``vehicle.velocity``,
    ``vehicle.acceleration`` and ``vehicle.angular_velocity`` (rad/s) in its
    pose convention; the Unity package's ``EmbodiedVehicleOutput`` and
    embodied-ui's ``CarlaEngine`` do. Telemetry is converted to CARLA's
    frame and units, as :class:`VehicleTelemetry` expects.

    Args:
        engine: An unstarted embodied-ui engine.
        convention: Pose convention of the engine; ``None`` asks a remote
            engine (``RemoteEngine.convention``) after it starts.
        extra: Function returning further controls for each step, e.g.
            lights from an interface reading.
    """

    def __init__(
        self,
        engine: Any,
        convention: str | None = None,
        extra: Callable[[], Mapping[str, float]] | None = None,
    ) -> None:
        self.engine = engine
        self.config = EngineSessionConfig(float(engine.dt))
        self.convention = convention
        self.extra = extra
        self._telemetry: VehicleTelemetry | None = None
        self._time = 0.0
        self._failed = False

    @property
    def active(self) -> bool:
        """Whether the session is started and healthy."""
        return self._telemetry is not None and not self._failed

    def start(self) -> EngineVehicleSession:
        """Start the engine and read the first telemetry without advancing."""
        from embodied_ui.engine import SessionInfo

        controls = tuple(f"vehicle.{name}" for name in VEHICLE_CONTROLS)
        state = self.engine.start(
            SessionInfo(self.config.fixed_delta_seconds, controls)
        )
        if self.convention is None:
            self.convention = str(getattr(self.engine, "convention", "flu"))
        self._telemetry = self._convert(state)
        return self

    def snapshot(self) -> VehicleTelemetry:
        """Return the latest telemetry without advancing."""
        if self._telemetry is None:
            raise RuntimeError("Start the session before reading feedback")
        return self._telemetry

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> VehicleTelemetry:
        """Reset the engine episode and recover a failed session."""
        self._failed = True
        state = self.engine.reset(seed=seed, options=options)
        self._telemetry = self._convert(state)
        self._failed = False
        return self._telemetry

    def step(self, command: VehicleCommand) -> VehicleTelemetry:
        """Apply one command, advance one period and return the telemetry."""
        from embodied_ui.engine import Command

        if not self.active:
            raise RuntimeError("Start a healthy session before stepping")
        try:
            controls: dict[str, float] = {
                f"vehicle.{key}": float(value)
                for key, value in command.as_dict().items()
            }
            if self.extra is not None:
                controls.update({k: float(v) for k, v in self.extra().items()})
            previous = self._telemetry
            assert previous is not None
            state = self.engine.step(Command(previous.elapsed_seconds, controls))
            telemetry = self._convert(state)
            elapsed = telemetry.elapsed_seconds - previous.elapsed_seconds
            if not math.isclose(
                elapsed,
                self.config.fixed_delta_seconds,
                rel_tol=1e-6,
                abs_tol=self.config.fixed_delta_seconds * 1e-6,
            ):
                raise SynchronizationError("engine time differs from the fixed delta")
            self._telemetry = telemetry
            return telemetry
        except Exception:
            self._failed = True
            self._stop_after_failure()
            raise

    def stop(self) -> None:
        """Nothing to queue: lockstep engines only move on a step."""

    def _stop_after_failure(self) -> None:
        self.stop()

    def close(self) -> None:
        """End the engine session."""
        self._telemetry = None
        self.engine.close()

    def __enter__(self) -> EngineVehicleSession:
        return self.start()

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _convert(self, state: Any) -> VehicleTelemetry:
        from embodied_ui.frames import CARLA, Pose, frame, rotation_to_unreal

        outputs = state.outputs
        try:
            pose = np.asarray(outputs["vehicle.pose"], dtype=float)
            velocity = outputs["vehicle.velocity"]
            acceleration = outputs.get("vehicle.acceleration", (0.0, 0.0, 0.0))
            omega = outputs["vehicle.angular_velocity"]
        except KeyError as error:
            raise KeyError(f"the engine reports no {error.args[0]!r} output") from None
        source = frame(self.convention or "flu")
        carla_pose = source.pose_to(CARLA, Pose(tuple(pose[:3]), tuple(pose[3:])))  # type: ignore[arg-type]
        return VehicleTelemetry(
            frame=int(state.frame),
            elapsed_seconds=float(state.time),
            location=_triple(carla_pose.pos),
            rotation=rotation_to_unreal(carla_pose.rotation()),
            velocity=_triple(source.point_to(CARLA, velocity)),
            acceleration=_triple(source.point_to(CARLA, acceleration)),
            angular_velocity=_triple(
                math.degrees(v) for v in source.pseudovector_to(CARLA, omega)
            ),
        )


def _triple(values: Any) -> tuple[float, float, float]:
    x, y, z = (float(v) for v in values)
    return (x, y, z)


class VehicleSessionEngine:
    """An embodied-ui engine around a myosuite-carla vehicle session.

    Controls ``vehicle.*`` become a :class:`VehicleCommand`; outputs are the
    session's telemetry in embodied-ui's vehicle schema (CARLA frame, SI
    units, angular velocity in rad/s) plus CARLA's ``vehicle.rotation``.

    Args:
        session: A :class:`CarlaSession` or :class:`SurrogateVehicle`.
        reset_vehicle: Reset a live vehicle with ``(session, seed, options)``.
            Surrogates restore the pose and speed captured at startup.
    """

    convention = "carla"

    def __init__(
        self,
        session: Any,
        *,
        reset_vehicle: Callable[[Any, int | None, Mapping[str, Any] | None], None]
        | None = None,
    ) -> None:
        self.session = session
        self.dt = float(session.config.fixed_delta_seconds)
        self._start: VehicleTelemetry | None = None
        self._reset_vehicle = reset_vehicle

    def start(self, info: Any) -> Any:
        if not math.isclose(info.dt, self.dt, rel_tol=1e-9):
            raise ValueError("session dt differs from the vehicle session's")
        if not getattr(self.session, "active", False):
            self.session.start()
        self._start = self.session.snapshot()
        return self._state(self._start)

    def reset(
        self, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> Any:
        from .vehicle_model import SurrogateVehicle

        if self._start is None:
            raise RuntimeError("Start the engine before resetting")
        if self._reset_vehicle is not None:
            self._reset_vehicle(self.session, seed, options)
        elif isinstance(self.session, SurrogateVehicle):
            initial = self._start
            yaw = math.radians(initial.rotation[1])
            speed = initial.velocity[0] * math.cos(yaw) + initial.velocity[
                1
            ] * math.sin(yaw)
            self.session.reset(
                location=initial.location, yaw=initial.rotation[1], speed=speed
            )
        else:
            raise RuntimeError(
                "reset_vehicle is required to reset a live vehicle session"
            )
        telemetry = self.session.snapshot()
        return self._state(telemetry)

    def step(self, command: Any) -> Any:
        values = {
            name.partition(".")[2]: value
            for name, value in command.controls.items()
            if name.startswith("vehicle.") and not math.isnan(float(value))
        }
        unknown = set(values) - set(VEHICLE_CONTROLS)
        if unknown:
            raise KeyError(f"unknown vehicle controls {sorted(unknown)}")
        vehicle = VehicleCommand(
            throttle=min(1.0, max(0.0, float(values.get("throttle", 0.0)))),
            steer=min(1.0, max(-1.0, float(values.get("steer", 0.0)))),
            brake=min(1.0, max(0.0, float(values.get("brake", 0.0)))),
            hand_brake=float(values.get("hand_brake", 0.0)) >= 0.5,
            reverse=float(values.get("reverse", 0.0)) >= 0.5,
            manual_gear_shift=float(values.get("manual_gear_shift", 0.0)) >= 0.5,
            gear=int(round(float(values.get("gear", 0.0)))),
        )
        return self._state(self.session.step(vehicle))

    def close(self) -> None:
        self.session.close()

    def _state(self, telemetry: VehicleTelemetry) -> Any:
        from embodied_ui.engine import EngineState
        from embodied_ui.frames import Pose, rotation_from_unreal

        pose = Pose.from_matrix(
            rotation_from_unreal(telemetry.rotation), telemetry.location
        )
        return EngineState(
            telemetry.frame,
            telemetry.elapsed_seconds,
            {
                "vehicle.pose": [*pose.pos, *pose.quat],
                "vehicle.location": list(telemetry.location),
                "vehicle.rotation": list(telemetry.rotation),
                "vehicle.velocity": list(telemetry.velocity),
                "vehicle.acceleration": list(telemetry.acceleration),
                "vehicle.angular_velocity": [
                    math.radians(v) for v in telemetry.angular_velocity
                ],
                "vehicle.speed": telemetry.speed,
            },
        )

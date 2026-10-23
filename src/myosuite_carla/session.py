"""Exclusive synchronous CARLA sessions and frame-aligned vehicle feedback."""

from __future__ import annotations

import importlib
import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

from .controls import VehicleCommand


class SynchronizationError(RuntimeError):
    """The simulators cannot continue in lockstep."""


@dataclass(frozen=True)
class SessionConfig:
    """Configure one exclusive CARLA ticking client.

    Args:
        fixed_delta_seconds: Shared MyoSuite and CARLA control period.
        timeout_seconds: RPC and tick deadline.
        max_substep_delta_time: Maximum CARLA physics substep.
        max_substeps: Maximum physics substeps per CARLA frame.
        allow_existing_synchronous: Explicitly take over an already synchronous world.
        no_rendering_mode: Disable CARLA rendering, including camera output.
    """

    fixed_delta_seconds: float = 0.02
    timeout_seconds: float = 10.0
    max_substep_delta_time: float = 0.01
    max_substeps: int = 10
    allow_existing_synchronous: bool = False
    no_rendering_mode: bool = False

    def __post_init__(self) -> None:
        for name in (
            "fixed_delta_seconds",
            "timeout_seconds",
            "max_substep_delta_time",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.max_substeps, bool) or not isinstance(
            self.max_substeps, int
        ):
            raise ValueError("max_substeps must be an integer")
        if self.max_substeps <= 0:
            raise ValueError("max_substeps must be positive")
        if self.fixed_delta_seconds > self.max_substep_delta_time * self.max_substeps:
            raise ValueError("CARLA fixed delta exceeds its physics substep budget")


@dataclass(frozen=True)
class VehicleTelemetry:
    """A single CARLA actor snapshot, in CARLA's left-handed world frame.

    Positions are metres, rotations degrees, linear velocities metres/second,
    acceleration metres/second squared, and angular velocities degrees/second.
    """

    frame: int
    elapsed_seconds: float
    location: tuple[float, float, float]
    rotation: tuple[float, float, float]
    velocity: tuple[float, float, float]
    acceleration: tuple[float, float, float]
    angular_velocity: tuple[float, float, float]

    @property
    def speed(self) -> float:
        """Return world linear speed in metres/second."""
        return math.sqrt(sum(value * value for value in self.velocity))

    def as_dict(self) -> dict[str, Any]:
        """Return serializable feedback with explicit frame and speed."""
        return {**asdict(self), "speed": self.speed}


def _vector(value: Any) -> tuple[float, float, float]:
    return (float(value.x), float(value.y), float(value.z))


class CarlaSession:
    """Manage world settings and a borrowed or explicitly owned vehicle.

    Args:
        client: CARLA Client, from the same build as the server.
        vehicle: Vehicle in the client's current world; disable autopilot first.
        config: Synchronous simulation settings.
        api: Optional CARLA module for dependency injection.
        own_vehicle: Destroy this vehicle on close, including startup failures.
    """

    def __init__(
        self,
        client: Any,
        vehicle: Any,
        config: SessionConfig = SessionConfig(),
        *,
        api: Any | None = None,
        own_vehicle: bool = False,
    ) -> None:
        self.client = client
        self.vehicle = vehicle
        self.config = config
        self.world = client.get_world()
        self._api = api
        self._own_vehicle = own_vehicle
        self._original_settings: Any = None
        self._active = False
        self._closed = False
        self._failed = False
        self._stop_pending = False
        self._telemetry: VehicleTelemetry | None = None
        self.client_version = ""
        self.server_version = ""

    @property
    def active(self) -> bool:
        """Return whether this session can advance frames."""
        return self._active and not self._failed and not self._closed

    def start(self) -> CarlaSession:
        """Validate the build pair, acquire ticking, and capture the initial frame."""
        if self._closed or self._failed:
            raise RuntimeError("Create a new session after closing or failing")
        if self._active:
            return self
        try:
            if self._api is None:
                self._api = importlib.import_module("carla")
            self.client.set_timeout(self.config.timeout_seconds)
            self.client_version = self.client.get_client_version()
            self.server_version = self.client.get_server_version()
            if self.client_version != self.server_version:
                raise RuntimeError(
                    f"CARLA client {self.client_version!r} and server "
                    f"{self.server_version!r} differ; install the matching UE5.8 wheel"
                )
            if not str(self.vehicle.type_id).startswith("vehicle."):
                raise ValueError("Expected a CARLA vehicle actor")
            if self.vehicle.get_world().id != self.world.id:
                raise ValueError("Vehicle and client belong to different CARLA worlds")
            original = self.world.get_settings()
            if original.synchronous_mode and not self.config.allow_existing_synchronous:
                raise SynchronizationError(
                    "World already has a synchronous owner; stop it before taking over"
                )
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.config.fixed_delta_seconds
            settings.substepping = True
            settings.max_substep_delta_time = self.config.max_substep_delta_time
            settings.max_substeps = self.config.max_substeps
            settings.no_rendering_mode = self.config.no_rendering_mode
            self._original_settings = original
            self._stop_pending = True
            self.world.apply_settings(settings)
            self._active = True
            self._telemetry = self._read_snapshot()
            return self
        except Exception:
            self._failed = True
            try:
                self.close()
            except Exception as error:
                warnings.warn(str(error), stacklevel=2)
            raise

    def _read_snapshot(self, expected_frame: int | None = None) -> VehicleTelemetry:
        snapshot = self.world.get_snapshot()
        if expected_frame is not None and snapshot.frame != expected_frame:
            raise SynchronizationError("Another client advanced CARLA during this step")
        actor = snapshot.find(self.vehicle.id)
        if actor is None:
            raise SynchronizationError("Vehicle disappeared from the CARLA snapshot")
        transform = actor.get_transform()
        return VehicleTelemetry(
            frame=int(snapshot.frame),
            elapsed_seconds=float(snapshot.timestamp.elapsed_seconds),
            location=_vector(transform.location),
            rotation=(
                float(transform.rotation.pitch),
                float(transform.rotation.yaw),
                float(transform.rotation.roll),
            ),
            velocity=_vector(actor.get_velocity()),
            acceleration=_vector(actor.get_acceleration()),
            angular_velocity=_vector(actor.get_angular_velocity()),
        )

    def snapshot(self) -> VehicleTelemetry:
        """Read frame-aligned feedback without advancing either simulator."""
        if not self.active or self._telemetry is None:
            raise RuntimeError("Start the CARLA session before reading feedback")
        current = self._read_snapshot(self._telemetry.frame)
        self._telemetry = current
        return current

    def _apply(self, command: VehicleCommand) -> None:
        if self._api is None:
            raise RuntimeError("Start the CARLA session before applying controls")
        control = self._api.VehicleControl(**command.as_dict())
        batch_command = self._api.command.ApplyVehicleControl(self.vehicle.id, control)
        responses = self.client.apply_batch_sync([batch_command], False)
        if len(responses) != 1:
            raise RuntimeError(
                f"CARLA control batch returned {len(responses)} responses; expected one"
            )
        if responses[0].has_error():
            raise RuntimeError(f"CARLA control command failed: {responses[0].error}")

    def step(self, command: VehicleCommand) -> VehicleTelemetry:
        """Apply a measured command, tick once, and return that frame's feedback."""
        if not self.active or self._telemetry is None:
            raise RuntimeError("Start a healthy CARLA session before stepping")
        try:
            self.snapshot()
            self._apply(command)
            previous = self._telemetry
            frame = self.world.tick(self.config.timeout_seconds)
            if frame != previous.frame + 1:
                raise SynchronizationError("CARLA frame gap: another client is ticking")
            telemetry = self._read_snapshot(frame)
            elapsed = telemetry.elapsed_seconds - previous.elapsed_seconds
            if elapsed <= 0 or not math.isclose(
                elapsed,
                self.config.fixed_delta_seconds,
                rel_tol=1e-6,
                abs_tol=self.config.fixed_delta_seconds * 1e-6,
            ):
                raise SynchronizationError(
                    "CARLA elapsed time differs from the fixed delta"
                )
            self._telemetry = telemetry
            return telemetry
        except Exception:
            self._failed = True
            self._stop_after_failure()
            raise

    def stop(self) -> None:
        """Submit zero throttle and full braking without consuming a simulation tick."""
        if self._api is not None:
            self._apply(VehicleCommand(brake=1.0, hand_brake=True))

    def _stop_after_failure(self) -> None:
        try:
            self.stop()
        except Exception as error:
            warnings.warn(
                f"Could not queue CARLA emergency braking: {error}", stacklevel=2
            )

    def close(self) -> None:
        """Brake, restore all world settings, and destroy only an owned vehicle."""
        if (
            self._closed
            and self._original_settings is None
            and not self._own_vehicle
            and not self._stop_pending
        ):
            return
        self._closed = True
        self._active = False
        errors: list[Exception] = []
        if self._stop_pending:
            try:
                self.stop()
                self._stop_pending = False
            except Exception as error:
                errors.append(error)
        if self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
                self._original_settings = None
            except Exception as error:
                errors.append(error)
        if self._own_vehicle:
            try:
                if not self.vehicle.destroy():
                    raise RuntimeError("CARLA could not destroy the owned vehicle")
                self._own_vehicle = False
                self._stop_pending = False
            except Exception as error:
                errors.append(error)
        if errors:
            raise RuntimeError(
                "CARLA cleanup failed: " + "; ".join(map(str, errors))
            ) from errors[0]

    def __enter__(self) -> CarlaSession:
        """Start this session as a context manager."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release resources while preserving an exception from the caller."""
        try:
            self.close()
        except Exception as error:
            if exc_value is None:
                raise
            warnings.warn(str(error), stacklevel=2)

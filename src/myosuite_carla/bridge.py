"""One MuJoCo control step to one CARLA frame."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import mujoco

from .inputs import ControlMapper, ControlSample
from .session import CarlaSession, SynchronizationError, VehicleTelemetry

ControlHook = Callable[[CarlaSession, ControlSample], None]


@dataclass(frozen=True)
class BridgeFrame:
    """Physical inputs and the resulting vehicle feedback at one shared time."""

    sample: ControlSample
    telemetry: VehicleTelemetry
    simulation_time: float

    def as_dict(self) -> dict[str, Any]:
        """Return the bridge info payload without live CARLA objects."""
        return {
            "inputs": dict(self.sample.values),
            "control": self.sample.command.as_dict(),
            "vehicle": self.telemetry.as_dict(),
            "simulation_time": self.simulation_time,
        }


class CarlaBridge:
    """Measure physical mechanisms after each MuJoCo control step.

    Args:
        mapper: Bindings compiled against the driver's MuJoCo model.
        session: Active exclusive CARLA session.
        control_dt: Driver control period, equal to CARLA's fixed delta.
        hooks: Extra measured controls such as lights or doors, before each tick.
    """

    def __init__(
        self,
        mapper: ControlMapper,
        session: CarlaSession,
        control_dt: float,
        *,
        hooks: tuple[ControlHook, ...] = (),
    ) -> None:
        if not math.isfinite(control_dt) or control_dt <= 0:
            raise ValueError("control_dt must be finite and positive")
        if not math.isclose(
            control_dt, session.config.fixed_delta_seconds, rel_tol=1e-9, abs_tol=0
        ):
            raise ValueError("MyoSuite control_dt must equal CARLA fixed_delta_seconds")
        self.mapper = mapper
        self.session = session
        self.control_dt = float(control_dt)
        self.hooks = tuple(hooks)
        self._last_time: float | None = None
        self._last_carla_time: float | None = None
        self._failed = False

    def reset(self, data: mujoco.MjData) -> BridgeFrame:
        """Anchor a driver episode to the current CARLA frame without ticking.

        Vehicle repositioning and scenario resets remain the caller's responsibility.
        """
        self._failed = True
        self._last_time = None
        self._last_carla_time = None
        try:
            if not math.isfinite(float(data.time)):
                raise ValueError("MuJoCo time must be finite")
            self.mapper.reset()
            sample = self.mapper.sample(data)
            telemetry = self.session.snapshot()
            self._last_time = float(data.time)
            self._last_carla_time = telemetry.elapsed_seconds
            self._failed = False
            return BridgeFrame(sample, telemetry, self._last_time)
        except Exception:
            self.session._stop_after_failure()
            raise

    def step(self, data: mujoco.MjData) -> BridgeFrame:
        """Read one completed driver step and advance CARLA exactly once."""
        if self._last_time is None or self._failed:
            raise RuntimeError("Reset a healthy bridge before stepping")
        try:
            current = float(data.time)
            duration = current - self._last_time
            if duration <= 0 or not math.isclose(
                duration, self.control_dt, rel_tol=1e-6, abs_tol=self.control_dt * 1e-6
            ):
                raise SynchronizationError(
                    "MuJoCo did not advance by one control period"
                )
            sample = self.mapper.sample(data)
            for hook in self.hooks:
                hook(self.session, sample)
            telemetry = self.session.step(sample.command)
            assert self._last_carla_time is not None
            if not math.isclose(
                telemetry.elapsed_seconds - self._last_carla_time,
                current - self._last_time,
                rel_tol=1e-6,
                abs_tol=self.control_dt * 1e-6,
            ):
                raise SynchronizationError("MuJoCo and CARLA clocks diverged")
            self._last_time = current
            self._last_carla_time = telemetry.elapsed_seconds
            return BridgeFrame(sample, telemetry, current)
        except Exception:
            self._failed = True
            self.session._stop_after_failure()
            raise

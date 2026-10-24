"""A fitted single-track surrogate of a CARLA vehicle for fast offline rollouts.

:class:`SurrogateVehicle` stands in for :class:`~myosuite_carla.CarlaSession`
where CARLA is unavailable or too slow (training, batched rollouts, continuous
integration). It has the session's ``config``, ``start``, ``snapshot``,
``step`` and ``close``, so ``CarlaBridge`` and ``DriveRecorder`` work with it
unchanged. :func:`fit_vehicle_model` identifies its parameters from recorded
CARLA drives and :func:`evaluate_vehicle_model` reports open-loop errors on
held-out drives.

The model, in CARLA's left-handed world (x forward at yaw 0, y right), with
signed forward speed ``u`` and yaw rate ``r`` (positive turns right)::

    du/dt = D(T, |u|) - c0 sgn(u) - c1 u - c2 u |u|
            - (kb brake + kh hand_brake) sgn(u) - g sin(pitch)
    r     = u tan(G(|u|) S) / L

``T`` and ``S`` are throttle and steer after first-order lags. ``D`` is a
non-negative drive-acceleration table over throttle and speed (negated in
reverse), which absorbs the engine torque curve and automatic gearbox, and
``G`` an effective road-wheel angle table over speed, which absorbs the
steering curve and tyre slip. Tables are interpolated linearly and held
constant beyond their last knot. Braking never reverses the car. Load
transfer, individual gear shifts, suspension and lateral slip dynamics are not
modelled; the held-out errors in ``VALIDATION.md`` bound what that costs.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .controls import VehicleCommand
from .session import SessionConfig, VehicleTelemetry

GRAVITY = 9.81
#: Speed below which the car is treated as standing still, m/s.
STANDSTILL = 0.05
SPEED_KNOTS = (0.0, 1.0, 2.5, 4.0, 6.0, 8.0, 11.0, 15.0, 20.0)
THROTTLE_KNOTS = (0.25, 0.5, 0.75, 1.0)


def _tuple(values: Any) -> tuple[Any, ...]:
    return tuple(
        _tuple(value) if isinstance(value, (list, tuple, np.ndarray)) else float(value)
        for value in values
    )


@dataclass(frozen=True)
class VehicleModelParams:
    """Surrogate parameters; see the module docstring for the equations.

    Args:
        wheelbase: Axle distance L in metres.
        speed_knots: Increasing speeds of the tables, from 0, m/s.
        throttle_knots: Increasing throttles of the drive tables in (0, 1];
            drive is zero at zero throttle.
        drive: Forward drive acceleration, one row per throttle knot and one
            column per speed knot, m/s^2.
        reverse_drive: Reverse drive acceleration per throttle knot, m/s^2.
        steer_gain: Effective road-wheel angle at full steer per speed knot, rad.
        rolling: Rolling resistance c0, m/s^2.
        linear_drag: Linear resistance c1, 1/s.
        quadratic_drag: Aerodynamic resistance c2, 1/m.
        brake_decel: Deceleration kb at full brake, m/s^2.
        hand_brake_decel: Deceleration kh with the hand brake, m/s^2.
        throttle_lag: Throttle time constant, s.
        steer_lag: Steer time constant, s.
    """

    wheelbase: float = 3.075
    speed_knots: tuple[float, ...] = SPEED_KNOTS
    throttle_knots: tuple[float, ...] = THROTTLE_KNOTS
    drive: tuple[tuple[float, ...], ...] = tuple(
        (3.0 * throttle,) * len(SPEED_KNOTS) for throttle in THROTTLE_KNOTS
    )
    reverse_drive: tuple[float, ...] = tuple(2.0 * value for value in THROTTLE_KNOTS)
    steer_gain: tuple[float, ...] = (0.5,) * len(SPEED_KNOTS)
    rolling: float = 0.15
    linear_drag: float = 0.0
    quadratic_drag: float = 0.002
    brake_decel: float = 8.0
    hand_brake_decel: float = 4.0
    throttle_lag: float = 0.2
    steer_lag: float = 0.1

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            converted = (
                _tuple(value)
                if isinstance(value, (list, tuple, np.ndarray))
                else float(value)
            )
            flat = np.asarray(converted, dtype=float).ravel()
            if not np.all(np.isfinite(flat)) or np.any(flat < 0):
                raise ValueError(f"{item.name} must be finite and non-negative")
            object.__setattr__(self, item.name, converted)
        speeds, throttles = (
            np.asarray(self.speed_knots),
            np.asarray(self.throttle_knots),
        )
        if speeds[0] != 0 or np.any(np.diff(speeds) <= 0):
            raise ValueError("speed_knots must increase from 0")
        if throttles[0] <= 0 or throttles[-1] > 1 or np.any(np.diff(throttles) <= 0):
            raise ValueError("throttle_knots must increase within (0, 1]")
        if np.asarray(self.drive).shape != (len(throttles), len(speeds)):
            raise ValueError("drive must have a row per throttle knot")
        if len(self.reverse_drive) != len(throttles):
            raise ValueError("reverse_drive needs one value per throttle knot")
        gains = np.asarray(self.steer_gain)
        if gains.shape != speeds.shape or np.any(gains <= 0) or np.any(gains >= 1.45):
            raise ValueError("steer_gain needs one angle in (0, 1.45) per speed knot")
        if self.wheelbase <= 0:
            raise ValueError("wheelbase must be positive")

    def drive_acceleration(self, throttle: float, speed: float, reverse: bool) -> float:
        """Return the signed drive acceleration D for a lagged throttle."""
        knots = (0.0, *self.throttle_knots)
        if reverse:
            return -float(np.interp(throttle, knots, (0.0, *self.reverse_drive)))
        rows = [np.interp(abs(speed), self.speed_knots, row) for row in self.drive]
        return float(np.interp(throttle, knots, (0.0, *rows)))

    def road_wheel_angle(self, speed: Any) -> Any:
        """Return the effective road-wheel angle at full steer for a speed."""
        return np.interp(np.abs(speed), self.speed_knots, self.steer_gain)

    def as_dict(self) -> dict[str, Any]:
        """Return the parameters as JSON-ready floats and lists."""
        return {
            item.name: (
                np.asarray(getattr(self, item.name)).tolist()
                if isinstance(getattr(self, item.name), tuple)
                else getattr(self, item.name)
            )
            for item in fields(self)
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VehicleModelParams:
        """Build parameters from :meth:`as_dict` output."""
        return cls(**{item.name: values[item.name] for item in fields(cls)})


def _lag(values: NDArray[np.float64], tau: float, dt: float) -> NDArray[np.float64]:
    """First-order lag of a held input sequence, steady at its first value."""
    alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
    out: NDArray[np.float64] = np.empty(len(values), dtype=float)
    state = float(values[0]) if len(values) else 0.0
    for index, value in enumerate(values):
        state += alpha * (float(value) - state)
        out[index] = state
    return out


def _yaw_rate(params: VehicleModelParams, speed: Any, steer: Any) -> Any:
    angle = np.clip(params.road_wheel_angle(speed) * np.asarray(steer), -1.4, 1.4)
    return np.asarray(speed) * np.tan(angle) / params.wheelbase


def _acceleration(
    params: VehicleModelParams,
    speed: float,
    throttle: float,
    brake: float,
    hand_brake: float,
    reverse: bool,
    pitch_deg: float,
) -> tuple[float, float]:
    """Return (propulsive, opposing) acceleration; opposing never reverses."""
    drive = params.drive_acceleration(throttle, speed, reverse)
    drive -= GRAVITY * math.sin(math.radians(pitch_deg))
    opposing = (
        params.rolling
        + params.linear_drag * abs(speed)
        + params.quadratic_drag * speed * speed
        + params.brake_decel * brake
        + params.hand_brake_decel * hand_brake
    )
    return drive, opposing


def _advance_speed(speed: float, drive: float, opposing: float, dt: float) -> float:
    if abs(speed) < STANDSTILL and abs(drive) <= opposing:
        # Static friction and brakes hold the car until drive exceeds them.
        return 0.0
    if abs(speed) < STANDSTILL:
        return speed + math.copysign(abs(drive) - opposing, drive) * dt
    new = speed + (drive - math.copysign(opposing, speed)) * dt
    # Resistance and brakes stop the car; only drive may reverse it.
    if new * speed < 0 and abs(drive) <= opposing:
        return 0.0
    return new


class SurrogateVehicle:
    """A ``CarlaSession`` stand-in driven by :class:`VehicleModelParams`.

    Args:
        params: Identified surrogate parameters; defaults to the CARLA 0.10.0
            Nissan Patrol fit (:func:`carla_nissan_patrol`).
        config: Only ``fixed_delta_seconds`` is used.
        location: Initial reference-point location in CARLA's world, metres.
        yaw: Initial heading in degrees.
        speed: Initial signed forward speed in m/s.
    """

    def __init__(
        self,
        params: VehicleModelParams | None = None,
        config: SessionConfig | None = None,
        *,
        location: Sequence[float] = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        speed: float = 0.0,
    ) -> None:
        self.params = params if params is not None else carla_nissan_patrol()
        self.config = config if config is not None else SessionConfig()
        self._active = False
        self.reset(location, yaw, speed)

    @property
    def active(self) -> bool:
        """Return whether the surrogate can advance frames."""
        return self._active

    def reset(
        self,
        location: Sequence[float] = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        speed: float = 0.0,
    ) -> VehicleTelemetry:
        """Place the vehicle; lags and time restart from rest."""
        x, y, z = (float(value) for value in location)
        self._position = [x, y, z]
        self._yaw = math.radians(float(yaw))
        self._speed = float(speed)
        self._throttle = self._steer = self._rate = 0.0
        self._accel = self._lateral = 0.0
        self._frame = 0
        self._elapsed = 0.0
        self._telemetry = self._read()
        return self._telemetry

    def start(self) -> SurrogateVehicle:
        """Activate stepping; the surrogate has no world settings to change."""
        self._active = True
        return self

    def snapshot(self) -> VehicleTelemetry:
        """Return the current frame's telemetry without advancing."""
        if not self._active:
            raise RuntimeError("Start the surrogate before reading feedback")
        return self._telemetry

    def step(self, command: VehicleCommand) -> VehicleTelemetry:
        """Apply a command for one fixed period and return the new telemetry."""
        if not self._active:
            raise RuntimeError("Start the surrogate before stepping")
        if not isinstance(command, VehicleCommand):
            raise TypeError("command must be a VehicleCommand")
        params, dt = self.params, self.config.fixed_delta_seconds
        for name, target, tau in (
            ("_throttle", command.throttle, params.throttle_lag),
            ("_steer", command.steer, params.steer_lag),
        ):
            alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
            value = getattr(self, name)
            setattr(self, name, value + alpha * (target - value))
        drive, opposing = _acceleration(
            params,
            self._speed,
            self._throttle,
            command.brake,
            float(command.hand_brake),
            command.reverse,
            0.0,
        )
        speed = _advance_speed(self._speed, drive, opposing, dt)
        rate = float(_yaw_rate(params, speed, self._steer))
        heading = self._yaw + 0.5 * rate * dt
        mean = 0.5 * (speed + self._speed)
        self._position[0] += mean * math.cos(heading) * dt
        self._position[1] += mean * math.sin(heading) * dt
        self._yaw += rate * dt
        self._accel = (speed - self._speed) / dt
        self._lateral = speed * rate
        self._speed, self._rate = speed, rate
        self._frame += 1
        self._elapsed += dt
        self._telemetry = self._read()
        return self._telemetry

    def _read(self) -> VehicleTelemetry:
        forward = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        right = np.array([-math.sin(self._yaw), math.cos(self._yaw), 0.0])
        acceleration = self._accel * forward + self._lateral * right
        return VehicleTelemetry(
            frame=self._frame,
            elapsed_seconds=self._elapsed,
            location=(self._position[0], self._position[1], self._position[2]),
            rotation=(0.0, math.degrees(self._yaw), 0.0),
            velocity=(
                float(self._speed * forward[0]),
                float(self._speed * forward[1]),
                0.0,
            ),
            acceleration=(
                float(acceleration[0]),
                float(acceleration[1]),
                float(acceleration[2]),
            ),
            angular_velocity=(0.0, 0.0, math.degrees(self._rate)),
        )

    def stop(self) -> None:
        """Match ``CarlaSession``; the surrogate has no queued control to brake."""

    def _stop_after_failure(self) -> None:
        """Match ``CarlaSession``; the surrogate has nothing to stop."""

    def close(self) -> None:
        """Deactivate stepping."""
        self._active = False

    def __enter__(self) -> SurrogateVehicle:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()


@dataclass(frozen=True)
class DriveSegment:
    """One continuous logged drive at a fixed period.

    ``states`` arrays have one more entry than ``commands``: command ``k``
    drives state ``k`` to state ``k + 1``.

    Args:
        name: Segment label.
        dt: Period in seconds.
        throttle, brake, steer, hand_brake, reverse: Commands, length N.
        x, y: Reference-point location, length N + 1, metres.
        yaw_deg, pitch_deg: Heading and pitch, length N + 1, degrees.
        speed: Signed forward speed, length N + 1, m/s.
        yaw_rate: Yaw rate, length N + 1, rad/s (positive turns right).
    """

    name: str
    dt: float
    throttle: NDArray[np.float64]
    brake: NDArray[np.float64]
    steer: NDArray[np.float64]
    hand_brake: NDArray[np.float64]
    reverse: NDArray[np.float64]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    yaw_deg: NDArray[np.float64]
    pitch_deg: NDArray[np.float64]
    speed: NDArray[np.float64]
    yaw_rate: NDArray[np.float64]

    @property
    def duration(self) -> float:
        """Return the commanded duration in seconds."""
        return len(self.throttle) * self.dt

    @classmethod
    def from_columns(
        cls, columns: Mapping[str, Any], name: str = "", dt: float | None = None
    ) -> DriveSegment:
        """Build a segment from ``DriveRecorder`` columns (one row per frame).

        Row ``k`` holds the command sent in frame ``k`` and the telemetry after
        that frame, so the first row's command is dropped.
        """
        time = np.asarray(columns["time_s"], dtype=float)
        if len(time) < 3:
            raise ValueError("A segment needs at least three frames")
        period = float(np.median(np.diff(time))) if dt is None else float(dt)
        if not np.allclose(np.diff(time), period, rtol=1e-3, atol=1e-6):
            raise ValueError("Segment frames must be evenly spaced")

        def column(key: str) -> NDArray[np.float64]:
            return np.asarray(columns[key], dtype=float)

        yaw, pitch = np.radians(column("yaw_deg")), np.radians(column("pitch_deg"))
        forward = np.column_stack(
            [np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)]
        )
        velocity = np.column_stack(
            [column("vx_mps"), column("vy_mps"), column("vz_mps")]
        )
        return cls(
            name=name,
            dt=period,
            throttle=column("command_throttle")[1:],
            brake=column("command_brake")[1:],
            steer=column("command_steer")[1:],
            hand_brake=column("command_hand_brake")[1:],
            reverse=column("command_reverse")[1:],
            x=column("x_m"),
            y=column("y_m"),
            yaw_deg=column("yaw_deg"),
            pitch_deg=column("pitch_deg"),
            speed=np.sum(forward * velocity, axis=1),
            yaw_rate=np.radians(column("wz_dps")),
        )


def segments_from_recording(
    recording: Any, *, by: str = "phase", min_frames: int = 10
) -> list[DriveSegment]:
    """Split a ``Recording`` into segments of consecutive equal ``by`` labels."""
    columns = recording.columns
    labels = list(columns[by]) if by in columns else [""] * len(recording)
    frames = np.asarray(columns["frame"], dtype=float)
    segments: list[DriveSegment] = []
    start = 0
    for end in range(1, len(labels) + 1):
        if (
            end == len(labels)
            or labels[end] != labels[start]
            or frames[end] != frames[end - 1] + 1
        ):
            if end - start >= min_frames:
                part = {key: values[start:end] for key, values in columns.items()}
                segments.append(DriveSegment.from_columns(part, str(labels[start])))
            start = end
    return segments


def simulate(
    params: VehicleModelParams, segment: DriveSegment, *, use_pitch: bool = True
) -> dict[str, NDArray[np.float64]]:
    """Open-loop rollout from a segment's initial state with its commands."""
    dt = segment.dt
    count = len(segment.throttle) + 1
    out = {key: np.empty(count) for key in ("x", "y", "yaw_deg", "speed", "yaw_rate")}
    x, y = float(segment.x[0]), float(segment.y[0])
    yaw = math.radians(float(segment.yaw_deg[0]))
    speed, rate = float(segment.speed[0]), float(segment.yaw_rate[0])
    throttle = _lag(segment.throttle, params.throttle_lag, dt)
    steer = _lag(segment.steer, params.steer_lag, dt)
    out["x"][0], out["y"][0], out["yaw_deg"][0] = x, y, math.degrees(yaw)
    out["speed"][0], out["yaw_rate"][0] = speed, rate
    for index in range(count - 1):
        pitch = float(segment.pitch_deg[index]) if use_pitch else 0.0
        drive, opposing = _acceleration(
            params,
            speed,
            float(throttle[index]),
            float(segment.brake[index]),
            float(segment.hand_brake[index]),
            bool(segment.reverse[index] > 0.5),
            pitch,
        )
        new = _advance_speed(speed, drive, opposing, dt)
        rate = float(_yaw_rate(params, new, steer[index]))
        heading = yaw + 0.5 * rate * dt
        x += 0.5 * (speed + new) * math.cos(heading) * dt
        y += 0.5 * (speed + new) * math.sin(heading) * dt
        yaw += rate * dt
        speed = new
        out["x"][index + 1], out["y"][index + 1] = x, y
        out["yaw_deg"][index + 1] = math.degrees(yaw)
        out["speed"][index + 1], out["yaw_rate"][index + 1] = speed, rate
    return out


def _hats(values: NDArray[np.float64], knots: Sequence[float]) -> NDArray[np.float64]:
    """Linear-interpolation weights of each knot, held beyond the last knot."""
    eye = np.eye(len(knots))
    return np.column_stack([np.interp(values, knots, row) for row in eye])


def _second_differences(count: int) -> NDArray[np.float64]:
    matrix = np.zeros((max(count - 2, 0), count))
    for row in range(count - 2):
        matrix[row, row : row + 3] = (1.0, -2.0, 1.0)
    return matrix


def _longitudinal_design(
    segments: Iterable[DriveSegment],
    params: VehicleModelParams,
    throttle_lag: float,
    min_speed: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rows, targets = [], []
    throttle_knots = (0.0, *params.throttle_knots)
    for segment in segments:
        throttle = _lag(segment.throttle, throttle_lag, segment.dt)
        before, after = segment.speed[:-1], segment.speed[1:]
        moving = (np.abs(before) > min_speed) & (np.abs(after) > min_speed)
        moving &= np.sign(before) == np.sign(after)
        sign = np.sign(before)
        reverse = segment.reverse > 0.5
        throttle_hats = _hats(throttle, throttle_knots)[:, 1:]
        speed_hats = _hats(np.abs(before), params.speed_knots)
        cells = (throttle_hats[:, :, None] * speed_hats[:, None, :]).reshape(
            len(before), -1
        )
        design = np.column_stack(
            [
                cells * ~reverse[:, None],
                -throttle_hats * reverse[:, None],
                -sign,
                -before,
                -before * np.abs(before),
                -sign * segment.brake,
                -sign * segment.hand_brake,
            ]
        )
        target = (after - before) / segment.dt + GRAVITY * np.sin(
            np.radians(segment.pitch_deg[:-1])
        )
        rows.append(design[moving])
        targets.append(target[moving])
    return np.vstack(rows), np.concatenate(targets)


def fit_vehicle_model(
    segments: Sequence[DriveSegment],
    *,
    wheelbase: float = 3.075,
    template: VehicleModelParams | None = None,
    throttle_lags: Sequence[float] = tuple(np.linspace(0.0, 0.6, 25)),
    smoothing: float = 1.0,
    min_speed: float = 0.1,
) -> tuple[VehicleModelParams, dict[str, Any]]:
    """Identify surrogate parameters from logged CARLA drives.

    The longitudinal terms are non-negative least squares on measured speed
    changes, with second-difference smoothing of the drive table across speed
    and throttle, for every candidate throttle lag; the lag with the smallest
    data residual wins. The steering table and lag are bounded nonlinear least
    squares on measured yaw rate above 0.5 m/s, also smoothed. Requires SciPy.

    Args:
        segments: Logged drives, e.g. from :func:`segments_from_recording`.
        wheelbase: Axle distance in metres.
        template: Supplies the table knots; defaults to the module knots.
        throttle_lags: Candidate throttle time constants in seconds.
        smoothing: Weight of the table smoothness penalties.
        min_speed: Longitudinal samples need this speed before and after, m/s.

    Returns:
        Parameters and a fit report with residual RMS values and sample counts.
    """
    if not segments:
        raise ValueError("Provide at least one segment")
    optimize = importlib.import_module("scipy.optimize")
    knots = (
        template if template is not None else VehicleModelParams(wheelbase=wheelbase)
    )
    count_t, count_u = len(knots.throttle_knots), len(knots.speed_knots)
    cells = count_t * count_u
    smoothed = cells + count_t
    # Smooth the forward table across speed and throttle and the reverse table
    # across throttle, so cells without data interpolate their neighbours.
    penalty = np.vstack(
        [
            np.pad(
                np.kron(np.eye(count_t), _second_differences(count_u)),
                ((0, 0), (0, count_t)),
            ),
            np.pad(
                np.kron(_second_differences(count_t), np.eye(count_u)),
                ((0, 0), (0, count_t)),
            ),
            np.pad(_second_differences(count_t), ((0, 0), (cells, 0))),
        ]
    )
    best: tuple[float, float, NDArray[np.float64], int] | None = None
    for lag in throttle_lags:
        design, target = _longitudinal_design(segments, knots, float(lag), min_speed)
        weight = smoothing * math.sqrt(len(target) / cells)
        augmented = np.vstack(
            [
                design,
                np.pad(weight * penalty, ((0, 0), (0, design.shape[1] - smoothed))),
            ]
        )
        padded = np.concatenate([target, np.zeros(len(penalty))])
        coefficients, _ = optimize.nnls(
            augmented, padded, maxiter=50 * augmented.shape[1]
        )
        residual = float(np.linalg.norm(design @ coefficients - target))
        if best is None or residual < best[0]:
            best = (residual, float(lag), coefficients, len(target))
    assert best is not None
    residual, throttle_lag, coefficients, samples = best
    drive = coefficients[:cells].reshape(count_t, count_u)
    reverse = coefficients[cells : cells + count_t]
    rolling, linear, quadratic, brake, hand = coefficients[cells + count_t :]

    lateral_penalty = _second_differences(count_u)

    def lateral_residual(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        gains, lag = theta[:-1], theta[-1]
        trial = VehicleModelParams(
            wheelbase=wheelbase, speed_knots=knots.speed_knots, steer_gain=gains
        )
        parts = []
        for segment in segments:
            speed = segment.speed[1:]
            steer = _lag(segment.steer, lag, segment.dt)
            mask = np.abs(speed) > 0.5
            parts.append((_yaw_rate(trial, speed, steer) - segment.yaw_rate[1:])[mask])
        data = np.concatenate(parts)
        scale = smoothing * math.sqrt(len(data) / count_u) * 0.05
        return np.concatenate([data, scale * (lateral_penalty @ gains)])

    lateral = optimize.least_squares(
        lateral_residual,
        x0=np.r_[np.full(count_u, 0.3), 0.1],
        bounds=(np.r_[np.full(count_u, 0.005), 0.0], np.r_[np.full(count_u, 1.4), 1.0]),
    )
    gains, steer_lag = lateral.x[:-1].copy(), float(lateral.x[-1])
    # Knots without steering data hold the nearest identified gain rather than
    # extrapolating the smoothing trend, which can drive gains to zero.
    speeds = np.concatenate([np.abs(segment.speed[1:]) for segment in segments])
    steers = np.concatenate([np.abs(segment.steer) for segment in segments])
    steering = (steers > 0.02) & (speeds > 0.5)
    support = _hats(speeds[steering], knots.speed_knots).sum(axis=0)
    identified = support >= 0.01 * support.max() if steering.any() else support > 0
    if identified.any():
        known = np.flatnonzero(identified)
        for index in np.flatnonzero(~identified):
            gains[index] = gains[known[np.argmin(np.abs(known - index))]]
    params = VehicleModelParams(
        wheelbase=wheelbase,
        speed_knots=knots.speed_knots,
        throttle_knots=knots.throttle_knots,
        drive=drive,
        reverse_drive=reverse,
        steer_gain=gains,
        rolling=rolling,
        linear_drag=linear,
        quadratic_drag=quadratic,
        brake_decel=brake,
        hand_brake_decel=hand,
        throttle_lag=throttle_lag,
        steer_lag=steer_lag,
    )
    data_rows = len(lateral.fun) - len(lateral_penalty)
    throttles = np.concatenate([segment.throttle for segment in segments])
    reversing = np.concatenate([segment.reverse > 0.5 for segment in segments])
    report = {
        "steer_knots_identified": [bool(value) for value in identified],
        "envelope": {
            "max_speed_mps": float(speeds.max()),
            "steering_speed_mps": [
                float(np.percentile(speeds[steering], 5)) if steering.any() else 0.0,
                float(np.percentile(speeds[steering], 95)) if steering.any() else 0.0,
            ],
            "max_steer": float(steers.max()),
            "max_reverse_throttle": float(
                throttles[reversing].max() if reversing.any() else 0.0
            ),
        },
        "longitudinal_samples": samples,
        "longitudinal_rms_mps2": residual / math.sqrt(max(samples, 1)),
        "lateral_samples": int(data_rows),
        "lateral_rms_deg_s": float(
            np.degrees(np.sqrt(np.mean(lateral.fun[:data_rows] ** 2)))
        ),
        "segments": len(segments),
        "seconds": float(sum(segment.duration for segment in segments)),
        "smoothing": smoothing,
    }
    return params, report


def evaluate_vehicle_model(
    params: VehicleModelParams,
    segments: Sequence[DriveSegment],
    *,
    horizons: Sequence[float] = (1.0, 3.0),
    group: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Open-loop errors of the surrogate on logged drives.

    Each segment is rolled out from its initial state with its logged
    commands. Position errors at each horizon are compared with constant
    velocity extrapolation, the usual trivial baseline.

    Args:
        params: Surrogate parameters.
        segments: Logged drives not used for fitting.
        horizons: Seconds after each segment start for position errors.
        group: Maps a segment name to a group for per-group summaries.

    Returns:
        Per-segment metrics and duration-weighted means, overall and by group.
    """
    rows = []
    for segment in segments:
        rollout = simulate(params, segment)
        time = np.arange(len(segment.x)) * segment.dt
        speed_error = rollout["speed"] - segment.speed
        rate_error = np.degrees(rollout["yaw_rate"] - segment.yaw_rate)
        row: dict[str, Any] = {
            "name": segment.name,
            "seconds": segment.duration,
            "speed_rmse_mps": float(np.sqrt(np.mean(speed_error**2))),
            "peak_speed_mps": float(np.max(np.abs(segment.speed))),
            "yaw_rate_rmse_deg_s": float(np.sqrt(np.mean(rate_error**2))),
            "peak_yaw_rate_deg_s": float(np.degrees(np.max(np.abs(segment.yaw_rate)))),
            "heading_change_deg": float(
                (segment.yaw_deg[-1] - segment.yaw_deg[0] + 180) % 360 - 180
            ),
            "final_heading_error_deg": float(
                (rollout["yaw_deg"][-1] - segment.yaw_deg[-1] + 180) % 360 - 180
            ),
            "final_position_error_m": float(
                math.hypot(
                    rollout["x"][-1] - segment.x[-1], rollout["y"][-1] - segment.y[-1]
                )
            ),
        }
        heading = math.radians(float(segment.yaw_deg[0]))
        velocity = segment.speed[0] * np.array([math.cos(heading), math.sin(heading)])
        for horizon in horizons:
            index = int(round(horizon / segment.dt))
            if index >= len(time):
                continue
            actual = np.array([segment.x[index], segment.y[index]])
            model = np.array([rollout["x"][index], rollout["y"][index]])
            baseline = np.array([segment.x[0], segment.y[0]]) + velocity * time[index]
            row[f"position_error_{horizon:g}s_m"] = float(
                np.linalg.norm(model - actual)
            )
            row[f"baseline_error_{horizon:g}s_m"] = float(
                np.linalg.norm(baseline - actual)
            )
        rows.append(row)

    def summarise(selected: list[dict[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for key in selected[0]:
            if key in ("name", "seconds"):
                continue
            present = [
                (abs(row[key]), row["seconds"]) for row in selected if key in row
            ]
            values = np.asarray(present)
            out[key] = float(np.sum(values[:, 0] * values[:, 1]) / np.sum(values[:, 1]))
        out["seconds"] = float(sum(row["seconds"] for row in selected))
        out["segments"] = len(selected)
        return out

    result: dict[str, Any] = {
        "segments": rows,
        "summary": summarise(rows) if rows else {},
    }
    if group is not None and rows:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(group(row["name"]), []).append(row)
        result["by_group"] = {
            name: summarise(items) for name, items in sorted(groups.items())
        }
    return result


#: CARLA 0.10.0 (UE5.5) ``vehicle.nissan.patrol`` on Town10HD_Opt, fitted by
#: ``examples/fit_vehicle_model.py`` to seeds 0 and 3 of
#: ``evidence/ue55-vehicle-sysid.zip``. Held-out errors are in
#: ``evidence/ue55-vehicle-surrogate.json`` and VALIDATION.md; steering is
#: identified from about 2.4 to 13.6 m/s and reverse up to throttle 0.53.
_PATROL: dict[str, Any] = {
    "wheelbase": 3.075,
    "speed_knots": [0, 1, 2.5, 4, 6, 8, 11, 15, 20],
    "throttle_knots": [0.25, 0.5, 0.75, 1],
    "drive": [
        [
            0,
            0.168804,
            0.350664,
            0.523192,
            0.700473,
            0.918991,
            1.21782,
            1.59201,
            1.99169,
        ],
        [
            0.671293,
            1.05237,
            1.51458,
            2.08456,
            2.56769,
            2.8009,
            2.80264,
            2.79174,
            2.77574,
        ],
        [1.37009, 1.98893, 2.76382, 3.74813, 4.577, 4.86496, 4.55806, 4.06629, 3.54201],
        [
            2.10678,
            2.96627,
            4.03232,
            5.49725,
            6.75196,
            7.12467,
            6.47271,
            5.36556,
            4.28336,
        ],
    ],
    "reverse_drive": [0.14466, 0.98024, 1.8164, 2.65256],
    "steer_gain": [
        0.489392,
        0.489392,
        0.386428,
        0.282451,
        0.261281,
        0.31202,
        0.205496,
        0.101791,
        0.101791,
    ],
    "rolling": 0.287323,
    "linear_drag": 0,
    "quadratic_drag": 0.00660362,
    "brake_decel": 6.36968,
    "hand_brake_decel": 4.92694,
    "throttle_lag": 0.075,
    "steer_lag": 0.037593,
}


def carla_nissan_patrol() -> VehicleModelParams:
    """Return the surrogate identified for CARLA 0.10.0's Nissan Patrol."""
    return VehicleModelParams.from_dict(_PATROL)

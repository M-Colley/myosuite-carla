"""Constrained longitudinal refinement using complete vehicle speed rollouts."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .vehicle_model import DriveSegment, VehicleModelParams, simulate


@dataclass(frozen=True)
class LongitudinalRolloutFitConfig:
    """Small bounded correction to an existing physical vehicle model.

    Two positive drive gains act at 15 and 20 m/s. The gain stays at one
    through 8 m/s. Positive speed gains preserve the original table's
    monotonicity in throttle. Remaining parameters scale quadratic drag and
    service braking, and replace the throttle lag. Steering stays unchanged.
    """

    lower: tuple[float, ...] = (0.5, 0.5, 0.5, 0.75, 0.0)
    upper: tuple[float, ...] = (1.5, 1.5, 1.5, 1.25, 0.3)
    prior_normalization: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 0.075)
    regularization_weight: float = 0.01
    max_nfev: int = 80
    ftol: float = 1e-7
    xtol: float = 1e-7
    gtol: float = 1e-7

    def __post_init__(self) -> None:
        low, high, normal = map(
            np.asarray, (self.lower, self.upper, self.prior_normalization)
        )
        if any(values.shape != (5,) for values in (low, high, normal)):
            raise ValueError("Longitudinal fitting needs five parameter bounds")
        if (
            not all(np.all(np.isfinite(values)) for values in (low, high, normal))
            or np.any(low >= high)
            or np.any(low[:4] <= 0)
            or low[4] < 0
            or np.any(normal <= 0)
        ):
            raise ValueError("Invalid longitudinal bounds or prior normalization")
        if (
            not np.isfinite(self.regularization_weight)
            or self.regularization_weight < 0
            or self.max_nfev < 1
            or any(
                not np.isfinite(value) or value <= 0
                for value in (self.ftol, self.xtol, self.gtol)
            )
        ):
            raise ValueError("Invalid longitudinal optimizer settings")


def longitudinal_correction(
    baseline: VehicleModelParams, values: Sequence[float]
) -> VehicleModelParams:
    """Apply positive drive and resistance corrections without changing steering."""
    theta: NDArray[np.float64] = np.asarray(values, dtype=float)
    if theta.shape != (5,) or not np.all(np.isfinite(theta)):
        raise ValueError("Provide five finite longitudinal correction values")
    if np.any(theta[:4] <= 0) or theta[4] < 0:
        raise ValueError("Drive and deceleration scales must be positive")
    if np.any(np.diff(np.asarray(baseline.drive), axis=0) < 0):
        raise ValueError("The baseline drive table must be monotonic in throttle")
    gains = np.interp(
        baseline.speed_knots, (0.0, 8.0, 15.0, 20.0), (1.0, 1.0, theta[0], theta[1])
    )
    drive = np.asarray(baseline.drive) * gains[None, :]
    return replace(
        baseline,
        drive=tuple(tuple(float(value) for value in row) for row in drive),
        quadratic_drag=baseline.quadratic_drag * float(theta[2]),
        brake_decel=baseline.brake_decel * float(theta[3]),
        throttle_lag=float(theta[4]),
    )


def fit_longitudinal_rollout(
    segments: Sequence[DriveSegment],
    baseline: VehicleModelParams,
    *,
    segment_weights: Sequence[float] | None = None,
    config: LongitudinalRolloutFitConfig | None = None,
) -> tuple[VehicleModelParams, dict[str, Any]]:
    """Refine five longitudinal parameters with SciPy bounded least squares.

    Every residual uses a full rollout from the recorded initial state.
    Future measured speed and pitch are excluded. Segment weights determine
    each trajectory's contribution to the speed MSE. A small normalized
    parameter penalty retains support from the baseline calibration.
    """
    if not segments:
        raise ValueError("Provide at least one complete drive segment")
    settings = config if config is not None else LongitudinalRolloutFitConfig()
    weights: NDArray[np.float64] = np.asarray(
        np.ones(len(segments)) if segment_weights is None else segment_weights,
        dtype=float,
    )
    if (
        weights.shape != (len(segments),)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        raise ValueError("Provide finite nonnegative weights for every segment")
    weights = weights / weights.sum()
    initial = np.array((1.0, 1.0, 1.0, 1.0, baseline.throttle_lag))
    low, high = np.asarray(settings.lower), np.asarray(settings.upper)
    if np.any(initial < low) or np.any(initial > high):
        raise ValueError("The baseline parameters must lie within the fit bounds")
    longitudinal_correction(baseline, initial)
    scales = [
        np.sqrt(weight / len(segment.speed))
        for weight, segment in zip(weights, segments)
    ]
    prior_scale = np.sqrt(settings.regularization_weight / len(initial))

    def data_residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        params = longitudinal_correction(baseline, values)
        return np.concatenate(
            [
                (simulate(params, segment, use_pitch=False)["speed"] - segment.speed)
                * scale
                for segment, scale in zip(segments, scales)
            ]
        )

    def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
        prior = prior_scale * (values - initial) / settings.prior_normalization
        return np.concatenate((data_residual(values), prior))

    optimize = importlib.import_module("scipy.optimize")
    result = optimize.least_squares(
        residual,
        initial,
        bounds=(low, high),
        method="trf",
        loss="linear",
        x_scale="jac",
        max_nfev=settings.max_nfev,
        ftol=settings.ftol,
        xtol=settings.xtol,
        gtol=settings.gtol,
    )
    params = longitudinal_correction(baseline, result.x)
    return params, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
        "njev": int(result.njev) if result.njev is not None else None,
        "parameters": result.x.tolist(),
        "settings": asdict(settings),
        "baseline_speed_mse": float(np.sum(data_residual(initial) ** 2)),
        "fitted_speed_mse": float(np.sum(data_residual(result.x) ** 2)),
        "penalized_objective": float(2.0 * result.cost),
        "segments": len(segments),
        "states": sum(len(segment.speed) for segment in segments),
        "segment_weights": weights.tolist(),
        "use_pitch": False,
        "measured_speed": False,
        "monotonic_drive": True,
    }

"""Joint fit of the steering rise rate and lag of design.md.

The gain table, exponent and longitudinal parameters stay at
``carla_nissan_patrol()``; the fall rate is fixed at 5.0 per second. The
objective is the lateral fit's: the pooled squared yaw-rate residual with the
measured speed over frames above 0.5 m/s. It is tabulated per segment over a
grid of rise rates and lags, so the grid minimum and its segment bootstrap
need no refitting. The rate limit is the library's; the lag is the library's
first-order lag written as a filter.

Usage: python fit.py --sysid artifacts/vehicle-sysid --wide
artifacts/vehicle-sysid-wide --fresh artifacts/steer-rate-limit --output OUT.json
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import signal

from myosuite_carla import Recording
from myosuite_carla.vehicle_model import (
    _rate_limit,
    carla_nissan_patrol,
    evaluate_vehicle_model,
    segments_from_recording,
)

RISES = np.round(np.arange(1.0, 5.0001, 0.05), 2)
LAGS = np.round(np.arange(0.0, 0.1001, 0.002), 3)
FALL = 5.0
BOOTSTRAP = 1000
SEED = 20261104
SPLIT = 3.0


def kind(name):
    return name.rsplit("-", 1)[-1]


def load(directories):
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
        if kind(segment.name).startswith("steer")
    ]


def lagged(values, tau, dt):
    """The library's ``_lag``: a first-order lag steady at its first value."""
    alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
    zi = [(1.0 - alpha) * values[0]]
    return signal.lfilter([alpha], [1.0, -(1.0 - alpha)], values, zi=zi)[0]


def residuals(params, segment, gains, steer):
    angle = np.clip(gains * params.steer_input(steer), -1.4, 1.4)
    speed = segment.speed[1:]
    return speed * np.tan(angle) / params.wheelbase - segment.yaw_rate[1:]


def steering_speed(segment):
    steering = np.abs(segment.steer) > 0
    return float(np.mean(np.abs(segment.speed[1:][steering])))


def tabulate(params, segments):
    """Masked squared residual per segment, rise rate and lag."""
    table = np.zeros((len(segments), len(RISES), len(LAGS)))
    for index, segment in enumerate(segments):
        gains = params.road_wheel_angle(segment.speed[1:])
        mask = np.abs(segment.speed[1:]) > 0.5
        for row, rise in enumerate(RISES):
            limited = _rate_limit(segment.steer, float(rise), FALL, segment.dt)
            for column, lag in enumerate(LAGS):
                error = residuals(
                    params, segment, gains, lagged(limited, lag, segment.dt)
                )
                table[index, row, column] = np.sum(error[mask] ** 2)
    return table


def minimum(total):
    row, column = np.unravel_index(np.argmin(total), total.shape)
    return float(RISES[row]), float(LAGS[column])


def fit(table, frames, rng):
    """Grid minimum, bootstrap intervals and profiles of a segment set."""
    rise, lag = minimum(table.sum(axis=0))
    draws = []
    for _ in range(BOOTSTRAP):
        counts = np.bincount(
            rng.integers(len(table), size=len(table)), minlength=len(table)
        )
        draws.append(minimum(np.tensordot(counts, table, axes=1)))
    draws = np.asarray(draws)
    total = table.sum(axis=0)
    row, column = (
        int(np.argmin(np.abs(RISES - rise))),
        int(np.argmin(np.abs(LAGS - lag))),
    )

    def rmse(values):
        return [float(np.degrees(math.sqrt(v / frames))) for v in values]

    return {
        "segments": len(table),
        "frames": int(frames),
        "rise_rate": rise,
        "lag_s": lag,
        "rmse_deg_s": rmse([total[row, column]])[0],
        "rise_rate_95": [float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
        "lag_s_95": [float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])],
        "rise_profile_at_lag": dict(
            zip(map(float, RISES), rmse(total[:, column]), strict=True)
        ),
        "lag_profile_at_rise": dict(
            zip(map(float, LAGS), rmse(total[row, :]), strict=True)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default = carla_nissan_patrol()
    segments = load(
        [args.sysid / "seed0", args.sysid / "seed3"]
        + [args.wide / f"seed{seed}" for seed in (10, 12)]
        + [args.fresh / f"seed{seed}" for seed in (14, 15, 16, 17)]
    )

    # The fast path must reproduce the library at a point off the grid.
    engine = replace(default, steer_rise_rate=2.5, steer_fall_rate=FALL)
    library = evaluate_vehicle_model(engine, segments, measured_speed=True)
    largest = 0.0
    for segment, row in zip(segments, library["segments"], strict=True):
        limited = _rate_limit(segment.steer, 2.5, FALL, segment.dt)
        steer = lagged(limited, default.steer_lag, segment.dt)
        gains = default.road_wheel_angle(segment.speed[1:])
        error = residuals(default, segment, gains, steer)
        fast = math.degrees(math.sqrt(np.sum(error**2) / (len(error) + 1)))
        largest = max(largest, abs(fast - row["yaw_rate_rmse_deg_s"]))
    if largest > 1e-9:
        raise SystemExit(f"fast path differs from the library by {largest}")

    table = tabulate(default, segments)
    masks = np.array([np.sum(np.abs(segment.speed[1:]) > 0.5) for segment in segments])
    rng = np.random.default_rng(SEED)
    report = {
        "fast_path_largest_difference_deg_s": largest,
        "training": {
            "kinds": {
                name: sum(kind(s.name) == name for s in segments)
                for name in sorted({kind(s.name) for s in segments})
            },
        },
        "fall_rate": FALL,
        "grid": {
            "rises": [float(RISES[0]), float(RISES[-1]), 0.05],
            "lags_s": [float(LAGS[0]), float(LAGS[-1]), 0.002],
        },
        "all": fit(table, masks.sum(), rng),
    }
    slow = np.array([steering_speed(segment) < SPLIT for segment in segments])
    report["below_3_mps"] = fit(table[slow], masks[slow].sum(), rng)
    report["at_or_above_3_mps"] = fit(table[~slow], masks[~slow].sum(), rng)

    # Reference points on the training objective.
    def reference(params):
        total = 0.0
        for segment in segments:
            limited = _rate_limit(
                segment.steer,
                params.steer_rise_rate,
                params.steer_fall_rate,
                segment.dt,
            )
            steer = lagged(limited, params.steer_lag, segment.dt)
            gains = params.road_wheel_angle(segment.speed[1:])
            error = residuals(params, segment, gains, steer)
            total += np.sum(error[np.abs(segment.speed[1:]) > 0.5] ** 2)
        return float(np.degrees(math.sqrt(total / masks.sum())))

    report["training_rmse_deg_s"] = {
        "default": reference(default),
        "engine_rates_default_lag": reference(engine),
    }
    candidate = replace(
        default,
        steer_rise_rate=report["all"]["rise_rate"],
        steer_fall_rate=FALL,
        steer_lag=report["all"]["lag_s"],
    )
    report["candidate"] = candidate.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for name in ("all", "below_3_mps", "at_or_above_3_mps"):
        values = report[name]
        print(
            name,
            {
                key: values[key]
                for key in (
                    "segments",
                    "rise_rate",
                    "rise_rate_95",
                    "lag_s",
                    "lag_s_95",
                    "rmse_deg_s",
                )
            },
        )
    print("training rmse", report["training_rmse_deg_s"])
    print("fast path", largest)


if __name__ == "__main__":
    main()

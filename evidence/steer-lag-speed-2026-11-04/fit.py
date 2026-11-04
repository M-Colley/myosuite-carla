"""Joint fit of the steer lag and its speed slope of design.md.

The rate limit is fixed at the engine's rates (rise 2.5, fall 5.0 per second);
the gain table, exponent and longitudinal parameters stay at
``carla_nissan_patrol()``. The objective is the lateral fit's: the pooled
squared yaw-rate residual with the measured speed over frames above 0.5 m/s.
The lag's time constant is ``lag + slope |u|`` with the measured speed before
each frame. The residual is tabulated per segment over the whole grid at once
(time runs in Python, the grid in NumPy), so the grid minimum and its segment
bootstrap need no refitting.

Usage: python fit.py --sysid artifacts/vehicle-sysid --wide
artifacts/vehicle-sysid-wide --first artifacts/steer-rate-limit --second
artifacts/steer-rate-lag --previous FIT.json --output OUT.json
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from myosuite_carla import Recording
from myosuite_carla.vehicle_model import (
    VehicleModelParams,
    _rate_limit,
    carla_nissan_patrol,
    evaluate_vehicle_model,
    segments_from_recording,
)

LAGS = np.round(np.arange(0.0, 0.0601, 0.002), 3)
SLOPES = np.round(np.arange(0.0, 0.00801, 0.0002), 4)
RISE, FALL = 2.5, 5.0
BOOTSTRAP = 1000
SEED = 20261104
CHECK = (0.0305, 0.0021)


def kind(name):
    return name.rsplit("-", 1)[-1]


def load(directories):
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
        if kind(segment.name).startswith("steer")
    ]


def residuals(params, segment, lags, slopes):
    """Yaw-rate residuals per frame, one column per (lag, slope) pair."""
    commands = _rate_limit(segment.steer, RISE, FALL, segment.dt)
    before, after = np.abs(segment.speed[:-1]), segment.speed[1:]
    gains = params.road_wheel_angle(after)
    state = np.full(len(lags), float(commands[0]))
    out = np.empty((len(commands), len(lags)))
    for index, command in enumerate(commands):
        tau = lags + slopes * before[index]
        with np.errstate(divide="ignore"):
            alpha = np.where(tau > 0, 1.0 - np.exp(-segment.dt / tau), 1.0)
        state = state + alpha * (command - state)
        angle = np.clip(gains[index] * params.steer_input(state), -1.4, 1.4)
        out[index] = after[index] * np.tan(angle) / params.wheelbase
    return out - segment.yaw_rate[1:, None]


def minimum(total):
    row, column = np.unravel_index(np.argmin(total), total.shape)
    return float(LAGS[row]), float(SLOPES[column])


def rmse(value, frames):
    return float(np.degrees(math.sqrt(value / frames)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default = carla_nissan_patrol()
    segments = load(
        [args.sysid / "seed0", args.sysid / "seed3"]
        + [args.wide / f"seed{seed}" for seed in (10, 12)]
        + [args.first / f"seed{seed}" for seed in (14, 15, 16, 17)]
        + [args.second / f"seed{seed}" for seed in (18, 19, 20, 21)]
    )

    # The fast path must reproduce the library at a point off the grid.
    lag, slope = CHECK
    probe = replace(
        default,
        steer_rise_rate=RISE,
        steer_fall_rate=FALL,
        steer_lag=lag,
        steer_lag_slope=slope,
    )
    library = evaluate_vehicle_model(probe, segments, measured_speed=True)
    largest = 0.0
    for segment, row in zip(segments, library["segments"], strict=True):
        error = residuals(default, segment, np.array([lag]), np.array([slope]))[:, 0]
        fast = math.degrees(math.sqrt(np.sum(error**2) / (len(error) + 1)))
        largest = max(largest, abs(fast - row["yaw_rate_rmse_deg_s"]))
    if largest > 1e-9:
        raise SystemExit(f"fast path differs from the library by {largest}")

    lags, slopes = (grid.ravel() for grid in np.meshgrid(LAGS, SLOPES, indexing="ij"))
    table = np.empty((len(segments), len(LAGS), len(SLOPES)))
    frames = 0
    for index, segment in enumerate(segments):
        mask = np.abs(segment.speed[1:]) > 0.5
        error = residuals(default, segment, lags, slopes)[mask]
        table[index] = np.sum(error**2, axis=0).reshape(len(LAGS), len(SLOPES))
        frames += int(mask.sum())

    total = table.sum(axis=0)
    best_lag, best_slope = minimum(total)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        counts = np.bincount(
            rng.integers(len(table), size=len(table)), minlength=len(table)
        )
        draws.append(minimum(np.tensordot(counts, table, axes=1)))
    draws = np.asarray(draws)
    row = int(np.argmin(np.abs(LAGS - best_lag)))
    column = int(np.argmin(np.abs(SLOPES - best_slope)))
    zero = int(np.argmin(total[:, 0]))

    previous = VehicleModelParams.from_dict(
        json.loads(args.previous.read_text(encoding="utf-8"))["candidate"]
    )
    candidate = replace(
        default,
        steer_rise_rate=RISE,
        steer_fall_rate=FALL,
        steer_lag=best_lag,
        steer_lag_slope=best_slope,
    )
    references = {}
    for name, params in (
        ("default", default),
        (
            "engine_rates_default_lag",
            replace(probe, steer_lag=default.steer_lag, steer_lag_slope=0.0),
        ),
        ("previous_constant_lag", previous),
        ("candidate", candidate),
    ):
        result = evaluate_vehicle_model(params, segments, measured_speed=True)
        references[name] = result["summary"]["yaw_rate_rmse_deg_s"]

    report = {
        "fast_path_largest_difference_deg_s": largest,
        "training": {
            "segments": len(segments),
            "frames_above_0.5_mps": frames,
            "kinds": {
                name: sum(kind(s.name) == name for s in segments)
                for name in sorted({kind(s.name) for s in segments})
            },
        },
        "rates": [RISE, FALL],
        "grid": {
            "lags_s": [float(LAGS[0]), float(LAGS[-1]), 0.002],
            "slopes_s_per_mps": [float(SLOPES[0]), float(SLOPES[-1]), 0.0002],
        },
        "lag_s": best_lag,
        "slope_s_per_mps": best_slope,
        "lag_s_95": [float(v) for v in np.percentile(draws[:, 0], [2.5, 97.5])],
        "slope_s_per_mps_95": [
            float(v) for v in np.percentile(draws[:, 1], [2.5, 97.5])
        ],
        "slope_interval_excludes_zero": bool(np.percentile(draws[:, 1], 2.5) > 0),
        "time_constant_s": {
            f"{speed:g} m/s": best_lag + best_slope * speed
            for speed in (1.0, 2.0, 5.0, 10.0, 14.0)
        },
        "masked_rmse_deg_s": {
            "candidate": rmse(total[row, column], frames),
            "best_constant_lag": rmse(total[zero, 0], frames),
            "best_constant_lag_s": float(LAGS[zero]),
        },
        "evaluate_measured_speed_rmse_deg_s": references,
        "slope_profile_at_lag": {
            float(value): rmse(total[row, index], frames)
            for index, value in enumerate(SLOPES)
        },
        "lag_profile_at_slope": {
            float(value): rmse(total[index, column], frames)
            for index, value in enumerate(LAGS)
        },
        "candidate": candidate.as_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for key in (
        "training",
        "lag_s",
        "lag_s_95",
        "slope_s_per_mps",
        "slope_s_per_mps_95",
        "slope_interval_excludes_zero",
        "time_constant_s",
        "masked_rmse_deg_s",
        "evaluate_measured_speed_rmse_deg_s",
        "fast_path_largest_difference_deg_s",
    ):
        print(key, json.dumps(report[key]))


if __name__ == "__main__":
    main()

"""Post-hoc diagnosis of the rejected refit; not part of the fixed design.

1. Speed or steering: the yaw-rate error of each model with the measured speed
   (as in the lateral fit, |v| > 0.5 m/s) against its open-loop error.
2. Tyre relaxation: the current default's steering map followed by a lag of
   the road-wheel angle with time constant ``sigma / v``. Only ``sigma`` is
   fitted, on training seeds 10 and 12 with the measured speed, and evaluated
   on held-out seeds 11 and 13 and the original held-out seeds 1, 2 and 4.
3. Input rate limit: Chaos's ``SteeringInputRate`` (UE5.8 engine defaults
   RiseRate 2.5/s while the magnitude grows, FallRate 5.0/s) applied to the
   command before the steering curve, with and without the default lag. No
   parameter is fitted.

Usage: python diagnose.py --wide artifacts/vehicle-sysid-wide --sysid
artifacts/vehicle-sysid --refit REFIT.json --output OUT.json
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import optimize

from myosuite_carla import Recording
from myosuite_carla.vehicle_model import (
    VehicleModelParams,
    carla_nissan_patrol,
    evaluate_vehicle_model,
    segments_from_recording,
)


def load(directories, steering_only=True):
    segments = [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
    ]
    if steering_only:
        segments = [
            s for s in segments if s.name.rsplit("-", 1)[-1].startswith("steer")
        ]
    return segments


def lagged(values, tau, dt):
    alpha = 1.0 if tau <= 0 else 1.0 - math.exp(-dt / tau)
    out, state = np.empty(len(values)), float(values[0])
    for index, value in enumerate(values):
        state += alpha * (float(value) - state)
        out[index] = state
    return out


def rate_limited(values, dt, rise=2.5, fall=5.0):
    """Chaos FVehicleInputRateConfig::InterpInputValue on a held command."""
    out, current = np.empty(len(values)), 0.0
    for index, value in enumerate(values):
        delta = float(value) - current
        rising = ((delta > 0) == (current > 0)) or (delta != 0 and current == 0)
        limit = dt * (rise if rising else fall)
        current += min(max(delta, -limit), limit)
        out[index] = current
    return out


def forced_rates(params, segment, sigma=0.0, limit=False):
    """Yaw rate with the measured speed; sigma > 0 lags the road-wheel angle."""
    speed = segment.speed[1:]
    command = rate_limited(segment.steer, segment.dt) if limit else segment.steer
    steer = lagged(command, params.steer_lag, segment.dt)
    angle = np.clip(
        params.road_wheel_angle(speed) * params.steer_input(steer), -1.4, 1.4
    )
    if sigma > 0:
        state = float(angle[0])
        for index, target in enumerate(angle):
            rate = abs(float(speed[index])) / sigma
            state += (1.0 - math.exp(-segment.dt * rate)) * (float(target) - state)
            angle[index] = state
    return speed * np.tan(angle) / params.wheelbase


def forced_rmse(params, segments, sigma=0.0, limit=False):
    """Duration-weighted per-segment yaw-rate RMSE in deg/s with measured speed."""
    total, seconds = 0.0, 0.0
    for segment in segments:
        mask = np.abs(segment.speed[1:]) > 0.5
        if not mask.any():
            continue
        rates = forced_rates(params, segment, sigma, limit)
        error = np.degrees(rates - segment.yaw_rate[1:])
        total += float(np.sqrt(np.mean(error[mask] ** 2))) * segment.duration
        seconds += segment.duration
    return total / seconds


def pooled_residual(params, segments, sigma):
    parts = []
    for segment in segments:
        mask = np.abs(segment.speed[1:]) > 0.5
        parts.append(
            (forced_rates(params, segment, sigma) - segment.yaw_rate[1:])[mask]
        )
    return float(np.sum(np.concatenate(parts) ** 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--refit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default = carla_nissan_patrol()
    refit = VehicleModelParams.from_dict(
        json.loads(args.refit.read_text(encoding="utf-8"))["params"]
    )
    train = load([args.wide / "seed10", args.wide / "seed12"])
    held = [
        s
        for s in load([args.wide / "seed11", args.wide / "seed13"])
        if not s.name.endswith("-steer")
    ]
    original = load([args.sysid / f"seed{seed}" for seed in (1, 2, 4)])
    report = {
        "segments": {
            "train": len(train),
            "held_new": len(held),
            "original": len(original),
        }
    }

    split = {}
    for name, params in (("default", default), ("refit", refit)):
        split[name] = {
            "new_open_loop": evaluate_vehicle_model(params, held)["summary"][
                "yaw_rate_rmse_deg_s"
            ],
            "new_measured_speed": forced_rmse(params, held),
            "original_open_loop": evaluate_vehicle_model(params, original)["summary"][
                "yaw_rate_rmse_deg_s"
            ],
            "original_measured_speed": forced_rmse(params, original),
        }
    report["speed_or_steering"] = split

    fit = optimize.minimize_scalar(
        lambda sigma: pooled_residual(default, train, sigma),
        bounds=(0.0, 3.0),
        method="bounded",
        options={"xatol": 1e-3},
    )
    sigma = float(fit.x)
    report["relaxation"] = {
        "sigma_m": sigma,
        "train_measured_speed": {
            "without": forced_rmse(default, train),
            "with": forced_rmse(default, train, sigma),
        },
        "new_measured_speed": {
            "without": forced_rmse(default, held),
            "with": forced_rmse(default, held, sigma),
        },
        "original_measured_speed": {
            "without": forced_rmse(default, original),
            "with": forced_rmse(default, original, sigma),
        },
    }
    unlagged = replace(default, steer_lag=0.0)
    report["rate_limit"] = {
        name: {
            "without": forced_rmse(default, segments),
            "with": forced_rmse(default, segments, limit=True),
            "with_no_lag": forced_rmse(unlagged, segments, limit=True),
        }
        for name, segments in (
            ("train_measured_speed", train),
            ("new_measured_speed", held),
            ("original_measured_speed", original),
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

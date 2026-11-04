"""Post-hoc look at the low-speed sweeps; not part of the fixed design.

The rate limit failed check (b) of design.md only on ``steer_low``. This
script reports, with the measured speed: each ``steer_low`` segment's error
with and without the limit against its amplitude, frequency, peak command
rate and speed; the rise-rate profile of each kind (fall fixed at 5); and the
limit without the steering lag, which was fitted without the limit.

Usage: python diagnose.py --fresh artifacts/steer-rate-limit --sysid
artifacts/vehicle-sysid --output OUT.json
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from myosuite_carla import Recording
from myosuite_carla.vehicle_model import (
    carla_nissan_patrol,
    evaluate_vehicle_model,
    segments_from_recording,
)

SEEDS = (14, 15, 16, 17)
NEW = ("steer_low", "steer_hold", "steer_doublet")


def load(directories):
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
    ]


def kind(name):
    return name.rsplit("-", 1)[-1]


def by_kind(params, segments):
    result = evaluate_vehicle_model(params, segments, group=kind, measured_speed=True)
    return {
        name: values["yaw_rate_rmse_deg_s"]
        for name, values in result["by_group"].items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default = carla_nissan_patrol()
    limited = replace(default, steer_rise_rate=2.5, steer_fall_rate=5.0)
    directories = [args.fresh / f"seed{seed}" for seed in SEEDS]
    new = [s for s in load(directories) if kind(s.name) in NEW]
    original = load([args.sysid / f"seed{seed}" for seed in (1, 2, 4)])
    logs = {}
    for directory in directories:
        log = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
        logs.update({row["segment"]: row for row in log["segments"]})

    low = [s for s in new if kind(s.name) == "steer_low"]
    before = evaluate_vehicle_model(default, low, measured_speed=True)["segments"]
    after = evaluate_vehicle_model(limited, low, measured_speed=True)["segments"]
    rows = []
    for segment, base, candidate in zip(low, before, after):
        log = logs[segment.name]
        steering = np.abs(segment.steer) > 0
        rows.append(
            {
                "segment": segment.name,
                "amplitude": log["steer_amplitude"],
                "frequency_hz": log["steer_frequency"],
                "peak_command_rate_per_s": 2
                * math.pi
                * log["steer_frequency"]
                * abs(log["steer_amplitude"]),
                "mean_speed_mps": float(np.mean(np.abs(segment.speed[1:][steering]))),
                "default_deg_s": base["yaw_rate_rmse_deg_s"],
                "limited_deg_s": candidate["yaw_rate_rmse_deg_s"],
            }
        )
    change = np.array([row["limited_deg_s"] - row["default_deg_s"] for row in rows])
    report = {
        "steer_low": {
            "segments": rows,
            "worse": int(np.sum(change > 0)),
            "better": int(np.sum(change < 0)),
            "correlation_of_change_with": {
                key: float(np.corrcoef([row[key] for row in rows], change)[0, 1])
                for key in (
                    "amplitude",
                    "frequency_hz",
                    "peak_command_rate_per_s",
                    "mean_speed_mps",
                )
            },
        }
    }
    rises = [float(value) for value in np.arange(0.5, 8.01, 0.25)]
    profiles = {
        rise: by_kind(replace(limited, steer_rise_rate=rise), new) for rise in rises
    }
    report["rise_profile_by_kind"] = {
        name: {
            "best_rise": min(rises, key=lambda rise: profiles[rise][name]),
            "profile": {rise: profiles[rise][name] for rise in rises},
        }
        for name in NEW
    }
    unlagged = replace(limited, steer_lag=0.0)
    report["without_lag"] = {
        "default": by_kind(default, new),
        "limited": by_kind(limited, new),
        "limited_without_lag": by_kind(unlagged, new),
        "original_default": evaluate_vehicle_model(
            default, original, measured_speed=True
        )["summary"]["yaw_rate_rmse_deg_s"],
        "original_limited_without_lag": evaluate_vehicle_model(
            unlagged, original, measured_speed=True
        )["summary"]["yaw_rate_rmse_deg_s"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    summary = {key: report["steer_low"][key] for key in ("worse", "better")}
    summary["correlation"] = report["steer_low"]["correlation_of_change_with"]
    print(json.dumps(summary, indent=1))
    for name, values in report["rise_profile_by_kind"].items():
        print(name, "best rise", values["best_rise"])
    print(json.dumps(report["without_lag"], indent=1))


if __name__ == "__main__":
    main()

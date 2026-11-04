"""Post-hoc pooling of both rate limit test sets; not part of either design.

The engine-rate model (rise 2.5, fall 5.0, default lag) failed check (b) on
seeds 14-17 only for the low-speed sweeps. This script reports the
measured-speed yaw-rate RMSE by kind on seeds 14-17, on seeds 18-21 and on
both together, for the default, the engine-rate model and the jointly fitted
candidate.

Usage: python posthoc.py --candidate FIT.json --first artifacts/steer-rate-limit
--second artifacts/steer-rate-lag --output OUT.json
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from myosuite_carla import Recording
from myosuite_carla.vehicle_model import (
    VehicleModelParams,
    carla_nissan_patrol,
    evaluate_vehicle_model,
    segments_from_recording,
)


def load(directories):
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
        if segment.name.rsplit("-", 1)[-1].startswith("steer")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default = carla_nissan_patrol()
    fitted = json.loads(args.candidate.read_text(encoding="utf-8"))["candidate"]
    models = {
        "default": default,
        "engine_rates": replace(default, steer_rise_rate=2.5, steer_fall_rate=5.0),
        "candidate": VehicleModelParams.from_dict(fitted),
    }
    first = load([args.first / f"seed{seed}" for seed in (14, 15, 16, 17)])
    second = load([args.second / f"seed{seed}" for seed in (18, 19, 20, 21)])
    report = {}
    for label, segments in (
        ("seeds 14-17", first),
        ("seeds 18-21", second),
        ("seeds 14-21", first + second),
    ):
        report[label] = {}
        for name, params in models.items():
            result = evaluate_vehicle_model(
                params,
                segments,
                group=lambda key: key.rsplit("-", 1)[-1],
                measured_speed=True,
            )
            report[label][name] = {
                group: {
                    "yaw_rate_rmse_deg_s": values["yaw_rate_rmse_deg_s"],
                    "segments": values["segments"],
                }
                for group, values in result["by_group"].items()
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    for label, values in report.items():
        print(label)
        for name, groups in values.items():
            cells = ", ".join(
                f"{group} {metrics['yaw_rate_rmse_deg_s']:.3f} ({metrics['segments']})"
                for group, metrics in groups.items()
            )
            print(f"  {name}: {cells}")


if __name__ == "__main__":
    main()

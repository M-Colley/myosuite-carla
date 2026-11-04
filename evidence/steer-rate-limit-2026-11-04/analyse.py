"""Control, adoption rule and secondary analyses of design.md.

Errors are duration-weighted means of per-segment RMSE from
``evaluate_vehicle_model``; measured-speed errors pass ``measured_speed=True``.
The default is ``carla_nissan_patrol()``; the candidate adds Chaos's steering
input rate limit (rise 2.5, fall 5.0 per second) and changes nothing else.

Usage: python analyse.py --fresh artifacts/steer-rate-limit --sysid
artifacts/vehicle-sysid --muscle DIR... --output OUT.json
"""

import argparse
import json
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
CONTROL_LIMIT = 2 * 0.337
REDUCTION = 0.25
ORIGINAL_TOLERANCE = 1.01
MUSCLE_TOLERANCE = 1.05
KEEP = (
    "yaw_rate_rmse_deg_s",
    "final_heading_error_deg",
    "final_position_error_m",
    "position_error_3s_m",
)


def load(directories):
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
    ]


def kind(name):
    return name.rsplit("-", 1)[-1]


def summary(params, segments, group=None, measured=False):
    result = evaluate_vehicle_model(
        params, segments, group=group, measured_speed=measured
    )
    out = {"all": {key: result["summary"].get(key) for key in KEEP}}
    out["all"]["segments"] = result["summary"].get("segments")
    for name, values in result.get("by_group", {}).items():
        out[name] = {key: values.get(key) for key in KEEP}
        out[name]["segments"] = values["segments"]
    return out


def band(segment):
    amplitude = float(np.max(np.abs(segment.steer)))
    if amplitude <= 0.51:
        return "|S| <= 0.51"
    return "|S| 0.51-0.75" if amplitude <= 0.75 else "|S| > 0.75"


def collection(directory):
    log = json.loads((directory / "collection.json").read_text(encoding="utf-8"))
    out = {}
    for row in log["segments"]:
        counts = out.setdefault(row["kind"], {"recorded": 0, "kept": 0})
        counts["recorded"] += 1
        counts["kept"] += bool(row["kept"])
    return {
        "server_version": log["server_version"],
        "client_version": log["client_version"],
        "map": log["map"],
        "started_utc": log["started_utc"],
        "finished_utc": log["finished_utc"],
        "kinds": out,
    }


def yaw(params, segments, measured=True):
    result = evaluate_vehicle_model(params, segments, measured_speed=measured)
    return result["summary"]["yaw_rate_rmse_deg_s"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--muscle", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default = carla_nissan_patrol()
    models = {
        "default": default,
        "limited": replace(default, steer_rise_rate=2.5, steer_fall_rate=5.0),
    }
    directories = [args.fresh / f"seed{seed}" for seed in SEEDS]
    fresh = load(directories)
    control = [segment for segment in fresh if kind(segment.name) == "steer"]
    new = [segment for segment in fresh if kind(segment.name) in NEW]
    original = load([args.sysid / f"seed{seed}" for seed in (1, 2, 4)])
    muscle = load(args.muscle)
    bands = {segment.name: band(segment) for segment in new}

    report = {
        "collection": {
            directory.name: collection(directory) for directory in directories
        },
        "segments": {
            "control": len(control),
            "new": len(new),
            "new_by_kind": {
                name: sum(kind(s.name) == name for s in new) for name in NEW
            },
            "original_held_out": len(original),
            "muscle_driven": len(muscle),
        },
    }
    control_rmse = summary(default, control)["all"]["yaw_rate_rmse_deg_s"]
    report["control"] = {
        "default_yaw_rate_rmse_deg_s": control_rmse,
        "limit_deg_s": CONTROL_LIMIT,
        "passed": bool(control_rmse <= CONTROL_LIMIT),
    }
    report["models"] = {
        name: {
            "new_measured_speed": summary(params, new, group=kind, measured=True),
            "new_measured_speed_by_band": summary(
                params, new, group=lambda key: bands[key], measured=True
            ),
            "new_open_loop": summary(params, new, group=kind),
            "control_open_loop": summary(params, control)["all"],
            "original_open_loop": summary(params, original, group=kind),
            "muscle_driven": summary(params, muscle)["all"],
            "muscle_driven_phases": {
                phase: values
                for phase, values in summary(
                    params, muscle, group=lambda key: key
                ).items()
                if phase in ("steer_left", "steer_right", "straighten")
            },
        }
        for name, params in models.items()
    }

    base, candidate = report["models"]["default"], report["models"]["limited"]

    def metric(model, part, group="all", key="yaw_rate_rmse_deg_s"):
        return model[part][group][key]

    before = metric(base, "new_measured_speed")
    after = metric(candidate, "new_measured_speed")
    checks = {
        "control": report["control"]["passed"],
        "a_measured_speed_reduction_at_least_25pct": after <= (1 - REDUCTION) * before,
        # A kind without kept segments fails.
        "b_each_kind_decreases": all(
            name in base["new_measured_speed"]
            and metric(candidate, "new_measured_speed", name)
            < metric(base, "new_measured_speed", name)
            for name in NEW
        ),
        "c_open_loop_yaw_rate_not_higher": metric(candidate, "new_open_loop")
        <= metric(base, "new_open_loop"),
    }
    original_values = {
        "yaw_rate_rmse_deg_s": (
            metric(base, "original_open_loop"),
            metric(candidate, "original_open_loop"),
        )
    }
    for group, values in base["original_open_loop"].items():
        if values["position_error_3s_m"] is not None:
            original_values[f"position_error_3s_m {group}"] = (
                values["position_error_3s_m"],
                candidate["original_open_loop"][group]["position_error_3s_m"],
            )
    checks["d_original_within_1pct"] = all(
        b <= ORIGINAL_TOLERANCE * a for a, b in original_values.values()
    )
    muscle_values = {
        key: (base["muscle_driven"][key], candidate["muscle_driven"][key])
        for key in ("yaw_rate_rmse_deg_s", "position_error_3s_m")
    }
    checks["e_muscle_driven_within_5pct"] = all(
        b <= MUSCLE_TOLERANCE * a for a, b in muscle_values.values()
    )
    report["rule"] = {
        "measured_speed_yaw_rate_rmse_deg_s": (before, after),
        "reduction": 1 - after / before,
        "original": original_values,
        "muscle_driven": muscle_values,
        "checks": checks,
        "adopt": all(checks.values()),
    }

    rises = [float(value) for value in np.arange(0.5, 6.01, 0.25)]
    falls = [0.0] + [float(value) for value in np.arange(1.0, 12.01, 0.5)]
    rise_profile = {
        rise: yaw(replace(default, steer_rise_rate=rise, steer_fall_rate=5.0), new)
        for rise in rises
    }
    fall_profile = {
        fall: yaw(replace(default, steer_rise_rate=2.5, steer_fall_rate=fall), new)
        for fall in falls
    }
    report["profiles"] = {
        "rise_with_fall_5": rise_profile,
        "fall_with_rise_2.5 (0 = unlimited)": fall_profile,
        "best_rise": min(rise_profile, key=rise_profile.get),
        "best_fall": min(fall_profile, key=fall_profile.get),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("segments", "control")}, indent=1))
    for name, values in report["models"].items():
        print(name)
        for part in ("new_measured_speed", "new_measured_speed_by_band"):
            for group, metrics in values[part].items():
                print(f"  {part} {group}: {json.dumps(metrics)}")
        print(f"  new_open_loop all: {json.dumps(values['new_open_loop']['all'])}")
        print(f"  control: {json.dumps(values['control_open_loop'])}")
        print(f"  original all: {json.dumps(values['original_open_loop']['all'])}")
        print(f"  muscle-driven: {json.dumps(values['muscle_driven'])}")
        for phase, metrics in values["muscle_driven_phases"].items():
            print(f"  {phase}: {json.dumps(metrics)}")
    print("rule", json.dumps(report["rule"], indent=1))
    print("best rise", report["profiles"]["best_rise"])
    print("best fall", report["profiles"]["best_fall"])


if __name__ == "__main__":
    main()

"""Control, adoption rule and secondary analyses of design.md.

Errors are duration-weighted means of per-segment RMSE from
``evaluate_vehicle_model``; measured-speed errors pass ``measured_speed=True``.
The default is ``carla_nissan_patrol()``; the candidate is read from the
committed fit report. The engine-rate model of the previous test (rise 2.5,
fall 5.0, default lag) is reported for comparison only.

Usage: python analyse.py --candidate FIT.json --fresh artifacts/steer-rate-lag
--wide artifacts/vehicle-sysid-wide --sysid artifacts/vehicle-sysid --muscle
DIR... --output OUT.json
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

SEEDS = (18, 19, 20, 21)
NEW = ("steer_low", "steer_hold", "steer_doublet")
SWEEPS = ("steer", "steer_fast")
CONTROL_LIMIT = 2 * 0.337
REDUCTION = 0.25
TOLERANCE = 1.05
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fresh", type=Path, required=True)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--muscle", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    default = carla_nissan_patrol()
    fitted = json.loads(args.candidate.read_text(encoding="utf-8"))["candidate"]
    models = {
        "default": default,
        "candidate": VehicleModelParams.from_dict(fitted),
        "engine_rates": replace(default, steer_rise_rate=2.5, steer_fall_rate=5.0),
    }
    directories = [args.fresh / f"seed{seed}" for seed in SEEDS]
    fresh = [s for s in load(directories) if kind(s.name).startswith("steer")]
    new = [segment for segment in fresh if kind(segment.name) in NEW]
    control = [segment for segment in fresh if kind(segment.name) == "steer"]
    inspected = load([args.wide / f"seed{seed}" for seed in (11, 13)])
    original = load([args.sysid / f"seed{seed}" for seed in (1, 2, 4)])
    muscle = load(args.muscle)

    report = {
        "candidate": {
            key: fitted[key]
            for key in ("steer_rise_rate", "steer_fall_rate", "steer_lag")
        },
        "collection": {
            directory.name: collection(directory) for directory in directories
        },
        "segments": {
            "fresh_by_kind": {
                name: sum(kind(s.name) == name for s in fresh) for name in NEW + SWEEPS
            },
            "seeds_11_13": len(inspected),
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
            "fresh_measured_speed": summary(params, fresh, group=kind, measured=True),
            "new_measured_speed": summary(params, new, measured=True)["all"],
            "new_open_loop": summary(params, new)["all"],
            "seeds_11_13_measured_speed": summary(
                params, inspected, group=kind, measured=True
            ),
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

    base, candidate = report["models"]["default"], report["models"]["candidate"]
    key = "yaw_rate_rmse_deg_s"

    def by_kind(model, name):
        groups = model["fresh_measured_speed"]
        return groups[name][key] if name in groups else None

    before, after = (
        base["new_measured_speed"][key],
        candidate["new_measured_speed"][key],
    )
    sweeps = {name: (by_kind(base, name), by_kind(candidate, name)) for name in SWEEPS}
    kinds = {name: (by_kind(base, name), by_kind(candidate, name)) for name in NEW}
    original_values = {
        key: (
            base["original_open_loop"]["all"][key],
            candidate["original_open_loop"]["all"][key],
        )
    }
    for group, values in base["original_open_loop"].items():
        if values["position_error_3s_m"] is not None:
            original_values[f"position_error_3s_m {group}"] = (
                values["position_error_3s_m"],
                candidate["original_open_loop"][group]["position_error_3s_m"],
            )
    muscle_values = {
        name: (base["muscle_driven"][name], candidate["muscle_driven"][name])
        for name in (key, "position_error_3s_m")
    }

    # A kind without kept segments fails its check.
    checks = {
        "control": report["control"]["passed"],
        "a_new_reduction_at_least_25pct": after <= (1 - REDUCTION) * before,
        "b_each_new_kind_decreases": all(
            a is not None and b < a for a, b in kinds.values()
        ),
        "c_sweeps_within_5pct": all(
            a is not None and b <= TOLERANCE * a for a, b in sweeps.values()
        ),
        "d_new_open_loop_not_higher": candidate["new_open_loop"][key]
        <= base["new_open_loop"][key],
        "e_original_within_5pct": all(
            b <= TOLERANCE * a for a, b in original_values.values()
        ),
        "f_muscle_driven_within_5pct": all(
            b <= TOLERANCE * a for a, b in muscle_values.values()
        ),
    }
    report["rule"] = {
        "new_measured_speed": (before, after),
        "reduction": 1 - after / before,
        "new_by_kind": kinds,
        "sweeps": sweeps,
        "new_open_loop": (base["new_open_loop"][key], candidate["new_open_loop"][key]),
        "original": original_values,
        "muscle_driven": muscle_values,
        "checks": checks,
        "adopt": all(checks.values()),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("candidate", "segments", "control")}))
    for name, values in report["models"].items():
        print(name)
        for group, metrics in values["fresh_measured_speed"].items():
            print(f"  fresh measured speed {group}: {metrics[key]:.4f}")
        print(f"  new open loop: {values['new_open_loop'][key]:.4f}")
        for group, metrics in values["seeds_11_13_measured_speed"].items():
            print(f"  seeds 11, 13 {group}: {metrics[key]:.4f}")
        print(f"  original all: {values['original_open_loop']['all'][key]:.4f}")
        print(f"  muscle-driven: {json.dumps(values['muscle_driven'])}")
        for phase, metrics in values["muscle_driven_phases"].items():
            print(f"  {phase}: {metrics[key]:.4f}")
    print("rule", json.dumps(report["rule"], indent=1))


if __name__ == "__main__":
    main()

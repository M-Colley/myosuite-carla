"""Control, extrapolation, steady-state curve and adoption rule of design.md.

Errors are duration-weighted means of per-segment open-loop RMSE from
``evaluate_vehicle_model``. The current default is ``carla_nissan_patrol()``;
the linear and refitted parameters are read from fit reports. The settled
road-wheel angle of a hold is ``atan(L * yaw_rate / v)`` over the second half
of the held command, with the yaw rate one frame after the command.

Usage: python analyse.py --wide artifacts/vehicle-sysid-wide --sysid
artifacts/vehicle-sysid --linear evidence/ue55-vehicle-surrogate.json --refit
REFIT.json --muscle DIR... --output OUT.json
"""

import argparse
import json
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

CONTROL_LIMIT = 2 * 0.337  # twice the original held-out steering 2-10 m/s
NARROW = 0.51  # largest identified steer of seeds 0-4
SLOW = 2.4  # lowest identified steering speed of seeds 0-4
TOLERANCE = 1.05


def load(directories):
    return [
        segment
        for directory in directories
        for segment in segments_from_recording(Recording.load(directory))
    ]


def kind(name):
    return name.rsplit("-", 1)[-1]


def band(segment):
    steering = np.abs(segment.steer) > 0
    amplitude = float(np.max(np.abs(segment.steer)))
    speed = float(np.mean(np.abs(segment.speed[1:][steering])))
    return (
        f"{'wide' if amplitude > NARROW else 'narrow'} |S|, "
        f"{'slow' if speed < SLOW else 'identified'} speed"
    )


def summary(params, segments, group=None):
    result = evaluate_vehicle_model(params, segments, group=group)
    keep = ("yaw_rate_rmse_deg_s", "final_heading_error_deg", "position_error_3s_m")
    out = {"all": {key: result["summary"].get(key) for key in keep}}
    out["all"]["segments"] = result["summary"].get("segments")
    for name, values in result.get("by_group", {}).items():
        out[name] = {key: values.get(key) for key in keep}
        out[name]["segments"] = values["segments"]
    return out


def holds(segments, wheelbase):
    """Settled (steer, speed, road-wheel angle) per hold segment."""
    rows = []
    for segment in segments:
        if kind(segment.name) != "steer_hold":
            continue
        held = np.flatnonzero(segment.steer != 0)
        if len(held) < 10:
            continue
        settled = held[len(held) // 2 :]
        speed = segment.speed[settled + 1]
        rate = segment.yaw_rate[settled + 1]
        if np.min(speed) < 0.3:
            continue
        angle = np.arctan(wheelbase * rate / speed)
        rows.append(
            {
                "segment": segment.name,
                "steer": float(segment.steer[held[0]]),
                "speed_mps": float(np.mean(speed)),
                "angle_rad": float(np.mean(angle)),
                "angle_sd_rad": float(np.std(angle)),
            }
        )
    return rows


def curve(rows):
    steer = np.array([row["steer"] for row in rows])
    angle = np.array([row["angle_rad"] for row in rows])

    def residual(theta):
        return theta[0] * np.sign(steer) * np.abs(steer) ** theta[1] - angle

    fit = optimize.least_squares(residual, x0=[0.9, 1.5], bounds=([0, 0.3], [2, 4]))
    jacobian = fit.jac
    dof = max(len(rows) - 2, 1)
    variance = float(np.sum(fit.fun**2) / dof)
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * variance
    return {
        "holds": len(rows),
        "gain_rad": float(fit.x[0]),
        "exponent": float(fit.x[1]),
        "exponent_se": float(np.sqrt(covariance[1, 1])),
        "rmse_deg": float(np.degrees(np.sqrt(np.mean(fit.fun**2)))),
    }


def predicted(params, rows):
    out = []
    for row in rows:
        model = float(
            params.road_wheel_angle(row["speed_mps"]) * params.steer_input(row["steer"])
        )
        out.append(np.degrees(model - row["angle_rad"]))
    errors = np.asarray(out)
    amplitude = np.abs([row["steer"] for row in rows])
    report = {"rmse_deg": float(np.sqrt(np.mean(errors**2)))}
    for low, high in ((0.0, 0.51), (0.51, 0.75), (0.75, 1.01)):
        chosen = (amplitude > low) & (amplitude <= high)
        if np.any(chosen):
            report[f"|S| {low:.2f}-{min(high, 1):.2f}"] = {
                "holds": int(np.sum(chosen)),
                "mean_error_deg": float(np.mean(errors[chosen])),
                "rmse_deg": float(np.sqrt(np.mean(errors[chosen] ** 2))),
            }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--sysid", type=Path, required=True)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--refit", type=Path)
    parser.add_argument("--muscle", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    models = {
        "default": carla_nissan_patrol(),
        "linear": VehicleModelParams.from_dict(
            json.loads(args.linear.read_text(encoding="utf-8"))["params"]
        ),
    }
    if args.refit:
        models["refit"] = VehicleModelParams.from_dict(
            json.loads(args.refit.read_text(encoding="utf-8"))["params"]
        )
    held = load([args.wide / "seed11", args.wide / "seed13"])
    every = load([args.wide / f"seed{seed}" for seed in (10, 11, 12, 13)])
    control = [segment for segment in held if kind(segment.name) == "steer"]
    new = [segment for segment in held if kind(segment.name) != "steer"]
    original = load([args.sysid / f"seed{seed}" for seed in (1, 2, 4)])
    muscle = load(args.muscle)
    names = {segment.name: band(segment) for segment in new}

    report = {
        "segments": {
            "held_out_control": len(control),
            "held_out_new": len(new),
            "original_held_out": len(original),
            "muscle_driven": len(muscle),
            "bands": {
                name: list(names.values()).count(name) for name in set(names.values())
            },
        }
    }
    control_rmse = summary(models["default"], control)["all"]["yaw_rate_rmse_deg_s"]
    report["control"] = {
        "default_yaw_rate_rmse_deg_s": control_rmse,
        "limit_deg_s": CONTROL_LIMIT,
        "passed": bool(control_rmse <= CONTROL_LIMIT),
    }
    rows = holds(every, models["default"].wheelbase)
    report["steady_state"] = {
        "fit": curve(rows),
        "prediction_error": {
            name: predicted(params, rows) for name, params in models.items()
        },
        "holds": rows,
    }
    report["models"] = {}
    for name, params in models.items():
        report["models"][name] = {
            "new_by_kind": summary(params, new, group=kind),
            "new_by_band": summary(params, new, group=lambda key: names[key]),
            "control": summary(params, control)["all"],
            "original_by_kind": summary(params, original, group=kind),
            "muscle_driven": summary(params, muscle)["all"],
        }

    if "refit" in models:
        base, refit = report["models"]["default"], report["models"]["refit"]
        checks = {
            "control": report["control"]["passed"],
            "a_new_yaw_rate_decreases": refit["new_by_kind"]["all"][
                "yaw_rate_rmse_deg_s"
            ]
            < base["new_by_kind"]["all"]["yaw_rate_rmse_deg_s"],
            "b_original_yaw_rate_within_5pct": refit["original_by_kind"]["all"][
                "yaw_rate_rmse_deg_s"
            ]
            <= TOLERANCE * base["original_by_kind"]["all"]["yaw_rate_rmse_deg_s"],
        }
        position = {}
        for group, values in base["original_by_kind"].items():
            if values["position_error_3s_m"] is None:
                continue
            position[f"original {group}"] = (
                values["position_error_3s_m"],
                refit["original_by_kind"][group]["position_error_3s_m"],
            )
        position["muscle-driven"] = (
            base["muscle_driven"]["position_error_3s_m"],
            refit["muscle_driven"]["position_error_3s_m"],
        )
        checks["c_position_within_5pct"] = all(
            after <= TOLERANCE * before for before, after in position.values()
        )
        report["rule"] = {
            "checks": checks,
            "position_3s_m": position,
            "adopt": all(checks.values()),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("segments", "control")}, indent=1))
    print("steady state", json.dumps(report["steady_state"]["fit"]))
    for name, values in report["steady_state"]["prediction_error"].items():
        print(f"  {name} prediction", json.dumps(values))
    for name, values in report["models"].items():
        print(name)
        for part in ("new_by_kind", "new_by_band"):
            for group, metrics in values[part].items():
                print(f"  {part} {group}: {json.dumps(metrics)}")
        print(f"  control: {json.dumps(values['control'])}")
        print(f"  original all: {json.dumps(values['original_by_kind']['all'])}")
        print(f"  muscle-driven: {json.dumps(values['muscle_driven'])}")
    if "rule" in report:
        print("rule", json.dumps(report["rule"], indent=1))


if __name__ == "__main__":
    main()

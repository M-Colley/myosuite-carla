"""Fit and evaluate the prospective CARLA 1.0 longitudinal experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap

from myosuite_carla import Recording
from myosuite_carla.longitudinal import (
    LongitudinalRolloutFitConfig,
    fit_longitudinal_rollout,
)
from myosuite_carla.vehicle_model import (
    DriveSegment,
    VehicleModelParams,
    segments_from_recording,
    simulate,
)

PROJECT = Path(__file__).resolve().parents[1]
METRICS = ("primary", "speed_rmse_mps", "final_position_error_m")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_batches(
    root: Path,
    seeds: list[int],
    design: dict[str, Any],
    *,
    allow_ineligible: bool = False,
) -> tuple[list[DriveSegment], dict[str, Any], dict[str, Any]]:
    expected = {
        "status": "completed",
        "client_version": design["target"]["carla"],
        "server_version": design["target"]["carla"],
        "map": design["target"]["map"],
        "blueprint": design["target"]["vehicle"],
        "control_dt": design["target"]["dt_s"],
        "script_sha256": design["collector_sha256"],
        "counts": design["counts"],
        "spawn_indices": design["spawn_indices"],
        "spawn_points_used": len(design["spawn_indices"]),
        "min_straight_m": design["min_straight_m"],
        "straight_limit_m": design["straight_limit_m"],
        "max_road_pitch_deg": design["max_road_pitch_deg"],
        "max_lateral_acceleration_mps2": design["max_lateral_acceleration_mps2"],
        "steer_input_exponent": design["collection_steer_exponent"],
        "launch_profile_grid": design["launch_profile_grid"],
    }
    grid = design["launch_profile_grid"]
    profiles = Counter(
        product(grid["target_speeds_mps"], grid["throttles"], grid["brakes"])
    )
    segments, metadata, coverage = [], {}, {}
    for seed in seeds:
        directory = root / f"seed{seed}"
        log = read_json(directory / "collection.json")
        for field, value in expected.items():
            if log.get(field) != value:
                raise ValueError(f"Seed {seed} collection {field} differs from design")
        if log.get("seed") != seed:
            raise ValueError(f"Seed identity differs in {directory}")
        attempts = log["segments"]
        if Counter(row["kind"] for row in attempts) != Counter(design["counts"]):
            raise ValueError(f"Attempt counts differ in seed {seed}")
        actual = Counter(
            (row["target_speed"], row["level"], row["brake"]) for row in attempts
        )
        if actual != profiles:
            raise ValueError(f"Assigned launch profiles differ in seed {seed}")
        names = [row["segment"] for row in attempts]
        if len(set(names)) != len(names) or any(
            not name.startswith(f"s{seed}-") or not name.endswith("-launch")
            for name in names
        ):
            raise ValueError(f"Trial identities differ in seed {seed}")
        accepted = [row for row in attempts if row["kept"]]
        rule = design["coverage"]
        target_counts = Counter(row["target_speed"] for row in accepted)
        target_throttle = Counter(
            (row["target_speed"], row["level"]) for row in accepted
        )
        brake_counts = Counter(row["brake"] for row in accepted)
        eligible = not (
            len(accepted) < rule["minimum_accepted_per_seed"]
            or any(
                target_counts[t] < rule["minimum_per_target"]
                for t in grid["target_speeds_mps"]
            )
            or any(
                target_throttle[t, u] < rule["minimum_per_target_throttle"]
                for t, u in product(grid["target_speeds_mps"], grid["throttles"])
            )
            or any(brake_counts[b] < rule["minimum_per_brake"] for b in grid["brakes"])
        )
        coverage[str(seed)] = {
            "eligible": eligible,
            "attempted": len(attempts),
            "accepted": len(accepted),
            "rejected": len(attempts) - len(accepted),
            "per_target": {str(t): target_counts[t] for t in grid["target_speeds_mps"]},
            "per_target_throttle": {
                f"{t}:{u}": target_throttle[t, u]
                for t, u in product(grid["target_speeds_mps"], grid["throttles"])
            },
            "per_brake": {str(b): brake_counts[b] for b in grid["brakes"]},
        }
        if not eligible and not allow_ineligible:
            raise ValueError(f"Frozen coverage rules failed in seed {seed}")
        if any(
            row.get("status") != "completed"
            or row.get("completed") is not True
            or row.get("targeted_launch") is not True
            or row.get("target_reached") is not True
            or row.get("incomplete_reason") is not None
            or row.get("collision") is not False
            or row.get("frame_limit_exceeded") is not False
            or not np.isfinite(row.get("max_roll_deg", float("inf")))
            or row.get("max_roll_deg", float("inf")) > 3
            for row in accepted
        ):
            raise ValueError(f"Accepted trial quality differs in seed {seed}")
        if not accepted:
            continue
        recording = Recording.load(directory)
        if any(recording.metadata.get(k) != v for k, v in expected.items()) or (
            recording.metadata.get("seed") != seed
            or recording.metadata.get("vehicle") != design["target"]["vehicle"]
        ):
            raise ValueError(f"Recording metadata differ in seed {seed}")
        batch = segments_from_recording(recording)
        if Counter(s.name for s in batch) != Counter(
            row["segment"] for row in accepted
        ):
            raise ValueError(f"Saved trials differ from accepted trials in seed {seed}")
        if any(
            not np.isclose(s.dt, design["target"]["dt_s"], rtol=1e-3, atol=1e-6)
            for s in batch
        ):
            raise ValueError(f"Saved sample period differs in seed {seed}")
        segments.extend(batch)
        metadata.update({row["segment"]: dict(row, seed=seed) for row in accepted})
    return segments, metadata, coverage


def trajectory_weights(
    segments: list[DriveSegment], metadata: dict[str, Any]
) -> list[float]:
    cells = Counter(
        (metadata[s.name]["seed"], metadata[s.name]["target_speed"]) for s in segments
    )
    return [
        1.0
        / (
            len(cells)
            * cells[metadata[s.name]["seed"], metadata[s.name]["target_speed"]]
        )
        for s in segments
    ]


def score_rows(rows: list[dict[str, Any]], targets: list[float]) -> dict[str, Any]:
    """Average trials within targets and seeds without weighting long trajectories."""
    if not rows or any(
        not np.isfinite(row[key]) or row[key] < 0
        for row in rows
        for key in ("speed_mse", "speed_rmse_mps", "final_position_error_m")
    ):
        raise ValueError("Scoring requires finite nonnegative trial errors")
    seeds = {}
    for seed in sorted({row["seed"] for row in rows}):
        groups = {}
        for target in targets:
            group = [
                row
                for row in rows
                if row["seed"] == seed and row["target_speed"] == target
            ]
            if not group:
                raise ValueError("Every seed requires every target speed")
            groups[str(target)] = {
                "primary": float(np.mean([row["speed_mse"] for row in group])),
                "speed_rmse_mps": float(
                    np.mean([row["speed_rmse_mps"] for row in group])
                ),
                "final_position_error_m": float(
                    np.mean([row["final_position_error_m"] for row in group])
                ),
            }
        seeds[str(seed)] = {
            k: float(np.mean([g[k] for g in groups.values()])) for k in METRICS
        } | {"targets": groups}
    return {
        "rows": rows,
        "seeds": seeds,
        "mean": {k: float(np.mean([s[k] for s in seeds.values()])) for k in METRICS},
        "p95_trial_speed_rmse_mps": float(
            np.quantile([row["speed_rmse_mps"] for row in rows], 0.95)
        ),
        "maximum_trial_speed_rmse_mps": float(
            max(row["speed_rmse_mps"] for row in rows)
        ),
    }


def evaluate(
    params: VehicleModelParams,
    segments: list[DriveSegment],
    metadata: dict[str, Any],
    targets: list[float],
    *,
    strict: bool = True,
) -> dict[str, Any]:
    rows = []
    for segment in segments:
        prediction = simulate(params, segment, use_pitch=False)
        error = prediction["speed"] - segment.speed
        meta = metadata[segment.name]
        row = {
            "name": segment.name,
            "seed": meta["seed"],
            "target_speed": meta["target_speed"],
            "throttle": meta["level"],
            "brake": meta["brake"],
            "seconds": segment.duration,
            "speed_mse": float(np.mean(error**2)),
            "speed_rmse_mps": float(np.sqrt(np.mean(error**2))),
            "signed_speed_error_mps": float(error.mean()),
            "signed_acceleration_error_mps2": float(np.diff(error).mean() / segment.dt),
            "peak_observed_speed_mps": float(np.max(np.abs(segment.speed))),
            "final_position_error_m": float(
                np.hypot(
                    prediction["x"][-1] - segment.x[-1],
                    prediction["y"][-1] - segment.y[-1],
                )
            ),
        }
        row["residual_speed_bins"] = {}
        for low, high in zip((0, 8, 15, 17, 20, 22), (8, 15, 17, 20, 22, np.inf)):
            mask = (np.abs(segment.speed[:-1]) >= low) & (
                np.abs(segment.speed[:-1]) < high
            )
            if np.any(mask):
                row["residual_speed_bins"][str(low)] = {
                    "states": int(mask.sum()),
                    "signed_speed_error_mps": float(error[1:][mask].mean()),
                    "signed_acceleration_error_mps2": float(
                        (np.diff(error)[mask] / segment.dt).mean()
                    ),
                }
        drive = np.flatnonzero(segment.throttle > 0)
        braking = np.flatnonzero(segment.brake > 0)
        index = np.arange(len(segment.throttle))
        coast = (
            (index > drive[-1]) & (index < braking[0])
            if len(drive) and len(braking)
            else np.zeros(len(index), dtype=bool)
        )
        for phase, mask in (
            ("launch", segment.throttle > 0),
            ("coast", coast),
            ("brake", segment.brake > 0),
        ):
            row[f"{phase}_speed_rmse_mps"] = (
                float(np.sqrt(np.mean(error[1:][mask] ** 2))) if np.any(mask) else None
            )
        if len(braking):
            start = int(braking[0])
            observed = np.flatnonzero(np.abs(segment.speed[start:]) < 0.02)
            predicted = np.flatnonzero(np.abs(prediction["speed"][start:]) < 0.02)
            row["predicted_stop_reached"] = bool(len(predicted))
            if len(observed) and len(predicted):
                a, b = start + int(observed[0]), start + int(predicted[0])
                row["stopping_time_error_s"] = abs(b - a) * segment.dt
                measured_distance = np.hypot(
                    segment.x[a] - segment.x[start], segment.y[a] - segment.y[start]
                )
                predicted_distance = np.hypot(
                    prediction["x"][b] - prediction["x"][start],
                    prediction["y"][b] - prediction["y"][start],
                )
                row["stopping_distance_error_m"] = float(
                    abs(predicted_distance - measured_distance)
                )
            else:
                row["stopping_time_error_s"] = row["stopping_distance_error_m"] = None
        rows.append(row)
    try:
        return score_rows(rows, targets)
    except ValueError as error:
        if strict:
            raise
        return {"rows": rows, "scoring_error": str(error)}


def compare_test(
    old: dict[str, Any], new: dict[str, Any], design: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen paired seed interval and all tail and target guards."""
    seeds = list(map(str, design["test_seeds"]))
    if set(old["seeds"]) != set(seeds) or set(new["seeds"]) != set(seeds):
        raise ValueError("The comparison requires every fixed test seed")
    differences = np.array(
        [new["seeds"][s]["primary"] - old["seeds"][s]["primary"] for s in seeds]
    )
    acceptance = design["acceptance"]
    interval = bootstrap(
        (differences,),
        np.mean,
        method="percentile",
        confidence_level=acceptance["confidence"],
        n_resamples=acceptance["resamples"],
        random_state=np.random.default_rng(acceptance["bootstrap_seed"]),
    ).confidence_interval
    guards = {
        k: new["mean"][k] <= old["mean"][k]
        for k in ("speed_rmse_mps", "final_position_error_m")
    }
    guards.update(
        {
            k: new[k] <= old[k]
            for k in ("p95_trial_speed_rmse_mps", "maximum_trial_speed_rmse_mps")
        }
    )
    for target in design["launch_profile_grid"]["target_speeds_mps"]:
        a = np.mean(
            [s["targets"][str(target)]["primary"] for s in old["seeds"].values()]
        )
        b = np.mean(
            [s["targets"][str(target)]["primary"] for s in new["seeds"].values()]
        )
        guards[f"target_{target}"] = bool(b <= 1.1 * a + 0.01)
    return {
        "paired_seed_primary_change": {
            "mean": float(differences.mean()),
            "differences": differences.tolist(),
            "interval": [float(interval.low), float(interval.high)],
            "resampling_unit": "seed batch",
        },
        "guards": guards,
        "decision": (
            "accepted_targeted_longitudinal"
            if interval.high < 0 and all(guards.values())
            else "experimental_longitudinal"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="Fit complete accepted trials for diagnostics; never select a model",
    )
    args = parser.parse_args()
    if args.diagnostic and args.stage != "fit":
        parser.error("Diagnostic fitting cannot be used for final testing")
    design = read_json(args.design)
    baseline_path = PROJECT / design["baseline"]["selection_file"]
    if digest(baseline_path) != design["baseline"]["selection_sha256"]:
        raise ValueError("Frozen baseline selection differs")
    baseline = VehicleModelParams.from_dict(read_json(baseline_path)["parameters"])
    if digest(PROJECT / design["collector_file"]) != design["collector_sha256"]:
        raise ValueError("Collector source differs from the frozen protocol")
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.stage}-report.json"
    if destination.exists() or (args.stage == "fit" and any(args.out.iterdir())):
        raise FileExistsError("Existing analysis evidence must be preserved")
    sources = {
        name: hashlib.sha256(
            (PROJECT / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in (
            "src/myosuite_carla/longitudinal.py",
            "src/myosuite_carla/vehicle_model.py",
            "examples/calibrate_longitudinal.py",
        )
    }
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": digest(args.design),
        "source_sha256": sources,
        "source_hash_normalization": "LF line endings",
        "use_pitch": False,
        "measured_speed": False,
    }
    targets = design["launch_profile_grid"]["target_speeds_mps"]
    if args.stage == "fit":
        if any((args.test_data / f"seed{s}").exists() for s in design["test_seeds"]):
            raise ValueError("Final test seeds must remain uncollected during fitting")
        training, train_meta, report["training_coverage"] = load_batches(
            args.data, design["train_seeds"], design, allow_ineligible=True
        )
        validation, val_meta, report["validation_coverage"] = load_batches(
            args.data, design["validation_seeds"], design, allow_ineligible=True
        )
        eligible = all(
            row["eligible"]
            for split in ("training_coverage", "validation_coverage")
            for row in report[split].values()
        )
        report["protocol_eligible"] = eligible
        report["diagnostic_only"] = args.diagnostic
        settings = design["fit"]
        config = LongitudinalRolloutFitConfig(
            **{
                k: tuple(settings[k]) if isinstance(settings[k], list) else settings[k]
                for k in LongitudinalRolloutFitConfig.__dataclass_fields__
            }
        )
        if training and (eligible or args.diagnostic):
            params, report["fit"] = fit_longitudinal_rollout(
                training,
                baseline,
                segment_weights=trajectory_weights(training, train_meta),
                config=config,
            )
            report["baseline"] = evaluate(
                baseline, validation, val_meta, targets, strict=eligible
            )
            report["candidate"] = evaluate(
                params, validation, val_meta, targets, strict=eligible
            )
        else:
            params = baseline
            report["fit"] = {
                "success": False,
                "skipped": "Insufficient eligible training coverage",
            }
        report["parameters"] = params.as_dict()
        report["selected"] = bool(
            eligible
            and not args.diagnostic
            and report["fit"]["success"]
            and report["candidate"]["mean"]["primary"]
            < report["baseline"]["mean"]["primary"]
            and report["candidate"]["p95_trial_speed_rmse_mps"]
            <= report["baseline"]["p95_trial_speed_rmse_mps"]
        )
        report["status"] = (
            "diagnostic_only"
            if args.diagnostic
            else "completed"
            if eligible
            else "coverage_failed_before_fitting"
        )
    else:
        fitted_path = args.out / "fit-report.json"
        fitted = read_json(fitted_path)
        frozen = read_json(args.out / "frozen-selection.json")
        if (
            not fitted["selected"]
            or fitted["design_sha256"] != report["design_sha256"]
            or fitted["source_sha256"] != sources
            or frozen["fit_report_sha256"] != digest(fitted_path)
            or frozen["parameters"] != fitted["parameters"]
        ):
            raise ValueError("A frozen improving validation fit is required")
        params = VehicleModelParams.from_dict(fitted["parameters"])
        segments, metadata, report["test_coverage"] = load_batches(
            args.test_data, design["test_seeds"], design
        )
        old = report["baseline"] = evaluate(baseline, segments, metadata, targets)
        new = report["candidate"] = evaluate(params, segments, metadata, targets)
        report.update(compare_test(old, new, design))
        report["parameters"] = params.as_dict()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save(destination, report)
    if args.stage == "fit":
        save(
            args.out / "frozen-selection.json",
            {
                "design_sha256": report["design_sha256"],
                "fit_report_sha256": digest(destination),
                "selected": report["selected"],
                "parameters": report["parameters"],
                "test_collected": False,
            },
        )
    print(f"Saved {destination}", flush=True)
    return 2 if report.get("status") == "coverage_failed_before_fitting" else 0


if __name__ == "__main__":
    raise SystemExit(main())

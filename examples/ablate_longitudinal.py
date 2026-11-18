"""Compare fixed derivative and rollout objectives on prospective moving probes."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap

from myosuite_carla import Recording
from myosuite_carla.driver_models import AccelerationControlConfig
from myosuite_carla.longitudinal import (
    LongitudinalRolloutFitConfig,
    fit_longitudinal_rollout,
)
from myosuite_carla.vehicle_model import (
    VehicleModelParams,
    segments_from_recording,
    simulate,
)

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "historical_longitudinal_protocol",
    Path(__file__).with_name("calibrate_longitudinal.py"),
)
assert SPEC is not None and SPEC.loader is not None
HISTORICAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HISTORICAL)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict[str, str]:
    return {
        name: hashlib.sha256(
            (PROJECT / name).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        for name in (
            "src/myosuite_carla/vehicle_model.py",
            "src/myosuite_carla/longitudinal.py",
            "examples/calibrate_longitudinal.py",
            "examples/ablate_longitudinal.py",
        )
    }


def load_probes(root: Path, design: dict[str, Any]):
    grid = design["probe_profile_grid"]
    expected_profiles = Counter(
        product(grid["speeds_mps"], grid["histories"], grid["actions"])
    )
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
        "probe_profile_grid": grid,
        "launch_profile_grid": None,
    }
    segments, metadata, coverage = [], {}, {}
    for seed in design["test_seeds"]:
        directory = root / f"seed{seed}"
        log = HISTORICAL.read_json(directory / "collection.json")
        if log.get("seed") != seed or any(log.get(k) != v for k, v in expected.items()):
            raise ValueError(
                f"Probe collection identity or configuration differs in seed {seed}"
            )
        rows = log["segments"]
        profiles = Counter(
            (
                r["probe_profile"]["speed_mps"],
                r["probe_profile"]["history"],
                r["probe_profile"]["action"],
            )
            for r in rows
        )
        if profiles != expected_profiles or Counter(r["kind"] for r in rows) != Counter(
            design["counts"]
        ):
            raise ValueError(f"Moving probe condition coverage differs in seed {seed}")
        names = [r["segment"] for r in rows]
        if len(set(names)) != len(names) or any(
            not n.startswith(f"s{seed}-") for n in names
        ):
            raise ValueError(f"Moving probe identities differ in seed {seed}")
        if any(
            r.get("kept") is not True
            or r.get("status") != "completed"
            or r.get("completed") is not True
            or r.get("target_reached") is not True
            or r.get("collision") is not False
            or r.get("frame_limit_exceeded") is not False
            or r.get("incomplete_reason") is not None
            or not np.isfinite(r.get("max_roll_deg", np.inf))
            or r.get("max_roll_deg", np.inf) > 3
            for r in rows
        ):
            raise ValueError(
                f"Every fixed moving probe must be eligible in seed {seed}"
            )
        recording = Recording.load(directory)
        if (
            any(recording.metadata.get(k) != v for k, v in expected.items())
            or recording.metadata.get("seed") != seed
            or recording.metadata.get("vehicle") != design["target"]["vehicle"]
        ):
            raise ValueError(f"Probe recording metadata differ in seed {seed}")
        batch = segments_from_recording(recording)
        if Counter(s.name for s in batch) != Counter(names):
            raise ValueError(f"Saved moving probes differ in seed {seed}")
        by_name = {row["segment"]: row for row in rows}
        for segment in batch:
            row = by_name[segment.name]
            origin = row.get("probe_origin_row")
            horizon = int(round(design["window"]["seconds"] / segment.dt))
            if (
                not np.isclose(
                    segment.dt, design["target"]["dt_s"], rtol=1e-3, atol=1e-6
                )
                or not isinstance(origin, int)
                or origin < 1
                or origin + horizon >= len(segment.speed)
            ):
                raise ValueError(f"Missing complete moving window in {segment.name}")
            target = row["probe_profile"]["speed_mps"]
            tolerance = grid["settings"]["entry_tolerance_mps"]
            if (
                abs(segment.speed[origin] - target) > tolerance
                or abs(row["probe_entry_speed_mps"] - target) > tolerance
            ):
                raise ValueError(f"Moving origin speed differs in {segment.name}")
            metadata[segment.name] = dict(row, seed=seed)
        segments.extend(batch)
        coverage[str(seed)] = {
            "attempted": len(rows),
            "accepted": len(rows),
            "conditions": len(profiles),
        }
    return segments, metadata, coverage


def moving_window(segment, origin: int, seconds: float):
    end = origin + int(round(seconds / segment.dt))
    return replace(
        segment,
        **{
            key: getattr(segment, key)[origin:end]
            for key in ("throttle", "brake", "steer", "hand_brake", "reverse")
        },
        **{
            key: getattr(segment, key)[origin : end + 1]
            for key in ("x", "y", "yaw_deg", "pitch_deg", "speed", "yaw_rate")
        },
    )


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows or any(
        not np.isfinite(row[k]) or row[k] < 0
        for row in rows
        for k in ("speed_mse", "speed_rmse_mps", "position_error_3s_m")
    ):
        raise ValueError("Moving probe scores must be finite and nonnegative")
    seeds = {
        str(seed): {
            "primary": float(
                np.mean([r["speed_mse"] for r in rows if r["seed"] == seed])
            ),
            "position_error_3s_m": float(
                np.mean([r["position_error_3s_m"] for r in rows if r["seed"] == seed])
            ),
            "history_effect_acceleration_error_mps2": float(
                np.mean(
                    [
                        r["acceleration_error_1s_mps2"]
                        for r in rows
                        if r["seed"] == seed and r["history"] == "braking"
                    ]
                )
                - np.mean(
                    [
                        r["acceleration_error_1s_mps2"]
                        for r in rows
                        if r["seed"] == seed and r["history"] == "acceleration"
                    ]
                )
            ),
        }
        for seed in sorted({row["seed"] for row in rows})
    }
    return {
        "rows": rows,
        "seeds": seeds,
        "mean": {
            k: float(np.mean([r[k] for r in seeds.values()]))
            for k in next(iter(seeds.values()))
        },
        "p95_trial_speed_rmse_mps": float(
            np.quantile([r["speed_rmse_mps"] for r in rows], 0.95)
        ),
        "maximum_trial_speed_rmse_mps": float(max(r["speed_rmse_mps"] for r in rows)),
    }


def evaluate(params, segments, metadata, design):
    rows = []
    for full in segments:
        meta = metadata[full.name]
        origin = meta["probe_origin_row"]
        segment = moving_window(full, origin, design["window"]["seconds"])
        prediction = simulate(
            params, segment, use_pitch=False, throttle_history=full.throttle[:origin]
        )
        error = prediction["speed"] - segment.speed
        one = int(round(1 / segment.dt))
        rows.append(
            {
                "name": full.name,
                "seed": meta["seed"],
                "speed_mps": meta["probe_profile"]["speed_mps"],
                "history": meta["probe_profile"]["history"],
                "action": meta["probe_profile"]["action"],
                "speed_mse": float(np.mean(error**2)),
                "speed_rmse_mps": float(np.sqrt(np.mean(error**2))),
                "speed_rmse_1s_mps": float(np.sqrt(np.mean(error[: one + 1] ** 2))),
                "position_error_1s_m": float(
                    np.hypot(
                        prediction["x"][one] - segment.x[one],
                        prediction["y"][one] - segment.y[one],
                    )
                ),
                "position_error_3s_m": float(
                    np.hypot(
                        prediction["x"][-1] - segment.x[-1],
                        prediction["y"][-1] - segment.y[-1],
                    )
                ),
                "acceleration_error_1s_mps2": float(
                    (error[one] - error[0]) / (one * segment.dt)
                ),
            }
        )
    return score_rows(rows)


def compare(derivative, rollout, design):
    seeds = list(map(str, design["test_seeds"]))
    if set(derivative["seeds"]) != set(seeds) or set(rollout["seeds"]) != set(seeds):
        raise ValueError("Every fixed seed is required for the objective comparison")
    if Counter(r["name"] for r in derivative["rows"]) != Counter(
        r["name"] for r in rollout["rows"]
    ):
        raise ValueError("Both objectives require identical moving probes")
    differences = np.array(
        [
            rollout["seeds"][s]["primary"] - derivative["seeds"][s]["primary"]
            for s in seeds
        ]
    )
    rule = design["acceptance"]
    interval = bootstrap(
        (differences,),
        np.mean,
        method="percentile",
        confidence_level=rule["confidence"],
        n_resamples=rule["resamples"],
        random_state=np.random.default_rng(rule["bootstrap_seed"]),
    ).confidence_interval
    guards = {
        "mean_3s_position_error": rollout["mean"]["position_error_3s_m"]
        <= derivative["mean"]["position_error_3s_m"]
    }
    guards.update(
        {
            k: rollout[k] <= derivative[k]
            for k in ("p95_trial_speed_rmse_mps", "maximum_trial_speed_rmse_mps")
        }
    )
    for field, values in (
        ("speed_mps", design["probe_profile_grid"]["speeds_mps"]),
        ("history", design["probe_profile_grid"]["histories"]),
        ("action", design["probe_profile_grid"]["actions"]),
    ):
        for value in values:
            old = np.mean(
                [r["speed_mse"] for r in derivative["rows"] if r[field] == value]
            )
            new = np.mean(
                [r["speed_mse"] for r in rollout["rows"] if r[field] == value]
            )
            guards[f"{field}:{value}"] = bool(new <= 1.1 * old + 0.01)
    return {
        "paired_seed_primary_change": {
            "mean": float(differences.mean()),
            "differences": differences.tolist(),
            "interval": [float(interval.low), float(interval.high)],
            "resampling_unit": "seed batch",
        },
        "guards": guards,
        "decision": "rollout_objective_supported"
        if interval.high < 0 and all(guards.values())
        else "rollout_objective_advantage_not_supported",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Historical development recordings only",
    )
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    args = parser.parse_args()
    design = HISTORICAL.read_json(args.design)
    historical_path = PROJECT / design["historical_development_protocol"]
    if (
        sha(historical_path) != design["historical_development_protocol_sha256"]
        or sha(PROJECT / design["collector_file"]) != design["collector_sha256"]
    ):
        raise ValueError("Frozen historical protocol or prospective collector differs")
    historical = HISTORICAL.read_json(historical_path)
    baseline_path = PROJECT / design["baseline"]["selection_file"]
    if sha(baseline_path) != design["baseline"]["selection_sha256"]:
        raise ValueError("The physical C3 baseline differs")
    baseline = VehicleModelParams.from_dict(
        HISTORICAL.read_json(baseline_path)["parameters"]
    )
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.stage}-report.json"
    if destination.exists() or (args.stage == "fit" and any(args.out.iterdir())):
        raise FileExistsError("Existing objective ablation evidence must be preserved")
    report = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": sha(args.design),
        "source_sha256": sources(),
        "source_hash_normalization": "LF line endings",
    }
    if args.stage == "fit":
        if any((args.test_data / f"seed{s}").exists() for s in design["test_seeds"]):
            raise ValueError(
                "Fresh moving probes must remain uncollected before model freezing"
            )
        training, train_meta, report["training_coverage"] = HISTORICAL.load_batches(
            args.data, design["train_seeds"], historical
        )
        validation, val_meta, report["historical_validation_coverage"] = (
            HISTORICAL.load_batches(
                args.data, design["historical_validation_seeds"], historical
            )
        )
        config = LongitudinalRolloutFitConfig(
            **{
                k: tuple(design["fit"][k])
                if isinstance(design["fit"][k], list)
                else design["fit"][k]
                for k in LongitudinalRolloutFitConfig.__dataclass_fields__
            }
        )
        report["models"] = {"C3": {"parameters": baseline.as_dict()}}
        for objective in ("derivative", "rollout"):
            print(f"Fitting fixed {objective} objective", flush=True)
            params, fitted = fit_longitudinal_rollout(
                training,
                baseline,
                segment_weights=HISTORICAL.trajectory_weights(training, train_meta),
                config=config,
                objective=objective,
            )
            AccelerationControlConfig(vehicle=params)
            if any(
                params.as_dict()[k] != value
                for k, value in baseline.as_dict().items()
                if k not in ("drive", "quadratic_drag", "brake_decel", "throttle_lag")
            ):
                raise ValueError("The fixed C3 parameter ownership changed")
            report["models"][objective] = {
                "parameters": params.as_dict(),
                "fit": fitted,
                "historical_validation": HISTORICAL.evaluate(
                    params,
                    validation,
                    val_meta,
                    historical["launch_profile_grid"]["target_speeds_mps"],
                ),
            }
        report["eligible"] = all(
            report["models"][name]["fit"]["success"]
            for name in ("derivative", "rollout")
        )
        report["historical_validation_role"] = design["historical_validation_role"]
    else:
        fitted_path = args.out / "fit-report.json"
        fitted = HISTORICAL.read_json(fitted_path)
        frozen = HISTORICAL.read_json(args.out / "frozen-models.json")
        if (
            not fitted["eligible"]
            or fitted["design_sha256"] != report["design_sha256"]
            or fitted["source_sha256"] != report["source_sha256"]
            or frozen["fit_report_sha256"] != sha(fitted_path)
            or frozen["models"]
            != {k: v["parameters"] for k, v in fitted["models"].items()}
        ):
            raise ValueError("Both unmodified objective fits must be frozen")
        segments, metadata, report["test_coverage"] = load_probes(
            args.test_data, design
        )
        report["models"] = {
            name: evaluate(
                VehicleModelParams.from_dict(value["parameters"]),
                segments,
                metadata,
                design,
            )
            for name, value in fitted["models"].items()
        }
        report.update(
            compare(report["models"]["derivative"], report["models"]["rollout"], design)
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    HISTORICAL.save(destination, report)
    if args.stage == "fit":
        HISTORICAL.save(
            args.out / "frozen-models.json",
            {
                "design_sha256": report["design_sha256"],
                "fit_report_sha256": sha(destination),
                "eligible": report["eligible"],
                "models": {k: v["parameters"] for k, v in report["models"].items()},
                "fresh_test_collected": False,
            },
        )
    print(f"Saved {destination}", flush=True)
    return 2 if report.get("eligible") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())

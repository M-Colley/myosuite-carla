"""Select a vehicle fit on validation batches, then evaluate sealed test batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import bootstrap

from myosuite_carla import Recording
from myosuite_carla.driver_models import AccelerationControlConfig
from myosuite_carla.vehicle_model import (
    VehicleModelParams,
    carla_nissan_patrol,
    evaluate_vehicle_model,
    fit_vehicle_model,
    segments_from_recording,
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def kind(name: str) -> str:
    return name.rsplit("-", 1)[-1]


def source_score(rows: list[dict[str, Any]], families: list[str]) -> dict[str, Any]:
    """Give each manoeuvre family equal weight within one random seed batch."""
    grouped: dict[str, Any] = {}
    for family in families:
        selected = [row for row in rows if kind(row["name"]) == family]
        if not selected:
            raise ValueError(f"Missing manoeuvre family: {family}")
        speed = np.array([r["speed_rmse_mps"] for r in selected])
        yaw = np.array([r["yaw_rate_rmse_deg_s"] for r in selected])
        grouped[family] = {
            "primary": float(np.mean(0.5 * (speed**2 + (yaw / 5.0) ** 2))),
            "speed_rmse_mps": float(speed.mean()),
            "yaw_rate_rmse_deg_s": float(yaw.mean()),
            "final_position_error_m": float(
                np.mean([r["final_position_error_m"] for r in selected])
            ),
            "segments": len(selected),
        }
    return {
        key: float(np.mean([row[key] for row in grouped.values()]))
        for key in (
            "primary",
            "speed_rmse_mps",
            "yaw_rate_rmse_deg_s",
            "final_position_error_m",
        )
    } | {"families": grouped}


def load_batches(
    data: Path, seeds: list[int], design: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    batches, coverage = {}, {}
    for seed in seeds:
        directory = data / f"seed{seed}"
        log = read_json(directory / "collection.json")
        if log["client_version"] != "1.0" or log["server_version"] != "1.0":
            raise ValueError("The calibration requires a matched CARLA 1.0 runtime")
        attempted = Counter(row["kind"] for row in log["segments"])
        kept = Counter(row["kind"] for row in log["segments"] if row["kept"])
        if dict(attempted) != design["counts"]:
            raise ValueError("Attempt counts differ from the frozen design")
        for family, count in design["counts"].items():
            if kept[family] < max(1, int(np.ceil(count / 2))):
                raise ValueError(
                    f"Insufficient accepted {family} trials in seed {seed}"
                )
        segments = segments_from_recording(Recording.load(directory))
        if len(segments) != sum(kept.values()):
            raise ValueError("Saved segments differ from the accepted collection log")
        batches[str(seed)] = segments
        coverage[str(seed)] = {
            "attempted": dict(attempted),
            "accepted": dict(kept),
            "rejected": sum(attempted.values()) - sum(kept.values()),
        }
    return batches, coverage


def evaluate(
    params: VehicleModelParams, batches: dict[str, Any], families: list[str]
) -> dict[str, Any]:
    reports = {
        seed: evaluate_vehicle_model(params, segments, group=kind, use_pitch=False)
        for seed, segments in batches.items()
    }
    scores = {
        seed: source_score(report["segments"], families)
        for seed, report in reports.items()
    }
    return {"reports": reports, "scores": scores}


def mean_score(result: dict[str, Any], metric: str = "primary") -> float:
    return float(np.mean([row[metric] for row in result["scores"].values()]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("fit", "test"), required=True)
    args = parser.parse_args()
    design = read_json(args.design)
    test_data = args.test_data if args.test_data is not None else args.data
    digest = hashlib.sha256(args.design.read_bytes()).hexdigest()
    families = list(design["counts"])
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f"{args.stage}-report.json"
    if destination.exists():
        raise ValueError("An existing stage report must be preserved")
    if args.stage == "fit" and any(args.out.iterdir()):
        raise ValueError("Fitting requires an empty result directory")
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "design_sha256": digest,
        "use_pitch": False,
        "measured_speed": False,
    }
    if args.stage == "fit":
        if any((test_data / f"seed{s}").exists() for s in design["test_seeds"]):
            raise ValueError("Test batches must remain uncollected during selection")
        training, report["training_coverage"] = load_batches(
            args.data, design["train_seeds"], design
        )
        validation, report["validation_coverage"] = load_batches(
            args.data, design["validation_seeds"], design
        )
        segments = [segment for batch in training.values() for segment in batch]
        report["legacy"] = evaluate(carla_nissan_patrol(), validation, families)
        report["candidates"] = {}
        for candidate in design["candidates"]:
            name = candidate["name"]
            print(f"Fitting {name}", flush=True)
            params, fit = fit_vehicle_model(
                segments,
                wheelbase=design["wheelbase_m"],
                smoothing=design["smoothing"],
                use_pitch=False,
                monotonic_drive=True,
                **candidate["parameters"],
            )
            AccelerationControlConfig(vehicle=params)
            result = {
                "parameters": params.as_dict(),
                "fit": fit,
                "validation": evaluate(params, validation, families),
            }
            report["candidates"][name] = result
            write_json(args.out / f"candidate-{name}.json", result)
            print(f"{name}: validation primary {mean_score(result['validation'])}")
        selected = min(
            report["candidates"],
            key=lambda name: mean_score(report["candidates"][name]["validation"]),
        )
        better = mean_score(report["candidates"][selected]["validation"]) < mean_score(
            report["legacy"]
        )
        report["selected"] = selected if better else None
        report["selection_data"] = "training and validation only"
    else:
        fitted = read_json(args.out / "fit-report.json")
        if fitted["design_sha256"] != digest or fitted["selected"] is None:
            raise ValueError("A frozen improving validation selection is required")
        selected = fitted["selected"]
        params = VehicleModelParams.from_dict(
            fitted["candidates"][selected]["parameters"]
        )
        testing, report["test_coverage"] = load_batches(
            test_data, design["test_seeds"], design
        )
        report["selected"] = selected
        report["legacy"] = evaluate(carla_nissan_patrol(), testing, families)
        report["candidate"] = evaluate(params, testing, families)
        differences = np.array(
            [
                report["candidate"]["scores"][seed]["primary"]
                - report["legacy"]["scores"][seed]["primary"]
                for seed in testing
            ]
        )
        interval = bootstrap(
            (differences,),
            np.mean,
            method="percentile",
            confidence_level=0.95,
            n_resamples=5000,
            random_state=np.random.default_rng(design["bootstrap_seed"]),
        ).confidence_interval
        report["paired_seed_primary_change"] = {
            "value": float(differences.mean()),
            "interval": [float(interval.low), float(interval.high)],
            "differences": differences.tolist(),
            "resampling_unit": "seed batch",
        }
        guard = {
            metric: mean_score(report["candidate"], metric)
            <= mean_score(report["legacy"], metric)
            for metric in (
                "speed_rmse_mps",
                "yaw_rate_rmse_deg_s",
                "final_position_error_m",
            )
        }
        for family in families:
            new = np.mean(
                [
                    r["families"][family]["primary"]
                    for r in report["candidate"]["scores"].values()
                ]
            )
            old = np.mean(
                [
                    r["families"][family]["primary"]
                    for r in report["legacy"]["scores"].values()
                ]
            )
            guard[family] = bool(new <= 1.25 * old + 0.01)
        report["guards"] = guard
        report["decision"] = (
            "accepted_contextual_calibration"
            if interval.high < 0 and all(guard.values())
            else "experimental_calibration"
        )
        report["parameters"] = params.as_dict()
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(destination, report)
    print(f"Saved {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

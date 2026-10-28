"""Measure desired and realized acceleration through muscle driven pedals."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def reference_grid_rmse(record: dict[str, Any]) -> float:
    """Score a common time grid, including early stops and feasible brake holds."""
    requested = record["desired_acceleration_mps2"]
    initial = record["initial_speed_mps"]
    samples = [row for row in record["samples"] if row["time_s"] >= 2]
    if not samples:
        raise ValueError("the reference grid needs samples after two seconds")
    reference = [
        requested if requested >= 0 or initial + requested * row["time_s"] > 0 else 0
        for row in samples
    ]
    errors = (
        np.asarray([row["actual_acceleration_mps2"] for row in samples]) - reference
    )
    return float(np.sqrt(np.mean(errors**2)))


def evaluate(seconds: float = 6.0) -> dict[str, Any]:
    from myosuite_carla.driver_models import (
        AccelerationControlConfig,
        ReflexDriver,
        ReflexDriverConfig,
        idm_acceleration,
    )
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.fullbody import FullBodyDriverConfig, build_fullbody_driver
    from myosuite_carla.scenario import Scenario

    driver = build_fullbody_driver(FullBodyDriverConfig.fast())
    profiles = {
        "legacy": ReflexDriverConfig(perception_delay=0.02),
        "inverse": ReflexDriverConfig(
            perception_delay=0.02, acceleration_control=AccelerationControlConfig()
        ),
    }
    episodes = []
    for label, config in profiles.items():
        for speed in (5.0, 10.0, 15.0):
            for desired in (-1.0, 0.0, 1.0):
                env = FullBodyDrivingEnv(
                    Scenario(
                        name="acceleration_tracking",
                        start_speed=speed,
                        target_speed=speed,
                        duration=seconds,
                    ),
                    driver=driver,
                )
                record: dict[str, Any] = {
                    "profile": label,
                    "initial_speed_mps": speed,
                    "desired_acceleration_mps2": desired,
                    "status": "failed",
                    "samples": [],
                }
                try:
                    env.reset(seed=931)
                    controller = ReflexDriver(env.dt, config)
                    for _ in range(math.ceil(seconds / env.dt)):
                        state = env.state
                        component = idm_acceleration(
                            state["speed_mps"],
                            state["target_speed_mps"],
                            None,
                            0,
                            config.idm,
                        )
                        action = controller.act(
                            state, acceleration_offset=desired - component
                        )
                        if abs(controller.demand[0] - desired) > 1e-10:
                            raise RuntimeError(
                                "synthetic demand differs from the request"
                            )
                        _, _, terminated, truncated, info = env.step(action)
                        if not all(
                            np.isfinite(value).all()
                            for value in (
                                driver.data.qpos,
                                driver.data.qvel,
                                driver.data.act,
                            )
                        ):
                            raise RuntimeError("nonfinite muscle state")
                        record["samples"].append(
                            {
                                "time_s": info["scenario"]["time"],
                                "speed_mps": env.state["speed_mps"],
                                "desired_acceleration_mps2": desired,
                                "actual_acceleration_mps2": env.state["a_forward_mps2"],
                                "measured_throttle": info["inputs"]["throttle"],
                                "measured_brake": info["inputs"]["brake"],
                            }
                        )
                        if terminated:
                            raise RuntimeError("synthetic tracking episode terminated")
                        if truncated:
                            record["status"] = "completed"
                            break
                    measured = [
                        row["actual_acceleration_mps2"]
                        for row in record["samples"]
                        if row["time_s"] >= 2 and row["speed_mps"] > 0.5
                    ]
                    if not measured:
                        raise RuntimeError("no feasible steady tracking samples")
                    errors = np.asarray(measured) - desired
                    record["tracking_rmse_mps2"] = float(np.sqrt(np.mean(errors**2)))
                    record["tracking_bias_mps2"] = float(np.mean(errors))
                    record["reference_grid_tracking_rmse_mps2"] = reference_grid_rmse(
                        record
                    )
                except Exception as error:
                    record["status"] = "failed"
                    record["error"] = f"{type(error).__name__}: {error}"
                finally:
                    env.close()
                    env.vehicle.close()
                    record["surrogate_closed"] = not env.vehicle.active
                episodes.append(record)
                print(
                    label,
                    speed,
                    desired,
                    record["status"],
                    record.get("tracking_rmse_mps2"),
                    flush=True,
                )
    return {
        "status": "completed"
        if all(row["status"] == "completed" for row in episodes)
        else "failed",
        "role": "engineering_tracking_consistency",
        "scoring": "after two seconds; positive speed above 0.5 m/s",
        "reference_grid_scoring": (
            "common times after two seconds; "
            "zero feasible acceleration after the reference vehicle stops"
        ),
        "configuration": {label: asdict(config) for label, config in profiles.items()},
        "episodes": episodes,
        "mean_tracking_rmse_mps2": {
            label: float(
                np.mean(
                    [
                        row["tracking_rmse_mps2"]
                        for row in episodes
                        if row["profile"] == label and "tracking_rmse_mps2" in row
                    ]
                )
            )
            for label in profiles
        },
        "mean_reference_grid_tracking_rmse_mps2": {
            label: float(
                np.mean(
                    [
                        row["reference_grid_tracking_rmse_mps2"]
                        for row in episodes
                        if row["profile"] == label
                        and "reference_grid_tracking_rmse_mps2" in row
                    ]
                )
            )
            for label in profiles
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

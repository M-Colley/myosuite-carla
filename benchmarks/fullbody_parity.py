"""Parity and speed of the full-body driver with and without pruned muscles.

Runs an identical scripted motor program through both models and compares
every free joint, mechanism sensor and mapped vehicle control at each 20 ms
control step. Prints JSON; ``--output`` also writes it.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from myosuite_carla import DriverIntent, FullBodyDriverConfig, build_fullbody_driver
from myosuite_carla.fullbody import FREE_JOINT_PREFIXES

PROGRAM = (
    (DriverIntent(left_foot=0.15, park_finger=0.06), 50),
    (DriverIntent(right_foot=0.012), 75),
    (DriverIntent(right_foot=0.008, steer=-0.14), 75),
    (DriverIntent(right_foot=0.008, steer=0.14), 75),
    (DriverIntent(left_foot=0.15, reverse_finger=0.05), 50),
    (DriverIntent(gear_finger=0.06), 15),
    (DriverIntent(), 60),
)


def _trace(config: FullBodyDriverConfig) -> dict:
    started = time.perf_counter()
    driver = build_fullbody_driver(config)
    build_seconds = time.perf_counter() - started
    model = driver.model
    joints = [
        model.joint(index).name
        for index in range(model.njnt)
        if model.joint(index).name.startswith(FREE_JOINT_PREFIXES)
    ]
    addresses = [model.jnt_qposadr[model.joint(name).id] for name in joints]
    driver.reset()
    qpos, controls = [], []
    started = time.perf_counter()
    for intent, periods in PROGRAM:
        for _ in range(periods):
            driver.advance(intent)
            qpos.append(driver.data.qpos[addresses].copy())
            values = driver.mapper.sample(driver.data).values
            controls.append([values[key] for key in sorted(values)])
    step_seconds = time.perf_counter() - started
    frames = len(qpos)
    return {
        "joints": joints,
        "qpos": np.array(qpos),
        "controls": np.array(controls),
        "channels": sorted(values),
        "muscles": int(model.nu),
        "tendons": int(model.ntendon),
        "build_seconds": build_seconds,
        "simulated_seconds": frames * config.control_dt,
        "wall_seconds": step_seconds,
        "real_time_factor": frames * config.control_dt / step_seconds,
    }


def _compare(reference: dict, variant: dict) -> dict:
    assert reference["joints"] == variant["joints"]
    assert reference["channels"] == variant["channels"]
    joint_error = np.abs(reference["qpos"] - variant["qpos"])
    control_error = np.abs(reference["controls"] - variant["controls"])
    return {
        **{
            k: variant[k]
            for k in (
                "muscles",
                "tendons",
                "build_seconds",
                "simulated_seconds",
                "wall_seconds",
                "real_time_factor",
            )
        },
        "speedup": reference["wall_seconds"] / variant["wall_seconds"],
        "max_free_joint_error": float(joint_error.max()),
        "worst_joint": reference["joints"][int(joint_error.max(axis=0).argmax())],
        "max_control_error_by_channel": {
            channel: float(control_error[:, index].max())
            for index, channel in enumerate(reference["channels"])
        },
        "rms_control_error_by_channel": {
            channel: float(np.sqrt(np.mean(control_error[:, index] ** 2)))
            for index, channel in enumerate(reference["channels"])
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    base = FullBodyDriverConfig()
    variants = {
        "full_1ms": base,
        "pruned_1ms": replace(base, prune_locked_muscles=True),
        "full_2ms": replace(base, timestep=0.002),
        "pruned_2ms": replace(base, timestep=0.002, prune_locked_muscles=True),
    }
    traces = {name: _trace(config) for name, config in variants.items()}
    reference = traces["full_1ms"]
    report = {
        "program_periods": sum(periods for _, periods in PROGRAM),
        "reference": "full_1ms",
        "free_joints": len(reference["joints"]),
        "variants": {
            name: _compare(reference, trace) for name, trace in traces.items()
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure bridge payload serialization; no server is required.

Run with the same interpreter before and after a change:
python benchmarks/serialization.py --output artifacts/serialization.json
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import timeit
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from myosuite_carla import VehicleCommand, VehicleTelemetry
from myosuite_carla.bridge import BridgeFrame
from myosuite_carla.inputs import ControlSample


def benchmark(iterations: int, repeats: int) -> dict[str, Any]:
    """Return timings for immutable command, telemetry and frame payloads."""
    command = VehicleCommand(throttle=0.3, steer=-0.25, brake=0.0, gear=2)
    telemetry = VehicleTelemetry(
        42,
        0.84,
        (1.0, 2.0, 3.0),
        (0.0, 5.0, 0.0),
        (3.0, 4.0, 0.0),
        (0.1, 0.2, 0.3),
        (0.0, 0.0, 1.0),
    )
    frame = BridgeFrame(
        ControlSample(command, {"throttle": 0.3, "steer": -0.25}), telemetry, 0.84
    )
    operations: dict[str, Callable[[], Any]] = {
        "command": command.as_dict,
        "telemetry": telemetry.as_dict,
        "frame": frame.as_dict,
    }

    def legacy_telemetry() -> dict[str, Any]:
        return {
            **asdict(telemetry),
            "speed": math.sqrt(sum(value * value for value in telemetry.velocity)),
        }

    def legacy_frame() -> dict[str, Any]:
        return {
            "inputs": dict(frame.sample.values),
            "control": asdict(command),
            "vehicle": legacy_telemetry(),
            "simulation_time": frame.simulation_time,
        }

    reference: dict[str, Callable[[], Any]] = {
        "command": lambda: asdict(command),
        "telemetry": legacy_telemetry,
        "frame": legacy_frame,
    }
    for name in operations:
        if operations[name]() != reference[name]():
            raise RuntimeError(f"Serialization payload changed: {name}")
    functions = [
        (name + suffix, function)
        for name in operations
        for suffix, function in (
            ("", operations[name]),
            ("_recursive_reference", reference[name]),
        )
    ]
    timings: dict[str, Any] = {name: [] for name, _ in functions}
    for repeat in range(repeats):
        # Alternate order so both implementations share the same machine load.
        for name, operation in functions[:: 1 if repeat % 2 == 0 else -1]:
            seconds = timeit.timeit(operation, number=iterations)
            timings[name].append(seconds / iterations * 1e6)
    timings = {
        name: {
            "median_microseconds": statistics.median(samples),
            "samples_microseconds": samples,
        }
        for name, samples in timings.items()
    }
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "iterations_per_repeat": iterations,
        "repeats": repeats,
        "timings": timings,
        "payload": frame.as_dict(),
    }


def main() -> None:
    """Write a repeatable local microbenchmark report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.repeats <= 0:
        parser.error("iterations and repeats must be positive")
    report = benchmark(args.iterations, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report["timings"], indent=2))


if __name__ == "__main__":
    main()

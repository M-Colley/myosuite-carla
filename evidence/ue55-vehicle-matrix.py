"""Run each muscle-operated acceptance trial for every four-wheel CARLA vehicle.

Supplementary evidence script. Unlike ``examples/validate_carla.py``, a failed
control does not stop the remaining controls or vehicles, so the matrix is
complete. Trials reuse the package's acceptance fixture and thresholds.
"""

from __future__ import annotations

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco

from myosuite_carla.validation import _spawn, _trial

SUMMARY = (
    "settling_frames",
    "excitation",
    "cruising_throttle",
    "peaks",
    "start_speed_mps",
    "end_speed_mps",
    "displacement_metres",
    "yaw_change_degrees",
    "vehicle_destroyed",
    "settings_restored",
    "error",
    "cleanup_errors",
)


def _physics(world: Any, blueprint: str, timeout: float) -> dict[str, Any]:
    vehicle = _spawn(world, blueprint)
    try:
        world.wait_for_tick(timeout)
        physics = vehicle.get_physics_control()
        wheels = list(physics.wheels)
        return {
            "mass_kg": float(physics.mass),
            "max_rpm": float(physics.max_rpm),
            "max_steer_angle_deg": max(float(w.max_steer_angle) for w in wheels),
            "max_brake_torque": max(float(w.max_brake_torque) for w in wheels),
        }
    finally:
        vehicle.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-output", type=Path)
    args = parser.parse_args()
    carla = importlib.import_module("carla")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    blueprints = sorted(
        b.id
        for b in world.get_blueprint_library().filter("vehicle.*")
        if b.has_attribute("number_of_wheels")
        and int(b.get_attribute("number_of_wheels")) == 4
    )
    report: dict[str, Any] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "client_version": client.get_client_version(),
        "server_version": client.get_server_version(),
        "map": world.get_map().name,
        "mujoco_version": mujoco.__version__,
        "vehicles": [],
    }
    full: list[dict[str, Any]] = []
    for blueprint in blueprints:
        row: dict[str, Any] = {"blueprint": blueprint, "trials": {}}
        try:
            row["physics"] = _physics(world, blueprint, args.timeout)
        except Exception as error:
            row["physics_error"] = f"{type(error).__name__}: {error}"
        for control in ("throttle", "brake", "steering"):
            evidence: dict[str, Any] = {}
            try:
                _trial(client, carla, control, blueprint, args.timeout, evidence)
            except Exception:
                pass  # The trial records its own error and cleanup evidence.
            full.append({"blueprint": blueprint, **evidence})
            summary = {key: evidence[key] for key in SUMMARY if key in evidence}
            summary["passed"] = bool(evidence.get("passed", False))
            summary["samples"] = len(evidence.get("samples", []))
            row["trials"][control] = summary
        row["passed"] = all(t["passed"] for t in row["trials"].values())
        report["vehicles"].append(row)
        print(
            blueprint,
            {c: t["passed"] for c, t in row["trials"].items()},
            flush=True,
        )
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["vehicles_passed"] = sum(row["passed"] for row in report["vehicles"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.full_output is not None:
        args.full_output.write_text(json.dumps(full, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Exercise the frozen vehicle in the physical embodied human control loop."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from myosuite_carla.driver_models import AccelerationControlConfig, ReflexDriverConfig
from myosuite_carla.empirical_following import (
    ClosedLoopConfig,
    ClosedLoopFollowingProtocol,
    FollowingProtocol,
)
from myosuite_carla.ngsim import FollowingTrial
from myosuite_carla.vehicle_model import VehicleModelParams


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.selection.open(encoding="utf-8") as stream:
        selection = json.load(stream)
    params = VehicleModelParams.from_dict(selection["parameters"])
    reflex = ReflexDriverConfig(
        perception_delay=0.02,
        acceleration_control=AccelerationControlConfig(vehicle=params),
    )
    time = np.arange(21) / 10.0
    speed = np.full_like(time, 5.0)
    trial = FollowingTrial(
        driver="scripted-patrol-coupling",
        time=time,
        position=5.0 * time,
        speed=speed,
        acceleration=np.zeros_like(time),
        gap=np.full_like(time, 18.0),
        leader_speed=speed.copy(),
    )
    source = FollowingProtocol([trial], desired_speed=10.0)
    with ClosedLoopFollowingProtocol(
        source,
        config=ClosedLoopConfig(seconds=None, keep_trajectories=True),
        reflex_config=reflex,
        vehicle_parameters=params,
    ) as protocol:
        result = protocol.report_pair(
            {}, {"time_headway": 2.0}, groups=source.groups, seed=42
        )
    result["candidate"] = selection["selected"]
    result["fit_report_sha256"] = selection["fit_report_sha256"]
    result["physical_runtime"] = {
        name: version(name) for name in ("myosuite", "myo-sim", "mujoco")
    }
    result["source_kind"] = "scripted control loop check"
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    assert result["status"] == "completed"
    assert result["muscles"] == 114 and result["body_builds"] == 1
    assert len(result["episodes"]) == 2
    assert all(
        row["coverage_fraction"] == 1
        and row["surrogate_closed"]
        and not row["collision"]
        and not row["off_road"]
        and row["vehicle_parameters"] == params.as_dict()
        for row in result["episodes"]
    )
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

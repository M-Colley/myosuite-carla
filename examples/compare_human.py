"""Compare the muscle-driven reflex driver with human driving (aggregates only).

Human windows are cut from each participant's lane-keeping segments (rows
whose label equals ``--keep``); model windows come from free-road drives of
the full body on the fitted vehicle surrogate, for several motor-noise levels
and seeds. Both become amplitude-free steering features and longitudinal
acceleration features (``myosuite_carla.human_comparison``). The report holds
feature medians, human-vs-human floor AUCs and model-vs-human AUCs. No human
row and no participant identifier is written: participants are numbered in
an arbitrary order.

The defaults match a CSV with columns ``time`` (s), ``Steering_angle``,
``Acceleration`` (m/s^2), ``speed`` (km/h), ``Participant``, ``Drive`` and
``label_grouped``. Speeds outside the surrogate's identified range make the
comparison context-mismatched; the report states both speed distributions.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from myosuite_carla import FullBodyDriverConfig, build_fullbody_driver
from myosuite_carla.driver_models import ReflexDriver
from myosuite_carla.driving_env import FullBodyDrivingEnv
from myosuite_carla.human_comparison import (
    FEATURES,
    SIGNAL_FEATURES,
    adequacy,
    discriminability,
    human_floor,
    summarize,
    window_features,
)
from myosuite_carla.scenario import Scenario


def _human(args: argparse.Namespace) -> dict[str, Any]:
    columns = [
        args.time,
        args.steer,
        args.accel,
        args.speed,
        args.participant,
        args.drive,
        args.label,
    ]
    frame = pd.read_csv(args.human, usecols=columns, low_memory=False)
    for name in (args.time, args.steer, args.accel, args.speed):
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame = frame.dropna(subset=[args.time, args.steer, args.accel])
    anonymous = {
        name: index
        for index, name in enumerate(
            np.random.default_rng(0).permutation(
                sorted(map(str, frame[args.participant].unique()))
            )
        )
    }
    features, participants, drives, speeds = [], [], [], []
    checks = {"steer": [0.0, 0.0, math.inf], "acceleration": [0.0, 0.0, math.inf]}
    for (participant, drive), group in frame.groupby([args.participant, args.drive]):
        group = group.sort_values(args.time)
        keep = (group[args.label] == args.keep).to_numpy()
        gaps = np.diff(group[args.time].to_numpy(), prepend=-np.inf) > 0.5
        segment = np.cumsum(~keep | gaps)
        for _, part in group[keep].groupby(segment[keep]):
            time = part[args.time].to_numpy()
            unique = np.r_[True, np.diff(time) > 0]
            for signal, column in (("steer", args.steer), ("acceleration", args.accel)):
                check = adequacy(time, part[column].to_numpy(), window=args.window)
                totals = checks[signal]
                totals[0] += check["windows"]
                totals[1] += check["constant_windows"]
                if check["resolution"] > 0:
                    totals[2] = min(totals[2], check["resolution"])
            rows = window_features(
                time[unique],
                part[args.steer].to_numpy()[unique],
                part[args.accel].to_numpy()[unique],
                window=args.window,
            )
            if len(rows):
                features.append(rows)
                index = anonymous[str(participant)]
                participants += [index] * len(rows)
                drives += [str(drive)] * len(rows)
                speeds.append(part[args.speed].to_numpy() * args.speed_scale)
    adequacy_report = {
        signal: {
            "distinct_values": int(frame[column].nunique()),
            "resolution": totals[2],
            "constant_window_fraction": totals[1] / max(totals[0], 1.0),
        }
        for (signal, totals), column in zip(checks.items(), (args.steer, args.accel))
    }
    return {
        "features": np.vstack(features),
        "participants": participants,
        "drives": drives,
        "speed": np.concatenate(speeds),
        "adequacy": adequacy_report,
    }


def _model(args: argparse.Namespace, noise: float) -> dict[str, Any]:
    driver = build_fullbody_driver(FullBodyDriverConfig.fast(motor_noise_signal=noise))
    scenario = Scenario(
        name="free",
        start_speed=args.model_speed,
        target_speed=args.model_speed,
        duration=args.seconds,
    )
    env = FullBodyDrivingEnv(scenario, driver=driver)
    features, groups, speeds = [], [], []
    for episode in range(args.episodes):
        policy = ReflexDriver(env.dt)
        env.reset(seed=episode)
        time, steer, accel = [], [], []
        while True:
            _, _, terminated, truncated, info = env.step(policy.act(env.state))
            if info["scenario"]["time"] >= args.warmup:
                time.append(info["scenario"]["time"])
                steer.append(info["inputs"]["steer"])
                accel.append(env.state["a_forward_mps2"])
                speeds.append(info["scenario"]["speed_mps"])
            if terminated or truncated:
                break
        rows = window_features(time, steer, accel, window=args.window)
        features.append(rows)
        groups += [f"episode{episode}"] * len(rows)
    return {
        "features": np.vstack(features),
        "groups": groups,
        "speed": np.asarray(speeds),
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return dict(zip(("q05", "q25", "median", "q75", "q95"), map(float, q)))


def main() -> int:
    """Build features, compute AUCs and write the aggregate report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, required=True)
    parser.add_argument("--time", default="time")
    parser.add_argument("--steer", default="Steering_angle")
    parser.add_argument("--accel", default="Acceleration")
    parser.add_argument("--speed", default="speed")
    parser.add_argument("--speed-scale", type=float, default=1 / 3.6)
    parser.add_argument("--participant", default="Participant")
    parser.add_argument("--drive", default="Drive")
    parser.add_argument("--label", default="label_grouped")
    parser.add_argument("--keep", default="remain")
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument(
        "--max-constant",
        type=float,
        default=0.5,
        help="drop features of a human signal constant in more windows than this",
    )
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument("--model-speed", type=float, default=13.0)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/human-comparison.json")
    )
    args = parser.parse_args()
    human = _human(args)
    # Features of a signal that is constant in most human windows are unusable.
    valid = [
        index
        for index, name in enumerate(FEATURES)
        for signal, names in SIGNAL_FEATURES.items()
        if name in names
        and human["adequacy"][signal]["constant_window_fraction"] < args.max_constant
    ]
    columns = [FEATURES[index] for index in valid]
    report: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "features": list(FEATURES),
        "compared_features": columns,
        "max_constant_window_fraction": args.max_constant,
        "window_s": args.window,
        "human": {
            "participants": len(set(human["participants"])),
            "windows": int(len(human["features"])),
            "speed_mps": _quantiles(human["speed"]),
            "adequacy": human["adequacy"],
            "features": summarize(human["features"]),
            "floor": (
                human_floor(
                    human["features"][:, valid],
                    human["participants"],
                    human["drives"],
                )
                if valid
                else None
            ),
        },
        "model": {},
    }
    human_groups = [
        f"{participant}:{drive}"
        for participant, drive in zip(human["participants"], human["drives"])
    ]
    for noise in args.noise:
        model = _model(args, noise)
        report["model"][f"{noise:g}"] = {
            "motor_noise_signal": noise,
            "episodes": args.episodes,
            "windows": int(len(model["features"])),
            "speed_mps": _quantiles(model["speed"]),
            "features": summarize(model["features"]),
            "versus_human": (
                discriminability(
                    model["features"][:, valid],
                    human["features"][:, valid],
                    model["groups"],
                    human_groups,
                    grouping="independent",
                )
                if valid
                else None
            ),
        }
        print(noise, report["model"][f"{noise:g}"]["versus_human"], flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"floor": report["human"]["floor"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

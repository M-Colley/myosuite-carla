"""Throughput of batched full-body rollouts on the vehicle surrogate.

Runs the reflex driver in ``gymnasium.vector.AsyncVectorEnv`` with one CPU
driver per worker process and reports aggregate steps per second and the
real-time factor per worker, after a warm-up, for each worker count. The
single-process environment without the vector wrapper is the baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from functools import partial
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from myosuite_carla.driver_models import ReflexDriver
from myosuite_carla.driving_env import make_driving_env


def _single(steps: int, scenario: str) -> dict[str, float]:
    env = make_driving_env(scenario)
    policy = ReflexDriver(env.dt)
    env.reset(seed=0)
    for _ in range(25):
        env.step(policy.act(env.state))
    started = time.perf_counter()
    for _ in range(steps):
        env.step(policy.act(env.state))
    elapsed = time.perf_counter() - started
    return {
        "workers": 1,
        "vectorized": False,
        "steps_per_second": steps / elapsed,
        "real_time_factor_per_worker": steps * env.dt / elapsed,
    }


def _vector(workers: int, steps: int, scenario: str) -> dict[str, float]:
    envs = gym.vector.AsyncVectorEnv(
        [partial(make_driving_env, scenario) for _ in range(workers)]
    )
    names = envs.get_attr("observation_names")[0]
    dt = envs.get_attr("dt")[0]
    policies = [ReflexDriver(dt) for _ in range(workers)]
    observations, _ = envs.reset(seed=list(range(workers)))

    def act(batch: np.ndarray) -> np.ndarray:
        intents = [
            policy.act(dict(zip(names, map(float, row))))
            for policy, row in zip(policies, batch)
        ]
        return np.asarray(
            [
                [
                    i.right_foot,
                    i.left_foot,
                    i.steer,
                    i.gear_finger,
                    i.reverse_finger,
                    i.park_finger,
                ]
                for i in intents
            ],
            dtype=np.float32,
        )

    for _ in range(25):
        observations, *_ = envs.step(act(observations))
    started = time.perf_counter()
    for _ in range(steps):
        observations, *_ = envs.step(act(observations))
    elapsed = time.perf_counter() - started
    envs.close()
    return {
        "workers": workers,
        "vectorized": True,
        "steps_per_second": workers * steps / elapsed,
        "real_time_factor_per_worker": steps * dt / elapsed,
    }


def main() -> int:
    """Measure and write the scaling record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--scenario", default="lead_vehicle_braking")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/vector-rollout.json")
    )
    args = parser.parse_args()
    results = [_single(args.steps, args.scenario)]
    print(results[-1], flush=True)
    for workers in args.workers:
        results.append(_vector(workers, args.steps, args.scenario))
        print(results[-1], flush=True)
    record = {
        "scenario": args.scenario,
        "steps_per_worker": args.steps,
        "driver": "FullBodyDriverConfig.fast() with the fitted surrogate",
        "machine": {
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

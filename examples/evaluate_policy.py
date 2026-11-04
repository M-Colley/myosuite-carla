"""Evaluate a trained driving policy against the reflex driver and human drivers.

Every evaluation episode runs twice from the same scenario and seed: with the
trained policy (deterministic) and with the reflex driver. The reflex driver
uses a zero demand residual in direct-policy evaluations. Per stage the report
gives success, collisions, lane error, minimum gap and return for both, and the paired
difference of returns with a bootstrap 95% interval.

For human-likeness, held-out NGSIM trials (one per driver, drivers never used
in training) are replayed: the lead vehicle follows the recorded leader and
the ego starts like the human follower. Both agents are compared with those
human followers by (a) the RMS gap error against the follower of the same
leader, (b) the judge's driver-grouped cross-validated AUC with gradient
boosting and (c) the per-feature overlap scoreboard, next to a human-vs-human
floor from two disjoint halves of the held-out source groups. Every replay
and its human source share a cross-validation group. Paired intervals
resample original source means.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


def _reflex_environment(env: Any) -> Any:
    """Reuse the physical task with a reflex baseline for each policy mode."""
    if env.mode != "direct":
        return env
    from myosuite_carla.training import DrivingTaskWrapper

    return DrivingTaskWrapper(
        env.unwrapped,
        mode="demand",
        vehicle_spread=0.0,
        expert=env.expert.config,
        takeover_population=False,
    )


def _run_chunk(job: dict[str, Any]) -> list[dict[str, Any]]:
    import torch

    torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC

    from myosuite_carla.human_comparison import following_features
    from myosuite_carla.ngsim import load_trials
    from myosuite_carla.training import (
        STAGES,
        SamplerConfig,
        ScenarioSampler,
        scenario_from_trial,
    )
    from myosuite_carla.training_sb3 import make_training_env

    run = Path(job["run"])
    config = json.loads((run / "config.json").read_text())
    mode = config["args"]["mode"]
    algo = PPO if config["args"]["algo"] == "ppo" else SAC
    model = algo.load(run / "model.zip", device="cpu")
    with (run / "vecnormalize.pkl").open("rb") as handle:
        normalizer = pickle.load(handle)
    normalizer.training = False
    env = make_training_env(0, mode=mode, stages=STAGES, vehicle_spread=0.0)
    env.takeover_population = False
    reflex_env = _reflex_environment(env)
    sampler = ScenarioSampler(STAGES, (), SamplerConfig(), len(STAGES) - 1)
    trials = load_trials(job["trials"]) if job["trials"] else []
    results = []
    for episode in job["episodes"]:
        if episode["kind"] == "ngsim":
            trial = trials[episode["index"]]
            scenario = scenario_from_trial(trial)
        else:
            rng = np.random.default_rng(episode["seed"])
            scenario = sampler.sample(rng, episode["kind"])
        for agent in ("policy", "reflex"):
            current_env = env if agent == "policy" else reflex_env
            observation, _ = current_env.reset(
                seed=episode["seed"], options={"scenario": scenario}
            )
            total = task = 0.0
            while True:
                if agent == "policy":
                    action, _ = model.predict(
                        normalizer.normalize_obs(observation), deterministic=True
                    )
                else:
                    action = np.zeros(current_env.action_space.shape)
                observation, reward, terminated, truncated, info = current_env.step(
                    action
                )
                total += reward
                terms = info["reward_terms"]
                task += reward - terms["residual"] - terms["smoothness"]
                if terminated or truncated:
                    break
            record = {
                **episode,
                "agent": agent,
                "return": float(total),
                "task_return": float(task),
                "success": bool(info["episode_success"]),
                **info["episode_metrics"],
            }
            if episode["kind"] == "ngsim":
                record["driver"] = trial.driver
                record["source_group"] = trial.group_id
            if "following_trial" in info:
                record["features"] = following_features(
                    info["following_trial"]
                ).tolist()
                record["gap"] = info["following_trial"].gap.tolist()
            results.append(record)
    return results


def _gap_rmse(row: dict[str, Any], trial: Any) -> float:
    """RMS gap difference over the shared 10 Hz samples of agent and human."""
    agent = np.asarray(row["gap"])
    count = min(len(agent), len(trial.gap))
    return float(np.sqrt(np.mean((agent[:count] - trial.gap[:count]) ** 2)))


def _source_means(
    differences: np.ndarray, groups: Sequence[Any] | None = None
) -> np.ndarray:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Paired differences must be a nonempty finite vector")
    identifiers = list(range(len(values))) if groups is None else list(groups)
    if len(identifiers) != len(values):
        raise ValueError("Each paired difference needs one source group")
    labels = np.asarray([str(group) for group in identifiers])
    return np.asarray(
        [np.mean(values[labels == group]) for group in dict.fromkeys(labels)]
    )


def _bootstrap(
    differences: np.ndarray, *, groups: Sequence[Any] | None = None
) -> list[float]:
    from scipy import stats

    units = _source_means(differences, groups)
    if len(units) < 2:
        raise ValueError("A bootstrap interval needs at least two source groups")
    if np.allclose(units, units[0]):
        return [float(np.mean(units))] * 2
    result = stats.bootstrap(
        (units,), np.mean, random_state=0, n_resamples=2000, method="percentile"
    )
    return [
        float(result.confidence_interval.low),
        float(result.confidence_interval.high),
    ]


def _paired_difference(
    differences: np.ndarray, groups: Sequence[Any]
) -> dict[str, Any]:
    units = _source_means(differences, groups)
    return {
        "mean": float(units.mean()),
        "ci95": _bootstrap(units) if len(units) >= 2 else None,
        "sources": len(units),
        "ci_method": "percentile bootstrap of source means",
        "ci_resamples": 2000,
        "ci_seed": 0,
    }


def _human_comparison(
    human: Sequence[Any],
    results: Sequence[dict[str, Any]],
    *,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare matched trials with the original source held out across labels."""
    from myosuite_carla.human_comparison import (
        FOLLOWING_FEATURES,
        discriminability,
        following_features,
        scoreboard,
    )

    human_features = np.vstack([following_features(trial) for trial in human])
    sources = [trial.group_id for trial in human]
    if len(set(sources)) < 2:
        raise ValueError("Paired evaluation needs at least two original source groups")
    trial_indices = list(range(len(human))) if indices is None else list(indices)
    if len(trial_indices) != len(human) or len(set(trial_indices)) != len(human):
        raise ValueError("Each human trial needs a unique evaluation index")
    unique_sources = list(dict.fromkeys(sources))
    first_half = set(unique_sources[: len(unique_sources) // 2])
    left = np.asarray([source in first_half for source in sources])
    floor = None
    if len(first_half) >= 2 and len(unique_sources) - len(first_half) >= 2:
        floor = discriminability(
            human_features[left],
            human_features[~left],
            [source for source in sources if source in first_half],
            [source for source in sources if source not in first_half],
            model="boosting",
            grouping="independent",
        )
    likeness: dict[str, Any] = {
        "trials": len(human),
        "sources": len(unique_sources),
        "source_groups": sources,
        "grouping_unit": "original source group",
    }
    errors = {}
    for agent in ("policy", "reflex"):
        candidates = [
            row for row in results if row["kind"] == "ngsim" and row["agent"] == agent
        ]
        lookup = {row["index"]: row for row in candidates}
        if len(lookup) != len(candidates) or set(lookup) != set(trial_indices):
            raise ValueError(
                "Paired results must match every human evaluation index once"
            )
        rows = [lookup[index] for index in trial_indices]
        if [row.get("source_group") for row in rows] != sources:
            raise ValueError(
                "Paired results must retain the original human source groups"
            )
        features = np.asarray([row["features"] for row in rows])
        errors[agent] = np.asarray(
            [_gap_rmse(row, trial) for row, trial in zip(rows, human)]
        )
        judge = discriminability(
            features,
            human_features,
            sources,
            sources,
            model="boosting",
            grouping="shared",
        )
        likeness[agent] = {
            "gap_rmse_m": {
                "mean": float(np.mean(errors[agent])),
                "median": float(np.median(errors[agent])),
            },
            "judge_auc": judge["auc"],
            "judge": judge,
            "scoreboard": scoreboard(human_features, features, FOLLOWING_FEATURES),
        }
    likeness["gap_rmse_difference_m"] = _paired_difference(
        errors["policy"] - errors["reflex"], sources
    )
    return {
        "human_floor_auc": floor["auc"] if floor is not None else None,
        "human_floor": floor,
        "human_likeness": likeness,
    }


def main() -> int:
    """Run the evaluation and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--ngsim-trials", type=int, default=120)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from myosuite_carla.ngsim import load_trials
    from myosuite_carla.training import STAGES

    output = args.output or args.run / "evaluation.json"
    episodes = [
        {"kind": stage, "seed": 10_000 + index}
        for stage in STAGES
        for index in range(args.episodes)
    ]
    trials_path = args.run / "ngsim_test.npz"
    human = []
    chosen: list[int] = []
    if trials_path.exists() and args.ngsim_trials > 0:
        bank = load_trials(trials_path)
        rng = np.random.default_rng(0)
        seen = set()
        for index in rng.permutation(len(bank)):
            if bank[index].group_id not in seen:
                seen.add(bank[index].group_id)
                chosen.append(int(index))
            if len(chosen) == args.ngsim_trials:
                break
        episodes += [
            {
                "kind": "ngsim",
                "seed": 20_000 + n,
                "index": index,
                "driver": bank[index].driver,
                "source_group": bank[index].group_id,
            }
            for n, index in enumerate(chosen)
        ]
        human = [bank[index] for index in chosen]
    chunks = [episodes[i :: args.workers] for i in range(args.workers)]
    jobs = [
        {
            "run": str(args.run),
            "episodes": chunk,
            "trials": str(trials_path) if human else None,
        }
        for chunk in chunks
        if chunk
    ]
    with ProcessPoolExecutor(len(jobs)) as pool:
        results = [row for rows in pool.map(_run_chunk, jobs) for row in rows]

    report: dict[str, Any] = {"run": str(args.run), "stages": {}}
    for stage in [*STAGES, "ngsim"]:
        rows = {
            agent: sorted(
                (
                    row
                    for row in results
                    if row["kind"] == stage and row["agent"] == agent
                ),
                key=lambda row: row["seed"],
            )
            for agent in ("policy", "reflex")
        }
        if not rows["policy"]:
            continue
        summary = {}
        for agent, items in rows.items():
            gaps = [row["min_gap_m"] for row in items if np.isfinite(row["min_gap_m"])]
            summary[agent] = {
                "episodes": len(items),
                "success_rate": float(np.mean([row["success"] for row in items])),
                "collisions": int(sum(row["collision"] for row in items)),
                "off_road": int(sum(row["off_road"] for row in items)),
                "lane_rms_m": float(np.mean([row["lane_rms_m"] for row in items])),
                "min_gap_m": float(np.min(gaps)) if gaps else None,
                "return": float(np.mean([row["return"] for row in items])),
                "task_return": float(np.mean([row["task_return"] for row in items])),
            }
        differences = np.asarray(
            [
                p["task_return"] - r["task_return"]
                for p, r in zip(rows["policy"], rows["reflex"])
            ]
        )
        summary["task_return_difference"] = _paired_difference(
            differences,
            [
                row["source_group"] if stage == "ngsim" else row["seed"]
                for row in rows["policy"]
            ],
        )
        report["stages"][stage] = summary

    if human:
        report.update(_human_comparison(human, results, indices=chosen))
    output.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

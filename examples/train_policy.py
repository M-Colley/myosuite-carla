"""Train a full-body driving policy with Stable-Baselines3.

``--mode demand`` (default) learns a bounded correction to the reflex
driver's desired acceleration and steer, which its proprioceptive loops and
the muscles execute, so training starts from a working driver (``--mode
intent`` corrects the motor program instead); ``--mode direct`` learns the
motor program itself, warm-started by behaviour cloning from the reflex
driver and refined with DAgger rounds before reinforcement learning. Training
runs on the fitted vehicle surrogate with a curriculum over scenario stages.

With ``--ngsim`` the car-following stage replays human leaders from NGSIM
I-80 trials (``examples/download_ngsim.py``); trials are split by driver, the
held-out bank is saved for ``evaluate_policy.py``, and ``--style-weight`` adds
a trial-level human-likeness reward from a discriminator fit on the training
drivers only.

Outputs in ``--output-dir``: ``model.zip``, ``vecnormalize.pkl``,
``config.json`` (arguments, versions, commit), ``episodes.jsonl``,
``curriculum.json``, ``progress.csv`` and, with NGSIM, ``ngsim_train.npz`` and
``ngsim_test.npz``.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    """Train and save one policy run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("demand", "intent", "direct"), default="demand"
    )
    parser.add_argument("--algo", choices=("ppo", "sac"), default="ppo")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--curriculum-threshold", type=float, default=0.8)
    parser.add_argument("--curriculum-window", type=int, default=40)
    parser.add_argument("--ngsim", type=Path, help="NGSIM CSV from download_ngsim.py")
    parser.add_argument("--style-weight", type=float, default=2.0)
    parser.add_argument("--style-window", type=float, default=5.0)
    parser.add_argument("--motor-noise", type=float, default=0.0)
    parser.add_argument("--vehicle-spread", type=float, default=0.1)
    parser.add_argument("--bc-steps", type=int, default=20_000)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/runs/demand_ppo")
    )
    args = parser.parse_args()

    import torch
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure

    from myosuite_carla.human_comparison import following_features
    from myosuite_carla.ngsim import (
        following_trials,
        load_tracks,
        save_trials,
        split_by_driver,
        split_windows,
    )
    from myosuite_carla.training import STAGES
    from myosuite_carla.training_sb3 import (
        CurriculumCallback,
        EpisodeLogCallback,
        StyleRewardCallback,
        behaviour_clone,
        collect_expert,
        make_vec_env,
    )

    torch.set_num_threads(args.torch_threads)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    stages = tuple(args.stages or STAGES)
    trials_path = None
    human = None
    if args.ngsim:
        trials = following_trials(load_tracks(args.ngsim))
        train, test = split_by_driver(trials, seed=args.seed)
        trials_path = str(save_trials(train, output / "ngsim_train.npz"))
        save_trials(test, output / "ngsim_test.npz")
        human = np.vstack(
            [
                following_features(window)
                for window in split_windows(train, args.style_window)
            ]
        )
        print(f"NGSIM: {len(train)} training and {len(test)} held-out trials")
    residual = args.mode != "direct"
    venv = make_vec_env(
        args.n_envs,
        args.seed,
        mode=args.mode,
        stages=stages,
        stage=args.start_stage,
        trials_path=trials_path,
        motor_noise=args.motor_noise,
        vehicle_spread=args.vehicle_spread,
    )
    if args.algo == "sac":
        model = SAC(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            buffer_size=500_000,
            batch_size=256,
            learning_starts=0 if not residual else 5_000,
            gamma=0.995,
            train_freq=1,
            gradient_steps=1,
            policy_kwargs={"net_arch": [256, 256]},
            seed=args.seed,
            verbose=0,
        )
    else:
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=512,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.0,
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.5},
            seed=args.seed,
            verbose=0,
        )
    model.set_logger(configure(str(output), ["csv"]))
    record: dict = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) for key, value in vars(args).items()},
        "stages": list(stages),
        "commit": _commit(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "stable_baselines3": __import__("stable_baselines3").__version__,
        },
    }
    started = time.perf_counter()
    if not residual and args.bc_steps > 0:
        bc = []
        for round_ in range(args.dagger_rounds + 1):
            # Round 0 is behaviour cloning; later rounds let the learner act.
            observations, labels = collect_expert(
                venv,
                args.bc_steps,
                policy=model if round_ else None,
                expert_probability=1.0 if round_ == 0 else 0.5,
                seed=args.seed + round_,
            )
            if round_ == 0:
                data = (observations, labels)
            else:
                data = (
                    np.concatenate([data[0], observations]),
                    np.concatenate([data[1], labels]),
                )
            losses = behaviour_clone(model, *data, epochs=15, seed=args.seed)
            bc.append({"round": round_, "samples": len(data[0]), "loss": losses[-1]})
            print(f"imitation round {round_}: {len(data[0])} samples")
        record["imitation"] = bc
    curriculum = CurriculumCallback(
        len(stages),
        window=args.curriculum_window,
        threshold=args.curriculum_threshold,
        start=args.start_stage,
    )
    callbacks = [
        curriculum,
        EpisodeLogCallback(output / "episodes.jsonl"),
        CheckpointCallback(
            max(args.timesteps // (4 * args.n_envs), 1),
            str(output / "checkpoints"),
            save_vecnormalize=True,
        ),
    ]
    if human is not None and args.style_weight > 0:
        callbacks.insert(0, StyleRewardCallback(human, weight=args.style_weight))
    model.learn(args.timesteps, callback=CallbackList(callbacks), progress_bar=False)
    record["seconds"] = time.perf_counter() - started
    record["timesteps"] = int(model.num_timesteps)
    record["final_stage"] = curriculum.stage
    model.save(output / "model.zip")
    venv.save(str(output / "vecnormalize.pkl"))
    (output / "curriculum.json").write_text(
        json.dumps(curriculum.history, indent=1), encoding="utf-8"
    )
    (output / "config.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    venv.close()
    print(json.dumps({k: record[k] for k in ("seconds", "timesteps", "final_stage")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

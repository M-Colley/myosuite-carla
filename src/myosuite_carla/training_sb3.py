"""Stable-Baselines3 training of full-body driving policies.

Follows the conventions of arm_force_driving's training (Stable-Baselines3,
``VecNormalize`` saved with the policy, a residual on an expert controller and
a judge-based trial reward) in an independent implementation. Importing this
module imports Stable-Baselines3 and PyTorch.

* :func:`make_vec_env` builds normalized, monitored worker processes that each
  hold one full-body driver and one vehicle surrogate.
* :class:`CurriculumCallback` unlocks the next scenario stage once the recent
  success rate reaches a threshold.
* :class:`StyleRewardCallback` pays, at the end of each car-following
  episode, the clipped log-odds of a :class:`StyleDiscriminator` that is refit
  on human training trials against the latest agent trials.
* :class:`EpisodeLogCallback` writes every episode's metrics as JSON lines.
* :func:`collect_expert` and :func:`behaviour_clone` warm-start a direct
  policy from the reflex driver; repeated with the learner acting, they form
  DAgger (Ross, Gordon and Bagnell, 2011).
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecMonitor,
    VecNormalize,
)

from .driving_env import FullBodyDrivingEnv
from .fullbody import FullBodyDriverConfig, build_fullbody_driver
from .human_comparison import StyleDiscriminator, following_features
from .ngsim import load_trials
from .training import STAGES, DrivingTaskWrapper, SamplerConfig, ScenarioSampler


def make_training_env(
    seed: int = 0,
    *,
    mode: str = "demand",
    stages: Sequence[str] = STAGES,
    stage: int = 0,
    trials_path: str | None = None,
    motor_noise: float = 0.0,
    vehicle_spread: float = 0.1,
    fast: bool = True,
    sampler_config: SamplerConfig | None = None,
) -> DrivingTaskWrapper:
    """Build one training environment; picklable through ``functools.partial``."""
    config = (
        FullBodyDriverConfig.fast(motor_noise_signal=motor_noise)
        if fast
        else FullBodyDriverConfig(motor_noise_signal=motor_noise)
    )
    env = FullBodyDrivingEnv(driver=build_fullbody_driver(config))
    sampler = ScenarioSampler(
        tuple(stages),
        load_trials(trials_path) if trials_path else (),
        sampler_config or SamplerConfig(),
        stage,
    )
    return DrivingTaskWrapper(env, sampler, mode=mode, vehicle_spread=vehicle_spread)


def make_vec_env(
    n_envs: int, seed: int = 0, *, normalize: bool = True, **kwargs: Any
) -> VecEnv:
    """Monitored (and observation-normalized) vectorized training environments.

    Rewards are left unnormalized so a style reward added by a callback keeps
    its scale. ``kwargs`` go to :func:`make_training_env`.
    """
    factories: list[Callable[[], Any]] = [
        partial(make_training_env, seed + index, **kwargs) for index in range(n_envs)
    ]
    venv: VecEnv = (
        SubprocVecEnv(factories, start_method="spawn")
        if n_envs > 1
        else DummyVecEnv(factories)
    )
    venv = VecMonitor(venv)
    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    venv.seed(seed)
    return venv


class EpisodeLogCallback(BaseCallback):
    """Write each finished episode's metrics, success and rewards as JSON lines."""

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self._handle: Any = None

    def _on_training_start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_metrics" not in info:
                continue
            record = {
                "timesteps": int(self.num_timesteps),
                "success": bool(info["episode_success"]),
                "return": float(info.get("episode", {}).get("r", np.nan)),
                "style_log_odds": info.get("style_log_odds"),
                **info["episode_metrics"],
            }
            self._handle.write(json.dumps(record) + "\n")
            stage = record["stage"]
            self.logger.record_mean(
                f"episode/{stage}_success", float(record["success"])
            )
            self.logger.record_mean(f"episode/{stage}_lane_rms_m", record["lane_rms_m"])
        return True

    def _on_training_end(self) -> None:
        if self._handle is not None:
            self._handle.close()


class CurriculumCallback(BaseCallback):
    """Unlock the next stage once recent episodes succeed often enough.

    Args:
        stages: Number of stages in the sampler.
        window: Episodes in the success window.
        threshold: Success rate that unlocks the next stage.
        start: Initially unlocked stage.
    """

    def __init__(
        self, stages: int, window: int = 50, threshold: float = 0.8, start: int = 0
    ) -> None:
        super().__init__()
        self.stages = int(stages)
        self.window = int(window)
        self.threshold = float(threshold)
        self.stage = int(start)
        self.history: list[dict[str, Any]] = []
        self._recent: deque[bool] = deque(maxlen=self.window)

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode_success" in info:
                self._recent.append(bool(info["episode_success"]))
        rate = float(np.mean(self._recent)) if self._recent else 0.0
        self.logger.record("curriculum/stage", self.stage)
        self.logger.record("curriculum/success_rate", rate)
        if (
            len(self._recent) == self.window
            and rate >= self.threshold
            and self.stage < self.stages - 1
        ):
            self.stage += 1
            self.training_env.env_method("set_stage", self.stage)
            self.history.append(
                {
                    "timesteps": int(self.num_timesteps),
                    "stage": self.stage,
                    "rate": rate,
                }
            )
            self._recent.clear()
        return True


class StyleRewardCallback(BaseCallback):
    """Trial-level human-likeness reward from a refit discriminator.

    Whenever the environment reports a car-following window (or, without
    windows, a whole trial at the episode end) the policy is paid ``weight``
    times the discriminator's clipped log-odds that it is human, added to that
    step's reward before it is stored. Paying per window keeps the reward
    within the discount horizon. The discriminator is refit on the human
    training features (windows of the same length) against the latest
    ``bank_size`` agent windows every ``refit_every`` new ones.

    Args:
        human: Human training trial features (:data:`FOLLOWING_FEATURES`).
        weight: Reward per unit log-odds.
        refit_every: New agent trials between refits.
        bank_size: Agent trials kept for fitting.
        min_agent: Agent trials needed before the first fit.
        key: Info entry holding the trial: ``"following_window"`` or
            ``"following_trial"``.
    """

    def __init__(
        self,
        human: NDArray[np.float64],
        weight: float = 1.0,
        refit_every: int = 25,
        bank_size: int = 400,
        min_agent: int = 25,
        key: str = "following_window",
    ) -> None:
        super().__init__()
        self.key = key
        self.human = np.asarray(human, dtype=float)
        self.weight = float(weight)
        self.refit_every = int(refit_every)
        self.min_agent = int(min_agent)
        self.discriminator = StyleDiscriminator()
        self.bank: deque[NDArray[np.float64]] = deque(maxlen=int(bank_size))
        self._new = 0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        for index, info in enumerate(self.locals["infos"]):
            trial = info.get(self.key)
            if trial is None:
                continue
            features = following_features(trial)
            if self.discriminator.fitted:
                evidence = float(self.discriminator.log_odds(features)[0])
                rewards[index] += self.weight * evidence
                info["style_log_odds"] = evidence
                self.logger.record_mean("style/log_odds", evidence)
            self.bank.append(features)
            self._new += 1
        if len(self.bank) >= self.min_agent and (
            not self.discriminator.fitted or self._new >= self.refit_every
        ):
            self.discriminator.fit(self.human, np.vstack(self.bank))
            self._new = 0
        return True


def collect_expert(
    venv: VecEnv,
    steps: int,
    *,
    policy: BaseAlgorithm | None = None,
    expert_probability: float = 1.0,
    seed: int = 0,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Roll out and label every observation with the reflex driver's action.

    With ``policy`` and ``expert_probability`` below one, each step's executed
    action is the learner's with that complement probability, as in DAgger;
    the label is always the expert's. Observations are as the policy sees
    them (normalized by ``VecNormalize`` if present).
    """
    rng = np.random.default_rng(seed)
    # Wrappers such as VecNormalize keep their own, empty, reset_infos.
    base = getattr(venv, "unwrapped", venv)
    observation = venv.reset()
    expert = np.asarray(
        [info["expert_action"] for info in base.reset_infos], dtype=np.float32
    )
    observations, labels = [], []
    for _ in range(max(1, steps // venv.num_envs)):
        observations.append(np.asarray(observation, dtype=np.float32))
        labels.append(expert)
        action = expert
        if policy is not None and rng.random() >= expert_probability:
            action, _ = policy.predict(observation, deterministic=True)
        observation, _, dones, infos = venv.step(action)
        expert = np.asarray([info["expert_action"] for info in infos], dtype=np.float32)
        for index, done in enumerate(dones):
            if done:
                # The reset observation is new; label it with the reset's expert.
                expert[index] = base.reset_infos[index]["expert_action"]
    return np.concatenate(observations), np.concatenate(labels)


def behaviour_clone(
    model: BaseAlgorithm,
    observations: NDArray[np.float32],
    actions: NDArray[np.float32],
    *,
    epochs: int = 20,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Fit the policy's deterministic action to expert actions (mean squared error).

    Works for SAC/TD3 actors and for PPO/A2C Gaussian policies; returns the
    loss per epoch.
    """
    torch.manual_seed(seed)
    policy = model.policy
    network = getattr(policy, "actor", policy)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    inputs = torch.as_tensor(observations, dtype=torch.float32, device=policy.device)
    targets = torch.as_tensor(actions, dtype=torch.float32, device=policy.device)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(inputs), generator=generator)
        total = 0.0
        for start in range(0, len(inputs), batch_size):
            batch = order[start : start + batch_size]
            if hasattr(policy, "actor"):
                predicted = policy.actor(inputs[batch], deterministic=True)
            else:
                predicted = policy.get_distribution(inputs[batch]).mode()
            loss = torch.mean((predicted - targets[batch]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        losses.append(total / len(inputs))
    return losses

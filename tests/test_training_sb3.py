"""Stable-Baselines3 callbacks, behaviour cloning and expert collection."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sklearn")

from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.logger import configure  # noqa: E402

from myosuite_carla.human_comparison import FOLLOWING_FEATURES  # noqa: E402
from myosuite_carla.ngsim import FollowingTrial  # noqa: E402
from myosuite_carla.training_sb3 import (  # noqa: E402
    CurriculumCallback,
    StyleRewardCallback,
    behaviour_clone,
    collect_expert,
    make_vec_env,
)


def _trial(gap: float) -> FollowingTrial:
    time = np.arange(201) / 10.0
    return FollowingTrial(
        driver="sim",
        time=time,
        position=8.0 * time,
        speed=np.full_like(time, 8.0),
        acceleration=0.3 * np.sin(time),
        gap=np.full_like(time, gap),
        leader_speed=np.full_like(time, 8.0),
    )


def test_curriculum_unlocks_stages_on_recent_success() -> None:
    model = PPO("MlpPolicy", "Pendulum-v1", n_steps=64, batch_size=64)
    model.set_logger(configure(None, []))
    callback = CurriculumCallback(stages=3, window=4, threshold=0.75)
    callback.init_callback(model)
    calls: list[tuple[str, int]] = []
    model.env = SimpleNamespace(
        env_method=lambda name, value: calls.append((name, value))
    )
    for outcome in (True, False, True, True, True, True, True, True, True):
        callback.locals = {"infos": [{"episode_success": outcome}]}
        callback.on_step()
    assert calls == [("set_stage", 1), ("set_stage", 2)]
    assert callback.stage == 2 and len(callback.history) == 2


def test_style_reward_is_paid_once_the_discriminator_is_fit() -> None:
    model = PPO("MlpPolicy", "Pendulum-v1", n_steps=64, batch_size=64)
    model.set_logger(configure(None, []))
    rng = np.random.default_rng(0)
    # Human trials keep short gaps; the agent's long gaps look unhuman.
    from myosuite_carla.human_comparison import following_features

    human = np.vstack([following_features(_trial(g)) for g in rng.uniform(5, 8, 60)])
    assert human.shape[1] == len(FOLLOWING_FEATURES)
    callback = StyleRewardCallback(human, weight=0.5, refit_every=5, min_agent=10)
    callback.init_callback(model)
    paid = []
    for index in range(20):
        rewards = np.zeros(2)
        infos: list[dict[str, Any]] = [{"following_window": _trial(30.0 + index)}, {}]
        callback.locals = {"rewards": rewards, "infos": infos}
        callback.on_step()
        paid.append(rewards.copy())
    assert np.all(np.asarray(paid)[:10] == 0.0)
    assert np.all(np.asarray(paid)[10:, 0] < 0.0) and np.all(
        np.asarray(paid)[:, 1] == 0
    )
    assert callback.discriminator.fitted and len(callback.bank) == 20


@pytest.mark.parametrize("algorithm", [PPO, SAC, TD3])
def test_behaviour_cloning_fits_an_expert(algorithm: Any) -> None:
    model = algorithm("MlpPolicy", "Pendulum-v1", seed=0)
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(2000, 3)).astype(np.float32)
    actions = np.tanh(observations @ np.array([[0.8], [-0.5], [0.3]])).astype(
        np.float32
    )
    losses = behaviour_clone(model, observations, actions, epochs=30, seed=0)
    assert losses[-1] < 0.25 * losses[0] and losses[-1] < 0.02


def test_expert_collection_on_the_full_body_environment() -> None:
    pytest.importorskip("myosuite.core.model_builder")
    pytest.importorskip("myo_sim")
    venv = make_vec_env(1, seed=0, mode="direct", vehicle_spread=0.0, style_window=3.0)
    try:
        assert venv.get_attr("style_window") == [3.0]
        observations, labels = collect_expert(venv, 30)
        assert observations.shape == (30, venv.observation_space.shape[0])
        assert labels.shape == (30, 3) and np.all(np.abs(labels) <= 1.0)
        model = PPO("MlpPolicy", venv, n_steps=32, batch_size=32, seed=0)
        mixed, _ = collect_expert(venv, 20, policy=model, expert_probability=0.5)
        assert mixed.shape[0] == 20
    finally:
        venv.close()

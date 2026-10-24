"""The unregistered full-body driving task on the fitted vehicle surrogate."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from myosuite_carla import DriveRecorder, FullBodyDriverConfig


@pytest.fixture(scope="module")
def driver() -> Any:
    pytest.importorskip("myosuite.core.model_builder")
    myo_sim = pytest.importorskip("myo_sim")
    try:
        myo_sim.load_spec("myofullbody")
    except Exception as error:  # pragma: no cover - depends on myo-sim release
        pytest.skip(f"myo_sim has no myofullbody: {error}")
    from myosuite_carla import build_fullbody_driver

    return build_fullbody_driver(FullBodyDriverConfig.fast())


def _episode(env: Any) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    from myosuite_carla.driver_models import ReflexDriver

    policy = ReflexDriver(env.dt)
    env.reset(seed=0)
    policy.reset()
    infos = []
    while True:
        _, reward, terminated, truncated, info = env.step(policy.act(env.state))
        assert math.isfinite(reward)
        infos.append(info)
        if terminated or truncated:
            return infos, info, terminated


def test_spaces_names_and_reset_contract(driver: Any) -> None:
    from myosuite_carla.driving_env import FullBodyDrivingEnv

    env = FullBodyDrivingEnv(driver=driver)
    with pytest.raises(gym.error.ResetNeeded):
        env.step(np.zeros(6))
    observation, info = env.reset(seed=3)
    assert observation.shape == env.observation_space.shape
    assert len(env.observation_names) == observation.shape[0]
    assert len(set(env.observation_names)) == len(env.observation_names)
    assert env.action_space.shape == (6,)
    assert info["events"][0]["type"] == "scenario_start"
    observation, reward, terminated, truncated, info = env.step(np.zeros(6))
    assert np.all(np.isfinite(observation))
    assert set(info["reward_terms"]) >= {"lane", "speed", "effort", "collision"}
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    with pytest.raises(ValueError):
        env.step(np.zeros(5))
    # Out-of-range actions are clipped to the motor program's limits.
    assert env.intent([2.0, -1.0, 0.9, 0, 0, 0]).steer == 0.5


def test_reflex_driver_stops_behind_a_braking_lead(driver: Any) -> None:
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.scenario import lead_vehicle_braking

    recorder = DriveRecorder(body=False)
    env = FullBodyDrivingEnv(lead_vehicle_braking(), driver=driver, recorder=recorder)
    infos, last, terminated = _episode(env)
    assert not terminated
    gaps = [info["scenario"]["gap_m"] for info in infos]
    assert min(gaps) > 1.0
    assert last["scenario"]["speed_mps"] < 0.2
    # The left foot brakes; IDM asks for comfortable decelerations.
    assert max(info["inputs"]["brake"] for info in infos) > 0.1
    recording = recorder.recording()
    assert len(recording) == len(infos)
    assert [event["type"] for event in recorder.metadata["events"]][:2] == [
        "scenario_start",
        "lead_brake",
    ]


def test_reflex_driver_takes_over_after_its_takeover_time(driver: Any) -> None:
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.scenario import takeover_request

    env = FullBodyDrivingEnv(takeover_request(), driver=driver)
    infos, last, terminated = _episode(env)
    events = [event for info in infos for event in info["events"]]
    takeover = next(event for event in events if event["type"] == "driver_takeover")
    assert 2.72 <= takeover["reaction_time"] <= 3.5
    automated = [info for info in infos if info["automation"]]
    # The automation's command, not the resting wheel, drove until the takeover.
    assert any(
        info["command"]["throttle"] != info["inputs"]["throttle"] for info in automated
    )
    assert not terminated and not last["automation"]
    assert abs(last["scenario"]["lane_offset_m"]) < 0.5
    assert abs(last["scenario"]["speed_mps"] - 12.0) < 2.0


def test_reflex_driver_completes_a_lane_change(driver: Any) -> None:
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.scenario import lane_change

    env = FullBodyDrivingEnv(lane_change(), driver=driver)
    infos, last, terminated = _episode(env)
    assert not terminated
    assert last["scenario"]["target_offset_m"] == pytest.approx(-3.5)
    assert abs(last["scenario"]["lane_offset_m"]) < 0.5
    assert not any(info["inputs"]["hand_brake"] for info in infos)

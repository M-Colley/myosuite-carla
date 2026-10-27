"""Curriculum scenarios, residual and direct action modes, and episode reports."""

from __future__ import annotations

import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from myosuite_carla import FullBodyDriverConfig
from myosuite_carla.ngsim import FollowingTrial
from myosuite_carla.scenario import LeadVehicle, ScenarioMonitor, lead_vehicle_braking
from myosuite_carla.training import (
    MODES,
    STAGES,
    ScenarioSampler,
    curved_lane,
    perturb_vehicle,
    scenario_from_trial,
)
from myosuite_carla.vehicle_model import carla_nissan_patrol


@pytest.fixture(scope="module")
def driver() -> Any:
    pytest.importorskip("myosuite.core.model_builder")
    pytest.importorskip("myo_sim")
    from myosuite_carla import build_fullbody_driver

    return build_fullbody_driver(FullBodyDriverConfig.fast())


def _trial(seconds: float = 20.0) -> FollowingTrial:
    time = np.arange(int(seconds * 10) + 1) / 10.0
    leader = 8.0 + np.sin(time / 3.0)
    return FollowingTrial(
        driver="7:1",
        time=time,
        position=8.0 * time,
        speed=np.full_like(time, 8.0),
        acceleration=np.zeros_like(time),
        gap=np.full_like(time, 12.0),
        leader_speed=leader,
    )


def test_lead_profile_replays_recorded_speeds() -> None:
    trial = _trial()
    scenario = scenario_from_trial(trial)
    assert scenario.start_speed == 8.0 and scenario.lead.gap == 12.0
    monitor = ScenarioMonitor(scenario)
    from types import SimpleNamespace

    ego = SimpleNamespace(
        location=(5.0, 0.0, 0.0), rotation=(0, 0, 0), velocity=(8, 0, 0)
    )
    monitor.reset(ego)
    speeds = [monitor.update(ego, 0.1)["lead_speed_mps"] for _ in range(100)]
    np.testing.assert_allclose(
        speeds, np.interp(trial.time[1:101], trial.time, trial.leader_speed)
    )
    with pytest.raises(ValueError):
        LeadVehicle(profile=((0.0, 1.0), (0.0, 2.0)))
    with pytest.raises(ValueError):
        LeadVehicle(profile=((0.0, -1.0),))


def test_sampler_draws_every_stage_and_respects_the_curriculum() -> None:
    rng = np.random.default_rng(0)
    sampler = ScenarioSampler(STAGES, (_trial(),))
    assert {sampler.sample(rng).name for _ in range(20)} == {"free"}
    sampler.stage = len(STAGES) - 1
    drawn = {sampler.sample(rng).name for _ in range(200)}
    assert drawn == set(STAGES)
    follow = sampler.sample(rng, "follow")
    assert follow.lead.profile and follow.duration == pytest.approx(20.0)
    assert sampler.sample(rng, "takeover").takeover is not None
    assert sampler.sample(rng, "lane_change").lane_changes
    lane = curved_lane(rng, 400.0)
    assert lane.length >= 400.0
    with pytest.raises(ValueError):
        ScenarioSampler(("free", "unknown"))


def test_vehicle_perturbation_stays_within_its_spread() -> None:
    base = carla_nissan_patrol()
    rng = np.random.default_rng(1)
    for _ in range(20):
        params = perturb_vehicle(base, rng, 0.1)
        ratio = params.brake_decel / base.brake_decel
        assert 0.9 <= ratio <= 1.1
        assert np.all(np.asarray(params.steer_gain) <= 1.4)
    assert perturb_vehicle(base, rng, 0.0) is base


def _episode(env: Any, action: Any) -> tuple[list[tuple[float, float]], dict]:
    env.reset(seed=0)
    out = []
    while True:
        _, _, terminated, truncated, info = env.step(action)
        out.append((info["scenario"]["speed_mps"], info["scenario"]["gap_m"]))
        if terminated or truncated:
            return out, info


def test_zero_residual_reproduces_the_reflex_driver(driver: Any) -> None:
    from myosuite_carla.driver_models import ReflexDriver
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.training import DrivingTaskWrapper

    scenario = lead_vehicle_braking(duration=6.0, lead=LeadVehicle(brake_time=2.0))
    env = FullBodyDrivingEnv(scenario, driver=driver)
    policy = ReflexDriver(env.dt)
    env.reset(seed=0)
    policy.reset()
    reference = []
    while True:
        _, _, terminated, truncated, info = env.step(policy.act(env.state))
        reference.append((info["scenario"]["speed_mps"], info["scenario"]["gap_m"]))
        if terminated or truncated:
            break
    for mode in MODES[:2]:
        wrapper = DrivingTaskWrapper(
            FullBodyDrivingEnv(scenario, driver=driver),
            mode=mode,
            vehicle_spread=0.0,
            takeover_population=False,
        )
        assert wrapper.observation_space.shape[0] == 73 + 2 * len(wrapper.fields)
        trace, info = _episode(wrapper, np.zeros(wrapper.action_space.shape))
        assert trace == reference
        assert info["episode_success"] and "following_trial" in info
        assert len(info["following_trial"].time) == 61
        # The 6 s follow ends its second 5 s style window only at 10 s.
        assert "following_window" not in info


def test_direct_policy_evaluation_uses_the_reflex_driver(driver: Any) -> None:
    from myosuite_carla.driver_models import ReflexDriver
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.scenario import Scenario
    from myosuite_carla.training import DrivingTaskWrapper

    namespace = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "evaluate_policy.py")
    )
    base = FullBodyDrivingEnv(Scenario(duration=1.0), driver=driver)
    direct = DrivingTaskWrapper(base, mode="direct", vehicle_spread=0.0)
    baseline = namespace["_reflex_environment"](direct)
    assert baseline.unwrapped is base
    baseline.reset(seed=0)
    reference = ReflexDriver(base.dt)
    expected = reference.act(base.state)
    actual = baseline.intent_for(np.zeros(baseline.action_space.shape))
    assert actual == expected
    assert actual.left_foot == 0.0


def test_residual_waits_for_the_takeover_and_direct_mode_maps_bounds(
    driver: Any,
) -> None:
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.scenario import TakeoverRequest, takeover_request
    from myosuite_carla.training import DrivingTaskWrapper

    scenario = takeover_request(duration=6.0, takeover=TakeoverRequest(time=1.0))
    wrapper = DrivingTaskWrapper(
        FullBodyDrivingEnv(scenario, driver=driver),
        vehicle_spread=0.0,
        takeover_population=False,
    )
    wrapper.reset(seed=0)
    before = None
    for _ in range(100):
        # Before the driver re-engages, even a full brake residual is ignored.
        _, _, _, _, info = wrapper.step(np.array([-1.0, 0.0]))
        if before is None and info["automation"] and info["intent"]["left_foot"] == 0:
            before = info
    assert before is not None
    direct = DrivingTaskWrapper(
        FullBodyDrivingEnv(scenario, driver=driver), mode="direct", vehicle_spread=0.0
    )
    direct.reset(seed=0)
    intent = direct.intent_for(np.array([-1.0, 1.0, 0.0]))
    assert intent.right_foot == 0.0 and intent.left_foot == 1.0 and intent.steer == 0.0
    assert np.all(np.abs(direct.expert_action()) <= 1.0)
    with pytest.raises(ValueError):
        DrivingTaskWrapper(FullBodyDrivingEnv(scenario, driver=driver), mode="nope")
    assert math.isfinite(before["reward_terms"]["smoothness"])


def test_speed_reward_follows_the_headway_behind_a_lead(driver: Any) -> None:
    from myosuite_carla.driving_env import FullBodyDrivingEnv, TaskConfig
    from myosuite_carla.training import DrivingTaskWrapper

    scenario = lead_vehicle_braking(duration=12.0)
    env = DrivingTaskWrapper(
        FullBodyDrivingEnv(scenario, driver=driver, task=TaskConfig()),
        vehicle_spread=0.0,
        takeover_population=False,
    )
    env.reset(seed=0)
    windows = bound = 0
    while True:
        _, _, terminated, truncated, info = env.step(np.zeros(2))
        windows += "following_window" in info
        state, target = info["scenario"], info["scenario"]["target_speed_mps"]
        # The desired speed is the headway-limited one, not the free-road target.
        desired = min(target, max((state["gap_m"] - 2.0) / 1.6, 0.0))
        bound += desired < target
        expected = -(((state["speed_mps"] - desired) / target) ** 2)
        assert info["reward_terms"]["speed"] == pytest.approx(expected)
        if terminated or truncated:
            break
    assert windows == 2 and bound > 0

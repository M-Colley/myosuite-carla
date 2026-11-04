"""The motor interface tracks physical acceleration demands continuously."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from myosuite_carla.driver_models import (
    AccelerationControlConfig,
    ReflexDriver,
    ReflexDriverConfig,
    acceleration_pedals,
    idm_acceleration,
)
from myosuite_carla.vehicle_model import carla_nissan_patrol


def _state(speed=10.0):
    return {
        "speed_mps": speed,
        "target_speed_mps": 16.0,
        "near_angle_rad": 0.0,
        "far_angle_rad": 0.0,
        "lead_present": 0.0,
        "input_throttle": 0.0,
        "input_brake": 0.0,
    }


def _act(driver, state, acceleration):
    desired = idm_acceleration(
        state["speed_mps"], state["target_speed_mps"], None, 0, driver.config.idm
    )
    return driver.act(state, acceleration_offset=acceleration - desired)


@pytest.mark.parametrize("speed", [0.5, 2.0, 5.0, 10.0, 16.0, 20.0])
@pytest.mark.parametrize("acceleration", [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0])
def test_inverse_reproduces_feasible_vehicle_acceleration(speed, acceleration):
    config = AccelerationControlConfig()
    throttle, brake = acceleration_pedals(acceleration, speed, config)
    p = config.vehicle
    actual = (
        p.drive_acceleration(throttle, speed, False)
        - p.rolling
        - p.linear_drag * speed
        - p.quadratic_drag * speed**2
        - p.brake_decel * brake
    )
    assert actual == pytest.approx(acceleration, abs=1e-12)
    assert throttle == 0 or brake == 0
    assert 0 <= throttle <= 1 and 0 <= brake <= 1


def test_pedal_targets_are_continuous_at_coast_and_previous_branch_boundary():
    config = AccelerationControlConfig()
    p, speed = config.vehicle, 10.0
    coast = -p.rolling - p.linear_drag * speed - p.quadratic_drag * speed**2
    for boundary in (coast, -0.4):
        before = np.asarray(acceleration_pedals(boundary - 1e-8, speed, config))
        after = np.asarray(acceleration_pedals(boundary + 1e-8, speed, config))
        assert np.max(np.abs(after - before)) < 1e-7


def test_inverse_saturates_and_releases_throttle_at_standstill():
    config = AccelerationControlConfig()
    assert acceleration_pedals(1000, 5, config) == (1.0, 0.0)
    assert acceleration_pedals(-1000, 5, config) == (0.0, 1.0)
    assert acceleration_pedals(0, 0, config) == (0.0, 0.0)
    assert acceleration_pedals(-1, 0, config)[0] == 0
    for acceleration, speed in ((float("nan"), 0), (0, float("inf")), (0, -1)):
        with pytest.raises(ValueError):
            acceleration_pedals(acceleration, speed, config)


def test_new_controller_closes_brake_feedback_and_resets_inactive_integrals():
    config = ReflexDriverConfig(
        perception_delay=0.02, acceleration_control=AccelerationControlConfig()
    )
    driver = ReflexDriver(0.02, config)
    state = _state()
    first = _act(driver, state, -3)
    for _ in range(20):
        last = _act(driver, state, -3)
    assert last.left_foot > first.left_foot
    assert first.right_foot == last.right_foot == 0
    assert driver._brake_integral > 0
    forward = _act(driver, state, 1)
    assert forward.left_foot == 0 and forward.right_foot > 0
    assert driver._brake_integral == 0
    idle = driver.act(dict(state, automation=1.0, takeover_requested=0.0))
    assert idle.left_foot == idle.right_foot == 0
    assert driver._throttle_integral == driver._brake_integral == 0
    driver.reset()
    assert driver.target_pedals == (0, 0)
    assert driver._throttle_integral == driver._brake_integral == 0


def test_feedback_is_bounded_and_rejects_reverse_or_nonfinite_measurements():
    config = ReflexDriverConfig(
        perception_delay=0.02, acceleration_control=AccelerationControlConfig()
    )
    driver = ReflexDriver(0.02, config)
    state = _state()
    for _ in range(200):
        intent = _act(driver, state, -3)
    assert 0 <= intent.left_foot <= 1
    assert abs(driver._brake_integral) <= config.acceleration_control.integral_limit
    saturated = ReflexDriver(
        1.0,
        replace(
            config,
            acceleration_control=AccelerationControlConfig(
                pedal_integral=1.0, integral_limit=1.0
            ),
        ),
    )
    for _ in range(10):
        intent = _act(saturated, state, -100)
    assert intent.left_foot <= 1 and saturated._brake_integral < 1
    with pytest.raises(ValueError, match="forward"):
        _act(driver, dict(state, input_reverse=1), 0)
    with pytest.raises(ValueError, match="finite"):
        _act(driver, dict(state, input_brake=float("nan")), -3)


def test_configuration_roundtrip_preserves_typed_vehicle_and_controller():
    config = ReflexDriverConfig(acceleration_control=AccelerationControlConfig())
    restored = ReflexDriverConfig.from_dict(json.loads(json.dumps(asdict(config))))
    assert restored == config
    assert isinstance(restored.acceleration_control, AccelerationControlConfig)
    assert isinstance(restored.throttle_map, tuple)
    legacy = ReflexDriverConfig.from_dict({"idm": {"time_headway": 2}})
    assert legacy.acceleration_control is None and legacy.idm.time_headway == 2


@pytest.mark.parametrize(
    "change",
    [
        {"pedal_integral": -1},
        {"integral_limit": 2},
        {"integral_limit": float("nan")},
        {"pedal_integral": True},
    ],
)
def test_invalid_servo_assumptions_are_rejected(change):
    with pytest.raises(ValueError):
        AccelerationControlConfig(**change)


def test_vehicle_inverse_rejects_nonmonotonic_throttle_and_absent_brakes():
    p = carla_nissan_patrol()
    with pytest.raises(ValueError, match="brake"):
        AccelerationControlConfig(vehicle=replace(p, brake_decel=0))
    with pytest.raises(ValueError, match="monotonic"):
        AccelerationControlConfig(vehicle=replace(p, drive=(p.drive[-1], *p.drive[1:])))


def test_legacy_control_trace_preserves_the_previous_pedal_and_steering_math():
    driver = ReflexDriver(0.02, ReflexDriverConfig(perception_delay=0.04))
    reference = {
        0: (0.013333629475469172, 0.0, -6e-05),
        7: (0.01365623201243196, 0.0, 0.04149072),
        12: (0.0, 0.15, 0.07082379770700636),
        23: (0.0, 0.15, 0.13352717551451188),
    }
    for index in range(24):
        state = _state(5 + index * 0.25)
        state.update(
            target_speed_mps=16 if index < 8 else 4,
            near_angle_rad=0.001 * index,
            far_angle_rad=0.002 * index,
            lead_present=1,
            gap_m=20 if index < 12 else 5,
            closing_speed_mps=0 if index < 12 else 3,
            input_throttle=0.2,
            input_brake=0.1,
            input_steer=0.01,
        )
        intent = driver.act(state)
        if index in reference:
            assert (intent.right_foot, intent.left_foot, intent.steer) == pytest.approx(
                reference[index], abs=1e-15, rel=0
            )


def test_actual_muscle_vehicle_loop_holds_zero_acceleration_more_accurately():
    pytest.importorskip("myosuite.core.model_builder")
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.fullbody import FullBodyDriverConfig, build_fullbody_driver
    from myosuite_carla.scenario import Scenario

    body = build_fullbody_driver(FullBodyDriverConfig.fast())
    errors = []
    for control in (None, AccelerationControlConfig()):
        env = FullBodyDrivingEnv(
            Scenario(name="acceleration_tracking", start_speed=10, duration=4),
            driver=body,
        )
        driver = ReflexDriver(
            env.dt,
            ReflexDriverConfig(perception_delay=env.dt, acceleration_control=control),
        )
        try:
            env.reset(seed=0)
            samples = []
            for index in range(200):
                _, _, terminated, _, _ = env.step(_act(driver, env.state, 0.0))
                assert not terminated
                if index >= 50:
                    samples.append(abs(env.state["a_forward_mps2"]))
            errors.append(float(np.mean(samples)))
        finally:
            env.close()
            env.vehicle.close()
    assert errors[1] < errors[0] * 0.5, errors
    assert errors[1] < 0.15, errors

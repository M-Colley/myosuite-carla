"""Muscle conditions, motor noise and posture variants of the full-body driver."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from myosuite_carla import DriverIntent, FullBodyDriverConfig


def _build(**overrides: Any) -> Any:
    pytest.importorskip("myosuite.core.muscle_conditions")
    myo_sim = pytest.importorskip("myo_sim")
    try:
        myo_sim.load_spec("myofullbody")
    except Exception as error:  # pragma: no cover - depends on myo-sim release
        pytest.skip(f"myo_sim has no myofullbody: {error}")
    from myosuite_carla import build_fullbody_driver

    return build_fullbody_driver(FullBodyDriverConfig.fast(**overrides))


@pytest.fixture(scope="module")
def healthy() -> Any:
    return _build()


def _hold(driver: Any, intent: DriverIntent, periods: int = 75) -> dict[str, float]:
    driver.reset(seed=0)
    for _ in range(periods):
        driver.advance(intent)
        values = driver.mapper.sample(driver.data).values
    return values


def _peak(driver: Any, name: str) -> float:
    return float(driver.model.actuator_gainprm[driver.model.actuator(name).id, 2])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"strength": 0.0}, "strength"),
        ({"muscle_strength": (("*_l", 3.0),)}, "muscle_strength"),
        ({"fatigue_sex": "X"}, "fatigue_sex"),
        ({"initial_fatigue": 1.0}, "initial_fatigue"),
        ({"motor_noise_signal": -0.1}, "motor_noise_signal"),
        ({"posture": (("knee_angle_r", float("nan")),)}, "posture"),
    ],
)
def test_condition_settings_are_validated(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FullBodyDriverConfig(**overrides)


def test_weak_left_side_scales_forces_and_weakens_the_brake_foot(healthy: Any) -> None:
    weak = _build(muscle_strength=(("*_l", 0.5),))
    for name in ("soleus_l", "BIClong_l", "FDP2_l"):
        assert _peak(weak, name) == pytest.approx(0.5 * _peak(healthy, name))
    for name in ("soleus_r", "BIClong", "FDP2"):
        assert _peak(weak, name) == pytest.approx(_peak(healthy, name))
    braking = DriverIntent(left_foot=0.04)
    assert _hold(weak, braking)["brake"] < _hold(healthy, braking)["brake"] - 0.05
    with pytest.raises(ValueError, match="matches no muscle"):
        _build(muscle_strength=(("no_such_muscle*", 0.5),))


def test_sarcopenia_halves_every_peak_force(healthy: Any) -> None:
    frail = _build(strength=0.5)
    np.testing.assert_allclose(
        frail.model.actuator_gainprm[:, 2], 0.5 * healthy.model.actuator_gainprm[:, 2]
    )
    # The cockpit is fitted to the weaker body at rest, so nothing is pressed.
    assert all(value == 0.0 for value in _hold(frail, DriverIntent(), 50).values())


def test_pre_fatigued_muscles_press_less(healthy: Any) -> None:
    tired = _build(fatigue=True, initial_fatigue=0.98)
    throttle = DriverIntent(right_foot=0.03)
    assert (
        _hold(tired, throttle)["throttle"] < _hold(healthy, throttle)["throttle"] - 0.1
    )
    assert tired.measurement()["driving_fatigued_fraction"] == pytest.approx(
        0.98, abs=0.01
    )
    assert "driving_fatigued_fraction" not in healthy.measurement()


def test_motor_noise_is_seeded_and_perturbs_steering(healthy: Any) -> None:
    noisy = _build(motor_noise_signal=0.2)

    def trace(driver: Any, seed: int) -> np.ndarray:
        driver.reset(seed=seed)
        out = []
        for _ in range(60):
            driver.advance(DriverIntent(steer=0.05))
            out.append(driver.mapper.sample(driver.data).values["steer"])
        return np.asarray(out)

    first, again, other = trace(noisy, 1), trace(noisy, 1), trace(noisy, 2)
    np.testing.assert_array_equal(first, again)
    assert np.max(np.abs(first - other)) > 1e-3
    np.testing.assert_array_equal(trace(healthy, 1), trace(healthy, 2))


def test_posture_override_refits_the_cockpit() -> None:
    reclined = _build(posture=(("hip_flexion_r", 1.2), ("hip_flexion_l", 1.2)))
    model = reclined.model
    for side in "rl":
        address = model.jnt_qposadr[model.joint(f"hip_flexion_{side}").id]
        # Posture locks are soft equality constraints that yield slightly.
        assert reclined.rest_qpos[address] == pytest.approx(1.2, abs=5e-3)
    for key in ("throttle", "brake"):
        assert reclined.initial_gaps[key] == pytest.approx(
            reclined.config.clearance, abs=3e-4
        )
    assert _hold(reclined, DriverIntent(right_foot=0.03))["throttle"] > 0.3
    with pytest.raises(ValueError, match="unknown joints"):
        _build(posture=(("no_such_joint", 0.1),))

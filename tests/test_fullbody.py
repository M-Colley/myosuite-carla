"""Offline checks of the seated MyoSuite full-body driver and its scenario."""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from myosuite_carla import (
    CarlaBridge,
    DriverIntent,
    FullBodyDriverConfig,
    SequentialShifter,
    SessionConfig,
)


@pytest.fixture(scope="module")
def driver() -> Any:
    pytest.importorskip("myosuite.core.model_builder")
    myo_sim = pytest.importorskip("myo_sim")
    try:
        myo_sim.load_spec("myofullbody")
    except Exception as error:  # pragma: no cover - depends on myo-sim release
        pytest.skip(f"myo_sim has no myofullbody: {error}")
    from myosuite_carla import build_fullbody_driver

    return build_fullbody_driver()


def drive(driver: Any, intent: DriverIntent, periods: int) -> tuple[dict, dict]:
    worst: dict[str, float] = {}
    for _ in range(periods):
        driver.advance(intent)
        values = driver.mapper.sample(driver.data).values
        for key, value in values.items():
            worst[key] = max(worst.get(key, 0.0), abs(value))
    return driver.mapper.sample(driver.data).values, worst


def test_importing_the_package_loads_neither_carla_nor_myosuite() -> None:
    code = (
        "import sys, myosuite_carla; "
        "print(any(name in ('carla', 'myosuite', 'myo_sim') or name.startswith("
        "('carla.', 'myosuite.', 'myo_sim.')) for name in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_one_full_body_model_contains_every_mechanism(driver: Any) -> None:
    import mujoco

    model = driver.model
    muscles = np.count_nonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
    assert muscles == model.nu == 416
    for name in (
        "cockpit_throttle_joint",
        "cockpit_brake_joint",
        "cockpit_steering_joint",
        "cradle_r_spin",
        "cradle_l_spin",
        "gear_button_joint",
        "reverse_button_joint",
        "park_button_joint",
    ):
        assert model.joint(name).id >= 0
    gaps = driver.initial_gaps
    for key in ("throttle", "brake", "cradle_r_lower", "cradle_l_lower"):
        assert gaps[key] == pytest.approx(driver.config.clearance, abs=3e-4)
    for key in ("cradle_r_upper", "cradle_l_upper"):
        assert 0.0 < gaps[key] < 2e-3
    for key in ("gear", "reverse", "park"):
        assert gaps[key] > 1e-3


def test_rest_pose_releases_every_control(driver: Any) -> None:
    driver.reset()
    values, worst = drive(driver, DriverIntent(), 50)
    assert all(value == 0.0 for value in worst.values())
    assert values["gear"] == values["manual_gear_shift"] == 0.0
    assert not np.any(driver.data.qfrc_applied)
    assert not np.any(driver.data.xfrc_applied)


@pytest.mark.parametrize(
    "field, channel, other",
    [("right_foot", "throttle", "brake"), ("left_foot", "brake", "throttle")],
)
def test_each_foot_presses_only_its_pedal(
    driver: Any, field: str, channel: str, other: str
) -> None:
    driver.reset()
    drive(driver, DriverIntent(), 10)
    held, worst = drive(driver, DriverIntent(**{field: 0.03}), 50)
    measurement = driver.measurement()
    assert held[channel] > 0.4
    assert worst[other] == 0.0
    assert measurement[f"{channel}_contact_newtons"] > 5.0
    released, _ = drive(driver, DriverIntent(), 50)
    assert released[channel] == 0.0


def test_push_pull_hands_turn_the_wheel_both_ways(driver: Any) -> None:
    for offset in (-0.14, 0.14):
        driver.reset()
        held, worst = drive(driver, DriverIntent(steer=offset), 60)
        assert np.sign(held["steer"]) == np.sign(offset)
        assert abs(held["steer"]) > 0.25
        assert driver.measurement()["steering_contact_newtons"] > 1.0
        assert worst["hand_brake"] == worst["reverse"] == 0.0
        assert worst["throttle"] == worst["brake"] == 0.0
        released, _ = drive(driver, DriverIntent(), 50)
        assert abs(released["steer"]) < 0.02


@pytest.mark.parametrize(
    "field, channel", [("reverse_finger", "reverse"), ("park_finger", "hand_brake")]
)
def test_finger_buttons_hold_and_release(driver: Any, field: str, channel: str) -> None:
    driver.reset()
    drive(driver, DriverIntent(), 10)
    held = []
    for _ in range(150):
        values, _ = drive(driver, DriverIntent(**{field: 0.05}), 1)
        held.append(values[channel] == 1.0)
    # The digit reaches its button within 0.5 s and then holds it steadily.
    assert all(held[25:])
    released, _ = drive(driver, DriverIntent(), 25)
    assert released[channel] == 0.0


def test_paddle_pulls_cycle_manual_gears(driver: Any) -> None:
    driver.reset()
    drive(driver, DriverIntent(), 5)
    gears = []
    for _ in range(4):
        drive(driver, DriverIntent(gear_finger=0.06), 15)
        values, _ = drive(driver, DriverIntent(), 25)
        gears.append((values["gear"], values["manual_gear_shift"]))
    assert gears == [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (0.0, 0.0)]


def test_sequential_shifter_counts_each_pull_once() -> None:
    shifter = SequentialShifter("paddle", travel=0.01, max_gear=2)
    reading = {"value": 0.0}
    data = SimpleNamespace(sensor=lambda name: SimpleNamespace(data=[reading["value"]]))
    sequence = [0.0, 0.007, 0.009, 0.009, 0.004, 0.002, 0.008, 0.0, 0.009, 0.0]
    gears = []
    for value in sequence:
        reading["value"] = value
        gear = shifter.read_gear(None, data)
        assert shifter.read_gear(None, data) == gear
        gears.append((gear, shifter.read_manual(None, data)))
    assert [gear for gear, _ in gears] == [0, 1, 1, 1, 1, 1, 2, 2, 0, 0]
    assert gears[-1][1] == 0.0
    shifter.reset()
    assert shifter.gear == 0


@pytest.mark.parametrize(
    "kwargs",
    [{"right_foot": 1.5}, {"park_finger": -0.1}, {"steer": 0.6}, {"left_foot": np.nan}],
)
def test_driver_intent_rejects_invalid_excitation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DriverIntent(**kwargs)


def test_config_requires_whole_physics_steps() -> None:
    with pytest.raises(ValueError, match="whole number"):
        FullBodyDriverConfig(timestep=0.003)
    with pytest.raises(ValueError, match="positive"):
        FullBodyDriverConfig(clearance=0.0)


def test_advance_rejects_external_forces(driver: Any) -> None:
    driver.reset()
    driver.data.xfrc_applied[1, 0] = 1.0
    with pytest.raises(RuntimeError, match="External forces"):
        driver.advance(DriverIntent())
    driver.reset()
    with pytest.raises(TypeError):
        driver.advance({"right_foot": 0.1})


def test_bridge_sends_measured_full_body_controls(driver: Any, simulation: Any) -> None:
    session = simulation.session(SessionConfig(fixed_delta_seconds=0.02))
    session.start()
    bridge = CarlaBridge(driver.mapper, session, driver.config.control_dt)
    driver.reset()
    bridge.reset(driver.data)
    for _ in range(40):
        driver.advance(DriverIntent(right_foot=0.012))
        bridge.step(driver.data)
    pressed = simulation.vehicle.controls[-1]
    assert pressed.throttle > 0.3 and pressed.brake == 0.0
    assert not pressed.reverse and not pressed.hand_brake
    for _ in range(40):
        driver.advance(DriverIntent(reverse_finger=0.05, park_finger=0.05))
        bridge.step(driver.data)
    held = simulation.vehicle.controls[-1]
    assert held.reverse and held.hand_brake
    assert held.throttle == 0.0
    session.close()


def test_scenario_checks_follow_phase_outcomes() -> None:
    from myosuite_carla.fullbody_validation import _checks, fullbody_scenario

    scenario = fullbody_scenario()
    names = [phase.name for phase in scenario]
    assert len(names) == len(set(names))

    def phase(**overrides: Any) -> dict[str, Any]:
        summary = {
            "frames": 100,
            "start_speed_mps": 0.0,
            "end_speed_mps": 0.0,
            "signed_displacement_metres": 0.0,
            "yaw_change_degrees": 0.0,
            "max_command": {"throttle": 0.5, "brake": 0.0, "steer": 0.0},
            "min_steer": 0.0,
            "max_steer": 0.0,
            "applied_true_frames": {
                "hand_brake": 0,
                "reverse": 0,
                "manual_gear_shift": 0,
            },
            "applied_gears": [1],
            "end_applied": {
                "hand_brake": False,
                "reverse": False,
                "manual_gear_shift": False,
                "gear": 0,
            },
        }
        summary.update(overrides)
        return summary

    phases = {name: phase() for name in names}
    phases["throttle"] = phase(end_speed_mps=1.2, signed_displacement_metres=1.6)
    phases["handbrake_hold_throttle"] = phase(
        applied_true_frames={"hand_brake": 100, "reverse": 0, "manual_gear_shift": 0}
    )
    phases["steer_left"] = phase(yaw_change_degrees=-7.0, min_steer=-0.4)
    phases["steer_right"] = phase(yaw_change_degrees=6.0, max_steer=0.5)
    phases["brake"] = phase(
        end_speed_mps=0.01,
        max_command={"throttle": 0.0, "brake": 1.0, "steer": 0.0},
    )
    phases["reverse_throttle"] = phase(
        signed_displacement_metres=-1.2,
        applied_true_frames={"hand_brake": 0, "reverse": 100, "manual_gear_shift": 0},
    )
    for gear in (1, 2, 3):
        phases[f"gear_{gear}_throttle"] = phase(
            end_applied={
                "hand_brake": False,
                "reverse": False,
                "manual_gear_shift": True,
                "gear": gear,
            }
        )
    phases["handbrake_stop"] = phase(
        start_speed_mps=1.6,
        end_speed_mps=0.01,
        max_command={"throttle": 0.0, "brake": 0.0, "steer": 0.0},
        end_applied={
            "hand_brake": True,
            "reverse": False,
            "manual_gear_shift": False,
            "gear": 1,
        },
    )
    results = _checks(phases)
    assert len(results) == 12
    assert all(result["passed"] for result in results)
    phases["reverse_throttle"]["signed_displacement_metres"] = 0.3
    phases["handbrake_stop"]["max_command"]["brake"] = 0.4
    failed = {result["name"] for result in _checks(phases) if not result["passed"]}
    assert failed == {
        "finger-held reverse drives backwards",
        "parking brake stops the rolling car",
    }

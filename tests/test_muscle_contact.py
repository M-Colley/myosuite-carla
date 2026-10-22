"""Real MyoSuite muscle excitation reaches vehicle commands through contact."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from myosuite_carla.validation import build_muscle_driver


@dataclass(frozen=True)
class MuscleContactResult:
    command: float
    peak_travel: float
    peak_contact_force: float
    peak_muscle_force: float


def _run_muscle_contact(
    control: str,
    excitation: float,
    *,
    contacts_enabled: bool = True,
) -> MuscleContactResult:
    pytest.importorskip("myo_sim")
    pytest.importorskip("myosuite.core.model_builder")
    driver = build_muscle_driver(control, contacts_enabled=contacts_enabled)
    for _ in range(30):
        driver.advance(excitation)
    assert np.all(np.isfinite(driver.data.qpos))
    assert np.all(np.isfinite(driver.data.sensordata))
    channel = "steer" if control == "steering" else control
    return MuscleContactResult(
        command=getattr(driver.mapper.sample(driver.data).command, channel),
        peak_travel=driver.peaks["travel"],
        peak_contact_force=driver.peaks["contact_newtons"],
        peak_muscle_force=driver.peaks["muscle_newtons"],
    )


@pytest.mark.parametrize("control", ["throttle", "brake", "steering"])
def test_real_muscles_move_controls_only_through_physical_contact(control: str) -> None:
    idle = _run_muscle_contact(control, 0.0)
    active = _run_muscle_contact(control, 0.05)
    disconnected = _run_muscle_contact(control, 0.05, contacts_enabled=False)

    assert active.peak_muscle_force > 100
    assert active.peak_contact_force > 10
    assert abs(active.command) > 0.1
    assert abs(idle.command) < 0.01
    assert disconnected.peak_contact_force == 0.0
    assert disconnected.peak_travel == 0.0
    assert disconnected.command == 0.0
    if control == "steering":
        assert active.peak_travel > 0.25
        assert active.command < -0.1
        assert idle.peak_travel < 0.01
    else:
        assert active.peak_travel > 0.045
        assert active.command > 0.8
        assert idle.peak_travel < 0.01

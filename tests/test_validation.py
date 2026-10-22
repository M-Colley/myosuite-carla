"""Real MuJoCo checks for the fixtures shared with actual-server trials."""

from __future__ import annotations

import numpy as np
import pytest

from myosuite_carla.validation import build_muscle_driver


@pytest.mark.parametrize("control", ["throttle", "brake", "steering"])
def test_fixture_sustains_muscle_contact_for_vehicle_trial(control: str) -> None:
    driver = build_muscle_driver(control)
    channel = "steer" if control == "steering" else control
    for _ in range(150):
        driver.advance(0.05 if control == "steering" else 0.01)
    assert abs(getattr(driver.mapper.sample(driver.data).command, channel)) > 0.1
    assert driver.peaks["contact_newtons"] > 10
    assert not np.any(driver.data.qfrc_applied)
    assert not np.any(driver.data.xfrc_applied)


def test_validation_fixture_rejects_external_forces_and_invalid_excitation() -> None:
    driver = build_muscle_driver("throttle")
    with pytest.raises(ValueError, match="excitation"):
        driver.advance(float("nan"))
    driver.data.qfrc_applied[0] = 1
    with pytest.raises(RuntimeError, match="External forces"):
        driver.advance()

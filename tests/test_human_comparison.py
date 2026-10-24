"""Window features, adequacy checks and classifier discriminability."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("sklearn")

from myosuite_carla.human_comparison import (  # noqa: E402
    FEATURES,
    adequacy,
    discriminability,
    human_floor,
    summarize,
    window_features,
)


def _signals(
    frequency: float, amplitude: float, seconds: float = 60.0, rate: float = 50.0
):
    time = np.arange(0.0, seconds, 1.0 / rate)
    steer = 0.2 * np.sin(2 * math.pi * frequency * time)
    accel = amplitude * np.sin(2 * math.pi * 0.2 * time)
    return time, steer, accel


def test_features_recover_known_sinusoids() -> None:
    time, steer, accel = _signals(0.3, 0.5)
    rows = window_features(time, steer, accel, window=10.0)
    assert rows.shape == (6, len(FEATURES))
    frequency, reversals, accel_sd, jerk = np.median(rows, axis=0)
    assert frequency == pytest.approx(0.3, rel=0.05)
    # Two reversals per cycle: 0.3 Hz is 36 per minute.
    assert reversals == pytest.approx(36.0, abs=6.0)
    assert accel_sd == pytest.approx(0.5 / math.sqrt(2), rel=0.05)
    assert jerk == pytest.approx(0.5 * 2 * math.pi * 0.2 / math.sqrt(2), rel=0.05)
    # Units and amplitude do not change the steering features.
    scaled = window_features(time, 57.3 * steer, accel, window=10.0)
    np.testing.assert_allclose(scaled[:, :2], rows[:, :2], rtol=1e-6)
    assert window_features(time[:20], steer[:20], accel[:20]).shape == (
        0,
        len(FEATURES),
    )
    with pytest.raises(ValueError):
        window_features(time[::-1], steer, accel)


def test_adequacy_flags_quantized_signals() -> None:
    time, steer, _ = _signals(0.3, 0.5)
    assert adequacy(time, steer)["constant_windows"] == 0
    coarse = np.round(steer, 0)  # 0.2 amplitude rounds to zero everywhere
    report = adequacy(time, coarse)
    assert report["constant_windows"] == report["windows"] == 5
    assert report["distinct"] == 1


def test_discriminability_separates_only_different_populations() -> None:
    rng = np.random.default_rng(0)

    def population(shift: float, groups: int) -> tuple[np.ndarray, list[str]]:
        rows = rng.normal(shift, 1.0, size=(groups * 20, len(FEATURES)))
        return rows, [f"g{index // 20}" for index in range(len(rows))]

    same_a, groups_a = population(0.0, 6)
    same_b, groups_b = population(0.0, 6)
    assert abs(discriminability(same_a, same_b, groups_a, groups_b)["auc"] - 0.5) < 0.15
    far, groups_far = population(3.0, 6)
    assert discriminability(same_a, far, groups_a, groups_far)["auc"] > 0.95
    with pytest.raises(ValueError, match="two groups"):
        discriminability(same_a, far, ["one"] * len(same_a), groups_far)
    participants = [index // 40 for index in range(len(same_a) + len(same_b))]
    drives = [index // 20 % 2 for index in range(len(participants))]
    floor = human_floor(np.vstack([same_a, same_b]), participants, drives)
    assert len(floor["auc"]) == 6 and floor["min"] <= floor["median"] <= floor["max"]
    assert set(summarize(same_a)) == set(FEATURES)

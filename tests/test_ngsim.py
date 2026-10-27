"""NGSIM loading, trial extraction and identical processing of simulated drives."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

from myosuite_carla.human_comparison import (  # noqa: E402
    FOLLOWING_FEATURES,
    StyleDiscriminator,
    following_features,
    overlap_coefficient,
    scoreboard,
)
from myosuite_carla.ngsim import (  # noqa: E402
    FEET,
    following_trials,
    load_tracks,
    load_trials,
    resample_following,
    save_trials,
    split_by_driver,
    split_windows,
)

HEADER = (
    "vehicle_id,frame_id,global_time,local_x,local_y,v_length,v_vel,v_acc,"
    "lane_id,preceding,space_headway\n"
)


def _write_ngsim(path: Path, pairs: int = 3, seconds: float = 50.0) -> None:
    """Leader-follower pairs in separate lanes at 8 m/s with a 15 m gap."""
    rows = []
    frames = int(seconds * 10)
    for pair in range(pairs):
        leader, follower = 100 + 2 * pair, 101 + 2 * pair
        for frame in range(frames):
            time_ms = 1_113_433_100_000 + frame * 100
            lead_y = (30.0 + 8.0 * frame / 10) / FEET
            follow_y = (8.0 * frame / 10 + 0.01 * np.sin(frame)) / FEET
            lane = pair + 1
            rows.append(
                f"{leader},{frame},{time_ms},10,{lead_y:.3f},15,26,0,{lane},0,0\n"
            )
            rows.append(
                f"{follower},{frame},{time_ms},10,{follow_y:.3f},15,26,0,{lane},"
                f"{leader},{lead_y - follow_y:.3f}\n"
            )
    path.write_text(HEADER + "".join(rows))


def test_trials_recover_gap_speed_and_drivers(tmp_path: Path) -> None:
    csv = tmp_path / "ngsim.csv"
    _write_ngsim(csv)
    tracks = load_tracks(csv)
    assert tracks["track"].nunique() == 6
    assert tracks["y"].max() == pytest.approx(30.0 + 8.0 * 49.9, rel=1e-3)
    trials = following_trials(tracks, duration=20.0)
    # Two non-overlapping 20 s windows fit in each 50 s follow.
    assert len(trials) == 6 and len({trial.driver for trial in trials}) == 3
    trial = trials[0]
    assert len(trial.time) == 201
    # Gap = leader front - leader length - follower front = 30 - 15 * 0.3048.
    np.testing.assert_allclose(trial.gap[20:-20], 30.0 - 15 * FEET, atol=0.05)
    np.testing.assert_allclose(trial.speed[20:-20], 8.0, atol=0.05)
    assert following_trials(tracks, max_speed=5.0) == []
    train, test = split_by_driver(trials, test_fraction=0.34, seed=1)
    assert {t.driver for t in train}.isdisjoint({t.driver for t in test})
    restored = load_trials(save_trials(trials, tmp_path / "trials.npz"))
    assert [t.driver for t in restored] == [t.driver for t in trials]
    np.testing.assert_array_equal(restored[0].gap, trial.gap)


def test_simulated_follow_uses_the_same_processing() -> None:
    # A 50 Hz simulated follow at constant speeds becomes the same 10 Hz trial.
    time = np.arange(0.0, 20.0 + 1e-9, 0.02)
    trial = resample_following(time, 8.0 * time, 20.0 + 8.0 * time, 4.8, "sim")
    assert trial.time[0] == 0.0 and len(trial.time) == 201
    np.testing.assert_allclose(trial.gap, 15.2, atol=1e-9)
    np.testing.assert_allclose(trial.speed, 8.0, atol=1e-9)
    np.testing.assert_allclose(trial.acceleration, 0.0, atol=1e-9)
    features = dict(zip(FOLLOWING_FEATURES, following_features(trial)))
    assert features["thw_mean_s"] == pytest.approx(15.2 / 8.0)
    assert features["brake_fraction"] == 0.0


def test_overlap_scoreboard_and_style_discriminator() -> None:
    rng = np.random.default_rng(0)
    assert overlap_coefficient([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert overlap_coefficient([0, 0.1], [10, 10.1]) == 0.0
    human = rng.normal(0, 1, (200, len(FOLLOWING_FEATURES)))
    agent = rng.normal(2, 1, (200, len(FOLLOWING_FEATURES)))
    board = scoreboard(human, agent)
    assert board["total"] == len(FOLLOWING_FEATURES) and board["passed"] == 0
    assert scoreboard(human, human)["passed"] == len(FOLLOWING_FEATURES)
    discriminator = StyleDiscriminator(clip=4.0)
    with pytest.raises(RuntimeError):
        discriminator.log_odds(human[:1])
    discriminator.fit(human, agent)
    evidence = discriminator.log_odds(np.vstack([np.zeros(8), np.full(8, 2.0)]))
    assert evidence[0] > 1.0 and evidence[1] < -1.0
    assert np.all(np.abs(discriminator.log_odds(rng.normal(9, 1, (5, 8)))) <= 4.0)


def test_trials_split_into_windows_of_equal_length() -> None:
    time = np.arange(0.0, 20.0 + 1e-9, 0.02)
    trial = resample_following(time, 8.0 * time, 20.0 + 8.0 * time, 4.8, "sim")
    windows = split_windows([trial], 5.0)
    assert len(windows) == 4 and {len(w.time) for w in windows} == {51}
    assert windows[1].time[0] == 0.0 and windows[1].time[-1] == pytest.approx(5.0)
    np.testing.assert_array_equal(windows[1].gap, trial.gap[50:101])


def test_source_groups_survive_windows_archives_and_driver_splits(
    tmp_path: Path,
) -> None:
    time = np.arange(0.0, 20.0 + 1e-9, 0.02)
    trials = [
        resample_following(
            time,
            8.0 * time,
            20.0 + 8.0 * time,
            4.8,
            driver=f"simulation:{i}",
            source_group=f"human:{i // 2}",
        )
        for i in range(6)
    ]
    windows = split_windows(trials, 5.0)
    assert [window.group_id for window in windows] == [
        trial.group_id for trial in trials for _ in range(4)
    ]
    restored = load_trials(save_trials(windows, tmp_path / "paired.npz"))
    assert [trial.group_id for trial in restored] == [
        trial.group_id for trial in windows
    ]
    train, test = split_by_driver(restored, test_fraction=0.34, seed=1)
    assert {trial.group_id for trial in train}.isdisjoint(
        trial.group_id for trial in test
    )
    assert len({trial.group_id for trial in test}) == 1
    unpaired = replace(trials[0], source_group=None)
    assert unpaired.group_id == unpaired.driver


def test_legacy_trial_archive_uses_driver_as_source_group(tmp_path: Path) -> None:
    time = np.arange(21, dtype=float) / 10
    arrays = {
        field: np.asarray([time])
        for field in (
            "time",
            "position",
            "speed",
            "acceleration",
            "gap",
            "leader_speed",
        )
    }
    path = tmp_path / "legacy.npz"
    np.savez(path, drivers=np.asarray(["human:1"]), **arrays)
    restored = load_trials(path)
    assert restored[0].group_id == "human:1"

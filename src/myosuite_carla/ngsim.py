"""Human car-following trials from NGSIM vehicle trajectories.

The Next Generation Simulation (NGSIM) trajectories of the US Federal Highway
Administration (CC BY-SA 3.0, ``examples/download_ngsim.py``) record every
vehicle's front-centre position at 10 Hz from video. A *track* is one
vehicle's continuous record; vehicle IDs repeat across recording periods, so
tracks are split wherever the record has a gap. Each track is one driver.

A car-following *trial* is a fixed-length window in which a car follows the
same leader in the same lane, both are tracked throughout and neither exceeds
``max_speed``. Positions are smoothed with a Savitzky-Golay filter, and speed
and acceleration are its derivatives, because NGSIM's own speed and
acceleration columns amplify video-tracking noise (Montanino and Punzo, 2015,
Transp. Res. Part B 80:82). :func:`resample_following` applies the same
processing to simulated drives, so both sides of a comparison are treated
alike. Requires pandas and SciPy.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FEET = 0.3048
RATE = 10.0
#: Savitzky-Golay window (samples at 10 Hz) and polynomial order.
SMOOTHING = (21, 3)


@dataclass(frozen=True)
class FollowingTrial:
    """One car-following window at 10 Hz.

    Args:
        driver: Track identifier of the follower, used as the grouping unit.
        time: Seconds from the window start.
        position: Follower front position along the road, metres.
        speed: Follower speed, m/s.
        acceleration: Follower acceleration, m/s^2.
        gap: Bumper-to-bumper gap to the leader, metres.
        leader_speed: Leader speed, m/s.
    """

    driver: str
    time: NDArray[np.float64]
    position: NDArray[np.float64]
    speed: NDArray[np.float64]
    acceleration: NDArray[np.float64]
    gap: NDArray[np.float64]
    leader_speed: NDArray[np.float64]

    @property
    def duration(self) -> float:
        """Return the window length in seconds."""
        return float(self.time[-1] - self.time[0])


def _smooth(position: NDArray[np.float64]) -> tuple[NDArray[np.float64], ...]:
    signal = importlib.import_module("scipy.signal")
    window, order = SMOOTHING
    window = min(window, len(position) - (1 - len(position) % 2))
    if window <= order:
        raise ValueError("track too short to smooth")
    dt = 1.0 / RATE
    smoothed = signal.savgol_filter(position, window, order)
    speed = signal.savgol_filter(position, window, order, deriv=1, delta=dt)
    accel = signal.savgol_filter(position, window, order, deriv=2, delta=dt)
    return smoothed, speed, accel


def resample_following(
    time: Sequence[float],
    position: Sequence[float],
    leader_position: Sequence[float],
    leader_length: float,
    driver: str = "",
) -> FollowingTrial:
    """Bring a simulated follow onto the NGSIM grid and processing.

    Front positions are interpolated to 10 Hz and smoothed exactly like the
    human tracks; the gap is the leader's rear minus the follower's front.
    """
    t: NDArray[np.float64] = np.asarray(time, dtype=float)
    # Relative times keep 0.1 s steps exact for large epoch timestamps.
    t = t - t[0]
    grid = np.arange(int(round(t[-1] * RATE)) + 1) / RATE
    follower = np.interp(grid, t, np.asarray(position, dtype=float))
    leader = np.interp(grid, t, np.asarray(leader_position, dtype=float))
    smoothed, speed, accel = _smooth(follower)
    leader_smoothed, leader_speed, _ = _smooth(leader)
    return FollowingTrial(
        driver=driver,
        time=grid,
        position=smoothed,
        speed=speed,
        acceleration=accel,
        gap=leader_smoothed - leader_length - smoothed,
        leader_speed=leader_speed,
    )


def load_tracks(path: str | Path) -> Any:
    """Read a downloaded NGSIM CSV into SI units with one track per driver.

    Returns:
        A pandas DataFrame sorted by track and time with columns ``track``,
        ``vehicle_id``, ``time`` (s), ``y`` (front position, m), ``lane``,
        ``preceding`` (vehicle ID, 0 for none) and ``length`` (m).
    """
    pandas = importlib.import_module("pandas")
    frame = pandas.read_csv(
        path,
        usecols=[
            "vehicle_id",
            "global_time",
            "local_y",
            "v_length",
            "lane_id",
            "preceding",
        ],
    )
    frame = frame.drop_duplicates(["vehicle_id", "global_time"]).sort_values(
        ["vehicle_id", "global_time"]
    )
    frame["time"] = frame["global_time"] / 1000.0
    gaps = frame.groupby("vehicle_id")["time"].diff()
    starts = gaps.isna() | (gaps > 1.5 / RATE)
    frame["track"] = (
        frame["vehicle_id"].astype(str)
        + ":"
        + starts.groupby(frame["vehicle_id"]).cumsum().astype(str)
    )
    return pandas.DataFrame(
        {
            "track": frame["track"],
            "vehicle_id": frame["vehicle_id"].astype(int),
            "time": frame["time"],
            "y": frame["local_y"] * FEET,
            "lane": frame["lane_id"].astype(int),
            "preceding": frame["preceding"].astype(int),
            "length": frame["v_length"] * FEET,
        }
    ).reset_index(drop=True)


def following_trials(
    tracks: Any,
    *,
    duration: float = 20.0,
    max_speed: float = 16.0,
    min_speed: float = 0.0,
    max_trials_per_driver: int = 3,
) -> list[FollowingTrial]:
    """Cut car-following windows of ``duration`` seconds from NGSIM tracks.

    A window needs one leader and one lane throughout, both vehicles tracked
    at every sample, a positive gap and speeds within ``[min_speed,
    max_speed]`` after smoothing. Windows of a driver do not overlap; at most
    ``max_trials_per_driver`` are kept per driver.
    """
    samples = int(round(duration * RATE)) + 1
    key = (tracks["time"] * RATE).round().astype(np.int64)
    leaders = tracks.assign(key=key)[["vehicle_id", "key", "y", "length"]].rename(
        columns={"vehicle_id": "preceding", "y": "leader_y", "length": "leader_length"}
    )
    merged = tracks.assign(key=key).merge(
        leaders.drop_duplicates(["preceding", "key"]),
        on=["preceding", "key"],
        how="left",
    )
    trials: list[FollowingTrial] = []
    for track, group in merged.groupby("track", sort=False):
        if len(group) < samples:
            continue
        time = group["time"].to_numpy()
        leader = group["preceding"].to_numpy()
        lane = group["lane"].to_numpy()
        leader_y = group["leader_y"].to_numpy()
        valid = (leader != 0) & np.isfinite(leader_y)
        steady = np.r_[True, np.abs(np.diff(time) - 1.0 / RATE) < 0.02]
        changes = (
            np.r_[True, (leader[1:] != leader[:-1]) | (lane[1:] != lane[:-1])]
            | ~steady
            | ~valid
        )
        runs: NDArray[np.int_] = np.cumsum(changes)
        kept = 0
        for run in np.unique(runs[valid]):
            indices = np.flatnonzero((runs == run) & valid)
            for first in range(indices[0], indices[-1] + 2 - samples, samples):
                if kept >= max_trials_per_driver:
                    break
                window = slice(first, first + samples)
                trial = resample_following(
                    time[window],
                    group["y"].to_numpy()[window],
                    leader_y[window],
                    float(group["leader_length"].to_numpy()[first]),
                    driver=str(track),
                )
                speeds = np.r_[trial.speed, trial.leader_speed]
                if (
                    np.all(trial.gap > 0)
                    and speeds.max() <= max_speed
                    and speeds.min() >= min_speed
                ):
                    trials.append(trial)
                    kept += 1
    return trials


def split_windows(
    trials: Sequence[FollowingTrial], seconds: float
) -> list[FollowingTrial]:
    """Cut trials into consecutive windows of ``seconds`` (shared end samples)."""
    step = int(round(seconds * RATE))
    windows = []
    for trial in trials:
        for start in range(0, len(trial.time) - step, step):
            part = slice(start, start + step + 1)
            windows.append(
                FollowingTrial(
                    driver=trial.driver,
                    time=trial.time[part] - trial.time[start],
                    position=trial.position[part],
                    speed=trial.speed[part],
                    acceleration=trial.acceleration[part],
                    gap=trial.gap[part],
                    leader_speed=trial.leader_speed[part],
                )
            )
    return windows


def split_by_driver(
    trials: Sequence[FollowingTrial], test_fraction: float = 0.3, seed: int = 0
) -> tuple[list[FollowingTrial], list[FollowingTrial]]:
    """Split trials into training and held-out sets with no shared driver."""
    drivers = sorted({trial.driver for trial in trials})
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(drivers)[: int(round(test_fraction * len(drivers)))])
    train = [trial for trial in trials if trial.driver not in held]
    test = [trial for trial in trials if trial.driver in held]
    return train, test


_FIELDS = ("time", "position", "speed", "acceleration", "gap", "leader_speed")


def save_trials(trials: Sequence[FollowingTrial], path: str | Path) -> Path:
    """Write trials to a compressed ``.npz`` (equal-length windows)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.stack([getattr(t, name) for t in trials]) for name in _FIELDS}
    np.savez_compressed(
        target, drivers=np.asarray([t.driver for t in trials]), **arrays
    )
    return target


def load_trials(path: str | Path) -> list[FollowingTrial]:
    """Read trials written by :func:`save_trials`."""
    with np.load(path, allow_pickle=False) as data:
        drivers = data["drivers"].tolist()
        arrays = {name: data[name] for name in _FIELDS}
    return [
        FollowingTrial(
            driver=str(driver), **{name: arrays[name][i] for name in _FIELDS}
        )
        for i, driver in enumerate(drivers)
    ]

"""Compare driving behaviour with human data through aggregate window features.

Signals are resampled to a common rate, low-pass filtered (which also removes
the steps of sample-and-hold logging) and cut into fixed windows. Each window
becomes a feature vector:

* ``steer_frequency_hz``: Rice's mean frequency of the steering signal,
  ``rms(d steer/dt) / (2 pi sd(steer))``, independent of units and amplitude;
* ``reversal_rate_per_min``: steering reversals larger than 10% of the
  window's 5-95% steering range, after the reversal-rate idea of Markkula and
  Engstrom (2006), with an amplitude-free gap;
* ``accel_sd_mps2`` and ``jerk_rms_mps3``: longitudinal acceleration spread and
  jerk, in physical units.

:func:`discriminability` is the cross-validated area under the ROC curve of a
logistic-regression classifier that separates two sets of windows, with
whole groups (participants, drives or simulated episodes) held out together.
0.5 means indistinguishable. :func:`human_floor` gives the same AUC for each
human participant against all others, the natural between-driver spread that
a model need not undercut. Only aggregates leave these functions.
Requires SciPy and scikit-learn.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

FEATURES = (
    "steer_frequency_hz",
    "reversal_rate_per_min",
    "accel_sd_mps2",
    "jerk_rms_mps3",
)
#: Features computed from each source signal.
SIGNAL_FEATURES = {
    "steer": FEATURES[:2],
    "acceleration": FEATURES[2:],
}


def _reversals(signal: NDArray[np.float64], gap: float) -> int:
    """Count direction changes that exceed ``gap`` from the last extremum."""
    if gap <= 0:
        return 0
    count, direction, extreme = 0, 0, float(signal[0])
    low = high = extreme
    for value in signal[1:]:
        value = float(value)
        if direction == 0:
            low, high = min(low, value), max(high, value)
            if value - low >= gap:
                direction, extreme = 1, value
            elif high - value >= gap:
                direction, extreme = -1, value
        elif direction > 0:
            if value > extreme:
                extreme = value
            elif extreme - value >= gap:
                count, direction, extreme = count + 1, -1, value
        else:
            if value < extreme:
                extreme = value
            elif value - extreme >= gap:
                count, direction, extreme = count + 1, 1, value
    return count


def window_features(
    time: Sequence[float],
    steer: Sequence[float],
    acceleration: Sequence[float],
    *,
    window: float = 10.0,
    rate: float = 10.0,
    cutoff: float = 2.0,
) -> NDArray[np.float64]:
    """Return one row of :data:`FEATURES` per complete window.

    Args:
        time: Monotonic sample times in seconds.
        steer: Steering signal in any unit (wheel angle or normalized steer).
        acceleration: Longitudinal acceleration in m/s^2.
        window: Window length in seconds.
        rate: Common resampling rate in Hz.
        cutoff: Second-order Butterworth low-pass cut-off in Hz.
    """
    signal = importlib.import_module("scipy.signal")
    t: NDArray[np.float64] = np.asarray(time, dtype=float)
    if len(t) < 2 or np.any(np.diff(t) <= 0):
        raise ValueError("time must increase")
    grid = np.arange(t[0], t[-1], 1.0 / rate)
    samples = int(round(window * rate))
    if len(grid) < max(samples, 16):
        return np.zeros((0, len(FEATURES)))
    b, a = signal.butter(2, cutoff / (rate / 2))
    steer_grid = signal.filtfilt(b, a, np.interp(grid, t, np.asarray(steer, float)))
    accel_grid = signal.filtfilt(
        b, a, np.interp(grid, t, np.asarray(acceleration, float))
    )
    rows = []
    for start in range(0, len(grid) - samples + 1, samples):
        part = steer_grid[start : start + samples]
        accel = accel_grid[start : start + samples]
        deviation = float(np.std(part))
        rate_rms = float(np.sqrt(np.mean(np.gradient(part, 1.0 / rate) ** 2)))
        spread = float(np.subtract(*np.percentile(part, [95, 5])))
        rows.append(
            (
                rate_rms / (2 * math.pi * deviation) if deviation > 1e-9 else 0.0,
                _reversals(part, 0.1 * spread) * 60.0 / window,
                float(np.std(accel)),
                float(np.sqrt(np.mean(np.gradient(accel, 1.0 / rate) ** 2))),
            )
        )
    return np.asarray(rows, dtype=float)


def adequacy(
    time: Sequence[float], values: Sequence[float], *, window: float = 10.0
) -> dict[str, float]:
    """Resolution diagnostics of one logged signal over complete windows.

    A signal that is constant within most windows (coarse quantization or
    sparse logging) cannot support the window features computed from it.

    Returns:
        Complete windows, windows with a constant raw value, distinct values
        and the smallest non-zero step.
    """
    t: NDArray[np.float64] = np.asarray(time, dtype=float)
    v: NDArray[np.float64] = np.asarray(values, dtype=float)
    steps = np.abs(np.diff(v))
    steps = steps[steps > 0]
    blocks = np.floor((t - t[0]) / window).astype(int)
    complete = np.unique(blocks[blocks < blocks.max()]) if len(blocks) else []
    constant = [float(np.ptp(v[blocks == block])) == 0.0 for block in complete]
    return {
        "windows": float(len(constant)),
        "constant_windows": float(sum(constant)),
        "distinct": float(np.unique(v).size),
        "resolution": float(steps.min()) if steps.size else 0.0,
    }


def discriminability(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    first_groups: Sequence[Any],
    second_groups: Sequence[Any],
    *,
    folds: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """Cross-validated AUC of a classifier separating two sets of windows.

    Groups are held out together (``StratifiedGroupKFold``), so the classifier
    is always tested on drivers or episodes it has not seen.

    Returns:
        ``auc``, window counts and the number of folds used.
    """
    model_selection = importlib.import_module("sklearn.model_selection")
    metrics = importlib.import_module("sklearn.metrics")
    pipeline = importlib.import_module("sklearn.pipeline")
    preprocessing = importlib.import_module("sklearn.preprocessing")
    linear = importlib.import_module("sklearn.linear_model")
    features = np.vstack([first, second])
    labels = np.r_[np.zeros(len(first)), np.ones(len(second))]
    groups = np.asarray(
        [f"a:{group}" for group in first_groups]
        + [f"b:{group}" for group in second_groups]
    )
    usable = min(
        folds,
        len(set(groups[labels == 0])),
        len(set(groups[labels == 1])),
    )
    if usable < 2:
        raise ValueError("Each side needs at least two groups")
    splitter = model_selection.StratifiedGroupKFold(
        n_splits=usable, shuffle=True, random_state=seed
    )
    scores = np.zeros(len(labels))
    for train, test in splitter.split(features, labels, groups):
        classifier = pipeline.make_pipeline(
            preprocessing.StandardScaler(),
            linear.LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        classifier.fit(features[train], labels[train])
        scores[test] = classifier.predict_proba(features[test])[:, 1]
    return {
        "auc": float(metrics.roc_auc_score(labels, scores)),
        "windows": [int(len(first)), int(len(second))],
        "folds": int(usable),
    }


def human_floor(
    features: NDArray[np.float64],
    participants: Sequence[Any],
    drives: Sequence[Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """AUC of each participant against all others, grouped by drive.

    Returns:
        Per-participant AUCs (anonymous order) and their median and range.
    """
    people: NDArray[Any] = np.asarray(participants)
    runs: NDArray[Any] = np.asarray(drives)
    values = []
    for participant in sorted(set(people.tolist())):
        mine = people == participant
        groups_mine = [f"{participant}:{drive}" for drive in runs[mine]]
        groups_rest = [
            f"{other}:{drive}" for other, drive in zip(people[~mine], runs[~mine])
        ]
        try:
            result = discriminability(
                features[mine], features[~mine], groups_mine, groups_rest, **kwargs
            )
        except ValueError:
            continue
        values.append(result["auc"])
    return {
        "auc": values,
        "median": float(np.median(values)) if values else math.nan,
        "min": float(np.min(values)) if values else math.nan,
        "max": float(np.max(values)) if values else math.nan,
    }


def summarize(features: NDArray[np.float64]) -> dict[str, dict[str, float]]:
    """Median and interquartile range of every feature."""
    return {
        name: {
            "median": float(np.median(features[:, index])),
            "q25": float(np.percentile(features[:, index], 25)),
            "q75": float(np.percentile(features[:, index], 75)),
        }
        for index, name in enumerate(FEATURES)
    }

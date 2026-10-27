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
a model need not undercut. Classification reports retain row fold assignments
for auditing source separation. Raw signals remain local.

Car-following trials (:class:`~myosuite_carla.ngsim.FollowingTrial`, human or
simulated through the same processing) become :data:`FOLLOWING_FEATURES`.
:func:`scoreboard` compares each feature's distributions by their histogram
overlap coefficient, passing at 0.60 or more as in arm_force_driving's judge,
and :class:`StyleDiscriminator` turns a gradient-boosted classifier's log-odds
into a trial-level style reward (GAIL, Ho and Ermon, 2016, at trial level).
Requires SciPy and scikit-learn.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any, Literal

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
#: Trial-level car-following features.
FOLLOWING_FEATURES = (
    "thw_mean_s",
    "thw_sd_s",
    "thw_p10_s",
    "gap_p10_m",
    "accel_sd_mps2",
    "jerk_rms_mps3",
    "relative_speed_abs_mps",
    "brake_fraction",
)
#: Overlap coefficient at or above which a feature counts as human-like.
OVL_PASS = 0.60


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
        bounds = np.percentile(part, [5, 95])
        spread = float(bounds[1] - bounds[0])
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


def following_features(trial: Any) -> NDArray[np.float64]:
    """Return :data:`FOLLOWING_FEATURES` of one 10 Hz car-following trial.

    Time headway is the gap over ``max(speed, 1 m/s)`` so that standstill
    stays finite; jerk is the derivative of the smoothed acceleration.
    """
    speed = np.asarray(trial.speed, dtype=float)
    gap = np.asarray(trial.gap, dtype=float)
    accel = np.asarray(trial.acceleration, dtype=float)
    headway = gap / np.maximum(speed, 1.0)
    dt = float(np.median(np.diff(trial.time)))
    return np.asarray(
        [
            float(np.mean(headway)),
            float(np.std(headway)),
            float(np.percentile(headway, 10)),
            float(np.percentile(gap, 10)),
            float(np.std(accel)),
            float(np.sqrt(np.mean(np.gradient(accel, dt) ** 2))),
            float(np.mean(np.abs(np.asarray(trial.leader_speed) - speed))),
            float(np.mean(accel < -0.5)),
        ]
    )


def overlap_coefficient(
    first: Sequence[float], second: Sequence[float], bins: int = 20
) -> float:
    """Shared area of two normalized histograms on their common range."""
    a: NDArray[np.float64] = np.asarray(first, dtype=float)
    b: NDArray[np.float64] = np.asarray(second, dtype=float)
    low, high = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    if high <= low:
        return 1.0
    edges = np.linspace(low, high, bins + 1)
    pa = np.histogram(a, edges)[0] / len(a)
    pb = np.histogram(b, edges)[0] / len(b)
    return float(np.minimum(pa, pb).sum())


def scoreboard(
    human: NDArray[np.float64],
    agent: NDArray[np.float64],
    names: Sequence[str] = FOLLOWING_FEATURES,
) -> dict[str, Any]:
    """Per-feature medians and overlap coefficients of two trial banks."""
    rows = {}
    for index, name in enumerate(names):
        ovl = overlap_coefficient(human[:, index], agent[:, index])
        rows[name] = {
            "human_median": float(np.median(human[:, index])),
            "agent_median": float(np.median(agent[:, index])),
            "ovl": ovl,
            "passed": ovl >= OVL_PASS,
        }
    return {
        "features": rows,
        "passed": int(sum(row["passed"] for row in rows.values())),
        "total": len(rows),
    }


class StyleDiscriminator:
    """Gradient-boosted human-vs-agent classifier for a trial-level reward.

    :meth:`fit` uses human training trials and the latest agent trials with
    balanced class weights; :meth:`log_odds` is then the evidence that a new
    trial is human, which a policy can be paid at the end of each trial.

    Args:
        clip: Bound of the returned log-odds.
        seed: Classifier seed.
    """

    def __init__(self, clip: float = 5.0, seed: int = 0) -> None:
        self.clip = float(clip)
        self.seed = int(seed)
        self.model: Any = None

    @property
    def fitted(self) -> bool:
        """Return whether :meth:`fit` has run."""
        return self.model is not None

    def fit(
        self, human: NDArray[np.float64], agent: NDArray[np.float64]
    ) -> StyleDiscriminator:
        """Fit on human (label 1) and agent (label 0) feature rows."""
        ensemble = importlib.import_module("sklearn.ensemble")
        features = np.vstack([human, agent])
        labels = np.r_[np.ones(len(human)), np.zeros(len(agent))]
        model = ensemble.HistGradientBoostingClassifier(
            max_iter=100, max_depth=3, class_weight="balanced", random_state=self.seed
        )
        self.model = model.fit(features, labels)
        return self

    def log_odds(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the clipped log-odds that each row is human."""
        if self.model is None:
            raise RuntimeError("Fit the discriminator first")
        rows = np.atleast_2d(np.asarray(features, dtype=float))
        probability = np.clip(self.model.predict_proba(rows)[:, 1], 1e-6, 1 - 1e-6)
        return np.clip(np.log(probability / (1 - probability)), -self.clip, self.clip)


def discriminability(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
    first_groups: Sequence[Any],
    second_groups: Sequence[Any],
    *,
    folds: int = 5,
    seed: int = 0,
    model: str = "logistic",
    grouping: Literal["independent", "shared"] = "independent",
) -> dict[str, Any]:
    """Cross-validated AUC of a classifier separating two sets of windows.

    Whole source groups form test folds. Independent populations use
    ``StratifiedKFold`` over group IDs and their population labels. Shared
    sources use ``StratifiedGroupKFold`` over rows. ``model`` is
    ``"logistic"`` (standardized logistic regression) or ``"boosting"``
    (gradient-boosted trees, a stricter judge for nonlinear differences).
    ``grouping="shared"`` holds a matching source ID out across both
    populations, as required when a simulation replays a human trial.
    ``"independent"`` gives each population its own group namespace.

    Returns:
        AUC, row and source counts, grouping mode and fold assignments for
        each population in input row order. No raw feature values are returned.
    """
    model_selection = importlib.import_module("sklearn.model_selection")
    metrics = importlib.import_module("sklearn.metrics")
    pipeline = importlib.import_module("sklearn.pipeline")
    preprocessing = importlib.import_module("sklearn.preprocessing")
    linear = importlib.import_module("sklearn.linear_model")
    ensemble = importlib.import_module("sklearn.ensemble")
    if model not in ("logistic", "boosting"):
        raise ValueError("model must be 'logistic' or 'boosting'")
    if grouping not in ("independent", "shared"):
        raise ValueError("grouping must be 'independent' or 'shared'")
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[1] != second.shape[1]
        or first.shape[1] == 0
        or not np.all(np.isfinite(first))
        or not np.all(np.isfinite(second))
    ):
        raise ValueError("Each population needs finite rows with matching features")
    if len(first_groups) != len(first) or len(second_groups) != len(second):
        raise ValueError("Each feature row needs one source group")
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer of at least two")
    raw_groups = [
        [str(group) for group in values] for values in (first_groups, second_groups)
    ]
    if any(not group for values in raw_groups for group in values):
        raise ValueError("Source group identifiers must be nonempty")
    features = np.vstack([first, second])
    labels = np.r_[np.zeros(len(first)), np.ones(len(second))]
    groups = np.asarray(
        [
            f"{population}:{group}" if grouping == "independent" else group
            for population, values in enumerate(raw_groups)
            for group in values
        ]
    )
    counts = [len(set(groups[labels == label])) for label in (0, 1)]
    usable = min(
        folds,
        *counts,
    )
    if usable < 2:
        raise ValueError("Each side needs at least two groups")
    if grouping == "independent":
        unique_groups, first_rows = np.unique(groups, return_index=True)
        splitter = model_selection.StratifiedKFold(
            n_splits=usable, shuffle=True, random_state=seed
        )
        splits = [
            (
                np.flatnonzero(np.isin(groups, unique_groups[train])),
                np.flatnonzero(np.isin(groups, unique_groups[test])),
            )
            for train, test in splitter.split(unique_groups, labels[first_rows])
        ]
        splitter_name = "StratifiedKFold"
    else:
        splitter = model_selection.StratifiedGroupKFold(
            n_splits=usable, shuffle=True, random_state=seed
        )
        splits = list(splitter.split(features, labels, groups))
        splitter_name = "StratifiedGroupKFold"
    scores = np.zeros(len(labels))
    assignments: NDArray[np.int_] = np.full(len(labels), -1, dtype=int)
    fold_groups = []
    for train, test in splits:
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[test])) != 2:
            raise ValueError(
                "Each fold needs both populations. "
                "Use more source groups or fewer folds."
            )
    for index, (train, test) in enumerate(splits):
        classifier = (
            pipeline.make_pipeline(
                preprocessing.StandardScaler(),
                linear.LogisticRegression(max_iter=1000, class_weight="balanced"),
            )
            if model == "logistic"
            else ensemble.HistGradientBoostingClassifier(
                max_iter=100, max_depth=3, class_weight="balanced", random_state=seed
            )
        )
        classifier.fit(features[train], labels[train])
        scores[test] = classifier.predict_proba(features[test])[:, 1]
        assignments[test] = index
        fold_groups.append(
            {
                "train": len(set(groups[train])),
                "test": len(set(groups[test])),
                "train_populations": [
                    len(set(groups[train][labels[train] == label])) for label in (0, 1)
                ],
                "test_populations": [
                    len(set(groups[test][labels[test] == label])) for label in (0, 1)
                ],
            }
        )
    return {
        "auc": float(metrics.roc_auc_score(labels, scores)),
        "windows": [int(len(first)), int(len(second))],
        "folds": int(usable),
        "model": model,
        "seed": int(seed),
        "splitter": splitter_name,
        "grouping": grouping,
        "groups": counts,
        "independent_groups": len(set(groups)),
        "shared_groups": len(set(raw_groups[0]) & set(raw_groups[1]))
        if grouping == "shared"
        else 0,
        "fold_groups": fold_groups,
        "fold_assignments": [
            assignments[: len(first)].tolist(),
            assignments[len(first) :].tolist(),
        ],
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


def summarize(
    features: NDArray[np.float64], names: Sequence[str] = FEATURES
) -> dict[str, dict[str, float]]:
    """Median and interquartile range of every feature."""
    return {
        name: {
            "median": float(np.median(features[:, index])),
            "q25": float(np.percentile(features[:, index], 25)),
            "q75": float(np.percentile(features[:, index], 75)),
        }
        for index, name in enumerate(names)
    }

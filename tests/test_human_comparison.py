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


def test_shared_sources_stay_together_across_populations(monkeypatch) -> None:
    import sklearn.model_selection

    actual = sklearn.model_selection.StratifiedGroupKFold
    observed = []

    class AuditedSplit(actual):
        def split(self, features, labels, groups):
            for train, test in super().split(features, labels, groups):
                observed.append((train.copy(), test.copy(), groups.copy()))
                yield train, test

    monkeypatch.setattr(sklearn.model_selection, "StratifiedGroupKFold", AuditedSplit)
    rng = np.random.default_rng(3)
    first = rng.normal(size=(24, 2))
    second = first + rng.normal(0, 0.2, first.shape)
    sources = [f"driver:{i // 4}" for i in range(len(first))]
    report = discriminability(first, second, sources, sources, grouping="shared")
    assert report["grouping"] == "shared"
    assert report["independent_groups"] == 6
    assert report["shared_groups"] == 6
    assert report["groups"] == [6, 6]
    for train, test, groups in observed:
        assert set(groups[train]).isdisjoint(set(groups[test]))
        for source in sources:
            indices = {i for i, value in enumerate(sources * 2) if value == source}
            assert indices <= set(train) or indices <= set(test)


def test_independent_grouping_preserves_population_namespaces() -> None:
    rng = np.random.default_rng(4)
    rows = rng.normal(size=(40, 2))
    groups = [f"g{i // 10}" for i in range(len(rows))]
    implicit = discriminability(rows, rows + 1, groups, groups)
    explicit = discriminability(rows, rows + 1, groups, groups, grouping="independent")
    assert implicit == explicit
    assert implicit["independent_groups"] == 8
    assert implicit["shared_groups"] == 0


@pytest.mark.parametrize(
    "sizes_a,sizes_b,folds",
    [([20] * 6, [20] * 6, 5), ([1, 39], [1, 20, 5, 17, 6, 1, 13, 2, 9, 7], 5)],
)
def test_independent_folds_keep_each_population_with_unequal_source_sizes(
    monkeypatch, sizes_a, sizes_b, folds
) -> None:
    import sklearn.model_selection

    class LegacyGroupSplit:
        def __init__(self, **kwargs):
            pass

        def split(self, features, labels, groups):
            # Older group splitters can place rare-class groups in one fold.
            yield np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)

    monkeypatch.setattr(
        sklearn.model_selection, "StratifiedGroupKFold", LegacyGroupSplit
    )
    rng = np.random.default_rng(8)
    populations = [rng.normal(size=(sum(sizes), 2)) for sizes in (sizes_a, sizes_b)]
    identifiers = [
        [f"group:{i}" for i, size in enumerate(sizes) for _ in range(size)]
        for sizes in (sizes_a, sizes_b)
    ]
    report = discriminability(*populations, *identifiers, folds=folds)
    assert report["splitter"] == "StratifiedKFold"
    assert report["folds"] == min(folds, len(sizes_a), len(sizes_b))
    for source_ids, assignments in zip(identifiers, report["fold_assignments"]):
        assignments = np.asarray(assignments)
        for source in set(source_ids):
            assert len(set(assignments[np.asarray(source_ids) == source])) == 1
        assert set(assignments) == set(range(report["folds"]))
    for fold in report["fold_groups"]:
        assert min(fold["train_populations"]) >= 1
        assert min(fold["test_populations"]) >= 1


@pytest.mark.parametrize("groups", [["one"] * 4, ["one", "one", "two"]])
def test_shared_grouping_rejects_insufficient_or_misaligned_sources(groups) -> None:
    rows = np.arange(8, dtype=float).reshape(4, 2)
    with pytest.raises(ValueError):
        discriminability(rows, rows + 1, groups, groups, grouping="shared")

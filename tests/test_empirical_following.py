"""Observed state IDM fitting preserves source groups and independent targets."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from myosuite_carla.driver_models import IdmConfig, idm_acceleration
from myosuite_carla.empirical_following import (
    FIXED_ERROR_SCALES,
    METRICS,
    FollowingProtocol,
    closed_loop_following,
    idm_config,
    idm_parameters,
    load_following_protocol,
)
from myosuite_carla.ngsim import FollowingTrial, save_trials


def _trial(
    name: str,
    *,
    config: IdmConfig = IdmConfig(),
    seconds: float = 5.0,
    acceleration_offset: float = 0.0,
    source_group: str | None = None,
    phase: float = 0.0,
) -> FollowingTrial:
    time = np.arange(round(seconds * 10) + 1) / 10
    speed = 5 + np.sin(time + phase)
    leader = speed - 0.5 * np.sin(0.5 * time + phase)
    gap = 15 + 3 * np.cos(time / 2 + phase)
    acceleration = np.asarray(
        [
            idm_acceleration(v, 16, g, v - lv, config)
            for v, g, lv in zip(speed, gap, leader)
        ]
    )
    position = np.r_[0.0, np.cumsum(0.5 * (speed[:-1] + speed[1:]) * np.diff(time))]
    return FollowingTrial(
        driver=name,
        time=time,
        position=position,
        speed=speed,
        acceleration=acceleration + acceleration_offset,
        gap=gap,
        leader_speed=leader,
        source_group=source_group,
    )


def test_original_source_identity_and_equal_trial_weight_survive_preparation() -> None:
    trials = [
        _trial("window-one", source_group="original", seconds=3, acceleration_offset=1),
        _trial(
            "window-two", source_group="original", seconds=10, acceleration_offset=3
        ),
        _trial("other", acceleration_offset=5),
    ]
    protocol = FollowingProtocol(trials)
    assert protocol.groups == ("ngsim:original", "ngsim:other")
    prediction = protocol(idm_parameters(), protocol.groups, seed=100)
    assert prediction["ngsim:original"] == {
        "acceleration_rmse": pytest.approx(2),
        "acceleration_bias": pytest.approx(-2),
    }
    assert prediction["ngsim:other"]["acceleration_rmse"] == pytest.approx(5)
    assert protocol.metadata["group_kind"] == "vehicle_track"
    assert protocol.metadata["population_source_groups"] == 2
    assert protocol.metadata["population_trials"] == 3
    assert protocol.observed == {
        group: {"acceleration_rmse": 0, "acceleration_bias": 0}
        for group in protocol.groups
    }


def test_prepared_trials_do_not_change_when_the_callers_arrays_change() -> None:
    trial = _trial("one")
    protocol = FollowingProtocol([trial])
    before = protocol({}, protocol.groups, 0)
    trial.acceleration[:] = 999
    assert protocol({}, protocol.groups, 1) == before
    with pytest.raises(ValueError):
        protocol.trials_for(protocol.groups[0])[0].gap[:] = 0


def test_source_subset_is_seeded_before_calibration_and_preserves_population() -> None:
    trials = [_trial(f"source:{i}") for i in range(12)]
    first = FollowingProtocol(trials, max_groups=5, seed=17)
    second = FollowingProtocol(list(reversed(trials)), max_groups=5, seed=17)
    assert first.groups == second.groups
    assert len(first.observed) == 5
    assert first.metadata["population_source_groups"] == 12
    assert first.metadata["selected_source_groups"] == 5
    assert first.metadata["desired_speed_mps"] == 16
    with pytest.raises(ValueError, match="unknown selected"):
        first({}, ("a different source",), 0)


def test_saved_trial_loading_retains_original_source_group(tmp_path: Any) -> None:
    path = tmp_path / "trials.npz"
    save_trials([_trial("replay", source_group="source")], path)
    protocol = load_following_protocol(path)
    assert protocol.groups == ("ngsim:source",)
    assert protocol.trials_for("ngsim:source")[0].driver == "replay"


def test_known_idm_parameter_is_recovered_on_unseen_source_groups() -> None:
    evaluation = pytest.importorskip("embodied_human.driving_evaluation")
    inference = pytest.importorskip("embodied_human.inference")
    truth = IdmConfig(time_headway=1.5)
    protocol = FollowingProtocol(
        [_trial(str(i), config=truth, phase=i / 10) for i in range(9)]
    )
    parameters = idm_parameters(replace(truth, time_headway=0.5))
    report = evaluation.calibrate_grouped(
        protocol.observed,
        protocol,
        [inference.Prior("time_headway", 0.2, 2.2)],
        parameters,
        metrics=METRICS,
        scales=FIXED_ERROR_SCALES,
        configuration={"human": {}, "driver": {}, "reflex": {"idm": parameters}},
        provenance={"protocol": "synthetic known IDM acceleration"},
        samples=32,
        accept=0.25,
        bootstrap=0,
    )
    selected = report["profile"]["parameters"]
    assert selected["time_headway"] == pytest.approx(1.5, abs=0.06)
    held = report["evaluations"]["test"]
    assert (
        held["selected"]["metrics"]["acceleration_rmse"]["mae"]["value"]
        < held["baseline"]["metrics"]["acceleration_rmse"]["mae"]["value"] / 10
    )
    assert set(report["partitions"]["test"]).isdisjoint(report["partitions"]["train"])
    assert set(report["partitions"]["test"]).isdisjoint(
        report["partitions"]["validation"]
    )


@pytest.mark.parametrize(
    "parameters",
    [
        {"desired_speed": 20},
        {"max_acceleration": 0},
        {"comfortable_deceleration": -1},
        {"min_gap": -1},
        {"time_headway": float("nan")},
    ],
)
def test_invalid_physical_parameters_are_rejected(parameters: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        idm_config(parameters)


def test_invalid_observations_fail_before_fitting() -> None:
    trial = _trial("one")
    with pytest.raises(ValueError, match="positive gaps"):
        FollowingProtocol([replace(trial, gap=np.zeros_like(trial.gap))])
    with pytest.raises(ValueError, match="positive gaps"):
        FollowingProtocol([replace(trial, leader_speed=trial.leader_speed[:-1])])


def test_closed_loop_unavailable_dependency_has_explicit_status(
    monkeypatch: Any,
) -> None:
    import builtins

    original = builtins.__import__

    def import_without_human(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "embodied_human":
            raise ModuleNotFoundError("embodied_human is unavailable")
        return original(name, *args, **kwargs)

    protocol = FollowingProtocol([_trial("source")])
    monkeypatch.setattr(builtins, "__import__", import_without_human)
    report = closed_loop_following(protocol, {}, {}, groups=protocol.groups)
    assert report["status"] == "unavailable"
    assert "ModuleNotFoundError" in report["error"]
    assert report["episodes"] == []
    assert report["selected_source_groups"] == ["ngsim:source"]


def test_closed_loop_requires_an_explicit_selected_source_subset() -> None:
    protocol = FollowingProtocol([_trial("source")])
    with pytest.raises(ValueError, match="held out subset"):
        closed_loop_following(protocol, {}, {}, groups=[])
    with pytest.raises(ValueError, match="held out subset"):
        closed_loop_following(protocol, {}, {}, groups=["different"])

"""Observed state IDM fitting preserves source groups and independent targets."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from myosuite_carla.driver_models import IdmConfig, idm_acceleration
from myosuite_carla.empirical_following import (
    CLOSED_LOOP_METRICS,
    CLOSED_LOOP_SCALES,
    FIXED_ERROR_SCALES,
    METRICS,
    ClosedLoopConfig,
    ClosedLoopFollowingProtocol,
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


def test_old_source_exclusions_precede_seeded_selection_and_retain_population() -> None:
    trials = [_trial(str(i)) for i in range(20)]
    excluded = tuple(f"ngsim:{i}" for i in range(10))
    first = FollowingProtocol(trials, max_groups=8, seed=0, excluded_groups=excluded)
    second = FollowingProtocol(
        list(reversed(trials)), max_groups=8, seed=0, excluded_groups=excluded
    )
    assert first.groups == second.groups
    assert set(first.groups).isdisjoint(excluded)
    assert len(first.groups) == 8
    assert first.metadata["population_source_groups"] == 20
    assert first.metadata["excluded_population_groups"] == 10
    assert first.metadata["excluded_source_groups"] == sorted(excluded)
    with pytest.raises(ValueError, match="no eligible"):
        FollowingProtocol(trials[:2], excluded_groups=excluded)


@pytest.mark.parametrize("excluded", ["ngsim:source", b"ngsim:source"])
def test_source_exclusions_require_a_sequence_of_source_identities(excluded) -> None:
    with pytest.raises(ValueError, match="excluded_groups"):
        FollowingProtocol([_trial("source")], excluded_groups=excluded)


@pytest.mark.parametrize(
    "settings",
    [
        {"seconds": 0.1},
        {"seconds": float("inf")},
        {"seconds": True},
        {"trials_per_source": 0},
        {"trials_per_source": True},
        {"keep_trajectories": 1},
    ],
)
def test_closed_loop_settings_reject_invalid_physical_objectives(settings) -> None:
    with pytest.raises(ValueError):
        ClosedLoopConfig(**settings)


def _fake_physical_protocol(monkeypatch, trials):
    import myosuite_carla.empirical_following as module

    protocol = ClosedLoopFollowingProtocol(
        FollowingProtocol(trials), config=ClosedLoopConfig(trials_per_source=2)
    )
    executions = []

    def ensure_driver():
        protocol._driver = object()
        protocol._body_metadata = {"body_builds": 1}

    def execute(driver, trial, parameters, *, group, trial_index, seed, **kwargs):
        executions.append((parameters, group, trial_index, seed))
        return {
            "status": "completed",
            "source_group": group,
            "original_source_group": trial.group_id,
            "trial_index": trial_index,
            "seed": seed,
            "parameters": idm_parameters(parameters),
            "collision": False,
            "off_road": False,
            "surrogate_closed": True,
            "coverage_fraction": 1.0,
            "failure_kind": None,
            "speed_rmse_mps": float(trial.acceleration[0]),
            "gap_rmse_m": 2.0,
            "acceleration_rmse_mps2": 3.0,
            "time_s": [0.0, 20.0],
        }

    monkeypatch.setattr(protocol, "_ensure_driver", ensure_driver)
    monkeypatch.setattr(module, "_run_closed_loop_episode", execute)
    return protocol, executions, execute


def test_physical_objective_averages_fixed_trials_without_reweighting_sources(
    monkeypatch,
):
    first = _trial("first", source_group="a")
    second = _trial("second", source_group="a")
    third = _trial("third", source_group="b")
    first.acceleration[:] = 1.0
    second.acceleration[:] = 3.0
    third.acceleration[:] = 9.0
    protocol, executions, _ = _fake_physical_protocol(
        monkeypatch, [first, second, third]
    )
    result = protocol({}, tuple(reversed(protocol.groups)), 17)
    assert result["ngsim:a"]["speed_rmse_mps"] == 2.0
    assert result["ngsim:b"]["speed_rmse_mps"] == 9.0
    assert protocol.observed == {
        group: dict.fromkeys(CLOSED_LOOP_METRICS, 0.0) for group in protocol.groups
    }
    assert CLOSED_LOOP_SCALES == dict(zip(CLOSED_LOOP_METRICS, (1.0, 5.0, 1.0)))
    assert len(executions) == 3
    assert protocol.metadata["stochastic"] is True
    assert protocol.metadata["selected_trials"] == 3


def test_episode_cache_binds_exact_settings_source_and_seed_and_copies_records(
    monkeypatch,
):
    protocol, executions, _ = _fake_physical_protocol(
        monkeypatch, [_trial("a"), _trial("b")]
    )
    groups = protocol.groups
    first = protocol({}, groups, 0)
    assert protocol({}, tuple(reversed(groups)), 0) == first
    protocol({}, (groups[1],), 0)
    assert len(executions) == 2
    assert protocol.diagnostics["cache_hits"] == 3
    old_seeds = {(group, index): seed for _, group, index, seed in executions}
    protocol({"time_headway": 2.0}, groups, 0)
    assert len(executions) == 4
    assert {(group, index): seed for _, group, index, seed in executions} == old_seeds
    protocol({}, groups, 1)
    assert len(executions) == 6
    assert all(
        seed != old_seeds[(group, index)] for _, group, index, seed in executions[-2:]
    )
    report = protocol.report_pair({}, {}, groups=groups, seed=0)
    assert len(executions) == 6
    assert len(report["episodes"]) == 4
    report["episodes"][0]["time_s"][0] = 999
    assert protocol.episodes[0]["time_s"][0] == 0
    protocol.close()
    with pytest.raises(RuntimeError, match="closed"):
        protocol({}, groups, 0)


@pytest.mark.parametrize(
    "change",
    [
        {"collision": True, "failure_kind": "collision"},
        {"off_road": True, "failure_kind": "road_departure"},
        {"coverage_fraction": 0.9, "failure_kind": "incomplete_horizon"},
        {"speed_rmse_mps": float("nan"), "failure_kind": "computational_error"},
        {"surrogate_closed": False, "failure_kind": "cleanup_error"},
        {"status": "failed", "error": "RuntimeError: solver error"},
    ],
)
def test_every_failed_required_episode_makes_candidate_infeasible(monkeypatch, change):
    import myosuite_carla.empirical_following as module

    protocol, executions, execute = _fake_physical_protocol(
        monkeypatch, [_trial("a"), _trial("b")]
    )

    def failing(*args, **kwargs):
        record = execute(*args, **kwargs)
        if kwargs["group"] == "ngsim:b":
            record.update(change)
        return record

    monkeypatch.setattr(module, "_run_closed_loop_episode", failing)
    with pytest.raises(RuntimeError, match="infeasible"):
        protocol({}, protocol.groups, 0)
    assert len(executions) == 2
    assert len(protocol.diagnostics["failed_episodes"]) == 1
    assert protocol.diagnostics["failed_episodes"][0]["source_group"] == "ngsim:b"
    report = protocol.report_pair({}, {}, groups=protocol.groups)
    assert report["status"] == "failed"
    assert len(report["episodes"]) == 4
    assert len(executions) == 2


def test_reusable_physical_protocol_runs_the_actual_body_and_releases_vehicles():
    pytest.importorskip("embodied_human")
    pytest.importorskip("myosuite.core.model_builder")
    source = FollowingProtocol([_trial("source", seconds=1)])
    with ClosedLoopFollowingProtocol(
        source, config=ClosedLoopConfig(seconds=1.0)
    ) as protocol:
        baseline = protocol({}, protocol.groups, 100)
        assert all(np.isfinite(list(baseline[protocol.groups[0]].values())))
        pair = protocol.report_pair({}, {}, groups=protocol.groups, seed=100)
        assert pair["status"] == "completed"
        assert pair["muscles"] == 114
        assert pair["body_builds"] == 1
        assert protocol.diagnostics["episode_runs"] == 1
        assert all(row["surrogate_closed"] for row in pair["episodes"])
        assert pair["episodes"][0]["seed"] == pair["episodes"][1]["seed"]
        assert (
            pair["episodes"][0]["predicted_speed_mps"]
            == pair["episodes"][1]["predicted_speed_mps"]
        )
        assert all(row["full_source_horizon"] for row in pair["episodes"])
    assert protocol.diagnostics["closed"] is True


@pytest.mark.parametrize("failure", ["collision", "off_road"])
def test_early_safety_termination_retains_cause_when_scoring_is_unavailable(
    monkeypatch, failure
):
    from types import SimpleNamespace

    eh = pytest.importorskip("embodied_human")
    from embodied_human.agents import driver as human_driver

    from myosuite_carla import driving_env, ngsim, training
    from myosuite_carla.empirical_following import _run_closed_loop_episode

    initial = {
        "time": 0.0,
        "collision": False,
        "off_road": False,
        "ego_front_s_m": 0.0,
        "lead_front_s_m": 20.0,
        "lead_length_m": 4.0,
    }
    final = dict(initial, time=0.02, **{failure: True})

    class Environment:
        dt = 0.02

        def __init__(self, *args, **kwargs):
            self.monitor = SimpleNamespace(events=[], footprints={})
            self.vehicle = SimpleNamespace(active=True, close=self.close_vehicle)

        def reset(self, **kwargs):
            return None, {"scenario": initial}

        def step(self, action):
            return None, 0.0, True, False, {"scenario": final}

        def close(self):
            pass

        def close_vehicle(self):
            self.vehicle.active = False

    agent = SimpleNamespace(reset=lambda **kwargs: None, act=lambda: None)
    monkeypatch.setattr(human_driver, "AttentiveDriver", lambda *args, **kwargs: agent)
    monkeypatch.setattr(eh, "build_human", lambda **kwargs: object())
    monkeypatch.setattr(driving_env, "FullBodyDrivingEnv", Environment)
    monkeypatch.setattr(
        training, "scenario_from_trial", lambda *args: SimpleNamespace(duration=5.0)
    )

    def too_short(*args, **kwargs):
        raise ValueError("track too short to smooth")

    monkeypatch.setattr(ngsim, "resample_following", too_short)
    body = SimpleNamespace(
        data=SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1), act=np.zeros(1))
    )
    record = _run_closed_loop_episode(
        body,
        _trial("a"),
        IdmConfig(),
        desired_speed=16,
        group="ngsim:a",
        trial_index=0,
        seed=0,
        config=ClosedLoopConfig(seconds=None),
    )
    assert record["status"] == "terminated"
    assert record["failure_kind"] == (
        "collision" if failure == "collision" else "road_departure"
    )
    assert record["surrogate_closed"] is True
    assert "too short" in record["error"]

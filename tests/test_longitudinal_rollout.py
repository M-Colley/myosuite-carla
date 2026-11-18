"""Longitudinal refinement preserves steering and controls full rollout errors."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from myosuite_carla.longitudinal import (
    LongitudinalRolloutFitConfig,
    fit_longitudinal_rollout,
    longitudinal_correction,
)
from myosuite_carla.vehicle_model import (
    DriveSegment,
    carla_1_0_nissan_patrol,
    simulate,
)

pytest.importorskip("scipy")


def synthetic_segment(params, initial_speed, throttle, brake):
    drive = np.r_[np.zeros(20), np.full(80, throttle), np.zeros(100)]
    braking = np.r_[np.zeros(120), np.full(80, brake)]
    count = len(drive)
    states = np.zeros(count + 1)
    segment = DriveSegment(
        name="synthetic",
        dt=0.05,
        throttle=drive,
        brake=braking,
        steer=np.zeros(count),
        hand_brake=np.zeros(count),
        reverse=np.zeros(count),
        x=states.copy(),
        y=states.copy(),
        yaw_deg=states.copy(),
        pitch_deg=states.copy(),
        speed=np.r_[initial_speed, np.zeros(count)],
        yaw_rate=states.copy(),
    )
    prediction = simulate(params, segment, use_pitch=False)
    return replace(segment, **prediction)


def test_positive_correction_preserves_monotonicity_and_all_steering_terms():
    baseline = carla_1_0_nissan_patrol()
    corrected = longitudinal_correction(baseline, (0.7, 1.4, 1.1, 0.9, 0.12))
    assert np.all(np.diff(np.asarray(corrected.drive), axis=0) >= 0)
    assert corrected.drive[0][:6] == baseline.drive[0][:6]
    for key in (
        "wheelbase",
        "speed_knots",
        "throttle_knots",
        "reverse_drive",
        "steer_gain",
        "steer_exponent",
        "steer_rise_rate",
        "steer_fall_rate",
        "steer_lag",
        "steer_lag_slope",
        "rolling",
        "linear_drag",
        "hand_brake_decel",
    ):
        assert getattr(corrected, key) == getattr(baseline, key)
    assert (
        longitudinal_correction(baseline, (1, 1, 1, 1, baseline.throttle_lag))
        == baseline
    )


def test_full_rollout_fit_recovers_synthetic_high_speed_dynamics():
    baseline = carla_1_0_nissan_patrol()
    gain = np.interp(baseline.speed_knots, (0, 8, 15, 20), (1, 1, 1.08, 1.16))
    truth = replace(
        baseline,
        drive=tuple(map(tuple, np.asarray(baseline.drive) * gain)),
        quadratic_drag=baseline.quadratic_drag * 0.95,
        brake_decel=baseline.brake_decel * 1.08,
        throttle_lag=0.12,
    )
    segments = [
        synthetic_segment(truth, *values)
        for values in product((0, 14, 19), (0.75, 1.0), (0.4, 0.8))
    ]
    params, report = fit_longitudinal_rollout(
        segments,
        baseline,
        config=LongitudinalRolloutFitConfig(regularization_weight=0, max_nfev=40),
    )
    assert report["success"]
    assert report["fitted_speed_mse"] < report["baseline_speed_mse"] * 1e-4
    np.testing.assert_allclose(
        report["parameters"], (1.08, 1.16, 0.95, 1.08, 0.12), atol=0.003
    )
    assert params.steer_gain == baseline.steer_gain


def test_rollout_fit_ignores_unavailable_future_pitch():
    baseline = carla_1_0_nissan_patrol()
    segment = synthetic_segment(baseline, 14, 0.75, 0.4)
    perturbed = replace(segment, pitch_deg=np.linspace(-30, 30, len(segment.speed)))
    fitted, report = fit_longitudinal_rollout([perturbed], baseline)
    assert report["baseline_speed_mse"] == 0
    assert report["fitted_speed_mse"] == 0
    assert fitted == baseline


SPEC = importlib.util.spec_from_file_location(
    "longitudinal_example",
    Path(__file__).parents[1] / "examples/calibrate_longitudinal.py",
)
assert SPEC is not None and SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def make_row(seed, target, error):
    return {
        "seed": seed,
        "target_speed": target,
        "speed_mse": error**2,
        "speed_rmse_mps": error,
        "final_position_error_m": error,
    }


def test_scoring_and_fitting_weights_balance_targets_and_seeds():
    rows = [make_row(1, 17.0, 0)] * 10 + [make_row(1, 22.0, 2)]
    rows += [make_row(2, 17.0, 1), make_row(2, 22.0, 3)]
    score = EXAMPLE.score_rows(rows, [17.0, 22.0])
    assert score["mean"]["primary"] == 3.5
    assert score["mean"]["speed_rmse_mps"] == 1.5
    segments = [SimpleNamespace(name=str(i)) for i in range(len(rows))]
    metadata = {
        s.name: dict(row, target_speed=row["target_speed"])
        for s, row in zip(segments, rows)
    }
    weights = EXAMPLE.trajectory_weights(segments, metadata)
    assert sum(weights[:10]) == pytest.approx(0.25)
    assert weights[10:] == pytest.approx([0.25, 0.25, 0.25])


def test_mean_improvement_cannot_hide_worse_tail_trial():
    design = {
        "test_seeds": [1, 2, 3],
        "launch_profile_grid": {"target_speeds_mps": [17.0, 22.0]},
        "acceptance": {"confidence": 0.95, "resamples": 5000, "bootstrap_seed": 42},
    }
    old = [
        make_row(seed, target, 1)
        for seed, target, _ in product((1, 2, 3), (17.0, 22.0), range(6))
    ]
    new = [make_row(row["seed"], row["target_speed"], 0.5) for row in old]
    new[0] = make_row(1, 17.0, 1.5)
    baseline = EXAMPLE.score_rows(old, [17.0, 22.0])
    candidate = EXAMPLE.score_rows(new, [17.0, 22.0])
    result = EXAMPLE.compare_test(baseline, candidate, design)
    assert candidate["mean"]["primary"] < baseline["mean"]["primary"]
    assert result["paired_seed_primary_change"]["interval"][1] < 0
    assert not result["guards"]["maximum_trial_speed_rmse_mps"]
    assert result["decision"] == "experimental_longitudinal"


@pytest.fixture
def prospective_batch(tmp_path, monkeypatch):
    design = json.loads(
        (
            Path(__file__).parents[1]
            / "evidence/carla-1.0-longitudinal-2026-11-18/design.json"
        ).read_bytes()
    )
    grid = design["launch_profile_grid"]
    metadata = {
        "status": "completed",
        "seed": 1100,
        "client_version": "1.0",
        "server_version": "1.0",
        "map": design["target"]["map"],
        "blueprint": design["target"]["vehicle"],
        "vehicle": design["target"]["vehicle"],
        "control_dt": 0.02,
        "script_sha256": design["collector_sha256"],
        "counts": design["counts"],
        "spawn_indices": design["spawn_indices"],
        "spawn_points_used": 3,
        "min_straight_m": 350.0,
        "straight_limit_m": 600.0,
        "max_road_pitch_deg": 1.0,
        "max_lateral_acceleration_mps2": 3.0,
        "steer_input_exponent": 1.0,
        "launch_profile_grid": grid,
    }
    attempts = [
        {
            "segment": f"s1100-{i:02d}-launch",
            "kind": "launch",
            "target_speed": target,
            "level": throttle,
            "brake": brake,
            "kept": True,
            "status": "completed",
            "completed": True,
            "targeted_launch": True,
            "target_reached": True,
            "incomplete_reason": None,
            "collision": False,
            "max_roll_deg": 0.0,
            "frame_limit_exceeded": False,
        }
        for i, (target, throttle, brake) in enumerate(
            product(grid["target_speeds_mps"], grid["throttles"], grid["brakes"])
        )
    ]
    segments = [SimpleNamespace(name=row["segment"], dt=0.02) for row in attempts]
    monkeypatch.setattr(
        EXAMPLE.Recording, "load", lambda _: SimpleNamespace(metadata=metadata)
    )
    monkeypatch.setattr(EXAMPLE, "segments_from_recording", lambda _: segments)
    folder = tmp_path / "seed1100"
    folder.mkdir()
    return tmp_path, design, metadata, attempts


def load_prospective(case):
    root, design, metadata, attempts = case
    (root / "seed1100/collection.json").write_text(
        json.dumps(dict(metadata, segments=attempts))
    )
    return EXAMPLE.load_batches(root, [1100], design)


def test_complete_balanced_grid_passes_prospective_loader(prospective_batch):
    segments, _, coverage = load_prospective(prospective_batch)
    assert len(segments) == 12
    assert coverage["1100"]["accepted"] == 12


def test_loader_rejects_missing_high_speed_support(prospective_batch):
    for row in prospective_batch[3]:
        if row["target_speed"] == 22:
            row["kept"] = False
    with pytest.raises(ValueError, match="coverage rules failed"):
        load_prospective(prospective_batch)


def test_loader_rejects_accepted_trial_without_assigned_speed(prospective_batch):
    prospective_batch[3][0]["target_reached"] = False
    with pytest.raises(ValueError, match="Accepted trial quality"):
        load_prospective(prospective_batch)


def test_loader_rejects_wrong_target_assignment(prospective_batch):
    prospective_batch[3][0]["target_speed"] = 18
    with pytest.raises(ValueError, match="Assigned launch profiles differ"):
        load_prospective(prospective_batch)


def test_loader_rejects_changed_collector(prospective_batch):
    prospective_batch[2]["script_sha256"] = "changed"
    with pytest.raises(ValueError, match="script_sha256 differs"):
        load_prospective(prospective_batch)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_ineligible_diagnostics_cannot_select_or_authorize_testing(
    tmp_path, monkeypatch, diagnostic
):
    design = (
        Path(__file__).parents[1]
        / "evidence/carla-1.0-longitudinal-2026-11-18/design.json"
    )
    output = tmp_path / "analysis"
    arguments = [
        "calibrate_longitudinal",
        "--design",
        str(design),
        "--data",
        str(tmp_path / "data"),
        "--test-data",
        str(tmp_path / "future-test"),
        "--out",
        str(output),
        "--stage",
        "fit",
    ]
    if diagnostic:
        arguments.append("--diagnostic")
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(
        EXAMPLE,
        "load_batches",
        lambda _, seeds, *a, **k: (
            [SimpleNamespace(name="trial")],
            {"trial": {"seed": seeds[0], "target_speed": 17}},
            {
                str(seed): {"eligible": False, "attempted": 12, "accepted": 1}
                for seed in seeds
            },
        ),
    )
    calls = []

    def fitting(segments, baseline, **kwargs):
        calls.append(segments)
        return baseline, {"success": True}

    monkeypatch.setattr(EXAMPLE, "fit_longitudinal_rollout", fitting)
    monkeypatch.setattr(
        EXAMPLE,
        "evaluate",
        lambda *a, **k: {"rows": [], "scoring_error": "missing target support"},
    )
    assert EXAMPLE.main() == (0 if diagnostic else 2)
    report = json.loads((output / "fit-report.json").read_bytes())
    frozen = json.loads((output / "frozen-selection.json").read_bytes())
    assert bool(calls) == diagnostic
    assert not report["protocol_eligible"]
    assert report["diagnostic_only"] == diagnostic
    assert report["selected"] is frozen["selected"] is False

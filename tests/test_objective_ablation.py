"""The prospective objective comparison retains shared physical ownership."""

from __future__ import annotations

import importlib.util
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "objective_ablation", ROOT / "examples/ablate_longitudinal.py"
)
assert SPEC is not None and SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def row(seed, speed, history, action, error):
    return {
        "name": f"s{seed}-{speed}-{history}-{action}",
        "seed": seed,
        "speed_mps": speed,
        "history": history,
        "action": action,
        "speed_mse": error**2,
        "speed_rmse_mps": error,
        "position_error_3s_m": error,
        "acceleration_error_1s_mps2": -error if history == "acceleration" else error,
    }


def test_tail_guard_prevents_positive_objective_claim_despite_better_mean():
    design = json.loads(
        (
            ROOT / "evidence/carla-1.0-objective-ablation-2026-11-18/design.json"
        ).read_bytes()
    )
    rows = [
        row(*values, 1.0)
        for values in product(
            design["test_seeds"],
            (17.0, 20.0),
            ("acceleration", "braking"),
            ("throttle", "brake"),
        )
    ]
    candidate = [
        dict(r, speed_mse=0.25, speed_rmse_mps=0.5, position_error_3s_m=0.5)
        for r in rows
    ]
    candidate[0].update(speed_mse=2.25, speed_rmse_mps=1.5)
    old, new = EXAMPLE.score_rows(rows), EXAMPLE.score_rows(candidate)
    comparison = EXAMPLE.compare(old, new, design)
    assert new["mean"]["primary"] < old["mean"]["primary"]
    assert comparison["paired_seed_primary_change"]["interval"][1] < 0
    assert not comparison["guards"]["maximum_trial_speed_rmse_mps"]
    assert comparison["decision"] == "rollout_objective_advantage_not_supported"


def test_both_objectives_freeze_without_selection_on_historical_diagnostics(
    tmp_path, monkeypatch
):
    design = ROOT / "evidence/carla-1.0-objective-ablation-2026-11-18/design.json"
    output = tmp_path / "analysis"
    monkeypatch.setattr(
        "sys.argv",
        [
            "ablate_longitudinal",
            "--design",
            str(design),
            "--data",
            str(tmp_path / "development"),
            "--test-data",
            str(tmp_path / "future"),
            "--out",
            str(output),
            "--stage",
            "fit",
        ],
    )
    monkeypatch.setattr(
        EXAMPLE.HISTORICAL,
        "load_batches",
        lambda _, seeds, __: (
            [SimpleNamespace(name="trial")],
            {"trial": {"seed": seeds[0], "target_speed": 17.0}},
            {str(s): {"eligible": True, "accepted": 12} for s in seeds},
        ),
    )
    objectives = []

    def fitting(segments, baseline, *, objective, **kwargs):
        objectives.append(objective)
        return baseline, {"success": True, "objective": objective}

    monkeypatch.setattr(EXAMPLE, "fit_longitudinal_rollout", fitting)
    monkeypatch.setattr(
        EXAMPLE.HISTORICAL, "evaluate", lambda *a: {"diagnostic_only": True}
    )
    assert EXAMPLE.main() == 0
    assert objectives == ["derivative", "rollout"]
    report = json.loads((output / "fit-report.json").read_bytes())
    frozen = json.loads((output / "frozen-models.json").read_bytes())
    assert frozen["eligible"] is report["eligible"] is True
    assert set(frozen["models"]) == {"C3", "derivative", "rollout"}
    assert (
        frozen["models"]["C3"]
        == frozen["models"]["derivative"]
        == frozen["models"]["rollout"]
    )
    assert frozen["fresh_test_collected"] is False

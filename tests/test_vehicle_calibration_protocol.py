"""Scientific guards of the frozen vehicle comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("scipy")

SPEC = importlib.util.spec_from_file_location(
    "vehicle_calibration_example",
    Path(__file__).parents[1] / "examples/evaluate_vehicle_calibration.py",
)
assert SPEC is not None and SPEC.loader is not None
EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXAMPLE)


def row(name, speed=0, yaw=0, position=0):
    return {
        "name": name,
        "speed_rmse_mps": speed,
        "yaw_rate_rmse_deg_s": yaw,
        "final_position_error_m": position,
    }


def test_family_weights_do_not_follow_unequal_attempt_counts():
    rows = [row(f"s10-{i}-launch") for i in range(10)]
    rows.append(row("s10-11-steer", speed=10))
    score = EXAMPLE.source_score(rows, ["launch", "steer"])
    assert score["primary"] == 25
    assert score["speed_rmse_mps"] == 5


def test_yaw_units_and_position_remain_separate():
    score = EXAMPLE.source_score([row("s10-0-steer", 2, 5, 9)], ["steer"])
    assert score["primary"] == 2.5
    assert score["final_position_error_m"] == 9
    with pytest.raises(ValueError, match="Missing manoeuvre"):
        EXAMPLE.source_score([row("s10-0-steer")], ["steer", "launch"])


def test_selection_refuses_already_collected_test_batches(tmp_path, monkeypatch):
    design = tmp_path / "design.json"
    design.write_text(json.dumps({"counts": {"launch": 1}, "test_seeds": [30]}))
    data = tmp_path / "data"
    (data / "seed30").mkdir(parents=True)
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_vehicle_calibration",
            "--stage",
            "fit",
            "--design",
            str(design),
            "--data",
            str(data),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="Test batches must remain uncollected"):
        EXAMPLE.main()

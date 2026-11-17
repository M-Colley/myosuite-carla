"""Scientific guards of the frozen vehicle comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture
def batch_case(tmp_path, monkeypatch):
    design = {
        "target": {
            "carla": "1.0",
            "map": "Carla/Maps/Town10HD_Opt",
            "vehicle": "vehicle.nissan.patrol",
            "dt_s": 0.02,
        },
        "counts": {"launch": 2},
        "min_straight_m": 100.0,
        "max_road_pitch_deg": 1.0,
        "nominal_max_lateral_acceleration_mps2": 3.0,
        "collection_steer_exponent": 1.0,
    }
    metadata = {
        "status": "completed",
        "seed": 10,
        "client_version": "1.0",
        "server_version": "1.0",
        "map": "Carla/Maps/Town10HD_Opt",
        "blueprint": "vehicle.nissan.patrol",
        "vehicle": "vehicle.nissan.patrol",
        "control_dt": 0.02,
        "min_straight_m": 100.0,
        "max_road_pitch_deg": 1.0,
        "max_lateral_acceleration_mps2": 3.0,
        "steer_input_exponent": 1.0,
        "counts": {"launch": 2},
    }
    log = dict(metadata)
    log["segments"] = [
        {
            "segment": f"s10-{index:02d}-launch",
            "kind": "launch",
            "kept": index == 0,
            "status": "completed",
            "completed": True,
        }
        for index in range(2)
    ]
    recording = SimpleNamespace(metadata=metadata)
    segments = [SimpleNamespace(name="s10-00-launch", dt=0.02)]
    monkeypatch.setattr(EXAMPLE.Recording, "load", lambda _: recording)
    monkeypatch.setattr(EXAMPLE, "segments_from_recording", lambda _: segments)
    directory = tmp_path / "seed10"
    directory.mkdir()
    return {
        "data": tmp_path,
        "design": design,
        "log": log,
        "recording": recording,
        "segments": segments,
    }


def load_case(case):
    (case["data"] / "seed10" / "collection.json").write_text(json.dumps(case["log"]))
    return EXAMPLE.load_batches(case["data"], [10], case["design"])


def test_completed_matching_batch_preserves_accepted_coverage(batch_case):
    batches, coverage = load_case(batch_case)
    assert batches == {"10": batch_case["segments"]}
    assert coverage["10"] == {
        "attempted": {"launch": 2},
        "accepted": {"launch": 1},
        "rejected": 1,
    }


@pytest.mark.parametrize("source", ["log", "recording"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "running"),
        ("seed", 11),
        ("client_version", "0.10.0"),
        ("server_version", "0.10.0"),
        ("map", "Carla/Maps/Town01"),
        ("blueprint", "vehicle.audi.a2"),
        ("control_dt", 0.05),
        ("min_straight_m", 80.0),
        ("max_road_pitch_deg", 2.0),
        ("max_lateral_acceleration_mps2", 4.0),
        ("steer_input_exponent", 2.0),
        ("counts", {"launch": 3}),
    ],
)
def test_batch_rejects_identity_or_configuration_mismatch(
    batch_case, source, field, value
):
    metadata = (
        batch_case["log"] if source == "log" else batch_case["recording"].metadata
    )
    metadata[field] = value
    with pytest.raises(ValueError, match=field):
        load_case(batch_case)


def test_batch_rejects_wrong_recorded_vehicle(batch_case):
    batch_case["recording"].metadata["vehicle"] = "vehicle.audi.a2"
    with pytest.raises(ValueError, match="recording vehicle"):
        load_case(batch_case)


@pytest.mark.parametrize("field", ["status", "completed"])
def test_batch_rejects_incomplete_accepted_trial(batch_case, field):
    batch_case["log"]["segments"][0][field] = False
    with pytest.raises(ValueError, match="Accepted trials must be completed"):
        load_case(batch_case)


def test_equal_segment_count_does_not_hide_wrong_trial_identity(batch_case):
    batch_case["segments"][0].name = "s10-99-launch"
    with pytest.raises(ValueError, match="Saved segments differ"):
        load_case(batch_case)


def test_duplicate_saved_trial_does_not_replace_another_accepted_trial(batch_case):
    batch_case["log"]["segments"][1]["kept"] = True
    batch_case["segments"].append(SimpleNamespace(name="s10-00-launch", dt=0.02))
    with pytest.raises(ValueError, match="Saved segments differ"):
        load_case(batch_case)


@pytest.mark.parametrize("identity", ["s11-00-launch", "s10-00-steer", "duplicate"])
def test_batch_rejects_invalid_collection_identities(batch_case, identity):
    if identity == "duplicate":
        batch_case["log"]["segments"][1]["segment"] = "s10-00-launch"
    else:
        batch_case["log"]["segments"][0]["segment"] = identity
    with pytest.raises(ValueError, match="Invalid collection segment identities"):
        load_case(batch_case)


def test_batch_rejects_sample_period_that_disagrees_with_metadata(batch_case):
    batch_case["segments"][0].dt = 0.05
    with pytest.raises(ValueError, match="Saved segment dt"):
        load_case(batch_case)


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

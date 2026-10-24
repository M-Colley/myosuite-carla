"""Run the standalone offline example through the installed public API."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_offline_cockpit_script_from_outside_checkout(tmp_path: Path) -> None:
    """A fresh interpreter produces nonzero physical pedal and wheel controls."""
    pytest.importorskip("myosuite")
    script = Path(__file__).resolve().parents[1] / "examples" / "offline_cockpit.py"
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output["driver"] == "scripted_contact_probe"
    assert output["muscle_actuators"] > 0
    phases = {phase["phase"]: phase for phase in output["phases"]}
    assert phases["throttle"]["command"]["throttle"] > 0.3
    assert phases["steering"]["command"]["steer"] > 0.01
    for phase in ("throttle", "steering"):
        assert phases[phase]["peak_touch_newtons"] > 1.0
        assert phases[phase]["peak_contact_count"] > 0
    assert phases["released"]["command"]["throttle"] == 0.0
    assert phases["released"]["command"]["brake"] == 0.0
    assert phases["released"]["command"]["steer"] == 0.0


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize(
    "script",
    [
        "drive_scenario.py",
        "identify_vehicle.py",
        "fit_vehicle_model.py",
        "compare_human.py",
        "validate_fullbody.py",
        "download_ngsim.py",
        "train_policy.py",
        "evaluate_policy.py",
        "drive_with_interface.py",
    ],
)
def test_example_scripts_document_their_options(script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-I", str(EXAMPLES / script), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert "usage:" in result.stdout


def test_surrogate_takeover_scenario_runs_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("myosuite.core.model_builder")
    pytest.importorskip("myo_sim")
    output = tmp_path / "scenario.json"
    subprocess.run(
        [
            sys.executable,
            "-I",
            str(EXAMPLES / "drive_scenario.py"),
            "--scenario",
            "takeover_request",
            "--duration",
            "6",
            "--takeover-time",
            "1",
            "--fast",
            "--record",
            str(tmp_path / "recording"),
            "--formats",
            "csv",
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    report = json.loads(output.read_text())
    summary = report["summary"]
    assert report["status"] == "completed" and summary["frames"] == 300
    assert not summary["collision"] and not summary["off_road"]
    assert 2.72 <= summary["takeover_time_s"] <= 3.5
    assert (tmp_path / "recording" / "datapackage.json").exists()

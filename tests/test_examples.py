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

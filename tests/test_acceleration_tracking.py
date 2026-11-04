"""A stopped controller remains accountable until the reference vehicle stops."""

import importlib.util
import math
from pathlib import Path


def _score(actual):
    path = Path(__file__).parents[1] / "examples/check_acceleration_tracking.py"
    spec = importlib.util.spec_from_file_location("tracking_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference_grid_rmse(
        {
            "initial_speed_mps": 5.0,
            "desired_acceleration_mps2": -1.0,
            "samples": [
                {"time_s": time, "actual_acceleration_mps2": value}
                for time, value in zip((1.0, 2.0, 3.0, 5.0), actual)
            ],
        }
    )


def test_reference_brake_hold_has_zero_error_after_the_expected_stop():
    assert _score((-100.0, -1.0, -1.0, 0.0)) == 0.0


def test_early_stop_is_scored_on_the_same_reference_grid():
    assert math.isclose(_score((-100.0, -1.0, 0.0, 0.0)), math.sqrt(1 / 3))

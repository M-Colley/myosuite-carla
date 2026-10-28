"""Physics aligned video frames preserve the episode duration."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "examples/record_empirical_trials.py"
    spec = importlib.util.spec_from_file_location("empirical_video", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twenty_second_trial_has_exactly_five_hundred_video_frames():
    capture = _module().frame_steps(20, 0.02, 25)
    assert list(capture) == list(range(0, 1000, 2))
    assert len(capture) / 25 == 20
    assert 1000 not in capture


@pytest.mark.parametrize("fps", [0, 24, 60, True])
def test_unsupported_video_rate_is_rejected(fps):
    with pytest.raises(ValueError):
        _module().frame_steps(20, 0.02, fps)

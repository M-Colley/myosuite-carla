"""Physics aligned video frames preserve the episode duration."""

import importlib.util
import json
from dataclasses import asdict
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


def test_recorded_actuator_configuration_is_preserved():
    from myosuite_carla.driver_models import (
        AccelerationControlConfig,
        IdmConfig,
        ReflexDriverConfig,
    )

    original = ReflexDriverConfig(
        perception_delay=0.08,
        idm=IdmConfig(time_headway=0.8),
        acceleration_control=AccelerationControlConfig(pedal_integral=0.04),
    )
    restored = _module().controller_configuration(
        asdict(original.idm), 0.02, json.loads(json.dumps(asdict(original)))
    )
    assert restored == original


def test_video_rejects_conflicting_idm_and_controller_settings():
    from myosuite_carla.driver_models import IdmConfig, ReflexDriverConfig

    with pytest.raises(ValueError, match="differ"):
        _module().controller_configuration(
            asdict(IdmConfig(time_headway=0.8)), 0.02, asdict(ReflexDriverConfig())
        )


def test_earlier_video_reports_keep_their_original_controller_defaults():
    from myosuite_carla.driver_models import IdmConfig

    restored = _module().controller_configuration(asdict(IdmConfig()), 0.02, None)
    assert restored.perception_delay == 0.02
    assert restored.acceleration_control is None

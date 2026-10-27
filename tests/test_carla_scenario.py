"""CARLA lane sampling validates traversal parameters before map access."""

import math

import pytest

from myosuite_carla.carla_scenario import lane_from_carla


@pytest.mark.parametrize("name", ["length", "step"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_lane_sampling_requires_finite_positive_distances(
    name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=name):
        lane_from_carla(None, None, **{name: value})

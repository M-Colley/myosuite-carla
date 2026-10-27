"""CARLA lane sampling validates traversal parameters before map access."""

import math
from types import SimpleNamespace

import pytest

from myosuite_carla.carla_scenario import (
    CarlaLeadVehicle,
    lane_from_carla,
    vehicle_footprint,
)
from myosuite_carla.scenario import LeadVehicle, Scenario, ScenarioMonitor


@pytest.mark.parametrize("name", ["length", "step"])
@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, -math.inf, math.nan])
def test_lane_sampling_requires_finite_positive_distances(
    name: str, value: float
) -> None:
    with pytest.raises(ValueError, match=name):
        lane_from_carla(None, None, **{name: value})


def test_actor_bounding_box_supplies_dimensions_center_and_rotation() -> None:
    actor = SimpleNamespace(
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=2.8, y=1.1),
            location=SimpleNamespace(x=0.4, y=-0.1),
            rotation=SimpleNamespace(yaw=15.0),
        )
    )
    footprint = vehicle_footprint(actor)
    assert footprint.length == pytest.approx(5.6)
    assert footprint.width == pytest.approx(2.2)
    assert footprint.center == pytest.approx((0.4, -0.1))
    assert footprint.yaw == pytest.approx(15.0)


def test_live_lead_resolves_geometry_and_preserves_initial_bumper_gap() -> None:
    monitor = ScenarioMonitor(Scenario(lead=LeadVehicle(gap=10.0)))
    monitor.reset(
        SimpleNamespace(
            location=(5.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
        )
    )
    transforms: list = []
    actor = SimpleNamespace(
        bounding_box=SimpleNamespace(
            extent=SimpleNamespace(x=3.0, y=1.0),
            location=SimpleNamespace(x=0.3, y=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
        set_simulate_physics=lambda value: None,
        set_transform=transforms.append,
    )
    world = SimpleNamespace(
        get_blueprint_library=lambda: SimpleNamespace(find=lambda name: name),
        spawn_actor=lambda *args: actor,
    )
    api = SimpleNamespace(
        Location=lambda **kwargs: SimpleNamespace(**kwargs),
        Rotation=lambda **kwargs: SimpleNamespace(**kwargs),
        Transform=lambda location, rotation: SimpleNamespace(
            location=location, rotation=rotation
        ),
    )
    lead = CarlaLeadVehicle(monitor, world, api=api)
    lead.spawn(0.5)
    assert monitor.state["gap_m"] == pytest.approx(10.0)
    assert monitor.footprints["lead"]["length"] == 6.0
    assert transforms[-1].location.x == pytest.approx(20.1)
    before = monitor.state.copy()
    lead(SimpleNamespace(config=SimpleNamespace(fixed_delta_seconds=0.02)), None)
    assert monitor.state == before and monitor.time == 0.0
    monitor.update(
        SimpleNamespace(
            location=(5.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)
        ),
        0.02,
    )
    assert transforms[-1].location.x == pytest.approx(monitor.lead_pose()[0])

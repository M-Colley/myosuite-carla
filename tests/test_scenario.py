"""Scenario geometry, events, automation and the classical driver models."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from myosuite_carla import VehicleCommand
from myosuite_carla.driver_models import idm_acceleration
from myosuite_carla.scenario import (
    VEHICLE_LENGTH,
    Lane,
    LaneChange,
    LeadVehicle,
    Scenario,
    ScenarioMonitor,
    TakeoverRequest,
    VehicleFootprint,
    automation_command,
    lane_change,
    lead_vehicle_braking,
    takeover_request,
)
from myosuite_carla.vehicle_model import SurrogateVehicle


def _telemetry(x: float, y: float, yaw: float = 0.0, speed: float = 0.0):
    heading = math.radians(yaw)
    return SimpleNamespace(
        location=(x, y, 0.0),
        rotation=(0.0, yaw, 0.0),
        velocity=(speed * math.cos(heading), speed * math.sin(heading), 0.0),
    )


def test_lane_projection_is_right_positive_and_round_trips() -> None:
    lane = Lane(((0.0, 0.0), (100.0, 0.0), (100.0, 100.0)), width=3.0)
    assert lane.length == pytest.approx(200.0)
    # Heading +x, CARLA's right is +y.
    s, offset, yaw = lane.project(40.0, 1.5)
    assert (s, offset, yaw) == pytest.approx((40.0, 1.5, 0.0))
    x, y, yaw = lane.pose(150.0, offset=2.0)
    # Second leg heads +y; its right is -x.
    assert (x, y, yaw) == pytest.approx((98.0, 50.0, 90.0))
    assert lane.project(x, y)[:2] == pytest.approx((150.0, 2.0))
    straight = Lane.straight((10.0, 5.0), yaw=90.0, length=50.0)
    assert straight.pose(10.0)[:2] == pytest.approx((10.0, 15.0))
    with pytest.raises(ValueError):
        Lane(((0.0, 0.0), (0.0, 0.0)))


def test_lead_vehicle_braking_yields_gap_ttc_and_collision_event() -> None:
    scenario = lead_vehicle_braking(
        lead=LeadVehicle(gap=10.0, speed=10.0, brake_time=1.0, deceleration=5.0)
    )
    monitor = ScenarioMonitor(scenario)
    x, y, _ = scenario.start_pose()
    state = monitor.reset(_telemetry(x, y, speed=10.0))
    assert state["gap_m"] == pytest.approx(10.0)
    assert math.isinf(state["time_to_collision_s"])
    assert state["time_headway_s"] == pytest.approx(1.0)
    ego = x
    for _ in range(150):
        ego += 10.0 * 0.02
        state = monitor.update(_telemetry(ego, y, speed=10.0), 0.02)
    assert state["lead_speed_mps"] < 10.0
    assert state["time_to_collision_s"] < 10.0 or state["gap_m"] < 0
    for _ in range(200):
        ego += 10.0 * 0.02
        monitor.update(_telemetry(ego, y, speed=10.0), 0.02)
    kinds = [event["type"] for event in monitor.events]
    assert kinds[:2] == ["scenario_start", "lead_brake"]
    assert kinds.count("collision") == 1


def test_lead_reference_pose_matches_the_reported_bumper_gap() -> None:
    scenario = Scenario(lead=LeadVehicle(gap=10.0, speed=0.0, brake_time=None))
    monitor = ScenarioMonitor(scenario)
    ego_x, ego_y, _ = scenario.start_pose()
    state = monitor.reset(_telemetry(ego_x, ego_y))
    pose = monitor.lead_pose()
    assert pose is not None
    assert pose[0] - ego_x - VEHICLE_LENGTH == pytest.approx(state["gap_m"])


@pytest.mark.parametrize(
    "position, yaw, collision",
    [
        ((19.8, 3.5), 0.0, False),
        ((30.0, 0.0), 0.0, False),
        ((19.8, 3.0), 90.0, True),
        ((16.0, 0.0), 0.0, True),
        ((14.0, 0.0), 0.0, False),
    ],
)
def test_collision_uses_both_oriented_vehicle_footprints(
    position: tuple[float, float], yaw: float, collision: bool
) -> None:
    monitor = ScenarioMonitor(
        Scenario(lead=LeadVehicle(gap=10.0, speed=0.0, brake_time=None))
    )
    monitor.reset(_telemetry(5.0, 0.0))
    state = monitor.update(_telemetry(*position, yaw=yaw), 0.02)
    assert bool(state["collision"]) is collision
    assert any(event["type"] == "collision" for event in monitor.events) is collision


def test_adjacent_or_passed_lead_has_no_following_headway() -> None:
    monitor = ScenarioMonitor(
        Scenario(lead=LeadVehicle(gap=10.0, speed=0.0, brake_time=None))
    )
    for x, y in ((19.8, 3.5), (30.0, 0.0)):
        monitor.reset(_telemetry(5.0, 0.0))
        state = monitor.update(_telemetry(x, y, speed=10.0), 0.02)
        assert state["lead_present"] == 0.0
        assert math.isinf(state["time_to_collision_s"])
        assert math.isinf(state["time_headway_s"])


def test_lead_around_a_curve_remains_in_the_following_lane() -> None:
    scenario = Scenario(
        lane=Lane(((0.0, 0.0), (10.0, 0.0), (10.0, 100.0))),
        lead=LeadVehicle(gap=10.0, speed=0.0, brake_time=None),
    )
    monitor = ScenarioMonitor(scenario)
    state = monitor.reset(_telemetry(5.0, 0.0, speed=5.0))
    assert monitor.lead_pose() == pytest.approx((10.0, 9.8, 90.0))
    assert state["lead_present"] == 1.0
    assert state["gap_m"] == pytest.approx(10.0)
    assert state["time_to_collision_s"] == pytest.approx(2.0)
    assert state["time_headway_s"] == pytest.approx(2.0)
    assert state["collision"] == 0.0


def test_default_curved_sampler_keeps_its_same_lane_lead() -> None:
    import numpy as np

    from myosuite_carla.training import ScenarioSampler

    scenario = ScenarioSampler().sample(np.random.default_rng(2), "lead_brake")
    x, y, yaw = scenario.start_pose()
    state = ScenarioMonitor(scenario).reset(_telemetry(x, y, yaw, scenario.start_speed))
    assert state["gap_m"] == pytest.approx(35.35564351485701)
    assert state["lead_present"] == 1.0
    assert math.isfinite(state["time_headway_s"])


def test_custom_footprints_preserve_the_bumper_gap_and_touching_contact() -> None:
    ego = VehicleFootprint(length=6.0, width=2.0, center=(1.0, 0.0))
    lead = VehicleFootprint(length=8.0, width=2.5, center=(-1.0, 0.0))
    monitor = ScenarioMonitor(
        Scenario(
            ego_footprint=ego,
            lead=LeadVehicle(gap=10.0, speed=0.0, footprint=lead, brake_time=None),
        )
    )
    state = monitor.reset(_telemetry(5.0, 0.0))
    pose = monitor.lead_pose()
    assert pose is not None and pose[0] == pytest.approx(24.0)
    assert state["gap_m"] == pytest.approx(10.0)
    assert (
        state["lead_front_s_m"] - state["lead_length_m"] - state["ego_front_s_m"]
    ) == pytest.approx(state["gap_m"])
    assert lead.polygon(pose).bounds[0] - ego.polygon((5.0, 0.0, 0.0)).bounds[
        2
    ] == pytest.approx(10.0)
    contact = monitor.update(_telemetry(15.0, 0.0), 0.02)
    assert contact["gap_m"] == pytest.approx(0.0)
    assert contact["collision"] == 1.0


def test_local_box_rotation_and_center_follow_the_vehicle_heading() -> None:
    footprint = VehicleFootprint(length=6.0, width=2.0, center=(1.0, 0.0), yaw=90.0)
    assert footprint.polygon((10.0, 20.0, 90.0)).bounds == pytest.approx(
        (7.0, 20.0, 13.0, 22.0)
    )


def test_resolving_live_footprints_keeps_initial_gap_and_records_geometry() -> None:
    monitor = ScenarioMonitor(Scenario(lead=LeadVehicle(gap=10.0)))
    monitor.reset(_telemetry(5.0, 0.0))
    monitor.set_footprints(
        ego=VehicleFootprint(length=6.0),
        lead=VehicleFootprint(length=8.0, center=(0.5, 0.0)),
    )
    pose = monitor.lead_pose()
    assert pose is not None and pose[0] == pytest.approx(21.5)
    assert monitor.state["gap_m"] == pytest.approx(10.0)
    assert monitor.footprints["ego"]["length"] == 6.0
    assert monitor.footprints["lead"]["center"] == (0.5, 0.0)


def test_no_lead_has_no_collision_and_rotated_footprint_sets_road_boundary() -> None:
    monitor = ScenarioMonitor(Scenario())
    state = monitor.reset(_telemetry(5.0, 3.0, yaw=90.0))
    assert state["collision"] == state["lead_present"] == 0.0
    assert math.isinf(state["gap_m"])
    assert state["off_road"] == 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"length": 0.0},
        {"width": math.inf},
        {"center": (0.0, math.nan)},
        {"center": (0.0,)},
        {"yaw": math.inf},
    ],
)
def test_footprint_geometry_is_finite_and_positive(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        VehicleFootprint(**kwargs)


@pytest.mark.parametrize("source_group", ["", " ", 1])
def test_source_group_cannot_lose_its_declared_context(source_group) -> None:
    with pytest.raises(ValueError, match="source_group"):
        Scenario(source_group=source_group)


@pytest.mark.parametrize(
    "lead, expected_s, expected_speed",
    [
        (
            LeadVehicle(gap=10.0, speed=10.0, brake_time=0.5, deceleration=4.0),
            23.8,
            8.0,
        ),
        (
            LeadVehicle(gap=10.0, speed=10.0, profile=((0.0, 10.0), (0.5, 4.0))),
            23.3,
            4.0,
        ),
    ],
)
def test_lead_preview_preserves_state_and_matches_profile_or_braking_boundary(
    lead: LeadVehicle, expected_s: float, expected_speed: float
) -> None:
    monitor = ScenarioMonitor(Scenario(lead=lead))
    telemetry = _telemetry(5.0, 0.0)
    initial = monitor.reset(telemetry).copy()
    events = list(monitor.events)
    pose = monitor.preview_lead_pose(0.5)
    assert pose is not None and pose[0] == pytest.approx(expected_s)
    assert monitor.state == initial and monitor.events == events and monitor.time == 0.0
    state = monitor.update(telemetry, 0.5)
    assert monitor.lead_pose() == pose
    assert state["lead_speed_mps"] == expected_speed


@pytest.mark.parametrize("dt", [0.0, -1.0, math.nan, math.inf])
def test_lead_preview_requires_a_finite_positive_period(dt: float) -> None:
    with pytest.raises(ValueError, match="dt"):
        ScenarioMonitor(Scenario()).preview_lead_pose(dt)


def test_takeover_hands_over_on_input_or_starts_a_minimal_risk_stop() -> None:
    scenario = takeover_request(takeover=TakeoverRequest(time=1.0, budget=2.0))
    for inputs, expected in (
        ({"steer": 0.2, "throttle": 0.0, "brake": 0.0}, "driver_takeover"),
        ({"steer": 0.0, "throttle": 0.0, "brake": 0.0}, "minimal_risk_manoeuvre"),
    ):
        monitor = ScenarioMonitor(scenario)
        telemetry = _telemetry(*scenario.start_pose()[:2], speed=12.0)
        assert monitor.reset(telemetry)["automation"] == 1.0
        for _ in range(200):
            monitor.update(telemetry, 0.02, inputs)
        kinds = [event["type"] for event in monitor.events]
        assert "takeover_request" in kinds and expected in kinds
        if expected == "driver_takeover":
            takeover = monitor.events[kinds.index("driver_takeover")]
            assert takeover["reaction_time"] == pytest.approx(0.0, abs=0.03)
            assert not monitor.automation
        else:
            assert monitor.state["minimal_risk"] == 1.0
            stop = automation_command(monitor, telemetry)
            assert stop.brake > 0 and stop.throttle == 0


def test_lane_change_target_follows_a_raised_cosine() -> None:
    scenario = lane_change(
        lane_changes=(LaneChange(time=1.0, offset=-3.5, duration=2.0),)
    )
    monitor = ScenarioMonitor(scenario)
    telemetry = _telemetry(*scenario.start_pose()[:2])
    monitor.reset(telemetry)
    offsets = []
    for _ in range(200):
        offsets.append(monitor.update(telemetry, 0.02)["target_offset_m"])
    assert offsets[40] == pytest.approx(0.0, abs=1e-9)
    assert offsets[99] == pytest.approx(-1.75, abs=0.01)
    assert offsets[-1] == pytest.approx(-3.5)
    assert all(b <= a + 1e-12 for a, b in zip(offsets, offsets[1:]))
    # The ego is right of the new target, so its offset from it is positive.
    assert monitor.state["lane_offset_m"] == pytest.approx(3.5)


def test_automation_keeps_its_lane_and_follows_a_lead_on_the_surrogate() -> None:
    scenario = lead_vehicle_braking(start_speed=8.0, duration=30.0)
    scenario = Scenario(
        **{**scenario.__dict__, "lane": Lane(((0, 0), (400, 0), (400, 2), (900, 30)))}
    )
    x, y, yaw = scenario.start_pose()
    vehicle = SurrogateVehicle(location=(x, y + 0.5, 0.0), yaw=yaw, speed=8.0).start()
    monitor = ScenarioMonitor(scenario)
    monitor.reset(vehicle.snapshot())
    worst_offset, min_gap = 0.0, math.inf
    for _ in range(1500):
        command = automation_command(monitor, vehicle.snapshot())
        assert isinstance(command, VehicleCommand)
        state = monitor.update(vehicle.step(command), 0.02)
        if state["time"] > 3.0:
            worst_offset = max(worst_offset, abs(state["lane_offset_m"]))
        min_gap = min(min_gap, state["gap_m"])
    assert worst_offset < 0.5
    assert min_gap > 2.0
    assert state["speed_mps"] < 0.1


def test_idm_accelerates_on_free_road_and_brakes_when_close() -> None:
    assert idm_acceleration(0.0, 10.0, None, 0.0) == pytest.approx(1.0)
    assert idm_acceleration(10.0, 10.0, None, 0.0) == pytest.approx(0.0)
    assert idm_acceleration(10.0, 15.0, 8.0, 5.0) < -3.0
    # At the standstill gap the interaction term cancels the free term.
    assert idm_acceleration(0.0, 10.0, 2.0, 0.0) == pytest.approx(0.0)

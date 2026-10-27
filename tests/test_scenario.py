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

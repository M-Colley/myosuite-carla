"""Driving scenarios with a lane, a lead vehicle, lane changes and takeovers.

A :class:`Scenario` describes what happens around the ego vehicle; a
:class:`ScenarioMonitor` follows one run of it from vehicle telemetry and
reports lane geometry, the lead vehicle, the takeover state and timestamped
events. Nothing here imports CARLA: the lane is a polyline in CARLA's world
frame (left-handed, x forward at yaw 0, y right) and the lead vehicle moves
kinematically along it. :class:`~myosuite_carla.carla_scenario.CarlaLeadVehicle`
mirrors that lead vehicle as a CARLA actor.

Conventions: lateral offsets and angles are positive to the right of the lane
direction, matching CARLA's steer sign; distances along the lane are ``s``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray
from shapely import Polygon, box
from shapely.affinity import rotate, translate

from .controls import VehicleCommand

#: Default vehicle length, metres.
VEHICLE_LENGTH = 4.8


@dataclass(frozen=True)
class VehicleFootprint:
    """Planar vehicle bounding box relative to its telemetry reference point.

    Args:
        length: Box length in metres.
        width: Box width in metres.
        center: Box centre in local forward/right coordinates, metres.
        yaw: Box rotation relative to the vehicle heading, degrees.
    """

    length: float = VEHICLE_LENGTH
    width: float = 2.0
    center: tuple[float, float] = (0.0, 0.0)
    yaw: float = 0.0

    def __post_init__(self) -> None:
        for name in ("length", "width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        center = tuple(float(value) for value in self.center)
        if len(center) != 2 or not all(math.isfinite(value) for value in center):
            raise ValueError("center must contain two finite coordinates")
        if not math.isfinite(self.yaw):
            raise ValueError("yaw must be finite")
        object.__setattr__(self, "center", center)

    @cached_property
    def _local(self) -> Polygon:
        rectangle = box(
            -self.length / 2, -self.width / 2, self.length / 2, self.width / 2
        )
        return translate(rotate(rectangle, self.yaw, origin=(0, 0)), *self.center)

    def polygon(self, pose: tuple[float, float, float]) -> Polygon:
        """Return the footprint at a world ``(x, y, yaw_deg)`` pose."""
        x, y, yaw = pose
        return translate(rotate(self._local, yaw, origin=(0, 0)), x, y)


@dataclass(frozen=True)
class Lane:
    """A lane centre polyline with a constant width.

    Args:
        points: ``(N, 2)`` CARLA world ``x, y`` points along the travel direction.
        width: Lane width in metres.
    """

    points: tuple[tuple[float, float], ...]
    width: float = 3.5

    def __post_init__(self) -> None:
        points: NDArray[np.float64] = np.asarray(self.points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("points must be at least two (x, y) pairs")
        if not np.all(np.isfinite(points)):
            raise ValueError("points must be finite")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) <= 0):
            raise ValueError("points must not repeat")
        if not math.isfinite(self.width) or self.width <= 0:
            raise ValueError("width must be positive")
        object.__setattr__(
            self, "points", tuple((float(x), float(y)) for x, y in points)
        )

    @classmethod
    def straight(
        cls,
        start: Sequence[float] = (0.0, 0.0),
        yaw: float = 0.0,
        length: float = 3000.0,
        width: float = 3.5,
    ) -> Lane:
        """Return a straight lane from ``start`` along heading ``yaw`` degrees."""
        heading = math.radians(yaw)
        end = (
            start[0] + length * math.cos(heading),
            start[1] + length * math.sin(heading),
        )
        return cls(((float(start[0]), float(start[1])), end), width)

    @property
    def _arrays(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        points = np.asarray(self.points)
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return points, np.concatenate([[0.0], np.cumsum(lengths)])

    @property
    def length(self) -> float:
        """Return the polyline length in metres."""
        return float(self._arrays[1][-1])

    def pose(self, s: float, offset: float = 0.0) -> tuple[float, float, float]:
        """Return ``(x, y, yaw_deg)`` at distance ``s`` and a rightward offset."""
        points, cumulative = self._arrays
        s = float(np.clip(s, 0.0, cumulative[-1]))
        index = int(np.clip(np.searchsorted(cumulative, s) - 1, 0, len(points) - 2))
        direction = points[index + 1] - points[index]
        direction /= np.linalg.norm(direction)
        right = np.array([-direction[1], direction[0]])
        point = points[index] + (s - cumulative[index]) * direction + offset * right
        return (
            float(point[0]),
            float(point[1]),
            math.degrees(math.atan2(direction[1], direction[0])),
        )

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """Return ``(s, rightward offset, lane yaw_deg)`` of a world point."""
        points, cumulative = self._arrays
        starts, ends = points[:-1], points[1:]
        directions = ends - starts
        lengths = np.linalg.norm(directions, axis=1)
        units = directions / lengths[:, None]
        relative = np.array([x, y]) - starts
        along = np.clip(np.sum(relative * units, axis=1), 0.0, lengths)
        nearest = starts + along[:, None] * units
        index = int(np.argmin(np.linalg.norm(nearest - np.array([x, y]), axis=1)))
        unit = units[index]
        offset = float(unit[0] * relative[index][1] - unit[1] * relative[index][0])
        yaw = math.degrees(math.atan2(unit[1], unit[0]))
        return float(cumulative[index] + along[index]), offset, yaw


@dataclass(frozen=True)
class LeadVehicle:
    """A lead vehicle in the ego lane with a cruise-brake-cruise speed profile.

    Args:
        gap: Initial bumper-to-bumper gap ahead of the ego vehicle, metres.
        speed: Initial and cruising speed, m/s.
        brake_time: Scenario time at which it starts braking, s (None: never).
        deceleration: Braking deceleration, m/s^2.
        brake_to: Speed it brakes down to, m/s.
        resume_time: Time after which it accelerates back to ``speed``, s.
        acceleration: Re-acceleration, m/s^2.
        profile: ``(time, speed)`` samples of a recorded leader, e.g. from
            NGSIM. When given, the lead follows them by linear interpolation
            (holding the last speed) instead of the cruise-brake profile, and
            starts at the first sample's speed.
        footprint: Planar bounding box relative to the lead reference point.
    """

    gap: float = 30.0
    speed: float = 12.0
    brake_time: float | None = 8.0
    deceleration: float = 4.0
    brake_to: float = 0.0
    resume_time: float | None = None
    acceleration: float = 1.5
    profile: tuple[tuple[float, float], ...] = ()
    footprint: VehicleFootprint = field(default_factory=VehicleFootprint)

    def __post_init__(self) -> None:
        if not isinstance(self.footprint, VehicleFootprint):
            raise TypeError("footprint must be a VehicleFootprint")
        for name in ("gap", "speed", "deceleration", "brake_to", "acceleration"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.brake_to > self.speed:
            raise ValueError("brake_to must not exceed speed")
        profile = tuple((float(time), float(speed)) for time, speed in self.profile)
        times = np.asarray([time for time, _ in profile])
        speeds = np.asarray([speed for _, speed in profile])
        if profile and (
            not np.all(np.isfinite(times))
            or not np.all(np.isfinite(speeds))
            or np.any(speeds < 0)
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("profile needs increasing times and non-negative speeds")
        object.__setattr__(self, "profile", profile)

    def profile_speed(self, time: float) -> float:
        """Return the recorded speed at ``time``; requires a profile."""
        times, speeds = zip(*self.profile)
        return float(np.interp(time, times, speeds))


@dataclass(frozen=True)
class TakeoverRequest:
    """Automation drives until it requests the driver to take over.

    Args:
        time: Scenario time of the request, s.
        budget: Time the automation keeps driving after the request before it
            brakes to a standstill (a minimal-risk manoeuvre), s.
        modality: HMI modality label recorded with the event.
        steer_threshold: Measured wheel input that counts as a takeover.
        pedal_threshold: Measured pedal input that counts as a takeover.
    """

    time: float = 10.0
    budget: float = 7.0
    modality: str = "visual-auditory"
    steer_threshold: float = 0.05
    pedal_threshold: float = 0.1


@dataclass(frozen=True)
class LaneChange:
    """Move the target lane centre sideways from a given time.

    The target follows a raised-cosine path over ``duration`` seconds, a
    common lane-change reference, so drivers see a gradual far-point shift.

    Args:
        time: Scenario time of the instruction, s.
        offset: Rightward displacement of the target centre, metres (negative
            changes to the left lane).
        duration: Duration of the reference path, s (0 for a step).
    """

    time: float = 5.0
    offset: float = -3.5
    duration: float = 4.0


@dataclass(frozen=True)
class Scenario:
    """Everything around the ego vehicle for one run.

    Args:
        name: Label recorded with events.
        lane: Ego lane centre.
        start_s: Ego start distance along the lane, metres.
        start_speed: Ego start speed, m/s.
        target_speed: Desired speed given to drivers and automation, m/s.
        duration: Run length in seconds.
        lead: Optional lead vehicle.
        takeover: Optional automation-to-driver takeover.
        lane_changes: Instructions to move to an offset lane.
        road_half_width: Drivable half-width around the lane centre, metres;
            any footprint portion outside it counts as off the road.
        ego_footprint: Planar bounding box relative to the ego reference point.
        source_group: Shared source context for paired human and simulated runs.
    """

    name: str = "free"
    lane: Lane = field(default_factory=Lane.straight)
    start_s: float = 5.0
    start_speed: float = 0.0
    target_speed: float = 10.0
    duration: float = 30.0
    lead: LeadVehicle | None = None
    takeover: TakeoverRequest | None = None
    lane_changes: tuple[LaneChange, ...] = ()
    road_half_width: float = 5.25
    ego_footprint: VehicleFootprint = field(default_factory=VehicleFootprint)
    source_group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ego_footprint, VehicleFootprint):
            raise TypeError("ego_footprint must be a VehicleFootprint")
        if self.source_group is not None and (
            not isinstance(self.source_group, str) or not self.source_group.strip()
        ):
            raise ValueError("source_group must be a nonempty string or None")

    def start_pose(self) -> tuple[float, float, float]:
        """Return the ego start ``(x, y, yaw_deg)``."""
        return self.lane.pose(self.start_s)


def lead_vehicle_braking(**overrides: Any) -> Scenario:
    """A lead vehicle 30 m ahead at 12 m/s brakes at 4 m/s^2 to a stop at 8 s."""
    return replace(
        Scenario(
            name="lead_vehicle_braking",
            start_speed=12.0,
            target_speed=12.0,
            duration=20.0,
            lead=LeadVehicle(),
        ),
        **overrides,
    )


def takeover_request(**overrides: Any) -> Scenario:
    """Automation cruises at 12 m/s and requests a takeover at 10 s."""
    return replace(
        Scenario(
            name="takeover_request",
            start_speed=12.0,
            target_speed=12.0,
            duration=25.0,
            takeover=TakeoverRequest(),
        ),
        **overrides,
    )


def lane_change(**overrides: Any) -> Scenario:
    """Change to the left lane at 5 s while cruising at 10 m/s."""
    return replace(
        Scenario(
            name="lane_change",
            start_speed=10.0,
            target_speed=10.0,
            duration=15.0,
            lane_changes=(LaneChange(),),
        ),
        **overrides,
    )


SCENARIOS: dict[str, Callable[..., Scenario]] = {
    "free": Scenario,
    "lead_vehicle_braking": lead_vehicle_braking,
    "takeover_request": takeover_request,
    "lane_change": lane_change,
}


def _wrap(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class ScenarioMonitor:
    """Follow one run of a scenario from vehicle telemetry.

    Call :meth:`reset` with the start telemetry, then :meth:`update` once per
    frame with the new telemetry and the measured driver inputs. Events are
    appended to :attr:`events` as dictionaries with ``time`` and ``type``.
    Collision reports sampled planar contact, with a 1 nm numerical tolerance.
    The signed ``gap_m`` uses lane coordinates and projected bumper extents.
    ``lead_present`` requires an ahead box with overlapping lateral lane extents.
    Headway and TTC are infinite when that condition does not hold.
    ``ego_front_s_m`` and ``lead_front_s_m`` add projected front extents to
    reference arclengths. ``lead_length_m`` is the projected box length.
    """

    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        self.ego_footprint = scenario.ego_footprint
        self.lead_footprint = (
            scenario.lead.footprint if scenario.lead else VehicleFootprint()
        )
        self._telemetry: Any = None
        self.events: list[dict[str, Any]] = []
        self.time = 0.0
        self._lead_s = 0.0
        self._lead_speed = 0.0
        self._target_offset = 0.0
        self.automation = scenario.takeover is not None
        self._requested = False
        self._minimal_risk = False
        self.state: dict[str, float] = {}

    @property
    def footprints(self) -> dict[str, dict[str, Any]]:
        """Return the resolved footprints for recording and replay."""
        return {"ego": asdict(self.ego_footprint), "lead": asdict(self.lead_footprint)}

    def set_footprints(
        self,
        *,
        ego: VehicleFootprint | None = None,
        lead: VehicleFootprint | None = None,
    ) -> None:
        """Resolve live geometry, preserving the initial gap at scenario time zero."""
        for name, footprint in (("ego_footprint", ego), ("lead_footprint", lead)):
            if footprint is not None:
                if not isinstance(footprint, VehicleFootprint):
                    raise TypeError("footprints must be VehicleFootprint instances")
                setattr(self, name, footprint)
        if self._telemetry is not None:
            if self.time == 0.0 and self.scenario.lead is not None:
                self._place_lead(self._telemetry)
            self.state = self._measure(self._telemetry)

    def _place_lead(self, telemetry: Any) -> None:
        lead = self.scenario.lead
        if lead is None:
            return
        s, _, lane_yaw = self.scenario.lane.project(*telemetry.location[:2])
        front = self.ego_footprint.polygon(
            (0.0, 0.0, telemetry.rotation[1] - lane_yaw)
        ).bounds[2]
        rear = -self.lead_footprint._local.bounds[0]
        self._lead_s = s + front + rear + lead.gap

    def _event(self, kind: str, **details: Any) -> None:
        self.events.append({"time": round(self.time, 6), "type": kind, **details})

    def reset(self, telemetry: Any) -> dict[str, float]:
        """Start a run at scenario time zero from the ego telemetry."""
        scenario = self.scenario
        self.events = []
        self.time = 0.0
        self.automation = scenario.takeover is not None
        self._requested = self._minimal_risk = False
        self._target_offset = 0.0
        if scenario.lead is not None:
            self._place_lead(telemetry)
            self._lead_speed = (
                scenario.lead.profile_speed(0.0)
                if scenario.lead.profile
                else scenario.lead.speed
            )
        self._event("scenario_start", scenario=scenario.name)
        if self.automation:
            self._event("automation_engaged")
        self._telemetry = telemetry
        self.state = self._measure(telemetry)
        return self.state

    def _advance_lead(self, dt: float) -> None:
        lead = self.scenario.lead
        if lead is None:
            return
        if lead.profile:
            speed = lead.profile_speed(self.time)
            self._lead_s += 0.5 * (self._lead_speed + speed) * dt
            self._lead_speed = speed
            return
        braking = lead.brake_time is not None and self.time >= lead.brake_time
        resumed = lead.resume_time is not None and self.time >= lead.resume_time
        if braking and not resumed:
            if self._lead_speed > lead.brake_to:
                if not any(event["type"] == "lead_brake" for event in self.events):
                    self._event("lead_brake", deceleration=lead.deceleration)
            target, rate = lead.brake_to, lead.deceleration
        else:
            target, rate = lead.speed, lead.acceleration
        step = rate * dt
        self._lead_speed += float(np.clip(target - self._lead_speed, -step, step))
        self._lead_s += self._lead_speed * dt

    def update(
        self, telemetry: Any, dt: float, inputs: dict[str, float] | None = None
    ) -> dict[str, float]:
        """Advance scenario time by ``dt`` and measure the new frame."""
        self.time += dt
        scenario = self.scenario
        self._advance_lead(dt)
        target = 0.0
        for change in scenario.lane_changes:
            if self.time - dt < change.time <= self.time:
                self._event("lane_change_request", offset=change.offset)
            progress = (self.time - change.time) / max(change.duration, 1e-9)
            progress = min(max(progress, 0.0), 1.0)
            target += change.offset * (0.5 - 0.5 * math.cos(math.pi * progress))
        self._target_offset = target
        takeover = scenario.takeover
        if takeover is not None:
            if not self._requested and self.time >= takeover.time:
                self._requested = True
                self._event("takeover_request", modality=takeover.modality)
            if self.automation and self._requested and inputs is not None:
                pedal = max(inputs.get("throttle", 0.0), inputs.get("brake", 0.0))
                if (
                    abs(inputs.get("steer", 0.0)) > takeover.steer_threshold
                    or pedal > takeover.pedal_threshold
                ):
                    self.automation = False
                    self._event(
                        "driver_takeover",
                        reaction_time=self.time - takeover.time,
                        input="steer"
                        if abs(inputs.get("steer", 0.0)) > takeover.steer_threshold
                        else "pedal",
                    )
            if (
                self.automation
                and self._requested
                and not self._minimal_risk
                and self.time >= takeover.time + takeover.budget
            ):
                self._minimal_risk = True
                self._event("minimal_risk_manoeuvre")
        self._telemetry = telemetry
        self.state = self._measure(telemetry)
        if self.state["collision"] and not any(
            event["type"] == "collision" for event in self.events
        ):
            self._event("collision", other="lead")
        return self.state

    def _measure(self, telemetry: Any) -> dict[str, float]:
        scenario = self.scenario
        x, y = telemetry.location[:2]
        s, offset, lane_yaw = scenario.lane.project(x, y)
        yaw = telemetry.rotation[1]
        velocity = np.asarray(telemetry.velocity, dtype=float)
        heading = math.radians(yaw)
        speed = float(velocity[0] * math.cos(heading) + velocity[1] * math.sin(heading))
        local_bounds = self.ego_footprint.polygon((0.0, 0.0, yaw - lane_yaw)).bounds
        state = {
            "time": self.time,
            "s_m": s,
            "ego_front_s_m": s + local_bounds[2],
            "lane_offset_m": offset - self._target_offset,
            "heading_error_deg": _wrap(yaw - lane_yaw),
            "speed_mps": speed,
            "target_speed_mps": scenario.target_speed,
            "target_offset_m": self._target_offset,
            "off_road": float(
                offset + local_bounds[1] < -scenario.road_half_width
                or offset + local_bounds[3] > scenario.road_half_width
            ),
            "lead_present": 0.0,
            "lead_exists": float(scenario.lead is not None),
            "collision": 0.0,
            "gap_m": math.inf,
            "lead_speed_mps": 0.0,
            "lead_s_m": math.inf,
            "lead_front_s_m": math.inf,
            "lead_length_m": 0.0,
            "time_to_collision_s": math.inf,
            "time_headway_s": math.inf,
            "automation": float(self.automation),
            "takeover_requested": float(self._requested),
            "minimal_risk": float(self._minimal_risk),
        }
        if scenario.lead is not None:
            pose = self.lead_pose()
            assert pose is not None
            ego_shape = self.ego_footprint.polygon((x, y, yaw))
            lead_shape = self.lead_footprint.polygon(pose)
            lead_local_bounds = self.lead_footprint._local.bounds
            in_path = (
                lead_local_bounds[1] <= offset + local_bounds[3] + 1e-9
                and offset + local_bounds[1] <= lead_local_bounds[3] + 1e-9
                and self._lead_s + 0.5 * sum(lead_local_bounds[::2])
                >= s + 0.5 * sum(local_bounds[::2])
            )
            front = local_bounds[2]
            rear = -lead_local_bounds[0]
            gap = self._lead_s - s - front - rear
            closing = speed - self._lead_speed
            state.update(
                lead_present=float(in_path),
                # One nanometre admits contact despite coordinate roundoff.
                collision=float(ego_shape.distance(lead_shape) <= 1e-9),
                gap_m=gap,
                lead_speed_mps=self._lead_speed,
                lead_s_m=self._lead_s,
                lead_front_s_m=self._lead_s + lead_local_bounds[2],
                lead_length_m=lead_local_bounds[2] - lead_local_bounds[0],
                time_to_collision_s=max(gap, 0.0) / closing
                if in_path and closing > 1e-6
                else math.inf,
                time_headway_s=max(gap, 0.0) / speed
                if in_path and speed > 0.1
                else math.inf,
            )
        return state

    def lead_pose(self) -> tuple[float, float, float] | None:
        """Return the lead vehicle's ``(x, y, yaw_deg)`` reference point."""
        if self.scenario.lead is None:
            return None
        return self.scenario.lane.pose(self._lead_s)

    def preview_lead_pose(self, dt: float) -> tuple[float, float, float] | None:
        """Predict the next lead pose without advancing scenario state or events."""
        if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        future = copy(self)
        future.events = list(self.events)
        future.time += dt
        future._advance_lead(dt)
        return future.lead_pose()

    def view_angle(self, telemetry: Any, distance: float) -> float:
        """Angle in radians to the target lane centre ``distance`` m ahead.

        Positive angles lie to the right of the vehicle's heading, as used by
        two-point visual steering models.
        """
        x, y = telemetry.location[:2]
        s, _, _ = self.scenario.lane.project(x, y)
        px, py, _ = self.scenario.lane.pose(s + distance, self._target_offset)
        heading = math.radians(telemetry.rotation[1])
        forward = (px - x) * math.cos(heading) + (py - y) * math.sin(heading)
        right = -(px - x) * math.sin(heading) + (py - y) * math.cos(heading)
        return math.atan2(right, max(forward, 1e-6))


@dataclass(frozen=True)
class AutomationConfig:
    """Gains of the scenario automation (a lane-centring cruise control).

    Args:
        steer_gain: Steer per radian of far-point angle.
        preview: Far-point preview time, s.
        speed_gain: Throttle or brake per m/s of speed error.
        time_gap: Desired time gap to a lead vehicle, s.
        min_gap: Standstill gap to a lead vehicle, m.
    """

    steer_gain: float = 1.5
    preview: float = 1.2
    speed_gain: float = 0.25
    time_gap: float = 1.8
    min_gap: float = 5.0


def automation_command(
    monitor: ScenarioMonitor,
    telemetry: Any,
    config: AutomationConfig = AutomationConfig(),
) -> VehicleCommand:
    """Return the automation's vehicle command for the current frame."""
    state = monitor.state
    speed = state["speed_mps"]
    if state["minimal_risk"]:
        return VehicleCommand(brake=0.6 if speed > 0.05 else 1.0)
    preview = max(8.0, speed * config.preview)
    steer = float(
        np.clip(config.steer_gain * monitor.view_angle(telemetry, preview), -1, 1)
    )
    desired = state["target_speed_mps"]
    if state["lead_present"]:
        desired = min(
            desired, max(0.0, (state["gap_m"] - config.min_gap) / config.time_gap)
        )
    error = desired - speed
    if error >= 0:
        return VehicleCommand(
            throttle=float(np.clip(0.3 + config.speed_gain * error, 0, 1)), steer=steer
        )
    return VehicleCommand(
        brake=float(np.clip(-config.speed_gain * error, 0, 1)), steer=steer
    )

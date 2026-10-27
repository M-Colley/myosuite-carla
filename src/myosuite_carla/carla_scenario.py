"""Mirror scenarios in a live CARLA world without importing CARLA eagerly.

:func:`lane_from_carla` follows map waypoints into a :class:`~.scenario.Lane`,
and :class:`CarlaLeadVehicle` is a ``CarlaBridge`` control hook that moves a
physics-free CARLA actor to the scenario's lead-vehicle pose before every
tick, so cameras, sensors and other clients see the same lead vehicle.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

from .controls import VehicleCommand
from .scenario import Lane, Scenario, ScenarioMonitor


def lane_from_carla(
    world_map: Any, transform: Any, length: float = 400.0, step: float = 2.0
) -> Lane:
    """Return the lane that starts at a CARLA transform, following waypoints.

    At junctions the first successor is taken. The width is the start lane's.
    """
    for name, value in (("length", length), ("step", step)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    waypoint = world_map.get_waypoint(transform.location)
    points = [(waypoint.transform.location.x, waypoint.transform.location.y)]
    travelled = 0.0
    while travelled < length:
        following = waypoint.next(step)
        if not following:
            break
        waypoint = following[0]
        point = (waypoint.transform.location.x, waypoint.transform.location.y)
        if point != points[-1]:
            points.append(point)
        travelled += step
    return Lane(
        tuple(points), float(world_map.get_waypoint(transform.location).lane_width)
    )


class CarlaLeadVehicle:
    """Kinematic CARLA actor that follows a monitor's lead vehicle.

    Args:
        monitor: Scenario monitor whose lead pose is mirrored.
        world: CARLA world.
        blueprint: Vehicle blueprint id for the lead.
        height: Reference height above the lane, metres.
        api: Optional CARLA module for dependency injection.
    """

    def __init__(
        self,
        monitor: ScenarioMonitor,
        world: Any,
        blueprint: str = "vehicle.lincoln.mkz",
        height: float = 0.3,
        api: Any | None = None,
    ) -> None:
        self.monitor = monitor
        self.world = world
        self.blueprint = blueprint
        self.height = float(height)
        self.api = api
        self.actor: Any = None
        self._z = 0.0

    def spawn(self, ego_z: float) -> None:
        """Spawn the lead at its current scenario pose, physics disabled."""
        pose = self.monitor.lead_pose()
        if pose is None:
            return
        api = self.api or importlib.import_module("carla")
        self.api = api
        self._z = float(ego_z) + self.height
        blueprint = self.world.get_blueprint_library().find(self.blueprint)
        self.actor = self.world.spawn_actor(blueprint, self._transform(pose))
        self.actor.set_simulate_physics(False)

    def _transform(self, pose: tuple[float, float, float]) -> Any:
        x, y, yaw = pose
        api = self.api
        if api is None:
            raise RuntimeError("Spawn the lead vehicle before moving it")
        return api.Transform(api.Location(x=x, y=y, z=self._z), api.Rotation(yaw=yaw))

    def __call__(self, session: Any, sample: Any) -> None:
        """Bridge hook: place the lead before CARLA ticks."""
        pose = self.monitor.lead_pose()
        if self.actor is not None and pose is not None:
            self.actor.set_transform(self._transform(pose))

    def destroy(self) -> None:
        """Remove the lead actor if one was spawned."""
        if self.actor is not None:
            self.actor.destroy()
            self.actor = None


def teleport(session: Any, scenario: Scenario, *, settle_frames: int = 50) -> None:
    """Place a started session's vehicle at the scenario start.

    The vehicle is moved to the lane start just above the road, held with the
    brakes for ``settle_frames`` ticks and then given the scenario's start
    speed along its heading. One more tick applies that speed here, because
    CARLA reports the velocity jump as a one-frame acceleration of hundreds of
    m/s^2 that must not reach the body as a vehicle load.
    """
    api = getattr(session, "_api", None) or importlib.import_module("carla")
    vehicle = session.vehicle
    x, y, yaw = scenario.start_pose()
    probe = api.Location(x=x, y=y, z=vehicle.get_location().z)
    ground = session.world.get_map().get_waypoint(probe).transform.location.z
    vehicle.set_transform(
        api.Transform(api.Location(x=x, y=y, z=ground + 0.3), api.Rotation(yaw=yaw))
    )
    vehicle.set_target_velocity(api.Vector3D(0.0, 0.0, 0.0))
    vehicle.set_target_angular_velocity(api.Vector3D(0.0, 0.0, 0.0))
    for _ in range(settle_frames):
        session.step(VehicleCommand(brake=1.0))
    if scenario.start_speed:
        heading = math.radians(yaw)
        vehicle.set_target_velocity(
            api.Vector3D(
                scenario.start_speed * math.cos(heading),
                scenario.start_speed * math.sin(heading),
                0.0,
            )
        )
        session.step(VehicleCommand())

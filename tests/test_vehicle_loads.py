"""Frame conversions and d'Alembert loads from CARLA vehicle motion."""

from __future__ import annotations

import math
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite_carla.vehicle_loads import (
    InertialLoads,
    VehicleMotion,
    carla_axes,
    carla_rate_to_cockpit,
    carla_vector_to_cockpit,
)


def test_carla_axes_match_unreal_forward_right_up() -> None:
    np.testing.assert_allclose(carla_axes((0, 0, 0)), np.eye(3), atol=1e-12)
    # Yaw 90 degrees: forward is world +y, right is world -x (left-handed).
    np.testing.assert_allclose(
        carla_axes((0, 90, 0)), [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-12
    )
    # Pitch up 90 degrees: forward points up.
    np.testing.assert_allclose(carla_axes((90, 0, 0))[0], [0, 0, 1], atol=1e-12)
    axes = carla_axes((7.0, -33.0, 12.0))
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)


def test_cockpit_frame_is_left_backward_up_and_right_handed() -> None:
    np.testing.assert_allclose(carla_vector_to_cockpit((1, 0, 0)), [0, -1, 0])
    np.testing.assert_allclose(carla_vector_to_cockpit((0, 1, 0)), [-1, 0, 0])
    np.testing.assert_allclose(carla_vector_to_cockpit((0, 0, 1)), [0, 0, 1])
    left, backward, up = (np.eye(3)[i] for i in range(3))
    np.testing.assert_allclose(np.cross(left, backward), up)
    # A right turn is a positive left-handed yaw rate; in the cockpit the
    # forward axis (-y) must rotate toward the right (-x).
    omega = carla_rate_to_cockpit((0.0, 0.0, 1.0))
    forward = np.array([0.0, -1.0, 0.0])
    assert np.cross(omega, forward)[0] < 0


def test_motion_in_cockpit_converts_gravity_braking_and_turning() -> None:
    level = VehicleMotion().in_cockpit()
    np.testing.assert_allclose(level["gravity"], [0, 0, -9.81])
    braking = VehicleMotion(acceleration=(-8.0, 0.0, 0.0), rotation=(0, 135, 0))
    # A vehicle heading 135 degrees braking along its own axis: world vector.
    yaw = math.radians(135)
    world = (-8 * math.cos(yaw), -8 * math.sin(yaw), 0.0)
    turned = VehicleMotion(acceleration=world, rotation=(0, 135, 0)).in_cockpit()
    np.testing.assert_allclose(turned["acceleration"], [0, 8.0, 0], atol=1e-9)
    assert braking.in_cockpit()["lateral_right_acceleration"].shape == (1,)
    rate = VehicleMotion(angular_velocity=(0, 0, 90)).in_cockpit()["angular_velocity"]
    np.testing.assert_allclose(rate, [0, 0, -math.pi / 2])
    with pytest.raises(ValueError):
        VehicleMotion(acceleration=(np.nan, 0, 0))


def _two_bodies() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="a" pos="1 0 0"><geom size=".1" mass="2"/></body>'
        '<body name="b" pos="0 2 0"><geom size=".1" mass="3"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_inertial_loads_follow_dalembert() -> None:
    model, data = _two_bodies()
    loads = InertialLoads(model, [1, 2], origin=(0, 0, 0))
    braking = VehicleMotion(acceleration=(-4.0, 0.0, 0.0))
    forces = loads.forces(data, braking)
    # Braking pushes bodies forward (cockpit -y) by m * 4, plus gravity.
    np.testing.assert_allclose(forces[:, 1], [-8.0, -12.0])
    np.testing.assert_allclose(forces[:, 2], [-2 * 9.81, -3 * 9.81])
    spin = VehicleMotion(angular_velocity=(0.0, 0.0, math.degrees(2.0)))
    centrifugal = loads.forces(data, spin)
    # omega x (omega x r) is centripetal; its d'Alembert load points outward.
    np.testing.assert_allclose(centrifugal[0, :2], [2 * 4.0 * 1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(centrifugal[1, :2], [0.0, 3 * 4.0 * 2.0], atol=1e-9)
    loads.apply(data, braking)
    assert not np.any(data.xfrc_applied[0])
    np.testing.assert_allclose(data.xfrc_applied[1:, :3], forces)


def test_motion_from_telemetry_differentiates_rate() -> None:
    before = SimpleNamespace(
        acceleration=(0, 0, 0), angular_velocity=(0, 0, 10), rotation=(0, 0, 0)
    )
    now = SimpleNamespace(
        acceleration=(1, 2, 3), angular_velocity=(0, 0, 12), rotation=(1, 2, 3)
    )
    motion = VehicleMotion.from_telemetry(now, before, 0.02)
    assert motion.angular_acceleration == pytest.approx((0, 0, 100))
    assert VehicleMotion.from_telemetry(now).angular_acceleration == (0, 0, 0)
    with pytest.raises(ValueError):
        VehicleMotion.from_telemetry(now, before, 0.0)


def test_acceleration_at_includes_tangential_centripetal_and_coriolis_terms() -> None:
    motion = VehicleMotion(
        acceleration=(1.0, 2.0, 3.0),
        angular_velocity=(0.0, 0.0, math.degrees(2.0)),
        angular_acceleration=(0.0, 0.0, math.degrees(3.0)),
    )
    # In cockpit axes omega=-2z, alpha=-3z and the point is at +x.
    # Tangential=-3y, centripetal=-4x and Coriolis=+8x for v_rel=+2y.
    acceleration = motion.acceleration_at(
        (1.0, 0.0, 0.0),
        relative_velocity=(0.0, 2.0, 0.0),
        relative_acceleration=(0.5, 1.0, -1.0),
    )
    np.testing.assert_allclose(acceleration, [2.5, -3.0, 2.0], atol=1e-12)


def test_acceleration_at_projects_pitched_and_rolled_world_motion() -> None:
    axes = carla_axes((24.0, -37.0, 18.0))
    forward_acceleration = axes.T @ np.array([4.0, 0.0, 0.0])
    yaw_rate = axes.T @ np.array([0.0, 0.0, math.degrees(2.0)])
    roll_acceleration = axes.T @ np.array([math.degrees(3.0), 0.0, 0.0])
    motion = VehicleMotion(
        acceleration=tuple(forward_acceleration),
        angular_velocity=tuple(yaw_rate),
        angular_acceleration=tuple(roll_acceleration),
        rotation=(24.0, -37.0, 18.0),
    )
    # alpha=+3y and omega=-2z in the right-handed cockpit.
    np.testing.assert_allclose(
        motion.acceleration_at((1.0, 0.0, 1.0)), [-1.0, -4.0, -3.0], atol=1e-12
    )


@pytest.mark.parametrize(
    "field", ["offset", "relative_velocity", "relative_acceleration"]
)
def test_acceleration_at_requires_finite_vectors(field: str) -> None:
    arguments = {"offset": (0.0, 0.0, 0.0), field: (math.nan, 0.0, 0.0)}
    with pytest.raises(ValueError, match=field):
        VehicleMotion().acceleration_at(**arguments)

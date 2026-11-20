"""Tests for exporting CARLA VehicleMotion and NGSIM trials to base-profile arrays."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from myosuite_carla import motions_to_base_profile_kwargs
from myosuite_carla.ngsim import FollowingTrial
from myosuite_carla.vehicle_loads import InertialLoads, VehicleMotion


def test_vehicle_motion_base_kinematics_matches_inertial_loads() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody>"
        '<body name="seg" pos="0.4 -0.3 0.6"><geom size=".08" mass="3.5"/></body>'
        "</worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    motion = VehicleMotion(
        acceleration=(-4.5, 1.8, 0.5),
        angular_velocity=(3.0, -5.0, 25.0),
        rotation=(2.0, 40.0, -3.0),
        angular_acceleration=(-8.0, 12.0, -40.0),
    )
    loads = InertialLoads(model, [1], origin=(0.0, 0.0, 0.0))
    total_force = loads.forces(data, motion)[0]
    gravity_force = 3.5 * motion.in_cockpit()["gravity"]
    fict_force_expected = total_force - gravity_force

    kin = motion.as_base_kinematics_dict(frame="cockpit")
    r = data.xipos[1]
    a_fict = -(
        kin["lin_acc"]
        + np.cross(kin["ang_acc"], r)
        + np.cross(kin["ang_vel"], np.cross(kin["ang_vel"], r))
    )
    np.testing.assert_allclose(3.5 * a_fict, fict_force_expected, atol=1e-11)

    flu = motion.as_base_kinematics_dict(frame="flu")
    # Forward CARLA acceleration (+x) at zero attitude maps to +x FLU and -y cockpit.
    fwd_motion = VehicleMotion(acceleration=(3.0, 0.0, 0.0))
    np.testing.assert_allclose(
        fwd_motion.as_base_kinematics_dict(frame="cockpit")["lin_acc"],
        [0.0, -3.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        fwd_motion.as_base_kinematics_dict(frame="flu")["lin_acc"],
        [3.0, 0.0, 0.0],
        atol=1e-12,
    )
    assert np.linalg.norm(flu["ang_vel"]) == pytest.approx(
        np.linalg.norm(kin["ang_vel"])
    )

    with pytest.raises(ValueError, match="frame"):
        motion.as_base_kinematics_dict(frame="ned")


def test_motions_to_base_profile_kwargs_differentiates_angular_rate() -> None:
    dt = 0.05
    # Ramping yaw rate of 20 deg/s^2 in CARLA -> -radians(20) rad/s^2 in cockpit +z.
    motions = [
        VehicleMotion(
            acceleration=(-2.0, 0.0, 0.0),
            angular_velocity=(0.0, 0.0, 20.0 * (k * dt)),
        )
        for k in range(5)
    ]
    exported = motions_to_base_profile_kwargs(motions, dt=dt, frame="cockpit")
    assert set(exported) == {"times", "lin_acc", "ang_vel", "ang_acc"}
    np.testing.assert_allclose(exported["times"], np.arange(5) * dt)
    np.testing.assert_allclose(exported["lin_acc"][:, 1], np.full(5, 2.0))
    np.testing.assert_allclose(
        exported["ang_acc"][:, 2],
        np.full(5, -math.radians(20.0)),
        atol=1e-11,
    )

    with_offset_times = motions_to_base_profile_kwargs(
        motions, times=[10.0 + k * dt for k in range(5)], frame="flu"
    )
    np.testing.assert_allclose(with_offset_times["times"], np.arange(5) * dt)
    np.testing.assert_allclose(with_offset_times["lin_acc"][:, 0], np.full(5, -2.0))

    with pytest.raises(ValueError, match="at least 2"):
        motions_to_base_profile_kwargs([motions[0]], dt=dt)
    with pytest.raises(ValueError, match="exactly one of dt or times"):
        motions_to_base_profile_kwargs(motions, dt=dt, times=[0.0, 0.1, 0.2, 0.3, 0.4])


def test_following_trial_as_base_profile_kwargs() -> None:
    t = np.array([100.0, 100.1, 100.2, 100.3])
    acc = np.array([0.5, -1.2, -2.5, 0.0])
    trial = FollowingTrial(
        driver="track_1",
        time=t,
        position=np.array([0.0, 1.0, 2.0, 3.0]),
        speed=np.array([10.0, 9.9, 9.7, 9.6]),
        acceleration=acc,
        gap=np.array([20.0, 20.1, 20.3, 20.4]),
        leader_speed=np.full(4, 10.0),
    )

    cockpit_kw = trial.as_base_profile_kwargs(
        frame="cockpit",
        lateral_left_acc=[0.1, 0.2, 0.1, 0.0],
    )
    np.testing.assert_allclose(cockpit_kw["times"], [0.0, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(cockpit_kw["lin_acc"][:, 0], [0.1, 0.2, 0.1, 0.0])
    np.testing.assert_allclose(cockpit_kw["lin_acc"][:, 1], -acc)
    np.testing.assert_allclose(cockpit_kw["ang_vel"], np.zeros((4, 3)))

    flu_kw = trial.as_base_profile_kwargs(frame="flu")
    np.testing.assert_allclose(flu_kw["lin_acc"][:, 0], acc)
    np.testing.assert_allclose(flu_kw["lin_acc"][:, 1:], np.zeros((4, 2)))

    with pytest.raises(ValueError, match="frame"):
        trial.as_base_profile_kwargs(frame="body")

"""Physical cockpit composition, calibration, and real MuJoCo contacts."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from myosuite_carla.cockpit import (
    CockpitConfig,
    CockpitNames,
    PedalConfig,
    SteeringConfig,
    add_cockpit,
)


@pytest.fixture
def mujoco_module() -> ModuleType:
    return pytest.importorskip("mujoco")


@pytest.fixture
def builder_class() -> type[Any]:
    return pytest.importorskip("myosuite.core.model_builder").ModelBuilder


@pytest.mark.parametrize(
    "config, kwargs",
    [
        (PedalConfig, {"travel": 0}),
        (PedalConfig, {"stiffness": -1}),
        (PedalConfig, {"mass": float("nan")}),
        (PedalConfig, {"half_size": (0.1, 0.0, 0.1)}),
        (PedalConfig, {"friction": (-1.0, 0.0, 0.0)}),
        (PedalConfig, {"quaternion": (2.0, 0.0, 0.0, 0.0)}),
        (SteeringConfig, {"radius": -1}),
        (SteeringConfig, {"grip_radius": 0.2}),
        (SteeringConfig, {"angle_limit": float("inf")}),
        (SteeringConfig, {"segments": 7}),
        (SteeringConfig, {"segments": 8.5}),
        (SteeringConfig, {"segments": True}),
        (CockpitConfig, {"prefix": ""}),
        (CockpitConfig, {"prefix": "has space"}),
        (CockpitConfig, {"parent_body": ""}),
        (CockpitConfig, {"position": (0.0, float("inf"), 0.0)}),
    ],
)
def test_invalid_configuration(config: type[Any], kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        config(**kwargs)


def test_names_are_stable_and_validate_controls() -> None:
    names = CockpitNames("driver_")
    assert names.throttle_joint == "driver_throttle_joint"
    assert names.brake_position == "driver_brake_position"
    assert names.steering_position == "driver_steering_position"
    assert names.body("steering") == "driver_steering_body"
    assert names.touch_sensor("throttle") == "driver_throttle_touch"
    with pytest.raises(ValueError, match="control must"):
        names.body("typo")


@pytest.mark.parametrize("degree", [True, False])
def test_composition_has_passive_joints_and_calibrated_sensors(
    builder_class: type[Any],
    mujoco_module: ModuleType,
    degree: bool,
) -> None:
    mujoco = mujoco_module
    config = CockpitConfig(steering=SteeringConfig(angle_limit=2.25))
    source = mujoco.MjSpec()
    source.compiler.degree = degree
    source.option.timestep = 0.001
    builder = builder_class.from_spec(source)
    assert add_cockpit(builder, config) is builder
    model, spec = builder.build()
    assert model.nq == 3
    assert model.nu == 0
    assert model.nsensor == 9
    assert model.nmesh == 0
    assert model.opt.timestep == 0.001
    assert spec.compiler.degree == degree
    np.testing.assert_allclose(
        model.joint(config.names.throttle_joint).range, (0, 0.06)
    )
    np.testing.assert_allclose(
        model.joint(config.names.steering_joint).range, (-2.25, 2.25)
    )
    data = mujoco.MjData(model)
    for name, value in (
        (config.names.throttle_joint, 0.03),
        (config.names.brake_joint, 0.04),
        (config.names.steering_joint, -0.8),
    ):
        data.joint(name).qpos[0] = value
    mujoco.mj_forward(model, data)
    assert data.sensor(config.names.throttle_position).data[0] == pytest.approx(0.03)
    assert data.sensor(config.names.brake_position).data[0] == pytest.approx(0.04)
    assert data.sensor(config.names.steering_position).data[0] == pytest.approx(-0.8)


def test_placement_parent_and_repeated_builds(
    builder_class: type[Any],
    mujoco_module: ModuleType,
) -> None:
    mujoco = mujoco_module
    source = mujoco.MjSpec()
    source.worldbody.add_body(name="vehicle", pos=(1, 2, 3))
    config = CockpitConfig(
        parent_body="vehicle",
        position=(0.5, 0, 0),
        quaternion=(math.sqrt(0.5), 0, 0, math.sqrt(0.5)),
    )
    builder = add_cockpit(builder_class.from_spec(source), config)
    first, _ = builder.build()
    second, _ = builder.build()
    assert first is not second
    assert len(source.bodies) == 2
    data = mujoco.MjData(first)
    mujoco.mj_forward(first, data)
    np.testing.assert_allclose(data.body(config.names.root_body).xpos, (1.5, 2, 3))
    expected = np.array((1.5, 2, 3)) + np.array((-0.55, 0.12, 0.15))
    np.testing.assert_allclose(data.body(config.names.body("throttle")).xpos, expected)


def test_missing_parent_and_name_collision_fail(
    builder_class: type[Any],
    mujoco_module: ModuleType,
) -> None:
    with pytest.raises(KeyError, match="parent body"):
        add_cockpit(builder_class(), CockpitConfig(parent_body="absent")).build()
    builder = add_cockpit(builder_class())
    with pytest.raises(ValueError, match="already exists"):
        add_cockpit(builder).build()
    source = mujoco_module.MjSpec()
    source.worldbody.add_site(name="cockpit_throttle_position", size=(0.001, 0, 0))
    with pytest.raises(ValueError, match="cockpit_throttle_position"):
        add_cockpit(builder_class.from_spec(source)).build()


def test_independent_namespaces_compose(builder_class: type[Any]) -> None:
    builder = add_cockpit(builder_class())
    model, _ = add_cockpit(
        builder, CockpitConfig(prefix="second_", position=(2, 0, 0))
    ).build()
    assert model.nq == 6
    assert model.nsensor == 18


@pytest.mark.parametrize(
    "defaults",
    [
        '<joint pos="1 2 3"/>',
        '<geom pos="2 3 4" euler="10 20 30"/><site pos="3 4 5" euler="30 20 10"/>',
        '<geom type="capsule" fromto="1 2 3 4 5 6" size=".1"/>'
        '<site type="capsule" fromto="2 3 4 5 6 7" size=".1"/>',
    ],
)
def test_host_default_placements_do_not_shift_cockpit_mechanisms(
    builder_class: type[Any],
    mujoco_module: ModuleType,
    defaults: str,
) -> None:
    mujoco = mujoco_module
    source = mujoco.MjSpec.from_string(
        f"<mujoco><default>{defaults}</default><worldbody>"
        '<body name="host"><joint name="host_joint"/>'
        '<geom name="host_geom" size=".1 .1 .1" mass="1"/>'
        '<site name="host_site" size=".1 .1 .1"/>'
        "</body></worldbody></mujoco>"
    )
    original = source.compile()
    reference, _ = add_cockpit(builder_class()).build()
    model, _ = add_cockpit(builder_class.from_spec(source)).build()
    for kind, count, fields in (
        ("joint", reference.njnt, ("pos", "axis")),
        ("geom", reference.ngeom, ("pos", "quat")),
        ("site", reference.nsite, ("pos", "quat")),
        ("body", reference.nbody, ("mass", "ipos", "iquat", "inertia")),
    ):
        for index in range(1 if kind == "body" else 0, count):
            expected = getattr(reference, kind)(index)
            actual = getattr(model, kind)(expected.name)
            for field in fields:
                np.testing.assert_allclose(
                    getattr(actual, field), getattr(expected, field), atol=1e-14
                )
    for kind in ("joint", "geom", "site"):
        for field in ("pos", "axis") if kind == "joint" else ("pos", "quat"):
            np.testing.assert_array_equal(
                getattr(getattr(model, kind)(f"host_{kind}"), field),
                getattr(getattr(original, kind)(f"host_{kind}"), field),
            )


@pytest.mark.parametrize("inference", ["false", "auto", "true"])
@pytest.mark.parametrize("groups", [(0, 5), (1, 5)])
def test_physical_inertias_preserve_host_compiler_settings(
    builder_class: type[Any],
    mujoco_module: ModuleType,
    inference: str,
    groups: tuple[int, int],
) -> None:
    mujoco = mujoco_module
    source = mujoco.MjSpec.from_string(
        f'<mujoco><compiler inertiafromgeom="{inference}" '
        f'inertiagrouprange="{groups[0]} {groups[1]}"/>'
        '<worldbody><body name="host"><joint name="host_joint"/>'
        '<inertial pos=".01 .02 .03" mass="2.7" '
        'diaginertia=".2 .3 .4"/><geom size=".1"/>'
        "</body></worldbody></mujoco>"
    )
    original = source.compile()
    config = CockpitConfig(
        throttle=PedalConfig(half_size=(0.03, 0.02, 0.08), mass=0.7),
        steering=SteeringConfig(radius=0.21, segments=17, mass=2.3),
    )
    model, spec = add_cockpit(builder_class.from_spec(source), config).build()
    assert spec.compiler.inertiafromgeom == source.compiler.inertiafromgeom
    np.testing.assert_array_equal(spec.compiler.inertiagrouprange, groups)
    for field in ("mass", "ipos", "iquat", "inertia"):
        np.testing.assert_array_equal(
            getattr(model.body("host"), field), getattr(original.body("host"), field)
        )

    reference_spec = spec.copy()
    reference_spec.compiler.inertiafromgeom = (
        mujoco.mjtInertiaFromGeom.mjINERTIAFROMGEOM_TRUE
    )
    reference_spec.compiler.inertiagrouprange = (0, 5)
    reference = reference_spec.compile()
    for control in ("throttle", "brake", "steering"):
        body = model.body(config.names.body(control))
        assert spec.body(body.name).explicitinertial
        assert body.mass[0] == pytest.approx(getattr(config, control).mass)
        assert np.all(body.inertia > 0)
        for field in ("mass", "ipos", "iquat", "inertia"):
            np.testing.assert_allclose(
                getattr(body, field),
                getattr(reference.body(body.name), field),
                rtol=1e-12,
                atol=1e-14,
            )


def _pusher_model(
    builder_class: type[Any],
    mujoco: ModuleType,
    config: CockpitConfig,
) -> tuple[Any, Any]:
    builder = add_cockpit(builder_class().set_timestep(0.001), config)

    def add_pusher(spec: Any) -> Any:
        pusher = spec.worldbody.add_body(name="contact_pusher", mocap=True)
        pusher.add_geom(
            name="contact_pusher_geom",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=(0.02, 0, 0),
            mass=0.1,
            friction=(1, 0.005, 0.0001),
            contype=1,
            conaffinity=1,
        )
        return spec

    model, _ = builder.apply_transform(add_pusher).build()
    return model, mujoco.MjData(model)


@pytest.mark.parametrize("control", ["throttle", "brake"])
def test_real_contact_depresses_pedal_and_spring_releases_it(
    builder_class: type[Any],
    mujoco_module: ModuleType,
    control: str,
) -> None:
    mujoco = mujoco_module
    config = CockpitConfig()
    model, data = _pusher_model(builder_class, mujoco, config)
    pedal = getattr(config, control)
    joint_name = getattr(config.names, f"{control}_joint")
    center = np.array(pedal.position)
    peak_travel = 0.0
    peak_touch = 0.0
    contacts = 0
    for offset in np.linspace(-0.08, 0.015, 500):
        data.mocap_pos[0] = center + (0, offset, 0)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        peak_travel = max(peak_travel, data.joint(joint_name).qpos[0])
        peak_touch = max(
            peak_touch, data.sensor(config.names.touch_sensor(control)).data[0]
        )
        contacts += data.ncon
    assert contacts > 0
    assert peak_touch > 1.0
    assert peak_travel > 0.025
    assert peak_travel < pedal.travel + 0.005
    data.mocap_pos[0] = center + (0, -0.2, 0)
    for _ in range(1200):
        mujoco.mj_step(model, data)
    assert abs(data.joint(joint_name).qpos[0]) < 1e-4
    assert np.all(np.isfinite(data.qpos))


def test_real_contact_turns_wheel_and_centering_spring_returns_it(
    builder_class: type[Any],
    mujoco_module: ModuleType,
) -> None:
    mujoco = mujoco_module
    config = CockpitConfig()
    model, data = _pusher_model(builder_class, mujoco, config)
    center = np.array(config.steering.position)
    peak_angle = 0.0
    peak_touch = 0.0
    for height in np.linspace(0.07, -0.09, 600):
        data.mocap_pos[0] = center + (0.1, 0, height)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        peak_angle = max(peak_angle, data.joint(config.names.steering_joint).qpos[0])
        peak_touch = max(
            peak_touch, data.sensor(config.names.touch_sensor("steering")).data[0]
        )
    assert peak_touch > 1.0
    assert peak_angle > 0.25
    data.mocap_pos[0] = center + (0, -0.5, 0)
    for _ in range(3000):
        mujoco.mj_step(model, data)
    assert abs(data.joint(config.names.steering_joint).qpos[0]) < 0.03


def test_wheel_centering_is_optional(
    builder_class: type[Any],
    mujoco_module: ModuleType,
) -> None:
    mujoco = mujoco_module
    config = CockpitConfig(steering=SteeringConfig(stiffness=0.0))
    model, _ = add_cockpit(builder_class(), config).build()
    data = mujoco.MjData(model)
    data.joint(config.names.steering_joint).qpos[0] = 0.7
    for _ in range(300):
        mujoco.mj_step(model, data)
    assert data.joint(config.names.steering_joint).qpos[0] == pytest.approx(
        0.7, abs=1e-6
    )


@pytest.mark.parametrize("fragment", ["elbow", "arm", "leg"])
def test_composes_with_real_muscle_models(
    builder_class: type[Any],
    mujoco_module: ModuleType,
    fragment: str,
) -> None:
    pytest.importorskip("myo_sim")
    mujoco = mujoco_module
    builder = builder_class().place_fragment(fragment, pos=(3, 0, 1))
    bare, _ = builder.build()
    config = CockpitConfig()
    model, _ = add_cockpit(builder, config).build()
    assert bare.nu > 0
    assert model.nu == bare.nu
    assert model.nq == bare.nq + 3
    assert model.nsensor == bare.nsensor + 9
    assert model.nmesh == bare.nmesh
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert np.all(np.isfinite(data.sensordata))
    muscle_count = np.count_nonzero(
        model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE
    )
    assert muscle_count > 0
    for _ in range(10):
        mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))

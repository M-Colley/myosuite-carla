"""Passive collision geometry composed into a MyoSuite ModelBuilder recipe."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco
    from myosuite.core.model_builder import ModelBuilder

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


def _vector(value: tuple[float, ...], length: int, name: str) -> None:
    if len(value) != length or not all(math.isfinite(v) for v in value):
        raise ValueError(f"{name} must contain {length} finite numbers")


def _positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {requirement}")


def _placement(position: Vec3, quaternion: Quat) -> None:
    _vector(position, 3, "position")
    _vector(quaternion, 4, "quaternion")
    if not math.isclose(sum(v * v for v in quaternion), 1.0, abs_tol=1e-6):
        raise ValueError("quaternion must have unit length (w, x, y, z)")


@dataclass(frozen=True)
class PedalConfig:
    """A spring-loaded slide pedal; positive travel follows its local +Y axis.

    Args:
        position: Pedal center relative to the cockpit, in meters.
        quaternion: Local orientation as a unit (w, x, y, z) quaternion.
        half_size: Collision box half extents, in meters.
        travel: Maximum travel in meters; the released position is zero.
        stiffness: Return spring stiffness in N/m.
        damping: Viscous damping in N s/m.
        mass: Moving pedal mass in kg.
        friction: Sliding, torsional, and rolling contact friction.
    """

    position: Vec3 = (0.12, 0.55, 0.15)
    quaternion: Quat = (1.0, 0.0, 0.0, 0.0)
    half_size: Vec3 = (0.045, 0.015, 0.09)
    travel: float = 0.06
    stiffness: float = 500.0
    damping: float = 10.0
    mass: float = 0.5
    friction: Vec3 = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        _placement(self.position, self.quaternion)
        _vector(self.half_size, 3, "half_size")
        _vector(self.friction, 3, "friction")
        for i, value in enumerate(self.half_size):
            _positive(value, f"half_size[{i}]")
        for i, value in enumerate(self.friction):
            _positive(value, f"friction[{i}]", allow_zero=True)
        for name in ("travel", "stiffness", "mass"):
            _positive(getattr(self, name), name)
        _positive(self.damping, "damping", allow_zero=True)


@dataclass(frozen=True)
class SteeringConfig:
    """A passive wheel in the local XZ plane, rotating about local +Y.

    Positive rotation moves the top of the wheel to the driver's right.
    All angles are radians, independent of the host MJCF compiler settings.

    Args:
        position: Wheel hub relative to the cockpit, in meters.
        quaternion: Local orientation as a unit (w, x, y, z) quaternion.
        radius: Radius of the ring's capsule centerline, in meters.
        grip_radius: Radius of each ring capsule, in meters.
        segments: Number of collision capsules in the closed ring.
        angle_limit: Symmetric mechanical stop angle in radians.
        stiffness: Optional centering stiffness in N m/rad; zero disables it.
        damping: Viscous damping in N m s/rad.
        mass: Total moving wheel mass in kg, including spokes.
        friction: Sliding, torsional, and rolling contact friction.
    """

    position: Vec3 = (0.0, 0.32, 0.65)
    quaternion: Quat = (1.0, 0.0, 0.0, 0.0)
    radius: float = 0.18
    grip_radius: float = 0.016
    segments: int = 24
    angle_limit: float = math.pi * 1.5
    stiffness: float = 1.5
    damping: float = 0.4
    mass: float = 1.5
    friction: Vec3 = (1.0, 0.005, 0.0001)

    def __post_init__(self) -> None:
        _placement(self.position, self.quaternion)
        _vector(self.friction, 3, "friction")
        for name in ("radius", "grip_radius", "angle_limit", "mass"):
            _positive(getattr(self, name), name)
        for name in ("stiffness", "damping"):
            _positive(getattr(self, name), name, allow_zero=True)
        for i, value in enumerate(self.friction):
            _positive(value, f"friction[{i}]", allow_zero=True)
        if isinstance(self.segments, bool) or not isinstance(self.segments, int):
            raise ValueError("segments must be an integer")
        if self.segments < 8:
            raise ValueError("segments must be at least 8")
        if self.grip_radius >= self.radius:
            raise ValueError("grip_radius must be smaller than radius")


@dataclass(frozen=True)
class CockpitNames:
    """Stable names for binding compiled joints and scalar sensors.

    Args:
        prefix: Namespace used by the cockpit recipe.
    """

    prefix: str = "cockpit_"

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.prefix
        ):
            raise ValueError("prefix must be a nonempty MJCF identifier")

    @property
    def root_body(self) -> str:
        """Name of the fixed cockpit root body."""
        return f"{self.prefix}root"

    @property
    def throttle_joint(self) -> str:
        """Throttle travel joint name."""
        return f"{self.prefix}throttle_joint"

    @property
    def brake_joint(self) -> str:
        """Brake travel joint name."""
        return f"{self.prefix}brake_joint"

    @property
    def steering_joint(self) -> str:
        """Steering angle joint name."""
        return f"{self.prefix}steering_joint"

    @property
    def throttle_position(self) -> str:
        """Throttle joint-position sensor name (meters)."""
        return f"{self.prefix}throttle_position"

    @property
    def brake_position(self) -> str:
        """Brake joint-position sensor name (meters)."""
        return f"{self.prefix}brake_position"

    @property
    def steering_position(self) -> str:
        """Steering joint-position sensor name (radians)."""
        return f"{self.prefix}steering_position"

    def body(self, control: str) -> str:
        """Return the moving body name for throttle, brake, or steering."""
        return self._control_name(control, "body")

    def contact_site(self, control: str) -> str:
        """Return the contact-sensing site name for a control."""
        return self._control_name(control, "contact")

    def touch_sensor(self, control: str) -> str:
        """Return the scalar touch sensor name (normal force in newtons)."""
        return self._control_name(control, "touch")

    def velocity_sensor(self, control: str) -> str:
        """Return the scalar joint-velocity sensor name (m/s or rad/s)."""
        return self._control_name(control, "velocity")

    def _control_name(self, control: str, suffix: str) -> str:
        if control not in ("throttle", "brake", "steering"):
            raise ValueError("control must be throttle, brake, or steering")
        return f"{self.prefix}{control}_{suffix}"


@dataclass(frozen=True)
class CockpitConfig:
    """Placement and mechanics for a reusable physical cockpit assembly.

    Local +X points to the driver's right, +Y points forward, and +Z points up.
    Pedals move forward on contact; clockwise steering is positive. Geometry is
    composed into the same model as the muscles, so MuJoCo solves the contacts.

    Args:
        prefix: Unique namespace for every generated element.
        position: Cockpit root position in the parent frame, in meters.
        quaternion: Root orientation as a unit (w, x, y, z) quaternion.
        parent_body: Host body name, or worldbody for a stationary assembly.
        throttle: Throttle geometry and spring configuration.
        brake: Brake geometry and spring configuration.
        steering: Wheel geometry and spring configuration.
    """

    prefix: str = "cockpit_"
    position: Vec3 = (0.0, 0.0, 0.0)
    quaternion: Quat = (1.0, 0.0, 0.0, 0.0)
    parent_body: str = "worldbody"
    throttle: PedalConfig = field(default_factory=PedalConfig)
    brake: PedalConfig = field(
        default_factory=lambda: PedalConfig(
            position=(-0.12, 0.55, 0.15),
            stiffness=1500.0,
            damping=18.0,
        )
    )
    steering: SteeringConfig = field(default_factory=SteeringConfig)

    def __post_init__(self) -> None:
        CockpitNames(self.prefix)
        _placement(self.position, self.quaternion)
        if not isinstance(self.parent_body, str) or not self.parent_body:
            raise ValueError("parent_body must be a nonempty body name")
        if not isinstance(self.throttle, PedalConfig) or not isinstance(
            self.brake, PedalConfig
        ):
            raise TypeError("throttle and brake must be PedalConfig instances")
        if not isinstance(self.steering, SteeringConfig):
            raise TypeError("steering must be a SteeringConfig instance")

    @property
    def names(self) -> CockpitNames:
        """Names to use when binding this assembly to a CARLA mapping."""
        return CockpitNames(self.prefix)


def _set_explicit_inertias(bodies: tuple[mujoco.MjsBody, ...]) -> None:
    """Infer each mechanism's inertia from its geoms independently of the host."""
    import mujoco

    standalone = mujoco.MjSpec()
    for body in bodies:
        inertial_body = standalone.worldbody.add_body(name=body.name)
        for geom in body.geoms:
            inertial_body.add_geom(
                type=geom.type,
                size=geom.size,
                pos=geom.pos,
                quat=geom.quat,
                fromto=geom.fromto,
                mass=geom.mass,
                typeinertia=geom.typeinertia,
            )
    model = standalone.compile()
    for body in bodies:
        inertial = model.body(body.name)
        body.explicitinertial = True
        body.mass = float(inertial.mass[0])
        body.ipos = inertial.ipos
        body.iquat = inertial.iquat
        body.inertia = inertial.inertia


def add_cockpit(
    builder: ModelBuilder,
    config: CockpitConfig | None = None,
) -> ModelBuilder:
    """Append passive cockpit geometry to a ModelBuilder recipe.

    The transform creates three unactuated DOFs, position/velocity sensors,
    and scalar contact force sensors. It preserves host options and muscle
    actuators, and supports both CPU compilation and mjlab spec consumption.
    The caller positions a muscle model with ModelBuilder placement helpers.

    Args:
        builder: A MyoSuite ModelBuilder, optionally containing muscle fragments.
        config: Placement, namespace and mechanics; defaults to CockpitConfig().

    Returns:
        The same builder, allowing further recipe composition.

    Raises:
        ValueError: A generated name already exists when the recipe is built.
        KeyError: The configured parent body does not exist at build time.
    """
    config = config if config is not None else CockpitConfig()
    if not isinstance(config, CockpitConfig):
        raise TypeError("config must be a CockpitConfig instance")

    def transform(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        import mujoco

        names = config.names
        generated = {names.root_body}
        for control in ("throttle", "brake", "steering"):
            generated.update(
                f"{config.prefix}{control}_{suffix}"
                for suffix in (
                    "body",
                    "joint",
                    "contact",
                    "position",
                    "velocity",
                    "touch",
                )
            )
        generated.update(
            (f"{config.prefix}throttle_surface", f"{config.prefix}brake_surface")
        )
        generated.update(
            f"{config.prefix}steering_rim_{i:02d}"
            for i in range(config.steering.segments)
        )
        generated.update(f"{config.prefix}steering_spoke_{i}" for i in range(3))
        for collection in (
            spec.bodies,
            spec.joints,
            spec.geoms,
            spec.sites,
            spec.sensors,
        ):
            for element in collection:
                if element.name in generated:
                    raise ValueError(
                        f"cockpit element name already exists: {element.name!r}"
                    )
        if config.parent_body == "worldbody":
            parent = spec.worldbody
        else:
            parent = spec.body(config.parent_body)
            if parent is None:
                raise KeyError(
                    f"cockpit parent body does not exist: {config.parent_body!r}"
                )
        root = parent.add_body(
            name=names.root_body, pos=config.position, quat=config.quaternion
        )
        moving_bodies = []

        def add_sensors(
            control: str, body: mujoco.MjsBody, size: Vec3, site_type: int
        ) -> None:
            site = body.add_site(
                name=names.contact_site(control),
                type=site_type,
                size=size,
                pos=(0, 0, 0),
                quat=(1, 0, 0, 0),
                fromto=(math.nan,) * 6,
                rgba=(0.0, 0.0, 0.0, 0.0),
            )
            site.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
            for suffix, sensor_type, object_type, object_name in (
                (
                    "position",
                    mujoco.mjtSensor.mjSENS_JOINTPOS,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{config.prefix}{control}_joint",
                ),
                (
                    "velocity",
                    mujoco.mjtSensor.mjSENS_JOINTVEL,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{config.prefix}{control}_joint",
                ),
                (
                    "touch",
                    mujoco.mjtSensor.mjSENS_TOUCH,
                    mujoco.mjtObj.mjOBJ_SITE,
                    names.contact_site(control),
                ),
            ):
                spec.add_sensor(
                    name=f"{config.prefix}{control}_{suffix}",
                    type=sensor_type,
                    objtype=object_type,
                    objname=object_name,
                )

        for control, pedal in (("throttle", config.throttle), ("brake", config.brake)):
            body = root.add_body(
                name=names.body(control), pos=pedal.position, quat=pedal.quaternion
            )
            moving_bodies.append(body)
            body.add_joint(
                name=f"{config.prefix}{control}_joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                pos=(0, 0, 0),
                axis=(0, 1, 0),
                limited=True,
                range=(0, pedal.travel),
                ref=0,
                springref=0,
                stiffness=pedal.stiffness,
                damping=pedal.damping,
                springdamper=(0, 0),
                armature=0,
                frictionloss=0,
            )
            surface = body.add_geom(
                name=f"{config.prefix}{control}_surface",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=pedal.half_size,
                pos=(0, 0, 0),
                quat=(1, 0, 0, 0),
                fromto=(math.nan,) * 6,
                mass=pedal.mass,
                friction=pedal.friction,
                contype=1,
                conaffinity=1,
                condim=3,
                rgba=(0.22, 0.24, 0.26, 1),
            )
            surface.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
            add_sensors(
                control,
                body,
                (
                    pedal.half_size[0] + 1e-4,
                    pedal.half_size[1] + 1e-4,
                    pedal.half_size[2] + 1e-4,
                ),
                mujoco.mjtGeom.mjGEOM_BOX,
            )

        wheel = config.steering
        body = root.add_body(
            name=names.body("steering"), pos=wheel.position, quat=wheel.quaternion
        )
        moving_bodies.append(body)
        angle = (
            math.degrees(wheel.angle_limit)
            if spec.compiler.degree
            else wheel.angle_limit
        )
        body.add_joint(
            name=names.steering_joint,
            type=mujoco.mjtJoint.mjJNT_HINGE,
            pos=(0, 0, 0),
            axis=(0, 1, 0),
            limited=True,
            range=(-angle, angle),
            ref=0,
            springref=0,
            stiffness=wheel.stiffness,
            damping=wheel.damping,
            springdamper=(0, 0),
            armature=0,
            frictionloss=0,
        )
        geom_common = dict(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            pos=(0, 0, 0),
            quat=(1, 0, 0, 0),
            friction=wheel.friction,
            contype=1,
            conaffinity=1,
            condim=3,
        )
        for i in range(wheel.segments):
            a0 = 2 * math.pi * i / wheel.segments
            a1 = 2 * math.pi * (i + 1) / wheel.segments
            rim = body.add_geom(
                name=f"{config.prefix}steering_rim_{i:02d}",
                fromto=(
                    wheel.radius * math.cos(a0),
                    0,
                    wheel.radius * math.sin(a0),
                    wheel.radius * math.cos(a1),
                    0,
                    wheel.radius * math.sin(a1),
                ),
                size=(wheel.grip_radius, 0, 0),
                mass=wheel.mass * 0.85 / wheel.segments,
                rgba=(0.12, 0.13, 0.14, 1),
                **geom_common,
            )
            rim.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        for i in range(3):
            angle = 2 * math.pi * i / 3
            spoke = body.add_geom(
                name=f"{config.prefix}steering_spoke_{i}",
                fromto=(
                    0,
                    0,
                    0,
                    wheel.radius * math.cos(angle),
                    0,
                    wheel.radius * math.sin(angle),
                ),
                size=(wheel.grip_radius * 0.6, 0, 0),
                mass=wheel.mass * 0.15 / 3,
                rgba=(0.4, 0.42, 0.44, 1),
                **geom_common,
            )
            spoke.alt.type = mujoco.mjtOrientation.mjORIENTATION_QUAT
        add_sensors(
            "steering",
            body,
            (wheel.radius + wheel.grip_radius + 1e-4, 0, 0),
            mujoco.mjtGeom.mjGEOM_SPHERE,
        )
        _set_explicit_inertias(tuple(moving_bodies))
        return spec

    return builder.apply_transform(transform)

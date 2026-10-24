"""A seated MyoSuite full-body driver that operates every control with muscles.

The body is MyoSuite's ``myofullbody`` (416 muscles) from ``myo_sim``. Its
pelvis is fixed to the seat and gravity is disabled, as for the acceptance
fixtures. Trunk, hips, knees, subtalar joints, shoulders, forearm rotation and
wrists are held in a seated posture by joint-equality locks. Ankles, toes,
elbows, fingers and thumbs stay free and move only under muscle force.

Every vehicle control comes from a passive mechanism moved by body contact:

* right forefoot plantarflexion presses the throttle pedal;
* left forefoot plantarflexion presses the brake pedal;
* both hands rest in spinner cradles at 3 and 9 o'clock; push-pull elbow
  flexion/extension turns the wheel either way;
* the right index finger pulls a sequential gear paddle, the right little finger
  holds a reverse button and the left index finger holds a parking-brake button.
  Thumbs are not used: a single-tendon thumb press buckles under sustained load
  in this model, whereas the fingers hold steadily.

The mechanisms are placed from the body's own kinematics so each one starts
1.5 mm clear of the limb that operates it. Elbows use a proportional-integral-
derivative stretch reflex on antagonist muscle excitations; everything else is
open-loop excitation. This is a scripted demonstration driver, not a trained
policy, and it applies no external forces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import mujoco
import numpy as np

from .cockpit import CockpitConfig, PedalConfig, SteeringConfig, add_cockpit
from .inputs import (
    Binding,
    BipolarCalibration,
    CallbackSource,
    ControlMapper,
    SensorValue,
    UnipolarCalibration,
)

#: Locked seated-posture angles in radians; unlisted locked joints keep qpos0.
SEATED_POSTURE: dict[str, float] = {
    "hip_flexion_r": 1.35,
    "hip_flexion_l": 1.35,
    "hip_adduction_r": -0.1,
    "hip_adduction_l": -0.1,
    "knee_angle_r": 1.25,
    "knee_angle_l": 1.25,
    "elv_angle_r": 1.472,
    "elv_angle_l": 1.472,
    "shoulder_elv_r": 0.7491,
    "shoulder_elv_l": 0.7491,
    "shoulder_rot_r": 0.3139,
    "shoulder_rot_l": 0.3139,
    "pro_sup_r": -0.0585,
    "pro_sup_l": -0.0585,
}
#: Free joints left to muscles; every other independent joint is locked.
FREE_JOINT_PREFIXES = (
    "ankle_angle_",
    "mtp_angle_",
    "elbow_flexion_",
    "cmc_",
    "mp_flexion",
    "ip_flexion",
    "mcp",
    "pm",
    "md",
)
ELBOW_FLEXORS = ("BIClong", "BICshort", "BRA", "BRD")
ELBOW_EXTENSORS = ("TRIlong", "TRIlat", "TRImed", "ANC")
PLANTARFLEXORS = ("soleus", "gasmed", "gaslat")


def _side(names: tuple[str, ...], side: str) -> tuple[str, ...]:
    return tuple(name if side == "r" else f"{name}_l" for name in names)


@dataclass(frozen=True)
class FingerControl:
    """A button pressed by one digit's flexors and released by its extensors.

    Args:
        flexors: Muscles that press the button.
        extensors: Muscles that hold the digit off the button when released.
        pad: Distal collision capsule whose tip presses the button.
        contact_bodies: Digit bodies allowed to touch the button.
        palm: Body the button housing is fixed to.
    """

    flexors: tuple[str, ...]
    extensors: tuple[str, ...]
    pad: str
    contact_bodies: tuple[str, ...]
    palm: str


FINGER_CONTROLS: dict[str, FingerControl] = {
    "gear": FingerControl(
        ("FDS2", "FDP2"),
        ("EIP", "EDC2"),
        "distph2_coll_r",
        ("distph2_r", "midph2_r"),
        "thirdmc_r",
    ),
    "reverse": FingerControl(
        ("FDS5", "FDP5"),
        ("EDM", "EDC5"),
        "5distph_coll_r",
        ("distph5_r", "midph5_r"),
        "thirdmc_r",
    ),
    "park": FingerControl(
        ("FDS2_l", "FDP2_l"),
        ("EIP_l", "EDC2_l"),
        "distph2_coll_l",
        ("distph2_l", "midph2_l"),
        "thirdmc_l",
    ),
}


@dataclass(frozen=True)
class FullBodyDriverConfig:
    """Mechanics and timing of the seated full-body driver.

    Args:
        timestep: MuJoCo physics step in seconds.
        control_dt: Driver control period, equal to CARLA's fixed delta.
        clearance: Initial gap between each mechanism and its limb, in metres.
        wheel_offset: Wheel plane distance in front of the hands, in metres.
        steering_angle_limit: Wheel angle mapped to full steer, in radians.
        button_travel: Finger-button travel in metres.
        button_lead: Fingertip path length before a button is reached, metres.
        press_excitation: Flexor excitation used to trace fingertip paths.
        release_tone: Tibialis anterior excitation holding released feet.
        finger_release_tone: Extensor excitation holding released digits.
        elbow_gains: Stretch-reflex proportional, integral and derivative gains.
        elbow_tone: Common co-activation of elbow flexors and extensors.
        max_gear: Highest manual gear selected by the paddle.
        settle_seconds: Passive settling time before mechanisms are placed.
    """

    timestep: float = 0.001
    control_dt: float = 0.02
    clearance: float = 0.0015
    wheel_offset: float = 0.12
    steering_angle_limit: float = 0.6
    button_travel: float = 0.006
    button_lead: float = 0.008
    press_excitation: float = 0.08
    release_tone: float = 0.02
    finger_release_tone: float = 0.15
    elbow_gains: tuple[float, float, float] = (4.0, 8.0, 0.2)
    elbow_tone: float = 0.02
    max_gear: int = 3
    settle_seconds: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "timestep",
            "control_dt",
            "clearance",
            "wheel_offset",
            "steering_angle_limit",
            "button_travel",
            "button_lead",
            "press_excitation",
            "settle_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        steps = self.control_dt / self.timestep
        if not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-9):
            raise ValueError("control_dt must be a whole number of timesteps")
        if not 1 <= self.max_gear < 2**31:
            raise ValueError("max_gear must be a positive signed 32-bit integer")


@dataclass(frozen=True)
class DriverIntent:
    """One control period of the scripted motor program.

    Values are muscle excitations in [0, 1] or a reflex set-point offset; they
    are never vehicle controls. CARLA controls are measured from mechanisms.

    Args:
        right_foot: Right plantarflexor excitation (throttle pedal).
        left_foot: Left plantarflexor excitation (brake pedal).
        steer: Push-pull elbow set-point offset in radians; positive turns the
            wheel clockwise (right elbow extends, left elbow flexes).
        gear_finger: Right index flexor excitation (sequential gear paddle).
        reverse_finger: Right little-finger flexor excitation (reverse button).
        park_finger: Left index flexor excitation (parking-brake button).
    """

    right_foot: float = 0.0
    left_foot: float = 0.0
    steer: float = 0.0
    gear_finger: float = 0.0
    reverse_finger: float = 0.0
    park_finger: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "right_foot",
            "left_foot",
            "gear_finger",
            "reverse_finger",
            "park_finger",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a finite excitation in [0, 1]")
            object.__setattr__(self, name, value)
        steer = float(self.steer)
        if not math.isfinite(steer) or abs(steer) > 0.5:
            raise ValueError("steer must be a finite offset within 0.5 rad")
        object.__setattr__(self, "steer", steer)


class SequentialShifter:
    """Count measured paddle pulls into gears; 0 is automatic.

    Each rising edge of the paddle travel (above 60% after falling below 30%)
    selects the next manual gear, and a pull at ``max_gear`` returns to
    automatic. Reading the same data twice never counts twice.
    """

    def __init__(self, sensor: str, travel: float, max_gear: int) -> None:
        self.sensor = sensor
        self.travel = float(travel)
        self.max_gear = int(max_gear)
        self.reset()

    def reset(self) -> None:
        """Return to automatic with the paddle considered released."""
        self.gear = 0
        self._pressed = False

    def read_gear(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Update from the paddle sensor and return the selected gear."""
        fraction = float(data.sensor(self.sensor).data[0]) / self.travel
        if not self._pressed and fraction > 0.6:
            self._pressed = True
            self.gear = 0 if self.gear >= self.max_gear else self.gear + 1
        elif self._pressed and fraction < 0.3:
            self._pressed = False
        return float(self.gear)

    def read_manual(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        """Return 1 when a manual gear is selected; call after ``read_gear``."""
        return 1.0 if self.gear else 0.0


class _Reflex:
    """PID stretch reflex driving antagonist muscles toward an angle set-point."""

    def __init__(
        self,
        model: mujoco.MjModel,
        joint: str,
        flexors: tuple[str, ...],
        extensors: tuple[str, ...],
        gains: tuple[float, float, float],
        tone: float,
    ) -> None:
        index = model.joint(joint).id
        self.qpos = int(model.jnt_qposadr[index])
        self.dof = int(model.jnt_dofadr[index])
        self.flexors = [model.actuator(name).id for name in flexors]
        self.extensors = [model.actuator(name).id for name in extensors]
        self.gains = gains
        self.tone = tone
        self.setpoint = 0.0
        self.integral = 0.0

    def apply(self, data: mujoco.MjData, setpoint: float, dt: float) -> None:
        kp, ki, kd = self.gains
        error = setpoint - float(data.qpos[self.qpos])
        self.integral = float(np.clip(self.integral + error * dt, -0.1, 0.1))
        drive = kp * error + ki * self.integral - kd * float(data.qvel[self.dof])
        data.ctrl[self.flexors] = np.clip(self.tone + drive, 0.0, 1.0)
        data.ctrl[self.extensors] = np.clip(self.tone - drive, 0.0, 1.0)


def _frame(y_axis: np.ndarray, x_hint: np.ndarray) -> tuple[float, float, float, float]:
    y = np.asarray(y_axis, dtype=float)
    y = y / np.linalg.norm(y)
    x = np.asarray(x_hint, dtype=float) - np.dot(x_hint, y) * y
    x = x / np.linalg.norm(x)
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, np.column_stack([x, y, np.cross(x, y)]).ravel())
    return tuple(float(v) for v in quaternion)  # type: ignore[return-value]


def _local(rotation: np.ndarray, origin: np.ndarray, point: np.ndarray) -> np.ndarray:
    return rotation.T @ (np.asarray(point) - origin)


def _local_quat(
    rotation: np.ndarray, quaternion: tuple[float, ...]
) -> tuple[float, ...]:
    world = np.zeros(9)
    mujoco.mju_quat2Mat(world, np.asarray(quaternion))
    local = np.zeros(4)
    mujoco.mju_mat2Quat(local, (rotation.T @ world.reshape(3, 3)).ravel())
    return tuple(float(v) for v in local)


def _find_body(spec: mujoco.MjSpec, name: str) -> mujoco.MjsBody:
    """Find a body by iteration; MuJoCo 3.6 name lookup misses attached specs."""
    for body in spec.bodies:
        if body.name == name:
            return body
    raise KeyError(f"body does not exist: {name!r}")


def _body_spec(config: FullBodyDriverConfig) -> mujoco.MjSpec:
    import myo_sim  # type: ignore[import-untyped]

    spec = myo_sim.load_spec("myofullbody")
    spec.delete(spec.joint("root"))
    spec.option.gravity[:] = 0
    spec.option.timestep = config.timestep
    spec.visual.global_.offwidth = max(spec.visual.global_.offwidth, 1280)
    spec.visual.global_.offheight = max(spec.visual.global_.offheight, 720)
    model = spec.compile()
    coupled = {
        int(model.eq_obj1id[i])
        for i in range(model.neq)
        if int(model.eq_type[i]) == int(mujoco.mjtEq.mjEQ_JOINT)
    }
    for index in range(model.njnt):
        name = model.joint(index).name
        if index in coupled or name.startswith(FREE_JOINT_PREFIXES):
            continue
        value = SEATED_POSTURE.get(name, float(model.qpos0[model.jnt_qposadr[index]]))
        equality = spec.add_equality(
            name=f"seat_{name}", type=mujoco.mjtEq.mjEQ_JOINT, name1=name
        )
        # The equality data length differs between MuJoCo releases.
        data = np.zeros(len(equality.data))
        data[0] = value
        equality.data = data
    return spec


#: Settled elbow angle of the solved seated arm posture, in radians.
REST_ELBOW = 0.3784


def _muscle_groups() -> dict[str, tuple[tuple[str, ...], tuple[str, ...]]]:
    """Agonists and released-state antagonists of each motor-program channel."""
    groups: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "right_foot": (tuple(f"{name}_r" for name in PLANTARFLEXORS), ("tibant_r",)),
        "left_foot": (tuple(f"{name}_l" for name in PLANTARFLEXORS), ("tibant_l",)),
    }
    for field, control in (
        ("gear_finger", "gear"),
        ("reverse_finger", "reverse"),
        ("park_finger", "park"),
    ):
        groups[field] = (
            FINGER_CONTROLS[control].flexors,
            FINGER_CONTROLS[control].extensors,
        )
    return groups


def _excite(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: FullBodyDriverConfig,
    levels: dict[str, float],
) -> None:
    """Excite pressing groups; hold every released group with antagonist tone."""
    for name, (agonists, antagonists) in _muscle_groups().items():
        level = levels.get(name, 0.0)
        if level > 0:
            ids = [model.actuator(muscle).id for muscle in agonists]
            data.ctrl[ids] = level
        else:
            tone = (
                config.release_tone
                if name.endswith("_foot")
                else config.finger_release_tone
            )
            ids = [model.actuator(muscle).id for muscle in antagonists]
            data.ctrl[ids] = tone


def _settle(
    model: mujoco.MjModel, config: FullBodyDriverConfig
) -> tuple[mujoco.MjData, dict[str, float]]:
    data = mujoco.MjData(model)
    for name, value in SEATED_POSTURE.items():
        data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
    for side in "rl":
        data.qpos[model.jnt_qposadr[model.joint(f"elbow_flexion_{side}").id]] = (
            REST_ELBOW
        )
    mujoco.mj_forward(model, data)
    reflexes = _reflexes(model, config)
    for _ in range(round(config.settle_seconds / config.timestep)):
        data.ctrl[:] = 0
        for reflex in reflexes.values():
            reflex.apply(data, REST_ELBOW, config.timestep)
        _excite(model, data, config, {})
        mujoco.mj_step(model, data)
    return data, {side: reflex.integral for side, reflex in reflexes.items()}


def _reflexes(
    model: mujoco.MjModel,
    config: FullBodyDriverConfig,
    integrals: dict[str, float] | None = None,
) -> dict[str, _Reflex]:
    reflexes = {
        side: _Reflex(
            model,
            f"elbow_flexion_{side}",
            _side(ELBOW_FLEXORS, side),
            _side(ELBOW_EXTENSORS, side),
            config.elbow_gains,
            config.elbow_tone,
        )
        for side in "rl"
    }
    for side, value in (integrals or {}).items():
        reflexes[side].integral = value
    return reflexes


def _motion(
    model: mujoco.MjModel, qpos: np.ndarray, joint: str, body: str, delta: float
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    start = data.xpos[model.body(body).id].copy()
    data.qpos[model.jnt_qposadr[model.joint(joint).id]] += delta
    mujoco.mj_kinematics(model, data)
    step = data.xpos[model.body(body).id] - start
    return start, step / np.linalg.norm(step)


def _pad_point(model: mujoco.MjModel, data: mujoco.MjData, geom: str) -> np.ndarray:
    index = model.geom(geom).id
    axis = data.geom_xmat[index].reshape(3, 3)[:, 2] * model.geom_size[index][1]
    centre = data.geom_xpos[index]
    origin = data.xpos[model.geom_bodyid[index]]
    ends = (centre + axis, centre - axis)
    return max(ends, key=lambda end: float(np.linalg.norm(end - origin))).copy()


def _pad_path(
    model: mujoco.MjModel,
    rest: mujoco.MjData,
    field: str,
    control: FingerControl,
    config: FullBodyDriverConfig,
    integrals: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    data.qpos[:] = rest.qpos
    data.act[:] = rest.act
    mujoco.mj_forward(model, data)
    reflexes = _reflexes(model, config, integrals)
    previous = _pad_point(model, data, control.pad)
    length = 0.0
    for _ in range(round(1.0 / config.timestep)):
        data.ctrl[:] = 0
        for reflex in reflexes.values():
            reflex.apply(data, REST_ELBOW, config.timestep)
        _excite(model, data, config, {field: config.press_excitation})
        mujoco.mj_step(model, data)
        point = _pad_point(model, data, control.pad)
        step = float(np.linalg.norm(point - previous))
        if length + step >= config.button_lead:
            direction = (point - previous) / step
            # The pad surface leads the capsule end point by the capsule radius.
            radius = float(model.geom_size[model.geom(control.pad).id][0])
            return point + radius * direction, direction
        length += step
        previous = point
    raise RuntimeError(f"{control.pad} did not move {config.button_lead} m")


@dataclass
class _Layout:
    pedals: dict[str, tuple[np.ndarray, np.ndarray]]
    hands: dict[str, np.ndarray]
    tangent: np.ndarray
    buttons: dict[str, tuple[np.ndarray, np.ndarray]]
    palms: dict[str, tuple[np.ndarray, np.ndarray]]


def _layout(
    model: mujoco.MjModel,
    rest: mujoco.MjData,
    config: FullBodyDriverConfig,
    integrals: dict[str, float],
) -> _Layout:
    qpos = rest.qpos.copy()
    pedals = {
        side: _motion(model, qpos, f"ankle_angle_{side}", f"toes_{side}", -0.02)
        for side in "rl"
    }
    hands = {
        side: 0.5
        * (
            rest.xpos[model.body(f"secondmc_{side}").id]
            + rest.xpos[model.body(f"fifthmc_{side}").id]
        )
        for side in "rl"
    }
    tangent = np.sum(
        [
            _motion(model, qpos, f"elbow_flexion_{side}", f"capitate_{side}", 0.02)[1]
            for side in "rl"
        ],
        axis=0,
    )
    tangent = tangent / np.linalg.norm(tangent)
    fields = {"gear": "gear_finger", "reverse": "reverse_finger", "park": "park_finger"}
    buttons = {
        name: _pad_path(model, rest, fields[name], control, config, integrals)
        for name, control in FINGER_CONTROLS.items()
    }
    palms = {
        name: (
            rest.xpos[model.body(name).id].copy(),
            rest.xmat[model.body(name).id].reshape(3, 3).copy(),
        )
        for name in {control.palm for control in FINGER_CONTROLS.values()}
    }
    return _Layout(pedals, hands, tangent, buttons, palms)


def _build(
    config: FullBodyDriverConfig, layout: _Layout, shifts: dict[str, float]
) -> tuple[mujoco.MjModel, CockpitConfig]:
    from myosuite.core.model_builder import ModelBuilder

    tangent = layout.tangent
    forward = np.array([0.0, -1.0, 0.0])
    axis = forward - np.dot(forward, tangent) * tangent
    axis /= np.linalg.norm(axis)
    middle = 0.5 * (layout.hands["r"] + layout.hands["l"])
    right = layout.hands["r"] - middle
    right -= np.dot(right, axis) * axis
    radius = float(np.linalg.norm(right))
    right /= radius
    hub = middle + config.wheel_offset * axis
    wheel_rotation = np.column_stack([right, axis, np.cross(right, axis)])
    pedals = {}
    for side, control in (("r", "throttle"), ("l", "brake")):
        ball, direction = layout.pedals[side]
        pedals[control] = (
            tuple(ball + (0.03 + shifts.get(control, 0.0)) * direction),
            _frame(direction, np.array([-1.0, 0.0, 0.0])),
        )
    cockpit = CockpitConfig(
        throttle=PedalConfig(
            position=pedals["throttle"][0],
            quaternion=pedals["throttle"][1],
            half_size=(0.045, 0.008, 0.07),
        ),
        brake=PedalConfig(
            position=pedals["brake"][0],
            quaternion=pedals["brake"][1],
            half_size=(0.045, 0.008, 0.07),
            stiffness=1500.0,
            damping=18.0,
        ),
        steering=SteeringConfig(
            position=tuple(hub),
            quaternion=_frame(axis, right),
            radius=radius,
            angle_limit=config.steering_angle_limit,
        ),
    )
    names = cockpit.names

    def mechanisms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body_names = [body.name for body in spec.bodies if body.name]
        wheel = _find_body(spec, names.body("steering"))
        for side, sign in (("r", 1.0), ("l", -1.0)):
            cradle = wheel.add_body(
                name=f"cradle_{side}",
                pos=_local(wheel_rotation, hub, layout.hands[side]),
            )
            # Like a spinner knob, the cradle turns freely on its post so the
            # wheel's rotation does not tilt the plates against the hand.
            cradle.add_joint(
                name=f"cradle_{side}_spin",
                type=mujoco.mjtJoint.mjJNT_HINGE,
                axis=(0, 1, 0),
                limited=True,
                range=(-1.0, 1.0),
                stiffness=0.05,
                damping=0.02,
                armature=0.001,
                margin=0,
                ref=0,
                springref=0,
                frictionloss=0,
            )
            for plate, offset_sign in (("upper", 1.0), ("lower", -1.0)):
                gap = 0.026 + shifts.get(f"cradle_{side}_{plate}", 0.0)
                cradle.add_geom(
                    name=f"cradle_{side}_{plate}",
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=(0.03, 0.05, 0.004),
                    pos=(0.0, 0.0, offset_sign * gap),
                    mass=0.05,
                    rgba=(0.75, 0.55, 0.2, 1),
                    contype=1,
                    conaffinity=1,
                )
            cradle.add_geom(
                name=f"cradle_{side}_post",
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=(-sign * 0.02, 0.0, 0.0, -sign * 0.02, config.wheel_offset, 0.0),
                size=(0.006, 0.0, 0.0),
                mass=0.02,
                rgba=(0.4, 0.42, 0.44, 1),
                contype=0,
                conaffinity=0,
            )
        for name in body_names:
            if name != "world":
                spec.add_exclude(bodyname1=name, bodyname2=names.body("steering"))
        # Cradle plates bear on the metacarpals only; digits stay free to press.
        digits = ("firstmc", "proximal_thumb", "distal_thumb") + tuple(
            name
            for digit in range(2, 6)
            for name in (f"{digit}proxph", f"midph{digit}", f"distph{digit}")
        )
        for side in "rl":
            for name in digits:
                spec.add_exclude(bodyname1=f"{name}_{side}", bodyname2=f"cradle_{side}")
        for name, control in FINGER_CONTROLS.items():
            point, direction = layout.buttons[name]
            centre = point + (0.0025 + shifts.get(name, 0.0)) * direction
            palm_position, palm_rotation = layout.palms[control.palm]
            housing = _find_body(spec, control.palm).add_body(
                name=f"{name}_button_body",
                pos=_local(palm_rotation, palm_position, centre),
                quat=_local_quat(palm_rotation, _frame(direction, tangent)),
            )
            joint = housing.add_joint(
                name=f"{name}_button_joint",
                type=mujoco.mjtJoint.mjJNT_SLIDE,
                axis=(0, 1, 0),
                limited=True,
                range=(0.0, config.button_travel),
                stiffness=100.0,
                damping=0.5,
                margin=0,
                ref=0,
                springref=0,
                armature=0.02,
                frictionloss=0,
            )
            joint.solref_limit = (0.004, 1.0)
            housing.add_geom(
                name=f"{name}_button_surface",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.011, 0.0025, 0.011),
                mass=0.02,
                rgba=(0.85, 0.2, 0.2, 1),
                contype=0,
                conaffinity=0,
            )
            housing.add_site(
                name=f"{name}_button_contact",
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=(0.0111, 0.0026, 0.0111),
                rgba=(0, 0, 0, 0),
            )
            spec.add_sensor(
                name=f"{name}_button_position",
                type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                objtype=mujoco.mjtObj.mjOBJ_JOINT,
                objname=f"{name}_button_joint",
            )
            spec.add_sensor(
                name=f"{name}_button_touch",
                type=mujoco.mjtSensor.mjSENS_TOUCH,
                objtype=mujoco.mjtObj.mjOBJ_SITE,
                objname=f"{name}_button_contact",
            )
            for body in control.contact_bodies:
                for geom in _find_body(spec, body).geoms:
                    if geom.contype:
                        spec.add_pair(
                            geomname1=f"{name}_button_surface",
                            geomname2=geom.name,
                            condim=3,
                        )
        return spec

    builder = ModelBuilder.from_spec(_body_spec(config))
    model, _ = add_cockpit(builder, cockpit).apply_transform(mechanisms).build()
    return model, cockpit


def _copy_pose(
    source: mujoco.MjModel, qpos: np.ndarray, model: mujoco.MjModel, data: mujoco.MjData
) -> None:
    for index in range(source.njnt):
        target = model.jnt_qposadr[model.joint(source.joint(index).name).id]
        data.qpos[target] = qpos[source.jnt_qposadr[index]]


def _gap(
    model: mujoco.MjModel, data: mujoco.MjData, geom: str, bodies: tuple[str, ...]
) -> float:
    ids = {model.body(name).id for name in bodies}
    limb = [
        index
        for index in range(model.ngeom)
        if model.geom_bodyid[index] in ids and model.geom_contype[index]
    ]
    fromto = np.zeros(6)
    return min(
        mujoco.mj_geomDistance(model, data, model.geom(geom).id, other, 0.2, fromto)
        for other in limb
    )


_HAND_BODIES = (
    "secondmc",
    "thirdmc",
    "fourthmc",
    "fifthmc",
    "2proxph",
    "3proxph",
    "4proxph",
    "5proxph",
)


def _clearance_checks() -> dict[str, tuple[str, tuple[str, ...]]]:
    checks: dict[str, tuple[str, tuple[str, ...]]] = {
        "throttle": ("cockpit_throttle_surface", ("calcn_r", "toes_r")),
        "brake": ("cockpit_brake_surface", ("calcn_l", "toes_l")),
    }
    for side in "rl":
        hand = tuple(f"{name}_{side}" for name in _HAND_BODIES)
        for plate in ("upper", "lower"):
            checks[f"cradle_{side}_{plate}"] = (f"cradle_{side}_{plate}", hand)
    for name, control in FINGER_CONTROLS.items():
        checks[name] = (f"{name}_button_surface", control.contact_bodies)
    return checks


class FullBodyDriver:
    """Seated full-body driver whose muscles operate the composed cockpit.

    Build instances with :func:`build_fullbody_driver`. Call :meth:`reset`,
    then :meth:`advance` once per control period before ``CarlaBridge.step``.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cockpit: CockpitConfig,
        config: FullBodyDriverConfig,
        rest: tuple[np.ndarray, np.ndarray],
        integrals: dict[str, float],
        gaps: dict[str, float],
    ) -> None:
        self.model = model
        self.cockpit = cockpit
        self.config = config
        self.data = mujoco.MjData(model)
        self.rest_qpos, self.rest_act = (array.copy() for array in rest)
        self.rest_integrals = dict(integrals)
        self.initial_gaps = dict(gaps)
        self.shifter = SequentialShifter(
            "gear_button_position", config.button_travel, config.max_gear
        )
        names = cockpit.names
        self.mapper = ControlMapper(
            model,
            (
                Binding(
                    "throttle",
                    SensorValue(names.throttle_position),
                    UnipolarCalibration(0, cockpit.throttle.travel, deadzone=0.02),
                ),
                Binding(
                    "brake",
                    SensorValue(names.brake_position),
                    UnipolarCalibration(0, cockpit.brake.travel, deadzone=0.02),
                ),
                Binding(
                    "steer",
                    SensorValue(names.steering_position),
                    BipolarCalibration(
                        -cockpit.steering.angle_limit,
                        0,
                        cockpit.steering.angle_limit,
                        deadzone=0.02,
                    ),
                ),
                Binding(
                    "hand_brake",
                    SensorValue("park_button_position"),
                    UnipolarCalibration(0, config.button_travel, deadzone=0.3),
                ),
                Binding(
                    "reverse",
                    SensorValue("reverse_button_position"),
                    UnipolarCalibration(0, config.button_travel, deadzone=0.3),
                ),
                Binding("gear", CallbackSource(self.shifter.read_gear)),
                Binding("manual_gear_shift", CallbackSource(self.shifter.read_manual)),
            ),
            brake_priority=True,
        )
        self._reflexes = _reflexes(model, config)
        self._substeps = round(config.control_dt / config.timestep)
        driving = {
            muscle
            for agonists, antagonists in _muscle_groups().values()
            for muscle in agonists + antagonists
        }
        for side in "rl":
            driving.update(_side(ELBOW_FLEXORS + ELBOW_EXTENSORS, side))
        self._driving_muscles = sorted(model.actuator(name).id for name in driving)
        self.reset()

    def reset(self) -> mujoco.MjData:
        """Restore the settled seated pose, reflex state and automatic gear."""
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.rest_qpos
        self.data.act[:] = self.rest_act
        mujoco.mj_forward(self.model, self.data)
        for side, reflex in self._reflexes.items():
            reflex.integral = self.rest_integrals[side]
        self.shifter.reset()
        return self.data

    def advance(self, intent: DriverIntent) -> None:
        """Apply one control period of muscle excitation; no external forces."""
        if not isinstance(intent, DriverIntent):
            raise TypeError("intent must be a DriverIntent")
        if np.any(self.data.qfrc_applied) or np.any(self.data.xfrc_applied):
            raise RuntimeError("External forces are forbidden for the muscle driver")
        setpoints = {"r": REST_ELBOW - intent.steer, "l": REST_ELBOW + intent.steer}
        levels = {name: getattr(intent, name) for name in _muscle_groups()}
        for _ in range(self._substeps):
            self.data.ctrl[:] = 0
            for side, reflex in self._reflexes.items():
                reflex.apply(self.data, setpoints[side], self.config.timestep)
            _excite(self.model, self.data, self.config, levels)
            mujoco.mj_step(self.model, self.data)

    def measurement(self) -> dict[str, float]:
        """Return mechanism travel, contact force and largest driving-muscle force."""
        names = self.cockpit.names
        values = {
            f"{control}_travel": float(
                self.data.sensor(getattr(names, f"{control}_position")).data[0]
            )
            for control in ("throttle", "brake", "steering")
        }
        values.update(
            {
                f"{control}_contact_newtons": float(
                    self.data.sensor(names.touch_sensor(control)).data[0]
                )
                for control in ("throttle", "brake")
            }
        )
        values["steering_contact_newtons"] = float(
            sum(
                np.linalg.norm(self._contact_force(f"cradle_{side}_{plate}"))
                for side in "rl"
                for plate in ("upper", "lower")
            )
        )
        for name in FINGER_CONTROLS:
            values[f"{name}_button_travel"] = float(
                self.data.sensor(f"{name}_button_position").data[0]
            )
            values[f"{name}_button_contact_newtons"] = float(
                self.data.sensor(f"{name}_button_touch").data[0]
            )
        # Posture-held trunk muscles carry passive force; report the muscles
        # that operate the controls (feet, elbows and button digits).
        values["largest_muscle_newtons"] = float(
            np.max(np.abs(self.data.actuator_force[self._driving_muscles]))
        )
        return values

    def _contact_force(self, geom: str) -> np.ndarray:
        index = self.model.geom(geom).id
        total = np.zeros(3)
        force = np.zeros(6)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if index in (int(contact.geom1), int(contact.geom2)):
                mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                total += force[:3]
        return total


def build_fullbody_driver(
    config: FullBodyDriverConfig | None = None,
) -> FullBodyDriver:
    """Compose the seated full body, cockpit and finger buttons into one model.

    Requires MyoSuite's ``ModelBuilder`` and ``myo_sim`` with ``myofullbody``.
    The body first settles in its seated posture; mechanisms are then placed
    from its kinematics and fingertip paths, and moved along their travel
    axes until each is ``config.clearance`` from its limb.

    Args:
        config: Mechanics and timing; defaults to ``FullBodyDriverConfig()``.

    Returns:
        A driver at its settled rest pose with all controls released.
    """
    config = config if config is not None else FullBodyDriverConfig()
    if not isinstance(config, FullBodyDriverConfig):
        raise TypeError("config must be a FullBodyDriverConfig")
    body = _body_spec(config).compile()
    rest, integrals = _settle(body, config)
    layout = _layout(body, rest, config, integrals)
    checks = _clearance_checks()
    shifts = {key: 0.0 for key in checks}
    gaps: dict[str, float] = {}
    for iteration in range(3):
        model, cockpit = _build(config, layout, shifts)
        data = mujoco.MjData(model)
        _copy_pose(body, rest.qpos, model, data)
        mujoco.mj_forward(model, data)
        gaps = {key: _gap(model, data, *check) for key, check in checks.items()}
        if iteration < 2:
            # Buttons stay on the traced fingertip path; only pedals and
            # cradle plates are moved along their axes to the clearance.
            for key, gap in gaps.items():
                if key not in FINGER_CONTROLS:
                    shifts[key] += config.clearance - gap
    act = np.zeros(model.na)
    for index in range(body.nu):
        act[model.actuator(body.actuator(index).name).id] = rest.act[index]
    return FullBodyDriver(
        model, cockpit, config, (data.qpos.copy(), act), integrals, gaps
    )


def summary(driver: FullBodyDriver) -> dict[str, Any]:
    """Return model sizes and initial clearances for evidence reports."""
    model = driver.model
    return {
        "body": "myo_sim:myofullbody",
        "muscles": int(
            np.count_nonzero(model.actuator_dyntype == mujoco.mjtDyn.mjDYN_MUSCLE)
        ),
        "nq": int(model.nq),
        "timestep": float(model.opt.timestep),
        "control_dt": driver.config.control_dt,
        "initial_gaps_metres": dict(driver.initial_gaps),
        "config": {
            key: value
            for key, value in vars(replace(driver.config)).items()
            if not key.startswith("_")
        },
    }

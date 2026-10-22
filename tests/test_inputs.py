"""Physical inputs are exercised against real MuJoCo models and constraints."""

from __future__ import annotations

import math
from collections.abc import Callable
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite_carla.controls import VehicleCommand
from myosuite_carla.inputs import (
    Binding,
    BipolarCalibration,
    CallbackSource,
    ContactForce,
    ControlMapper,
    IdentityCalibration,
    JointPosition,
    SensorValue,
    UnipolarCalibration,
)


@pytest.fixture
def mechanism() -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Build scalar joints after a free joint to catch qpos-address mistakes."""
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="root" pos="0 0 3">
              <freejoint name="root_free"/>
              <geom size="0.1" mass="1"/>
            </body>
            <body name="wheel" pos="1 0 3">
              <joint name="steering" type="hinge" axis="0 1 0"/>
              <geom name="wheel_geom" size="0.1" mass="1"/>
            </body>
            <body name="pedal" pos="2 0 3">
              <joint name="pedal_travel" type="slide" axis="0 0 1"/>
              <geom name="pedal_geom" size="0.1" mass="1"/>
            </body>
            <body name="ball" pos="3 0 3">
              <joint name="ball_joint" type="ball"/>
              <geom size="0.1" mass="1"/>
            </body>
          </worldbody>
          <sensor>
            <jointpos name="pedal_position" joint="pedal_travel"/>
            <framepos name="wheel_position" objtype="body" objname="wheel"/>
          </sensor>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.mark.parametrize("released, pressed", [(0.0, 1.0), (2.0, -2.0)])
def test_unipolar_endpoints_and_saturation(released: float, pressed: float) -> None:
    calibration = UnipolarCalibration(released, pressed)
    assert calibration(released) == 0.0
    assert calibration(pressed) == 1.0
    assert calibration((released + pressed) / 2) == 0.5
    span = pressed - released
    assert calibration(released - span) == 0.0
    assert calibration(pressed + span) == 1.0


def test_unipolar_deadzone_and_exponent() -> None:
    calibration = UnipolarCalibration(0.0, 10.0, deadzone=0.2, exponent=2.0)
    assert calibration(2.0) == 0.0
    assert calibration(6.0) == pytest.approx(0.25)
    assert calibration(10.0) == 1.0


def test_bipolar_asymmetry_direction_and_deadzone() -> None:
    calibration = BipolarCalibration(-2.0, 1.0, 5.0, deadzone=0.2, exponent=2.0)
    assert calibration(-20.0) == -1.0
    assert calibration(50.0) == 1.0
    assert calibration(1.0) == 0.0
    assert calibration(0.5) == 0.0
    assert calibration(1.5) == 0.0
    assert calibration(-0.8) == pytest.approx(-0.25)
    assert calibration(3.4) == pytest.approx(0.25)
    inverted = BipolarCalibration(-2.0, 1.0, 5.0, invert=True)
    assert inverted(-2.0) == 1.0
    assert inverted(5.0) == -1.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: UnipolarCalibration(0.0, 0.0),
        lambda: UnipolarCalibration(0.0, math.inf),
        lambda: UnipolarCalibration(-1e308, 1e308),
        lambda: UnipolarCalibration(0.0, 1.0, deadzone=-0.1),
        lambda: UnipolarCalibration(0.0, 1.0, deadzone=1.0),
        lambda: UnipolarCalibration(0.0, 1.0, deadzone=math.nan),
        lambda: UnipolarCalibration(0.0, 1.0, exponent=0.0),
        lambda: UnipolarCalibration(0.0, 1.0, exponent=math.inf),
        lambda: BipolarCalibration(0.0, 0.0, 1.0),
        lambda: BipolarCalibration(1.0, 0.0, -1.0),
        lambda: BipolarCalibration(-1.0, math.nan, 1.0),
        lambda: BipolarCalibration(-1e308, 1e308, 1.5e308),
        lambda: BipolarCalibration(-1.0, 0.0, 1.0, deadzone=1.0),
        lambda: BipolarCalibration(-1.0, 0.0, 1.0, exponent=-1.0),
    ],
)
def test_invalid_calibration_configuration(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_calibration_readings(value: float) -> None:
    for calibration in (
        IdentityCalibration(),
        UnipolarCalibration(0.0, 1.0),
        BipolarCalibration(-1.0, 0.0, 1.0),
    ):
        with pytest.raises(ValueError, match="finite"):
            calibration(value)


def test_joint_position_uses_qpos_address(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    model, data = mechanism
    steering_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steering")
    assert steering_id == 1
    assert model.jnt_qposadr[steering_id] == 7
    data.qpos[7] = -0.7
    mapper = ControlMapper(
        model,
        [Binding("steer", JointPosition("steering"), BipolarCalibration(-1, 0, 1))],
    )
    assert mapper.sample(data).command.steer == pytest.approx(-0.7)
    data.qpos[7] = 0.4
    mapper.reset()
    assert mapper.sample(data).command.steer == pytest.approx(0.4)
    data.qpos[8] = 0.2
    assert JointPosition("pedal_travel").bind(model)(data) == 0.2


@pytest.mark.parametrize("name", ["steering", "pedal_travel"])
@pytest.mark.parametrize("integer_only_enums", [False, True])
def test_scalar_joint_types_bind_across_python_numpy_enum_versions(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
    name: str,
    integer_only_enums: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, data = mechanism
    joint = model.joint(name)
    data.qpos[int(joint.qposadr[0])] = 0.37
    if integer_only_enums:

        class IntegerEnum:
            def __init__(self, value: int) -> None:
                self.value = value

            def __int__(self) -> int:
                return self.value

        monkeypatch.setattr(
            mujoco,
            "mjtJoint",
            SimpleNamespace(
                mjJNT_HINGE=IntegerEnum(int(mujoco.mjtJoint.mjJNT_HINGE)),
                mjJNT_SLIDE=IntegerEnum(int(mujoco.mjtJoint.mjJNT_SLIDE)),
            ),
        )
    assert JointPosition(name).bind(model)(data) == pytest.approx(0.37)


@pytest.mark.parametrize("name", ["missing", "root_free", "ball_joint"])
def test_invalid_joint_names_and_nonscalar_joints(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData], name: str
) -> None:
    with pytest.raises(ValueError, match="(No MuJoCo joint|scalar)"):
        JointPosition(name).bind(mechanism[0])


def test_sensor_addresses_and_vector_components(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    model, data = mechanism
    data.qpos[8] = 0.6
    mujoco.mj_forward(model, data)
    mapper = ControlMapper(
        model,
        [
            Binding("throttle", SensorValue("pedal_position")),
            Binding("wheel_z", SensorValue("wheel_position", component=2)),
        ],
    )
    sample = mapper.sample(data)
    assert sample.command.throttle == pytest.approx(0.6)
    assert sample.values["wheel_z"] == 3.0


@pytest.mark.parametrize(
    "source,error",
    [
        (SensorValue("missing"), ValueError),
        (SensorValue("pedal_position", -1), ValueError),
        (SensorValue("pedal_position", 1), ValueError),
        (SensorValue("wheel_position", 3), ValueError),
        (SensorValue("wheel_position", 1.2), TypeError),
        (SensorValue("wheel_position", True), TypeError),
    ],
)
def test_bad_sensors(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData], source: SensorValue, error: type
) -> None:
    with pytest.raises(error):
        source.bind(mechanism[0])


def constant(value: float) -> CallbackSource:
    """Return a constant application-defined signal."""
    return CallbackSource(lambda model, data: value)


def test_brake_priority_preserves_calibrated_values(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    model, data = mechanism
    bindings = [Binding("throttle", constant(0.7)), Binding("brake", constant(0.2))]
    sample = ControlMapper(model, bindings).sample(data)
    assert sample.command.throttle == 0.0
    assert sample.command.brake == 0.2
    assert sample.values == {"throttle": 0.7, "brake": 0.2}
    assert (
        ControlMapper(model, bindings, brake_priority=False)
        .sample(data)
        .command.throttle
        == 0.7
    )


def test_optional_binary_gear_and_custom_channels(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    model, data = mechanism
    sample = ControlMapper(
        model,
        [
            Binding("hand_brake", constant(0.5)),
            Binding("reverse", constant(0.49)),
            Binding("manual_gear_shift", constant(1.0)),
            Binding("gear", constant(-1.0)),
            Binding("indicator_left", constant(4.2)),
        ],
    ).sample(data)
    assert sample.command == VehicleCommand(
        hand_brake=True, reverse=False, manual_gear_shift=True, gear=-1
    )
    assert sample.values["indicator_left"] == 4.2
    assert ControlMapper(model, []).sample(data).command == VehicleCommand()


@pytest.mark.parametrize(
    "channel,value",
    [
        ("throttle", 1.1),
        ("steer", -1.1),
        ("brake", -0.1),
        ("hand_brake", -0.1),
        ("reverse", 1.1),
        ("manual_gear_shift", 2.0),
        ("gear", 1.2),
        ("gear", float(2**31)),
        ("custom", math.nan),
        ("throttle", math.inf),
    ],
)
def test_invalid_sampled_channels(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData], channel: str, value: float
) -> None:
    model, data = mechanism
    mapper = ControlMapper(model, [Binding(channel, constant(value))])
    with pytest.raises(ValueError):
        mapper.sample(data)


def test_invalid_raw_joint_and_calibration_results(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    model, data = mechanism
    data.qpos[7] = math.nan
    mapper = ControlMapper(
        model,
        [Binding("steer", JointPosition("steering"), BipolarCalibration(-1, 0, 1))],
    )
    with pytest.raises(ValueError, match="finite"):
        mapper.sample(data)
    mapper = ControlMapper(
        model, [Binding("custom", constant(0.0), lambda value: math.nan)]
    )
    with pytest.raises(ValueError, match="calibrated custom"):
        mapper.sample(data)
    data.sensordata[0] = math.inf
    mapper = ControlMapper(model, [Binding("custom", SensorValue("pedal_position"))])
    with pytest.raises(ValueError, match="finite"):
        mapper.sample(data)


def test_custom_source_protocol_resolves_once_and_reads_current_data(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    model, data = mechanism

    class TimeSource:
        """Implement an application-defined source without a wrapper."""

        bind_calls = 0

        def bind(self, target: mujoco.MjModel) -> Callable[[mujoco.MjData], float]:
            """Resolve a reader for the selected model."""
            assert target is model
            self.bind_calls += 1
            return lambda state: state.time

    source = TimeSource()
    mapper = ControlMapper(model, [Binding("time", source)])
    assert mapper.sample(data).values["time"] == 0.0
    data.time = 3.0
    assert mapper.sample(data).values["time"] == 3.0
    assert source.bind_calls == 1


def test_duplicate_channels_rejected(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData],
) -> None:
    with pytest.raises(ValueError, match="one binding"):
        ControlMapper(
            mechanism[0],
            [Binding("brake", constant(0.1)), Binding("brake", constant(0.2))],
        )


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"throttle": math.nan}, ValueError),
        ({"steer": math.inf}, ValueError),
        ({"brake": 1.01}, ValueError),
        ({"throttle": -0.01}, ValueError),
        ({"hand_brake": 1.0}, TypeError),
        ({"gear": 1.5}, TypeError),
        ({"gear": True}, TypeError),
        ({"gear": 2**31}, ValueError),
    ],
)
def test_vehicle_command_validation(kwargs: dict, error: type) -> None:
    with pytest.raises(error):
        VehicleCommand(**kwargs)


def test_vehicle_command_dictionary() -> None:
    command = VehicleCommand(throttle=np.float64(0.4), gear=np.int64(-1))
    assert command.as_dict() == {
        "throttle": 0.4,
        "steer": 0.0,
        "brake": 0.0,
        "hand_brake": False,
        "reverse": False,
        "manual_gear_shift": False,
        "gear": -1,
    }


def test_contact_force_reads_real_constraint_force_and_pair_selection() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <geom name="floor" type="plane" size="3 3 0.1"/>
            <body pos="0 0 0.1">
              <freejoint/>
              <geom name="foot" type="box" size="0.05 0.04 0.02" mass="1"/>
            </body>
            <body pos="1 0 0.1">
              <freejoint/>
              <geom name="other_foot" type="sphere" size="0.05" mass="2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    force = ContactForce("foot", "floor").bind(model)
    reverse = ContactForce("floor", "foot").bind(model)
    other = ContactForce("floor", "other_foot").bind(model)
    assert data.ncon == 5
    assert force(data) == pytest.approx(9.81, abs=0.001)
    assert reverse(data) == force(data)
    assert other(data) == pytest.approx(19.62, abs=0.001)
    mapper = ControlMapper(
        model,
        [Binding("brake", ContactForce("foot", "floor"), UnipolarCalibration(0, 20))],
    )
    assert mapper.sample(data).command.brake == pytest.approx(9.81 / 20, abs=0.001)
    data.qpos[2] = 2.0
    mujoco.mj_forward(model, data)
    assert force(data) == 0.0


@pytest.mark.parametrize(
    "geoms", [("missing", "pedal_geom"), ("wheel_geom", "wheel_geom")]
)
def test_contact_force_invalid_geoms(
    mechanism: tuple[mujoco.MjModel, mujoco.MjData], geoms: tuple[str, str]
) -> None:
    with pytest.raises(ValueError):
        ContactForce(*geoms).bind(mechanism[0])

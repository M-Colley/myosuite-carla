"""Per-frame recordings and their open exports."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from myosuite_carla import DriveRecorder, Recording, VehicleTelemetry
from myosuite_carla.recording import ARM_FORCE_COLUMNS

MODEL = """
<mujoco>
  <worldbody>
    <body name="upper">
      <joint name="elbow_flexion_r" type="hinge" axis="0 1 0" range="0 2"/>
      <geom type="capsule" size=".02" fromto="0 0 0 0 0 -.3" mass="1"/>
      <body name="slider" pos="0 0 -.3">
        <joint name="knee_translation" type="slide" axis="1 0 0"/>
        <geom size=".02" mass=".1"/>
      </body>
    </body>
    <body name="cockpit_pedal">
      <joint name="cockpit_throttle_joint" type="slide" axis="0 0 1"/>
      <geom size=".02" mass=".1"/>
    </body>
  </worldbody>
  <actuator>
    <muscle name="BIClong" joint="elbow_flexion_r"/>
    <muscle name="TRIlong" joint="elbow_flexion_r"/>
  </actuator>
</mujoco>
"""


def _telemetry(frame: int, yaw: float, x: float) -> VehicleTelemetry:
    # Heading 90 degrees: forward is world +y and right is world -x.
    return VehicleTelemetry(
        frame=frame,
        elapsed_seconds=0.02 * frame,
        location=(x, 2.0 * frame, 0.0),
        rotation=(0.0, yaw, 0.0),
        velocity=(0.0, 3.0, 0.0),
        acceleration=(-2.0, 1.0, 0.0),
        angular_velocity=(0.0, 0.0, 5.0),
    )


def _recorded(frames: int = 5, body: bool = True) -> Recording:
    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    recorder = DriveRecorder(body=body, metadata={"map": "Town10HD_Opt"})
    for index in range(frames):
        data.ctrl[:] = (0.4, 0.1)
        mujoco.mj_step(model, data)
        recorder.add(
            data.time,
            _telemetry(index, 90.0, 0.0),
            command={"throttle": 0.3, "steer": 0.0, "hand_brake": False},
            inputs={"throttle": 0.3},
            applied={"throttle": 0.3, "hand_brake": False},
            intent={"right_foot": 0.012, "steer": 0.1},
            mechanics={"steering_travel": 0.05, "brake_contact_newtons": 2.0},
            phase="throttle" if index < 3 else "brake",
            model=model,
            data=data,
        )
    return recorder.recording()


def test_recorder_builds_named_columns_and_body_arrays() -> None:
    recording = _recorded()
    assert len(recording) == 5
    columns = recording.columns
    assert list(columns["phase"]) == ["throttle"] * 3 + ["brake"] * 2
    np.testing.assert_allclose(columns["a_forward_mps2"], 1.0, atol=1e-12)
    # World -x is the vehicle's right at heading 90 degrees: right-positive.
    np.testing.assert_allclose(columns["a_right_mps2"], 2.0, atol=1e-12)
    np.testing.assert_allclose(columns["speed_mps"], 3.0)
    assert columns["applied_hand_brake"][0] == 0.0
    # Mechanism joints are not body coordinates; slides stay in metres.
    assert recording.joints == ("elbow_flexion_r", "knee_translation")
    assert recording.joint_kinds == ("hinge", "slide")
    assert recording.muscles == ("BIClong", "TRIlong")
    assert recording.coordinates.shape == (5, 2)
    assert np.all(recording.activation[-1] > 0)
    # Tension is reported positive although MuJoCo's actuator force is negative.
    assert np.all(recording.force[-1] >= 0) and recording.force[-1, 0] > 0
    np.testing.assert_allclose(
        columns["effort_sum_activation_squared"],
        np.sum(recording.activation**2, axis=1),
    )


@pytest.mark.parametrize("integer_only_enums", [False, True])
def test_body_recording_normalizes_mujoco_enum_values(
    monkeypatch: pytest.MonkeyPatch, integer_only_enums: bool
) -> None:
    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    data.qpos[:] = (0.01, 0.12, 0.05)
    data.act[:] = (0.6, 0.2)
    mujoco.mj_forward(model, data)
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
        monkeypatch.setattr(
            mujoco,
            "mjtDyn",
            SimpleNamespace(mjDYN_MUSCLE=IntegerEnum(int(mujoco.mjtDyn.mjDYN_MUSCLE))),
        )
    recorder = DriveRecorder()
    recorder.add(0.0, _telemetry(0, 0.0, 0.0), command={}, model=model, data=data)
    result = recorder.recording()
    assert result.joints == ("elbow_flexion_r", "knee_translation")
    assert result.joint_kinds == ("hinge", "slide")
    assert result.muscles == ("BIClong", "TRIlong")
    np.testing.assert_array_equal(result.coordinates, [[0.01, 0.12]])
    np.testing.assert_array_equal(result.activation, [[0.6, 0.2]])


def test_recorder_rejects_changing_fields_and_empty_use() -> None:
    recorder = DriveRecorder(body=False)
    with pytest.raises(ValueError, match="Nothing"):
        recorder.recording()
    recorder.add(0.02, _telemetry(1, 0.0, 0.0), command={"throttle": 0.1})
    with pytest.raises(ValueError, match="same fields"):
        recorder.add(0.04, _telemetry(2, 0.0, 0.0), command={"brake": 0.1})
    with pytest.raises(ValueError, match="model"):
        DriveRecorder().add(
            0.04,
            _telemetry(2, 0.0, 0.0),
            command={"throttle": 0.1},
            data=SimpleNamespace(),
        )


def test_save_and_load_round_trip_with_units(tmp_path: Path) -> None:
    recording = _recorded()
    written = recording.save(tmp_path)
    assert {"frames", "coordinates", "activation", "force", "datapackage"} <= set(
        written
    )
    package = json.loads((tmp_path / "datapackage.json").read_text())
    assert package["profile"] == "tabular-data-package"
    fields = {
        item["name"]: item
        for item in next(
            resource
            for resource in package["resources"]
            if resource["name"] == "frames"
        )["schema"]["fields"]
    }
    assert fields["speed_mps"]["unit"] == "m/s"
    assert fields["mech_steering_travel"]["unit"] == "rad"
    assert fields["mech_brake_contact_newtons"]["unit"] == "N"
    assert fields["phase"]["type"] == "string"
    assert fields["frame"]["type"] == "integer"
    loaded = Recording.load(tmp_path)
    assert loaded.joints == recording.joints and loaded.muscles == recording.muscles
    np.testing.assert_array_equal(loaded.coordinates, recording.coordinates)
    np.testing.assert_array_equal(loaded.force, recording.force)
    np.testing.assert_array_equal(loaded.time, recording.time)
    assert list(loaded.columns["phase"]) == list(recording.columns["phase"])
    assert loaded.metadata["map"] == "Town10HD_Opt"


def test_arm_force_recording_matches_its_schema(tmp_path: Path) -> None:
    _recorded().save(tmp_path, formats=("csv", "arm_force"))
    manifest = json.loads((tmp_path / "arm_force" / "manifest.json").read_text())
    assert manifest["schema"] == 1
    assert manifest["trajectory_columns"] == list(ARM_FORCE_COLUMNS)
    assert manifest["fps"] == pytest.approx(1 / 0.002)
    with (tmp_path / "arm_force" / "trajectory.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == ARM_FORCE_COLUMNS
    assert float(rows[0]["time_s"]) == 0.0
    assert float(rows[0]["ay_mps2"]) == pytest.approx(2.0)
    assert float(rows[0]["heading_rad"]) == pytest.approx(math.pi / 2)
    assert math.isnan(float(rows[0]["lon"]))
    progress = [float(row["progress_frac"]) for row in rows]
    assert progress[0] == 0.0 and progress[-1] == 1.0
    assert progress == sorted(progress)


def test_opensim_storage_files_load_in_opensim(tmp_path: Path) -> None:
    recording = _recorded()
    recording.save(tmp_path, formats=("csv", "opensim"))
    text = (tmp_path / "opensim" / "coordinates.mot").read_text().splitlines()
    assert text[1:6] == [
        "version=1",
        "nRows=5",
        "nColumns=3",
        "inDegrees=yes",
        "endheader",
    ]
    assert text[6].split("\t") == ["time", "elbow_flexion_r", "knee_translation"]
    first = [float(value) for value in text[7].split("\t")]
    assert first[1] == pytest.approx(math.degrees(recording.coordinates[0, 0]))
    assert first[2] == pytest.approx(recording.coordinates[0, 1])
    opensim = pytest.importorskip("opensim")
    table = opensim.TimeSeriesTable(str(tmp_path / "opensim" / "coordinates.mot"))
    assert list(table.getColumnLabels()) == list(recording.joints)
    assert table.getNumRows() == 5
    assert table.getTableMetaDataAsString("inDegrees") == "yes"
    forces = opensim.TimeSeriesTable(str(tmp_path / "opensim" / "force.sto"))
    column = forces.getDependentColumn("BIClong").to_numpy()
    np.testing.assert_allclose(column, recording.force[:, 0])


def test_parquet_and_mcap_exports(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    pandas = pytest.importorskip("pandas")
    reader_module = pytest.importorskip("mcap.reader")
    recording = _recorded()
    written = recording.save(tmp_path, formats=("csv", "parquet", "mcap"))
    frame = pandas.read_parquet(written["parquet"])
    assert len(frame) == 5
    assert {"speed_mps", "elbow_flexion_r", "BIClong_activation"} <= set(frame)
    with written["mcap"].open("rb") as handle:
        reader = reader_module.make_reader(handle)
        messages: dict[str, list[dict]] = {}
        for _, channel, message in reader.iter_messages():
            messages.setdefault(channel.topic, []).append(json.loads(message.data))
    assert set(messages) == {
        "/drive/frame",
        "/body/coordinates",
        "/body/activation",
        "/body/force",
    }
    assert len(messages["/drive/frame"]) == 5
    assert messages["/body/force"][-1]["BIClong"] == pytest.approx(
        recording.force[-1, 0]
    )
    with pytest.raises(ValueError, match="Unknown"):
        recording.save(tmp_path, formats=("csv", "xlsx"))


def test_live_start_adds_geolocation() -> None:
    location = SimpleNamespace
    geo = SimpleNamespace(
        name="Town10HD_Opt",
        transform_to_geolocation=lambda loc: SimpleNamespace(
            longitude=loc.x * 1e-5, latitude=49.0 + loc.y * 1e-5
        ),
    )
    session = SimpleNamespace(
        _api=SimpleNamespace(
            Location=lambda x, y, z: location(x=x, y=y, z=z),
        ),
        world=SimpleNamespace(get_map=lambda: geo),
        vehicle=SimpleNamespace(type_id="vehicle.dodge.charger"),
        config=SimpleNamespace(fixed_delta_seconds=0.02),
    )
    recorder = DriveRecorder(body=False)
    recorder.start(session, None)
    for index in range(3):
        recorder.add(0.02 * index, _telemetry(index, 0.0, 100.0), command={})
    recording = recorder.recording()
    np.testing.assert_allclose(recording.columns["lon_deg"], 1e-3)
    np.testing.assert_allclose(recording.columns["lat_deg"], [49, 49.00002, 49.00004])
    assert recording.metadata["vehicle"] == "vehicle.dodge.charger"

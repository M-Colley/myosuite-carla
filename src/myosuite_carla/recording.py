"""Per-frame recordings of a driven body and its vehicle, in open formats.

A :class:`DriveRecorder` collects one row per control period: the driver
intent, measured mechanism inputs, the command sent to the vehicle, the
controls the vehicle applied, vehicle kinematics and mechanism measurements.
Body state (joint coordinates, muscle activation and force) is kept as named
arrays. :meth:`Recording.save` writes:

* ``frames.csv`` and body CSVs described by a Frictionless Data Package
  (``datapackage.json``) with units, descriptions and provenance;
* OpenSim Storage tables (``coordinates.mot`` in degrees, ``activation.sto``,
  ``force.sto``) labelled with the coordinate and muscle names MyoSuite's
  models share with their OpenSim sources;
* an arm_force_driving CARLA recording (``trajectory.csv`` and
  ``manifest.json`` schema 1; lateral acceleration right-positive);
* optionally Parquet (pandas with pyarrow) and MCAP (``mcap``) for Foxglove.

Vehicle quantities follow CARLA: a left-handed world, rotations as (pitch,
yaw, roll) in degrees and angular velocity in degrees per second.
"""

from __future__ import annotations

import csv
import importlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from .vehicle_loads import carla_axes

FORMATS = ("csv", "opensim", "arm_force", "parquet", "mcap")
_DEFAULT_FORMATS = ("csv", "opensim", "arm_force")

_VEHICLE_FIELDS: dict[str, tuple[str, str]] = {
    "x_m": ("m", "Vehicle location x, CARLA world frame"),
    "y_m": ("m", "Vehicle location y, CARLA world frame (left-handed)"),
    "z_m": ("m", "Vehicle location z, CARLA world frame"),
    "pitch_deg": ("deg", "Vehicle pitch"),
    "yaw_deg": ("deg", "Vehicle yaw (heading)"),
    "roll_deg": ("deg", "Vehicle roll"),
    "vx_mps": ("m/s", "World velocity x"),
    "vy_mps": ("m/s", "World velocity y"),
    "vz_mps": ("m/s", "World velocity z"),
    "ax_world_mps2": ("m/s^2", "World acceleration x"),
    "ay_world_mps2": ("m/s^2", "World acceleration y"),
    "az_world_mps2": ("m/s^2", "World acceleration z"),
    "wx_dps": ("deg/s", "World angular velocity x"),
    "wy_dps": ("deg/s", "World angular velocity y"),
    "wz_dps": ("deg/s", "World angular velocity z (yaw rate)"),
    "speed_mps": ("m/s", "Vehicle speed"),
    "a_forward_mps2": ("m/s^2", "Acceleration along the vehicle's forward axis"),
    "a_right_mps2": ("m/s^2", "Acceleration along the vehicle's right axis"),
    "a_up_mps2": ("m/s^2", "Acceleration along the vehicle's up axis"),
}
_PREFIXES = {
    "intent_": "Driver intent (muscle excitation or reflex offset)",
    "input_": "Measured mechanism input after calibration",
    "command_": "Vehicle command sent this frame",
    "applied_": "Control the vehicle reported as applied",
    "mech_": "Mechanism measurement",
}


def _unit(name: str) -> str:
    if name in _VEHICLE_FIELDS:
        return _VEHICLE_FIELDS[name][0]
    if name in ("time_s", "carla_elapsed_s"):
        return "s"
    if name in ("lon_deg", "lat_deg"):
        return "deg"
    if name.endswith("_newtons"):
        return "N"
    if name == "mech_steering_travel" or name == "intent_steer":
        return "rad"
    if name.endswith("_travel"):
        return "m"
    return "1"


def _description(name: str) -> str:
    fixed = {
        "time_s": "Driver simulation time at the end of the control period",
        "frame": "Simulator frame number",
        "carla_elapsed_s": "Vehicle simulator elapsed time",
        "phase": "Scenario phase label",
        "lon_deg": "Vehicle longitude from the map's geo-reference",
        "lat_deg": "Vehicle latitude from the map's geo-reference",
        "effort_sum_activation_squared": (
            "Sum over muscles of squared activation, a common effort measure"
        ),
    }
    if name in fixed:
        return fixed[name]
    if name in _VEHICLE_FIELDS:
        return _VEHICLE_FIELDS[name][1]
    for prefix, text in _PREFIXES.items():
        if name.startswith(prefix):
            return f"{text}: {name[len(prefix) :]}"
    return name


def _is_mechanism_joint(name: str) -> bool:
    return name.startswith(("cockpit_", "cradle_")) or name.endswith("_button_joint")


def _get(vehicle: Any, name: str) -> Any:
    return vehicle[name] if isinstance(vehicle, Mapping) else getattr(vehicle, name)


def _vehicle_columns(vehicle: Any) -> dict[str, float]:
    location, rotation = _get(vehicle, "location"), _get(vehicle, "rotation")
    velocity = _get(vehicle, "velocity")
    acceleration = _get(vehicle, "acceleration")
    rate = _get(vehicle, "angular_velocity")
    local = carla_axes(rotation) @ np.asarray(acceleration, dtype=float)
    values = (
        *location,
        *rotation,
        *velocity,
        *acceleration,
        *rate,
        math.hypot(*velocity),
        *local,
    )
    return {name: float(value) for name, value in zip(_VEHICLE_FIELDS, values)}


@dataclass
class Recording:
    """A finished recording: scalar columns plus named body arrays.

    Args:
        columns: Equal-length columns; ``phase`` holds strings, others numbers.
        joints: Body coordinate names (MuJoCo joint names).
        joint_kinds: ``"hinge"`` (radians) or ``"slide"`` (metres) per joint.
        muscles: Muscle names (MuJoCo actuator names).
        coordinates: ``(frames, joints)`` coordinate values.
        activation: ``(frames, muscles)`` activation in [0, 1].
        force: ``(frames, muscles)`` muscle tension in newtons (positive pulls).
        metadata: Provenance such as map, vehicle and software versions.
    """

    columns: dict[str, NDArray[Any]]
    joints: tuple[str, ...] = ()
    joint_kinds: tuple[str, ...] = ()
    muscles: tuple[str, ...] = ()
    coordinates: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    activation: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    force: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0)))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {len(values) for values in self.columns.values()}
        if len(lengths) > 1:
            raise ValueError("Recording columns differ in length")
        if "time_s" not in self.columns:
            raise ValueError("Recording needs a time_s column")
        frames = len(self.columns["time_s"])
        if self.joints and self.coordinates.shape != (frames, len(self.joints)):
            raise ValueError("coordinates must be (frames, joints)")
        if len(self.joint_kinds) != len(self.joints):
            raise ValueError("joint_kinds must name a kind for every joint")
        for name in ("activation", "force"):
            array = getattr(self, name)
            if self.muscles and array.shape != (frames, len(self.muscles)):
                raise ValueError(f"{name} must be (frames, muscles)")

    @property
    def time(self) -> NDArray[np.float64]:
        """Return the time column in seconds."""
        return np.asarray(self.columns["time_s"], dtype=float)

    def __len__(self) -> int:
        return len(self.columns["time_s"])

    def to_pandas(self, body: bool = False) -> Any:
        """Return a pandas DataFrame of the columns, optionally with body arrays."""
        pandas = importlib.import_module("pandas")
        frame = pandas.DataFrame(self.columns)
        if body and self.joints:
            frame = pandas.concat(
                [
                    frame,
                    pandas.DataFrame(self.coordinates, columns=list(self.joints)),
                    pandas.DataFrame(
                        self.activation,
                        columns=[f"{name}_activation" for name in self.muscles],
                    ),
                    pandas.DataFrame(
                        self.force, columns=[f"{name}_force" for name in self.muscles]
                    ),
                ],
                axis=1,
            )
        return frame

    def save(
        self, directory: str | Path, formats: Sequence[str] = _DEFAULT_FORMATS
    ) -> dict[str, Path]:
        """Write the recording; return the path of each written artefact.

        Args:
            directory: Output directory, created if needed.
            formats: Any of ``csv``, ``opensim``, ``arm_force``, ``parquet`` and
                ``mcap``. ``csv`` is always written because the data package
                describes it.
        """
        unknown = set(formats) - set(FORMATS)
        if unknown:
            raise ValueError(f"Unknown formats: {sorted(unknown)}")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        written = _write_tables(self, root)
        if "opensim" in formats and self.joints:
            written.update(_write_opensim(self, root / "opensim"))
        if "arm_force" in formats:
            written.update(_write_arm_force(self, root / "arm_force"))
        if "parquet" in formats:
            path = root / "frames.parquet"
            self.to_pandas(body=True).to_parquet(path, index=False)
            written["parquet"] = path
        if "mcap" in formats:
            written["mcap"] = _write_mcap(self, root / "drive.mcap")
        return written

    @classmethod
    def concatenate(cls, parts: Sequence[Recording]) -> Recording:
        """Join recordings with identical fields; metadata of the first is kept."""
        if not parts:
            raise ValueError("Provide at least one recording")
        first = parts[0]
        for part in parts[1:]:
            if (
                list(part.columns) != list(first.columns)
                or part.joints != first.joints
                or part.muscles != first.muscles
            ):
                raise ValueError("Recordings must have identical fields")
        arrays = {
            name: np.concatenate([part.columns[name] for part in parts])
            for name in first.columns
        }
        body: dict[str, Any] = {}
        if first.joints:
            body = {
                "joints": first.joints,
                "joint_kinds": first.joint_kinds,
                "muscles": first.muscles,
                **{
                    name: np.vstack([getattr(part, name) for part in parts])
                    for name in ("coordinates", "activation", "force")
                },
            }
        return cls(columns=arrays, metadata=dict(first.metadata), **body)

    @classmethod
    def load(cls, directory: str | Path) -> Recording:
        """Read a recording saved by :meth:`save` from its CSV tables."""
        root = Path(directory)
        package = json.loads((root / "datapackage.json").read_text(encoding="utf-8"))
        resources = {item["name"]: item for item in package["resources"]}
        columns = _read_csv(root / resources["frames"]["path"])
        typed: dict[str, NDArray[Any]] = {}
        for item in resources["frames"]["schema"]["fields"]:
            values = columns[item["name"]]
            typed[item["name"]] = (
                np.asarray(values, dtype=object)
                if item["type"] == "string"
                else np.asarray(values, dtype=float)
            )
        body: dict[str, Any] = {}
        meta = package.get("myosuite_carla", {})
        if "coordinates" in resources:
            coordinates = _read_csv(root / resources["coordinates"]["path"])
            joints = tuple(meta["joints"])
            body["joints"] = joints
            body["joint_kinds"] = tuple(meta["joint_kinds"])
            body["coordinates"] = np.column_stack(
                [np.asarray(coordinates[name], dtype=float) for name in joints]
            )
            muscles = tuple(meta["muscles"])
            body["muscles"] = muscles
            for name in ("activation", "force"):
                table = _read_csv(root / resources[name]["path"])
                body[name] = np.column_stack(
                    [np.asarray(table[muscle], dtype=float) for muscle in muscles]
                )
        return cls(columns=typed, metadata=meta.get("metadata", {}), **body)


class DriveRecorder:
    """Collect one row per control period; usable as a ``DriveObserver``.

    Args:
        body: Also record joint coordinates and muscle activation and force.
        metadata: Free-form provenance stored with the recording.
    """

    def __init__(
        self, *, body: bool = True, metadata: Mapping[str, Any] | None = None
    ) -> None:
        self.body = body
        self.metadata = dict(metadata or {})
        self._rows: list[dict[str, Any]] = []
        self._coordinates: list[NDArray[np.float64]] = []
        self._activation: list[NDArray[np.float64]] = []
        self._force: list[NDArray[np.float64]] = []
        self._body_layout: tuple[Any, ...] | None = None
        self._geolocate: Callable[[Sequence[float]], tuple[float, float]] | None = None

    def start(self, session: Any, driver: Any) -> None:
        """Capture map, vehicle and geolocation from a live CARLA session."""
        api = getattr(session, "_api", None) or importlib.import_module("carla")
        world_map = session.world.get_map()
        self.metadata.update(
            map=str(world_map.name),
            vehicle=str(session.vehicle.type_id),
            fixed_delta_seconds=float(session.config.fixed_delta_seconds),
        )

        def geolocate(location: Sequence[float]) -> tuple[float, float]:
            geo = world_map.transform_to_geolocation(api.Location(*location))
            return float(geo.longitude), float(geo.latitude)

        self._geolocate = geolocate

    def frame(self, record: Mapping[str, Any], driver: Any) -> None:
        """Record one frame from ``run_fullbody_validation``'s frame record."""
        self.add(
            record["simulation_time"],
            record["vehicle"],
            command=record["command"],
            inputs=record.get("inputs"),
            applied=record.get("applied"),
            intent=record.get("intent"),
            mechanics=record.get("mechanics"),
            phase=record.get("phase"),
            model=driver.model,
            data=driver.data,
        )

    def close(self) -> None:
        """Nothing to release; call :meth:`recording` for the result."""

    def add(
        self,
        time_s: float,
        vehicle: Any,
        *,
        command: Mapping[str, Any],
        inputs: Mapping[str, Any] | None = None,
        applied: Mapping[str, Any] | None = None,
        intent: Mapping[str, Any] | None = None,
        mechanics: Mapping[str, Any] | None = None,
        phase: str | None = None,
        model: mujoco.MjModel | None = None,
        data: mujoco.MjData | None = None,
    ) -> None:
        """Append one control period.

        Args:
            time_s: Driver simulation time in seconds.
            vehicle: ``VehicleTelemetry`` or its ``as_dict()`` mapping.
            command: The command sent to the vehicle.
            inputs: Measured mechanism inputs.
            applied: Controls the vehicle reported as applied.
            intent: Driver intent fields.
            mechanics: Mechanism measurements.
            phase: Scenario phase label.
            model: Driver model, required with ``data`` for body state.
            data: Driver data whose body state is recorded.
        """
        row: dict[str, Any] = {"time_s": float(time_s)}
        row["frame"] = int(_get(vehicle, "frame"))
        row["carla_elapsed_s"] = float(_get(vehicle, "elapsed_seconds"))
        if phase is not None:
            row["phase"] = str(phase)
        for prefix, values in (
            ("intent_", intent),
            ("input_", inputs),
            ("command_", command),
            ("applied_", applied),
        ):
            for key, value in (values or {}).items():
                row[prefix + key] = float(value)
        row.update(_vehicle_columns(vehicle))
        for key, value in (mechanics or {}).items():
            row["mech_" + key] = float(value)
        if self.body and data is not None:
            if model is None:
                raise ValueError("model is required to record body state")
            self._add_body(model, data, row)
        if self._rows and row.keys() != self._rows[0].keys():
            raise ValueError("Every frame must record the same fields")
        self._rows.append(row)

    def _add_body(
        self, model: mujoco.MjModel, data: mujoco.MjData, row: dict[str, Any]
    ) -> None:
        if self._body_layout is None:
            scalar = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            joints = [
                index
                for index in range(model.njnt)
                if model.jnt_type[index] in scalar
                and not _is_mechanism_joint(model.joint(index).name)
            ]
            muscles = [
                index
                for index in range(model.nu)
                if model.actuator_dyntype[index] == mujoco.mjtDyn.mjDYN_MUSCLE
            ]
            self._body_layout = (
                tuple(model.joint(index).name for index in joints),
                tuple(
                    "hinge"
                    if model.jnt_type[index] == mujoco.mjtJoint.mjJNT_HINGE
                    else "slide"
                    for index in joints
                ),
                np.asarray(model.jnt_qposadr[joints], dtype=int),
                tuple(model.actuator(index).name for index in muscles),
                np.asarray(muscles, dtype=int),
                np.asarray(model.actuator_actadr[muscles], dtype=int),
            )
        _, _, qposadr, _, muscles, actadr = self._body_layout
        activation = np.asarray(data.act[actadr], dtype=float).copy()
        self._coordinates.append(np.asarray(data.qpos[qposadr], dtype=float).copy())
        self._activation.append(activation)
        # MuJoCo reports muscle tension as negative actuator force.
        self._force.append(-np.asarray(data.actuator_force[muscles], dtype=float))
        row["effort_sum_activation_squared"] = float(np.sum(activation**2))

    def __len__(self) -> int:
        return len(self._rows)

    def recording(self) -> Recording:
        """Return the frames recorded so far as a :class:`Recording`."""
        if not self._rows:
            raise ValueError("Nothing has been recorded")
        names = list(self._rows[0])
        columns: dict[str, NDArray[Any]] = {
            name: (
                np.asarray([row[name] for row in self._rows], dtype=object)
                if name == "phase"
                else np.asarray([row[name] for row in self._rows], dtype=float)
            )
            for name in names
        }
        if self._geolocate is not None:
            points = [
                self._geolocate((row["x_m"], row["y_m"], row["z_m"]))
                for row in self._rows
            ]
            columns["lon_deg"] = np.asarray([point[0] for point in points])
            columns["lat_deg"] = np.asarray([point[1] for point in points])
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mujoco_version": mujoco.__version__,
            **self.metadata,
        }
        if self._body_layout is None or not self._coordinates:
            return Recording(columns=columns, metadata=metadata)
        joints, kinds, _, muscles, _, _ = self._body_layout
        return Recording(
            columns=columns,
            joints=joints,
            joint_kinds=kinds,
            muscles=muscles,
            coordinates=np.vstack(self._coordinates),
            activation=np.vstack(self._activation),
            force=np.vstack(self._force),
            metadata=metadata,
        )


def _write_csv(path: Path, header: Sequence[str], rows: Any) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _read_csv(path: Path) -> dict[str, list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        values: list[list[str]] = [[] for _ in header]
        for row in reader:
            for column, value in zip(values, row):
                column.append(value)
    return dict(zip(header, values))


def _cell(value: Any) -> str:
    return value if isinstance(value, str) else repr(float(value))


def _field(name: str, values: NDArray[Any]) -> dict[str, str]:
    kind = "string" if values.dtype == object else "number"
    if name == "frame":
        kind = "integer"
    return {
        "name": name,
        "type": kind,
        "unit": "" if kind == "string" else _unit(name),
        "description": _description(name),
    }


def _write_tables(recording: Recording, root: Path) -> dict[str, Path]:
    names = list(recording.columns)
    columns = [recording.columns[name] for name in names]
    rows = (
        [
            str(int(value)) if name == "frame" else _cell(value)
            for name, value in zip(names, values)
        ]
        for values in zip(*columns)
    )
    _write_csv(root / "frames.csv", names, rows)
    resources: list[dict[str, Any]] = [
        _resource(
            "frames",
            "frames.csv",
            [_field(name, recording.columns[name]) for name in names],
        )
    ]
    written = {"frames": root / "frames.csv"}
    if recording.joints:
        time = recording.time
        for name, header, array, units, text in (
            (
                "coordinates",
                recording.joints,
                recording.coordinates,
                ["rad" if kind == "hinge" else "m" for kind in recording.joint_kinds],
                "Joint coordinate",
            ),
            (
                "activation",
                recording.muscles,
                recording.activation,
                ["1"] * len(recording.muscles),
                "Muscle activation",
            ),
            (
                "force",
                recording.muscles,
                recording.force,
                ["N"] * len(recording.muscles),
                "Muscle tension",
            ),
        ):
            path = root / f"{name}.csv"
            _write_csv(
                path,
                ["time_s", *header],
                (
                    [_cell(moment), *(_cell(value) for value in values)]
                    for moment, values in zip(time, array)
                ),
            )
            fields = [_field("time_s", time)] + [
                {"name": label, "type": "number", "unit": unit, "description": text}
                for label, unit in zip(header, units)
            ]
            resources.append(_resource(name, path.name, fields))
            written[name] = path
    package = {
        "profile": "tabular-data-package",
        "name": "myosuite-carla-drive",
        "created": recording.metadata.get(
            "created_utc", datetime.now(timezone.utc).isoformat()
        ),
        "resources": resources,
        "myosuite_carla": {
            "metadata": recording.metadata,
            "joints": list(recording.joints),
            "joint_kinds": list(recording.joint_kinds),
            "muscles": list(recording.muscles),
            "conventions": {
                "vehicle": "CARLA left-handed world; rotation (pitch, yaw, roll) deg",
                "a_right_mps2": "right-positive vehicle-frame lateral acceleration",
                "muscle_force": "tension in newtons, positive when pulling",
            },
        },
    }
    path = root / "datapackage.json"
    path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    written["datapackage"] = path
    return written


def _resource(name: str, path: str, fields: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "profile": "tabular-data-resource",
        "format": "csv",
        "mediatype": "text/csv",
        "encoding": "utf-8",
        "schema": {"fields": fields},
    }


def _write_storage(
    path: Path,
    time: NDArray[np.float64],
    labels: Sequence[str],
    values: NDArray[np.float64],
    in_degrees: bool,
) -> Path:
    lines = [
        path.stem,
        "version=1",
        f"nRows={len(time)}",
        f"nColumns={len(labels) + 1}",
        f"inDegrees={'yes' if in_degrees else 'no'}",
        "endheader",
        "\t".join(["time", *labels]),
    ]
    lines += [
        "\t".join(repr(float(value)) for value in (moment, *row))
        for moment, row in zip(time, values)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_opensim(recording: Recording, root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    hinge = np.asarray([kind == "hinge" for kind in recording.joint_kinds])
    coordinates = recording.coordinates.copy()
    coordinates[:, hinge] = np.degrees(coordinates[:, hinge])
    time = recording.time
    return {
        "opensim_coordinates": _write_storage(
            root / "coordinates.mot", time, recording.joints, coordinates, True
        ),
        "opensim_activation": _write_storage(
            root / "activation.sto",
            time,
            recording.muscles,
            recording.activation,
            False,
        ),
        "opensim_force": _write_storage(
            root / "force.sto", time, recording.muscles, recording.force, False
        ),
    }


#: arm_force_driving's CARLA recording columns (``trajectory.csv``, schema 1).
ARM_FORCE_COLUMNS = (
    "time_s",
    "ax_mps2",
    "ay_mps2",
    "lon",
    "lat",
    "heading_rad",
    "speed_mps",
    "progress_frac",
)


def _write_arm_force(recording: Recording, root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    columns = recording.columns
    time = recording.time
    steps = np.hypot(np.diff(columns["x_m"]), np.diff(columns["y_m"]))
    travelled = np.concatenate([[0.0], np.cumsum(steps)])
    total = float(travelled[-1])
    progress = travelled / total if total > 0 else np.zeros_like(travelled)
    nan = np.full(len(time), np.nan)
    table = np.column_stack(
        [
            time - time[0],
            columns["a_forward_mps2"],
            columns["a_right_mps2"],
            columns.get("lon_deg", nan),
            columns.get("lat_deg", nan),
            np.radians(np.asarray(columns["yaw_deg"], dtype=float)),
            columns["speed_mps"],
            progress,
        ]
    )
    trajectory = root / "trajectory.csv"
    _write_csv(
        trajectory,
        ARM_FORCE_COLUMNS,
        ([_cell(value) for value in row] for row in table),
    )
    period = float(np.mean(np.diff(time))) if len(time) > 1 else math.nan
    manifest = {
        "schema": 1,
        "map": str(recording.metadata.get("map", "<unknown>")),
        "vehicle": str(recording.metadata.get("vehicle", "<unknown>")),
        "fps": 1.0 / period if period > 0 else 0.0,
        "trajectory_columns": list(ARM_FORCE_COLUMNS),
        "extra": {
            "source": "myosuite-carla",
            "geolocation": "lon_deg" in columns,
            "progress_frac": "cumulative planar distance over the recorded total",
            "metadata": recording.metadata,
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"arm_force_trajectory": trajectory, "arm_force_manifest": path}


def _row_message(
    labels: Sequence[str], array: NDArray[np.float64]
) -> Callable[[int], dict[str, Any]]:
    def message(index: int) -> dict[str, Any]:
        return {label: float(value) for label, value in zip(labels, array[index])}

    return message


def _write_mcap(recording: Recording, path: Path) -> Path:
    writer_module = importlib.import_module("mcap.writer")
    names = list(recording.columns)
    numeric = [name for name in names if recording.columns[name].dtype != object]
    topics: list[tuple[str, str, dict[str, Any], Callable[[int], dict[str, Any]]]] = [
        (
            "/drive/frame",
            "myosuite_carla.Frame",
            {name: {"type": "number"} for name in numeric},
            lambda index: {
                name: float(recording.columns[name][index]) for name in numeric
            },
        )
    ]
    if recording.joints:
        for topic, schema, labels, array in (
            (
                "/body/coordinates",
                "Coordinates",
                recording.joints,
                recording.coordinates,
            ),
            ("/body/activation", "Activation", recording.muscles, recording.activation),
            ("/body/force", "Force", recording.muscles, recording.force),
        ):
            topics.append(
                (
                    topic,
                    f"myosuite_carla.{schema}",
                    {label: {"type": "number"} for label in labels},
                    _row_message(labels, array),
                )
            )
    with path.open("wb") as handle:
        writer = writer_module.Writer(handle)
        writer.start(profile="", library="myosuite-carla")
        channels = []
        for topic, schema_name, properties, message in topics:
            schema = writer.register_schema(
                name=schema_name,
                encoding="jsonschema",
                data=json.dumps({"type": "object", "properties": properties}).encode(),
            )
            channels.append(
                (
                    writer.register_channel(
                        topic=topic, message_encoding="json", schema_id=schema
                    ),
                    message,
                )
            )
        for index, moment in enumerate(recording.time):
            stamp = int(round(float(moment) * 1e9))
            for channel, message in channels:
                writer.add_message(
                    channel_id=channel,
                    log_time=stamp,
                    publish_time=stamp,
                    data=json.dumps(message(index)).encode(),
                )
        writer.finish()
    return path

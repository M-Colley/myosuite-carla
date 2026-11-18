"""Collect designed drives in live CARLA to identify the vehicle surrogate.

Each segment teleports one vehicle to rest at a spawn point with at least
``--min-straight`` metres of straight road ahead (140 m for fast steering),
records half a second at rest and then one manoeuvre: a throttle launch, coast
and pedal stop; a hand-brake stop; a reverse launch; or a speed-held
sinusoidal steering sweep at 2-10 m/s (``steer``) or 8-14 m/s
(``steer_fast``); or a low-speed sweep (``steer_low``) or step and hold
(``steer_hold``) at 0.8-3 m/s up to full lock; or a step doublet
(``steer_doublet``: a held step, the opposite step and release) at 3-10 m/s.
Levels, speeds, amplitudes and frequencies are drawn from each seed. Segments
with a collision or a roll above 3 degrees are excluded from fitting.
All attempted trajectories are retained under ``seed<N>/attempts``.
Accepted frames are exported with ``DriveRecorder`` to ``--data/seed<N>``
with a ``collection.json`` log; fit with
``examples/fit_vehicle_model.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from myosuite_carla import (
    CarlaSession,
    DriveRecorder,
    Recording,
    SessionConfig,
    VehicleCommand,
)

DT = 0.02


def _straight(
    town: Any, transform: Any, limit: float = 300.0, *, max_road_pitch: float = 1.0
) -> float:
    waypoint = town.get_waypoint(transform.location)
    if waypoint is None or abs(waypoint.transform.rotation.pitch) > max_road_pitch:
        return 0.0
    start = waypoint.transform.rotation.yaw
    distance = 0.0
    while distance < limit:
        following = waypoint.next(2.0)
        if not following:
            break
        turn = (following[0].transform.rotation.yaw - start + 180) % 360 - 180
        if abs(turn) > 3 or abs(following[0].transform.rotation.pitch) > max_road_pitch:
            break
        distance += 2.0
        waypoint = following[0]
    return distance


class Segment:
    """Stateful command program for one manoeuvre; returns None when done."""

    def __init__(
        self,
        kind: str,
        rng: np.random.Generator,
        budget: float,
        *,
        steer_exponent: float = 1.0,
        max_lateral_acceleration: float = 3.0,
        launch_profile: tuple[float, float, float] | None = None,
        coast_seconds: float = 1.5,
    ) -> None:
        if not math.isfinite(steer_exponent) or not 0.5 <= steer_exponent <= 3:
            raise ValueError("steering input exponent must be within [0.5, 3]")
        if not math.isfinite(max_lateral_acceleration) or max_lateral_acceleration <= 0:
            raise ValueError("maximum lateral acceleration must be finite and positive")
        self.kind = kind
        self.budget = budget
        self.stage = "rest"
        self.frames = 0
        self.stage_frames = 0
        self.travelled = 0.0
        self.level = float(rng.uniform(0.2, 1.0))
        self.brake = float(rng.uniform(0.15, 1.0))
        self.target = float(rng.uniform(2.0, 10.0))
        self.amplitude = float(rng.uniform(0.05, 0.6))
        self.frequency = float(rng.uniform(0.15, 0.6))
        self.reverse_level = float(rng.uniform(0.2, 0.6))
        if kind == "steer_fast":
            self.target = float(rng.uniform(8.0, 14.0))
            self.amplitude = float(rng.uniform(0.03, 0.25))
            self.frequency = float(rng.uniform(0.4, 0.9))
        if kind in ("steer", "steer_fast") and steer_exponent != 2:
            lateral = float(
                rng.uniform(
                    min(0.3, max_lateral_acceleration / 2), max_lateral_acceleration
                )
            )
            angle = math.atan(lateral * 3.075 / self.target**2)
            self.amplitude = float(
                np.clip((angle / 0.92) ** (1 / steer_exponent), 0.01, 1.0)
            )
        self.hold = 0.0
        self.targeted_launch = launch_profile is not None
        self.target_reached = False
        self.incomplete_reason: str | None = None
        self.coast_seconds = coast_seconds
        if launch_profile is not None:
            self.target, self.level, self.brake = launch_profile
        if kind in ("steer_low", "steer_hold"):
            self.target = float(rng.uniform(0.8, 3.0))
            if kind == "steer_low":
                self.amplitude = float(rng.uniform(0.4, 1.0))
                self.frequency = float(rng.uniform(0.3, 0.8))
            else:
                side = float(rng.choice((-1.0, 1.0)))
                self.amplitude = side * float(rng.uniform(0.2, 1.0))
            limit = (
                math.atan(max_lateral_acceleration * 3.075 / self.target**2) / 0.92
            ) ** (1 / steer_exponent)
            self.amplitude = math.copysign(
                min(abs(self.amplitude), limit), self.amplitude
            )
            # Bound the heading excursion near 0.6 rad per half sweep or
            # hold, taking the road-wheel angle as at most 0.98 rad times
            # the steer (linear, so conservative).
            rate = self.target * math.tan(0.98 * abs(self.amplitude)) / 3.075
            self.frequency = min(max(self.frequency, rate / (math.pi * 0.6)), 1.2)
            self.hold = float(np.clip(0.6 / rate, 0.6, 2.5))
        if kind == "steer_doublet":
            self.target = float(rng.uniform(3.0, 10.0))
            lateral = float(
                rng.uniform(
                    min(0.5, max_lateral_acceleration / 2), max_lateral_acceleration
                )
            )
            # Invert the selected input curve for the target lateral acceleration.
            angle = math.atan(lateral * 3.075 / self.target**2)
            side = float(rng.choice((-1.0, 1.0)))
            self.amplitude = side * float(
                np.clip((angle / 0.92) ** (1.0 / steer_exponent), 0.05, 0.8)
            )
            self.hold = float(np.clip(0.25 * self.target / lateral, 0.4, 1.5))
        if kind.startswith("steer"):
            limit = (
                math.atan(max_lateral_acceleration * 3.075 / self.target**2) / 0.92
            ) ** (1 / steer_exponent)
            self.amplitude = math.copysign(
                min(abs(self.amplitude), limit), self.amplitude
            )

    def _next(self, stage: str) -> None:
        self.stage, self.stage_frames = stage, 0

    def command(self, speed: float, step: float) -> VehicleCommand | None:
        self.frames += 1
        self.stage_frames += 1
        self.travelled += abs(speed) * step
        time = self.stage_frames * step
        stopping = speed * speed / 8.0 + 2.0 * abs(speed) + 12.0
        if self.targeted_launch:
            stopping = speed * speed / 4.0 + 2.0 * abs(speed) + 25.0
        room = self.budget - self.travelled - stopping
        if self.stage == "rest":
            if time >= 0.5:
                self._next("drive")
            return VehicleCommand()
        if self.stage == "stop":
            if abs(speed) < 0.02 and time > 0.5:
                return None
            hand = self.kind == "hand_brake"
            return VehicleCommand(
                brake=0.0 if hand else self.brake,
                hand_brake=hand,
                reverse=self.kind == "reverse",
            )
        if self.kind in ("launch", "hand_brake"):
            if self.stage == "drive":
                if self.targeted_launch:
                    if speed >= self.target:
                        self.target_reached = True
                        self._next("coast")
                    elif room <= 0 or time >= 20.0:
                        self.incomplete_reason = (
                            "distance_limit" if room <= 0 else "time_limit"
                        )
                        self._next("stop")
                    return VehicleCommand(throttle=self.level)
                level = self.level if self.kind == "launch" else 0.6
                limit = 8.0 if self.kind == "launch" else 4.0
                if (
                    room <= 0
                    or time >= limit
                    or (self.kind == "hand_brake" and speed >= self.target)
                ):
                    self._next("coast" if self.kind == "launch" else "stop")
                return VehicleCommand(throttle=level)
            if self.stage == "coast":
                if time >= self.coast_seconds or room <= 0:
                    if self.targeted_launch and time < self.coast_seconds:
                        self.incomplete_reason = "coast_distance_limit"
                    self._next("stop")
                return VehicleCommand()
        if self.kind == "reverse":
            if self.stage == "drive":
                if time >= 2.5:
                    self._next("coast")
                return VehicleCommand(throttle=self.reverse_level, reverse=True)
            if self.stage == "coast":
                if time >= 1.0:
                    self._next("stop")
                return VehicleCommand(reverse=True)
        if self.kind in (
            "steer",
            "steer_fast",
            "steer_low",
            "steer_hold",
            "steer_doublet",
        ):
            if self.kind in ("steer_low", "steer_hold"):
                throttle = float(np.clip(0.2 + 0.4 * (self.target - speed), 0.0, 0.6))
            else:
                throttle = float(np.clip(0.3 + 0.25 * (self.target - speed), 0.0, 1.0))
            if self.stage == "drive":
                if speed >= self.target or time >= 8.0 or room <= 30:
                    held = self.kind in ("steer_hold", "steer_doublet")
                    self._next("hold" if held else "sweep")
                return VehicleCommand(throttle=throttle)
            if self.stage == "hold":
                if time >= self.hold or room <= 0:
                    doublet = self.kind == "steer_doublet"
                    self._next("counter" if doublet else "release")
                return VehicleCommand(throttle=throttle, steer=self.amplitude)
            if self.stage == "counter":
                if time >= self.hold or room <= 0:
                    self._next("release")
                return VehicleCommand(throttle=throttle, steer=-self.amplitude)
            if self.stage == "release":
                if time >= 1.5 or room <= 0:
                    self._next("stop")
                return VehicleCommand(throttle=throttle)
            if self.stage == "sweep":
                if time >= 1.0 / self.frequency or room <= 0:
                    self._next("stop")
                steer = self.amplitude * math.sin(2 * math.pi * self.frequency * time)
                return VehicleCommand(throttle=throttle, steer=steer)
        raise RuntimeError(f"Unknown segment state {self.kind}/{self.stage}")


def _run_seed(
    session: CarlaSession,
    api: Any,
    vehicle: Any,
    spawns: list[tuple[Any, float, float]],
    seed: int,
    counts: dict[str, int],
    collisions: list[int],
    *,
    steer_exponent: float,
    max_lateral_acceleration: float,
    directory: Path,
    metadata: dict[str, Any],
    launch_profiles: list[tuple[float, float, float]] | None = None,
    coast_seconds: float = 1.5,
) -> tuple[list[Recording], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    profiles = list(launch_profiles) if launch_profiles is not None else None
    if profiles is not None:
        rng.shuffle(profiles)
    launch_index = 0
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    rng.shuffle(kinds)
    kept: list[Recording] = []
    log: list[dict[str, Any]] = []
    long = [item for item in spawns if item[1] >= 140.0] or spawns
    for index, kind in enumerate(kinds):
        candidates = long if kind in ("steer_fast", "steer_doublet") else spawns
        spawn, budget, road_pitch = candidates[int(rng.integers(len(candidates)))]
        program = Segment(
            kind,
            rng,
            budget,
            steer_exponent=steer_exponent,
            max_lateral_acceleration=max_lateral_acceleration,
            launch_profile=(
                profiles[launch_index]
                if profiles is not None and kind == "launch"
                else None
            ),
            coast_seconds=coast_seconds,
        )
        if kind == "launch":
            launch_index += 1
        recorder = DriveRecorder(body=False)
        label = f"s{seed}-{index:02d}-{kind}"
        max_roll = 0.0
        command = VehicleCommand()
        completed = False
        entry: dict[str, Any] = {
            "segment": label,
            "kind": kind,
            "straight_m": budget,
            "road_pitch_deg": road_pitch,
            "kept": False,
            "status": "failed",
            "level": program.level,
            "brake": program.brake,
            "target_speed": program.target,
            "steer_amplitude": program.amplitude,
            "steer_frequency": program.frequency,
            "steer_hold_s": program.hold,
            "steer_input_exponent": steer_exponent,
            "targeted_launch": program.targeted_launch,
        }
        log.append(entry)
        try:
            collisions.clear()
            vehicle.set_transform(spawn)
            vehicle.set_target_velocity(api.Vector3D(0.0, 0.0, 0.0))
            vehicle.set_target_angular_velocity(api.Vector3D(0.0, 0.0, 0.0))
            for _ in range(75):
                session.step(VehicleCommand(brake=1.0, hand_brake=True))
            entry["settling_collision_frames"] = list(collisions)
            collisions.clear()
            telemetry = session.snapshot()
            while True:
                velocity = np.asarray(telemetry.velocity)
                yaw = math.radians(telemetry.rotation[1])
                speed = float(velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw))
                next_command = program.command(speed, DT)
                recorder.add(
                    telemetry.elapsed_seconds,
                    telemetry,
                    command=command.as_dict(),
                    phase=label,
                )
                max_roll = max(max_roll, abs(telemetry.rotation[2]))
                if collisions or max_roll > 3 or program.frames > 1500:
                    break
                if next_command is None:
                    completed = True
                    break
                command = next_command
                telemetry = session.step(command)
            for _ in range(3):
                session.step(VehicleCommand(brake=1.0))
            valid = (
                completed
                and not collisions
                and max_roll <= 3
                and program.incomplete_reason is None
            )
            entry.update(kept=valid, status="completed" if valid else "rejected")
            if valid:
                kept.append(recorder.recording())
        except Exception as error:
            entry["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            entry.update(
                frames=len(recorder),
                collision=bool(collisions),
                collision_frames=list(collisions),
                max_roll_deg=max_roll,
                program_stage=program.stage,
                completed=completed,
                frame_limit_exceeded=program.frames > 1500,
                target_reached=program.target_reached,
                incomplete_reason=program.incomplete_reason,
            )
            try:
                if len(recorder):
                    attempt = recorder.recording()
                    attempt.metadata.update(metadata)
                    attempt.metadata.update(entry, seed=seed)
                    entry["recording"] = f"attempts/{label}"
                    attempt.save(directory / "attempts" / label, formats=("csv",))
            except Exception as error:
                entry.update(
                    kept=False,
                    status="failed",
                    error={"type": type(error).__name__, "message": str(error)},
                )
                raise
            finally:
                (directory / "collection.json").write_text(
                    json.dumps(
                        {
                            **metadata,
                            "status": "failed"
                            if entry["status"] == "failed"
                            else "running",
                            "segments": log,
                        },
                        indent=1,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        print(f"{label}: {len(recorder)} frames, kept={entry['kept']}", flush=True)
    return kept, log


def main() -> int:
    """Collect designed drives, fit the surrogate and report held-out errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("CARLA_HOST", "localhost"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("CARLA_PORT", 2000))
    )
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--blueprint", default="vehicle.nissan.patrol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--min-straight", type=float, default=100.0)
    parser.add_argument("--straight-limit", type=float, default=300.0)
    parser.add_argument("--spawn-indices", type=int, nargs="+")
    parser.add_argument("--max-road-pitch", type=float, default=1.0)
    parser.add_argument("--max-lateral-acceleration", type=float, default=3.0)
    parser.add_argument(
        "--steer-exponent",
        type=float,
        default=None,
        help="input exponent; inferred as 1 for CARLA 1.0 or 2 for 0.10 if omitted",
    )
    parser.add_argument("--launch", type=int, default=10)
    parser.add_argument("--launch-targets", type=float, nargs="+")
    parser.add_argument(
        "--launch-throttles", type=float, nargs="+", default=[0.75, 1.0]
    )
    parser.add_argument("--launch-brakes", type=float, nargs="+", default=[0.4, 0.8])
    parser.add_argument("--launch-coast-seconds", type=float, default=1.5)
    parser.add_argument("--hand-brake", type=int, default=2)
    parser.add_argument("--reverse", type=int, default=3)
    parser.add_argument("--steer", type=int, default=12)
    parser.add_argument("--steer-fast", type=int, default=0)
    parser.add_argument("--steer-low", type=int, default=0)
    parser.add_argument("--steer-hold", type=int, default=0)
    parser.add_argument("--steer-doublet", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("artifacts/vehicle-sysid"))
    args = parser.parse_args()
    counts = {
        "launch": args.launch,
        "hand_brake": args.hand_brake,
        "reverse": args.reverse,
        "steer": args.steer,
        "steer_fast": args.steer_fast,
        "steer_low": args.steer_low,
        "steer_hold": args.steer_hold,
        "steer_doublet": args.steer_doublet,
    }
    if any(count < 0 for count in counts.values()) or not sum(counts.values()):
        parser.error("manoeuvre counts must be nonnegative with a positive total")
    if any(seed < 0 for seed in args.seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be unique and nonnegative")
    if args.spawn_indices is not None and (
        any(index < 0 for index in args.spawn_indices)
        or len(set(args.spawn_indices)) != len(args.spawn_indices)
    ):
        parser.error("spawn indices must be unique and nonnegative")
    launch_profiles = None
    if args.launch_targets is not None:
        for values, upper in (
            (args.launch_targets, 30),
            (args.launch_throttles, 1),
            (args.launch_brakes, 1),
        ):
            if len(set(values)) != len(values) or any(
                not math.isfinite(value) or not 0 < value <= upper for value in values
            ):
                parser.error("launch grid values must be distinct and within bounds")
        launch_profiles = list(
            product(args.launch_targets, args.launch_throttles, args.launch_brakes)
        )
        if counts["launch"] != len(launch_profiles):
            parser.error("launch count must equal the complete launch profile grid")
    if (
        not math.isfinite(args.min_straight)
        or args.min_straight <= 0
        or not math.isfinite(args.straight_limit)
        or args.straight_limit < args.min_straight
        or not math.isfinite(args.launch_coast_seconds)
        or args.launch_coast_seconds < 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.max_road_pitch)
        or not 0 <= args.max_road_pitch < 90
        or not math.isfinite(args.max_lateral_acceleration)
        or args.max_lateral_acceleration <= 0
        or (
            args.steer_exponent is not None
            and (
                not math.isfinite(args.steer_exponent)
                or not 0.5 <= args.steer_exponent <= 3
            )
        )
    ):
        parser.error("invalid timing, road pitch, acceleration or exponent setting")
    if args.data.exists() and any(args.data.iterdir()):
        parser.error(
            "the output directory must be empty; existing evidence is retained"
        )
    args.data.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        (args.data / f"seed{seed}").mkdir()
    common: dict[str, Any] = {
        "blueprint": args.blueprint,
        "min_straight_m": args.min_straight,
        "straight_limit_m": args.straight_limit,
        "spawn_indices": args.spawn_indices,
        "max_road_pitch_deg": args.max_road_pitch,
        "max_lateral_acceleration_mps2": args.max_lateral_acceleration,
        "counts": counts,
        "seeds": args.seeds,
        "control_dt": DT,
        "launch_profile_grid": (
            {
                "target_speeds_mps": args.launch_targets,
                "throttles": args.launch_throttles,
                "brakes": args.launch_brakes,
                "coast_seconds": args.launch_coast_seconds,
            }
            if launch_profiles is not None
            else None
        ),
        "command": "examples/identify_vehicle.py " + " ".join(sys.argv[1:]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }
    try:
        import carla

        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)
        common.update(
            server_version=client.get_server_version(),
            client_version=client.get_client_version(),
            carla_module=str(getattr(carla, "__file__", None)),
        )
        if common["server_version"] != common["client_version"]:
            raise RuntimeError("CARLA client and server builds differ")
        exponent = args.steer_exponent
        if exponent is None:
            release = common["server_version"].split("-", 1)[0]
            exponent = {"1.0": 1.0, "1.0.0": 1.0, "0.10.0": 2.0}.get(release)
            if exponent is None:
                raise ValueError(
                    "this CARLA release requires an explicit steer exponent"
                )
        common["steer_input_exponent"] = exponent
        world = client.get_world()
        if world.get_settings().synchronous_mode:
            raise RuntimeError("Stop the existing synchronous client first")
        town = world.get_map()
        spawn_points = town.get_spawn_points()
        if args.spawn_indices is not None:
            if any(index >= len(spawn_points) for index in args.spawn_indices):
                raise ValueError("A requested spawn index is outside this map")
            spawn_points = [spawn_points[index] for index in args.spawn_indices]
        spawns = [
            (
                spawn,
                length,
                float(town.get_waypoint(spawn.location).transform.rotation.pitch),
            )
            for spawn in spawn_points
            if (
                length := _straight(
                    town,
                    spawn,
                    limit=args.straight_limit,
                    max_road_pitch=args.max_road_pitch,
                )
            )
            >= args.min_straight
        ]
        common.update(map=town.name, spawn_points_used=len(spawns))
        if not spawns:
            raise RuntimeError(
                "No spawn point satisfies the straight road and pitch limits"
            )
        with ExitStack() as cleanup:
            blueprint = world.get_blueprint_library().find(args.blueprint)
            vehicle = world.try_spawn_actor(blueprint, spawns[0][0])
            if vehicle is None:
                raise RuntimeError("Could not spawn the vehicle")
            cleanup.callback(vehicle.destroy)
            collisions: list[int] = []
            sensor = world.spawn_actor(
                world.get_blueprint_library().find("sensor.other.collision"),
                carla.Transform(),
                attach_to=vehicle,
            )
            cleanup.callback(sensor.destroy)
            sensor.listen(lambda event: collisions.append(int(event.frame)))
            cleanup.callback(sensor.stop)
            session = CarlaSession(
                client,
                vehicle,
                SessionConfig(fixed_delta_seconds=DT, timeout_seconds=args.timeout),
                api=carla,
                own_vehicle=False,
            )
            cleanup.callback(session.close)
            session.start()
            for seed in args.seeds:
                directory = args.data / f"seed{seed}"
                metadata = {
                    **common,
                    "seed": seed,
                    "status": "running",
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                }
                kept, log = _run_seed(
                    session,
                    carla,
                    vehicle,
                    spawns,
                    seed,
                    counts,
                    collisions,
                    steer_exponent=exponent,
                    max_lateral_acceleration=args.max_lateral_acceleration,
                    directory=directory,
                    metadata=metadata,
                    launch_profiles=launch_profiles,
                    coast_seconds=args.launch_coast_seconds,
                )
                if kept:
                    recording = Recording.concatenate(kept)
                    recording.metadata.update(
                        common, vehicle=args.blueprint, seed=seed, status="completed"
                    )
                    recording.save(directory, formats=("csv",))
                (directory / "collection.json").write_text(
                    json.dumps(
                        {
                            **metadata,
                            "status": "completed",
                            "finished_utc": datetime.now(timezone.utc).isoformat(),
                            "segments": log,
                        },
                        indent=1,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        common["status"] = "completed"
    except Exception as error:
        common["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        common["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (args.data / "run.json").write_text(
            json.dumps(common, indent=1) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

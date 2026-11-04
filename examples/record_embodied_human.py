"""Record a local embodied-human driver in a rendering CARLA 0.10 server.

The chase camera and MuJoCo body share the same synchronous frame. The inset
shows the human model's gaze and SuRT state. Road perception uses the human
project's parametric geometry interface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import queue
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any


def _source(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "path": str(root),
        "revision": git("rev-parse", "HEAD"),
        "dirty": git("status", "--porcelain"),
        "source_sha256": {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((root / "src").rglob("*.py"))
        },
    }


def _json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(item) for item in value]
    if hasattr(value, "tolist"):
        return _json(value.tolist())
    if hasattr(value, "item"):
        return _json(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


@lru_cache
def _font(size: int) -> Any:
    from PIL import ImageFont

    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class Video:
    """Compose and encode synchronized vehicle, body and human state images."""

    def __init__(self, output: Path, driver: Any, title: str) -> None:
        import imageio.v2 as imageio
        import mujoco
        import numpy as np
        from PIL import Image, ImageDraw

        self.title = title
        self.renderer = mujoco.Renderer(driver.model, height=720, width=640)
        self.view = mujoco.MjvCamera()
        self.view.lookat[:] = (-0.03, -0.12, 0.95)
        self.view.distance, self.view.azimuth, self.view.elevation = 2.5, 155, -10
        driver.model.vis.headlight.ambient[:] = 0.4
        driver.model.vis.headlight.diffuse[:] = 0.6
        try:
            self.writer = imageio.get_writer(
                output,
                fps=25,
                codec="libx264",
                bitrate="8M",
                pixelformat="yuv420p",
                macro_block_size=2,
                ffmpeg_params=[
                    "-maxrate",
                    "8M",
                    "-bufsize",
                    "16M",
                    "-movflags",
                    "+faststart",
                ],
            )
        except Exception:
            self.renderer.close()
            raise
        card = Image.new("RGB", (1920, 1080), (15, 20, 28))
        draw = ImageDraw.Draw(card)
        for y, text, size in (
            (390, "Embodied human attentive driving", 48),
            (465, title, 36),
            (530, "Live CARLA 0.10.0 and MyoSuite muscle actuation", 30),
            (590, "Local embodied-human project; gaze model and SuRT", 26),
        ):
            draw.text((80, y), text, font=_font(size), fill="white")
        try:
            for _ in range(25):
                self.writer.append_data(np.asarray(card))
        except Exception:
            self.close()
            raise
        self.encoded = 0

    def frame(self, image: Any, env: Any, agent: Any, info: dict[str, Any]) -> None:
        import numpy as np
        from PIL import Image, ImageDraw

        pixels = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(720, 1280, 4)
        canvas = Image.new("RGB", (1920, 1080), (15, 20, 28))
        canvas.paste(Image.fromarray(pixels[:, :, 2::-1].copy()), (0, 0))
        self.renderer.update_scene(env.driver.data, camera=self.view)
        canvas.paste(Image.fromarray(self.renderer.render()), (1280, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((12, 12, 1010, 94), fill=(15, 20, 28))
        draw.text((28, 20), self.title, font=_font(29), fill="white")
        draw.text(
            (28, 59),
            "Controller: embodied-human AttentiveDriver",
            font=_font(22),
            fill="white",
        )
        draw.text((1300, 20), "MyoSuite full body", font=_font(27), fill="white")
        draw.text(
            (1300, 56),
            f"{env.driver.model.nu} muscles; vehicle loads",
            font=_font(21),
            fill="white",
        )
        state, inputs = info["scenario"], info["inputs"]
        draw.text(
            (28, 742),
            f"Time {state['time']:.2f} s     Speed {state['speed_mps'] * 3.6:.1f} km/h",
            font=_font(29),
            fill="white",
        )
        for y, name in ((810, "throttle"), (860, "brake"), (910, "steer")):
            value = float(inputs[name])
            draw.text((28, y), name, font=_font(24), fill="white")
            draw.rectangle((170, y + 4, 490, y + 30), outline=(190, 200, 210))
            start = 330 if name == "steer" else 170
            end = start + int(value * (160 if name == "steer" else 320))
            draw.rectangle(
                (min(start, end), y + 6, max(start, end), y + 28), fill=(80, 200, 165)
            )
            draw.text((510, y), f"{value:+.2f}", font=_font(24), fill="white")
        draw.text(
            (28, 978),
            f"Lane error {state['lane_offset_m']:+.2f} m",
            font=_font(25),
            fill="white",
        )
        draw.text(
            (28, 1020),
            "Measured controls from limb contact",
            font=_font(23),
            fill=(180, 200, 220),
        )
        area, gaze = agent.log.areas[-1], agent.log.gaze[-1]
        for y, text in (
            (742, "Human model state"),
            (792, f"Gaze: {area}"),
            (838, f"Gaze angles {gaze[0]:+.1f}, {gaze[1]:+.1f} degrees"),
            (884, f"Lateral belief SD {agent.log.belief_sd[-1]:.3f} m"),
            (930, "Perception: parametric road geometry"),
            (978, "Head pose remains posture constrained"),
        ):
            draw.text((650, y), text, font=_font(24), fill="white")
        if agent.task is not None:
            task_image = Image.fromarray(agent.task.image).resize((560, 280))
            if area == "display":
                p = agent.params
                gx = (0.5 + (gaze[0] - p.display_position[0]) / p.display_size[0]) * 560
                gy = (0.5 - (gaze[1] - p.display_position[1]) / p.display_size[1]) * 280
                ImageDraw.Draw(task_image).ellipse(
                    (gx - 10, gy - 10, gx + 10, gy + 10), outline=(255, 80, 80), width=3
                )
            canvas.paste(task_image, (1320, 778))
            draw.text(
                (1320, 738), "SuRT display and model gaze", font=_font(24), fill="white"
            )
        self.writer.append_data(np.asarray(canvas))
        self.encoded += 1

    def close(self) -> None:
        try:
            self.writer.close()
        finally:
            self.renderer.close()


def main() -> int:
    """Record one episode driven by the specified local human project."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--human-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "embodied-human",
    )
    parser.add_argument(
        "--scenario",
        choices=("lane_change", "lead_brake", "cruise"),
        default="lane_change",
    )
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--task", choices=("surt", "none"), default="surt")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/embodied-human-live/drive.mp4")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spawn", type=int, default=105)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0.02:
        parser.error("seconds must be finite and at least 0.02")
    human_repo = args.human_repo.resolve()
    if not (human_repo / "src/embodied_human/__init__.py").is_file():
        parser.error("human-repo must contain src/embodied_human")
    sys.path.insert(0, str(human_repo / "src"))
    import carla
    import embodied_human as eh
    import numpy as np
    from embodied_human.agents.driver import AttentiveDriver
    from embodied_human.tasks.surt import SuRTTask

    from myosuite_carla import (
        CarlaSession,
        DriveRecorder,
        FullBodyDriverConfig,
        SessionConfig,
        build_fullbody_driver,
    )
    from myosuite_carla.carla_scenario import (
        CarlaLeadVehicle,
        lane_from_carla,
        teleport,
    )
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.scenario import (
        LaneChange,
        LeadVehicle,
        Scenario,
        ScenarioMonitor,
    )

    if not Path(eh.__file__).resolve().is_relative_to(human_repo):
        raise RuntimeError("The imported human module is outside human-repo")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "status": "failed",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "scenario": args.scenario,
        "human_import": eh.__file__,
        "human_source": _source(human_repo),
        "bridge_source": _source(Path(__file__).resolve().parents[1]),
        "recording_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "carla",
                "mujoco",
                "myosuite",
                "numpy",
                "embodied-human",
                "myosuite-carla",
            )
        },
        "frames": [],
        "cleanup": [],
        "collision_contacts": [],
        "camera_frames_matched": 0,
    }
    setting_names = (
        "synchronous_mode",
        "fixed_delta_seconds",
        "substepping",
        "max_substep_delta_time",
        "max_substeps",
        "no_rendering_mode",
    )
    world = None
    before_actors: set[int] = set()
    before_settings: dict[str, Any] = {}

    def cleanup(label: str, function: Any, *values: Any) -> None:
        try:
            function(*values)
            report["cleanup"].append({"resource": label, "success": True})
        except Exception as error:
            report["cleanup"].append(
                {"resource": label, "success": False, "error": str(error)}
            )

    started = time.perf_counter()
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60)
        world = client.get_world()
        before_actors = {actor.id for actor in world.get_actors()}
        before_settings = {
            name: getattr(world.get_settings(), name) for name in setting_names
        }
        config = FullBodyDriverConfig.fast() if args.fast else FullBodyDriverConfig()
        driver = build_fullbody_driver(config)
        report["driver_config"] = asdict(config)
        report["muscles"] = driver.model.nu
        report["myosuite_import"] = getattr(
            sys.modules.get("myosuite"), "__file__", None
        )
        with ExitStack() as stack:
            spawn = world.get_map().get_spawn_points()[args.spawn]
            lane = lane_from_carla(
                world.get_map(), spawn, length=max(180.0, args.seconds * 12.0)
            )
            scenario = Scenario(
                name=args.scenario,
                lane=lane,
                duration=args.seconds,
                start_speed=8.0,
                target_speed=8.0,
                lane_changes=(LaneChange(time=3.0, offset=-3.5, duration=4.0),)
                if args.scenario == "lane_change"
                else (),
                lead=LeadVehicle(gap=20, speed=8, brake_time=4, deceleration=2.5)
                if args.scenario == "lead_brake"
                else None,
            )
            vehicle = world.spawn_actor(
                world.get_blueprint_library().find("vehicle.nissan.patrol"), spawn
            )
            stack.callback(cleanup, "ego", vehicle.destroy)
            vehicle.set_autopilot(False)
            session = CarlaSession(
                client,
                vehicle,
                SessionConfig(
                    fixed_delta_seconds=config.control_dt, timeout_seconds=60
                ),
                api=carla,
            )
            stack.callback(cleanup, "session", session.close)
            session.start()
            monitor = ScenarioMonitor(scenario)
            lead = CarlaLeadVehicle(monitor, world, api=carla)
            stack.callback(cleanup, "lead", lead.destroy)
            recorder = DriveRecorder() if args.record else None
            env = FullBodyDrivingEnv(
                scenario,
                driver=driver,
                vehicle=session,
                reset_vehicle=teleport,
                hooks=(lead,),
                recorder=recorder,
                monitor=monitor,
            )
            env.reset(seed=args.seed)
            lead.spawn(session.snapshot().location[2])
            task = (
                SuRTTask(rng=np.random.default_rng(args.seed))
                if args.task == "surt"
                else None
            )
            human = eh.build_human(seed=args.seed)
            agent = AttentiveDriver(human, env, task)
            agent.reset(seed=args.seed)
            report.update(
                server_version=client.get_server_version(),
                client_version=client.get_client_version(),
                map=world.get_map().name,
                human_parameters=human.provenance(),
                scenario_config=asdict(scenario),
            )
            rgb = world.get_blueprint_library().find("sensor.camera.rgb")
            for name, value in {
                "image_size_x": "1280",
                "image_size_y": "720",
                "fov": "90",
                # Capture each tick without a second floating point time threshold.
                "sensor_tick": "0",
            }.items():
                rgb.set_attribute(name, value)
            camera = world.spawn_actor(
                rgb,
                carla.Transform(
                    carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12)
                ),
                attach_to=vehicle,
            )
            stack.callback(cleanup, "camera", camera.destroy)
            stack.callback(cleanup, "camera_listener", camera.stop)
            images: queue.Queue[Any] = queue.Queue()
            camera.listen(images.put)
            collision = world.spawn_actor(
                world.get_blueprint_library().find("sensor.other.collision"),
                carla.Transform(),
                attach_to=vehicle,
            )
            stack.callback(cleanup, "collision_sensor", collision.destroy)
            stack.callback(cleanup, "collision_listener", collision.stop)
            collision.listen(
                lambda event: report["collision_contacts"].append(
                    {"frame": event.frame, "other": event.other_actor.type_id}
                )
            )
            title = (
                "Lane change"
                if args.scenario == "lane_change"
                else "Lead vehicle braking"
                if args.scenario == "lead_brake"
                else "Lane keeping"
            )
            if args.task == "surt":
                title += " with SuRT"
            video = Video(args.output, driver, title)
            stack.callback(cleanup, "video", video.close)
            for step in range(math.ceil(args.seconds / env.dt)):
                intent = agent.act()
                _, _, terminated, truncated, info = env.step(intent)
                telemetry = session.snapshot()
                if not all(
                    np.isfinite(values).all()
                    for values in (
                        driver.data.qpos,
                        driver.data.qvel,
                        driver.data.act,
                        telemetry.location,
                        telemetry.rotation,
                        telemetry.velocity,
                        telemetry.acceleration,
                        telemetry.angular_velocity,
                    )
                ):
                    raise RuntimeError("The simulation contains a nonfinite state")
                applied = vehicle.get_control()
                applied_controls = {
                    name: float(getattr(applied, name))
                    for name in ("throttle", "brake", "steer")
                }
                if any(
                    not math.isclose(value, info["command"][name], abs_tol=1e-5)
                    for name, value in applied_controls.items()
                ):
                    raise RuntimeError("Applied CARLA controls differ from the command")
                try:
                    image = images.get(timeout=60)
                    while image.frame < telemetry.frame:
                        image = images.get(timeout=60)
                except queue.Empty as error:
                    raise TimeoutError(
                        f"No camera image for CARLA frame {telemetry.frame}"
                    ) from error
                if image.frame != telemetry.frame:
                    raise RuntimeError("CARLA camera and vehicle frames differ")
                report["camera_frames_matched"] += 1
                report["frames"].append(
                    {
                        "camera_frame": image.frame,
                        "vehicle": telemetry.as_dict(),
                        "intent": asdict(intent),
                        "inputs": info["inputs"],
                        "command": info["command"],
                        "applied_controls": applied_controls,
                        "scenario": info["scenario"],
                        "gaze_area": agent.log.areas[-1],
                        "gaze_degrees": agent.log.gaze[-1],
                        "belief_sd_m": agent.log.belief_sd[-1],
                        "head_acceleration_mps2": agent.log.head_acceleration[-1],
                        "head_rate_rad_s": agent.log.head_rate[-1],
                        "fatigue": agent.log.fatigue[-1],
                        "mechanics": driver.measurement(),
                    }
                )
                if step % 2 == 1 or terminated or truncated:
                    video.frame(image, env, agent, info)
                if terminated or truncated:
                    break
                if step % 100 == 0:
                    print(f"{args.scenario}: {step + 1} frames", flush=True)
            time.sleep(0.2)
            report.update(
                encoded_frames=video.encoded,
                seconds=info["scenario"]["time"],
                terminated=terminated,
                truncated=truncated,
                events=monitor.events,
                glance_metrics=agent.log.glance_metrics(),
                task_summary=task.result.summary() if task is not None else None,
                vehicle_footprints=monitor.footprints,
            )
            report["status"] = (
                "completed"
                if truncated and not terminated and not report["collision_contacts"]
                else "failed"
            )
            if recorder is not None and args.record:
                recorder.recording().save(args.record, formats=("csv",))
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["video_bytes"] = (
            args.output.stat().st_size if args.output.exists() else 0
        )
        try:
            report["world_settings_restored"] = world is not None and all(
                getattr(world.get_settings(), name) == value
                for name, value in before_settings.items()
            )
            report["actors_restored"] = (
                world is not None
                and {actor.id for actor in world.get_actors()} == before_actors
            )
        except Exception as error:
            report["world_settings_restored"] = False
            report["actors_restored"] = False
            report["cleanup"].append(
                {"resource": "verification", "success": False, "error": str(error)}
            )
        if (
            not report["world_settings_restored"]
            or not report["actors_restored"]
            or any(not item["success"] for item in report["cleanup"])
        ):
            report["status"] = "failed"
        args.output.with_suffix(".json").write_text(
            json.dumps(_json(report), indent=2, allow_nan=False), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                key: report.get(key)
                for key in (
                    "status",
                    "seconds",
                    "camera_frames_matched",
                    "encoded_frames",
                    "error",
                    "actors_restored",
                    "world_settings_restored",
                )
            }
        ),
        flush=True,
    )
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

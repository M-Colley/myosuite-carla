"""Render verified offline replays of the paired empirical following trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_steps(seconds: float, dt: float, fps: int) -> range:
    """Capture on the physics grid while omitting the terminal endpoint."""
    if not math.isfinite(seconds) or seconds <= 0 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("duration and physics timestep must be finite and positive")
    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be a positive integer")
    stride = 1 / (dt * fps)
    if not math.isclose(stride, round(stride), abs_tol=1e-9) or stride < 1:
        raise ValueError("frame rate must divide the physics step rate")
    return range(0, math.ceil(seconds / dt - 1e-8), round(stride))


@lru_cache
def font(size: int) -> Any:
    from PIL import ImageFont

    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


class BodyView:
    def __init__(self, driver: Any) -> None:
        import mujoco

        self.renderer = mujoco.Renderer(driver.model, height=720, width=640)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = (-0.03, -0.12, 0.95)
        self.camera.distance = 2.5
        self.camera.azimuth = 155
        self.camera.elevation = -10
        driver.model.vis.headlight.ambient[:] = 0.4
        driver.model.vis.headlight.diffuse[:] = 0.6

    def image(self, driver: Any) -> Any:
        from PIL import Image

        self.renderer.update_scene(driver.data, camera=self.camera)
        return Image.fromarray(self.renderer.render())

    def close(self) -> None:
        self.renderer.close()


class Panel:
    """One profile with measured controls, actual body, and trajectory schematic."""

    def __init__(
        self, path: Path, profile: str, group: str, trial: Any, fps: int, body: BodyView
    ) -> None:
        import imageio.v2 as imageio

        self.profile, self.group, self.trial, self.body = profile, group, trial, body
        self.colour = (55, 145, 210) if profile == "baseline" else (235, 135, 62)
        self.frames = 0
        self.writer = imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            macro_block_size=2,
            ffmpeg_params=[
                "-preset",
                "veryfast",
                "-crf",
                "21",
                "-movflags",
                "+faststart",
            ],
        )

    def frame(self, env: Any, state: dict[str, Any], inputs: dict[str, Any]) -> None:
        import numpy as np
        from PIL import Image, ImageDraw

        canvas = Image.new("RGB", (960, 1080), (15, 21, 30))
        draw = ImageDraw.Draw(canvas)
        title = "Baseline" if self.profile == "baseline" else "Fitted candidate"
        draw.text((22, 12), title, font=font(31), fill=self.colour)
        draw.text(
            (22, 57),
            f"Offline MyoSuite replay   {self.group}",
            font=font(23),
            fill="white",
        )
        t = float(state["time"])
        observed_gap = float(np.interp(t, self.trial.time, self.trial.gap))
        observed_speed = float(np.interp(t, self.trial.time, self.trial.speed))
        simulated_gap = float(state["gap_m"])
        scale = min(11.0, 650 / max(30.0, observed_gap, simulated_gap))
        leader_rear = 800
        length = max(22.0, 4.8 * scale)
        for y, label, gap, colour, outline in (
            (150, "Recorded trajectories", observed_gap, (230, 234, 240), True),
            (260, "Model trajectory", simulated_gap, self.colour, False),
        ):
            draw.text((22, y - 43), label, font=font(20), fill=colour)
            draw.rectangle((22, y - 5, 938, y + 44), fill=(40, 48, 60))
            displacement = float(state["lead_front_s_m"]) * scale
            for x in range(-80, 1000, 90):
                start = x - displacement % 90
                draw.line(
                    (start, y + 20, start + 45, y + 20), fill=(90, 100, 114), width=2
                )
            draw.rounded_rectangle(
                (leader_rear, y, leader_rear + length, y + 37),
                radius=5,
                fill=(160, 169, 181),
            )
            front = leader_rear - gap * scale
            rectangle = (front - length, y, front, y + 37)
            draw.rounded_rectangle(
                rectangle,
                radius=5,
                fill=None if outline else colour,
                outline=colour,
                width=3,
            )
            draw.text((820, y - 30), "Leader", font=font(18), fill=(190, 199, 209))
            draw.text(
                (22, y + 48), f"Bumper gap {gap:.2f} m", font=font(20), fill=colour
            )
        canvas.paste(self.body.image(env.driver), (0, 360))
        draw.text((658, 370), "MyoSuite body", font=font(25), fill="white")
        draw.text(
            (658, 410), f"{env.driver.model.nu} muscles", font=font(22), fill="white"
        )
        for y, text in (
            (465, f"Time {t:5.2f} s"),
            (505, f"Model {state['speed_mps']:.2f} m/s"),
            (545, f"Recorded {observed_speed:.2f} m/s"),
        ):
            draw.text((658, y), text, font=font(22), fill="white")
        draw.text((658, 605), "Measured controls", font=font(23), fill="white")
        for y, name in ((650, "throttle"), (740, "brake"), (830, "steer")):
            value = float(inputs.get(name, 0.0))
            draw.text((658, y), f"{name}  {value:+.2f}", font=font(23), fill="white")
            draw.rectangle((660, y + 37, 930, y + 62), outline=(160, 170, 185))
            start = 795 if name == "steer" else 660
            end = start + value * (135 if name == "steer" else 270)
            draw.rectangle(
                (min(start, end), y + 39, max(start, end), y + 60), fill=self.colour
            )
        draw.text((658, 930), "Recorded leader", font=font(20), fill=(190, 199, 209))
        draw.text((658, 963), "replayed at 50 Hz", font=font(20), fill=(190, 199, 209))
        draw.text((658, 1010), "Video at real time", font=font(20), fill="white")
        self.writer.append_data(np.asarray(canvas))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()


def verify_replay(
    states: list[dict[str, Any]], trial: Any, expected: dict[str, Any]
) -> dict[str, float]:
    import numpy as np

    from myosuite_carla.ngsim import RATE, resample_following

    times = np.asarray([row["time"] for row in states])
    horizon = math.floor(min(times[-1], trial.duration) * RATE + 1e-8) / RATE
    keep = times <= horizon + 1e-8
    predicted = resample_following(
        times[keep],
        np.asarray([row["ego_front_s_m"] for row in states])[keep],
        np.asarray([row["lead_front_s_m"] for row in states])[keep],
        states[0]["lead_length_m"],
        driver="video-replay",
        source_group=trial.group_id,
    )
    actual = {
        "time_s": predicted.time,
        "predicted_speed_mps": predicted.speed,
        "predicted_gap_m": predicted.gap,
        "predicted_acceleration_mps2": predicted.acceleration,
    }
    for name, signal in (
        ("speed", trial.speed),
        ("gap", trial.gap),
        ("acceleration", trial.acceleration),
    ):
        measured = np.interp(predicted.time, trial.time, signal)
        suffix = {"speed": "mps", "gap": "m", "acceleration": "mps2"}[name]
        values = actual[f"predicted_{name}_{suffix}"]
        actual[f"{name}_rmse_{suffix}"] = float(
            np.sqrt(np.mean((values - measured) ** 2))
        )
    errors = {}
    for name, values in actual.items():
        np.testing.assert_allclose(
            values, expected[name], atol=1e-8, rtol=0, err_msg=name
        )
        errors[name] = float(np.max(np.abs(np.asarray(values) - expected[name])))
    return errors


def replay(
    driver: Any,
    body: BodyView,
    trial: Any,
    expected: dict[str, Any],
    parameters: dict[str, float],
    path: Path,
    fps: int,
    desired_speed: float,
) -> dict[str, Any]:
    import embodied_human as eh
    import numpy as np
    from embodied_human.agents.driver import AttentiveDriver

    from myosuite_carla.driver_models import IdmConfig, ReflexDriver, ReflexDriverConfig
    from myosuite_carla.driving_env import FullBodyDrivingEnv
    from myosuite_carla.training import scenario_from_trial

    scenario = scenario_from_trial(trial, desired_speed)
    env = FullBodyDrivingEnv(scenario, driver=driver)
    temporary = path.with_suffix(".unverified.mp4")
    panel = None
    result: dict[str, Any] = {
        "status": "unverified",
        "seed": expected["seed"],
        "parameters": parameters,
    }
    try:
        _, initial = env.reset(seed=expected["seed"])
        agent = AttentiveDriver(
            eh.build_human(seed=expected["seed"]),
            env,
            reflex=ReflexDriver(
                env.dt,
                ReflexDriverConfig(
                    idm=IdmConfig(**parameters), perception_delay=env.dt
                ),
            ),
        )
        agent.reset(seed=expected["seed"])
        panel = Panel(
            temporary, expected["profile"], expected["source_group"], trial, fps, body
        )
        capture = frame_steps(scenario.duration, env.dt, fps)
        states = [initial["scenario"]]
        panel.frame(env, states[-1], dict(env._inputs))
        terminated = truncated = False
        for index in range(1, math.ceil(scenario.duration / env.dt) + 1):
            _, _, terminated, truncated, info = env.step(agent.act())
            if not all(
                np.isfinite(value).all()
                for value in (driver.data.qpos, driver.data.qvel, driver.data.act)
            ):
                raise RuntimeError("nonfinite MuJoCo state")
            states.append(info["scenario"])
            if index in capture:
                panel.frame(env, states[-1], info["inputs"])
            if terminated or truncated:
                break
        if (
            terminated
            or not truncated
            or any(row["collision"] or row["off_road"] for row in states)
        ):
            raise RuntimeError("replay did not complete safely")
        result["maximum_absolute_replay_errors"] = verify_replay(
            states, trial, expected
        )
        if panel.frames != len(capture):
            raise RuntimeError("video frame count differs from the physics schedule")
        result.update(
            status="verified",
            frames=panel.frames,
            fps=fps,
            seconds=len(capture) / fps,
            physics_steps=len(states) - 1,
        )
    finally:
        if panel is not None:
            panel.close()
        env.close()
        env.vehicle.close()
        result["surrogate_closed"] = not env.vehicle.active
    if not result["surrogate_closed"]:
        raise RuntimeError("surrogate cleanup failed")
    temporary.replace(path)
    result.update(file=path.name, sha256=sha256(path))
    return result


def ffmpeg(arguments: list[str]) -> None:
    import imageio_ffmpeg

    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            *arguments,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip())


def combine(left: Path, right: Path, output: Path) -> None:
    temporary = output.with_suffix(".pending.mp4")
    ffmpeg(
        [
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            "hstack=inputs=2",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--groups", nargs="+")
    args = parser.parse_args()
    from embodied_human.reporting import runtime_sources

    from myosuite_carla.empirical_following import load_following_protocol
    from myosuite_carla.fullbody import FullBodyDriverConfig, build_fullbody_driver

    report = json.loads(args.report.read_text())
    provenance = json.loads(
        (args.report.parent / report["provenance_report"]).read_text()
    )
    if (
        report["status"] != "evaluated"
        or sha256(args.source) != provenance["source"]["sha256"]
    ):
        raise ValueError(
            "report status or trajectory source differs from the calibration"
        )
    origins = runtime_sources()
    for name, original in provenance["runtime_sources"].items():
        if origins[name].get("python_source_sha256") != original.get(
            "python_source_sha256"
        ):
            raise ValueError(f"runtime source differs from the calibration: {name}")
    design = report["design"]
    config = design["configuration"]
    protocol = load_following_protocol(
        args.source,
        max_groups=sum(
            len(design["partitions"][role]) for role in ("train", "validation", "test")
        ),
        seed=config["seed"],
        desired_speed=config["desired_speed"],
        excluded_groups=design["excluded_source_groups"],
    )
    selected = {
        group
        for role in ("train", "validation", "test")
        for group in design["partitions"][role]
    }
    if set(protocol.groups) != selected:
        raise ValueError("selected source windows differ from the frozen design")
    body_config = FullBodyDriverConfig.fast()
    if (
        json.loads(json.dumps(asdict(body_config)))
        != report["closed_loop"]["driver_config"]
    ):
        raise ValueError("body configuration differs from the original calibration")
    groups = sorted(args.groups or design["partitions"]["test"])
    if not groups or set(groups) - set(design["partitions"]["test"]):
        raise ValueError("video groups must belong to the held out test partition")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    identity = {
        "report_sha256": sha256(args.report),
        "source_sha256": sha256(args.source),
        "fps": args.fps,
    }
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else {**identity, "trials": {}}
    )
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError(
            "output directory belongs to another report, source, or frame rate"
        )
    driver, body = None, None
    try:
        for group in groups:
            if group in manifest["trials"]:
                entry = manifest["trials"][group]
                if sha256(args.out / entry["file"]) != entry["sha256"]:
                    raise ValueError("an existing verified trial video changed")
                print(f"Reusing verified {group}", flush=True)
                continue
            if driver is None:
                driver = build_fullbody_driver(body_config)
                body = BodyView(driver)
            records = []
            stem = group.replace(":", "_")
            for label in ("baseline", "fitted"):
                expected = next(
                    row
                    for row in report["closed_loop"]["episodes"]
                    if row["source_group"] == group and row["profile"] == label
                )
                trial = protocol.trials_for(group)[expected["trial_index"]]
                print(f"Rendering {group} {label}", flush=True)
                records.append(
                    replay(
                        driver,
                        body,
                        trial,
                        expected,
                        report["closed_loop"]["profiles"][label],
                        args.out / f"{stem}-{label}.mp4",
                        args.fps,
                        config["desired_speed"],
                    )
                )
            paired = args.out / f"{stem}-paired.mp4"
            combine(*(args.out / row["file"] for row in records), paired)
            manifest["trials"][group] = {
                "status": "verified",
                "file": paired.name,
                "sha256": sha256(paired),
                "profiles": records,
                "seconds": records[0]["seconds"],
                "frames": records[0]["frames"],
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Verified paired video {paired}", flush=True)
    finally:
        if body is not None:
            body.close()
    playlist = args.out / "trials.txt"
    playlist.write_text(
        "".join(
            f"file '{manifest['trials'][group]['file']}'\n"
            for group in sorted(manifest["trials"])
        ),
        encoding="utf-8",
    )
    montage = args.out / "held-out-trials.mp4"
    ffmpeg(
        [
            "-f",
            "concat",
            "-safe",
            "1",
            "-i",
            str(playlist),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(montage),
        ]
    )
    manifest.update(
        montage={
            "file": montage.name,
            "sha256": sha256(montage),
            "seconds": sum(row["seconds"] for row in manifest["trials"].values()),
        },
        runtime_sources=origins,
        script_sha256=sha256(Path(__file__)),
        interpretation=(
            "Exact offline replay. Recorded trajectories are schematic references. "
            "The fitted candidate was rejected."
        ),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved montage {montage}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

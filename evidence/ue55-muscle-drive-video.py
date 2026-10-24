"""Record a MyoSuite muscle-driven CARLA drive as an annotated MP4.

Supplementary demonstration script; needs a rendering CARLA server (not
``-nullrhi``). One vehicle is driven continuously in three segments, each by
the package's real muscle/contact acceptance fixture: throttle from rest,
left steering while a scalar 0.3 cruise throttle is held, then braking to a
stop. Every MuJoCo control step is followed by exactly one CARLA frame. Each
video frame pairs CARLA's chase camera for that frame with a MuJoCo render of
the finger and mechanism, labelled with the measured controls and forces.
"""

from __future__ import annotations

import argparse
import importlib
import json
import queue
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from myosuite_carla import CarlaBridge, CarlaSession, SessionConfig
from myosuite_carla.inputs import Binding, CallbackSource, ControlMapper
from myosuite_carla.validation import (
    _select_blueprint,
    _settings,
    _settle_vehicle,
    _spawn,
    build_muscle_driver,
)

SEGMENTS = (
    ("throttle", 200, 0.015, "Muscles press the throttle pedal"),
    ("steering", 100, 0.05, "Muscles turn the wheel left (scalar 0.3 cruise)"),
    ("brake", 175, 0.01, "Muscles press the brake pedal"),
)
# Fixed views: lookat = scale * mechanism body position + offset (metres).
CAMERAS = {
    "throttle": (0.5, (0.0, 0.0, 0.12), 0.55, 138.0, -14.0),
    "brake": (0.5, (0.0, 0.0, 0.12), 0.55, 138.0, -14.0),
    "steering": (1.0, (0.0, 0.0, 0.1), 0.7, 150.0, -12.0),
}


def _font(size: int) -> Any:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bar(draw: Any, x: int, y: int, label: str, value: float, signed: bool) -> None:
    width, height = 260, 18
    draw.rectangle((x + 90, y, x + 90 + width, y + height), outline=(230, 230, 230))
    if signed:
        centre = x + 90 + width // 2
        end = centre + int(value * width / 2)
        draw.rectangle(
            (min(centre, end), y + 2, max(centre, end), y + height - 2),
            fill=(90, 170, 255),
        )
        draw.line((centre, y, centre, y + height), fill=(230, 230, 230))
    else:
        draw.rectangle(
            (x + 92, y + 2, x + 92 + int(value * (width - 4)), y + height - 2),
            fill=(255, 170, 60) if label == "brake" else (80, 220, 120),
        )
    draw.text((x, y - 3), f"{label}", font=_font(18), fill=(255, 255, 255))
    draw.text(
        (x + 100 + width, y - 3),
        f"{value:+.3f}" if signed else f"{value:.3f}",
        font=_font(18),
        fill=(255, 255, 255),
    )


def _panel(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(0, 0, 0, 150))
    image.alpha_composite(overlay)


def _compose(
    carla_rgb: np.ndarray,
    mujoco_rgb: np.ndarray,
    info: dict[str, Any],
) -> np.ndarray:
    scene = Image.fromarray(carla_rgb).convert("RGBA")
    _panel(scene, (16, 16, 640, 92))
    _panel(scene, (16, 560, 520, 704))
    draw = ImageDraw.Draw(scene)
    draw.text((30, 22), info["title"], font=_font(26), fill=(255, 255, 255))
    draw.text(
        (30, 58),
        f"CARLA 0.10.0 (UE5.5)  frame {info['frame']}  t = {info['time']:.2f} s  "
        f"speed {info['speed'] * 3.6:.1f} km/h",
        font=_font(18),
        fill=(220, 220, 220),
    )
    _bar(draw, 30, 580, "throttle", info["throttle"], False)
    _bar(draw, 30, 620, "brake", info["brake"], False)
    _bar(draw, 30, 660, "steer", info["steer"], True)
    body = Image.fromarray(mujoco_rgb).convert("RGBA")
    _panel(body, (12, 12, 628, 150))
    draw = ImageDraw.Draw(body)
    draw.text(
        (26, 18), "MyoSuite finger in MuJoCo", font=_font(26), fill=(255, 255, 255)
    )
    draw.text(
        (26, 56),
        f"{info['mechanism']}: angle {np.degrees(info['travel']):+.1f} deg"
        if info["mechanism"] == "steering joint"
        else f"{info['mechanism']}: travel {info['travel'] * 1000:.1f} mm",
        font=_font(18),
        fill=(220, 220, 220),
    )
    draw.text(
        (26, 84),
        f"contact {info['contact']:.1f} N   largest muscle {info['muscle']:.0f} N",
        font=_font(18),
        fill=(220, 220, 220),
    )
    draw.text(
        (26, 112),
        f"excitation {info['excitation']:.3f}; no external forces",
        font=_font(18),
        fill=(220, 220, 220),
    )
    frame = Image.new("RGB", (scene.width + body.width, scene.height))
    frame.paste(scene.convert("RGB"), (0, 0))
    frame.paste(body.convert("RGB"), (scene.width, 0))
    return np.asarray(frame)


def _title_card(width: int, height: int) -> np.ndarray:
    card = Image.new("RGB", (width, height), (12, 14, 18))
    draw = ImageDraw.Draw(card)
    lines = [
        ("MyoSuite muscles driving a CARLA vehicle", 44, (255, 255, 255)),
        (
            "myosuite-carla bridge  |  CARLA 0.10.0 (Unreal Engine 5.5)",
            26,
            (200, 200, 200),
        ),
        (
            "Muscle excitation -> finger contact -> passive pedal/wheel -> "
            "CARLA VehicleControl",
            22,
            (170, 200, 255),
        ),
        (
            "One 20 ms MuJoCo step per synchronous CARLA frame; real time playback",
            22,
            (170, 200, 255),
        ),
    ]
    y = height // 2 - 110
    for text, size, colour in lines:
        font = _font(size)
        left, _, right, bottom = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - (right - left)) // 2, y), text, font=font, fill=colour)
        y += bottom + 24
    return np.asarray(card)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/muscle_drive.mp4")
    )
    args = parser.parse_args()
    carla = importlib.import_module("carla")
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    original = _settings(world)
    if original["synchronous_mode"] or original["no_rendering_mode"]:
        raise RuntimeError("Use an asynchronous, rendering CARLA server")
    width, height, side = 1280, 720, 640
    drivers = {name: build_muscle_driver(name) for name, *_ in SEGMENTS}
    for driver in drivers.values():
        driver.model.vis.headlight.ambient[:] = 0.35
        driver.model.vis.headlight.diffuse[:] = 0.65
        driver.model.vis.global_.offwidth = max(driver.model.vis.global_.offwidth, side)
        driver.model.vis.global_.offheight = max(
            driver.model.vis.global_.offheight, height
        )
    blueprint = _select_blueprint(world, None)
    vehicle = _spawn(world, blueprint)
    session = CarlaSession(
        client,
        vehicle,
        SessionConfig(timeout_seconds=args.timeout),
        api=carla,
        own_vehicle=True,
    )
    camera = None
    drive_start: float | None = None
    images: queue.Queue[Any] = queue.Queue()
    log: dict[str, Any] = {"blueprint": blueprint, "segments": [], "frames": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        args.output,
        fps=50,
        codec="libx264",
        quality=7,
        pixelformat="yuv420p",
        macro_block_size=16,
    )
    try:
        vehicle.set_autopilot(False)
        session.start()
        log["settling_frames"] = _settle_vehicle(session, lambda *a: None)
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("fov", "90")
        camera = world.spawn_actor(
            camera_bp,
            carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12)),
            attach_to=vehicle,
        )
        camera.listen(images.put)
        card = _title_card(width + side, height)
        for _ in range(100):
            writer.append_data(card)
        for name, frames, excitation, title in SEGMENTS:
            driver = drivers[name]
            if name == "steering":
                driver.mapper = ControlMapper(
                    driver.model,
                    [
                        *driver.mapper.bindings,
                        Binding("throttle", CallbackSource(lambda model, data: 0.3)),
                    ],
                )
            renderer = mujoco.Renderer(driver.model, height, side)
            view = mujoco.MjvCamera()
            mechanism = getattr(driver.config.names, f"{name}_joint")
            body = driver.model.jnt_bodyid[driver.model.joint(mechanism).id]
            mujoco.mj_forward(driver.model, driver.data)
            scale, offset, distance, azimuth, elevation = CAMERAS[name]
            view.lookat[:] = scale * driver.data.xpos[body] + np.asarray(offset)
            view.distance, view.azimuth, view.elevation = distance, azimuth, elevation
            bridge = CarlaBridge(driver.mapper, session, 0.02)
            start = bridge.reset(driver.data).telemetry
            if drive_start is None:
                drive_start = start.elapsed_seconds
            log["segments"].append(
                {
                    "control": name,
                    "frames": frames,
                    "excitation": excitation,
                    "start_frame": start.frame,
                    "start_speed_mps": start.speed,
                }
            )
            for _ in range(frames):
                driver.advance(excitation)
                step = bridge.step(driver.data)
                telemetry, command = step.telemetry, step.sample.command
                image = images.get(timeout=args.timeout)
                while image.frame < telemetry.frame:
                    image = images.get(timeout=args.timeout)
                if image.frame != telemetry.frame:
                    raise RuntimeError("Camera frame does not match the CARLA tick")
                bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                    height, width, 4
                )
                renderer.update_scene(driver.data, view)
                measured = driver.measurement()
                info = {
                    "title": title,
                    "frame": telemetry.frame,
                    "time": telemetry.elapsed_seconds - drive_start,
                    "speed": telemetry.speed,
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "steer": command.steer,
                    "mechanism": mechanism.replace("cockpit_", "").replace("_", " "),
                    "travel": measured["travel"],
                    "contact": measured["contact_newtons"],
                    "muscle": measured["muscle_newtons"],
                    "excitation": excitation,
                }
                writer.append_data(
                    _compose(bgra[:, :, 2::-1].copy(), renderer.render(), info)
                )
                log["frames"].append({**info, "carla_image_frame": image.frame})
            renderer.close()
            log["segments"][-1]["end_speed_mps"] = telemetry.speed
            log["segments"][-1]["yaw_degrees"] = telemetry.rotation[1]
    finally:
        writer.close()
        if camera is not None:
            camera.stop()
            camera.destroy()
        session.close()
        log["settings_restored"] = _settings(world) == original
        args.output.with_suffix(".json").write_text(json.dumps(log, indent=1))
    print(
        json.dumps(
            {k: log[k] for k in ("blueprint", "segments", "settings_restored")},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

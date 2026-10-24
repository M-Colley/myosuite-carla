"""Record the live full-body drive: CARLA chase camera beside the MuJoCo body.

Supplementary demonstration script; needs a rendering CARLA server (not
``-nullrhi``). It runs ``run_fullbody_validation`` with an observer that
attaches an RGB camera to the vehicle and pairs every synchronous CARLA frame
with an off-screen MuJoCo render of the seated MyoSuite full body. Overlays
show the measured controls, CARLA's applied gear state and contact forces.
"""

from __future__ import annotations

import argparse
import json
import queue
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from myosuite_carla.fullbody_validation import run_fullbody_validation
from myosuite_carla.validation import LiveValidationError

TITLES = {
    "hold": "Left foot brakes, left index holds the parking brake",
    "handbrake_hold_throttle": "Right foot presses throttle; parking brake holds car",
    "release_handbrake": "Left index releases the parking brake",
    "throttle": "Right foot presses the throttle",
    "steer_left": "Hands push-pull the wheel left; foot keeps throttle",
    "steer_right": "Hands push-pull the wheel right; foot keeps throttle",
    "straighten": "Hands return the wheel",
    "brake": "Left foot presses the brake",
    "select_reverse": "Right little finger holds reverse",
    "reverse_throttle": "Reverse: right foot presses the throttle",
    "reverse_brake": "Reverse: left foot brakes",
    "leave_reverse": "Little finger releases reverse",
    "shift_1": "Right index pulls the paddle: manual gear 1",
    "release_paddle_1": "Paddle released",
    "gear_1_throttle": "Gear 1: right foot presses the throttle",
    "shift_2": "Right index pulls the paddle: gear 2",
    "gear_2_throttle": "Gear 2",
    "shift_3": "Right index pulls the paddle: gear 3",
    "gear_3_throttle": "Gear 3",
    "shift_automatic": "Fourth pull returns to automatic",
    "coast": "Coasting",
    "handbrake_stop": "Left index applies the parking brake; no pedal",
    "park": "Parked",
}


def _font(size: int) -> Any:
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _shade(image: Image.Image, box: tuple[int, int, int, int]) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(box, fill=(0, 0, 0, 150))
    image.alpha_composite(overlay)


def _bar(draw: Any, y: int, label: str, value: float, signed: bool) -> None:
    x, width, height = 30, 240, 18
    draw.rectangle((x + 90, y, x + 90 + width, y + height), outline=(230, 230, 230))
    if signed:
        centre = x + 90 + width // 2
        end = centre + int(value * width / 2)
        draw.rectangle(
            (min(centre, end), y + 2, max(centre, end), y + height - 2),
            fill=(90, 170, 255),
        )
        draw.line((centre, y, centre, y + height), fill=(230, 230, 230))
        text = f"{value:+.2f}"
    else:
        colour = (255, 170, 60) if label == "brake" else (80, 220, 120)
        draw.rectangle(
            (x + 92, y + 2, x + 92 + int(value * (width - 4)), y + height - 2),
            fill=colour,
        )
        text = f"{value:.2f}"
    draw.text((x, y - 3), label, font=_font(18), fill=(255, 255, 255))
    draw.text((x + 100 + width, y - 3), text, font=_font(18), fill=(255, 255, 255))


class VideoObserver:
    """Attach a chase camera and write one composed video frame per tick."""

    def __init__(self, output: Path, carla: Any) -> None:
        self.output = output
        self.carla = carla
        self.width, self.height, self.side = 1280, 720, 640
        self.images: queue.Queue[Any] = queue.Queue()
        self.camera: Any = None
        self.writer: Any = None
        self.renderer: Any = None
        self.view = mujoco.MjvCamera()
        self.matched = 0
        self.drive_start: float | None = None

    def start(self, session: Any, driver: Any) -> None:
        world = session.world
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(self.width))
        blueprint.set_attribute("image_size_y", str(self.height))
        blueprint.set_attribute("fov", "90")
        self.camera = world.spawn_actor(
            blueprint,
            self.carla.Transform(
                self.carla.Location(x=-6.5, z=2.8), self.carla.Rotation(pitch=-12)
            ),
            attach_to=session.vehicle,
        )
        self.camera.listen(self.images.put)
        driver.model.vis.headlight.ambient[:] = 0.4
        driver.model.vis.headlight.diffuse[:] = 0.6
        self.renderer = mujoco.Renderer(driver.model, self.height, self.side)
        self.view.lookat[:] = (-0.03, -0.12, 0.95)
        self.view.distance, self.view.azimuth, self.view.elevation = 2.5, 155, -10
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            self.output,
            fps=50,
            codec="libx264",
            quality=7,
            pixelformat="yuv420p",
            macro_block_size=16,
        )
        card = Image.new("RGB", (self.width + self.side, self.height), (12, 14, 18))
        draw = ImageDraw.Draw(card)
        lines = [
            ("One MyoSuite full body drives a CARLA vehicle", 44, (255, 255, 255)),
            (
                "myofullbody, 416 muscles  |  CARLA 0.10.0 (Unreal Engine 5.5)",
                26,
                (200, 200, 200),
            ),
            (
                "Feet: throttle and brake pedals   Hands: push-pull wheel cradles",
                22,
                (170, 200, 255),
            ),
            (
                "Fingers: gear paddle, reverse and parking-brake buttons",
                22,
                (170, 200, 255),
            ),
            (
                "Controls are measured from mechanism contact; one 20 ms MuJoCo "
                "step per synchronous CARLA frame",
                20,
                (170, 200, 255),
            ),
        ]
        y = self.height // 2 - 140
        for text, size, colour in lines:
            font = _font(size)
            left, _, right, bottom = draw.textbbox((0, 0), text, font=font)
            draw.text(
                ((card.width - (right - left)) // 2, y), text, font=font, fill=colour
            )
            y += bottom + 22
        frame = np.asarray(card)
        for _ in range(125):
            self.writer.append_data(frame)

    def frame(self, record: dict[str, Any], driver: Any) -> None:
        image = self.images.get(timeout=60)
        while image.frame < record["frame"]:
            image = self.images.get(timeout=60)
        if image.frame != record["frame"]:
            raise RuntimeError("Camera frame does not match the CARLA tick")
        self.matched += 1
        if self.drive_start is None:
            self.drive_start = record["vehicle"]["elapsed_seconds"]
        bgra = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
            self.height, self.width, 4
        )
        scene = Image.fromarray(bgra[:, :, 2::-1].copy()).convert("RGBA")
        command, applied = record["command"], record["applied"]
        vehicle, mechanics = record["vehicle"], record["mechanics"]
        _shade(scene, (16, 16, 760, 92))
        _shade(scene, (16, 560, 470, 704))
        draw = ImageDraw.Draw(scene)
        draw.text(
            (30, 22), TITLES[record["phase"]], font=_font(26), fill=(255, 255, 255)
        )
        gear = (
            f"M{applied['gear']}"
            if applied["manual_gear_shift"]
            else ("R" if applied["reverse"] else "A")
        )
        draw.text(
            (30, 58),
            f"t = {vehicle['elapsed_seconds'] - self.drive_start:5.2f} s   "
            f"{vehicle['speed'] * 3.6:5.1f} km/h   gear {gear}   "
            f"parking brake {'ON' if applied['hand_brake'] else 'off'}",
            font=_font(18),
            fill=(220, 220, 220),
        )
        _bar(draw, 580, "throttle", command["throttle"], False)
        _bar(draw, 620, "brake", command["brake"], False)
        _bar(draw, 660, "steer", command["steer"], True)
        self.renderer.update_scene(driver.data, self.view)
        body = Image.fromarray(self.renderer.render()).convert("RGBA")
        _shade(body, (12, 12, 628, 180))
        draw = ImageDraw.Draw(body)
        draw.text(
            (26, 18),
            "MyoSuite full body (MuJoCo)",
            font=_font(26),
            fill=(255, 255, 255),
        )
        rows = [
            f"feet: throttle {mechanics['throttle_contact_newtons']:5.1f} N   "
            f"brake {mechanics['brake_contact_newtons']:5.1f} N",
            f"hands on wheel cradles {mechanics['steering_contact_newtons']:5.1f} N",
            f"fingers: gear {mechanics['gear_button_contact_newtons']:4.1f} N  "
            f"reverse {mechanics['reverse_button_contact_newtons']:4.1f} N  "
            f"park {mechanics['park_button_contact_newtons']:4.1f} N",
            f"largest driving muscle {mechanics['largest_muscle_newtons']:4.0f} N; "
            "no external forces",
        ]
        for index, text in enumerate(rows):
            draw.text((26, 56 + 28 * index), text, font=_font(17), fill=(220, 220, 220))
        frame = Image.new("RGB", (self.width + self.side, self.height))
        frame.paste(scene.convert("RGB"), (0, 0))
        frame.paste(body.convert("RGB"), (self.width, 0))
        self.writer.append_data(np.asarray(frame))

    def close(self) -> None:
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
        if self.writer is not None:
            self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--target-ref", default="0.10.0")
    parser.add_argument("--build-ref", default="0.10.0")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fullbody.mp4"))
    args = parser.parse_args()
    import carla  # type: ignore[import-not-found]

    observer = VideoObserver(args.output, carla)
    try:
        report = run_fullbody_validation(
            args.host,
            args.port,
            timeout=args.timeout,
            target_ref=args.target_ref,
            build_ref=args.build_ref,
            observer=observer,
        )
    except LiveValidationError as error:
        report = error.report
    report["video"] = {"path": str(args.output), "matched_frames": observer.matched}
    args.output.with_suffix(".json").write_text(json.dumps(report), encoding="utf-8")
    print(report["status"].upper(), "matched camera frames", observer.matched)
    for check in report.get("checks", []):
        print(
            f"  [{'x' if check['passed'] else ' '}] {check['name']}: {check['detail']}"
        )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

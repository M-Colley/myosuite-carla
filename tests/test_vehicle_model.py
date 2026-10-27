"""The fitted single-track vehicle surrogate."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from myosuite_carla import DriveRecorder, VehicleCommand
from myosuite_carla.session import SessionConfig
from myosuite_carla.vehicle_model import (
    SPEED_KNOTS,
    THROTTLE_KNOTS,
    SurrogateVehicle,
    VehicleModelParams,
    carla_nissan_patrol,
    evaluate_vehicle_model,
    fit_vehicle_model,
    segments_from_recording,
)

TRUTH = VehicleModelParams(
    drive=tuple(
        tuple((0.5 + 3.0 * throttle) * (1.0 + 0.03 * speed) for speed in SPEED_KNOTS)
        for throttle in THROTTLE_KNOTS
    ),
    reverse_drive=(0.4, 0.9, 1.4, 1.9),
    steer_gain=tuple(0.5 - 0.02 * speed for speed in SPEED_KNOTS),
    rolling=0.2,
    quadratic_drag=0.004,
    brake_decel=7.0,
    hand_brake_decel=4.0,
    throttle_lag=0.15,
    steer_lag=0.08,
)


def _drive(vehicle: SurrogateVehicle, commands, recorder=None, label=""):
    telemetry = vehicle.snapshot()
    previous = VehicleCommand()
    if recorder is not None:
        recorder.add(0.0, telemetry, command=previous.as_dict(), phase=label)
    for index, command in enumerate(commands):
        telemetry = vehicle.step(command)
        if recorder is not None:
            recorder.add(
                telemetry.elapsed_seconds,
                telemetry,
                command=command.as_dict(),
                phase=label,
            )
    return telemetry


def test_surrogate_launches_brakes_and_reverses_physically() -> None:
    with SurrogateVehicle(TRUTH) as vehicle:
        rest = _drive(vehicle, [VehicleCommand()] * 50)
        assert rest.speed == 0.0 and rest.location == (0.0, 0.0, 0.0)
        moving = _drive(vehicle, [VehicleCommand(throttle=0.6)] * 150)
        assert moving.speed > 5.0 and moving.location[0] > 5.0
        # Heading 0 drives along +x; the right axis is +y in CARLA's world.
        assert abs(moving.location[1]) < 1e-9
        stopped = _drive(vehicle, [VehicleCommand(brake=1.0)] * 150)
        assert stopped.speed == 0.0
        held = _drive(vehicle, [VehicleCommand(brake=1.0)] * 20)
        assert held.location == stopped.location
        backwards = _drive(vehicle, [VehicleCommand(throttle=0.5, reverse=True)] * 100)
        assert backwards.velocity[0] < -0.5
        assert backwards.location[0] < stopped.location[0]
    with pytest.raises(RuntimeError):
        vehicle.step(VehicleCommand())


def test_right_steer_turns_right_with_matching_acceleration() -> None:
    from myosuite_carla.recording import _vehicle_columns

    with SurrogateVehicle(TRUTH, yaw=90.0, speed=5.0) as vehicle:
        telemetry = _drive(vehicle, [VehicleCommand(throttle=0.3, steer=0.4)] * 50)
    assert telemetry.rotation[1] > 90.0
    assert telemetry.angular_velocity[2] > 0
    columns = _vehicle_columns(telemetry)
    yaw_rate = math.radians(telemetry.angular_velocity[2])
    assert columns["a_right_mps2"] == pytest.approx(telemetry.speed * yaw_rate)


def test_params_validate_and_round_trip_through_json() -> None:
    restored = VehicleModelParams.from_dict(json.loads(json.dumps(TRUTH.as_dict())))
    assert restored == TRUTH
    with pytest.raises(ValueError, match="steer_gain"):
        VehicleModelParams(steer_gain=(2.0,) * len(SPEED_KNOTS))
    with pytest.raises(ValueError, match="speed_knots"):
        VehicleModelParams(speed_knots=(1.0, 0.5) + SPEED_KNOTS[2:])
    with pytest.raises(ValueError, match="non-negative"):
        VehicleModelParams(brake_decel=-1.0)
    assert isinstance(carla_nissan_patrol(), VehicleModelParams)


def _synthetic(seed: int):
    rng = np.random.default_rng(seed)
    recorder = DriveRecorder(body=False)
    vehicle = SurrogateVehicle(TRUTH, SessionConfig(fixed_delta_seconds=0.02)).start()
    for index in range(12):
        vehicle.reset()
        throttle = float(rng.uniform(0.2, 1.0))
        steer = float(rng.uniform(-0.5, 0.5))
        commands = (
            [VehicleCommand()] * 10
            + [VehicleCommand(throttle=throttle)] * int(rng.integers(150, 300))
            + [
                VehicleCommand(throttle=throttle, steer=steer * math.sin(0.05 * k))
                for k in range(120)
            ]
            + [VehicleCommand()] * 50
            + [VehicleCommand(brake=float(rng.uniform(0.2, 1.0)))] * 150
        )
        if index % 4 == 0:
            commands += [VehicleCommand(hand_brake=True)] * 20
            commands += [VehicleCommand(throttle=0.6)] * 100
            commands += [VehicleCommand(hand_brake=True)] * 200
        if index % 3 == 0:
            commands += [VehicleCommand(throttle=throttle, reverse=True)] * 120
            commands += [VehicleCommand(brake=0.5, reverse=True)] * 100
        part = DriveRecorder(body=False)
        _drive(vehicle, commands, part, f"s{seed}-{index}")
        recorder._rows.extend(
            {**row, "frame": row["frame"] + 100_000 * index} for row in part._rows
        )
    return segments_from_recording(recorder.recording())


def test_fit_recovers_known_parameters_from_its_own_drives() -> None:
    pytest.importorskip("scipy")
    train, test = _synthetic(0), _synthetic(1)
    params, report = fit_vehicle_model(train, smoothing=0.05)
    assert report["longitudinal_samples"] > 5000
    assert params.brake_decel == pytest.approx(TRUTH.brake_decel, rel=0.05)
    assert params.hand_brake_decel == pytest.approx(TRUTH.hand_brake_decel, rel=0.1)
    assert params.throttle_lag == pytest.approx(TRUTH.throttle_lag, abs=0.03)
    assert params.steer_lag == pytest.approx(TRUTH.steer_lag, abs=0.03)
    identified = np.asarray(report["steer_knots_identified"])
    gains = np.asarray(params.steer_gain)
    assert identified.sum() >= 3
    np.testing.assert_allclose(
        gains[identified], np.asarray(TRUTH.steer_gain)[identified], rtol=0.1
    )
    # Knots without steering data hold the nearest identified gain.
    known = np.flatnonzero(identified)
    for index in np.flatnonzero(~identified):
        nearest = known[np.argmin(np.abs(known - index))]
        assert gains[index] == gains[nearest]
    result = evaluate_vehicle_model(params, test, group=lambda name: name.split("-")[0])
    summary = result["summary"]
    assert summary["speed_rmse_mps"] < 0.1
    assert summary["position_error_3s_m"] < 0.2 * summary["baseline_error_3s_m"]
    assert set(result["by_group"]) == {"s1"}


def test_fit_supports_custom_speed_and_throttle_knots() -> None:
    pytest.importorskip("scipy")
    template = VehicleModelParams(
        speed_knots=(0.0, 5.0, 10.0),
        throttle_knots=(0.5, 1.0),
        drive=((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
        reverse_drive=(0.5, 1.0),
        steer_gain=(0.3, 0.3, 0.3),
    )
    # Use an actual rolling segment with enough steering and speed support.
    segment = _synthetic(0)[0]
    params, report = fit_vehicle_model(
        [segment], template=template, throttle_lags=(0.0,)
    )
    assert params.speed_knots == template.speed_knots
    assert params.throttle_knots == template.throttle_knots
    assert report["lateral_samples"] > 0


def test_bridge_drives_the_surrogate_like_a_carla_session() -> None:
    from types import SimpleNamespace

    from myosuite_carla import CarlaBridge
    from myosuite_carla.inputs import ControlSample

    vehicle = SurrogateVehicle(TRUTH).start()
    command = VehicleCommand(throttle=0.5)
    mapper = SimpleNamespace(
        reset=lambda: None,
        sample=lambda data: ControlSample(command, {"throttle": 0.5}),
    )
    bridge = CarlaBridge(mapper, vehicle, vehicle.config.fixed_delta_seconds)
    data = SimpleNamespace(time=0.0)
    bridge.reset(data)
    for _ in range(100):
        data.time += 0.02
        frame = bridge.step(data)
    assert frame.telemetry.speed > 2.0
    assert frame.telemetry.frame == 100

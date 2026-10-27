"""Mix and match with embodied-ui: devices in the driver, engines and vehicles."""

from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

embodied_ui = pytest.importorskip("embodied_ui")

from embodied_ui import Indicator, Interface, Pedal, Pose, ToggleSwitch  # noqa: E402
from embodied_ui.backends import LoopbackEngine  # noqa: E402
from embodied_ui.engine import Bridge, Route, Routes  # noqa: E402

from myosuite_carla import DriverIntent, FullBodyDriverConfig, VehicleCommand  # noqa: E402
from myosuite_carla.interfaces import (  # noqa: E402
    EngineVehicleSession,
    VehicleSessionEngine,
    interface_hook,
)
from myosuite_carla.vehicle_model import SurrogateVehicle  # noqa: E402

DT = 0.02


def unity_car(command: Any, time: float) -> dict[str, Any]:
    """A Unity car at 5 m/s along +z, yawed 90 degrees right, turning left."""
    s = math.sqrt(0.5)
    return {
        "vehicle.pose": [1.0, 0.5, 5.0 * time, s, 0.0, s, 0.0],
        "vehicle.velocity": [0.0, 0.0, 5.0],
        "vehicle.acceleration": [0.0, 0.0, command.controls["vehicle.throttle"]],
        "vehicle.angular_velocity": [0.0, -0.5, 0.0],
    }


def test_engine_vehicle_session_converts_unity_telemetry_to_carla() -> None:
    session = EngineVehicleSession(
        LoopbackEngine(
            DT,
            unity_car,
            reset=lambda seed: unity_car(
                SimpleNamespace(controls={"vehicle.throttle": 0.0}), 0.0
            ),
        ),
        convention="unity",
    )
    with session:
        first = session.snapshot()
        telemetry = session.step(VehicleCommand(throttle=0.4))
    assert first.frame == 0 and telemetry.frame == 1
    assert telemetry.elapsed_seconds == pytest.approx(DT)
    # Unity (x right, y up, z forward) to CARLA (x forward, y right, z up).
    np.testing.assert_allclose(telemetry.location, (5.0 * DT, 1.0, 0.5))
    np.testing.assert_allclose(telemetry.velocity, (5.0, 0.0, 0.0))
    np.testing.assert_allclose(telemetry.acceleration, (0.4, 0.0, 0.0))
    assert telemetry.rotation[1] == pytest.approx(90.0)  # yawed right
    # A left turn is a negative yaw rate in both left-handed frames.
    np.testing.assert_allclose(
        telemetry.angular_velocity, (0, 0, -math.degrees(0.5)), atol=1e-9
    )
    commands = session.engine.commands
    assert commands[-1].controls["vehicle.throttle"] == pytest.approx(0.4)
    assert set(commands[-1].controls) >= {
        "vehicle.steer",
        "vehicle.brake",
        "vehicle.gear",
    }


def test_engine_vehicle_session_detects_clock_slips() -> None:
    class Slipping(LoopbackEngine):
        def step(self, command: Any) -> Any:
            state = super().step(command)
            return type(state)(state.frame, state.time * 1.5, state.outputs)

    session = EngineVehicleSession(
        Slipping(
            DT,
            unity_car,
            reset=lambda seed: unity_car(
                SimpleNamespace(controls={"vehicle.throttle": 0.0}), 0.0
            ),
        ),
        convention="unity",
    )
    session.start()
    with pytest.raises(Exception, match="fixed delta"):
        session.step(VehicleCommand())
    with pytest.raises(RuntimeError, match="healthy"):
        session.step(VehicleCommand())


def test_engine_vehicle_session_resets_engine_and_recovers_after_failure() -> None:
    engine = LoopbackEngine(
        DT,
        unity_car,
        reset=lambda seed: unity_car(
            SimpleNamespace(controls={"vehicle.throttle": 0.0}), 0.0
        ),
    )
    with EngineVehicleSession(engine, convention="unity") as session:
        session.step(VehicleCommand(throttle=0.4))
        session._failed = True
        first = session.reset(seed=7)
        assert session.active
        assert first.frame == 0 and first.elapsed_seconds == 0.0
        assert not engine.commands
        assert session.step(VehicleCommand()).frame == 1


def test_vehicle_session_engine_restores_initial_surrogate_state() -> None:
    vehicle = SurrogateVehicle(location=(3.0, 4.0, 0.5), yaw=90.0, speed=2.0)
    engine = VehicleSessionEngine(vehicle)
    initial = engine.start(SimpleNamespace(dt=DT))
    for k in range(100):
        engine.step(SimpleNamespace(controls={"vehicle.throttle": 1.0}, time=k * DT))
    reset = engine.reset(seed=7)
    assert reset.frame == 0 and reset.time == 0.0
    np.testing.assert_allclose(
        reset.outputs["vehicle.pose"], initial.outputs["vehicle.pose"]
    )
    np.testing.assert_allclose(
        reset.outputs["vehicle.velocity"], initial.outputs["vehicle.velocity"]
    )
    again = engine.step(SimpleNamespace(controls={"vehicle.throttle": 1.0}, time=0.0))
    engine.reset(seed=7)
    repeated = engine.step(
        SimpleNamespace(controls={"vehicle.throttle": 1.0}, time=0.0)
    )
    assert repeated.outputs == again.outputs
    engine.close()


def test_vehicle_session_engine_uses_explicit_live_reset() -> None:
    vehicle = SurrogateVehicle().start()
    calls: list[Any] = []

    def reset(session: Any, seed: int | None, options: Any) -> None:
        calls.append((seed, options))
        session.reset(speed=3.0)

    engine = VehicleSessionEngine(vehicle, reset_vehicle=reset)
    engine.start(SimpleNamespace(dt=DT))
    state = engine.reset(seed=11, options={"trial": 2})
    assert calls == [(11, {"trial": 2})]
    assert state.outputs["vehicle.speed"] == pytest.approx(3.0)
    engine.close()


def test_vehicle_session_engine_requires_live_reset_callback() -> None:
    with SurrogateVehicle() as vehicle:
        borrowed = SimpleNamespace(
            config=vehicle.config, active=True, snapshot=vehicle.snapshot
        )
        engine = VehicleSessionEngine(borrowed)
        engine.start(SimpleNamespace(dt=DT))
        with pytest.raises(RuntimeError, match="reset_vehicle is required"):
            engine.reset(seed=1)


def test_a_mujoco_pedal_drives_the_fitted_vehicle_surrogate() -> None:
    import mujoco
    from embodied_ui.probe import Probe

    interface = Interface("car", (Pedal("throttle"),))
    spec = mujoco.MjSpec()
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    probe = Probe(pos=(0, 0, 0.05), stiffness=5000.0)
    probe.attach(spec)
    interface.attach(spec)
    model = spec.compile()
    data = mujoco.MjData(model)
    runtime = interface.bind(model)
    tip = probe.bind(model)
    engine = VehicleSessionEngine(SurrogateVehicle())
    routes = Routes(routes=(Route("throttle.value", "vehicle.throttle"),))
    with Bridge(runtime, engine, routes, engine_frame="carla") as bridge:
        bridge.reset(data)
        for k in range(150):
            tip.target(data, (0, 0, 0.016 - (0.03 if k > 10 else 0.0)))
            runtime.step(data, 10)
            frame = bridge.step(data)
    assert frame.command.controls["vehicle.throttle"] > 0.4
    assert frame.state.outputs["vehicle.speed"] > 1.0
    assert frame.state.frame == 150


def test_interface_hook_switches_carla_lights() -> None:
    pytest.importorskip("carla")
    import carla

    states: list[Any] = []
    vehicle = SimpleNamespace(
        set_light_state=states.append, get_light_state=lambda: states[-1]
    )
    reading = embodied_ui.Reading(0.0, {"hazard.state": 1.0, "beam.state": 0.0})
    outputs: list[dict[str, float]] = []
    runtime = SimpleNamespace(set_outputs=lambda data, values: outputs.append(values))
    driver = SimpleNamespace(
        interface_reading=lambda: reading, interfaces=(runtime,), data=None
    )
    routes = Routes(
        routes=(
            Route("hazard.state", "lights.hazard"),
            Route("beam.state", "lights.low_beam"),
        ),
        feedback=(Route("lights.right_blinker", "lamp.on"),),
    )
    interface_hook(driver, routes)(SimpleNamespace(vehicle=vehicle), None)
    flags = carla.VehicleLightState
    assert int(states[0]) == int(flags.LeftBlinker | flags.RightBlinker)
    assert outputs == [{"lamp.on": 1.0}]


@pytest.fixture(scope="module")
def driver_with_interface() -> Any:
    pytest.importorskip("myosuite.core.model_builder")
    myo_sim = pytest.importorskip("myo_sim")
    try:
        myo_sim.load_spec("myofullbody")
    except Exception as error:  # pragma: no cover - depends on myo-sim release
        pytest.skip(f"myo_sim has no myofullbody: {error}")
    from myosuite_carla import build_fullbody_driver

    # A hazard switch on the wheel hub facing the driver (+y is backward in
    # the wheel frame's axis) and a lamp next to it.
    dashboard = Interface(
        "dash",
        (
            ToggleSwitch(
                "hazard",
                Pose((0.0, 0.05, 0.0), (0.5, 0.5, 0.5, 0.5)),
                mount="cockpit_steering_body",
            ),
            Indicator(
                "lamp",
                Pose((0.03, 0.05, 0.0), (0.5, 0.5, 0.5, 0.5)),
                mount="cockpit_steering_body",
            ),
        ),
    )
    return build_fullbody_driver(FullBodyDriverConfig.fast(interfaces=(dashboard,)))


def test_devices_compose_into_the_full_body_driver(driver_with_interface: Any) -> None:
    driver = driver_with_interface
    model = driver.model
    assert model.joint("dash/hazard/angle").id >= 0
    assert (
        model.body("dash/hazard").parentid[0] == model.body("cockpit_steering_body").id
    )
    reading = driver.interface_reading()
    assert reading["hazard.state"] in (0.0, 1.0)
    # Reading twice in one period returns the same object.
    assert driver.interface_reading() is reading
    joint = model.joint("dash/hazard/angle").qposadr[0]
    driver.data.qpos[joint] = 0.3
    for _ in range(10):
        driver.advance(DriverIntent())
    reading = driver.interface_reading()
    assert reading["hazard.state"] == 1.0
    assert [e.kind for e in reading.events] in ([], ["toggle"])
    from myosuite_carla.fullbody import summary

    report = summary(driver)
    json.dumps(report)
    assert report["config"]["interfaces"][0]["name"] == "dash"


def test_interface_feel_keeps_the_driver_free_of_external_forces(
    driver_with_interface: Any,
) -> None:
    driver = driver_with_interface
    driver.reset()
    for _ in range(5):
        driver.advance(DriverIntent())
    switch = driver.model.joint("dash/hazard/angle").dofadr[0]
    # The switch's detents are the only applied force besides the wheel feel.
    applied = np.flatnonzero(driver.data.qfrc_applied)
    assert set(applied) <= {switch, driver._steering_dof}

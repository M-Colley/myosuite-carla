"""Real MuJoCo stepping against frame-accurate CARLA doubles."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from myosuite_carla import (
    Binding,
    CallbackSource,
    CarlaBridge,
    CarlaWrapper,
    ControlMapper,
    JointPosition,
    SessionConfig,
    SynchronizationError,
    UnipolarCalibration,
)


def driver_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep="0.002" gravity="0 0 0"/>'
        '<worldbody><body><joint name="pedal" type="slide"/>'
        '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
    )


class Driver(gym.Env):
    def __init__(self) -> None:
        self.model = driver_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.action_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)
        self.end = False
        self.fail = False
        self.closed = False

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, Any]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos.astype(np.float32).copy(), {"seed": seed}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self.fail:
            raise RuntimeError("driver failed")
        self.data.qfrc_applied[0] = float(action[0])
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        mujoco.mj_forward(self.model, self.data)
        return (
            self.data.qpos.astype(np.float32).copy(),
            7.0,
            self.end,
            False,
            {"driver": 1},
        )

    def close(self) -> None:
        self.closed = True


def mapper(model: mujoco.MjModel) -> ControlMapper:
    return ControlMapper(
        model,
        [Binding("throttle", JointPosition("pedal"), UnipolarCalibration(0, 0.001))],
    )


def test_physical_motion_controls_vehicle_and_preserves_gym_contract(
    simulation: Any,
) -> None:
    driver = Driver()
    with simulation.session() as session:
        bridge = CarlaBridge(mapper(driver.model), session, 0.02)
        env = CarlaWrapper(driver, bridge)
        assert env.action_space is driver.action_space
        assert env.observation_space is driver.observation_space
        with pytest.raises(gym.error.ResetNeeded):
            env.step([1])
        observation, info = env.reset(seed=42)
        assert observation.dtype == np.float32
        assert info["seed"] == 42
        assert info["carla"]["vehicle"]["frame"] == 100
        observation, reward, terminated, truncated, info = env.step([1])
        assert reward == 7 and not terminated and not truncated
        assert observation[0] > 0
        assert info["driver"] == 1
        assert info["carla"]["control"]["throttle"] > 0
        assert info["carla"]["vehicle"]["frame"] == 101
        assert info["carla"]["simulation_time"] == pytest.approx(0.02)
        assert (
            simulation.vehicle.controls[-1].throttle
            == info["carla"]["control"]["throttle"]
        )
        driver.end = True
        env.step([0])
        assert simulation.vehicle.controls[-1].brake == 1
        with pytest.raises(gym.error.ResetNeeded):
            env.step([0])
        env.reset(seed=42)
        assert simulation.world.frame == 102
        env.step([1])


def test_bad_timing_never_advances_carla(simulation: Any) -> None:
    model = driver_model()
    data = mujoco.MjData(model)
    with simulation.session() as session:
        with pytest.raises(ValueError, match="equal"):
            CarlaBridge(mapper(model), session, 0.01)
        bridge = CarlaBridge(mapper(model), session, 0.02)
        with pytest.raises(RuntimeError, match="Reset"):
            bridge.step(data)
        bridge.reset(data)
        mujoco.mj_step(model, data)
        with pytest.raises(SynchronizationError, match="MuJoCo"):
            bridge.step(data)
        assert simulation.world.frame == 100
        assert simulation.vehicle.controls[-1].brake == 1


def test_extensions_read_measured_extra_signals_before_tick(simulation: Any) -> None:
    model = driver_model()
    data = mujoco.MjData(model)
    extra = ControlMapper(
        model, [Binding("lights", CallbackSource(lambda m, d: d.time))]
    )
    events: list[Any] = []
    with simulation.session() as session:
        bridge = CarlaBridge(
            extra,
            session,
            0.02,
            hooks=(lambda s, v: events.append((s.world.frame, v.values["lights"])),),
        )
        bridge.reset(data)
        mujoco.mj_step(model, data, nstep=10)
        bridge.step(data)
        assert events == [(100, pytest.approx(0.02))]


def test_nonfinite_signal_and_hook_failures_brake(simulation: Any) -> None:
    model = driver_model()
    data = mujoco.MjData(model)
    with simulation.session() as session:
        source = ControlMapper(
            model,
            [
                Binding(
                    "throttle",
                    CallbackSource(lambda m, d: float("nan") if d.time else 0.0),
                )
            ],
        )
        bridge = CarlaBridge(source, session, 0.02)
        bridge.reset(data)
        mujoco.mj_step(model, data, nstep=10)
        with pytest.raises(ValueError, match="finite"):
            bridge.step(data)
        assert simulation.world.frame == 100
        assert simulation.vehicle.controls[-1].brake == 1
        bridge = CarlaBridge(
            mapper(model),
            session,
            0.02,
            hooks=(lambda s, v: (_ for _ in ()).throw(RuntimeError("hook failed")),),
        )
        bridge.reset(data)
        mujoco.mj_step(model, data, nstep=10)
        with pytest.raises(RuntimeError, match="hook failed"):
            bridge.step(data)
        assert simulation.world.frame == 100


def test_driver_exception_brakes_and_requires_reset(simulation: Any) -> None:
    driver = Driver()
    with simulation.session() as session:
        env = CarlaWrapper(driver, CarlaBridge(mapper(driver.model), session, 0.02))
        env.reset()
        driver.fail = True
        with pytest.raises(RuntimeError, match="driver failed"):
            env.step([1])
        assert simulation.vehicle.controls[-1].brake == 1
        with pytest.raises(gym.error.ResetNeeded):
            env.step([1])


def test_wrapper_rejects_different_model_and_timestep(simulation: Any) -> None:
    driver = Driver()
    session = simulation.session()
    with pytest.raises(ValueError, match="model"):
        CarlaWrapper(driver, CarlaBridge(mapper(driver_model()), session, 0.02))
    driver.frame_skip = 5
    with pytest.raises(ValueError, match="frame_skip"):
        CarlaWrapper(driver, CarlaBridge(mapper(driver.model), session, 0.02))


def test_close_ownership(simulation: Any) -> None:
    driver = Driver()
    session = simulation.session().start()
    env = CarlaWrapper(
        driver, CarlaBridge(mapper(driver.model), session, 0.02), close_session=True
    )
    env.close()
    assert driver.closed
    assert not session.active


def test_reset_failure_invalidates_direct_bridge_and_brakes(simulation: Any) -> None:
    model = driver_model()
    data = mujoco.MjData(model)
    with simulation.session() as session:
        bridge = CarlaBridge(mapper(model), session, 0.02)
        bridge.reset(data)
        data.qvel[0] = 0.01
        mujoco.mj_step(model, data, nstep=10)
        bridge.step(data)
        assert simulation.vehicle.controls[-1].throttle > 0
        data.time = float("nan")
        with pytest.raises(ValueError, match="finite"):
            bridge.reset(data)
        assert simulation.vehicle.controls[-1].brake == 1
        data.time = 0.02
        with pytest.raises(RuntimeError, match="Reset"):
            bridge.step(data)


def test_tiny_period_cannot_advance_without_mujoco_step(simulation: Any) -> None:
    model = driver_model()
    data = mujoco.MjData(model)
    with simulation.session(SessionConfig(fixed_delta_seconds=1e-10)) as session:
        bridge = CarlaBridge(mapper(model), session, 1e-10)
        bridge.reset(data)
        with pytest.raises(SynchronizationError, match="MuJoCo"):
            bridge.step(data)
        assert simulation.world.frame == 100

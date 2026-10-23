"""Fault injection preserves wrapper reset and resource ownership contracts."""

from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from myosuite_carla import CarlaBridge, CarlaWrapper, ControlMapper
from myosuite_carla.bridge import BridgeFrame


class Driver(gym.Env):
    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(
            '<mujoco><option timestep=".002"/></mujoco>'
        )
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.closed = False

    def reset(
        self, *, seed: int | None = None, options: Any = None
    ) -> tuple[Any, dict]:
        super().reset(seed=seed)
        return np.zeros(1), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict]:
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        return np.zeros(1), 0.0, False, False, {}

    def close(self) -> None:
        self.closed = True


def test_failed_reset_feedback_export_keeps_wrapper_guarded(
    simulation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = Driver()
    with simulation.session() as session:
        env = CarlaWrapper(
            driver, CarlaBridge(ControlMapper(driver.model, []), session, 0.02)
        )

        def fail_export(frame: BridgeFrame) -> dict:
            raise RuntimeError("feedback export failed")

        with monkeypatch.context() as patch:
            patch.setattr(BridgeFrame, "as_dict", fail_export)
            with pytest.raises(RuntimeError, match="feedback export failed"):
                env.reset()
        with pytest.raises(gym.error.ResetNeeded):
            env.step([0.0])
        assert simulation.world.frame == 100
        assert simulation.vehicle.controls[-1].brake == 1.0
        env.reset()
        env.step([0.0])


def test_close_preserves_driver_error_and_attempts_session_cleanup(
    simulation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    driver = Driver()
    session = simulation.session().start()
    env = CarlaWrapper(
        driver,
        CarlaBridge(ControlMapper(driver.model, []), session, 0.02),
        close_session=True,
    )
    driver_error = RuntimeError("driver close failed")
    session_error = ValueError("session close failed")
    attempted: list[str] = []

    def close_driver() -> None:
        attempted.append("driver")
        raise driver_error

    def close_session() -> None:
        attempted.append("session")
        raise session_error

    with monkeypatch.context() as patch:
        patch.setattr(driver, "close", close_driver)
        patch.setattr(session, "close", close_session)
        with pytest.raises(RuntimeError, match="driver close failed") as captured:
            env.close()
    assert captured.value is driver_error
    assert captured.value.__cause__ is session_error
    assert attempted == ["driver", "session"]
    session.close()


@pytest.mark.parametrize("close_session", ["false", 0, 1, None])
def test_close_ownership_flag_requires_boolean(
    simulation: Any, close_session: object
) -> None:
    driver = Driver()
    session = simulation.session()
    with pytest.raises(TypeError, match="close_session"):
        CarlaWrapper(
            driver,
            CarlaBridge(ControlMapper(driver.model, []), session, 0.02),
            close_session=close_session,
        )

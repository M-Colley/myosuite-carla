"""Gymnasium adapter retaining the driver's muscle action and observation spaces."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym

from .bridge import CarlaBridge


class CarlaWrapper(gym.Wrapper):
    """Add CARLA feedback to info while preserving the driver's task contract.

    Args:
        env: CPU MyoSuite environment created with ``myosuite.make_env``.
        bridge: Bridge bound to exactly this environment's compiled model.
        close_session: Close the session when the wrapper closes.
    """

    def __init__(
        self, env: gym.Env, bridge: CarlaBridge, *, close_session: bool = False
    ) -> None:
        if not isinstance(close_session, bool):
            raise TypeError("close_session must be a bool")
        super().__init__(env)
        driver = env.unwrapped
        if getattr(driver, "model", None) is not bridge.mapper.model:
            raise ValueError("Bridge mapper must be bound to env.unwrapped.model")
        if not hasattr(driver, "data"):
            raise TypeError("A CPU MuJoCo driver environment is required")
        control_dt = float(driver.model.opt.timestep) * int(driver.frame_skip)
        if not math.isclose(control_dt, bridge.control_dt, rel_tol=1e-9, abs_tol=0):
            raise ValueError(
                "Driver frame_skip and timestep differ from bridge control_dt"
            )
        self.bridge = bridge
        self.close_session = close_session
        self._needs_reset = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the muscle task and anchor feedback to the current CARLA frame."""
        self._needs_reset = True
        try:
            observation, info = self.env.reset(seed=seed, options=options)
            frame = self.bridge.reset(self.env.unwrapped.data)
            feedback = {**info, "carla": frame.as_dict()}
            self._needs_reset = False
            return observation, feedback
        except Exception:
            self.bridge.session._stop_after_failure()
            raise

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step muscles, measure the mechanisms, and return CARLA feedback in info."""
        if self._needs_reset:
            raise gym.error.ResetNeeded("Reset the driver before stepping")
        try:
            observation, reward, terminated, truncated, info = self.env.step(action)
            frame = self.bridge.step(self.env.unwrapped.data)
            if terminated or truncated:
                self._needs_reset = True
                self.bridge.session.stop()
            return (
                observation,
                reward,
                terminated,
                truncated,
                {**info, "carla": frame.as_dict()},
            )
        except Exception:
            self._needs_reset = True
            self.bridge.session._stop_after_failure()
            raise

    def close(self) -> None:
        """Close the driver and, if requested, its CARLA session."""
        self._needs_reset = True
        try:
            self.bridge.session._stop_after_failure()
        finally:
            try:
                self.env.close()
            except Exception as driver_error:
                if self.close_session:
                    try:
                        self.bridge.session.close()
                    except Exception as session_error:
                        raise driver_error from session_error
                raise
            else:
                if self.close_session:
                    self.bridge.session.close()

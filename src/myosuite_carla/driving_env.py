"""An unregistered Gymnasium task in which the full-body driver drives a scenario.

:class:`FullBodyDrivingEnv` advances the muscle-driven :class:`FullBodyDriver`
one control period per step, measures its mechanisms, drives a vehicle in
lockstep through :class:`CarlaBridge` and follows a :class:`Scenario`. The
vehicle is the fitted :class:`SurrogateVehicle` by default (fast, CPU only) or
any started ``CarlaSession``. Before a takeover request the scenario
automation drives while the driver's inputs are still measured.

Actions are the driver's motor program (:class:`DriverIntent`): foot and
finger excitations and the push-pull elbow set-point offset. Observations are
named (see :attr:`FullBodyDrivingEnv.observation_names`): vehicle and lane
state, the lead vehicle, takeover state, measured mechanism inputs, driving
joint angles and velocities and driving muscle activations. Rewards are a
weighted sum of named terms reported in ``info["reward_terms"]``.

The environment is intentionally not registered. MyoSuite registration through
``myosuite.make_env`` is a follow-up that must meet MyoSuite's CPU/mjlab task
contract.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .bridge import CarlaBridge
from .fullbody import (
    DriverIntent,
    FullBodyDriver,
    FullBodyDriverConfig,
    build_fullbody_driver,
)
from .recording import DriveRecorder
from .scenario import (
    SCENARIOS,
    AutomationConfig,
    Scenario,
    ScenarioMonitor,
    automation_command,
)
from .session import SessionConfig
from .vehicle_loads import VehicleMotion
from .vehicle_model import SurrogateVehicle

ACTION_FIELDS = (
    "right_foot",
    "left_foot",
    "steer",
    "gear_finger",
    "reverse_finger",
    "park_finger",
)
#: Free joints whose angles and velocities are observed.
DRIVING_JOINTS = (
    "ankle_angle_r",
    "ankle_angle_l",
    "elbow_flexion_r",
    "elbow_flexion_l",
    "mcp2_flexion_r",
    "mcp5_flexion_r",
    "mcp2_flexion_l",
)
_SCENARIO_FEATURES = (
    "speed_mps",
    "target_speed_mps",
    "lane_offset_m",
    "heading_error_rad",
    "near_angle_rad",
    "far_angle_rad",
    "lead_present",
    "gap_m",
    "closing_speed_mps",
    "inverse_ttc_per_s",
    "automation",
    "takeover_requested",
    "time_since_request_s",
)
_VEHICLE_FEATURES = ("a_forward_mps2", "a_right_mps2", "yaw_rate_rad_s")
_INPUT_FEATURES = (
    "throttle",
    "brake",
    "steer",
    "hand_brake",
    "reverse",
    "gear",
    "manual_gear_shift",
)


@dataclass(frozen=True)
class TaskConfig:
    """Reward weights, termination and perception settings.

    Args:
        lane: Weight of squared lane offset in half-lane widths.
        heading: Weight of squared heading error in radians.
        speed: Weight of squared speed error relative to the target speed.
        headway: Weight of time-headway shortfall below ``safe_headway``.
        comfort: Weight of squared jerk (per 10 m/s^3) and lateral
            acceleration (per 3 m/s^2), each capped at 25.
        effort: Weight of the mean squared driving-muscle activation.
        automation: Per-second penalty while automation still drives after a
            takeover request.
        collision: Terminal penalty for hitting the lead vehicle.
        off_road: Terminal penalty for leaving the road.
        safe_headway: Time headway below which the headway term acts, s.
        follow_headway: Time gap that limits the desired speed behind a lead
            vehicle, s; the speed term then rewards ``min(target, (gap -
            follow_gap) / follow_headway)`` as the Intelligent Driver Model's
            equilibrium does, instead of the free-road target.
        follow_gap: Standstill gap behind a lead vehicle, m.
        near_distance: Near-point distance for the observation, m.
        far_preview: Far-point preview time for the observation, s.
        gap_clip: Gap observed when no lead vehicle is close, m.
    """

    lane: float = 1.0
    heading: float = 0.5
    speed: float = 1.0
    headway: float = 1.0
    comfort: float = 0.1
    effort: float = 0.1
    automation: float = 1.0
    collision: float = 100.0
    off_road: float = 50.0
    safe_headway: float = 1.0
    follow_headway: float = 1.6
    follow_gap: float = 2.0
    near_distance: float = 8.0
    far_preview: float = 2.0
    gap_clip: float = 100.0

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{item.name} must be finite and non-negative")
        for name in ("safe_headway", "near_distance"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")


def _reset_surrogate(vehicle: Any, scenario: Scenario) -> None:
    x, y, yaw = scenario.start_pose()
    vehicle.reset((x, y, 0.0), yaw, scenario.start_speed)
    vehicle.start()


class FullBodyDrivingEnv(gym.Env[NDArray[np.float32], NDArray[np.float32]]):
    """Drive a scenario with the muscle-actuated full body.

    Args:
        scenario: What happens around the ego vehicle.
        task: Reward weights and perception settings.
        driver: An existing driver; built from ``driver_config`` otherwise.
        driver_config: Configuration used when ``driver`` is not given.
        vehicle: Session-like vehicle; defaults to a :class:`SurrogateVehicle`.
        reset_vehicle: Places ``vehicle`` at the scenario start; required for a
            CARLA session (see :func:`~myosuite_carla.carla_scenario.teleport`).
        hooks: Bridge control hooks, e.g. a ``CarlaLeadVehicle``.
        automation: Gains of the pre-takeover automation.
        recorder: Optional recorder of every frame and the scenario events.
        monitor: Monitor to use, e.g. one shared with a `CarlaLeadVehicle`;
            it must follow `scenario`.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | None = None,
        task: TaskConfig | None = None,
        *,
        driver: FullBodyDriver | None = None,
        driver_config: FullBodyDriverConfig | None = None,
        vehicle: Any = None,
        reset_vehicle: Callable[[Any, Scenario], None] | None = None,
        hooks: tuple[Callable[[Any, Any], None], ...] = (),
        automation: AutomationConfig | None = None,
        recorder: DriveRecorder | None = None,
        monitor: ScenarioMonitor | None = None,
    ) -> None:
        super().__init__()
        self.scenario = scenario if scenario is not None else Scenario()
        self.task = task if task is not None else TaskConfig()
        self.driver = (
            driver if driver is not None else build_fullbody_driver(driver_config)
        )
        dt = self.driver.config.control_dt
        if vehicle is None:
            vehicle = SurrogateVehicle(config=SessionConfig(fixed_delta_seconds=dt))
            reset_vehicle = reset_vehicle or _reset_surrogate
        if reset_vehicle is None:
            raise ValueError("reset_vehicle is required for a non-surrogate vehicle")
        self.vehicle = vehicle
        self._reset_vehicle = reset_vehicle
        self._hooks = tuple(hooks)
        self.automation_config = (
            automation if automation is not None else AutomationConfig()
        )
        self.recorder = recorder
        if monitor is not None and monitor.scenario is not self.scenario:
            raise ValueError("monitor must follow the environment's scenario")
        self.monitor = (
            monitor if monitor is not None else ScenarioMonitor(self.scenario)
        )
        model = self.driver.model
        self._joints = [
            (
                int(model.jnt_qposadr[model.joint(name).id]),
                int(model.jnt_dofadr[model.joint(name).id]),
            )
            for name in DRIVING_JOINTS
        ]
        self._muscles = list(self.driver._driving_muscles)
        self._actadr = [int(model.actuator_actadr[index]) for index in self._muscles]
        self.observation_names = (
            *_SCENARIO_FEATURES,
            *_VEHICLE_FEATURES,
            *(f"input_{name}" for name in _INPUT_FEATURES),
            *(f"{name}_angle" for name in DRIVING_JOINTS),
            *(f"{name}_velocity" for name in DRIVING_JOINTS),
            *(f"{model.actuator(index).name}_activation" for index in self._muscles),
        )
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (len(self.observation_names),), np.float32
        )
        self.action_space = gym.spaces.Box(
            np.array([0, 0, -0.5, 0, 0, 0], dtype=np.float32),
            np.array([1, 1, 0.5, 1, 1, 1], dtype=np.float32),
        )
        self.dt = dt
        self.bridge: CarlaBridge | None = None
        self.state: dict[str, float] = {}
        self._telemetry: Any = None
        self._inputs: dict[str, float] = {}
        self._previous_accel = 0.0
        self._needs_reset = True

    def _observe(self) -> NDArray[np.float32]:
        state, telemetry, task = self.monitor.state, self._telemetry, self.task
        yaw = math.radians(telemetry.rotation[1])
        acceleration = np.asarray(telemetry.acceleration, dtype=float)
        forward = acceleration[0] * math.cos(yaw) + acceleration[1] * math.sin(yaw)
        right = -acceleration[0] * math.sin(yaw) + acceleration[1] * math.cos(yaw)
        speed = state["speed_mps"]
        ttc = state["time_to_collision_s"]
        request = self.scenario.takeover
        since = (
            state["time"] - request.time
            if request is not None and state["takeover_requested"]
            else 0.0
        )
        named = {
            "speed_mps": speed,
            "target_speed_mps": state["target_speed_mps"],
            "lane_offset_m": state["lane_offset_m"],
            "heading_error_rad": math.radians(state["heading_error_deg"]),
            "near_angle_rad": self.monitor.view_angle(telemetry, task.near_distance),
            "far_angle_rad": self.monitor.view_angle(
                telemetry, max(task.near_distance + 2.0, speed * task.far_preview)
            ),
            "lead_present": state["lead_present"],
            "gap_m": min(state["gap_m"], task.gap_clip),
            "closing_speed_mps": (
                speed - state["lead_speed_mps"] if state["lead_present"] else 0.0
            ),
            "inverse_ttc_per_s": 1.0 / ttc if math.isfinite(ttc) and ttc > 0 else 0.0,
            "automation": state["automation"],
            "takeover_requested": state["takeover_requested"],
            "time_since_request_s": since,
            "a_forward_mps2": forward,
            "a_right_mps2": right,
            "yaw_rate_rad_s": math.radians(telemetry.angular_velocity[2]),
        }
        self.state = {**named, **{f"input_{k}": v for k, v in self._inputs.items()}}
        data = self.driver.data
        vector = np.concatenate(
            [
                [named[name] for name in (*_SCENARIO_FEATURES, *_VEHICLE_FEATURES)],
                [float(self._inputs.get(name, 0.0)) for name in _INPUT_FEATURES],
                [data.qpos[qpos] for qpos, _ in self._joints],
                [data.qvel[dof] for _, dof in self._joints],
                data.act[self._actadr],
            ]
        )
        return vector.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Reset the body, place the vehicle and start the scenario."""
        super().reset(seed=seed)
        self.driver.reset(seed=seed)
        self._reset_vehicle(self.vehicle, self.scenario)
        self.bridge = CarlaBridge(
            self.driver.mapper, self.vehicle, self.dt, hooks=self._hooks
        )
        frame = self.bridge.reset(self.driver.data)
        self._telemetry = frame.telemetry
        self._inputs = dict(frame.sample.values)
        actor = getattr(self.vehicle, "vehicle", None)
        if actor is not None:
            from .carla_scenario import vehicle_footprint

            self.monitor.set_footprints(ego=vehicle_footprint(actor))
        self.monitor.reset(frame.telemetry)
        self._previous_accel = 0.0
        self._needs_reset = False
        observation = self._observe()
        return observation, {
            "scenario": dict(self.monitor.state),
            "events": list(self.monitor.events),
            "vehicle_footprints": self.monitor.footprints,
        }

    def set_scenario(self, scenario: Scenario) -> None:
        """Use another scenario from the next reset, with a fresh monitor."""
        if not isinstance(scenario, Scenario):
            raise TypeError("scenario must be a Scenario")
        self.scenario = scenario
        self.monitor = ScenarioMonitor(scenario)
        self._needs_reset = True

    def intent(self, action: Any) -> DriverIntent:
        """Convert an action vector into a validated driver intent."""
        values = np.asarray(action, dtype=float)
        if values.shape != (len(ACTION_FIELDS),) or not np.all(np.isfinite(values)):
            raise ValueError(f"action must be {len(ACTION_FIELDS)} finite numbers")
        values = np.clip(values, self.action_space.low, self.action_space.high)
        return DriverIntent(**dict(zip(ACTION_FIELDS, map(float, values))))

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Advance body, vehicle and scenario by one control period."""
        if self._needs_reset or self.bridge is None:
            raise gym.error.ResetNeeded("Reset the environment before stepping")
        intent = action if isinstance(action, DriverIntent) else self.intent(action)
        previous = self._telemetry
        self.driver.advance(intent)
        command = (
            automation_command(self.monitor, previous, self.automation_config)
            if self.monitor.automation
            else None
        )
        frame = self.bridge.step(self.driver.data, command=command)
        telemetry = frame.telemetry
        self.driver.set_vehicle_motion(
            VehicleMotion.from_telemetry(telemetry, previous, self.dt)
        )
        self._telemetry = telemetry
        self._inputs = dict(frame.sample.values)
        events_before = len(self.monitor.events)
        state = self.monitor.update(telemetry, self.dt, self._inputs)
        observation = self._observe()
        terms = self._reward_terms(state)
        collided = bool(state["collision"])
        off_road = bool(state["off_road"])
        terminated = collided or off_road
        truncated = state["time"] >= self.scenario.duration - 1e-9
        if self.recorder is not None:
            self.recorder.add(
                frame.simulation_time,
                telemetry,
                command=frame.command.as_dict(),
                inputs=self._inputs,
                intent=vars(intent),
                mechanics=self.driver.measurement(),
                phase=self.scenario.name,
                model=self.driver.model,
                data=self.driver.data,
            )
            self.recorder.metadata["events"] = list(self.monitor.events)
            self.recorder.metadata["vehicle_footprints"] = self.monitor.footprints
        if terminated or truncated:
            self._needs_reset = True
        info = {
            "scenario": dict(state),
            "events": self.monitor.events[events_before:],
            "inputs": dict(self._inputs),
            "command": frame.command.as_dict(),
            "automation": bool(self.monitor.automation),
            "reward_terms": terms,
            "collision": collided,
            "off_road": off_road,
            "vehicle_footprints": self.monitor.footprints,
        }
        reading = self.driver.interface_reading()
        if reading is not None:
            info["interface"] = reading.as_dict()
        return observation, float(sum(terms.values())), terminated, truncated, info

    def _reward_terms(self, state: dict[str, float]) -> dict[str, float]:
        task, scenario = self.task, self.scenario
        driving = not state["automation"]
        half_lane = 0.5 * scenario.lane.width
        accel = self.state["a_forward_mps2"]
        jerk = (accel - self._previous_accel) / self.dt
        self._previous_accel = accel
        headway = state["time_headway_s"]
        activation = self.driver.data.act[self._actadr]
        target = max(scenario.target_speed, 1.0)
        desired = scenario.target_speed
        if state["lead_present"] and math.isfinite(state["gap_m"]):
            spacing = (state["gap_m"] - task.follow_gap) / max(
                task.follow_headway, 1e-6
            )
            desired = min(desired, max(spacing, 0.0))
        terms = {
            "lane": -task.lane * (state["lane_offset_m"] / half_lane) ** 2 * driving,
            "heading": -task.heading
            * math.radians(state["heading_error_deg"]) ** 2
            * driving,
            "speed": -task.speed
            * ((state["speed_mps"] - desired) / target) ** 2
            * driving,
            "headway": -task.headway
            * max(0.0, (task.safe_headway - headway) / task.safe_headway)
            if math.isfinite(headway)
            else 0.0,
            # Each comfort term is capped so one transient cannot dominate.
            "comfort": -task.comfort
            * (
                min((jerk / 10.0) ** 2, 25.0)
                + min((self.state["a_right_mps2"] / 3.0) ** 2, 25.0)
            ),
            "effort": -task.effort * float(np.mean(activation**2)),
            "automation": -task.automation
            * self.dt
            * float(state["automation"] and state["takeover_requested"]),
            "collision": -task.collision * state["collision"],
            "off_road": -task.off_road * float(state["off_road"]),
        }
        return {name: float(value) for name, value in terms.items()}

    def close(self) -> None:
        """Release the vehicle; a CARLA session is stopped, not closed."""
        self._needs_reset = True
        stop = getattr(self.vehicle, "_stop_after_failure", None)
        if stop is not None:
            stop()


def make_driving_env(
    scenario: str | Scenario = "free", *, fast: bool = True, **kwargs: Any
) -> FullBodyDrivingEnv:
    """Build an environment, and its own driver, by scenario name.

    Picklable via ``functools.partial`` for ``gymnasium.vector.AsyncVectorEnv``,
    where each worker process builds one CPU driver and one surrogate vehicle.

    Args:
        scenario: A name in :data:`~myosuite_carla.scenario.SCENARIOS` or a
            :class:`Scenario`.
        fast: Use :meth:`FullBodyDriverConfig.fast` unless ``driver_config``
            is given.
        **kwargs: Passed to :class:`FullBodyDrivingEnv`.
    """
    chosen = SCENARIOS[scenario]() if isinstance(scenario, str) else scenario
    if fast and "driver" not in kwargs and "driver_config" not in kwargs:
        kwargs["driver_config"] = FullBodyDriverConfig.fast()
    return FullBodyDrivingEnv(chosen, **kwargs)

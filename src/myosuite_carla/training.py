"""Training components for full-body driving policies.

These wrap :class:`~myosuite_carla.driving_env.FullBodyDrivingEnv` for
reinforcement learning without importing a learning library; the
Stable-Baselines3 callbacks and imitation helpers are in
:mod:`myosuite_carla.training_sb3`.

* :class:`ScenarioSampler` draws a scenario per episode from curriculum
  stages: free driving on randomly curved roads, car following that replays a
  recorded human leader (NGSIM), a braking lead vehicle, a lane change and an
  automation takeover. Earlier stages stay in the mix after a stage unlocks.
* :class:`DrivingTaskWrapper` maps a normalized action in one of three
  modes. ``demand`` (default) adds a bounded residual to the
  :class:`~.driver_models.ReflexDriver`'s desired acceleration and steer, which
  its proprioceptive loops and the muscles then execute, as arm_force_driving's
  residual acts on the demand side of its control law; ``intent`` adds it to
  the driver intent; ``direct`` maps the action to the intent. A zero residual
  reproduces the reflex driver exactly, and its last demand or intent is
  appended to the observation. It randomizes the vehicle surrogate's
  parameters, penalizes residual size and action changes, and at the end of
  each episode reports success and, for car following, the trial on NGSIM's
  10 Hz processing for a human-likeness judge.

Observations stay raw; normalize them in the vectorized environment (for
example ``VecNormalize``) and save the statistics with the policy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from .driver_models import ReflexDriver, ReflexDriverConfig
from .fullbody import DriverIntent
from .ngsim import FollowingTrial, resample_following
from .scenario import (
    VEHICLE_LENGTH,
    Lane,
    LaneChange,
    LeadVehicle,
    Scenario,
    TakeoverRequest,
)
from .vehicle_model import SurrogateVehicle, VehicleModelParams

STAGES = ("free", "follow", "lead_brake", "lane_change", "takeover")
MODES = ("demand", "intent", "direct")
#: Default fields and residual scales per mode.
MODE_FIELDS = {
    "demand": (("acceleration", "steer"), (2.0, 0.2)),
    "intent": (("right_foot", "left_foot", "steer"), (0.02, 0.05, 0.1)),
    "direct": (("right_foot", "left_foot", "steer"), (1.0, 1.0, 1.0)),
}
#: Bounds of every intent field (excitation or elbow set-point offset).
INTENT_BOUNDS = {
    "right_foot": (0.0, 1.0),
    "left_foot": (0.0, 1.0),
    "steer": (-0.5, 0.5),
    "gear_finger": (0.0, 1.0),
    "reverse_finger": (0.0, 1.0),
    "park_finger": (0.0, 1.0),
}


def curved_lane(
    rng: np.random.Generator,
    length: float,
    *,
    min_radius: float = 150.0,
    max_radius: float = 600.0,
    curve_probability: float = 0.6,
    step: float = 2.0,
) -> Lane:
    """Return a lane of straights and constant-radius arcs, 40-120 m each."""
    points = [(0.0, 0.0)]
    heading = 0.0
    travelled = 0.0
    while travelled < length:
        piece = float(rng.uniform(40.0, 120.0))
        curvature = 0.0
        if rng.random() < curve_probability:
            radius = float(rng.uniform(min_radius, max_radius))
            curvature = float(rng.choice([-1.0, 1.0])) / radius
        for _ in range(max(1, int(piece / step))):
            heading += curvature * step
            x, y = points[-1]
            points.append((x + step * math.cos(heading), y + step * math.sin(heading)))
            travelled += step
    return Lane(tuple(points))


def scenario_from_trial(
    trial: FollowingTrial, top_speed: float = 16.0, curved: Lane | None = None
) -> Scenario:
    """Replay a human trial's leader in front of the ego vehicle.

    The ego starts at the human follower's speed and gap, and the lead
    follows the recorded leader speed, so the ego's response can be compared
    with the human follower's for the same leader.
    """
    length = trial.duration * (top_speed + 3.0) + 150.0
    return Scenario(
        name="follow",
        lane=curved if curved is not None else Lane.straight(length=length),
        start_speed=float(trial.speed[0]),
        target_speed=float(top_speed),
        duration=trial.duration,
        lead=LeadVehicle(
            gap=float(trial.gap[0]),
            speed=float(trial.leader_speed[0]),
            brake_time=None,
            profile=tuple(zip(trial.time.tolist(), trial.leader_speed.tolist())),
        ),
    )


@dataclass(frozen=True)
class SamplerConfig:
    """Ranges of the scenario parameters drawn per episode.

    Args:
        duration: Episode length in seconds (NGSIM trials keep their own).
        speed: Target and lead speed range, m/s (within the surrogate's
            identified range).
        curved: Draw curved roads for every stage.
        lead_gap: Initial gap range of a braking lead, m.
        lead_deceleration: Braking deceleration range, m/s^2.
        lane_change_duration: Reference duration range of a lane change, s.
        takeover_budget: Automation budget after a takeover request, s.
    """

    duration: float = 20.0
    speed: tuple[float, float] = (6.0, 13.0)
    curved: bool = True
    lead_gap: tuple[float, float] = (15.0, 40.0)
    lead_deceleration: tuple[float, float] = (2.0, 6.0)
    lane_change_duration: tuple[float, float] = (3.0, 6.0)
    takeover_budget: tuple[float, float] = (4.0, 8.0)


@dataclass
class ScenarioSampler:
    """Draw episode scenarios from the unlocked curriculum stages.

    Args:
        stages: Stage names in curriculum order, from :data:`STAGES`.
        trials: Human car-following trials replayed by the ``follow`` stage;
            without them that stage uses a smooth random leader profile.
        config: Parameter ranges.
        stage: Index of the highest unlocked stage.
    """

    stages: tuple[str, ...] = STAGES
    trials: Sequence[FollowingTrial] = field(default_factory=tuple)
    config: SamplerConfig = field(default_factory=SamplerConfig)
    stage: int = 0

    def __post_init__(self) -> None:
        unknown = set(self.stages) - set(STAGES)
        if unknown or not self.stages:
            raise ValueError(f"stages must be drawn from {STAGES}")
        self.stage = int(np.clip(self.stage, 0, len(self.stages) - 1))

    def _lane(self, rng: np.random.Generator, length: float) -> Lane:
        if self.config.curved:
            return curved_lane(rng, length)
        return Lane.straight(length=length)

    def sample(self, rng: np.random.Generator, name: str | None = None) -> Scenario:
        """Draw a scenario of ``name`` or of a random unlocked stage."""
        config = self.config
        kind = name or str(rng.choice(self.stages[: self.stage + 1]))
        speed = float(rng.uniform(*config.speed))
        duration = config.duration
        if kind == "follow" and self.trials:
            trial = self.trials[int(rng.integers(len(self.trials)))]
            top = config.speed[1] + 3.0
            length = trial.duration * (top + 3.0) + 150.0
            return scenario_from_trial(trial, top, self._lane(rng, length))
        if kind == "follow":
            times = np.arange(0.0, duration + 0.5, 0.5)
            phases = rng.uniform(0, 2 * math.pi, 3)
            periods = rng.uniform(4.0, 15.0, 3)
            profile = speed + sum(
                rng.uniform(0.5, 2.0) * np.sin(2 * math.pi * times / period + phase)
                for period, phase in zip(periods, phases)
            )
            profile = np.clip(profile, 0.0, config.speed[1] + 2.0)
            lead = LeadVehicle(
                gap=float(rng.uniform(*config.lead_gap)),
                speed=float(profile[0]),
                brake_time=None,
                profile=tuple(zip(times.tolist(), profile.tolist())),
            )
            start, target = float(profile[0]), config.speed[1] + 3.0
        elif kind == "lead_brake":
            brake = float(rng.uniform(3.0, 0.6 * duration))
            lead = LeadVehicle(
                gap=float(rng.uniform(*config.lead_gap)),
                speed=speed,
                brake_time=brake,
                deceleration=float(rng.uniform(*config.lead_deceleration)),
                brake_to=float(rng.uniform(0.0, 0.5 * speed)),
                resume_time=float(rng.uniform(brake + 3.0, brake + 8.0)),
            )
            start, target = speed, speed
        else:
            lead = None
            start, target = float(rng.uniform(0.5, 1.0)) * speed, speed
        length = duration * (target + 3.0) + 150.0
        scenario = Scenario(
            name=kind,
            lane=self._lane(rng, length),
            start_speed=start,
            target_speed=target,
            duration=duration,
            lead=lead,
        )
        if kind == "lane_change":
            scenario = replace(
                scenario,
                lane_changes=(
                    LaneChange(
                        time=float(rng.uniform(3.0, 8.0)),
                        offset=float(rng.choice([-3.5, 3.5])),
                        duration=float(rng.uniform(*config.lane_change_duration)),
                    ),
                ),
            )
        if kind == "takeover":
            scenario = replace(
                scenario,
                start_speed=speed,
                takeover=TakeoverRequest(
                    time=float(rng.uniform(3.0, 8.0)),
                    budget=float(rng.uniform(*config.takeover_budget)),
                ),
            )
        return scenario


def perturb_vehicle(
    params: VehicleModelParams, rng: np.random.Generator, spread: float
) -> VehicleModelParams:
    """Scale drive, brake, steering, resistance and lags by U(1-s, 1+s) each."""
    if spread <= 0:
        return params

    def factor() -> float:
        return float(rng.uniform(1.0 - spread, 1.0 + spread))

    drive, brake, steer, resist, lag = (factor() for _ in range(5))
    gains = np.clip(np.asarray(params.steer_gain) * steer, 0.005, 1.4)
    return replace(
        params,
        drive=np.asarray(params.drive) * drive,
        reverse_drive=np.asarray(params.reverse_drive) * drive,
        brake_decel=params.brake_decel * brake,
        hand_brake_decel=params.hand_brake_decel * brake,
        steer_gain=gains,
        rolling=params.rolling * resist,
        quadratic_drag=params.quadratic_drag * resist,
        throttle_lag=params.throttle_lag * lag,
        steer_lag=params.steer_lag * lag,
    )


def _success(stage: str, record: dict[str, Any]) -> bool:
    if record["collision"] or record["off_road"]:
        return False
    if record["lane_rms_m"] > 0.5:
        return False
    if stage == "free":
        return record["final_speed_error"] < 0.2
    if stage == "lane_change":
        return abs(record["final_lane_offset_m"]) < 0.5
    if stage == "takeover":
        return record["takeover_time_s"] is not None
    if stage in ("follow", "lead_brake"):
        return record["min_gap_m"] > 1.0
    return True


class DrivingTaskWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Normalized actions, curriculum scenarios and episode reports.

    Args:
        env: The full-body driving environment.
        sampler: Draws a scenario at every reset; None keeps the env's own.
        mode: ``"demand"``, ``"intent"`` or ``"direct"`` (see the module).
        fields: Controlled fields; defaults per mode (:data:`MODE_FIELDS`).
            Intent fields the policy does not control come from the reflex
            driver (residual modes) or stay zero (direct).
        residual_scale: Largest residual per field in m/s^2, normalized steer
            or intent units; defaults per mode.
        residual_penalty: Reward cost per mean squared action.
        smoothness_penalty: Reward cost per mean squared action change.
        vehicle_spread: Relative spread of the surrogate's parameters per
            episode; 0 disables it (and it never applies to CARLA).
        expert: Reflex-driver configuration.
        action_filter: Time constant of a first-order low-pass filter on the
            action, s; 0 disables it. Human control is low-pass, and an
            unfiltered 50 Hz residual makes acceleration and jerk unhumanly
            large. The filter state is appended to the observation.
        style_window: Seconds of following per reported style window; each
            complete window is put in ``info["following_window"]`` so a
            human-likeness reward can be paid during the episode.
        takeover_population: Draw the reflex driver's takeover time per
            episode from a lognormal with the meta-analytic mean 2.72 s and
            standard deviation 1.45 s (Zhang et al., 2019); else keep the
            configured value. Residuals act only once the reflex driver is
            engaged, so a policy cannot take over faster than its driver.
    """

    def __init__(
        self,
        env: Any,
        sampler: ScenarioSampler | None = None,
        *,
        mode: str = "demand",
        fields: Sequence[str] | None = None,
        residual_scale: Sequence[float] | None = None,
        residual_penalty: float = 0.05,
        smoothness_penalty: float = 0.05,
        vehicle_spread: float = 0.1,
        expert: ReflexDriverConfig | None = None,
        takeover_population: bool = True,
        style_window: float = 5.0,
        action_filter: float = 0.3,
    ) -> None:
        super().__init__(env)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        default_fields, default_scale = MODE_FIELDS[mode]
        fields = tuple(fields or default_fields)
        residual_scale = tuple(residual_scale or default_scale)
        allowed = default_fields if mode == "demand" else tuple(INTENT_BOUNDS)
        if set(fields) - set(allowed) or not fields:
            raise ValueError(f"fields must be drawn from {allowed}")
        if len(residual_scale) != len(fields):
            raise ValueError("residual_scale needs one value per field")
        self.sampler = sampler
        self.mode = mode
        self.residual = mode != "direct"
        self.fields = fields
        self.scale: NDArray[np.float64] = np.asarray(residual_scale, dtype=float)
        self.residual_penalty = float(residual_penalty)
        self.smoothness_penalty = float(smoothness_penalty)
        self.vehicle_spread = float(vehicle_spread)
        self.expert = ReflexDriver(env.unwrapped.dt, expert)
        self._expert_config = self.expert.config
        self.takeover_population = bool(takeover_population)
        self.style_window = float(style_window)
        if not math.isfinite(action_filter) or action_filter < 0:
            raise ValueError("action_filter must be finite and non-negative")
        self.action_filter = float(action_filter)
        dt = env.unwrapped.dt
        self._alpha = 1.0 if action_filter == 0 else 1.0 - math.exp(-dt / action_filter)
        self._filtered = np.zeros(len(self.fields))
        self._window_start = 0
        vehicle = env.unwrapped.vehicle
        self._base_params = (
            vehicle.params if isinstance(vehicle, SurrogateVehicle) else None
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (len(self.fields),), np.float32)
        extra = len(self.fields) if self.residual else 0
        filtered = len(self.fields) if self.action_filter > 0 else 0
        size = env.observation_space.shape[0] + extra + filtered
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (size,), np.float32)
        self.observation_names = (
            tuple(env.unwrapped.observation_names)
            + tuple(f"expert_{name}" for name in self.fields[:extra])
            + tuple(f"filtered_{name}" for name in self.fields[:filtered])
        )
        self._expert_intent = DriverIntent()
        self._previous = np.zeros(len(self.fields))
        self._history: dict[str, list[float]] = {}

    def set_stage(self, stage: int) -> None:
        """Unlock curriculum stages up to ``stage`` (used by callbacks)."""
        if self.sampler is not None:
            self.sampler.stage = int(np.clip(stage, 0, len(self.sampler.stages) - 1))

    def expert_action(self) -> NDArray[np.float32]:
        """The reflex driver's current intent as a normalized action.

        For a residual policy the expert action is zero by construction.
        """
        if self.residual:
            return np.zeros(len(self.fields), dtype=np.float32)
        values = []
        for name in self.fields:
            low, high = INTENT_BOUNDS[name]
            value = getattr(self._expert_intent, name)
            values.append(2.0 * (value - low) / (high - low) - 1.0)
        return np.clip(np.asarray(values, dtype=np.float32), -1.0, 1.0)

    def _observe(self, observation: NDArray[np.float32]) -> NDArray[np.float32]:
        state: NDArray[np.float32] = (
            np.asarray(self._filtered, dtype=np.float32)
            if self.action_filter > 0
            else np.zeros(0, dtype=np.float32)
        )
        if not self.residual:
            return np.concatenate([observation, state])
        if self.mode == "demand":
            accel, steer = self.expert.demand
            values = {"acceleration": accel, "steer": steer}
            expert = [values[name] for name in self.fields]
        else:
            expert = [getattr(self._expert_intent, name) for name in self.fields]
        return np.concatenate(
            [observation, np.asarray(expert, dtype=np.float32), state]
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Draw a scenario and vehicle, reset the body and the reflex driver.

        ``options`` may hold a ``scenario`` to use, or a ``stage`` name to draw.
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        env = self.env.unwrapped
        options = dict(options or {})
        if "scenario" in options:
            env.set_scenario(options.pop("scenario"))
        elif self.sampler is not None:
            env.set_scenario(
                self.sampler.sample(self.np_random, options.pop("stage", None))
            )
        if self._base_params is not None:
            env.vehicle.params = perturb_vehicle(
                self._base_params, self.np_random, self.vehicle_spread
            )
        observation, info = self.env.reset(seed=seed, options=options or None)
        if self.takeover_population:
            # Lognormal with mean 2.72 s and SD 1.45 s (Zhang et al., 2019).
            sigma = math.sqrt(math.log(1.0 + (1.45 / 2.72) ** 2))
            mu = math.log(2.72) - 0.5 * sigma**2
            self.expert.config = replace(
                self._expert_config,
                takeover_time=float(self.np_random.lognormal(mu, sigma)),
            )
        self.expert.reset()
        if self.mode != "demand":
            self._expert_intent = self.expert.act(env.state)
        self._previous = np.zeros(len(self.fields))
        self._filtered = np.zeros(len(self.fields))
        self._history = {
            key: []
            for key in ("time", "s", "lead_s", "lane", "speed_error", "gap", "ttc")
        }
        # The start state opens the trial, as a human trial opens with it.
        self._record(env.monitor.state, env)
        self._window_start = 0
        info = {
            **info,
            "stage": env.scenario.name,
            "expert_action": self.expert_action(),
            "takeover_time_s": self.expert.config.takeover_time,
        }
        return self._observe(observation), info

    def intent_for(self, action: Any) -> DriverIntent:
        """Return the intent that a normalized action produces now.

        In ``demand`` mode this advances the reflex driver once.
        """
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if action.shape != (len(self.fields),):
            raise ValueError(f"action must have {len(self.fields)} values")
        if self.mode == "demand":
            offsets = dict(zip(self.fields, self.scale * action))
            return self.expert.act(
                self.env.unwrapped.state,
                acceleration_offset=float(offsets.get("acceleration", 0.0)),
                steer_offset=float(offsets.get("steer", 0.0)),
            )
        values = {
            name: (getattr(self._expert_intent, name) if self.residual else 0.0)
            for name in INTENT_BOUNDS
        }
        for index, name in enumerate(self.fields):
            low, high = INTENT_BOUNDS[name]
            if self.residual:
                value = values[name] + self.scale[index] * action[index]
            else:
                value = low + 0.5 * (action[index] + 1.0) * (high - low)
            values[name] = float(np.clip(value, low, high))
        return DriverIntent(**values)

    def step(
        self, action: Any
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Apply the action, update the reflex driver and report episodes."""
        action = np.clip(np.asarray(action, dtype=float).reshape(-1), -1.0, 1.0)
        if self.residual and not self.expert.engaged:
            action = np.zeros_like(action)
        self._filtered = self._filtered + self._alpha * (action - self._filtered)
        intent = self.intent_for(self._filtered)
        observation, reward, terminated, truncated, info = self.env.step(intent)
        penalty = (
            self.residual_penalty * float(np.mean(self._filtered**2)) * self.residual
        )
        smooth = self.smoothness_penalty * float(
            np.mean((action - self._previous) ** 2)
        )
        self._previous = action
        reward = float(reward) - penalty - smooth
        info["reward_terms"] = {
            **info["reward_terms"],
            "residual": -penalty,
            "smoothness": -smooth,
        }
        env = self.env.unwrapped
        self._record(info["scenario"], env)
        if self.mode != "demand":
            self._expert_intent = self.expert.act(env.state)
        info["expert_action"] = self.expert_action()
        info["intent"] = vars(intent)
        window = self._following_window(env)
        if window is not None:
            info["following_window"] = window
        if terminated or truncated:
            info.update(self._episode_report(info))
        return self._observe(observation), reward, terminated, truncated, info

    def _record(self, state: dict[str, float], env: Any) -> None:
        history = self._history
        history["time"].append(state["time"])
        history["s"].append(state["s_m"])
        history["lead_s"].append(state.get("lead_s_m", math.nan))
        driving = not state["automation"]
        history["lane"].append(state["lane_offset_m"] if driving else math.nan)
        target = max(state["target_speed_mps"], 1.0)
        history["speed_error"].append(abs(state["speed_mps"] - target) / target)
        history["gap"].append(state["gap_m"])
        history["ttc"].append(state["time_to_collision_s"])

    def _following_window(self, env: Any) -> FollowingTrial | None:
        """Return the last ``style_window`` seconds of following once complete."""
        history = self._history
        if env.scenario.lead is None or self.style_window <= 0:
            return None
        start = self._window_start
        if history["time"][-1] - history["time"][start] < self.style_window - 1e-9:
            return None
        self._window_start = len(history["time"]) - 1
        window = slice(start, len(history["time"]))
        lead: NDArray[np.float64] = np.asarray(history["lead_s"][window], dtype=float)
        if not np.all(np.isfinite(lead)):
            return None
        return resample_following(
            history["time"][window],
            history["s"][window],
            lead.tolist(),
            VEHICLE_LENGTH,
            driver=env.scenario.name,
        )

    def _episode_report(self, info: dict[str, Any]) -> dict[str, Any]:
        env = self.env.unwrapped
        history = self._history
        lanes: NDArray[np.float64] = np.asarray(history["lane"], dtype=float)
        lane = lanes[np.isfinite(lanes)]
        gaps: NDArray[np.float64] = np.asarray(history["gap"], dtype=float)
        events = env.monitor.events
        takeover = [event for event in events if event["type"] == "driver_takeover"]
        record = {
            "stage": env.scenario.name,
            "collision": bool(any(event["type"] == "collision" for event in events)),
            "off_road": bool(info["off_road"]),
            "lane_rms_m": float(np.sqrt(np.mean(lane**2))) if lane.size else 0.0,
            "final_lane_offset_m": float(lane[-1]) if lane.size else 0.0,
            "final_speed_error": float(history["speed_error"][-1]),
            "min_gap_m": float(np.min(gaps)) if np.isfinite(gaps).any() else math.inf,
            "min_ttc_s": float(np.min(history["ttc"])),
            "takeover_time_s": takeover[0]["reaction_time"] if takeover else None,
            "seconds": float(history["time"][-1]),
        }
        report: dict[str, Any] = {
            "episode_metrics": record,
            "episode_success": _success(record["stage"], record),
        }
        lead_s: NDArray[np.float64] = np.asarray(history["lead_s"], dtype=float)
        if (
            env.scenario.lead is not None
            and np.all(np.isfinite(lead_s))
            and len(lead_s) > 30
        ):
            report["following_trial"] = resample_following(
                history["time"],
                history["s"],
                lead_s.tolist(),
                VEHICLE_LENGTH,
                driver=env.scenario.name,
            )
        return report

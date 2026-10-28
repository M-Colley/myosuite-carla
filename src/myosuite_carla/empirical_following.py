"""Empirical following component fitting and closed loop body evaluation.

The component protocol predicts acceleration from recorded vehicle states.
Closed loop evaluation separately replays the recorded leader and lets the
attentive human, muscles and vehicle surrogate generate the follower motion.
NGSIM identifiers denote source vehicle tracks, not identified participants.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .driver_models import IdmConfig, idm_acceleration
from .ngsim import RATE, FollowingTrial, following_trials, load_tracks, load_trials

METRICS = ("acceleration_rmse", "acceleration_bias")
FIXED_ERROR_SCALES = {"acceleration_rmse": 1.0, "acceleration_bias": 0.5}


def idm_parameters(config: IdmConfig = IdmConfig()) -> dict[str, float]:
    """Return a complete IDM parameter dictionary, including the exponent."""
    return {name: float(value) for name, value in asdict(config).items()}


def idm_config(parameters: Mapping[str, float]) -> IdmConfig:
    """Apply explicit IDM overrides and validate their physical domains."""
    values = idm_parameters()
    if set(parameters) - set(values):
        raise ValueError("parameters contain unknown IDM fields")
    values.update({name: float(value) for name, value in parameters.items()})
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("IDM parameters must be finite")
    if values["time_headway"] < 0 or values["min_gap"] < 0:
        raise ValueError("IDM time headway and minimum gap must be nonnegative")
    if any(
        values[name] <= 0
        for name in ("max_acceleration", "comfortable_deceleration", "exponent")
    ):
        raise ValueError("IDM acceleration, deceleration and exponent must be positive")
    return IdmConfig(**values)


def _validate_trial(trial: FollowingTrial) -> None:
    arrays = [
        np.asarray(getattr(trial, name), dtype=float)
        for name in ("time", "position", "speed", "acceleration", "gap", "leader_speed")
    ]
    if (
        not trial.group_id.strip()
        or len(arrays[0]) < 5
        or any(array.ndim != 1 or array.shape != arrays[0].shape for array in arrays)
        or any(not np.isfinite(array).all() for array in arrays)
        or np.any(np.diff(arrays[0]) <= 0)
        or abs(float(arrays[0][0])) > 1e-8
        or np.any(arrays[2] < 0)
        or np.any(arrays[4] <= 0)
        or np.any(arrays[5] < 0)
    ):
        raise ValueError(
            "following trials need finite aligned states and positive gaps"
        )


class FollowingProtocol:
    """Deterministic IDM acceleration errors averaged equally across trials.

    Observed targets are zero errors relative to measured acceleration. They
    are not a claim that human acceleration is zero. Each original source
    contributes one summary, irrespective of its number of windows.
    """

    def __init__(
        self,
        trials: Sequence[FollowingTrial],
        *,
        desired_speed: float = 16.0,
        max_groups: int | None = 96,
        seed: int = 0,
        dataset_id: str = "ngsim",
    ) -> None:
        if not math.isfinite(desired_speed) or desired_speed <= 0:
            raise ValueError("desired_speed must be finite and positive")
        if not dataset_id.strip():
            raise ValueError("dataset_id must be nonempty")
        if max_groups is not None and (
            isinstance(max_groups, bool)
            or not isinstance(max_groups, int)
            or max_groups < 1
        ):
            raise ValueError("max_groups must be a positive integer or None")
        if not trials:
            raise ValueError("at least one following trial is required")
        population: dict[str, list[FollowingTrial]] = {}
        for trial in trials:
            _validate_trial(trial)
            population.setdefault(f"{dataset_id}:{trial.group_id}", []).append(trial)
        self.population_groups = len(population)
        self.population_trials = len(trials)
        ordered = sorted(population)
        if max_groups is not None and len(ordered) > max_groups:
            ordered = sorted(
                np.random.default_rng(seed).choice(ordered, max_groups, replace=False)
            )
        self.groups = tuple(str(group) for group in ordered)
        self.desired_speed = float(desired_speed)
        self.seed = seed
        self.dataset_id = dataset_id
        self._trials: dict[str, tuple[FollowingTrial, ...]] = {}
        for group in self.groups:
            frozen = []
            for trial in sorted(
                population[group],
                key=lambda value: (
                    value.driver,
                    float(value.position[0]),
                    value.duration,
                ),
            ):
                values: dict[str, NDArray[np.float64]] = {}
                for name in (
                    "time",
                    "position",
                    "speed",
                    "acceleration",
                    "gap",
                    "leader_speed",
                ):
                    array = np.array(getattr(trial, name), dtype=float, copy=True)
                    array.flags.writeable = False
                    values[name] = array
                frozen.append(replace(trial, **values))
            self._trials[group] = tuple(frozen)

    @property
    def observed(self) -> dict[str, dict[str, float]]:
        """Return zero acceleration error targets for the selected sources."""
        return {group: dict.fromkeys(METRICS, 0.0) for group in self.groups}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return selection, units and the interpretation of this protocol."""
        return {
            "dataset_id": self.dataset_id,
            "role": "recorded_state_acceleration_component",
            "group_kind": "vehicle_track",
            "population_source_groups": self.population_groups,
            "population_trials": self.population_trials,
            "selected_source_groups": len(self.groups),
            "selected_trials": sum(len(trials) for trials in self._trials.values()),
            "source_groups": list(self.groups),
            "selection_seed": self.seed,
            "desired_speed_mps": self.desired_speed,
            "metrics": dict.fromkeys(METRICS, "m/s^2"),
            "observed_target_definition": "zero error to recorded acceleration",
            "trial_weighting": "equal trials within source; equal source weights",
            "stochastic": False,
            "state_conditioning": [
                "recorded speed",
                "recorded gap",
                "recorded closing speed",
            ],
        }

    def trials_for(self, group: str) -> tuple[FollowingTrial, ...]:
        """Return the immutable selected windows of one namespaced source."""
        if group not in self._trials:
            raise ValueError(f"unknown selected source group {group!r}")
        return self._trials[group]

    def __call__(
        self, parameters: Mapping[str, float], groups: tuple[str, ...], seed: int
    ) -> dict[str, dict[str, float]]:
        """Predict teacher forced IDM acceleration errors for requested sources."""
        config = idm_config(parameters)
        if len(set(groups)) != len(groups):
            raise ValueError("requested source groups must be unique")
        summaries = {}
        for group in groups:
            errors = []
            for trial in self.trials_for(group):
                predicted: NDArray[np.float64] = np.fromiter(
                    (
                        idm_acceleration(
                            speed, self.desired_speed, gap, speed - leader, config
                        )
                        for speed, gap, leader in zip(
                            trial.speed, trial.gap, trial.leader_speed
                        )
                    ),
                    dtype=float,
                    count=len(trial.time),
                )
                residual = predicted - trial.acceleration
                if not np.isfinite(residual).all():
                    raise ValueError("IDM produced nonfinite acceleration residuals")
                errors.append(
                    [float(np.sqrt(np.mean(residual**2))), float(np.mean(residual))]
                )
            means = np.mean(errors, axis=0)
            summaries[group] = dict(zip(METRICS, map(float, means)))
        return summaries


def load_following_protocol(path: str | Path, **settings: Any) -> FollowingProtocol:
    """Load a saved trial NPZ or the existing public NGSIM CSV format."""
    source = Path(path)
    trials = (
        load_trials(source)
        if source.suffix.lower() == ".npz"
        else following_trials(load_tracks(source), max_speed=16.0)
    )
    return FollowingProtocol(trials, **settings)


def closed_loop_following(
    protocol: FollowingProtocol,
    baseline_parameters: Mapping[str, float],
    fitted_parameters: Mapping[str, float],
    *,
    groups: Sequence[str],
    max_trials: int = 6,
    seed: int = 0,
    seconds: float | None = None,
) -> dict[str, Any]:
    """Replay leaders for an explicitly supplied held out source subset.

    One fast body is reused across both profiles and all sequential episodes.
    Each follower starts at the recorded speed and gap. Subsequent follower
    states come exclusively from the human model, muscles and fitted surrogate.
    """
    if (
        not groups
        or len(set(groups)) != len(groups)
        or set(groups) - set(protocol.groups)
    ):
        raise ValueError("groups must be a unique nonempty selected held out subset")
    if (
        isinstance(max_trials, bool)
        or not isinstance(max_trials, int)
        or max_trials < 1
    ):
        raise ValueError("max_trials must be a positive integer")
    if seconds is not None and (not math.isfinite(seconds) or seconds < 0.4):
        raise ValueError("seconds must be finite and at least 0.4")
    baseline, fitted = idm_config(baseline_parameters), idm_config(fitted_parameters)
    order = np.random.default_rng(seed).permutation(sorted(groups)).tolist()
    chosen: list[tuple[str, int, FollowingTrial]] = []
    for index in range(max(len(protocol.trials_for(group)) for group in order)):
        for group in order:
            trials = protocol.trials_for(group)
            if index < len(trials) and len(chosen) < max_trials:
                chosen.append((group, index, trials[index]))
    report: dict[str, Any] = {
        "status": "unavailable",
        "role": "held_out_closed_loop",
        "group_kind": "vehicle_track",
        "requested_source_groups": sorted(groups),
        "selected_source_groups": sorted({group for group, _, _ in chosen}),
        "selected_trials": len(chosen),
        "secondary_task": None,
        "controller": "embodied_human.agents.driver.AttentiveDriver",
        "vehicle": "offline fitted CARLA 0.10.0 Nissan Patrol surrogate",
        "desired_speed_mps": protocol.desired_speed,
        "profiles": {"baseline": asdict(baseline), "fitted": asdict(fitted)},
        "episodes": [],
    }
    try:
        import embodied_human as eh
        from embodied_human.agents.driver import AttentiveDriver

        from .driver_models import ReflexDriver, ReflexDriverConfig
        from .driving_env import FullBodyDrivingEnv
        from .fullbody import FullBodyDriverConfig, build_fullbody_driver
        from .ngsim import resample_following
        from .training import scenario_from_trial
        from .vehicle_model import carla_nissan_patrol

        driver_config = FullBodyDriverConfig.fast()
        driver = build_fullbody_driver(driver_config)
    except ImportError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    report.update(
        status="completed",
        human_import=eh.__file__,
        human_version=eh.__version__,
        body_builds=1,
        muscles=driver.model.nu,
        driver_config=asdict(driver_config),
        vehicle_parameters=carla_nissan_patrol().as_dict(),
    )
    for profile, config in (("baseline", baseline), ("fitted", fitted)):
        for episode, (group, trial_index, trial) in enumerate(chosen):
            record: dict[str, Any] = {
                "profile": profile,
                "source_group": group,
                "original_source_group": trial.group_id,
                "trial_index": trial_index,
                "seed": seed + episode,
                "status": "failed",
                "original_duration_s": trial.duration,
            }
            scenario = scenario_from_trial(trial, protocol.desired_speed)
            if seconds is not None:
                scenario = replace(scenario, duration=min(seconds, scenario.duration))
            record["requested_duration_s"] = scenario.duration
            env = FullBodyDrivingEnv(scenario, driver=driver)
            try:
                _, initial = env.reset(seed=seed + episode)
                reflex_config = ReflexDriverConfig(idm=config, perception_delay=env.dt)
                agent = AttentiveDriver(
                    eh.build_human(seed=seed + episode),
                    env,
                    reflex=ReflexDriver(env.dt, reflex_config),
                )
                agent.reset(seed=seed + episode)
                states = [initial["scenario"]]
                terminated = truncated = False
                for _ in range(math.ceil(scenario.duration / env.dt)):
                    _, _, terminated, truncated, info = env.step(agent.act())
                    if not all(
                        np.isfinite(values).all()
                        for values in (
                            driver.data.qpos,
                            driver.data.qvel,
                            driver.data.act,
                        )
                    ):
                        raise RuntimeError("the MuJoCo body contains nonfinite state")
                    states.append(info["scenario"])
                    if terminated or truncated:
                        break
                times = np.asarray([state["time"] for state in states])
                record.update(
                    seconds=float(times[-1]),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    collision=any(bool(state["collision"]) for state in states),
                    off_road=any(bool(state["off_road"]) for state in states),
                    events=list(env.monitor.events),
                    vehicle_footprints=env.monitor.footprints,
                )
                horizon = (
                    math.floor(min(times[-1], scenario.duration) * RATE + 1e-8) / RATE
                )
                keep = times <= horizon + 1e-8
                predicted = resample_following(
                    times[keep],
                    np.asarray([state["ego_front_s_m"] for state in states])[keep],
                    np.asarray([state["lead_front_s_m"] for state in states])[keep],
                    states[0]["lead_length_m"],
                    driver=f"simulation:{profile}:{group}:{trial_index}",
                    source_group=trial.group_id,
                )
                observed_speed = np.interp(predicted.time, trial.time, trial.speed)
                observed_gap = np.interp(predicted.time, trial.time, trial.gap)
                observed_accel = np.interp(
                    predicted.time, trial.time, trial.acceleration
                )
                record.update(
                    status="completed"
                    if truncated and not terminated
                    else "terminated",
                    compared_seconds=float(predicted.time[-1]),
                    coverage_fraction=float(predicted.time[-1] / scenario.duration),
                    speed_rmse_mps=float(
                        np.sqrt(np.mean((predicted.speed - observed_speed) ** 2))
                    ),
                    gap_rmse_m=float(
                        np.sqrt(np.mean((predicted.gap - observed_gap) ** 2))
                    ),
                    acceleration_rmse_mps2=float(
                        np.sqrt(np.mean((predicted.acceleration - observed_accel) ** 2))
                    ),
                    time_s=predicted.time.tolist(),
                    predicted_speed_mps=predicted.speed.tolist(),
                    observed_speed_mps=observed_speed.tolist(),
                    predicted_gap_m=predicted.gap.tolist(),
                    observed_gap_m=observed_gap.tolist(),
                )
            except Exception as error:
                record["status"] = "failed"
                record["error"] = f"{type(error).__name__}: {error}"
            finally:
                env.close()
                env.vehicle.close()
                record["surrogate_closed"] = not env.vehicle.active
            if record["status"] != "completed":
                report["status"] = "failed"
            report["episodes"].append(record)
    return report

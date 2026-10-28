"""Empirical following component fitting and closed loop body evaluation.

The component protocol predicts acceleration from recorded vehicle states.
Closed loop evaluation separately replays the recorded leader and lets the
attentive human, muscles and vehicle surrogate generate the follower motion.
NGSIM identifiers denote source vehicle tracks, not identified participants.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .driver_models import IdmConfig, idm_acceleration
from .ngsim import RATE, FollowingTrial, following_trials, load_tracks, load_trials

METRICS = ("acceleration_rmse", "acceleration_bias")
FIXED_ERROR_SCALES = {"acceleration_rmse": 1.0, "acceleration_bias": 0.5}
CLOSED_LOOP_METRICS = (
    "speed_rmse_mps",
    "gap_rmse_m",
    "acceleration_rmse_mps2",
)
CLOSED_LOOP_SCALES = dict(zip(CLOSED_LOOP_METRICS, (1.0, 5.0, 1.0)))


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
        excluded_groups: Sequence[str] = (),
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
        if isinstance(excluded_groups, (str, bytes)) or any(
            not isinstance(group, str) or not group.strip() for group in excluded_groups
        ):
            raise ValueError("excluded_groups must contain nonempty source identities")
        excluded = tuple(sorted(set(excluded_groups)))
        self.excluded_groups = excluded
        self.excluded_population_groups = len(set(excluded) & population.keys())
        ordered = sorted(set(population) - set(excluded))
        if not ordered:
            raise ValueError("source exclusions leave no eligible following trials")
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
            "excluded_source_groups": list(self.excluded_groups),
            "excluded_population_groups": self.excluded_population_groups,
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


@dataclass(frozen=True)
class ClosedLoopConfig:
    """Physical objective settings shared across candidates and partitions.

    A complete recorded window preserves the source derivative filter at both
    endpoints. Shorter horizons are useful for software checks and are labelled
    explicitly in the episode reports.
    """

    seconds: float | None = 20.0
    trials_per_source: int = 1
    keep_trajectories: bool = True

    def __post_init__(self) -> None:
        if self.seconds is not None and (
            isinstance(self.seconds, bool)
            or not math.isfinite(self.seconds)
            or self.seconds < 0.4
        ):
            raise ValueError("seconds must be finite and at least 0.4")
        if (
            isinstance(self.trials_per_source, bool)
            or not isinstance(self.trials_per_source, int)
            or self.trials_per_source < 1
        ):
            raise ValueError("trials_per_source must be a positive integer")
        if not isinstance(self.keep_trajectories, bool):
            raise ValueError("keep_trajectories must be boolean")


def _episode_complete(record: Mapping[str, Any]) -> bool:
    values = [record.get(name) for name in (*CLOSED_LOOP_METRICS, "coverage_fraction")]
    return (
        record.get("status") == "completed"
        and not record.get("collision", True)
        and not record.get("off_road", True)
        and record.get("surrogate_closed") is True
        and all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
            for value in values
        )
        and abs(float(record["coverage_fraction"]) - 1.0) <= 1e-8
    )


def _source_seed(seed: int, group: str, trial_index: int) -> int:
    """Keep simulation seeds stable when source order or batching changes."""
    identity = f"{seed}\0{group}\0{trial_index}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")


def _run_closed_loop_episode(
    driver: Any,
    trial: FollowingTrial,
    parameters: IdmConfig,
    *,
    desired_speed: float,
    group: str,
    trial_index: int,
    seed: int,
    config: ClosedLoopConfig,
) -> dict[str, Any]:
    """Run one muscle driven follower and retain completion and error evidence."""
    import embodied_human as eh
    from embodied_human.agents.driver import AttentiveDriver

    from .driver_models import ReflexDriver, ReflexDriverConfig
    from .driving_env import FullBodyDrivingEnv
    from .ngsim import resample_following
    from .training import scenario_from_trial

    record: dict[str, Any] = {
        "source_group": group,
        "original_source_group": trial.group_id,
        "trial_index": trial_index,
        "trial_driver": trial.driver,
        "seed": seed,
        "parameters": asdict(parameters),
        "status": "failed",
        "failure_kind": "computational_error",
        "original_duration_s": trial.duration,
    }
    env: Any = None
    try:
        scenario = scenario_from_trial(trial, desired_speed)
        if config.seconds is not None:
            scenario = replace(
                scenario, duration=min(config.seconds, scenario.duration)
            )
        record["requested_duration_s"] = scenario.duration
        record["full_source_horizon"] = math.isclose(
            scenario.duration, trial.duration, rel_tol=0, abs_tol=1e-8
        )
        env = FullBodyDrivingEnv(scenario, driver=driver)
        _, initial = env.reset(seed=seed)
        reflex_config = ReflexDriverConfig(idm=parameters, perception_delay=env.dt)
        agent = AttentiveDriver(
            eh.build_human(seed=seed),
            env,
            reflex=ReflexDriver(env.dt, reflex_config),
        )
        agent.reset(seed=seed)
        states = [initial["scenario"]]
        terminated = truncated = False
        for _ in range(math.ceil(scenario.duration / env.dt)):
            _, _, terminated, truncated, info = env.step(agent.act())
            if not all(
                np.isfinite(values).all()
                for values in (driver.data.qpos, driver.data.qvel, driver.data.act)
            ):
                raise RuntimeError("the MuJoCo body contains nonfinite state")
            states.append(info["scenario"])
            if terminated or truncated:
                break
        times = np.asarray([state["time"] for state in states])
        collision = any(bool(state["collision"]) for state in states)
        off_road = any(bool(state["off_road"]) for state in states)
        record.update(
            seconds=float(times[-1]),
            terminated=bool(terminated),
            truncated=bool(truncated),
            collision=collision,
            off_road=off_road,
            events=list(env.monitor.events),
            vehicle_footprints=env.monitor.footprints,
        )
        horizon = math.floor(min(times[-1], scenario.duration) * RATE + 1e-8) / RATE
        keep = times <= horizon + 1e-8
        predicted = resample_following(
            times[keep],
            np.asarray([state["ego_front_s_m"] for state in states])[keep],
            np.asarray([state["lead_front_s_m"] for state in states])[keep],
            states[0]["lead_length_m"],
            driver=f"simulation:{group}:{trial_index}",
            source_group=trial.group_id,
        )
        observed_speed = np.interp(predicted.time, trial.time, trial.speed)
        observed_gap = np.interp(predicted.time, trial.time, trial.gap)
        observed_accel = np.interp(predicted.time, trial.time, trial.acceleration)
        record.update(
            status="completed" if truncated and not terminated else "terminated",
            compared_seconds=float(predicted.time[-1]),
            coverage_fraction=float(predicted.time[-1] / scenario.duration),
            speed_rmse_mps=float(
                np.sqrt(np.mean((predicted.speed - observed_speed) ** 2))
            ),
            gap_rmse_m=float(np.sqrt(np.mean((predicted.gap - observed_gap) ** 2))),
            acceleration_rmse_mps2=float(
                np.sqrt(np.mean((predicted.acceleration - observed_accel) ** 2))
            ),
            failure_kind=(
                "collision"
                if collision
                else "road_departure"
                if off_road
                else "terminated"
                if terminated
                else "incomplete_horizon"
                if predicted.time[-1] < scenario.duration - 1e-8
                else None
            ),
        )
        if config.keep_trajectories:
            record.update(
                time_s=predicted.time.tolist(),
                predicted_speed_mps=predicted.speed.tolist(),
                observed_speed_mps=observed_speed.tolist(),
                predicted_gap_m=predicted.gap.tolist(),
                observed_gap_m=observed_gap.tolist(),
                predicted_acceleration_mps2=predicted.acceleration.tolist(),
                observed_acceleration_mps2=observed_accel.tolist(),
            )
    except Exception as error:
        safety_failure = (
            "collision"
            if record.get("collision")
            else "road_departure"
            if record.get("off_road")
            else None
        )
        record.update(
            status="terminated" if safety_failure else "failed",
            failure_kind=safety_failure or "computational_error",
        )
        record["error"] = f"{type(error).__name__}: {error}"
    finally:
        if env is not None:
            try:
                env.close()
                env.vehicle.close()
                record["surrogate_closed"] = not env.vehicle.active
            except Exception as error:
                record.update(status="failed", failure_kind="cleanup_error")
                record["cleanup_error"] = f"{type(error).__name__}: {error}"
        else:
            record["surrogate_closed"] = True
    return record


class ClosedLoopFollowingProtocol:
    """Calibrate the physical human and vehicle loop with equal source weights.

    Every requested source contributes its fixed trial set. An incomplete or
    unsafe episode makes the objective infeasible. Episode caches bind the
    complete IDM settings, source trial, and exact simulation seed.
    """

    def __init__(
        self,
        protocol: FollowingProtocol,
        *,
        config: ClosedLoopConfig = ClosedLoopConfig(),
    ) -> None:
        self.config = config
        self.groups = tuple(protocol.groups)
        self.desired_speed = protocol.desired_speed
        self._source_metadata = deepcopy(protocol.metadata)
        self._trials = {
            group: protocol.trials_for(group)[: config.trials_per_source]
            for group in self.groups
        }
        self._driver: Any = None
        self._body_metadata: dict[str, Any] = {}
        self._cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._cache_hits = 0
        self._closed = False

    def __enter__(self) -> ClosedLoopFollowingProtocol:
        self._ensure_driver()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Release the owned MuJoCo body after all episode vehicles are closed."""
        self._driver = None
        self._closed = True

    def _ensure_driver(self) -> None:
        if self._closed:
            raise RuntimeError("closed loop protocol has been closed")
        if self._driver is not None:
            return
        import embodied_human as eh

        from .fullbody import FullBodyDriverConfig, build_fullbody_driver
        from .vehicle_model import carla_nissan_patrol

        driver_config = FullBodyDriverConfig.fast()
        self._driver = build_fullbody_driver(driver_config)
        self._body_metadata = {
            "human_import": eh.__file__,
            "human_version": eh.__version__,
            "body_builds": 1,
            "muscles": self._driver.model.nu,
            "driver_config": asdict(driver_config),
            "vehicle_parameters": carla_nissan_patrol().as_dict(),
        }

    @property
    def observed(self) -> dict[str, dict[str, float]]:
        """Return zero trajectory error targets for every selected source."""
        return {group: dict.fromkeys(CLOSED_LOOP_METRICS, 0.0) for group in self.groups}

    @property
    def metadata(self) -> dict[str, Any]:
        """Export objective settings, source coverage, and physical provenance."""
        return {
            **deepcopy(self._source_metadata),
            "role": "physical_closed_loop_calibration",
            "configuration": asdict(self.config),
            "metrics": dict(zip(CLOSED_LOOP_METRICS, ("m/s", "m", "m/s^2"))),
            "fixed_error_scales": dict(CLOSED_LOOP_SCALES),
            "observed_target_definition": "zero trajectory error to recorded follower",
            "trial_weighting": "equal trials within source; equal source weights",
            "trial_selection": "first fixed source windows after deterministic sorting",
            "selected_trials": sum(map(len, self._trials.values())),
            "state_conditioning": ["recorded initial speed", "recorded initial gap"],
            "stochastic": True,
            "leader": "recorded trajectory replay",
            "failure_policy": "any incomplete or unsafe episode is infeasible",
            "seed_policy": "SHA256 of base seed, source identity, and trial index",
            **deepcopy(self._body_metadata),
        }

    @property
    def episodes(self) -> list[dict[str, Any]]:
        """Export each physically executed episode once with its parameters."""
        return deepcopy(list(self._cache.values()))

    @property
    def diagnostics(self) -> dict[str, Any]:
        """Return execution counts and retained failure identities."""
        failures = [
            {
                key: record.get(key)
                for key in (
                    "source_group",
                    "trial_index",
                    "seed",
                    "parameters",
                    "status",
                    "failure_kind",
                    "error",
                    "coverage_fraction",
                )
            }
            for record in self._cache.values()
            if not _episode_complete(record)
        ]
        return {
            "body_builds": self._body_metadata.get("body_builds", 0),
            "episode_runs": len(self._cache),
            "cache_hits": self._cache_hits,
            "failed_episodes": failures,
            "closed": self._closed,
        }

    def _requested_groups(self, groups: Sequence[str]) -> tuple[str, ...]:
        if (
            not groups
            or len(set(groups)) != len(groups)
            or set(groups) - set(self.groups)
        ):
            raise ValueError("groups must be a unique nonempty selected source subset")
        return tuple(sorted(groups))

    def _episode_for(
        self, parameters: IdmConfig, group: str, trial_index: int, seed: int
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("closed loop protocol has been closed")
        key = (tuple(asdict(parameters).values()), group, trial_index, seed)
        if key in self._cache:
            self._cache_hits += 1
            return deepcopy(self._cache[key])
        self._ensure_driver()
        record = _run_closed_loop_episode(
            self._driver,
            self._trials[group][trial_index],
            parameters,
            desired_speed=self.desired_speed,
            group=group,
            trial_index=trial_index,
            seed=seed,
            config=self.config,
        )
        self._cache[key] = deepcopy(record)
        return record

    def __call__(
        self, parameters: Mapping[str, float], groups: tuple[str, ...], seed: int
    ) -> dict[str, dict[str, float]]:
        """Return complete source trajectory errors or an infeasible objective."""
        requested = self._requested_groups(groups)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        config = idm_config(parameters)
        records: dict[str, list[dict[str, Any]]] = {}
        for group in requested:
            records[group] = [
                self._episode_for(
                    config, group, index, _source_seed(seed, group, index)
                )
                for index in range(len(self._trials[group]))
            ]
        failures = [
            record
            for rows in records.values()
            for record in rows
            if not _episode_complete(record)
        ]
        if failures:
            identities = [
                (row["source_group"], row["trial_index"], row.get("failure_kind"))
                for row in failures
            ]
            raise RuntimeError(
                f"closed loop objective has infeasible episodes: {identities}"
            )
        return {
            group: {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in CLOSED_LOOP_METRICS
            }
            for group, rows in records.items()
        }

    def report_pair(
        self,
        baseline: Mapping[str, float],
        fitted: Mapping[str, float],
        *,
        groups: Sequence[str],
        seed: int = 0,
    ) -> dict[str, Any]:
        """Export complete matched baseline and candidate physical episodes."""
        requested = self._requested_groups(groups)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        profiles = {"baseline": idm_config(baseline), "fitted": idm_config(fitted)}
        episodes = []
        for label, parameters in profiles.items():
            for group in requested:
                for index in range(len(self._trials[group])):
                    record = self._episode_for(
                        parameters, group, index, _source_seed(seed, group, index)
                    )
                    episodes.append({**record, "profile": label, "base_seed": seed})
        return {
            "status": "completed"
            if all(map(_episode_complete, episodes))
            else "failed",
            "role": "paired_physical_closed_loop",
            "group_kind": "vehicle_track",
            "requested_source_groups": list(requested),
            "selected_source_groups": list(requested),
            "selected_trials": sum(len(self._trials[group]) for group in requested),
            "secondary_task": None,
            "controller": "embodied_human.agents.driver.AttentiveDriver",
            "vehicle": "offline fitted CARLA 0.10.0 Nissan Patrol surrogate",
            "desired_speed_mps": self.desired_speed,
            "profiles": {key: asdict(value) for key, value in profiles.items()},
            "configuration": asdict(self.config),
            "episodes": episodes,
            **deepcopy(self._body_metadata),
        }


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
    """Replay matched profiles for an explicitly supplied held out subset.

    This compatibility entry point preserves the original total trial limit
    and episode seed offsets. Calibration uses ClosedLoopFollowingProtocol.
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
    config = ClosedLoopConfig(seconds=seconds, trials_per_source=max_trials)
    profiles = {
        "baseline": idm_config(baseline_parameters),
        "fitted": idm_config(fitted_parameters),
    }
    order = np.random.default_rng(seed).permutation(sorted(groups)).tolist()
    chosen: list[tuple[str, int]] = []
    for index in range(max(len(protocol.trials_for(group)) for group in order)):
        for group in order:
            if index < len(protocol.trials_for(group)) and len(chosen) < max_trials:
                chosen.append((group, index))
    report: dict[str, Any] = {
        "status": "unavailable",
        "role": "held_out_closed_loop",
        "group_kind": "vehicle_track",
        "requested_source_groups": sorted(groups),
        "selected_source_groups": sorted({group for group, _ in chosen}),
        "selected_trials": len(chosen),
        "secondary_task": None,
        "controller": "embodied_human.agents.driver.AttentiveDriver",
        "vehicle": "offline fitted CARLA 0.10.0 Nissan Patrol surrogate",
        "desired_speed_mps": protocol.desired_speed,
        "profiles": {name: asdict(value) for name, value in profiles.items()},
        "episodes": [],
    }
    session = ClosedLoopFollowingProtocol(protocol, config=config)
    try:
        session._ensure_driver()
    except ImportError as error:
        report["error"] = f"{type(error).__name__}: {error}"
        session.close()
        return report
    try:
        report.update(status="completed", **deepcopy(session._body_metadata))
        for label, parameters in profiles.items():
            for episode, (group, index) in enumerate(chosen):
                record = session._episode_for(parameters, group, index, seed + episode)
                report["episodes"].append({**record, "profile": label})
                if not _episode_complete(record):
                    report["status"] = "failed"
    finally:
        session.close()
    return report

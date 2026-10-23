"""Fail closed on corrupt feedback and ownership failures before startup."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest

from myosuite_carla import (
    SessionConfig,
    SynchronizationError,
    VehicleCommand,
    VehicleTelemetry,
)


@pytest.mark.parametrize("name", ["allow_existing_synchronous", "no_rendering_mode"])
@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_session_policy_flags_require_booleans(name: str, value: Any) -> None:
    with pytest.raises(TypeError, match=name):
        SessionConfig(**{name: value})


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_invalid_ownership_flag_never_takes_vehicle(
    simulation: Any, value: Any
) -> None:
    with pytest.raises(TypeError, match="own_vehicle"):
        simulation.session(own_vehicle=value)
    assert simulation.vehicle.destroy_count == 0
    assert simulation.world.events == []


def test_substep_count_must_fit_carla_integer_field() -> None:
    with pytest.raises(ValueError, match="32-bit"):
        SessionConfig(max_substeps=2**31)


@pytest.mark.parametrize(
    "name",
    [
        "fixed_delta_seconds",
        "timeout_seconds",
        "max_substep_delta_time",
    ],
)
def test_boolean_is_not_a_timing_value(name: str) -> None:
    with pytest.raises(TypeError, match=name):
        SessionConfig(**{name: True})


def corrupt_snapshot(simulation: Any, field: str, value: float) -> None:
    """Replace one snapshot component while preserving its real fixture shape."""
    snapshot = simulation.world.get_snapshot()
    actor = snapshot.find(simulation.vehicle.id)
    if field == "elapsed_seconds":
        snapshot.timestamp.elapsed_seconds = value
    elif field == "rotation":
        transform = actor.get_transform()
        transform.rotation.pitch = value
        actor.get_transform = lambda: transform
    elif field == "location":
        transform = actor.get_transform()
        transform.location.x = value
        actor.get_transform = lambda: transform
    else:
        setattr(actor, "get_" + field, lambda: SimpleNamespace(x=value, y=0, z=0))
    simulation.world.get_snapshot = lambda: snapshot


@pytest.mark.parametrize(
    "field",
    [
        "elapsed_seconds",
        "location",
        "rotation",
        "velocity",
        "acceleration",
        "angular_velocity",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_start_rejects_nonfinite_feedback_and_restores_settings(
    simulation: Any, field: str, value: float
) -> None:
    corrupt_snapshot(simulation, field, value)
    session = simulation.session(own_vehicle=True)
    with pytest.raises(SynchronizationError, match="nonfinite"):
        session.start()
    assert not session.active
    assert simulation.vehicle.destroy_count == 1
    assert not simulation.world.settings.synchronous_mode


@pytest.mark.parametrize("fault", ["external_tick", "missing", "nonfinite"])
def test_public_snapshot_failure_brakes_and_invalidates_session(
    simulation: Any, fault: str
) -> None:
    session = simulation.session().start()
    if fault == "external_tick":
        simulation.world.frame += 1
    elif fault == "missing":
        simulation.world.missing_actor = True
    else:
        corrupt_snapshot(simulation, "velocity", math.nan)
    with pytest.raises(SynchronizationError):
        session.snapshot()
    assert not session.active
    assert simulation.vehicle.controls[-1].throttle == 0
    assert simulation.vehicle.controls[-1].brake == 1
    assert not any(event[0] == "tick" for event in simulation.world.events)
    with pytest.raises(RuntimeError, match="healthy"):
        session.step(VehicleCommand(throttle=0.5))
    session.close()


def test_corrupt_preflight_never_submits_throttle_or_ticks(simulation: Any) -> None:
    session = simulation.session().start()
    corrupt_snapshot(simulation, "elapsed_seconds", math.nan)
    with pytest.raises(SynchronizationError, match="nonfinite"):
        session.step(VehicleCommand(throttle=0.8))
    assert all(command.throttle == 0 for command in simulation.vehicle.controls)
    assert simulation.vehicle.controls[-1].brake == 1
    assert simulation.world.frame == 100
    assert not session.active
    session.close()


@pytest.mark.parametrize("owned", [False, True])
def test_initial_world_rpc_failure_releases_only_owned_vehicle(
    simulation: Any, owned: bool
) -> None:
    def fail() -> None:
        raise TimeoutError("initial world RPC failed")

    simulation.client.get_world = fail
    with pytest.raises(TimeoutError, match="initial world"):
        simulation.session(own_vehicle=owned)
    assert simulation.vehicle.destroy_count == int(owned)
    assert simulation.world.events == []


def test_initial_world_rpc_error_survives_cleanup_error(simulation: Any) -> None:
    def fail() -> None:
        raise TimeoutError("initial world RPC failed")

    simulation.client.get_world = fail
    simulation.vehicle.destroy = lambda: False
    with pytest.warns(UserWarning, match="cleanup"):
        with pytest.raises(TimeoutError, match="initial world"):
            simulation.session(own_vehicle=True)


def test_telemetry_payload_is_independent_and_speed_avoids_overflow() -> None:
    telemetry = VehicleTelemetry(
        100,
        2.0,
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (1e200, 1e200, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 90.0),
    )
    payload = telemetry.as_dict()
    assert math.isfinite(payload["speed"])
    assert payload["speed"] == pytest.approx(math.hypot(1e200, 1e200))
    payload["location"] = (0.0, 0.0, 0.0)
    payload["frame"] = -1
    assert telemetry.location == (1.0, 2.0, 3.0)
    assert telemetry.as_dict()["frame"] == 100


def test_mutable_telemetry_vectors_cannot_alias_exported_feedback() -> None:
    location = [1.0, 2.0, 3.0]
    telemetry = VehicleTelemetry(
        100, 2.0, location, (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
    )
    payload = telemetry.as_dict()
    location[0] = 99.0
    assert telemetry.location == (1.0, 2.0, 3.0)
    assert payload["location"] == (1.0, 2.0, 3.0)


def test_incomplete_telemetry_vector_is_rejected() -> None:
    with pytest.raises(ValueError, match="location"):
        VehicleTelemetry(
            100, 2.0, (1.0, 2.0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)
        )

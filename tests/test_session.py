"""Session lifetime, API contract and clock failure tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from myosuite_carla import SessionConfig, SynchronizationError, VehicleCommand


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fixed_delta_seconds": 0},
        {"fixed_delta_seconds": float("nan")},
        {"timeout_seconds": -1},
        {"max_substep_delta_time": float("inf")},
        {"max_substeps": 0},
        {"max_substeps": True},
        {"max_substeps": 2.5},
        {"fixed_delta_seconds": 0.11},
    ],
)
def test_bad_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SessionConfig(**kwargs)


def test_control_before_tick_and_coherent_feedback(simulation: Any) -> None:
    original = simulation.world.get_settings()
    with simulation.session() as session:
        assert session.start() is session
        assert session.snapshot().frame == 100
        state = session.step(VehicleCommand(throttle=0.4, steer=-0.5, gear=2))
        assert state.frame == 101
        assert state.elapsed_seconds == pytest.approx(2.02)
        assert state.speed == 5
        assert state.rotation == (4, 5, 6)
        assert state.angular_velocity == (0, 0, 90)
        assert simulation.world.events[1:5] == [
            ("batch", 1, False),
            ("control", 0.4, 0),
            ("batch_complete", 1),
            ("tick", 10),
        ]
        assert simulation.vehicle.controls[-1].steer == -0.5
        assert state.as_dict()["speed"] == 5
    assert vars(simulation.world.settings) == vars(original)
    assert simulation.vehicle.controls[-1].brake == 1
    assert simulation.vehicle.destroy_count == 0
    session.close()


def test_capture_anchor_after_apply_settings_ticks(simulation: Any) -> None:
    simulation.world.settings_ticks = 3
    with simulation.session() as session:
        assert session.snapshot().frame == 103
        assert session.step(VehicleCommand()).frame == 104


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        ("error", "control command failed: actor unavailable"),
        ("missing", "returned 0 responses"),
        ("extra", "returned 2 responses"),
        ("timeout", "batch timeout"),
    ],
)
def test_rejected_batch_brakes_without_ticking_and_cleans_up(
    simulation: Any, fault: str, message: str
) -> None:
    session = simulation.session(own_vehicle=True).start()
    submit = simulation.client.apply_batch_sync
    attempts = 0

    def reject_once(commands: list[Any], do_tick: bool) -> list[Any]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            if fault == "timeout":
                raise TimeoutError("batch timeout")
            if fault == "missing":
                return []
            response = SimpleNamespace(
                error="actor unavailable", has_error=lambda: fault == "error"
            )
            return [response, response] if fault == "extra" else [response]
        return submit(commands, do_tick)

    simulation.client.apply_batch_sync = reject_once
    with pytest.raises((RuntimeError, TimeoutError), match=message):
        session.step(VehicleCommand(throttle=0.7))
    assert attempts == 2
    assert not session.active
    assert not any(event[0] == "tick" for event in simulation.world.events)
    assert simulation.vehicle.controls[-1].brake == 1
    assert simulation.vehicle.controls[-1].throttle == 0
    session.close()
    assert simulation.vehicle.destroy_count == 1
    assert not simulation.world.settings.synchronous_mode


def test_close_retries_rejected_stop_without_ticking_or_repeating_restoration(
    simulation: Any,
) -> None:
    session = simulation.session().start()
    submit = simulation.client.apply_batch_sync
    simulation.client.apply_batch_sync = lambda commands, do_tick: [
        SimpleNamespace(error="stop rejected", has_error=lambda: True)
    ]
    with pytest.raises(RuntimeError, match="cleanup failed.*stop rejected"):
        session.close()
    assert not simulation.world.settings.synchronous_mode
    assert simulation.vehicle.destroy_count == 0
    simulation.client.apply_batch_sync = submit
    session.close()
    assert simulation.vehicle.controls[-1].brake == 1
    assert simulation.vehicle.controls[-1].hand_brake
    assert not any(event[0] == "tick" for event in simulation.world.events)
    assert simulation.world.events.count(("settings", False)) == 1


def test_reject_build_mismatch_before_world_mutation(simulation: Any) -> None:
    simulation.client.server_version = "other-build"
    with pytest.raises(RuntimeError, match="differ"):
        simulation.session().start()
    assert simulation.world.events == []


def test_reject_existing_owner_unless_explicit(simulation: Any) -> None:
    simulation.world.settings.synchronous_mode = True
    with pytest.raises(SynchronizationError, match="owner"):
        simulation.session().start()
    with simulation.session(SessionConfig(allow_existing_synchronous=True)) as session:
        session.step(VehicleCommand())
    assert simulation.world.settings.synchronous_mode


@pytest.mark.parametrize(
    "fault", ["gap", "external_tick", "wrong_delta", "missing", "timeout"]
)
def test_tick_failures_brake_and_require_new_session(
    simulation: Any, fault: str
) -> None:
    session = simulation.session().start()
    if fault == "gap":
        simulation.world.frame_increment = 2
    elif fault == "external_tick":
        simulation.world.frame += 1
    elif fault == "wrong_delta":
        simulation.world.time_increment = 0.01
    elif fault == "missing":
        simulation.world.missing_actor = True
    else:
        simulation.world.fail_tick = True
    with pytest.raises((SynchronizationError, TimeoutError)):
        session.step(VehicleCommand(throttle=0.5))
    assert simulation.vehicle.controls[-1].throttle == 0
    assert simulation.vehicle.controls[-1].brake == 1
    assert not session.active
    with pytest.raises(RuntimeError, match="healthy"):
        session.step(VehicleCommand())
    session.close()
    assert not simulation.world.settings.synchronous_mode


def test_borrowed_vehicle_is_never_destroyed_on_startup_error(simulation: Any) -> None:
    simulation.vehicle.type_id = "walker.test"
    with pytest.raises(ValueError, match="vehicle"):
        simulation.session().start()
    assert simulation.vehicle.destroy_count == 0


def test_owned_vehicle_is_destroyed_once_on_startup_error(simulation: Any) -> None:
    simulation.client.server_version = "different"
    session = simulation.session(own_vehicle=True)
    with pytest.raises(RuntimeError, match="differ"):
        session.start()
    session.close()
    assert simulation.vehicle.destroy_count == 1


def test_world_mismatch(simulation: Any) -> None:
    simulation.vehicle.get_world = lambda: type("OtherWorld", (), {"id": 8})()
    with pytest.raises(ValueError, match="different"):
        simulation.session().start()


def test_cleanup_attempts_settings_and_destroy_after_control_failure(
    simulation: Any,
) -> None:
    session = simulation.session(own_vehicle=True).start()
    simulation.vehicle.fail_control = True
    with pytest.raises(RuntimeError, match="cleanup"):
        session.close()
    assert not simulation.world.settings.synchronous_mode
    assert simulation.vehicle.destroy_count == 1


def test_context_preserves_original_exception_when_cleanup_fails(
    simulation: Any,
) -> None:
    with pytest.warns(UserWarning, match="cleanup"):
        with pytest.raises(ValueError, match="original"):
            with simulation.session():
                simulation.world.fail_restore = True
                raise ValueError("original")


def test_session_must_be_started_and_cannot_restart_after_close(
    simulation: Any,
) -> None:
    session = simulation.session()
    with pytest.raises(RuntimeError, match="Start"):
        session.snapshot()
    with pytest.raises(RuntimeError, match="Start"):
        session.step(VehicleCommand())
    session.close()
    with pytest.raises(RuntimeError, match="new session"):
        session.start()


def test_close_retries_only_failed_resources(simulation: Any) -> None:
    session = simulation.session(own_vehicle=True).start()
    simulation.world.fail_restore = True
    with pytest.raises(RuntimeError, match="restore failed"):
        session.close()
    assert simulation.vehicle.destroy_count == 1
    simulation.world.fail_restore = False
    session.close()
    assert not simulation.world.settings.synchronous_mode
    assert simulation.vehicle.destroy_count == 1


def test_startup_preserves_error_when_owned_destroy_fails(simulation: Any) -> None:
    simulation.client.server_version = "different"
    simulation.vehicle.destroy = lambda: False
    session = simulation.session(own_vehicle=True)
    with pytest.warns(UserWarning, match="cleanup"):
        with pytest.raises(RuntimeError, match="differ"):
            session.start()
    simulation.vehicle.destroy = lambda: True
    session.close()

"""A strict CARLA API double; no server or CARLA wheel required."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from myosuite_carla import CarlaSession, SessionConfig


def vector(x: float = 0, y: float = 0, z: float = 0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


class FakeWorld:
    def __init__(self) -> None:
        self.id = 7
        self.frame = 100
        self.elapsed = 2.0
        self.settings = SimpleNamespace(
            synchronous_mode=False,
            fixed_delta_seconds=None,
            substepping=False,
            max_substep_delta_time=0.008,
            max_substeps=7,
            no_rendering_mode=False,
            unrelated_field="preserve",
        )
        self.events: list[Any] = []
        self.fail_tick = False
        self.missing_actor = False
        self.frame_increment = 1
        self.time_increment: float | None = None
        self.settings_ticks = 0
        self.fail_restore = False

    def get_settings(self) -> Any:
        return copy.deepcopy(self.settings)

    def apply_settings(self, settings: Any) -> int:
        self.events.append(("settings", settings.synchronous_mode))
        if self.fail_restore and not settings.synchronous_mode:
            raise RuntimeError("restore failed")
        self.settings = copy.deepcopy(settings)
        if settings.synchronous_mode:
            self.frame += self.settings_ticks
            self.elapsed += self.settings_ticks * (settings.fixed_delta_seconds or 0)
        return self.frame

    def tick(self, seconds: float) -> int:
        self.events.append(("tick", seconds))
        if self.fail_tick:
            raise TimeoutError("tick timed out")
        self.frame += self.frame_increment
        self.elapsed += (
            self.settings.fixed_delta_seconds
            if self.time_increment is None
            else self.time_increment
        )
        return self.frame

    def get_snapshot(self) -> Any:
        actor = SimpleNamespace(
            get_transform=lambda: SimpleNamespace(
                location=vector(1, 2, 3),
                rotation=SimpleNamespace(pitch=4, yaw=5, roll=6),
            ),
            get_velocity=lambda: vector(3, 4, 0),
            get_acceleration=lambda: vector(0.1, 0.2, 0.3),
            get_angular_velocity=lambda: vector(0, 0, 90),
        )
        return SimpleNamespace(
            frame=self.frame,
            timestamp=SimpleNamespace(elapsed_seconds=self.elapsed),
            find=lambda actor_id: (
                None if self.missing_actor or actor_id != 42 else actor
            ),
        )


class FakeVehicle:
    def __init__(self, world: FakeWorld) -> None:
        self.id = 42
        self.type_id = "vehicle.test"
        self.world = world
        self.controls: list[Any] = []
        self.destroy_count = 0
        self.fail_control = False

    def get_world(self) -> FakeWorld:
        return self.world

    def apply_control(self, control: Any) -> None:
        raise AssertionError("Controls must use the synchronous command batch")

    def receive_control(self, control: Any) -> None:
        self.world.events.append(("control", control.throttle, control.brake))
        self.controls.append(control)
        if self.fail_control:
            raise RuntimeError("actor unavailable")

    def destroy(self) -> bool:
        self.destroy_count += 1
        return True


@dataclass
class FakeResponse:
    error: str = ""

    def has_error(self) -> bool:
        return bool(self.error)


class FakeClient:
    def __init__(self, world: FakeWorld, vehicle: FakeVehicle) -> None:
        self.world = world
        self.vehicle = vehicle
        self.client_version = "0.10.0-dev"
        self.server_version = "0.10.0-dev"
        self.timeout = 0.0

    def get_world(self) -> FakeWorld:
        return self.world

    def set_timeout(self, seconds: float) -> None:
        self.timeout = seconds

    def get_client_version(self) -> str:
        return self.client_version

    def get_server_version(self) -> str:
        return self.server_version

    def apply_batch_sync(
        self, commands: list[Any], do_tick: bool
    ) -> list[FakeResponse]:
        assert do_tick is False
        self.world.events.append(("batch", len(commands), do_tick))
        responses = []
        for command in commands:
            assert command.actor_id == self.vehicle.id
            try:
                self.vehicle.receive_control(command.control)
                responses.append(FakeResponse())
            except RuntimeError as error:
                responses.append(FakeResponse(str(error)))
        self.world.events.append(("batch_complete", len(responses)))
        return responses


@pytest.fixture
def simulation() -> Any:
    world = FakeWorld()
    vehicle = FakeVehicle(world)
    client = FakeClient(world, vehicle)
    api = SimpleNamespace(
        VehicleControl=lambda **kwargs: SimpleNamespace(**kwargs),
        command=SimpleNamespace(
            ApplyVehicleControl=lambda actor_id, control: SimpleNamespace(
                actor_id=actor_id, control=control
            )
        ),
    )

    def session(config: SessionConfig = SessionConfig(), **kwargs: Any) -> CarlaSession:
        return CarlaSession(client, vehicle, config, api=api, **kwargs)

    return SimpleNamespace(
        world=world, vehicle=vehicle, client=client, api=api, session=session
    )

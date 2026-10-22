"""Physical driver interaction between MyoSuite and CARLA UE5.8."""

from .bridge import BridgeFrame, CarlaBridge, ControlHook
from .cockpit import (
    CockpitConfig,
    CockpitNames,
    PedalConfig,
    SteeringConfig,
    add_cockpit,
)
from .controls import VehicleCommand
from .inputs import (
    Binding,
    BipolarCalibration,
    CallbackSource,
    ContactForce,
    ControlMapper,
    ControlSample,
    IdentityCalibration,
    JointPosition,
    SensorValue,
    UnipolarCalibration,
)
from .presets import cockpit_mapper
from .session import CarlaSession, SessionConfig, SynchronizationError, VehicleTelemetry
from .wrapper import CarlaWrapper

__version__ = "0.1.0"

__all__ = [
    "Binding",
    "BipolarCalibration",
    "BridgeFrame",
    "CallbackSource",
    "CarlaBridge",
    "CarlaSession",
    "CarlaWrapper",
    "CockpitConfig",
    "CockpitNames",
    "ContactForce",
    "ControlHook",
    "ControlMapper",
    "ControlSample",
    "IdentityCalibration",
    "JointPosition",
    "PedalConfig",
    "SensorValue",
    "SessionConfig",
    "SteeringConfig",
    "SynchronizationError",
    "UnipolarCalibration",
    "VehicleCommand",
    "VehicleTelemetry",
    "add_cockpit",
    "cockpit_mapper",
]

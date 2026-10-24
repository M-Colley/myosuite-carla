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
from .fullbody import (
    DriverIntent,
    FullBodyDriver,
    FullBodyDriverConfig,
    SequentialShifter,
    build_fullbody_driver,
)
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
    "DriverIntent",
    "FullBodyDriver",
    "FullBodyDriverConfig",
    "IdentityCalibration",
    "JointPosition",
    "PedalConfig",
    "SensorValue",
    "SequentialShifter",
    "SessionConfig",
    "SteeringConfig",
    "SynchronizationError",
    "UnipolarCalibration",
    "VehicleCommand",
    "VehicleTelemetry",
    "add_cockpit",
    "build_fullbody_driver",
    "cockpit_mapper",
]

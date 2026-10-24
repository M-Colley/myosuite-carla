"""Classical human driver models that drive the full body through its muscles.

:class:`ReflexDriver` is a closed-loop baseline policy for
:class:`~myosuite_carla.driving_env.FullBodyDrivingEnv`. It combines

* two-point visual steering (Salvucci and Gray, 2004, Perception 33:1233):
  the steering rate is ``k_far * d(theta_far)/dt + k_near * d(theta_near)/dt
  + k_i * theta_near`` for the angles to a near and a far point on the lane;
* the Intelligent Driver Model for speed and car following (Treiber, Hennecke
  and Helbing, 2000, Phys. Rev. E 62:1805);
* a perceptual delay and, after a takeover request, a takeover time whose
  default is the meta-analytic mean of 2.72 s (Zhang, de Winter, Varotto,
  Happee and Martens, 2019, Transp. Res. Part F 64:285).

The desired wheel and pedal positions are turned into the motor program by
integral feedback on the measured mechanism inputs, as proprioception would:
the elbow set-point offset tracks the wheel, and foot excitations track the
pedals. The model structures are published; the gains are tuned for the
fitted CARLA Nissan Patrol and are not the cited studies' estimates.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .fullbody import DriverIntent


@dataclass(frozen=True)
class TwoPointSteeringConfig:
    """Two-point steering gains on normalized steer per radian (per second)."""

    k_far: float = 6.0
    k_near: float = 2.0
    k_integral: float = 1.0


@dataclass(frozen=True)
class IdmConfig:
    """Intelligent Driver Model parameters (Treiber et al., 2000, urban values).

    Args:
        max_acceleration: a, m/s^2.
        comfortable_deceleration: b, m/s^2.
        time_headway: T, s.
        min_gap: s0, m.
        exponent: Free-road acceleration exponent delta.
    """

    max_acceleration: float = 1.0
    comfortable_deceleration: float = 1.5
    time_headway: float = 1.6
    min_gap: float = 2.0
    exponent: float = 4.0


def idm_acceleration(
    speed: float,
    desired_speed: float,
    gap: float | None,
    closing_speed: float,
    config: IdmConfig = IdmConfig(),
) -> float:
    """Return the Intelligent Driver Model acceleration in m/s^2."""
    free = 1.0 - (max(speed, 0.0) / max(desired_speed, 0.1)) ** config.exponent
    interaction = 0.0
    if gap is not None and math.isfinite(gap):
        desired_gap = config.min_gap + max(
            0.0,
            speed * config.time_headway
            + speed
            * closing_speed
            / (
                2 * math.sqrt(config.max_acceleration * config.comfortable_deceleration)
            ),
        )
        interaction = (desired_gap / max(gap, 0.1)) ** 2
    return config.max_acceleration * (free - interaction)


#: Static maps of the default seated driver at rest, measured by holding each
#: intent for 1.5 s (2 s for steering): (mechanism input, intent) pairs.
THROTTLE_MAP = (
    (0.0, 0.0),
    (0.258, 0.004),
    (0.287, 0.006),
    (0.348, 0.008),
    (0.405, 0.01),
    (0.461, 0.012),
    (0.553, 0.016),
    (0.625, 0.02),
    (0.765, 0.03),
    (0.941, 0.05),
    (1.0, 0.08),
)
BRAKE_MAP = (
    (0.0, 0.0),
    (0.387, 0.02),
    (0.568, 0.04),
    (0.701, 0.06),
    (0.798, 0.08),
    (0.871, 0.1),
    (1.0, 0.15),
)
STEER_MAP = (
    (-0.877, -0.3),
    (-0.533, -0.2),
    (-0.253, -0.1),
    (-0.1, -0.05),
    (0.0, 0.0),
    (0.105, 0.05),
    (0.262, 0.1),
    (0.641, 0.2),
    (0.801, 0.3),
)


def _inverse(table: tuple[tuple[float, float], ...], value: float) -> float:
    inputs, intents = zip(*table)
    return float(np.interp(value, inputs, intents))


@dataclass(frozen=True)
class ReflexDriverConfig:
    """Timing, pedal mapping and proprioceptive loop gains.

    Args:
        perception_delay: Delay of the perceived scenario state, s.
        takeover_time: Time from a takeover request to driving, s.
        steering: Two-point steering gains.
        idm: Car-following parameters.
        throttle_bias: Throttle that holds cruising speed at zero acceleration.
        throttle_per_mps2: Throttle per m/s^2 of desired acceleration.
        brake_per_mps2: Brake per m/s^2 of desired deceleration.
        coast_band: Desired decelerations smaller than this coast, m/s^2.
        throttle_map, brake_map, steer_map: Measured static maps used as
            feedforward inverses.
        steer_integral: Integral gain of the wheel position loop, rad/s.
        pedal_integral: Integral gain of the pedal loops, excitation/s.
        max_steer_rate: Largest change of desired normalized steer per second.
    """

    perception_delay: float = 0.2
    takeover_time: float = 2.72
    steering: TwoPointSteeringConfig = TwoPointSteeringConfig()
    idm: IdmConfig = IdmConfig()
    throttle_bias: float = 0.3
    throttle_per_mps2: float = 0.25
    brake_per_mps2: float = 0.15
    coast_band: float = 0.4
    throttle_map: tuple[tuple[float, float], ...] = THROTTLE_MAP
    brake_map: tuple[tuple[float, float], ...] = BRAKE_MAP
    steer_map: tuple[tuple[float, float], ...] = STEER_MAP
    steer_integral: float = 0.3
    pedal_integral: float = 0.05
    max_steer_rate: float = 0.8


class ReflexDriver:
    """Closed-loop baseline that maps the environment state to driver intents.

    Use :meth:`act` with ``env.state`` (named observations) every step; call
    :meth:`reset` at every episode start.
    """

    def __init__(self, dt: float, config: ReflexDriverConfig | None = None) -> None:
        self.dt = float(dt)
        self.config = config if config is not None else ReflexDriverConfig()
        self.reset()

    def reset(self) -> None:
        """Forget perceived history and release every control."""
        delay = max(1, round(self.config.perception_delay / self.dt))
        self._history: deque[dict[str, float]] = deque(maxlen=delay)
        self._previous: dict[str, float] | None = None
        self._steer = 0.0
        self._steer_integral = 0.0
        self._pedal_integral = 0.0
        self._engaged = True
        #: Last desired acceleration (m/s^2) and normalized steer, offsets included.
        self.demand = (0.0, 0.0)

    @property
    def engaged(self) -> bool:
        """Return whether the driver is in the loop (not awaiting a takeover)."""
        return self._engaged

    def act(
        self,
        state: Mapping[str, float],
        *,
        acceleration_offset: float = 0.0,
        steer_offset: float = 0.0,
    ) -> DriverIntent:
        """Return the next intent from the current named state.

        Args:
            state: Named observations, e.g. ``FullBodyDrivingEnv.state``.
            acceleration_offset: Added to the desired acceleration, m/s^2.
            steer_offset: Added to the desired normalized steer (not
                integrated), so a learned residual acts on the demand and is
                executed by the proprioceptive loops and the muscles.
        """
        config = self.config
        self._history.append(dict(state))
        perceived = self._history[0]
        current = dict(state)
        if perceived.get("automation", 0.0):
            # Out of the loop until the takeover time has elapsed.
            self._engaged = (
                perceived.get("takeover_requested", 0.0) > 0
                and perceived.get("time_since_request_s", 0.0) >= config.takeover_time
            )
        else:
            self._engaged = True
        if not self._engaged:
            self._previous = perceived
            self._steer = current.get("input_steer", 0.0)
            self.demand = (0.0, 0.0)
            return DriverIntent()
        steering = config.steering
        previous = self._previous or perceived
        rate = (
            steering.k_far
            * (perceived["far_angle_rad"] - previous["far_angle_rad"])
            / self.dt
            + steering.k_near
            * (perceived["near_angle_rad"] - previous["near_angle_rad"])
            / self.dt
            + steering.k_integral * perceived["near_angle_rad"]
        )
        self._previous = perceived
        rate = float(np.clip(rate, -config.max_steer_rate, config.max_steer_rate))
        self._steer = float(np.clip(self._steer + rate * self.dt, -1.0, 1.0))
        desired = float(np.clip(self._steer + steer_offset, -1.0, 1.0))
        measured = current.get("input_steer", 0.0)
        self._steer_integral = float(
            np.clip(
                self._steer_integral
                + config.steer_integral * (desired - measured) * self.dt,
                -0.3,
                0.3,
            )
        )
        offset = float(
            np.clip(
                _inverse(config.steer_map, desired) + self._steer_integral,
                -0.5,
                0.5,
            )
        )

        gap = perceived["gap_m"] if perceived.get("lead_present", 0.0) else None
        accel = idm_acceleration(
            perceived["speed_mps"],
            perceived["target_speed_mps"],
            gap,
            perceived.get("closing_speed_mps", 0.0),
            config.idm,
        ) + float(acceleration_offset)
        self.demand = (accel, desired)
        right = left = 0.0
        if accel > -config.coast_band:
            throttle = float(
                np.clip(
                    config.throttle_bias + config.throttle_per_mps2 * accel, 0.0, 1.0
                )
            )
            error = throttle - current.get("input_throttle", 0.0)
            self._pedal_integral = float(
                np.clip(
                    self._pedal_integral + config.pedal_integral * error * self.dt,
                    -0.02,
                    0.02,
                )
            )
            right = _inverse(config.throttle_map, throttle) + self._pedal_integral
        else:
            self._pedal_integral = 0.0
            brake = float(
                np.clip(config.brake_per_mps2 * (-accel - config.coast_band), 0.0, 1.0)
            )
            left = _inverse(config.brake_map, brake)
        return DriverIntent(
            right_foot=float(np.clip(right, 0.0, 1.0)),
            left_foot=float(np.clip(left, 0.0, 1.0)),
            steer=offset,
        )

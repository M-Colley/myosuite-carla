# CARLA compatibility contract

This is Mark Colley's standalone
[`myosuite-carla`](https://github.com/M-Colley/myosuite-carla) package. It depends
on MyoSuite and CARLA without modifying either project's package layout.

The integration targets CARLA's **`ue58-dev`** branch for Unreal Engine 5.8,
which becomes **CARLA 1.0**. The API review was made against commit
[`defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`](https://github.com/carla-simulator/carla/tree/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5)
(2026-10-22) and repeated on 2026-11-04 against the `1.0` tag; see
[CARLA 1.0](#carla-10). Actual live validation has passed against the published
CARLA 0.10.0 / UE5.5 release; UE5.8 and CARLA 1.0 runtime validation remain
pending. See [VALIDATION.md](VALIDATION.md) for measurements and build provenance.

## MyoSuite dependency

Cockpit composition requires `myosuite.core.model_builder.ModelBuilder` from
MyoSuite's `dev` branch. The offline validation used MyoSuite commit
`c0af9c656de2aa1807924e63d0d5320d77aa1ebd` (package version `3.0.0`). Install a
separate `dev` checkout in editable mode as shown in [README.md](README.md).
There is no implicit PyPI MyoSuite dependency: a release carrying a numerically
compatible version may have a different API. The standalone package itself
supports Python 3.10+, MuJoCo 3.6+ and Gymnasium 1.0 through 1.2; your CARLA wheel
may restrict the Python minor version and platform further.

The 2026-10-27 integration review also checked current MyoSuite `dev` commit `230f198458b501c423a0892b78da60419ffc49a3`. The final integration tests include the fatigue reset fix from [MyoSuite PR 575](https://github.com/MyoHub/myosuite/pull/575). Embodied human driving and embodied UI composition use their public CPU APIs. Vehicle adapters restore the initial surrogate pose and speed on reset. `VehicleSessionEngine` requires an explicit `reset_vehicle(session, seed, options)` callback for live vehicle resets.

Version 0.1.1 adds Shapely 2 or newer for planar vehicle footprints. Embodied human 0.1.1 uses `VehicleMotion.acceleration_at` for head sensing and requires this bridge version. Following trial archives use schema 2 to retain source groups. The reader accepts the earlier archives. Shared source grouping is selected explicitly for paired human and simulated evaluations.

## Acceleration control and empirical profiles

Actuator calibration affects the inferred human following parameters. The legacy fixed throttle bias produced positive acceleration for a zero acceleration request. `AccelerationControlConfig` optionally inverts the vehicle drive curve and drag model, then tracks measured pedal positions. `ReflexDriver` returns muscle intents through the existing body interface. Its default configuration retains the legacy controller.

The inverse requires vehicle specific `VehicleModelParams`. Its default identifies the CARLA 0.10 Nissan Patrol surrogate. Forward driving on a level road is assumed. Reverse driving raises an error, and finite pedal travel limits achievable acceleration. For a custom vehicle, identify its drive, drag and braking parameters with `examples/identify_vehicle.py` and `examples/fit_vehicle_model.py`. Supply the fitted parameters to the controller and the evaluated surrogate. Live execution requires an additional check against that actual CARLA vehicle.

```python
from dataclasses import asdict
import json

from myosuite_carla.driver_models import AccelerationControlConfig, ReflexDriverConfig
from myosuite_carla.vehicle_model import SurrogateVehicle, carla_nissan_patrol

vehicle_parameters = carla_nissan_patrol()
vehicle = SurrogateVehicle(params=vehicle_parameters)
reflex_configuration = ReflexDriverConfig(
    acceleration_control=AccelerationControlConfig(vehicle=vehicle_parameters)
)
saved_configuration = json.loads(json.dumps(asdict(reflex_configuration)))
restored_configuration = ReflexDriverConfig.from_dict(saved_configuration)
assert restored_configuration.acceleration_control.vehicle == vehicle.params
```

Profiles must preserve the complete controller and vehicle configuration. Pedal targets and resulting motion depend on that full configuration. `ReflexDriverConfig.from_dict` restores the nested typed configuration. In embodied human, `ResolvedDrivingProfile.make_reflex_config` uses this method. Physical reports also retain the body configuration, simulation timestep, source window and seed. `ClosedLoopFollowingProtocol` binds the complete controller configuration to its cache identity.

Longitudinal control currently receives exact simulated speed, bumper gap, closing speed and acceleration. The attentive human estimates lateral state. Consequently, acceleration tracking measures the actuator interface under these telemetry assumptions. The engineering test uses the same vehicle model for inversion and simulation. Human sensing accuracy requires separate empirical evaluation.

The reflex history uses `N = max(1, round(perception_delay / dt))` entries. Appending the current state before reading the oldest entry gives a steady history lag of `(N - 1) * dt`. The retained study used `perception_delay = dt = 0.02 s`, which adds zero history lag. Record the configured delay, timestep and realized lag when comparing human response times. Changes to this timing require a new comparison with the retained legacy traces.

The independent trajectory test remains inconclusive. Combined normalized error decreased by 9.10%, while bumper gap RMSE increased. Its paired source interval includes zero. Retain speed, gap and acceleration errors separately, match source identities and seeds, and select configurations before opening test results. The [retained study](evidence/actuation-improvement-2026-10-28/summary.json) supports continued experimental use of the correction. Its live CARLA validation remains pending.

## Install the matching client

Build the server and Python client from the same CARLA checkout, or use the
wheel supplied with the server package. The wheel must match the operating
system, architecture, and Python minor version. Do **not** install `carla` from
PyPI for this target: CARLA's UE5.8 quick start identifies that package as the
UE4 client. See the pinned [quick start](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/start_quickstart.md#L54)
and [source build instructions](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/README.md).

`CarlaSession` requires equal client and server version strings. Equality is a
useful mismatch check, but cannot prove the branch or commit. The reviewed
[CMake version](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/CMakeLists.txt#L37)
is `0.10.0`; Unreal Engine 5.8 is not CARLA version 5.8. From the `1.0` tag the
version is `1.0`, without a patch component. Record the CARLA source
commit alongside experiment results.

## CARLA 1.0

Re-checked on 2026-11-04 against the
[`1.0` tag](https://github.com/carla-simulator/carla/tree/ecb0af47fcd991c0ec6f48433df2a09721b2fec8)
(`ecb0af47`, branch `ue5/1.0`, 2026-11-02), 40 commits after `defabe35`, and
`ue58-dev` at `a3c6d7f4` (2026-11-04), seven commits later. The GitHub release
was not yet published. CARLA's download server had a Linux `CARLA_1.0.tar.gz`
(13,985,818,522 bytes) last modified 2026-11-02 09:36 UTC, before the version
commit and the tag, so its source commit and version string are unverified; no
Windows package was found. No CARLA 1.0 server has been run with this bridge.

- **API.** Every call made by the bridge, `embodied-ui`'s `CarlaEngine` and the
  examples is unchanged: `VehicleControl`, `command.ApplyVehicleControl`,
  `apply_batch_sync`, `World.tick`/`get_settings`/`apply_settings`,
  `WorldSettings`, actor transforms and velocities, light state, spawning and
  `Map.get_waypoint`. LibCarla's client sources are identical to `defabe35`,
  so the batch-then-tick ordering above still holds. No bridge code change is
  needed for the API.
- **Version.** From upstream `8d30115a8` (in the tag) servers and wheels report
  `1.0`; earlier `ue58-dev` builds, including `defabe35` and the client
  validation snapshot `ee5859b4`, report `0.10.0`. The version string now
  distinguishes 1.0 from the UE5.5 release, and `CarlaSession` rejects a
  0.10.0 wheel against a 1.0 server.
- **Steering input curve.** Since upstream `674a3a70e` (2026-08-12, in
  `defabe35` and 1.0) CARLA forces Chaos's steering input curve to
  `LinearFunction`. UE5's default `SquaredFunction` maps a command `s` to about
  `s|s|` of the steering range. The retained 0.10.0 drives used the squared
  curve: in every speed band from 2 to 14 m/s the best exponent is 1.95 to 2.0,
  the effective road-wheel angle is about 0.9 rad × `s|s|`, and a linear fit
  leaves 17 to 313 times the squared error; within single sweeps the median
  exponent is 1.95 ([analysis](evidence/carla-1.0-review/steer-input-curve.json)).
  A 0.1 command therefore turns the road wheels roughly an order of magnitude
  more under 1.0 (upstream measured about 7° against 0.6°). Steering that was
  identified or tuned on 0.10.0 or its surrogate does not transfer: the
  `carla_nissan_patrol()` steering table, `AutomationConfig.steer_gain`, the
  two-point steering gains and the calibration of cabin wheel travel to `steer`.
  The surrogate now identifies the exponent (`steer_exponent`, 1.97 for the
  0.10.0 Patrol); its earlier linear table misspecified 0.10.0 and had
  4 to 8 times the held-out yaw-rate error (see VALIDATION.md). Re-identify the
  vehicle on 1.0, where the exponent should come out near 1, before using it
  there. Chaos also rate-limits the raw steering input (UE5.8 engine
  defaults: the magnitude rises at 2.5 and falls at 5.0 per second). CARLA's
  C++ in the 1.0 tag and `ue58-dev` overrides only the curve. The vehicle
  blueprints were not inspected, so the 1.0 rates are unconfirmed. On 0.10.0
  a joint fit recovered a rise rate of 2.5, and `carla_nissan_patrol()` now
  applies these rates with a steer lag that grows with speed (see
  VALIDATION.md). Check both on 1.0 before reusing them.
- **Vehicle physics.** UE5.8 Chaos and the 1.0 vehicle content differ from
  UE5.5. All fitted drive, brake, drag and lag parameters were identified on
  0.10.0 and need re-identification on 1.0.
- **Geolocation.** `Map.transform_to_geolocation` in `defabe35` passed CARLA's
  `y` (south) as the northing, so latitude grew southwards; upstream
  `2466fff9c` (in the tag) corrected it. 0.10.0 and 1.0 both increase latitude
  northwards (towards −`y`), but 0.10.0 offsets a scaled Mercator projection while 1.0
  uses the map's `GeoProjection`, so recorded coordinates are not
  interchangeable. `DriveRecorder` stores the server version with each
  recording.
- **Physics control.** 1.0 applies `WheelPhysicsControl` fields at runtime
  (upstream `538ecebea`) and reads and writes
  `forward_gear_ratios`/`reverse_gear_ratios`. Upstream found that the tyre
  friction of 3.5 written at spawn never reached Chaos, which used the
  blueprint value (2.0 on most vehicles), and removed it. 0.10.0 has the same
  spawn code, so its `get_physics_control()` reports 3.5 and applying that
  control back raises the grip. The bridge applies no physics control.
- **Packages and wheels.** The tag's release workflow builds wheels for Python
  3.10, 3.11 and 3.12 and a package can ship one per version; install the wheel
  whose `cpXY` tag matches the interpreter. There is no 1.0 wheel for Python
  3.13 or 3.14.
- **After the tag.** `ue58-dev` adds a flat package layout (`CarlaUnreal.sh`
  or `.exe` at the package root, `46caf61b6`), Windows build fixes
  (`a38a43462`) and first-gear reporting while Autoware's acceleration control
  drives (`570779f0c`). `VehicleControl` driving is unaffected; the reverse
  and gear lag measured on 0.10.0 is untested on 1.0.
- **Client fixes.** Of the nine client fixes in the validation snapshot
  `ee5859b4`, 1.0 contains an independent gear-ratio conversion fix. The
  episode exception and map synchronization fix
  ([PR 9999](https://github.com/carla-simulator/carla/pull/9999)) was approved
  but not merged before the tag; the other seven were closed unmerged. A stock
  1.0 client lacks them.

## Physical coupling and controls

MuJoCo simulates muscle forces, limb motion, pedal contacts, and steering-wheel
motion. The mapper reads those mechanisms and sends the resulting scalar inputs
to CARLA, which simulates the vehicle and road. CARLA does not share a contact
solver with MuJoCo. Its Python vehicle API provides no pedal reaction force,
clutch input, or steering-wheel force feedback. Any return force model must be
implemented explicitly in MuJoCo. The bridge applies no inertial load by itself;
the full-body driver applies gravity, the d'Alembert loads of the measured
vehicle motion and a steering-feel torque when the caller passes
`VehicleMotion.from_telemetry(...)` to `set_vehicle_motion` (`vehicle_loads.py`).

The reviewed [VehicleControl contract](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/PythonAPI/docs/control.yml#L6)
supports throttle/brake in `[0, 1]`, steering in `[-1, 1]` (positive right),
boolean handbrake/reverse/manual shifting, and an integer gear. Steering is a
normalized input, not a road-wheel or cabin steering-wheel angle. Calibrate the
MuJoCo wheel's travel and sign for the selected CARLA vehicle. CARLA's
speed-dependent steering curve can change the resulting road-wheel angle.

`vehicle.get_control()` reports the gearbox state for `reverse` and `gear`. With
CARLA 0.10.0 a commanded reverse or manual gear appeared there two frames
(40 ms) after it was applied, and up to three frames in the 2026-10-24 runs;
throttle, brake, steering, handbrake and the manual-shift flag matched on the
same frame. The full-body validation therefore bounds reverse/gear agreement to
25 frames instead of requiring the same frame.

In CARLA 0.10.0 `get_physics_control()` returns zero `wheels[i].location`
values, and its `forward_gear_ratios`/`reverse_gear_ratios` cannot be read
from Python (no converter for `std::vector<float>`). The vehicle surrogate
therefore uses a published wheelbase and fits effective drive and steering
tables instead of using CARLA's drivetrain parameters. CARLA 1.0 makes the gear
ratios readable; see [CARLA 1.0](#carla-10) for its steering curve. Setting a speed with
`set_target_velocity` right after `set_transform` appears in the next frame's
`get_acceleration()` as a one-frame spike (594 m/s^2 for 0 to 12 m/s);
`carla_scenario.teleport` ticks once more so that spike never reaches the body.

Additional measured channels are available to application hooks. CARLA supports
[vehicle lights and door commands](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/PythonAPI/docs/actor.yml#L543),
subject to the vehicle asset. These commands do not create a shared physical
door/contact model. The API's `set_wheel_steer_direction` concerns the vehicle's
**road wheels**, changes their appearance only, and requires
`set_wheel_animation_override(True)` in this branch. It is not the steering
control path or a cabin steering-wheel setter.

## Time and lifecycle

One completed MyoSuite control step maps to one CARLA frame, with
`frame_skip * model.opt.timestep == fixed_delta_seconds`. Mechanisms are sampled
after the MyoSuite step; the command is then applied for CARLA's next tick. This
is discrete, sequential coupling rather than a shared continuous physics solve.
The wrapper preserves the muscle task's action, observation, reward, and
termination contract, adding vehicle feedback to `info["carla"]`.

Controls use `client.apply_batch_sync([ApplyVehicleControl(...)], False)` and
check the response before the single explicit `world.tick`. The acknowledged
batch prevents asynchronous `vehicle.apply_control` delivery from racing the
tick. The batch itself requests no tick. This ordering is supported by the
pinned [client implementation](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/LibCarla/source/carla/client/detail/Client.cpp)
and was necessary to pass actual fast-loop control equality checks.

Only one client may tick the CARLA world. Stop other ticking applications and
disable the controlled vehicle's autopilot. If a Traffic Manager is in use,
configure it for synchronous operation too. The session enables synchronous
mode, fixed delta, and physics substepping, requiring
`fixed_delta_seconds <= max_substep_delta_time * max_substeps`. A CARLA physics
substep of `0.01 s` or less is recommended by the pinned
[synchrony guide](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/core_synchrony.md).

CARLA's [`World.apply_settings`](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/LibCarla/source/carla/client/World.cpp#L58)
can advance simulation frames while establishing a fixed delta. The session
anchors its frame/time after this call. Restoring a previous fixed delta on
close can likewise advance frames. A queued emergency brake takes effect when
CARLA processes the control; `stop()` does not tick or guarantee an immediate
physical stop.

Do not interleave external `vehicle.apply_control` calls while a session owns
the vehicle. Batches bypass the Python vehicle handle's private sticky-control
cache. After handing back a borrowed vehicle, send its first external command
with `client.apply_batch_sync(..., False)` as well; a cached identical command
sent with `vehicle.apply_control` can otherwise be suppressed while the final
brake remains active.

Driver `reset(seed=...)` resets the MyoSuite episode and re-anchors the clocks.
It does not reload CARLA, reposition its vehicle, reset traffic, or seed CARLA.
For repeatable scenarios, follow CARLA's determinism requirements: enable
synchronous fixed timing before a world reload, reload between repetitions,
seed the Traffic Manager when used, and check batched command failures. Do not
load or reload a world while a session is active; recreate the actor and session
after changing worlds.

## Feedback and backend scope

Feedback uses a single actor snapshot from the returned CARLA frame. Coordinates
are CARLA's left-handed world frame: x forward, y right, z up. Location is in
metres; linear velocity in m/s; acceleration in m/s²; angular velocity in
**degrees/s**. Rotation tuples are `(pitch, yaw, roll)` in degrees. See
[coordinate conventions](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/core_coordinate_conventions.md).
These values are not expressed in the driver's local frame. Explicitly
transform them before using them in biomechanical feedback controllers.

The bridge/session/wrapper run one CPU MuJoCo driver against one CARLA vehicle.
The cockpit's `ModelBuilder` recipe can be reused when constructing a GPU mjlab
task, but this package does not provide a batched CARLA server, a registered
driving task or mjlab twin, or CPU/GPU training parity. New driving environments
must follow MyoSuite's paired CPU/mjlab registration and cross-backend contract.

# CARLA compatibility contract

This is Mark Colley's standalone
[`myosuite-carla`](https://github.com/M-Colley/myosuite-carla) package. It depends
on MyoSuite and CARLA without modifying either project's package layout.

The integration targets CARLA's **`ue58-dev`** branch for Unreal Engine 5.8.
The API review was made against commit
[`defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`](https://github.com/carla-simulator/carla/tree/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5)
(2026-10-22). Actual live validation has passed against the published CARLA
0.10.0 / UE5.5 release; UE5.8 runtime validation remains pending. See
[VALIDATION.md](VALIDATION.md) for measurements and build provenance.

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
is `0.10.0`; Unreal Engine 5.8 is not CARLA version 5.8. Record the CARLA source
commit alongside experiment results.

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
tables instead of using CARLA's drivetrain parameters. Setting a speed with
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

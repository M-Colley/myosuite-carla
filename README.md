# MyoSuite–CARLA

Mark Colley's [`myosuite-carla`](https://github.com/M-Colley/myosuite-carla) is a
standalone installable package connecting a physical MuJoCo
driver to a CARLA **Unreal Engine 5.8 `ue58-dev`** vehicle. Muscles move the limbs;
limbs contact passive pedals and a steering wheel; measured mechanism travel or
contact force becomes `carla.VehicleControl`. CARLA simulates the vehicle.

**Validation status:** with a real CARLA 0.10.0 / UE5.5 server, 312 tests pass
with zero skips (94% statement coverage). One seated MyoSuite full body, loaded
by gravity and the measured vehicle motion, operates throttle, brake,
push-pull steering, reverse, sequential gears and the parking brake in a
single live drive, with all 13 physical checks passing in both the default and
the fast configuration. The same body drives lead-vehicle braking, takeover and
lane-change scenarios in CARLA without collision. The single-finger acceptance
fixtures pass on two Windows PCs, all 11 four-wheel vehicles, a rendered server
and a 15,000-frame soak. Actual **UE5.8** validation remains pending a runnable
build; see [measured results](VALIDATION.md) and [runtime status](RUNTIME_STATUS.md).

The package provides:

- A composable `ModelBuilder` cockpit: spring-loaded throttle/brake pedals,
  a collision steering wheel, mechanical stops, damping and position/touch sensors.
- Named joint, sensor-component, contact-force and custom signal bindings;
  asymmetric steering calibration, deadzones, nonlinear response and saturation.
- Throttle, brake, steering, handbrake, reverse, manual shifting and gear support;
  additional measured channels for application hooks such as lights and doors.
- An exclusive synchronous session with acknowledged controls, matched clocks,
  frame-gap detection,
  frame-aligned feedback, emergency-brake commands and retryable cleanup.
- A Gymnasium wrapper preserving the muscle environment's action/observation
  spaces, rewards and termination; feedback is added to `info["carla"]`.
- A full-body driver (`myofullbody`, 416 muscles) that feels gravity, the
  vehicle's acceleration and rotation, and steering feel.
- Per-frame recordings exported as a Frictionless data package (CSV), OpenSim
  `.mot`/`.sto`, an arm_force_driving CARLA recording, Parquet and MCAP.
- A single-track vehicle surrogate fitted to CARLA drives, usable wherever a
  `CarlaSession` is, for training and batched rollouts without CARLA.
- An unregistered Gymnasium driving task with lead-vehicle, takeover and lane
  change scenarios, and a classical human driver model as a baseline policy.
- Muscle conditions and variability from MyoSuite (sarcopenia, selective
  weakness, 3CC-r fatigue, motor noise) and posture variants.
- An aggregate-only comparison with human driving data, with data-adequacy
  checks and human-vs-human floors.

## Install

For a clean Windows PC, follow [NEW_PC_SETUP.md](NEW_PC_SETUP.md): prerequisites,
private-repository authentication, pinned dependencies, UE5.8 source building
or the tested UE5.5 download, server startup, live tests and troubleshooting.

Use the Python minor version supported by your CARLA UE5.8 build. Clone this
package and MyoSuite into separate directories. MyoSuite's **`dev` branch** is
required for the `myosuite.core.model_builder.ModelBuilder` API; a version number
alone does not establish compatibility with that API.

```shell
git clone --branch dev https://github.com/MyoHub/myosuite.git ../myosuite
python -m pip install -e ../myosuite
python -m pip install -e ".[test,data,analysis]"
python -m pip install /path/to/your/ue58-build/PythonAPI/carla/dist/<matching-wheel>.whl
```

The `data` extra (pandas, pyarrow, mcap) enables Parquet and MCAP exports; the
`analysis` extra (SciPy, scikit-learn, pandas) enables vehicle-surrogate fitting
and the human-data comparison. `pip install opensim` additionally lets the tests
read the OpenSim exports back with OpenSim's own reader.

Run these commands from this package's root. If you already have a MyoSuite
`dev` checkout, install that path instead of cloning another copy. Use a virtual
environment for the installation. `pip install .` installs the bridge with
MuJoCo/Gymnasium/NumPy; MyoSuite and the matching CARLA client are installed
separately. Importing `myosuite_carla` does not import CARLA or MyoSuite, start a
server, or register environments. See [COMPATIBILITY.md](COMPATIBILITY.md) for
the tested MyoSuite revision and CARLA API contract.

**Use the client wheel from the same build as the UE5.8 server.** CARLA's
[UE5.8 quick start](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/start_quickstart.md)
warns against `pip install carla` for this branch. Version equality is checked at
session startup but cannot distinguish commits with identical version strings.
See [COMPATIBILITY.md](COMPATIBILITY.md) for the pinned upstream API contract.

## Compose the physical driver

Add the cockpit **before compiling** your driver's model, using the same recipe
in its CPU and mjlab implementations:

```python
from myosuite.core.model_builder import ModelBuilder
from myosuite_carla import CockpitConfig, add_cockpit

cockpit_config = CockpitConfig()
builder = ModelBuilder().attach_fragment("arm")
model, spec = add_cockpit(builder, cockpit_config).build()
```

Set `CockpitConfig.position`, `quaternion`, `parent_body`, pedal placements and
wheel placement to fit your limb or full-body model. Defaults are a generic rig,
not an anthropometrically fitted driver. The cockpit's local axes are **X right,
Y forward, Z up**; orient it explicitly relative to the host model. Pedals move
along local +Y. Positive wheel rotation moves its top right. All configuration
angles are radians and quaternions use `(w, x, y, z)`.

Install the compiled model in your normal `MyoGymnasiumEnv` subclass. Continue
using MyoSuite's muscle action stages and physics/finalization primitives. Ensure
the hand/foot collision geometry actually contacts the controls. The cockpit
adds three passive joints and nine sensors, without adding actuators. Rebuild
bindings whenever you recompile a model; joint IDs are not qpos addresses.

Existing reaching/locomotion checkpoints do not acquire a driving skill by
wrapping them. Fit the scene and train or supply a suitable driver policy. If
you create a registered driving task, follow MyoSuite's paired CPU/mjlab task
contract; this package adds no environment IDs or alternate registry.

## Run in lockstep

Create the driver through `myosuite.make_env` and choose a CARLA vehicle already
spawned in the current world. Disable its autopilot and stop other ticking
clients. Traffic Manager, if used by your scenario, must also be synchronous.

```python
from myosuite import make_env
from myosuite_carla import (
    CarlaBridge, CarlaSession, CarlaWrapper, SessionConfig, cockpit_mapper,
)

driver = make_env("your_registered_driver_id")
control_dt = driver.unwrapped.model.opt.timestep * driver.unwrapped.frame_skip
mapper = cockpit_mapper(driver.unwrapped.model, cockpit_config)
vehicle.set_autopilot(False)

try:
    with CarlaSession(client, vehicle, SessionConfig(fixed_delta_seconds=control_dt)) as session:
        bridge = CarlaBridge(mapper, session, control_dt)
        env = CarlaWrapper(driver, bridge)
        observation, info = env.reset(seed=0)
        while True:
            muscle_action = policy(observation, info)
            observation, reward, terminated, truncated, info = env.step(muscle_action)
            feedback = info["carla"]
            if terminated or truncated:
                break
finally:
    driver.close()
```

`client`, `vehicle`, `policy` and the registered driver's `cockpit_config` come
from your scenario/application. The runnable
[live_drive.py](examples/live_drive.py) handles connecting, spawning a vehicle,
running one episode, restoring settings and destroying its own vehicle:

```shell
python examples/live_drive.py --driver-factory your_driver:create_env --policy your_driver:policy --steps 1000
```

`create_env()` returns a CPU MyoSuite environment. Expose the configuration as
`env.unwrapped.cockpit_config` if it differs from the default. `policy(obs, info)`
returns the environment's existing muscle action. Both callbacks are local Python
functions; load your trained checkpoint inside your policy module.

For a custom MuJoCo loop, call `bridge.reset(data)` once, then advance the driver
by `control_dt`, call `mj_forward`, and call `bridge.step(data)` once per control
period. Multiple MuJoCo physics substeps per control period are supported. The
driver and CARLA control periods must match; differing rates fail before CARLA
advances. One bridge drives one vehicle on CPU. GPU batching and force feedback
are application work; the cockpit recipe itself is reusable on mjlab.

## Bind other models or measured controls

You can use an existing cockpit model instead of the supplied recipe:

```python
from myosuite_carla import (
    Binding, BipolarCalibration, ContactForce, ControlMapper,
    IdentityCalibration, JointPosition, SensorValue, UnipolarCalibration,
)

mapper = ControlMapper(model, [
    Binding("throttle", JointPosition("accelerator"), UnipolarCalibration(0.0, 0.06)),
    Binding("brake", ContactForce("foot_geom", "brake_geom"), UnipolarCalibration(0.0, 250.0)),
    Binding("steer", JointPosition("wheel"), BipolarCalibration(-4.7, 0.0, 4.7)),
    Binding("hand_brake", SensorValue("handbrake_position"), UnipolarCalibration(0.0, 0.04)),
    Binding("gear", SensorValue("selected_gear"), IdentityCalibration()),
])
```

Names and component indices are validated against the compiled model. Contact
forces sum all matching compressive normal contacts in newtons. Signal and
calibration outputs must be finite. Standard control ranges are validated;
calibrations saturate at endpoints. Boolean channels use a 0.5 threshold. Gear
must be an exact signed 32-bit integer; physically noisy gear mechanisms should
use an application-defined quantizing calibration.

Brake priority is enabled by default: any positive calibrated brake suppresses
throttle. Choose a deadzone for unloaded pedal noise; set `brake_priority=False`
for experiments that intentionally combine the pedals. Missing standard channels
default to zero/false. Additional channel names remain in `ControlSample.values`.

Extension hooks receive `(session, sample)` after measurement and before the
CARLA tick. For example, bind a custom `lights` channel and use:

```python
def update_lights(session, sample):
    state = carla.VehicleLightState.LowBeam if sample.values["lights"] >= 0.5 else carla.VehicleLightState.NONE
    session.vehicle.set_light_state(state)

bridge = CarlaBridge(mapper, session, control_dt, hooks=(update_lights,))
```

For door commands, track rising/falling edges in your hook. CARLA's API has no
clutch field or cabin steering-wheel pose/torque setter; a custom clutch signal
needs explicit application dynamics. Road-wheel animation commands do not steer
the vehicle's physics.

## One full-body driver

`build_fullbody_driver()` composes MyoSuite's `myofullbody` (416 muscles, from
`myo_sim`) with the cockpit into one MuJoCo model. The seated body works every
control by contact:

| Body part | Mechanism | CARLA control |
| --- | --- | --- |
| Right forefoot (plantarflexors) | Throttle pedal | `throttle` |
| Left forefoot (plantarflexors) | Brake pedal | `brake` |
| Both hands (elbow flexors/extensors) | Free-spinning cradles at 3 and 9 o'clock | `steer`, push-pull |
| Right index finger | Sequential paddle | `manual_gear_shift`, `gear` |
| Right little finger | Reverse button, latched | `reverse` |
| Left index finger | Parking-brake switch, latched | `hand_brake` |

```python
from myosuite_carla import CarlaBridge, DriverIntent, build_fullbody_driver

driver = build_fullbody_driver()
bridge = CarlaBridge(driver.mapper, session, driver.config.control_dt)
driver.reset()
bridge.reset(driver.data)
for _ in range(150):
    driver.advance(DriverIntent(right_foot=0.008, steer=-0.1))
    frame = bridge.step(driver.data)
```

`DriverIntent` holds muscle excitations and a push-pull elbow set-point offset,
never vehicle controls. CARLA receives only what the mechanisms measure. The
pelvis is fixed to the seat. Trunk, hips, knees, shoulders, forearm rotation
and wrists are held by joint-equality locks; ankles, toes, elbows and fingers
move only under muscle force. Elbows use a PID stretch reflex on antagonist
muscles, released limbs keep a small antagonist tone, and a pressing finger
co-excites its interossei and lumbrical (Valero-Cuevas et al., 1998).
Mechanisms are placed from the settled body's kinematics and simulated
fingertip paths, each starting clear of its limb.

Reverse and the parking brake are momentary controls that the vehicle latches,
as in production cars: each press toggles them. A press must be held for
`switch_hold` (0.2 s), and reverse also needs the brake pedal pressed; paddle
pulls must last `paddle_hold` (0.1 s). These interlocks reject brushes from the
steering hands. A rigid-capsule fingertip cannot hold a bottomed button under
gravity for more than about 1 to 7 s, so sustained finger holds are outside the
model's validity. `FullBodyDriverConfig.fast()` (2 ms physics, 114 of 416
muscles kept) runs 2.5 times faster with the same discrete controls and
continuous controls within 0.03; see [VALIDATION.md](VALIDATION.md).

`examples/validate_fullbody.py` drives a real vehicle through holding, throttle,
push-pull steering both ways, braking, reverse, three manual gears and parking
brake stops, with 13 physical checks. `--fast` uses the fast preset and
`--record DIR` exports every frame.

## Vehicle loads on the body

The cockpit frame moves with the car, so MuJoCo's own gravity is off and every
body segment instead carries the d'Alembert load
`m (g - a - alpha x r - omega x (omega x r))` from the vehicle's measured
acceleration, rotation and angular acceleration (`vehicle_loads.py`). After each
step call `driver.set_vehicle_motion(VehicleMotion.from_telemetry(telemetry,
previous, dt))`; `seat_offset` places the pelvis relative to CARLA's reference
point. The wheel also receives a self-aligning steering-feel torque of
`steering_feel` N m per m/s^2 of lateral acceleration. Mechanisms carry no
loads. Hard braking at 8 m/s^2 moves the body but presses no control.

## Record and export

`DriveRecorder` collects one row per control period: intent, measured inputs,
sent and applied commands, vehicle kinematics, mechanism measurements, joint
coordinates and muscle activation and force. It is also a `DriveObserver`:

```python
from myosuite_carla import DriveRecorder

recorder = DriveRecorder()
report = run_fullbody_validation("127.0.0.1", observer=recorder)
recorder.recording().save("artifacts/drive", formats=("csv", "opensim", "arm_force", "parquet", "mcap"))
```

| Output | Contents |
| --- | --- |
| `datapackage.json`, `frames.csv`, `coordinates.csv`, `activation.csv`, `force.csv` | Frictionless data package with units and descriptions |
| `opensim/coordinates.mot`, `activation.sto`, `force.sto` | OpenSim tables (coordinates in degrees), MyoSuite's OpenSim names |
| `arm_force/trajectory.csv`, `manifest.json` | arm_force_driving CARLA recording, schema 1, right-positive lateral acceleration, geolocation |
| `frames.parquet`, `drive.mcap` | Wide table; Foxglove-compatible JSON channels |

`Recording.load()` reads a saved recording back; `Recording.concatenate()`
joins runs. Data are exchanged by format only; no arm_force_driving code is used.

## Vehicle surrogate

`SurrogateVehicle` has `CarlaSession`'s interface, so `CarlaBridge` drives it
unchanged. Its single-track model uses drive-acceleration and steering-gain
tables, first-order lags, drag and brake terms. `carla_nissan_patrol()` holds
the parameters fitted to 319 s of designed CARLA 0.10.0 drives. On 526 s of
held-out drives the open-loop speed error is 0.53 m/s, and the position error
3 s ahead is 0.45 m against 6.0 m for constant-velocity extrapolation.
Steering is identified from 2.4 to 13.6 m/s. Collect and fit your own vehicle with:

```shell
python examples/identify_vehicle.py --seeds 0 1 2 --steer-fast 0
python examples/fit_vehicle_model.py --train artifacts/vehicle-sysid/seed0 --test artifacts/vehicle-sysid/seed1
```

## Driving task, scenarios and baseline driver

`FullBodyDrivingEnv` (`driving_env.py`) is an unregistered Gymnasium
environment: actions are the six `DriverIntent` fields; observations are named
(`env.observation_names`): lane geometry, near- and far-point angles, the lead
vehicle, takeover state, measured inputs, driving-joint angles and velocities,
and driving-muscle activations. Rewards are weighted named terms (`TaskConfig`).
The vehicle defaults to the surrogate; pass a started `CarlaSession` with
`reset_vehicle=carla_scenario.teleport` for CARLA.

`VehicleFootprint` in `scenario.py` configures the ego and lead lengths, widths, local bounding box centres and local yaw angles. Shapely computes planar contact between the oriented rectangles. Collision events, termination and penalties use this contact flag. The signed longitudinal bumper gap remains a separate measurement. Lead relevance and time to collision account for lateral overlap and whether the lead is ahead. CARLA runs resolve the rectangles from actor bounding boxes. Recordings retain the resolved geometry and front trajectories.

Scenarios (`scenario.py`) are a lane polyline plus a braking lead vehicle,
raised-cosine lane changes or an automation-to-driver takeover request, with
timestamped events (request, takeover with reaction time, minimal-risk
manoeuvre, collision). `ReflexDriver` (`driver_models.py`) is a baseline built
from two-point visual steering (Salvucci and Gray, 2004), the Intelligent
Driver Model (Treiber et al., 2000), a perception delay and the meta-analytic
2.72 s takeover time (Zhang et al., 2019). It acts only through the muscles.

```shell
python examples/drive_scenario.py --scenario takeover_request --backend surrogate
python examples/drive_scenario.py --scenario lead_vehicle_braking --backend carla --record artifacts/lead
```

`make_driving_env` builds one driver per process for
`gymnasium.vector.AsyncVectorEnv`: on this machine one process runs 226 steps
per second (4.5 times real time) and 8 processes 973 steps per second. The
package does not register environments; registration through
`myosuite.make_env` must follow MyoSuite's CPU/mjlab task contract.

## Train a policy

`training.py` and `training_sb3.py` follow arm_force_driving's conventions
(Stable-Baselines3, `VecNormalize` saved with the policy, a residual on an
expert controller and a judge-based trial reward) in independent code.

- `DrivingTaskWrapper` maps a normalized action in one of three modes:
  `demand` (default) adds a bounded residual to the reflex driver's desired
  acceleration (up to 2 m/s^2) and steer (up to 0.2), which its proprioceptive
  loops and the muscles execute; `intent` adds it to the motor program;
  `direct` controls the motor program. A zero residual reproduces the reflex
  driver exactly, and the residual acts only while that driver is engaged, so
  a policy cannot take over faster than its driver. Actions pass a 0.3 s
  low-pass filter, whose state is observed; without it, 50 Hz residuals made
  car following unhumanly jerky. Each episode draws the
  takeover time from a lognormal with the meta-analytic mean 2.72 s and
  standard deviation 1.45 s (Zhang et al., 2019).
- `ScenarioSampler` draws per episode from curriculum stages: free driving on
  randomly curved roads, car following that replays a recorded human leader,
  a braking lead vehicle, a lane change and a takeover. The surrogate's
  parameters are perturbed by up to 10% per episode.
- `CurriculumCallback` unlocks a stage at 80% recent success.
  `StyleRewardCallback` pays, at the end of each car-following episode, the
  log-odds of a gradient-boosted discriminator refit on human trials against
  the latest agent trials (trial-level GAIL). `behaviour_clone` and
  `collect_expert` warm-start a direct policy, with DAgger rounds.

Human car following comes from NGSIM I-80 (US DOT FHWA, CC BY-SA 3.0,
https://doi.org/10.21949/1504477). The data are downloaded, not committed;
positions are smoothed with a Savitzky-Golay filter, and simulated drives go
through the identical 10 Hz processing. Trials are split by driver, and only
training drivers inform the reward.

Paired evaluation retains `FollowingTrial.source_group` through replay, windowing and NPZ files. Its default grouping identity is the original driver. `discriminability(..., grouping="shared")` holds each source together across human and simulated populations. Reports retain source identities, fold assignments and classifier seeds. Paired confidence intervals resample source means. The independent grouping mode remains available for unrelated populations.

```shell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[rl]"
python examples/download_ngsim.py
python examples/train_policy.py --ngsim artifacts/ngsim/ngsim-i80-cars.csv --timesteps 1000000
python examples/evaluate_policy.py --run artifacts/runs/demand_ppo
```

`evaluate_policy.py` runs every episode twice from the same seed, with the
policy and with the reflex driver, and compares both with held-out human
drivers following the same recorded leaders: gap error against that human,
judge AUC and per-feature overlap.

Three 1 M-step pilots are recorded in [VALIDATION.md](VALIDATION.md#policy-training-pilots).
The third matches the reflex driver's task performance, transfers to CARLA
without collision and is not yet more human-like than the reflex driver.

## Muscle conditions and posture

`FullBodyDriverConfig` reuses MyoSuite's implementations:

| Setting | Effect |
| --- | --- |
| `strength=0.5` | Sarcopenia: every peak force halved (`apply_sarcopenia_to_spec`) |
| `muscle_strength=(("*_l", 0.4),)` | Weakness of muscles matching a name pattern |
| `fatigue=True`, `initial_fatigue`, `fatigue_sex` | 3CC-r `CumulativeFatigue` every physics step |
| `motor_noise_signal`, `motor_noise_constant` | Motor noise (`MotorNoiseCfg`), seeded by `reset(seed=...)` |
| `posture=(("hip_flexion_r", 1.2), ...)` | Seated posture overrides |

Strength and posture change the body before the cockpit is fitted, so
mechanisms are placed for the resulting body. Segment-length scaling of
`myofullbody` is not available.

## Compare with human data

`human_comparison.py` turns windows of steering and acceleration into
amplitude-free steering features and acceleration features, measures how well
a classifier separates two sets of windows (participant-grouped
cross-validated AUC) and gives human-vs-human floors.
`examples/compare_human.py` writes aggregates only and first checks whether the
human signals can support each feature.

## Feedback, resets and cleanup

`info["carla"]` contains `inputs`, the applied `control`, `simulation_time`, and
`vehicle`: CARLA frame, elapsed seconds, location, `(pitch, yaw, roll)` rotation,
velocity, acceleration, angular velocity and scalar speed. Feedback uses the
returned frame's actor snapshot. CARLA world coordinates are left-handed, X
forward/Y right/Z up; rotations and angular velocities are **degrees** and
**degrees/second**. Linear quantities use SI units. Transform feedback explicitly
before applying biomechanical loads.

CARLA 0.10.0 advances `elapsed_seconds` by the single-precision value of the
fixed delta. With 0.02 s, CARLA's clock falls behind MuJoCo's by 4.47e-10 s per
frame, about 80 microseconds per simulated hour. The bridge's per-step checks
allow for this. Align long recordings by CARLA frame instead of comparing the
two absolute clocks.

`reset(seed=...)` resets the MyoSuite driver and re-anchors the bridge without
advancing CARLA. It does not reset the vehicle, traffic, map or CARLA seeds. Reset
your scenario explicitly between experiments; recreate the session after a world
reload. This preserves existing MyoSuite reward/observation semantics and does
not promise deterministic CARLA scenarios from a driver seed alone.

The session borrows its vehicle by default and restores all captured world
settings on close. `own_vehicle=True` opts into actor destruction. Wrapper
`close_session=True` transfers session cleanup to that wrapper. Context managers
are recommended; cleanup failures are reported and failed resources can be
retried with `close()`. Startup/frame/time/RPC/input/hook failures prevent further
healthy stepping and submit zero throttle/full brake. Acknowledging a brake does not
guarantee an immediate physical stop in a paused or unreachable server. A failed
CARLA session must be replaced; a driver/mapper failure requires a successful
reset. Each terminated/truncated driver episode queues braking.

## Mix and match with embodied-ui

[embodied-ui](https://github.com/M-Colley/embodied-ui) is the engine-independent
package for physical interfaces: buttons, switches, knobs, levers,
touchscreens, displays and lamps operated through contact, bridged in
lockstep to Unity, CARLA and other engines. It plugs in three ways
(`pip install -e ../embodied-ui`; needs MuJoCo 3.7 or newer):

- **More devices for the driver.** `FullBodyDriverConfig(interfaces=(dash,))`
  composes embodied-ui interfaces into the driver model through the same
  `ModelBuilder` recipe. Mount devices on any body (for example
  `cockpit_steering_body`) or place them in the seated frame
  (`embodied_ui.MYOSUITE_SEATED`). Their feel acts every physics step;
  `driver.interface_reading()` and the environment's `info["interface"]`
  report their channels and events.
  `myosuite_carla.interfaces.interface_hook(driver, routes)` is a
  `CarlaBridge` hook. It routes channels to CARLA's lights, including
  `lights.hazard`, and the reported light state back to lamps.
- **Unity vehicles.** `EngineVehicleSession(engine)` implements the vehicle
  session protocol on any embodied-ui engine, for example a Unity car through
  `RemoteEngine` with the Unity package's `EmbodiedVehicleOutput`. It converts
  that engine's telemetry to CARLA's frame, so `CarlaBridge`,
  `FullBodyDrivingEnv` and the scenarios run unchanged.
- **Any body, these vehicles.** `VehicleSessionEngine(session)` exposes a
  `CarlaSession` or the fitted `SurrogateVehicle` as an embodied-ui engine.
  Any MuJoCo body with embodied-ui pedals and wheels can drive it.

```shell
python examples/drive_with_interface.py --backend surrogate
python examples/drive_with_interface.py --backend carla
python examples/drive_with_interface.py --backend unity --port 47474
```

The full-body driver's scripted motor programs operate only its own controls.
The example flips the hazard switch by hand, and operating new devices
through the muscles is future work. The live CARLA run is recorded in
[VALIDATION.md](VALIDATION.md#mix-and-match-with-embodied-ui). The Unity
backend has not been run, because no Unity Editor was available.

## Record an attentive human driver

The recording example runs `AttentiveDriver` from a specified local `embodied-human` checkout. The video combines a synchronized CARLA chase camera, the MuJoCo body, measured limb controls, model gaze, and the SuRT visual search task. Road perception uses the human project's parametric geometry interface. The head remains posture constrained. Each recording includes source hashes, versions, per frame telemetry, applied controls, collision sensor contacts, and cleanup results in a JSON file beside the video.

Install MyoSuite from `dev`, the matching CARLA client, and these optional dependencies. Start a rendering CARLA server with no other synchronous client. The default spawn index assumes `Town10HD_Opt`.

```shell
python -m pip install -e ".[video]" -e ../embodied-human
python examples/record_embodied_human.py --human-repo ../embodied-human --fast --scenario lane_change --seconds 12 --seed 0 --output artifacts/lane-change.mp4 --record artifacts/lane-change
python examples/record_embodied_human.py --human-repo ../embodied-human --fast --scenario lead_brake --seconds 12 --seed 0 --output artifacts/lead-brake.mp4 --record artifacts/lead-brake
```

The fast configuration retains 114 muscles and applies vehicle loads to the body. The two episodes evaluate a requested lateral displacement and a braking lead vehicle. The lead vehicle follows a scripted trajectory. These seeded integration demonstrations establish execution and physical coupling. Human driving validity requires comparison with measured human data. The actual CARLA 0.10.0 recording results are documented in [VALIDATION.md](VALIDATION.md).

## Empirical following calibration

`empirical_following.FollowingProtocol` compares the IDM acceleration component with recorded NGSIM states. Its observations are acceleration errors relative to the measurements. It preserves each original vehicle source through fitting and evaluation. The companion `closed_loop_following` function replays recorded leaders through `embodied-human.AttentiveDriver`, one reused MyoSuite body, and the fitted offline vehicle surrogate. It reports speed and bumper gap errors, collisions, road departures, and termination for baseline and fitted configurations.

The [embodied human campaign](https://github.com/M-Colley/embodied-human) connects these protocols to the other empirical sources. Vehicle tracks provide the NGSIM grouping unit. A component fit requires a separate check of its transfer to closed loop behavior. The reports preserve this distinction and retain failed episodes. [VALIDATION.md](VALIDATION.md) contains the actual commands and outcomes.

`ClosedLoopFollowingProtocol` supplies a physical calibration objective. Every candidate drives the attentive human, MyoSuite muscles, and fitted vehicle surrogate. The context manager reuses one body. Its episode cache binds the full IDM settings, source window, and simulation seed. Source seeds remain stable across request ordering and batching. Incomplete or unsafe episodes make the objective infeasible. `FollowingProtocol` accepts explicit source exclusions before selecting new source groups. The companion embodied human example `calibrate_closed_loop_following.py` fixes the partitions and acceptance criterion before fitting and retains every executed episode.

`ReflexDriverConfig.acceleration_control` enables a speed dependent inverse of the fitted vehicle model. `AccelerationControlConfig` calculates continuous throttle and brake targets and closes separate feedback loops on the measured pedals. Foot muscle excitations execute those targets. The optional configuration preserves the legacy controller when unset. `ReflexDriverConfig.from_dict` restores the complete nested vehicle and controller configuration for empirical profiles and video replay.

Vehicle calibration, profile restoration, longitudinal telemetry and realized perception delay follow the [acceleration control contract](COMPATIBILITY.md#acceleration-control-and-empirical-profiles).

The companion `evaluate_actuation_improvement.py` in embodied human compares three fixed configurations. These combine legacy actuation with default IDM, corrected actuation with default IDM, and corrected actuation with the previously fitted IDM. Development selection uses 32 previously inspected source tracks. The selected correction receives one independent test on 24 fresh tracks with two matched simulation seeds. Controller configurations are fixed within each physical protocol and included in cache identity.

The completed comparison reduced mean normalized trajectory error from 0.7686 to 0.6987. Its paired source change was −0.0699 (95% bootstrap CI [−0.1655, 0.0212]). Speed and acceleration RMSE decreased, while gap RMSE increased. The correction remains experimental and package defaults remain unchanged. [The retained evidence](evidence/actuation-improvement-2026-10-28/summary.json) includes all 192 development and test episodes, frozen source windows, complete configurations, and verified replay metadata in `study-evidence.zip`.

The following command records the first two fresh test sources with matched seed 117. The paired video places legacy actuation on the left and corrected actuation on the right. The complete controller configuration comes from each saved episode.

```shell
python examples/record_empirical_trials.py --source artifacts/ngsim/ngsim-i80-cars.csv --report artifacts/actuation-improvement-2026-10-28/run/fullreport.json --out artifacts/actuation-improvement-videos --groups ngsim:1136:2 ngsim:1193:2 --base-seed 117
```

The empirical recorder renders the retained test trials with the actual MyoSuite body and measured controls. Each paired video places the baseline on the left and the fitted candidate on the right. The road schematics show recorded trajectory references and simulated bumper gaps. Videos use the original model configurations and simulation seeds. Every replay must match the saved trajectory samples and RMSE values within an absolute tolerance of `1e-8` before its clip is published. The recorder checks source hashes and resumes previously verified clips.

```shell
python examples/record_empirical_trials.py --source artifacts/ngsim/ngsim-i80-cars.csv --report artifacts/closed-loop-calibration-2026-10-28/run/fullreport.json --out artifacts/empirical-trial-videos
```

The retained six test trials produce six paired 20 second clips and `held-out-trials.mp4`, a two minute montage at 25 fps. Install the `video` extra and use the matching embodied human and MyoSuite development sources recorded in the calibration provenance. These recordings depict offline replays through the fitted vehicle surrogate.

## Test

Run the offline contact demonstration and tests from the repository root:

```shell
python examples/offline_cockpit.py
python -m pytest tests -q
```

The default suite uses real MuJoCo physics and CARLA API doubles. It exercises
contact forces, physical pedal/wheel motion and spring return, real MyoSuite
muscle actuation and model composition, calibration, timing, Gymnasium integration
and failure cleanup. See [VALIDATION.md](VALIDATION.md) for measured results and
the dependency regression failures reproduced on the unchanged MyoSuite base.
The offline demo uses a scripted contact probe for a deterministic mechanism
check, not a trained driver policy.

Actual CARLA validation requires the matching client wheel and an already
running UE5.8 server with no other synchronous owner. Run the acceptance CLI:

```shell
python examples/validate_carla.py --host 127.0.0.1 --port 2000 --build-ref ue58-dev --build-commit ACTUAL_COMMIT_SHA --output artifacts/carla-live.json
```

Use the server build's actual commit. The runner uses real MyoSuite muscle
excitation and physical contact to move the throttle, brake and steering
mechanisms, then checks actual CARLA acceleration/displacement, deceleration and
turning. It also checks applied controls, frame timing, actor cleanup and world
restoration. Separate trials use a real MyoSuite finger fitted to each control;
these are acceptance fixtures, not a trained full-body driver. Scalar setup
commands settle/accelerate the vehicle, and the steering trial holds scalar
cruising throttle. The tested control comes from muscle/contact motion.

The runner discovers an available four-wheel vehicle, preferring the Nissan
Patrol; `--blueprint` selects an explicit vehicle. For supplementary testing of
the published UE5.5 release, use `--target-ref 0.10.0 --build-ref 0.10.0` and a
separate output file. Such a pass does not establish UE5.8 compatibility.

The JSON retains build declarations, client/server versions and measured
telemetry on success or failure. A missing `carla` module writes a **failed**
report and exits nonzero. RPC version equality does not independently prove the
server's source branch; keep build provenance alongside the report. See
[RUNTIME_STATUS.md](RUNTIME_STATUS.md) for the downloaded source/runtime and
UE5.8 build constraints, and [VALIDATION.md](VALIDATION.md) for actual results.

The same checks are available through explicitly enabled pytest. PowerShell:

```powershell
$env:CARLA_BRIDGE_LIVE = "1"
$env:CARLA_BUILD_REF = "ue58-dev"
$env:CARLA_BUILD_COMMIT = "ACTUAL_COMMIT_SHA"
$env:CARLA_LIVE_OUTPUT = "artifacts/carla-live.json"
python -m pytest tests/test_live_server.py -v
```

On POSIX set the equivalent environment variables before running pytest.
Optional `CARLA_HOST` and `CARLA_PORT` select the server; `CARLA_TARGET_REF`
selects the report's validation target (default `ue58-dev`). The test spawns and
destroys its own vehicles and requires an unoccupied straight road spawn with
35 m of road ahead. Offline success does not establish live-server validation.
With `CARLA_BRIDGE_LIVE=1`, `tests/test_live_fullbody.py` also runs the
full-body drive and writes `CARLA_FULLBODY_OUTPUT`
(default `artifacts/carla-fullbody.json`).

The [CI workflow](.github/workflows/ci.yml) installs MyoSuite `dev` in a separate
checkout and runs offline CPU tests on Linux and Windows, including the minimum
MuJoCo version. Its manual `live_carla` option runs the live tests on a
`self-hosted, carla-ue58` runner. Configure that runner's Python with MyoSuite
`dev`, the matching CARLA wheel, and an already running exclusive server before
enabling the option. Set `CARLA_HOST`/`CARLA_PORT` in the runner environment for
a remote server and `CARLA_BUILD_REF`/`CARLA_BUILD_COMMIT` for its actual build.
The job uploads measured JSON and JUnit results even if validation fails.
Hosted offline jobs cannot establish a live CARLA pass.

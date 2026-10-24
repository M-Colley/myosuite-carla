# MyoSuite–CARLA

Mark Colley's [`myosuite-carla`](https://github.com/M-Colley/myosuite-carla) is a
standalone installable package connecting a physical MuJoCo
driver to a CARLA **Unreal Engine 5.8 `ue58-dev`** vehicle. Muscles move the limbs;
limbs contact passive pedals and a steering wheel; measured mechanism travel or
contact force becomes `carla.VehicleControl`. CARLA simulates the vehicle.

**Validation status:** with a real CARLA 0.10.0 / UE5.5 server, 262 tests pass
with zero skips (97% statement coverage). One seated MyoSuite full body
operates throttle, brake, push-pull steering, reverse, sequential gears and the
parking brake in a single live drive, with all 12 physical checks passing. The
single-finger acceptance fixtures pass on two Windows PCs, all 11 four-wheel
vehicles, a rendered server and a 15,000-frame soak. Actual **UE5.8**
validation remains pending a runnable build; see
[measured results](VALIDATION.md) and [runtime status](RUNTIME_STATUS.md).

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
python -m pip install -e ".[test]"
python -m pip install /path/to/your/ue58-build/PythonAPI/carla/dist/<matching-wheel>.whl
```

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
| Right little finger | Reverse button, held | `reverse` |
| Left index finger | Parking-brake button, held | `hand_brake` |

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
pelvis is fixed to the seat and gravity is disabled. Trunk, hips, knees,
shoulders, forearm rotation and wrists are held by joint-equality locks; ankles,
toes, elbows and fingers move only under muscle force. Elbows use a PID stretch
reflex on antagonist muscles, and released limbs keep a small antagonist tone.
Mechanisms are placed from the settled body's kinematics and simulated
fingertip paths, each starting clear of its limb. This is a scripted motor
program for validation and demonstration, not a trained driving policy.

`examples/validate_fullbody.py` drives a real vehicle through holding, throttle,
push-pull steering both ways, braking, reverse, three manual gears and parking
brake stops, with 12 physical checks.

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

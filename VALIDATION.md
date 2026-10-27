# Validation record

## Integration review on 2026-10-27

The review used Windows, CPython 3.12.13, MuJoCo 3.17.0, NumPy 2.5.4, myo-sim 0.2.3 and Stable-Baselines3 2.9.0. MyoSuite was loaded from current `dev` commit `230f198458b501c423a0892b78da60419ffc49a3` with the fatigue reset fix in commit `37c8b863fe448f8fff2b29eb62a53a269b79d16f` ([upstream PR 575](https://github.com/MyoHub/myosuite/pull/575)). The sibling revisions were embodied-human `9d38317ec9f0e07c5a393212ab5d8fae210f5ac8` and embodied-ui `605862bd551401bfce91afb4afcf280b2bef98fd`.

The final bridge suite reports **372 passed and 4 live tests skipped** ([output](evidence/review-2026-10-27-bridge.txt), [JUnit](evidence/review-2026-10-27-bridge.xml)). The embodied human driving suite reports **5 passed** ([output](evidence/review-2026-10-27-human.txt), [JUnit](evidence/review-2026-10-27-human.xml)). Ruff, formatting, mypy and both distribution builds pass. The [environment record](evidence/review-2026-10-27-summary.json) includes dependency versions, source hashes and sibling revisions. CI now runs the bridge with the optional interface and training dependencies alongside these pinned embodied human and embodied UI revisions.

The fixes refresh derived MuJoCo state after each driver period, preserve button hold duration across repeated reads, reject malformed actions and invalid task distances, reset vehicle adapters, align lead placement with reported gap, support custom surrogate knots and TD3 cloning, preserve the requested style window and use the actual reflex evaluation baseline. Both live examples now release resources after setup failures. Lane sampling rejects traversal distances that would prevent progress.

The upstream fatigue regression mutated caller supplied arrays during stepping. It also failed on readonly arrays and changed repeated seeded episodes. All three regression cases pass after copying the reset vector. MyoSuite's focused fatigue, wrapper, reset and noise suites report 270 passed, 10 skipped and 6 expected failures. All applicable precommit hooks pass. Its required model builder, CPU terms, fragment, parity and registry suite reports 312 passed, 2 skipped, 4 expected failures and 4 identical parity failures before and after the fix. The existing failures are `myoChallengeBaodingP2-v1`, `myoChallengeRelocateP1-v0`, `myoHandReorient8-v0` and `myoLegRoughTerrainWalk-v0`.

The local commands use a separate MyoSuite worktree. Its source takes precedence over the existing editable installation:

```powershell
$env:PYTHONPATH = (Resolve-Path '../myosuite-dev-review').Path
.venv/Scripts/python.exe -m pytest tests -q --junitxml=test-results/review-dev-suite.xml
.venv/Scripts/python.exe -m pytest ../embodied-human/tests/test_driving.py -q --junitxml=test-results/review-embodied-human.xml
.venv/Scripts/python.exe -m ruff check src tests examples
.venv/Scripts/python.exe -m ruff format --check src tests examples
.venv/Scripts/python.exe -m mypy src/myosuite_carla --ignore-missing-imports --follow-imports=skip
.venv/Scripts/python.exe -m build
```

Five focused tests pass with the minimum MuJoCo 3.6.0. They cover derived state consistency in both driver presets, pedal isolation and steering in both directions. No live CARLA server was started for this review. The four live tests remain disabled. UE5.8 runtime validation remains pending.

Three scientific improvements remain. First, collision detection uses longitudinal gap alone. Adjacent lane passing therefore needs explicit vehicle footprints and lane occupancy. Second, embodied-human's `BodySensing.head_motion` omits rotational acceleration terms at the head's position. Its vestibular inputs need a shared rigid body kinematics calculation. Third, paired human and simulated trials need shared source identifiers during grouped evaluation. This prevents the same driving context from entering both training and test partitions.

## Earlier validation

Validated locally on Windows on 2026-10-23, with the researcher features below
validated on 2026-10-24. The CARLA source
contract is `ue58-dev` commit `defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`;
see [COMPATIBILITY.md](COMPATIBILITY.md).

This record accompanies Mark Colley's standalone `myosuite-carla` repository.
The initial offline results below were obtained before extraction from the
MyoSuite worktree; the commands use the equivalent standalone root layout.

**Recorded live result: 312 passed, no skips, 94% statement coverage** against an
actual CARLA **0.10.0 / Unreal Engine 5.5** server (CPython 3.12.13, NumPy
2.5.4, MuJoCo 3.17.0); see
[researcher features](#researcher-features-loads-recording-surrogate-task-and-conditions).
The previous result was **262 passed, no skips, 97% statement coverage** on the
second PC (NumPy 1.26.4, the package's `cp312-win_amd64` wheel).
It includes a live drive in which **one MyoSuite full body operates every
control**; see [full-body drive](#one-myosuite-full-body-operates-every-control).
The earlier first-PC result was **243 passed, no skips, 97% statement coverage**
with Python 3.12.10. This is supplementary live validation; **UE5.8 runtime
compatibility is still unproven**.
See [current JUnit results](evidence/deep-full-suite.xml),
[extended live checks](evidence/deep-live-session.json), and
[runtime acquisition](RUNTIME_STATUS.md). The same live result was reproduced
on a **second physical Windows PC**; see
[second-PC reproduction](#second-physical-pc-reproduction).

The final offline suite also passes **240 tests, 3 live tests skipped** on the
minimum supported MuJoCo 3.6.0; see
[minimum-version results](evidence/deep-minimum.xml). With the full-body driver,
the second PC's offline suite passes **258 tests, 4 live tests skipped** on
both MuJoCo 3.17.0 and the minimum 3.6.0 with NumPy 2.5.4
([3.6.0 JUnit](evidence/ue55-min-mujoco-3.6.xml)). Checking 3.6.0 found two
full-body incompatibilities, both now fixed: joint-equality data has 11 values
rather than 12, and `MjSpec.body(name)` misses bodies in attached specs. Ruff lint and
format checks, mypy and wheel/sdist builds pass.

Before the live-delivery fixes, the standalone suite passed **142 tests** under
both MuJoCo 3.6.0 and 3.16.0 with the live test skipped and **87%** coverage. See the saved
[offline test output](evidence/offline-mujoco-3.16.txt).
The [MuJoCo 3.6 output](evidence/offline-mujoco-3.6.txt) records the minimum-version run.
Focused input and bridge tests also pass on local Python 3.14.6. A regression
test reproduces and guards against the NumPy/pybind enum equality failure found
by Linux Python 3.14 CI; joint-type validation now compares builtin integers.

## Researcher features: loads, recording, surrogate, task and conditions

Validated on 2026-10-24 on the first PC against the downloaded **CARLA 0.10.0
/ UE5.5** package started with `-nullrhi -nosound -unattended -notrace
-carla-rpc-port=2000` (Town10HD_Opt, `vehicle.nissan.patrol`). Installing
`opensim==4.6`, `mcap` and `pyarrow` upgraded NumPy from 1.26.4 to 2.5.4;
`pip check` reported no conflicts and the suite passed before and after. An
unrelated TD-MPC2 training job ran on the same machine throughout, so timings
are conservative.

**Live full-body drive with vehicle loads.** All **13 checks** pass for the
default configuration and for `--fast`, and again inside the live suite
([compact evidence](evidence/ue55-fullbody-loads.json), every tenth frame). The
new check is that a second switch press releases the parking brake. Every run
restored world settings and actors; CARLA's gearbox reported reverse and gear
changes at most 3 and 2 frames late. Commands:

```powershell
python examples/validate_fullbody.py --host 127.0.0.1 --timeout 30 --target-ref 0.10.0 --build-ref 0.10.0 --output artifacts/ue55-fullbody-loads.json --record artifacts/ue55-fullbody-loads --formats csv,opensim,arm_force,parquet,mcap
python examples/validate_fullbody.py --host 127.0.0.1 --timeout 30 --target-ref 0.10.0 --build-ref 0.10.0 --fast --output artifacts/ue55-fullbody-fast.json
```

**Live suite.** With the environment variables of the earlier live runs
(`CARLA_BRIDGE_LIVE=1`, host `127.0.0.1`, target and build ref `0.10.0`),
`python -m pytest tests --cov=myosuite_carla` passed **312 tests with zero
skips and 94% statement coverage**
([JUnit](evidence/ue55-loads-live-suite.xml), [output](evidence/ue55-loads-live-suite.txt)).
`carla_scenario.py` is exercised only by the live scenario runs below. On the
minimum MuJoCo **3.6.0** (`.venv-min`, NumPy 2.5.4) the offline suite passed
**306 tests, 6 skipped**: the 4 live tests and the Parquet/MCAP and OpenSim
round trips, whose optional packages are not installed there
([JUnit](evidence/ue55-loads-mujoco-3.6.xml)).
Ruff lint and format, mypy and wheel/sdist builds pass.

**Recording interoperability on the live drive.** The 1,149-frame recording
was read back with each format's own reader: OpenSim 4.6 `TimeSeriesTable`
(122 coordinates in degrees, 416 muscle forces equal to the CSV), pandas
Parquet (1,149 x 1,020), the `mcap` reader (four topics of 1,149 messages) and
the arm_force_driving manifest (schema 1, its eight columns, CARLA geolocation).
File sizes and SHA-256 are in the evidence.

**Vehicle surrogate.** `examples/identify_vehicle.py` drove 97 designed
segments (launches, pedal and hand-brake stops, reverse, steering sweeps at
2-10 and 8-14 m/s) from rest on straight roads; 8 steering segments with a
collision or curb strike were discarded. `examples/fit_vehicle_model.py`
fitted seeds 0 and 3 (319 s) and evaluated open-loop rollouts on seeds 1, 2 and
4 (526 s), compared with constant-velocity extrapolation
([report](evidence/ue55-vehicle-surrogate.json), [data](evidence/ue55-vehicle-sysid.zip)):

| Held-out set | Speed RMSE | Yaw-rate RMSE (peak) | Position error at 3 s | Constant velocity at 3 s |
| --- | --- | --- | --- | --- |
| All 60 segments | 0.53 m/s | 0.76 deg/s (5.6) | 0.45 m | 5.99 m |
| Launches | 0.74 m/s | 0.04 deg/s | 0.52 m | 3.51 m |
| Steering 2-10 m/s | 0.29 m/s | 2.23 deg/s (18.1) | 0.28 m | 8.09 m |
| Steering 8-14 m/s | 0.51 m/s | 0.69 deg/s (2.9) | 0.62 m | 10.37 m |
| Reverse | 0.04 m/s | 0.00 deg/s | 0.06 m | 0.86 m |

Steering is identified from 2.4 to 13.6 m/s (5th-95th percentile) and reverse
up to throttle 0.53; table knots without data hold the nearest identified
value. On the muscle-driven CARLA drives (scenarios and every full-body phase)
the 3 s error is 0.20 m against 1.74 m for the lane change and 0.19 m against
0.51 m for the takeover. It is worse than the baseline (1.27 m against 0.22 m)
for lead-vehicle braking, which starts at a steady 12 m/s right after CARLA's
teleport-to-speed transient. Launches from rest at light throttle are the
weakest case (full-body `throttle` phase: 0.59 m/s RMSE at 1.1 m/s peak),
because CARLA engages the drive with a delay below 1 m/s.

**Scenarios on both backends.** The reflex driver drove each scenario through
the muscles on the surrogate and on CARLA (spawn 105, two same-direction
lanes, 170 m straight), with no collision and no off-road event
([evidence](evidence/ue55-scenarios.json)):

| Scenario | Metric | CARLA | Surrogate |
| --- | --- | --- | --- |
| Lead vehicle brakes 12 to 0 m/s at 4 m/s^2 | Minimum gap / TTC | 1.91 m / 2.71 s | 2.08 m / 2.69 s |
| Takeover request at 5 s | Driver takeover time | 3.00 s | 3.00 s |
| Lane change, 3.5 m left | Lateral error RMS / maximum | 0.53 / 1.42 m | 0.52 / 1.38 m |

Two scenario defects were found only live and fixed: a lane change at a spawn
with no drivable left lane mounted the curb (the script now refuses such
spawns), and CARLA reports the teleport-to-speed velocity jump as a 594 m/s^2
one-frame acceleration, which the body loads turned into a 1,000 N reflex
event; `teleport` now ticks once more before the episode starts.

**Defects found while adding vehicle loads.**

- Under gravity a fingertip pressing a bottomed button rolls off it after 1 to
  7 s; neither intrinsic co-activation nor friction removed this, so reverse
  and the parking brake became latched switches pressed briefly, as in
  production cars.
- Large steering motions pushed the left index into the parking-brake switch,
  and in the fast preset the right index into the gear paddle, so CARLA
  selected manual gear 1 early and refused reverse. Switches now need a 0.2 s
  hold (reverse also the brake pedal) and paddle pulls a 0.1 s hold.
- The reflex driver's lane keeping keeps a 0.57 Hz steering oscillation of
  0.0027 normalized steer (2.8 cm lane-position SD); it is negligible
  physically but present without any noise.

**Performance.** The fast preset keeps 114 of 416 muscles and steps at 2 ms
([parity](evidence/fullbody-parity.json), 400 control periods):

| Variant | Real-time factor | Speed-up | Largest continuous control difference |
| --- | --- | --- | --- |
| 416 muscles, 1 ms (reference) | 2.33 | 1.00 | - |
| 114 muscles, 1 ms | 2.94 | 1.26 | 0.029 steer |
| 416 muscles, 2 ms | 4.61 | 1.98 | 0.020 brake |
| 114 muscles, 2 ms (`fast()`) | 5.76 | 2.47 | 0.030 steer |

Boolean and gear channels are identical in every variant. Converting the
vehicle motion once per control period and precomputing excitation indices
raised the reference from 1.93 to 2.33 times real time with identical results.
Batched rollouts of `make_driving_env` in `AsyncVectorEnv` reach 226 steps per
second in one process and 380, 774 and 973 with 2, 4 and 8 workers
([scaling](evidence/vector-rollout.json)).

**Human data.** `examples/compare_human.py` read
`combined_eye_and_driving_df.csv` (10 participants, 3 drives each) and wrote
aggregates only ([report](evidence/human-comparison.json)). The data cannot
support a control comparison: logged steering has 174 distinct values, and in
98.3% of 10 s lane-keeping windows steering and acceleration are constant;
the front-vehicle distance is held for long stretches; human speeds are 20-36
m/s (5-95%), above the surrogate's identified range. The script therefore
reports no feature-based AUC. A first run without the adequacy check gave a
model-vs-human AUC of 0.995 that would have been misleading. The toolkit's
features and AUCs are verified on synthetic data in `tests/test_human_comparison.py`.

## Mix and match with embodied-ui

Run on 2026-10-24 on the first PC with embodied-ui 0.1.0 (commit `aa08385`),
MuJoCo 3.17.0, Python 3.12.13 and the CARLA 0.10.0 / UE5.5 package started
with `-nullrhi -nosound -unattended -notrace -carla-rpc-port=2000`.

**Offline.** `tests/test_interfaces.py` (6 tests) passes:

- Unity-convention telemetry converts to CARLA's frame: location, velocity,
  acceleration, yaw, and a left turn's negative yaw rate.
- The vehicle session detects clock slips.
- A servo fingertip on an embodied-ui pedal drives the fitted surrogate
  through `VehicleSessionEngine`.
- `interface_hook` sets both blinkers for `lights.hazard` and feeds the light
  state back to a lamp.
- Devices compose into the fast full-body driver, mounted on
  `cockpit_steering_body`, and are read once per period.
- The driver's only applied forces stay the wheel feel and the devices' own
  degrees of freedom.

The full offline suite passes: 332 passed and 4 live tests skipped.

**Live.** `examples/drive_with_interface.py --backend carla --seconds 6`
([report](evidence/ue55-interface-hazard.json)) ran as follows:

- **Setup:** the full-body driver (fast preset) carried an embodied-ui hazard
  toggle switch and a lamp on its steering-wheel hub. It drove Town10HD_Opt
  spawn 105 in `vehicle.nissan.patrol` with the reflex driver through its
  muscles, for 300 periods in 3.18 s
  wall time.
- **The switch:** it was flipped by setting its joint at 2.0 s, because no
  motor program reaches it. Its debounced state read 1 from
  2.02 s.
- **CARLA's lights:** the bridge hook set the hazard lights. CARLA reported
  light state 48 (left and right blinkers) at the end, and the lamp showed
  its on colour from that reported state.
- **Driving:** the vehicle reached 8.143 m/s with no
  collision or off-road termination.

## Policy training pilots

Run on 2026-10-24 on the first PC with Stable-Baselines3 2.9.0, PyTorch
2.14.1 (CPU build), Gymnasium 1.2.3 and NumPy 2.5.4, while an unrelated
TD-MPC2 job used the GPU.

**Human data.** `examples/download_ngsim.py` downloaded the NGSIM I-80 car
trajectories (US DOT FHWA, CC BY-SA 3.0, https://doi.org/10.21949/1504477;
4,377,903 rows, query and SHA-256 in
[the aggregate summary](evidence/ngsim-i80-following-summary.json)). The data
stay in `artifacts/`. Smoothed with a Savitzky-Golay filter, they give 8,878
car-following trials of 20 s from 4,346 drivers (speeds 1.7-10.8 m/s, 5th-95th
percentile; median time headway 1.98 s), split 70/30 by driver. Simulated
drives pass through the identical 10 Hz processing. Pilot 1 used an earlier
time grid that dropped the last sample of some windows (8,891 trials); its
held-out bank therefore differs slightly from pilots 2 and 3.

**Setup common to all pilots.** `examples/train_policy.py` trained PPO for
1,001,472 steps with 6 workers in about 30 minutes each. The policy adds a
residual to the reflex driver's desired acceleration (up to 2 m/s^2) and steer
(up to 0.2), executed by its proprioceptive loops and the muscles of the fast
preset on the vehicle surrogate. Other settings:

- curriculum over curved free roads, NGSIM-leader following, lead braking,
  lane changes and takeovers; every stage was unlocked by 168,000 steps;
- surrogate parameters perturbed by up to 10% per episode;
- per-episode takeover times from the meta-analytic lognormal;
- the trial-level style reward from training drivers only.

`examples/evaluate_policy.py` ran 10 paired episodes per stage, with policy
and reflex driver on the same seed, plus 120 held-out NGSIM drivers.

**What changed between pilots.**

- Pilot 1: its car-following speed term aimed at the 16 m/s free-road target,
  and the style reward (weight 0.5) came once per 20 s episode, nearly
  discounted away at gamma 0.995.
- Pilot 2: the desired speed behind a lead became `min(target, (gap - 2 m) /
  1.6 s)`, as in the Intelligent Driver Model equilibrium, and the style
  reward is paid every 5 s with weight 2.
- Pilot 3: additionally, the action passes a first-order low-pass filter
  (0.3 s), with the filter state in the observation.

All three pilots completed every evaluation episode without collision or
leaving the road. Task-return difference, policy minus reflex driver, with
paired-bootstrap 95% intervals (evaluated under pilot 2's corrected reward;
training penalties excluded):

| Stage | Pilot 1 | Pilot 2 | Pilot 3 |
| --- | --- | --- | --- |
| Free road, curved | -22.3 [-25.4, -19.3] | -7.1 [-9.2, -4.0] | -0.1 [-5.2, +3.1] |
| Following a random leader | -9.1 [-15.7, -0.8] | -43.1 [-50.2, -36.0] | +0.5 [-3.1, +4.3] |
| Lead brakes | -13.5 [-15.5, -10.2] | -39.4 [-46.2, -32.1] | +1.3 [-1.2, +3.1] |
| Lane change | -2.2 [-12.9, +7.3] | -4.9 [-10.8, +5.1] | +3.6 [-1.6, +10.6] |
| Takeover | -11.5 [-13.3, -9.8] | -6.7 [-8.0, -5.0] | +3.6 [+2.0, +5.7] |
| Held-out NGSIM leaders | -10.9 [-12.3, -9.6] | -22.7 [-27.3, -19.3] | -1.6 [-4.8, +0.2] |

Human-likeness on 120 held-out drivers following their own recorded leaders
(policy / reflex driver):

| Run | Judge AUC | Human floor | Features with overlap >= 0.60 | Gap RMSE vs. the human | Gap RMSE difference [95% CI] |
| --- | --- | --- | --- | --- | --- |
| Pilot 1 | 0.983 / 0.957 | 0.491 | 3 / 4 of 8 | 4.20 / 3.96 m | +0.25 [+0.16, +0.35] m |
| Pilot 2 | 0.985 / 0.932 | 0.455 | 2 / 5 of 8 | 4.71 / 3.17 m | +1.54 [+1.15, +1.95] m |
| Pilot 3 | 0.937 / 0.932 | 0.455 | 6 / 5 of 8 | 3.47 / 3.17 m | +0.30 [+0.15, +0.55] m |

Findings:

- **Pilots 1 and 2 were worse than the reflex driver.** The unfiltered 50 Hz
  residual made car following jerky: acceleration SD 1.21 and 1.17 m/s^2
  against 0.76 for humans. The style reward's log-odds stayed between -2 and
  -4 throughout pilot 2.
- **Pilot 3 matches the reflex driver.** Low-pass filtering brought the
  acceleration SD to 0.75 m/s^2. Task return does not differ significantly
  from the reflex driver's in five stages, is better on takeovers, and lane
  error falls in four of six.
- **Human-likeness did not improve beyond the reflex driver.** Judge AUC is
  0.94 against a 0.46 floor, and the gap error against the matched human is
  0.30 m larger.
- **Limits of the evidence.** These are single-seed pilots of 1 M steps;
  claims about learned human-likeness need longer training, several seeds and
  a reward-weight study.

**Transfer to CARLA.** With the CARLA 0.10.0 / UE5.5 server on `-nullrhi`,
`examples/drive_scenario.py --policy artifacts/runs/demand_ppo_pilot3 --fast`
drove the three scenarios on Town10HD_Opt spawn 105 through the muscles, with
the reflex driver on the same preset for comparison
([evidence](evidence/ue55-policy-scenarios.json)). There was no collision and
no off-road event, and the world was restored each time:

| Scenario | Metric | Policy | Reflex driver |
| --- | --- | --- | --- |
| Lead brakes 12 to 0 m/s | Minimum gap / TTC | 2.23 m / 1.95 s | 1.91 m / 2.71 s |
| Takeover request at 5 s | Takeover time | 3.00 s | 3.00 s |
| Lane change 3.5 m left | Lateral error RMS / maximum | 0.50 / 1.34 m | 0.54 / 1.41 m |

Learning curves, curricula and full evaluations:
[pilot 1](evidence/training-pilot1.json), [pilot 2](evidence/training-pilot2.json),
[pilot 3](evidence/training-pilot3.json). Model weights stay in `artifacts/`.

With the training code the offline suite passes **326 tests, 4 live tests
skipped** (MuJoCo 3.17) and **319 tests, 7 skipped** on MuJoCo 3.6.0, where
Stable-Baselines3 and the optional exporters are not installed. Ruff, mypy and
the wheel/sdist build pass. The live suite was not re-run after the training
code was added; its 312-test result above predates it, and its live tests do
not exercise the training modules.

## Deeper review, fault testing and performance

The same fresh Python 3.12 environment was used for a further independent audit.
The full live-enabled suite passed **243 tests with zero skips and 97% coverage**
in 65.07 seconds. The minimum-version environment (Python 3.13, MuJoCo 3.6.0)
passed **240 tests**, with the three explicit live-server tests skipped.
Ruff lint/format, source mypy and wheel/sdist builds also pass.

Confirmed fixes include:

- An inherited MuJoCo joint margin generated **46.1% brake input at rest**,
  without muscle excitation/contact. Cockpit joints now explicitly use zero
  margin, preserving the host model's own joint settings.
- NumPy integer quaternion arithmetic could overflow and accept a non-unit
  quaternion. Configuration values now become finite builtin floats and
  immutable tuples before arithmetic; later caller mutations cannot alter them.
- Bound input readers now reject data from a different compiled model before
  accessing arrays or native contact-force functions. Contact sums also reject
  overflow, even when individual contact values are finite.
- Nonfinite CARLA feedback is rejected before controls/ticking. Failed public
  snapshot reads brake and invalidate the session. Initial world-RPC failures
  release only explicitly owned vehicles and preserve the original error.
- Boolean ownership/ticking flags and CARLA's signed 32-bit substep field are
  checked before use. Failed reset feedback export leaves the wrapper guarded;
  cleanup preserves driver failures while retaining session cleanup errors.
- Telemetry vectors are detached immutable tuples. Primitive serializers avoid
  recursive dataclass copying, and speed uses overflow-resistant `math.hypot`.

The [extended actual-server report](evidence/deep-live-session.json) records
**1,200 consecutive real MyoSuite muscle/contact frames**, or 24 simulated
seconds, twice during this review. Maximum accumulated clock drift was
**0.536 microseconds**, and maximum requested/applied control error was
**2.96e-8**. The second-client test advanced the world independently, verified
that public `snapshot()` detected the frame gap and invalidated the session,
then observed full emergency braking in a subsequent streamed frame. Each
test destroyed its vehicle and restored all recorded settings and actor IDs.
This stress test checks timing, command delivery and finite state; it does not
assess collision avoidance or driving policy quality.

The independent [physical acceptance report](evidence/deep-live-physical.json)
also passes acceleration, braking and steering. The [final server probe](evidence/deep-runtime-final-probe.json)
shows no remaining vehicles and restored asynchronous settings. The server
processes started for this review were stopped. These live runs used the
downloaded **CARLA 0.10.0 / UE5.5** package; **UE5.8 remains untested live**.

Run the repeatable CPU serialization benchmark with:

```powershell
python benchmarks/serialization.py --output artifacts/serialization.json
```

Its [recorded interleaved measurements](evidence/serialization-benchmark.json)
alternate current functions and the previous recursive implementation under
the same machine load, seven repetitions of 20,000 calls. Complete frame
payload serialization measured **12.05 -> 1.02 microseconds**, approximately
**11.8x faster**, with equal payloads. This measures serialization CPU time;
CARLA rendering/network/physics performance is outside that measurement.

Further CARLA `ue58-dev` fixes were submitted under M-Colley:

- [PR #9994](https://github.com/carla-simulator/carla/pull/9994): successful
  metadata-only wheels, generated paths with spaces and a client-only build
  requesting Boost.Python while disabling it. Actual CMake templates
  build/import native fixture wheels; three packaging tests, two dependency
  selection tests and [Linux/Windows CI](https://github.com/M-Colley/carla/actions/runs/39933137062) pass.
- [PR #9995](https://github.com/carla-simulator/carla/pull/9995): invalid atomic
  compare-exchange failure ordering, corrected without changing object layout.
- [PR #9996](https://github.com/carla-simulator/carla/pull/9996): completed and
  timed-out future waiters retained objects and received redundant publications.
  Registration now lasts only for an active wait. Mutable/const lvalue
  exceptions also compile and retain owned copies. The private class layout
  changes, so LibCarla and consumers must be rebuilt together.
- [PR #9997](https://github.com/carla-simulator/carla/pull/9997): sensor listener
  queries raced subscriber changes, ROS visibility and forced activity;
  Dispatcher ROS methods also raced its stream map. One atomic reason mask and
  the existing map mutex preserve the prior visibility/reset rules.
- [PR #9998](https://github.com/carla-simulator/carla/pull/9998): the native RPC
  test exhausted empty polls and then hung joining a client still awaiting
  service; a TCP test connected to the wildcard bind address on Windows.
  Bounded elapsed-time waits, guaranteed cleanup and a loopback fixture fix
  both failures. Each corrected case passes five repeated executions. The
  Python version test now checks installed wheel metadata instead of requiring
  reachable Git tags; original failure and actual-module success are recorded.
- [PR #9999](https://github.com/carla-simulator/carla/pull/9999): pending Episode
  exceptions raced their flag/message readers, and map invalidation raced
  publication/consumption. A mutex with optional text and an atomic exchange
  preserve latest/empty messages and new-episode ordering.

Actual source-header tests passed on native MSVC and GCC. The combined future
and atomic suites passed **190 repetitions under ThreadSanitizer** and **190
under AddressSanitizer/UndefinedBehaviorSanitizer**, with no reports. These
counts mean 19 tests repeated ten times per sanitizer configuration. The
no-exception build passed 36 tests (12 repeated three times). A repeatable
`WorldSnapshot` publication benchmark is included in PR #9996: with 256
inactive waiters, publication measured **8.50-9.10 microseconds before** and
**23-30 nanoseconds after**; the zero-waiter case changed from 22-23 to 26-28 ns.
These are native primitive measurements, not end-to-end simulator FPS results.

The actual production streaming subset passes **11 cases repeated 20 times**,
or **220 case executions per configuration**, on native Windows/MSVC Debug,
Linux/GCC Release, ThreadSanitizer and AddressSanitizer/UndefinedBehaviorSanitizer.
Four original-code filters independently reproduce races. The hot listener
getter uses one atomic load, with no new mutex. Its interleaved Release
benchmark measured roughly 0.84–1.07 ns before and 0.86 ns after; differences
this small are noise-sensitive and do not establish a sensor FPS improvement.

The Episode tests build the **full production client and dependencies** and
drive genuine RPC/TCP world-state packets through its deserializer. Seven
cases repeated 20 times pass under each sanitizer configuration: **140 TSAN
executions and 140 ASan/UBSan executions**, with no reports. Original-source
traces reproduce the flag, string and map invalidation races. Independent
review checked message semantics, callback lifetime and publication ordering.

Saved [source review evidence](evidence/source-review/README.md) includes repeat
logs, original failures, build provenance and the listener benchmark. No Unreal
Engine or UE5.8 simulator runtime pass is claimed for these source patches.

## Complete UE5.8 client build and source suites

The complete production `carla-client` library and native Python extension
were built on Windows without the Unreal engine/project or simulator content.
The combined, published source is
[`ee5859b4e63a4b79c69b02e381c95fb5727e0874`](https://github.com/M-Colley/carla/tree/ee5859b4e63a4b79c69b02e381c95fb5727e0874),
based on `defabe35`, with the source/test fixes from PRs #9991–#9999.
All **35 changed source/build/test files** match their individual PR revisions;
see [the exact source map](evidence/source-review/patch-source-map.json).

The rebuilt integrated client passed **306 native tests in 57 suites**, with
no failures/errors/skips, in 62.304 seconds. The actual installed native Python
module passed **92 source Python API tests**, with no failures/errors/skips,
in 76.338 seconds. The latter includes all three native wheel packaging
fixtures under the activated compiler environment. See the
[native JUnit](evidence/source-review/client-tests-integrated.xml) and
[Python API JUnit](evidence/source-review/python-api-tests-integrated.xml).
The original broad run's failures and the focused fixes are retained alongside
the successful integrated reports.

Build inputs were MSVC 19.51 x64, CPython 3.12.10, CMake 4.4.2, Boost 1.90 and
NumPy 2.5.4 headers. The generated
`carla-0.10.0-cp312-cp312-win_amd64.whl` is **1,692,729 bytes**, SHA-256
`8aedb4423378ee0c12b7319de94e8566a5dca3ab7fc18fc84c2bfa1fd760322a`.
The client archive was rebuilt after copying source changes, and the complete
Python module was rebuilt/relinked against it. A preliminary wheel that used
older object files was explicitly excluded from final integrated validation.

With this actual wheel installed, the bridge passes **240 tests with three
explicit live-server skips** under both NumPy **1.26.4** and **2.5.4**. These
final runs use the clean MyoSuite clone at `c0af9c6`; the reports assert its
import path and the standalone bridge's `src` path. Real MuJoCo muscle/contact
outputs for throttle, brake and steering convert successfully into the
compiled `carla.VehicleControl` and `carla.command.ApplyVehicleControl` classes,
including signed 32-bit gear boundaries. No Unreal connection was made.
The [build manifest](evidence/source-review/build-manifest.json) records exact
commands, object/wheel hashes, dependency revisions, environment provenance,
all test counts and both native-binding smoke reports. Exploratory runs using
an older local MyoSuite worktree were superseded by these clean-clone runs.
The source distribution now includes the serialization benchmark and complete
evidence records (including their README/logs); a final wheel/sdist rebuild
and archive-content check passed.

The [fresh-PC guide](NEW_PC_SETUP.md#10-optional-reproduce-the-ue58-client-source-checks-without-the-engine)
now pins this source snapshot and the actual rpclib, GoogleTest and OpenDRIVE
fixture revisions for a separate client-only reproduction. These results
validate native client code and Python bindings. **No Unreal Engine 5.8
simulator build or live physics run is established by them.**

## Fresh-PC rehearsal and independent code audit

The [new-PC guide](NEW_PC_SETUP.md) was rehearsed in a **new standard Python
3.12 virtual environment**, using normal `python -m pip` and a freshly cloned,
separate MyoSuite repository at `c0af9c6`. This was dependency isolation on the
existing Windows PC, not a separate physical PC or a clean Windows installation.
No previous MyoSuite worktree supplied the fresh environment's imports.

[constraints/windows-py312-live.txt](constraints/windows-py312-live.txt) records
the 91 installed dependency/tool versions, excluding the two editable projects,
the separately installed CARLA wheel and pip. The base environment uses NumPy
1.26.4, MuJoCo 3.17.0 and Gymnasium 1.2.3. The fresh install resolved Ruff
0.17.2 and mypy 2.4.0, and both pass. `pip check` reports no broken requirements.

Independent reviews found and fixed:

- `CarlaSession` configured its RPC timeout after its first `get_world` call;
  the configured timeout now covers that initial RPC.
- Valid NumPy calibration endpoints could retain NumPy integer arithmetic and
  overflow for large spans; stored endpoints and shaping parameters are now
  validated Python floats.
- Host MJCF default joint positions, geom/site poses and alternative orientations
  could shift cockpit pivots/surfaces/sensors or prevent composition. Generated
  mechanisms now explicitly set their local placements and orientation mode,
  without modifying the host's elements.
- Vehicle destruction failures in acceptance cleanup could mask the original
  trial exception and omit settings evidence. Both errors and cleanup outcomes
  are retained, and successful measurements cannot pass failed cleanup.

The fresh live run additionally exposed a setup assumption: a newly spawned
car can still be bouncing after 0.5 s. Setup now waits an initial 25 frames,
then requires 10 consecutive frames below 0.05 m/s, bounded to 250 frames / 5 s.
The throttle fixture's previous excitation produced a marginal acceleration
signal for a fully settled car. Excitation is now 0.015 for throttle, 0.01 for
brake and 0.05 for steering. Real MuJoCo checks on 3.6/3.17 establish sustained
throttle contact at that stimulus; excessive excitation that lets the finger
bypass the pedal is excluded. Vehicle acceptance thresholds were unchanged.

Regressions cover NumPy span overflow, host defaults, timeout ordering, cleanup
error preservation, successful measurements with failed cleanup, interrupted
rest and a bounded settling failure. The composition/calibration cases fail
against their previous implementations.

The final fresh-environment CLI [report](evidence/fresh-pc-cli.json) passes all
three scenarios. Its measured outcomes:

| Control | Actual vehicle result | Setup settling frames |
| --- | --- | --- |
| Throttle | 0 → 2.249 m/s; moved 2.473 m in 3 s | 35 |
| Brake | 2.964 → 0.0002 m/s in 2 s | 75 |
| Steering | −25.77° yaw; moved 6.727 m in 2 s | 75 |

The independent full [pytest live report](evidence/fresh-pc-live.json) also
passes all three controls. All 170 tests passed in 25.52 s; statement coverage
was 928/960 = 97%. No tests were skipped. The final
[server probe](evidence/fresh-pc-final-probe.json) shows zero remaining vehicles
and restored asynchronous settings. The specific server process started for
this rehearsal was stopped. Installation, checks, demo and CMake Python-finder
outputs are saved alongside the reports. The finder check confirms the chosen
Python interpreter/development components; it is not a UE5.8 engine build.

This rehearsal validates the published **UE5.5** binary and the guide's Python
setup. The UE5.8 source-build procedure was reviewed against upstream source,
including the engine's DLSS-disabled mode; it remains unexecuted locally.

## Second physical PC reproduction

On 2026-10-23 the [new-PC guide](NEW_PC_SETUP.md) was followed on a separate
Windows 11 Pro machine (build 26300, Intel i7-14700K, 63.7 GB RAM, RTX 5080).
The bridge was cloned fresh at `12bead9`, MyoSuite freshly cloned at `c0af9c6`,
and a new virtual environment was created from uv-managed **CPython 3.12.13**
with the guide's constraints. `pip check` reported no broken requirements.

That machine already had the official CARLA 0.10.0 Windows package. Its
`VERSION` file names CARLA `ada75f92`, content `a32a1930` and engine `6cf20383`;
its `CHANGELOG` identifies the UE5.5 migration. Its cp312 wheel has the same
SHA-256 as the previously tested release wheel (`69047c3f...b99579`).
The server was started with `-nullrhi -nosound -unattended -notrace
-carla-rpc-port=2000`. Probe: client/server `0.10.0`, `Town10HD_Opt`, 155 spawn
points, 11 vehicle blueprints and asynchronous mode.

| Check | Result |
| --- | --- |
| Offline suite, ruff, mypy, wheel/sdist | 240 passed, 3 live skips; all checks pass |
| Acceptance CLI | Passed all three scenarios |
| Live-enabled suite | **243 passed, 0 skipped, 97% coverage**, 14.89 s |
| Final probe | No remaining vehicles; asynchronous settings restored |

The CLI's measured outcomes were:

| Control | Actual vehicle result | Peak contact | Peak muscle force |
| --- | --- | --- | --- |
| Throttle | 0 → 2.249 m/s; moved 2.473 m in 3 s | 21.76 N | 255.55 N |
| Brake | 2.964 → 0.0002 m/s in 2 s | 13.82 N | 251.49 N |
| Steering | −25.77° yaw; moved 6.727 m in 2 s | 75.59 N | 288.08 N |

The vehicle outcomes match the first PC's fresh-environment run to the
reported precision.
Evidence: [environment record](evidence/second-pc-0.10.0-environment.txt),
[CLI report](evidence/second-pc-0.10.0-cli.json),
[live pytest report](evidence/second-pc-0.10.0-pytest.json),
[JUnit](evidence/second-pc-0.10.0-suite.xml),
[suite summary](evidence/second-pc-0.10.0-suite.txt) and
[installed packages](evidence/second-pc-0.10.0-pip-freeze.txt). The server
processes started for this run were stopped afterwards.

This is a second-machine reproduction of the **UE5.5** result. It does not
establish UE5.8 runtime compatibility. The UE5.8 build attempt on the same
machine is recorded in [RUNTIME_STATUS.md](RUNTIME_STATUS.md#second-pc-ue58-build-attempt).

## Extended UE5.5 validation on the second PC

On 2026-10-24 the same package, bridge commit and environment were exercised
further. All runs used one exclusive ticking client and `Town10HD_Opt`.

**Fresh `-nullrhi` run.** The acceptance CLI passed, and the live-enabled suite
again passed **243 tests with zero skips** (97% coverage, 11.10 s). In its
[session report](evidence/ue55-nullrhi-session.json), 1,200 consecutive muscle
frames ran at 452 frames per wall second, with maximum clock drift 0.536 µs and
maximum applied-control error 2.95e-8. A competing client's tick was detected,
the session invalidated and full braking observed. See the
[JUnit](evidence/ue55-nullrhi-suite.xml) and [summary](evidence/ue55-nullrhi-suite.txt).

**Repeatability.** Five consecutive CLI runs all passed; outcomes are in
[ue55-repeatability.json](evidence/ue55-repeatability.json). MuJoCo contact and
muscle peaks were bit-identical across runs. Across runs, brake and steering
end speeds varied by less than 4e-6 m/s, displacements by less than 6e-6 m and
steering yaw by 8.4e-5°. Throttle end speed ranged
2.239–2.258 m/s (SD 0.0096 m/s), because each throttle trial starts from a
slightly different residual speed after settling (0.0027–0.0054 m/s).

**All four-wheel vehicles.** [ue55-vehicle-matrix.py](evidence/ue55-vehicle-matrix.py)
runs each muscle-operated trial separately for every four-wheel blueprint on
the server, so one failure cannot hide later results. **All 33 trials passed**
with the unchanged thresholds, vehicles destroyed and settings restored
([results](evidence/ue55-vehicle-matrix.json)):

| Vehicle | Mass | Throttle gain / distance | Brake | Steering yaw / speed |
| --- | --- | --- | --- | --- |
| Ford ambulance | 3,000 kg | +1.98 m/s / 2.18 m | 2.50 → 0.0001 m/s | −18.7° / 3.21 m/s |
| CarlaCola truck | 3,000 kg | +2.72 m/s / 3.02 m | 3.42 → 0.0000 m/s | −18.7° / 3.90 m/s |
| Dodge Charger | 1,920 kg | +2.96 m/s / 3.21 m | 3.89 → 0.0005 m/s | −33.6° / 4.69 m/s |
| Dodge Charger police | 1,920 kg | +2.60 m/s / 2.85 m | 3.34 → 0.0000 m/s | −28.8° / 3.99 m/s |
| Fire truck | 3,000 kg | +1.90 m/s / 2.15 m | 2.41 → 0.549 m/s | −12.3° / 2.70 m/s |
| Mitsubishi Fuso | 3,000 kg | +3.29 m/s / 3.71 m | 4.16 → 2.278 m/s | −20.2° / 4.70 m/s |
| Lincoln MKZ | 1,696 kg | +4.70 m/s / 5.04 m | 6.53 → 1.751 m/s | −55.6° / 8.10 m/s |
| Mini Cooper | 1,130 kg | +4.21 m/s / 4.62 m | 5.66 → 0.602 m/s | −47.9° / 6.75 m/s |
| Nissan Patrol | 2,355 kg | +2.25 m/s / 2.47 m | 2.96 → 0.0002 m/s | −25.8° / 3.93 m/s |
| Mercedes Sprinter | 3,000 kg | +1.88 m/s / 2.08 m | 2.35 → 0.0000 m/s | −17.2° / 2.73 m/s |
| Ford taxi | 1,920 kg | +3.02 m/s / 3.28 m | 4.10 → 0.0011 m/s | −37.7° / 5.17 m/s |

The identical muscle stimulus therefore produces vehicle-specific responses
from CARLA's own dynamics. The brake trial's fixed 2 s window does not stop
the faster or heavier vehicles completely, but every one meets the required
deceleration.

**Five-minute soak.** [ue55-soak.py](evidence/ue55-soak.py) ran **15,000
frames (300 simulated seconds)** in 29.1 s wall time
([report](evidence/ue55-soak.json)). Each 6 s cycle has a 2 s muscle press of
the throttle followed by a 4 s release with a scalar 0.5 brake. Every frame
advanced exactly one CARLA frame by 0.02 s. Maximum applied-control error was
2.95e-8. All 50 presses accelerated the car: +1.25 m/s for the first press
from the initial posture, then +1.62 to +2.02 m/s.
Server and client private memory stayed at 3.9 GB and 0.81 GB throughout.
Vehicle, settings and actor IDs were restored.

The soak found two things the 1,200-frame test could not:

- CARLA's elapsed time falls behind MuJoCo's by exactly
  `n × (0.02 − float32(0.02)) = n × 4.470e-10 s`. The first soak attempt
  stopped at frame 4,474 when this crossed the 2 µs absolute bound copied from
  the 1,200-frame test. The 1,200-frame test's 0.536 µs matches the same
  expression to 1e-11 s, consistent with CARLA advancing by the single-precision
  step. The soak therefore bounds the residual after this term; it stayed
  ≤ 3.6e-9 s, with final drift −6.70 µs. The bridge's per-step tolerance is
  unaffected. The README now recommends aligning long recordings by frame.
- With flexor excitation alone, the passive finger stays curled after release
  and drifts sideways. In an offline MuJoCo-only reproduction, it pressed the
  pedal in three bursts and missed it from the fourth on. Co-activating both
  abductors at 0.05, plus extensor 0.02 during release, gives an identical press
  every burst (mean throttle 0.5665, 22.65 N contact) and full release.
  This is a property of the scripted acceptance fixture, not of the bridge.

**Rendered server.** The package was restarted without `-nullrhi`, using
`-RenderOffScreen` at default quality on the RTX 5080. It used about 6 GB of
VRAM and 11.4 GB of RAM. The CLI passed, and the full live-enabled suite passed
**243 tests with zero skips** in 27.09 s ([JUnit](evidence/ue55-rendered-suite.xml),
[CLI report](evidence/ue55-rendered-cli.json),
[session report](evidence/ue55-rendered-session.json)). Compared with the
`-nullrhi` CLI run, end speeds agreed within 3.2e-5 m/s, displacements within
8e-6 m and yaw within 7.3e-4°. The 1,200-frame lockstep test ran at
110 frames per wall second, with the same 0.536 µs drift.

**Rendered video.** [ue55-muscle-drive-video.py](evidence/ue55-muscle-drive-video.py)
recorded one continuous drive with a camera attached to the vehicle. The
muscle throttle accelerated the Nissan Patrol from rest to 4.99 m/s in 4 s.
Muscle steering then turned it 42° left in 2 s, with a scalar 0.3 cruise
throttle. Finally, the muscle brake stopped it from 6.47 m/s in 2.54 s. Each 1920×720 frame
pairs the CARLA camera image from the same synchronous tick (all 475 matched)
with an off-screen MuJoCo render of the finger and mechanism. The MP4 (11.5 s,
50 fps, 28 MB) is kept outside Git under `artifacts/ue55/`. The
[per-frame log](evidence/ue55-muscle-drive-video.json) records the shown
controls, forces and camera frame numbers. The started servers were stopped
afterwards.

## One MyoSuite full body operates every control

The single-finger fixtures above test one mechanism per model. On 2026-10-24
they were superseded for whole-vehicle control by a single seated driver:
MyoSuite's `myofullbody` from `myo_sim` 0.2.3, **416 muscles**, composed with
the cockpit by `ModelBuilder` (`myosuite_carla.fullbody`). One model and one
motor program operate everything in one continuous CARLA drive:

| Body part and muscles | Mechanism | Control |
| --- | --- | --- |
| Right soleus/gastrocnemii; tibialis anterior to release | Throttle pedal | `throttle` |
| Left soleus/gastrocnemii; tibialis anterior to release | Brake pedal | `brake` |
| Both elbows' flexors/extensors, PID stretch reflex | Two free-spinning cradles, 3 and 9 o'clock | `steer` |
| Right FDS2/FDP2; EIP/EDC2 to release | Sequential paddle | `gear`, `manual_gear_shift` |
| Right FDS5/FDP5; EDM/EDC5 to release | Reverse button, held | `reverse` |
| Left FDS2/FDP2; EIP/EDC2 to release | Parking-brake button, held | `hand_brake` |

**Model.** The root joint is removed so the pelvis sits on the seat, and
gravity is off as in the fixtures. Trunk, hips (1.35 rad flexion, 0.1 rad
abduction), knees (1.25 rad), subtalar joints, shoulders, forearm rotation and
wrists are held by joint-equality locks. The arm posture was solved by forward
kinematics with the model's own joint couplings, placing both hands at chest
height with forearms forward and thumbs up. Ankles, toes, elbows and all digits
stay free. The body settles for 3 s with released-limb antagonist tone. Pedals
and cradle plates are then placed along their travel directions until they are
1.37–1.5 mm from the limb, measured with `mj_geomDistance`. Each finger button
is placed on that digit's own simulated pad path, 6–7 mm ahead of the rest pose.
Buttons contact only their digit through explicit MuJoCo contact pairs. The
cradle plates bear only on the metacarpals. `driver.advance` rejects external
forces.

**Offline checks** ([tests/test_fullbody.py](tests/test_fullbody.py)):

- each foot presses only its pedal and releases;
- push-pull reaches steer ±0.36–0.40 at a ±0.14 rad offset, with no pedal or
  button cross-talk;
- finger buttons hold steadily and release;
- four paddle pulls give gears 1, 2, 3, then automatic;
- the shifter counts each edge once;
- importing the package loads neither CARLA nor MyoSuite;
- the real `CarlaBridge` delivers the measured controls to the CARLA test
  double.

**Live drive.** [`examples/validate_fullbody.py`](examples/validate_fullbody.py)
and the opt-in [live test](tests/test_live_fullbody.py) run 23 scripted phases,
about 22 s of simulated driving. Every frame verifies one-frame lockstep and
CARLA's applied throttle, brake, steer, handbrake and manual-shift flag.
Reverse and manual gears must be reported within 25 frames; CARLA needed 2.
All five recorded runs passed every check. Three ran on a rendered server
(two CLI runs and the video run), and two on `-nullrhi` (the live pytest and a
following CLI run). The values below were identical to the reported precision
in four of them. The last `-nullrhi` CLI run differed slightly (1.48 m, −6.9°,
2.33 m/s):

| Check | Measured |
| --- | --- |
| Foot throttle accelerates | 0 → 1.07 m/s over 1.50 m in 3 s |
| Parking brake resists foot throttle | 0.01 m in 2 s at throttle 0.48, vs 1.50 m released |
| Push-pull steers left under foot throttle | −7.0° yaw, steer −0.43 |
| Push-pull steers right under foot throttle | +6.0° yaw, steer 0.55 |
| Foot brake stops the car | 2.34 → 0.012 m/s |
| Finger-held reverse drives backwards | −1.16 m along the heading |
| Reverse released | flag cleared |
| Paddle selects manual gears 1, 2, 3 | CARLA reported gears 1, 2, 3 |
| Paddle returns to automatic | manual shift cleared |
| Parking brake stops the rolling car | 1.56 → 0.01 m/s, brake pedal unused |

Measured contact forces were 12–15 N at the throttle foot and 88–102 N at the
brake foot. The hands bore 38–61 N on the cradles while steering, and the
fingers 1–4 N on the buttons. Driving-muscle forces peaked at 104–884 N. Net
heading after the left/right manoeuvre was −0.5°. Every vehicle was destroyed,
and world settings and actor IDs were restored.
Evidence: [phase summaries, checks and every tenth frame of all five runs](evidence/ue55-fullbody-drive.json),
[final suite JUnit](evidence/ue55-final-suite.xml) and [summary](evidence/ue55-final-suite.txt).
[ue55-fullbody-video.py](evidence/ue55-fullbody-video.py) recorded the
rendered run with a vehicle-attached camera; all 1,097 drive frames matched
their CARLA ticks. The 24.5 s MP4 is kept outside Git.

Defects found and fixed while building the driver:

- **Cockpit touch sensors under-read firm contacts.** Touch sites were 0.1 mm
  larger than their geoms. MuJoCo places a contact point midway through the
  penetration, so a foot pressing 0.4–0.7 mm into a pedal registered 0 N. The
  margin is now 2 mm; a touch sensor still counts only its own body's contacts.
  Pedal contact force now equals spring stiffness × travel. Contact values in
  earlier sections were measured with the old margin and may under-report.
- **Two hands on fixed cradles squeezed the wheel.** Turning the wheel tilted
  the plates and wedged the right hand (≈185 N steady force on a right turn).
  The cradles now spin freely on their posts, like spinner knobs: 3.9–13.6 N.
- **Single-tendon thumb presses buckle.** Under sustained FPL load the thumb's
  CMC joint snapped to its limits within 0.5–3 s, with or without opponens,
  abductor or extensor co-activation. Index and little fingers held every frame
  of a 5 s test;
  reverse therefore uses the little finger.
- **Fingertips sweep past small buttons.** Buttons placed on the initial
  tangent were missed as the finger curled. Placing them on the simulated pad
  path, with a stiffer end stop and a realistic button mass, made presses
  reliable.
- **Strict same-frame reverse/gear checks fail on real CARLA**, because
  `get_control()` reports the gearbox, which lags by two frames.

Limitations: this is a scripted motor program, not a learned or human-fitted
policy. Proximal joints are locked, gravity and vehicle inertial loads do not
act on the body, and braking uses the left foot. Buttons are a palm-strapped
remote, and reverse is hold-to-reverse. Muscle excitations were chosen to
produce moderate controls (throttle 0.38–0.48).

## Package verification

Offline checks ran against MuJoCo 3.6.0 and 3.16.0; the final live-enabled suite
used 3.17.0. They cover:

- Actual MyoSuite finger muscle excitation physically operating throttle, brake
  and steering; idle and disconnected-contact comparisons establish causality.
- Spring return, collision forces, joint-address and sensor-component resolution,
  custom calibration, endpoint saturation and simultaneous-pedal behavior.
- Cockpit composition with real elbow/arm/leg models, preserving all muscle
  actuators; explicit host inertias, all compiler inertia-inference settings,
  degree/radian compiler settings, placements and independent namespaces.
- Actual MuJoCo stepping with strict CARLA API doubles: control-before-tick
  ordering, frame alignment, version mismatch, frame gaps, timing errors, RPC
  failures, missing actors, invalid inputs, hook errors and retryable cleanup.
- Gymnasium action/observation/reward/termination preservation and reset handling.
- Offline example execution through the installed API, independent of the
  repository working directory. Importing the package loads neither CARLA nor
  MyoSuite.

The deterministic offline probe example produces throttle approximately `0.691`
and steering approximately `0.176`, then returns all controls to zero. Its probe
is scripted; separate causal tests exercise actual muscle actuation.

Historical result before the live-runner additions: **133 passed, 1 skipped**
on each MuJoCo version. The skipped test is the
explicitly gated live-server test. Source statement coverage on MuJoCo 3.16.0
is **97%**. Package type checking and Ruff pass, all changed-file pre-commit
hooks pass, and the wheel and source distribution build successfully.

Commands from the repository root:

```shell
python -m pytest tests -q
python -m pytest tests --cov=myosuite_carla --cov-report=term-missing -q
python -m build
python -m mypy src/myosuite_carla --ignore-missing-imports --follow-imports=skip
python -m ruff check src tests examples
python -m ruff format --check src tests examples
```

The standalone GitHub Actions workflow defines offline jobs for Linux Python
3.10/3.14 and Windows Python 3.13. It installs MyoSuite `dev` separately. The live
job is manually gated and requires a prepared self-hosted CARLA UE5.8 runner.
Those remote jobs were not executed locally.

## Historical MyoSuite dependency regression checks

The required model-builder, CPU terms, fragment-compatibility, parity and registry
modules produced **312 passed, 2 skipped, 4 expected failures and 4 failures**
under MuJoCo 3.16.0. All four failures were reproduced independently in a pristine
checkout of the `dev` base (`c0af9c6`), with the same interpreter/dependencies:

- `test_parity[myoChallengeBaodingP2-v1]`
- `test_parity[myoChallengeRelocateP1-v0]`
- `test_parity[myoHandReorient8-v0]`
- `test_parity[myoLegRoughTerrainWalk-v0]`

These are existing parity-baseline mismatches in the external MyoSuite
dependency. No environment implementation, registered contract or parity
baseline is changed by this standalone package.

`pre-commit run --all-files` passed the other hooks but identified pre-existing
indentation formatting in `myosuite/utils/examine_sim.py`; that unrelated
automatic edit was reverted. The changed-file hooks are checked separately.

## Initial live simulator results before the fresh-PC audit

The official release was downloaded using the `ue58-dev` repository's included
download procedure, then started with:

```powershell
$runtime = 'D:/carla-downloads/Carla-0.10.0-Win64-Shipping/Carla-0.10.0-Win64-Shipping'
Start-Process -FilePath "$runtime/CarlaUnreal.exe" -WorkingDirectory $runtime -ArgumentList '-nullrhi','-nosound','-unattended','-notrace','-carla-rpc-port=2000' -WindowStyle Hidden
```

The package's own `CHANGELOG` identifies UE5.5. Both client and server report
`0.10.0`; the map is `Carla/Maps/Town10HD_Opt`, with 155 spawn points and 11
vehicle blueprints. The selected vehicle was `vehicle.nissan.patrol`. Exact
engine patch and server source commit were not established; the shared version
string does not establish an engine branch. Archive/wheel hashes and paths are
saved in [the download manifest](evidence/carla-download-manifest.json).

Actual command from this repository root:

```powershell
.venv-carla/Scripts/python.exe examples/validate_carla.py --host 127.0.0.1 --port 2000 --timeout 30 --target-ref 0.10.0 --build-ref 0.10.0 --output artifacts/carla-0.10.0-live.json
```

The CLI passed. Its complete [measured telemetry](evidence/carla-0.10.0-live.json)
records every requested/applied control and frame, muscle/contact measurement,
vehicle pose, velocity and cleanup result. Outcomes:

| Muscle-operated control | Actual vehicle result | Peak contact | Peak muscle force |
| --- | --- | --- | --- |
| Throttle, 3 seconds | 0.025 → 0.930 m/s; moved 1.190 m | 13.82 N | 251.49 N |
| Brake, 2 seconds | 2.967 → 0.0002 m/s | 13.82 N | 251.49 N |
| Steering, 2 seconds | Yaw changed −25.79°; moved 6.733 m | 75.59 N | 288.08 N |

Setup commands settled the vehicle and accelerated the brake/steering trials;
the steering trial used scalar cruising throttle `0.3`. The measured control in
each trial came from actual MyoSuite muscle/contact motion. No external MuJoCo
forces were applied. These fitted finger fixtures establish physical coupling;
they are not a trained human driver or a combined hands-and-feet policy.

The entire suite then passed independently with the live test enabled:

```powershell
$env:CARLA_BRIDGE_LIVE = '1'
$env:CARLA_HOST = '127.0.0.1'
$env:CARLA_PORT = '2000'
$env:CARLA_TIMEOUT = '30'
$env:CARLA_TARGET_REF = '0.10.0'
$env:CARLA_BUILD_REF = '0.10.0'
$env:CARLA_LIVE_OUTPUT = 'artifacts/carla-0.10.0-pytest.json'
.venv-carla/Scripts/python.exe -m pytest tests --cov=myosuite_carla --cov-report=term-missing --junitxml=test-results/carla-live-suite.xml -q
```

That second [live report](evidence/carla-0.10.0-pytest.json) also passes all three
scenarios. Commands matched CARLA's applied controls within `1e-5`; each step
advanced exactly one frame by 0.02 seconds. Every trial destroyed its vehicle
and restored all captured world settings. A final
[RPC probe](evidence/carla-runtime-probe.json) confirmed no vehicles remained
and asynchronous/default settings were restored. The server process started
for validation was then stopped; its downloaded installation is retained.

### Defect found only by actual runtime testing

Initial live runs failed intermittently because `vehicle.apply_control()` sends
asynchronously: the next tick sometimes executed with the previous input. The
[retained failing report](evidence/carla-async-delivery-failed.json) includes
requested/applied values and cleanup evidence. The fix submits
`client.apply_batch_sync([ApplyVehicleControl(...)], False)`, checks the response,
then performs the existing single `world.tick`. The false flag prevents an
extra tick. Both passing live runs use this fix, with no relaxed assertions.
Regression tests cover acknowledgement-before-tick, rejected/malformed batches,
transport failures and retryable braking/cleanup.

## UE5.8 runtime boundary

The requested `ue58-dev` source was downloaded to `D:/carla-ue58-dev`, pinned to
`defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`. No runnable UE5.8 server or matching
wheel was obtainable from the inspected official release/artifact sources.
The earlier attempt recorded in [evidence/live-attempt.json](evidence/live-attempt.json) has
`status: failed`, error `ModuleNotFoundError: No module named 'carla'`, and no
executed scenarios. This is evidence of the unavailable dependency, not a live
CARLA pass. [RUNTIME_STATUS.md](RUNTIME_STATUS.md) records the installation,
release, artifact and machine investigation. A normal UE5.8 source build needs
more disk than this machine has available. The UE5.5 results above do not close
that target-specific validation gap.

The acceptance CLI and opt-in pytest now use real MyoSuite muscle excitation,
mechanism contact and measured travel to test actual vehicle acceleration,
braking and turning. Each control has its own fitted MyoSuite finger fixture;
setup phases use scalar commands to settle/accelerate the vehicle, and the
steering trial uses scalar cruising throttle. The acceptance mechanism is moved
by its muscles and contacts, without externally applied forces. The runner
checks applied commands, frame timing, actor destruction and restoration of
world settings, and preserves JSON telemetry when a scenario fails.

Run from the standalone root once a matching client/server pair is available:

```shell
python examples/validate_carla.py --host 127.0.0.1 --port 2000 --build-ref ue58-dev --build-commit ACTUAL_COMMIT_SHA --output artifacts/carla-live.json
```

The CLI exits nonzero and writes a failed report if CARLA is missing or a check
fails. Build ref/commit are caller declarations; RPC cannot attest source
provenance. Record the actual server build metadata with any future passing
report. `CARLA_BRIDGE_LIVE=1 python -m pytest tests/test_live_server.py -v` runs
the same checks, using `CARLA_HOST`, `CARLA_PORT`, `CARLA_BUILD_REF`,
`CARLA_BUILD_COMMIT`, `CARLA_TARGET_REF` and `CARLA_LIVE_OUTPUT` for configuration. Without explicit
enablement the live test is skipped. Offline success does not establish
live-server validation.

The package supplies coupling, physical mechanisms and test fixtures. It does
not supply a trained driving policy, fitted human driver, registered driving
task, batched GPU CARLA simulation or automatic biomechanical force feedback.

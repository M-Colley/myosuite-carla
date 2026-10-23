# Validation record

Validated locally on Windows on 2026-10-23. The CARLA source
contract is `ue58-dev` commit `defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`;
see [COMPATIBILITY.md](COMPATIBILITY.md).

This record accompanies Mark Colley's standalone `myosuite-carla` repository.
The initial offline results below were obtained before extraction from the
MyoSuite worktree; the commands use the equivalent standalone root layout.

**Current result: 170 passed, no skips, 97% statement coverage**, including an
actual CARLA **0.10.0 / Unreal Engine 5.5** server. Python 3.12.10, NumPy 1.26.4,
MuJoCo 3.17.0 and the server package's `cp312-win_amd64` wheel were used. This is
supplementary live validation; **UE5.8 runtime compatibility is still unproven**.
See [fresh-environment test output](evidence/fresh-pc-suite.txt),
[JUnit results](evidence/fresh-pc-suite.xml), and
[runtime acquisition](RUNTIME_STATUS.md).

The final offline suite also passes **169 tests, 1 live test skipped** on the
minimum supported MuJoCo 3.6.0; see
[minimum-version output](evidence/offline-audit-mujoco-3.6.txt). Ruff lint and
format checks, mypy and wheel/sdist builds pass.

Before the live-delivery fixes, the standalone suite passed **142 tests** under
both MuJoCo 3.6.0 and 3.16.0 with the live test skipped and **87%** coverage. See the saved
[offline test output](evidence/offline-mujoco-3.16.txt).
The [MuJoCo 3.6 output](evidence/offline-mujoco-3.6.txt) records the minimum-version run.
Focused input and bridge tests also pass on local Python 3.14.6. A regression
test reproduces and guards against the NumPy/pybind enum equality failure found
by Linux Python 3.14 CI; joint-type validation now compares builtin integers.

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

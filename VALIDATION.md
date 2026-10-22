# Validation record

Validated locally on Windows, Python 3.13.12, on 2026-10-23. The CARLA source
contract is `ue58-dev` commit `defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`;
see [COMPATIBILITY.md](COMPATIBILITY.md).

This record accompanies Mark Colley's standalone `myosuite-carla` repository.
The initial offline results below were obtained before extraction from the
MyoSuite worktree; the commands use the equivalent standalone root layout.

The standalone suite, including the sustained muscle fixtures and failed-live-
evidence checks, passes **142 tests** under both MuJoCo 3.6.0 and 3.16.0. The single skipped
test requires an actual CARLA server. Full source statement coverage is **87%**;
the real-server scenario paths remain unexecuted. See the saved
[offline test output](evidence/offline-mujoco-3.16.txt).
The [MuJoCo 3.6 output](evidence/offline-mujoco-3.6.txt) records the minimum-version run.
Focused input and bridge tests also pass on local Python 3.14.6. A regression
test reproduces and guards against the NumPy/pybind enum equality failure found
by Linux Python 3.14 CI; joint-type validation now compares builtin integers.

## Package verification

The offline suite runs against MuJoCo 3.6.0 and 3.16.0. It covers:

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

## Live-server boundary

No CARLA UE5.8 server or matching Python wheel was available locally. The actual
attempt recorded in [evidence/live-attempt.json](evidence/live-attempt.json) has
`status: failed`, error `ModuleNotFoundError: No module named 'carla'`, and no
executed scenarios. This is evidence of the unavailable dependency, not a live
CARLA pass. [RUNTIME_STATUS.md](RUNTIME_STATUS.md) records the installation,
release, artifact and machine investigation.

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
`CARLA_BUILD_COMMIT` and `CARLA_LIVE_OUTPUT` for configuration. Without explicit
enablement the live test is skipped. Offline success does not establish
live-server validation.

The package supplies coupling, physical mechanisms and test fixtures. It does
not supply a trained driving policy, fitted human driver, registered driving
task, batched GPU CARLA simulation or automatic biomechanical force feedback.

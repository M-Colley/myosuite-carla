# Test MyoSuite–CARLA on a fresh Windows PC

This guide targets **Windows 11 x64**. Checked on 2026-10-23. It covers two
different server choices; keep their results separate:

| Server choice | What it establishes | Acquisition |
| --- | --- | --- |
| CARLA `ue58-dev`, Unreal Engine **5.8** | The requested target, once its actual tests pass | Build the CARLA engine fork, content, simulator and matching client; see section 5A |
| Official CARLA **0.10.0**, Unreal Engine **5.5** | Reproduce the supplementary live tests already recorded here | Download the Windows package; see section 5B |

A shared CARLA `0.10.0` version string does **not** distinguish UE5.5 from
UE5.8. Preserve the server's engine version and source/build identity.
No successful UE5.8 engine/simulator build or live run is claimed by this guide. The
Windows build commands below were checked against the pinned upstream source
and its [official Windows guide](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/build_windows_ue5.md).

The Python install procedure was rehearsed in a new normal virtual environment
with a fresh MyoSuite clone on the existing PC. The updated CLI and full suite
passed against the actual UE5.5 release: **243 tests, zero skips, 97% coverage**,
including 1,200 consecutive muscle/contact frames and competing-client checks.
This verifies dependency isolation; it is not a clean Windows installation.
The guide was then followed on a **second physical PC** against the same UE5.5
release: 243 tests passed with zero skips and identical vehicle outcomes. See
[the audit record](VALIDATION.md#second-physical-pc-reproduction).

## 1. Machine, storage and accounts

- Use an Intel/AMD **x64** Windows PC. The supplied Windows wheel is not a
  Windows ARM64 or Linux binary.
- The recorded physics-only run worked on Windows 11, 32 GB RAM, with an RTX
  4050 laptop GPU. This is a tested configuration, not a measured minimum.
  Rendering/camera experiments need suitable GPU memory and current drivers.
- For the published release, allow **at least 40 GB free**, plus room for
  Python dependencies: the ZIP is 9.50 GiB and expands to 20.39 GiB.
- For a UE5.8 source build, reserve roughly **500 GB free on an internal SSD**
  as planning headroom. Upstream budgets about 225 GB for the engine alone and
  about 400 GB for the full build/package. Keep builds out of OneDrive,
  network shares and external disks. Several hours and large downloads should
  be expected; actual build time depends on the PC.
- Sign in to GitHub with access to **`M-Colley/myosuite-carla`**, currently a
  private repository. For source building, also link GitHub to Epic Games,
  accept the organization invitation, and verify access to the
  [CARLA Unreal fork](https://github.com/CarlaUnreal/UnrealEngine).
  The [Epic access instructions](https://www.unrealengine.com/en-US/ue-on-github)
  describe the account linking process.
- Local acceptance tests need no paid service, trained checkpoint, physical
  steering-wheel hardware, CUDA toolkit, PyTorch, WSL, Docker or ROS installation.
  They use the CPU MyoSuite model assets installed by `myo-sim`.

## 2. Install common tools

Install these first, then open a new **PowerShell** window:

1. [Git for Windows](https://git-scm.com/downloads/win), including Git LFS and
   Git Credential Manager.
2. **64-bit CPython 3.12** with the Python launcher. The recorded run used
   [Python 3.12.10](https://www.python.org/downloads/release/python-31210/).
   `cp312` wheels require Python 3.12, not 3.13. Other 3.12 patch releases have
   the same wheel tag; the recorded patch version is provided for reproduction.
3. The [Microsoft x64 Visual C++ runtime](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
   The CARLA package also contains `Engine/Extras/Redist/en-us/UEPrereqSetup_x64.exe`
   if additional Unreal prerequisites are needed.
4. [GitHub CLI](https://cli.github.com/) for authenticating the private clone,
   or use Git Credential Manager when Git prompts for GitHub sign-in.

Verify tools and authenticate:

```powershell
git --version
git lfs version
py -3.12 --version
py -3.12 -c "import platform,struct; print(platform.machine(), struct.calcsize('P')*8)"
gh auth login
gh auth setup-git
```

The last Python command must report a 64-bit process. GitHub access is needed
to clone the package; an access token is not part of the CARLA test itself.
If Python 3.12 was installed by uv, `py -3.12` may report no suitable runtime;
`py -0p` lists it under a tag such as `-V:Astral/CPython3.12.13`. Use that tag
in place of `-3.12` below; the second test PC used CPython 3.12.13 this way.

## 3. Clone separate repositories and create the test environment

Use a short local path. The rest of this guide uses `C:\sim`; replace it
consistently if using another internal SSD. Run each command separately and
stop on errors; a nonzero `$LASTEXITCODE` means a native command failed.

```powershell
$workRoot = 'C:\sim'
New-Item -ItemType Directory -Force $workRoot | Out-Null
Set-Location $workRoot
git clone https://github.com/M-Colley/myosuite-carla.git
git clone --branch dev --single-branch https://github.com/MyoHub/myosuite.git
git -C myosuite checkout --detach c0af9c656de2aa1807924e63d0d5320d77aa1ebd
Set-Location "$workRoot\myosuite-carla"
py -3.12 -m venv .venv
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
& $python -m pip install --upgrade pip
& $python -m pip install -c constraints/windows-py312-live.txt -e ../myosuite -e '.[test,lint,data,analysis]'
& $python -m pip check
```

The pinned MyoSuite commit is from **`dev`** and supplies the required
`ModelBuilder`. Do not replace it with the PyPI MyoSuite package for this
reproduction. The two repositories remain siblings. The constraints file
records the tested Windows/Python 3.12 dependency versions (NumPy 2.5.4 since
the 2026-10-24 validation; earlier records used 1.26.4); it deliberately
does not install CARLA, MyoSuite or this bridge from PyPI. It is a reproduction
snapshot, not a universal constraint set for other Python versions/platforms.

All following commands use the explicit interpreter, so activating a script
or changing PowerShell execution policy is unnecessary. In a new terminal:

```powershell
$workRoot = 'C:\sim'
Set-Location "$workRoot\myosuite-carla"
$python = (Resolve-Path '.venv\Scripts\python.exe').Path
```

## 4. Prove the Python side works before starting CARLA

```powershell
& $python -c "import sys,numpy,mujoco,gymnasium,myo_sim; from myosuite.core.model_builder import ModelBuilder; import myosuite_carla; print(sys.executable); print('numpy',numpy.__version__,'mujoco',mujoco.__version__,'gymnasium',gymnasium.__version__); print(myosuite_carla.__file__)"
& $python examples/offline_cockpit.py
Remove-Item Env:CARLA_BRIDGE_LIVE -ErrorAction SilentlyContinue
& $python -m pytest tests -q
& $python -m ruff check src tests examples
& $python -m ruff format --check src tests examples
& $python -m mypy src/myosuite_carla --ignore-missing-imports --follow-imports=skip
& $python -m build
```

The dependency probe must import `ModelBuilder` and `myo_sim`. The offline
example should show pedal/wheel contact and return to released controls.
There should be exactly **four skipped live-server tests**, plus one if the
optional `opensim` package is not installed, with every other test passing
(308 on the first PC on 2026-10-24). Extra skips can indicate missing
MyoSuite/model dependencies or extras; the full-body tests also need
`myo_sim`'s `myofullbody`, which myo-sim 0.2.3 provides.
Offline success alone is not a CARLA runtime pass.

## 5A. Build the requested UE5.8 server and client

Skip this section only when reproducing UE5.5 results or when a trusted UE5.8
package/server and its matching client wheel have already been supplied.
The release in section 5B cannot satisfy this target.

### Install build prerequisites

- Install **Visual Studio 2022**, using the
  [VS 2022 installer](https://aka.ms/vs/17/release/vs_community.exe). The pinned
  setup script looks specifically for Community/Professional/Enterprise under
  `Microsoft Visual Studio\2022`; a VS 2026-only installation is not equivalent.
- In Visual Studio Installer, import the CARLA checkout's
  `Unreal\CarlaUnreal\.vsconfig` after cloning it below. Include C++ desktop
  development, C++ game development, managed desktop tools, Windows 11 SDK
  10.0.22621, and MSVC v143 **14.38**, or upstream's supported 14.44.35211 or
  newer toolset. Install its requested .NET targeting/developer components.
- Enable Windows Developer Mode, as required by the upstream Windows build.
- Git LFS must be installed before cloning the content. Keep engine/content
  on the internal build SSD and retain substantial free space throughout.

### Download the exact CARLA source and matching fork branches

In PowerShell, from the workspace root:

```powershell
Set-Location $workRoot
git lfs install
git config --global core.longpaths true
git clone --branch ue58-dev --single-branch https://github.com/carla-simulator/carla.git carla-ue58
git -C carla-ue58 checkout --detach defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5
git -C carla-ue58 remote add colley https://github.com/M-Colley/carla.git
git -C carla-ue58 fetch colley codex/ue58-client-validation
git -C carla-ue58 checkout --detach ee5859b4e63a4b79c69b02e381c95fb5727e0874
git ls-remote https://github.com/CarlaUnreal/UnrealEngine.git refs/heads/ue58-dev-carla
git clone --branch ue58-dev-carla --single-branch https://github.com/CarlaUnreal/UnrealEngine.git UnrealEngine5_carla
git -C UnrealEngine5_carla checkout --detach 0703595a065dee51538720027ce9cc05bd464517
git clone --branch ue58-dev-carla --single-branch https://bitbucket.org/carla-simulator/carla-content.git carla-ue58/Unreal/CarlaUnreal/Content/Carla
git -C carla-ue58/Unreal/CarlaUnreal/Content/Carla checkout --detach bc499dedeae683ad154f586ca7f94a463e234fc7
git -C carla-ue58/Unreal/CarlaUnreal/Content/Carla lfs pull
```

If the engine query returns 404/permission denied, complete Epic/GitHub access
before retrying. Do not substitute the stock Epic launcher engine or the
UE5.5 `ue5-dev-carla` fork. The content directory must be the actual **`Carla`**
checkout, not just an empty parent named `Content`. Stop if LFS objects failed
to download.

These engine/content commits were checked on the date above; the engine's
`Build.version` identifies 5.8.0. Pinning them makes the intended source inputs
explicit, but does not turn this reviewed recipe into a verified UE5.8 build.

The pinned CARLA revision includes the source and regression tests from
[PRs #9991–#9999](https://github.com/M-Colley/carla/blob/ee5859b4e63a4b79c69b02e381c95fb5727e0874/VALIDATION_CLIENT.md),
which remain unmerged upstream. All 35 changed source/build/test files were
checked against their individual PR revisions. The actual client and complete
Python extension built from this snapshot; **306 native tests and 92 Python
API tests passed**, including the native wheel fixtures. This validates the
client portion. The full engine/simulator procedure below remains unexecuted.

The upstream `CarlaSetup.bat` automates much of this but also selects Python,
installs prerequisites and checks existing folders. The explicit sequence here
keeps the Python build environment and the required content checkout visible.
If using the upstream automation instead, pass the intended Python root and
verify its engine/content/Python decisions before reusing its build directory.

### Build in the VS 2022 x64 command prompt

Open **x64 Native Tools Command Prompt for VS 2022** from the Start menu. The
following block is **cmd.exe**, not PowerShell. Change `C:\sim` if needed.
This build environment is separate from the bridge's `.venv`.

```bat
cd /d C:\sim
py -3.12 -m venv ue58-build-env
set "CARLA_BUILD_PYTHON=C:\sim\ue58-build-env\Scripts\python.exe"
set "CARLA_UNREAL_ENGINE_PATH=C:\sim\UnrealEngine5_carla"
set "DLSS_SDK=disabled"
set "PATH=C:\sim\ue58-build-env\Scripts;%PATH%"
"%CARLA_BUILD_PYTHON%" -m pip install --upgrade pip
"%CARLA_BUILD_PYTHON%" -m pip install "cmake>=3.28" ninja
cd /d C:\sim\carla-ue58
"%CARLA_BUILD_PYTHON%" -m pip install -r requirements.txt "numpy==1.26.4"
cd /d C:\sim\UnrealEngine5_carla
call Setup.bat
call GenerateProjectFiles.bat
msbuild Engine\Intermediate\ProjectFiles\UE5.vcxproj /property:Configuration=Development_Editor /property:Platform=x64
```

Check the last command succeeds and
`Engine\Binaries\Win64\UnrealEditor.exe` exists before continuing. Engine
source access does not imply a completed engine build. An existing engine build
from an older `ue58-dev-carla` commit is not sufficient: the pinned CARLA source
uses `FPostProcessSettings::PathTracingLensAOV` and the
`SCS_PathTracingLensDistance`/`SCS_PathTracingLensTag` capture sources, which
the engine branch added after `cacb25b9`. Update such a checkout to the pinned
engine commit and rebuild; the header change rebuilds most of the editor.

The editor build needs substantial **commit memory**, not only physical RAM.
Unreal Build Accelerator kills new compiles once system commit reaches 95% of
its limit. Stop other memory-heavy jobs before building; on the second test PC,
unrelated training jobs held about 90% of commit and the rebuild could not
progress. The disabled DLSS mode
is suitable for this physics test; it avoids requiring the optional NVIDIA SDK.

Configure CARLA with the same Python root for both CMake Python finders, and
cook only the map used by these tests:

```bat
cd /d C:\sim\carla-ue58
cmake -G Ninja -S . -B Build --toolchain=CMake/Toolchain.cmake -DCMAKE_BUILD_TYPE=Release -DCARLA_UNREAL_ENGINE_PATH="%CARLA_UNREAL_ENGINE_PATH%" -DPython_ROOT_DIR=C:/sim/ue58-build-env/Scripts -DPython3_ROOT_DIR=C:/sim/ue58-build-env/Scripts -DPython_EXECUTABLE="%CARLA_BUILD_PYTHON%" -DPython3_EXECUTABLE="%CARLA_BUILD_PYTHON%" -DCARLA_DLSS_SDK_PATH=disabled -DCARLA_MAPS_TO_COOK=/Game/Carla/Maps/Town10HD_Opt
cmake --build Build
cmake --build Build --target carla-python-api-install
```

To create a portable package, start Zen in a separate command prompt and leave
it running during packaging:

```bat
"C:\sim\UnrealEngine5_carla\Engine\Binaries\Win64\zenserver.exe" --port 8558
```

Then, in the configured VS command prompt:

```bat
cd /d C:\sim\carla-ue58
cmake --build Build --target package
```

The expected package is
`C:\sim\carla-ue58\Build\Package\Carla-0.10.0-Win64-Shipping`, with a matching
wheel in its `PythonAPI\carla\dist`. A development package can be built with
`--target package-development` when debugging needs runtime logs. Do not treat
an uncooked game executable as a complete portable server.

Back in the original bridge **PowerShell** terminal, select the built package
and preserve provenance:

```powershell
$runtime = "$workRoot\carla-ue58\Build\Package\Carla-0.10.0-Win64-Shipping"
Test-Path "$runtime\CarlaUnreal.exe"
Get-Content "$workRoot\UnrealEngine5_carla\Engine\Build\Build.version"
git -C "$workRoot\carla-ue58" rev-parse HEAD
git -C "$workRoot\UnrealEngine5_carla" rev-parse HEAD
git -C "$workRoot\carla-ue58\Unreal\CarlaUnreal\Content\Carla" rev-parse HEAD
$env:CARLA_TARGET_REF = 'ue58-dev'
$env:CARLA_BUILD_REF = 'ue58-dev'
$env:CARLA_BUILD_COMMIT = (git -C "$workRoot\carla-ue58" rev-parse HEAD).Trim()
```

Require a real package executable, engine `MajorVersion: 5` /
`MinorVersion: 8`, the correct fork branches and actual recorded commits.
Keep build configuration/cache and source SHAs with the reports. Build ref
arguments below are declarations; the RPC server cannot attest its branch.
Continue with section 6 using this `$runtime`.

## 5B. Download the already tested UE5.5 release

This is the shorter route to reproducing the saved **supplementary** tests.
It needs no Visual Studio compiler, Epic source access or UE engine build.
The [official 0.10.0 release](https://github.com/carla-simulator/carla/releases/tag/0.10.0)
identifies the UE5.5 migration and links this Windows archive.

In the bridge PowerShell terminal:

```powershell
$downloads = "$workRoot\downloads"
$runtimeParent = "$workRoot\runtimes"
New-Item -ItemType Directory -Force $downloads,$runtimeParent | Out-Null
$archive = "$downloads\Carla-0.10.0-Win64-Shipping.zip"
curl.exe -fL --retry 3 --retry-delay 5 -C - -o $archive https://downloads.carlasim.com/Windows/Carla-0.10.0-Win64-Shipping.zip
if ($LASTEXITCODE) { throw 'CARLA download failed' }
$expectedArchiveHash = '0E02B7DD1F12EC5F2517ADF73F2944CC77ADBC6C34896DA0C604BC1E096C5682'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $expectedArchiveHash) { throw 'Archive differs from the tested download' }
& $python -m zipfile -e $archive $runtimeParent
if ($LASTEXITCODE) { throw 'CARLA extraction failed' }
$runtime = "$runtimeParent\Carla-0.10.0-Win64-Shipping"
Test-Path "$runtime\CarlaUnreal.exe"
Get-Content "$runtime\CHANGELOG" -TotalCount 15
$env:CARLA_TARGET_REF = '0.10.0'
$env:CARLA_BUILD_REF = '0.10.0'
Remove-Item Env:CARLA_BUILD_COMMIT -ErrorAction SilentlyContinue
```

The hash is the archive actually tested in this repository, not a separate
vendor signature. If extraction creates an extra nesting level, set `$runtime`
to the directory that directly contains `CarlaUnreal.exe`, `Engine`, and
`PythonAPI`. Retaining the ZIP makes later reproduction possible.

## 6. Install the client from the selected server package

Do this for **either** server choice. Do not reuse the 0.10.0 release wheel
against a source-built UE5.8 server merely because the version strings match.

```powershell
$wheels = @(Get-ChildItem "$runtime\PythonAPI\carla\dist" -Filter '*cp312*win_amd64.whl')
if ($wheels.Count -ne 1) { throw 'Expected exactly one matching CPython 3.12 Windows wheel' }
& $python -m pip install --force-reinstall -c constraints/windows-py312-live.txt $wheels[0].FullName
& $python -m pip check
& $python -c "import sys,carla; print(sys.executable); print(carla.__file__)"
Get-FileHash -LiteralPath $wheels[0].FullName -Algorithm SHA256
```

The tested release's wheel SHA-256 is
`69047c3f84b083112c147d718b5627dd5a77dd83a02c049fb078e90af8b99579`.
A newly built UE5.8 wheel has its own hash: record it rather than expecting the
release hash. If your UE5.8 package targets another Python minor version,
rebuild its client for 3.12 or create a compatible separate client environment;
do not rename a wheel's tag to force installation.

## 7. Start and probe the simulator

Use one server and one ticking client. Close other CARLA scenarios, manual
control clients and Traffic Manager scripts for this acceptance test. Verify
TCP 2000 is free before starting the selected package:

```powershell
if (Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue) { throw 'Port 2000 already has a server' }
New-Item -ItemType Directory -Force artifacts | Out-Null
$serverLauncher = Start-Process -FilePath "$runtime\CarlaUnreal.exe" -WorkingDirectory $runtime -ArgumentList '-nullrhi','-nosound','-unattended','-notrace','-carla-rpc-port=2000' -WindowStyle Hidden -PassThru
$serverLauncher.Id
```

First startup can take several minutes. Test TCP readiness with
`Test-NetConnection 127.0.0.1 -Port 2000`, then perform this actual RPC probe:

```powershell
& $python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(30); print('client',c.get_client_version()); print('server',c.get_server_version()); w=c.get_world(); print('map',w.get_map().name,'spawns',len(w.get_map().get_spawn_points())); s=w.get_settings(); print('sync',s.synchronous_mode,'fixed_dt',s.fixed_delta_seconds); print('vehicles',[b.id for b in w.get_blueprint_library().filter('vehicle.*')])"
```

Require matching client/server versions, `Town10HD_Opt`, available four-wheel
vehicles and `synchronous_mode=False`. The release tested here had 155 spawn
points. Do not print the entire `WorldSettings` object on 0.10.0: its formatter
can throw `Bad optional access` when fixed delta is unset; inspect individual
attributes as above. If RPC is refused, let startup finish or inspect crashes
before retrying. `-nullrhi` performs physics/control testing without rendered
images. **Do not spawn cameras with it.** For image-based tests, restart with a
supported rendering configuration and adequate GPU resources instead. On the
second test PC (RTX 5080), `-RenderOffScreen` in place of `-nullrhi` used about
6 GB of VRAM and 11.4 GB of RAM at default quality; the full live suite passed
there too and supported an attached RGB camera.

If using an already running remote server, skip the local process launch and
local TCP check. Set `CARLA_HOST`/`CARLA_PORT` to its reachable address before
the RPC probe and tests; substitute those values in the section 7 probe too.

## 8. Run actual muscle-driven acceptance tests

Set the target variables in section 5A or 5B first. From the bridge root:

```powershell
if (-not $env:CARLA_HOST) { $env:CARLA_HOST = '127.0.0.1' }
if (-not $env:CARLA_PORT) { $env:CARLA_PORT = '2000' }
$env:CARLA_TIMEOUT = '30'
& $python examples/validate_carla.py --host $env:CARLA_HOST --port $env:CARLA_PORT --timeout 30 --target-ref $env:CARLA_TARGET_REF --build-ref $env:CARLA_BUILD_REF --output "artifacts/$($env:CARLA_TARGET_REF)-cli.json"
if ($LASTEXITCODE) { throw 'Live CARLA acceptance failed; inspect the JSON error and samples' }
& $python examples/validate_fullbody.py --host $env:CARLA_HOST --port $env:CARLA_PORT --timeout 60 --target-ref $env:CARLA_TARGET_REF --build-ref $env:CARLA_BUILD_REF --output "artifacts/$($env:CARLA_TARGET_REF)-fullbody.json"
if ($LASTEXITCODE) { throw 'Live full-body drive failed; inspect the JSON checks and phases' }
$env:CARLA_BRIDGE_LIVE = '1'
$env:CARLA_LIVE_OUTPUT = "artifacts/$($env:CARLA_TARGET_REF)-pytest.json"
$env:CARLA_FULLBODY_OUTPUT = "artifacts/$($env:CARLA_TARGET_REF)-pytest-fullbody.json"
& $python -m pytest tests --cov=myosuite_carla --cov-report=term-missing --junitxml=artifacts/results.xml -q
if ($LASTEXITCODE) { throw 'Full live-enabled suite failed' }
```

The CLI writes evidence on success **and failure**. It separately exercises a
real MyoSuite finger physically moving each passive throttle/brake/wheel
mechanism; no trained policy is needed. Scalar setup commands settle the car
and accelerate brake/steering trials; steering uses scalar cruising throttle.
No MuJoCo external force is used to move the tested mechanism.

Setup brakes until the car is measurably at rest (10 consecutive frames below
0.05 m/s after the initial 0.5 s), with a 5 s limit. The JSON records
`settling_frames`, so a freshly spawned bouncing car is not assumed to have
settled after a fixed half-second.

Pass criteria enforced in code:

- Measured contact exceeds 10 N and peak muscle force exceeds 100 N.
- Throttle increases speed by more than 0.8 m/s and moves over 1 m in 3 s.
- Brake reduces speed by more than 1 m/s and below 70% of initial speed in 2 s.
- Negative steering turns the vehicle left by more than 2° in 2 s, with final
  speed above 1 m/s.
- Requested/applied control channels agree within `1e-5`; each bridge step
  advances exactly one CARLA frame with a 0.02 s interval.
- Each trial destroys its vehicle and restores the captured world settings.

The full-body CLI drives one seated MyoSuite `myofullbody` (416 muscles)
through 26 phases, about 23 s, loaded by gravity and the measured vehicle
motion: hold, a parking-brake switch press, throttle against the parking
brake, a second press that releases it, throttle, push-pull steering left and
right under throttle, braking, a reverse button press with the brake held,
reversing, three paddle-selected manual gears and a rolling parking-brake
stop. It prints 13 physical checks, all of which must pass; add `--fast` to
repeat it with the fast preset. See
[the researcher-feature record](VALIDATION.md#researcher-features-loads-recording-surrogate-task-and-conditions).
Allow about 20 s per run on `-nullrhi`; building the body takes about 3 s.

Check the live-enabled suite has **zero skips** and all tests pass. Separately
verify that the report's target/build labels describe the actual binary, using
the engine/build metadata preserved above. Code records caller declarations and
label mismatches; it cannot independently enforce truthful binary provenance.

Exact trajectories and runtimes can vary. Use the enforced thresholds rather
than requiring bit-identical telemetry. The package supplies physical coupling
and acceptance fixtures; it does not supply a trained full-body driving policy.

If the server is on another PC, run the simulator there and the client/tests
here with the server's reachable IP. Install a matching wheel for the **client
PC's** OS/Python, from the same CARLA build. Set `CARLA_HOST` accordingly. Permit
CARLA RPC/streaming ports (default 2000/2001) only as needed on the server's
private network/firewall; a remote server does not need this bridge installed.
The two computers must not have competing ticking clients. Keep the server
build identity with the client-side evidence. Run server process/port and stop
commands on the server machine; the local PID examples do not manage a remote
server.

For example, before the section 8 commands:

```powershell
$env:CARLA_HOST = '192.168.1.50'
$env:CARLA_PORT = '2000'
```

Replace the example IP with the actual server address. The defaults in section
8 preserve these values rather than resetting them to localhost.

### Optional: researcher features on the live server

These runs need the `data` and `analysis` extras. Each drives its own vehicle
and restores the world:

```powershell
& $python examples/validate_fullbody.py --host $env:CARLA_HOST --port $env:CARLA_PORT --timeout 60 --target-ref $env:CARLA_TARGET_REF --build-ref $env:CARLA_BUILD_REF --output artifacts/fullbody-recorded.json --record artifacts/fullbody-recorded --formats csv,opensim,arm_force,parquet,mcap
& $python examples/drive_scenario.py --backend carla --host $env:CARLA_HOST --scenario lead_vehicle_braking --record artifacts/scenario-lead
& $python examples/identify_vehicle.py --host $env:CARLA_HOST --seeds 0 1 2
& $python examples/fit_vehicle_model.py --train artifacts/vehicle-sysid/seed0 --test artifacts/vehicle-sysid/seed1 artifacts/vehicle-sysid/seed2
```

`drive_scenario.py --backend carla` uses Town10HD_Opt spawn 105 by default (two
lanes, 170 m straight); it refuses a lane change toward a non-drivable lane.
The identification takes about 10 minutes per three seeds on `-nullrhi`.

## 9. Save the reproduction record and stop only your server

```powershell
& $python -m pip freeze | Set-Content artifacts/python-packages.txt
git rev-parse HEAD | Set-Content artifacts/bridge-commit.txt
git -C ../myosuite rev-parse HEAD | Set-Content artifacts/myosuite-commit.txt
Get-FileHash -LiteralPath $wheels[0].FullName -Algorithm SHA256 | Format-List | Out-File artifacts/carla-wheel-hash.txt
& $python -c "import os,carla; c=carla.Client(os.environ.get('CARLA_HOST','127.0.0.1'),int(os.environ.get('CARLA_PORT','2000'))); c.set_timeout(30); w=c.get_world(); s=w.get_settings(); print('sync',s.synchronous_mode,'fixed_dt',s.fixed_delta_seconds); print('remaining vehicles',[v.id for v in w.get_actors().filter('vehicle.*')])"
```

Also preserve the CARLA archive/build metadata, engine/content SHAs for UE5.8,
test stdout, both JSON reports and JUnit XML. The runner restores its vehicle
and world settings; it does not stop the server process.

The Windows launcher can spawn a separate shipping executable. Inspect exact
paths/PIDs before stopping the instance you started; do not kill unrelated
Unreal or CARLA processes:

```powershell
Get-CimInstance Win32_Process -Filter "Name LIKE 'CarlaUnreal%'" | Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine
Get-NetTCPConnection -LocalPort 2000 -State Listen | Select-Object OwningProcess
```

After verifying the owning PID belongs to your selected `$runtime`, use
`Stop-Process -Id ACTUAL_VERIFIED_SERVER_PID`. Confirm port 2000 is no longer
listening. Stop the Zen process you started after packaging as well.

## 10. Optional: reproduce the UE5.8 client source checks without the engine

This builds the client library, full Python extension and offline source tests.
It needs a C++20 x64 MSVC developer command prompt and the build tools from 5A,
but needs neither Epic engine access nor simulator content. The recorded local
compiler was MSVC 19.51; this was an isolated build on the existing Windows PC.
Use a separate environment from the bridge's live-server environment.

In the x64 developer **cmd.exe** prompt, with `C:\sim` adjusted consistently:

```bat
cd /d C:\sim
git clone --branch codex/ue58-client-validation --single-branch https://github.com/M-Colley/carla.git carla-client-check
git -C carla-client-check checkout --detach ee5859b4e63a4b79c69b02e381c95fb5727e0874
git clone --depth 1 --branch carla https://github.com/carla-simulator/rpclib.git client-deps/rpclib
git -C client-deps/rpclib fetch --depth 1 origin bbfe141f4b5c1396cedb91071f8a839a6fa9de35
git -C client-deps/rpclib checkout --detach bbfe141f4b5c1396cedb91071f8a839a6fa9de35
git clone --depth 1 --branch v1.17.0 https://github.com/google/googletest.git client-deps/googletest
git -C client-deps/googletest checkout --detach 52eb8108c5bdec04579160ae17225d66034bd723
py -3.12 -m venv ue58-client-env
set "CLIENT_PYTHON=C:\sim\ue58-client-env\Scripts\python.exe"
set "PATH=C:\sim\ue58-client-env\Scripts;%PATH%"
set "CMAKE_BUILD_PARALLEL_LEVEL=2"
"%CLIENT_PYTHON%" -m pip install --upgrade pip
"%CLIENT_PYTHON%" -m pip install "cmake>=3.28" ninja
"%CLIENT_PYTHON%" -m pip install -r carla-client-check\requirements.txt -r carla-client-check\PythonAPI\test\requirements.txt "numpy==2.5.4"
cd /d C:\sim\carla-client-check
git clone --depth 1 --branch 0.1.4 https://github.com/carla-simulator/opendrive-test-files.git Build/test-content
git -C Build/test-content checkout --detach 5929b67cc414b611b701ad251fe1e6d30f465bba
cmake -S . -B Build/ClientReview -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_CARLA_UNREAL=OFF -DBUILD_CARLA_SERVER=OFF -DBUILD_CARLA_CLIENT=ON -DBUILD_PYTHON_API=ON -DBUILD_EXAMPLES=OFF -DBUILD_LIBCARLA_TESTS=ON -DBUILD_OSM_WORLD_RENDERER=OFF -DENABLE_ROS2=OFF -DENABLE_ROS2_DEMO=OFF -DENABLE_RSS=OFF -DENABLE_OSM2ODR=OFF -DENABLE_PYTORCH=OFF -DINSTALL_LIBCARLA=OFF -DFETCHCONTENT_SOURCE_DIR_RPCLIB=C:/sim/client-deps/rpclib -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=C:/sim/client-deps/googletest -DPython_EXECUTABLE="%CLIENT_PYTHON%" -DPython3_EXECUTABLE="%CLIENT_PYTHON%"
cmake --build Build/ClientReview --target carla-client libcarla_test_client carla-python-api --parallel 2
Build\ClientReview\LibCarla\libcarla_test_client.exe --gtest_output=xml:Build/native-client.xml
"%CLIENT_PYTHON%" -m pip install --force-reinstall --no-deps Build\ClientReview\PythonAPI\dist\carla-0.10.0-cp312-cp312-win_amd64.whl
cd PythonAPI\test\unit
"%CLIENT_PYTHON%" -m nose2 -v
cd /d C:\sim\myosuite-carla
"%CLIENT_PYTHON%" -m pip install -c constraints/windows-py312-live.txt -e ../myosuite -e ".[test,lint]"
"%CLIENT_PYTHON%" -m pip check
"%CLIENT_PYTHON%" -m pytest tests -q --junitxml=artifacts/ue58-client-offline.xml
```

Stop on nonzero exit codes. The original source's client-only Boost selection,
two native network tests and Git-tag-dependent Python version test failed
during this review; the pinned snapshot includes their fixes. The full source
suite passed 306 native cases and 92 Python cases with no skips. Keep the
configure/build output, source/dependency SHAs, XML and wheel hash with the run.
Use this client wheel with a server built from the same source when performing
live UE5.8 acceptance. These offline checks do not provide that server.

The final three commands reuse the clean MyoSuite clone from section 3 and
select the guide's NumPy 1.26.4/OpenCV versions in this isolated environment.
The new native wheel passed **240 bridge tests with 3 live-server skips** under
both NumPy 1.26.4 and 2.5.4. Real MyoSuite muscle/contact outputs also converted
successfully to the compiled CARLA control and command classes for all three
mechanisms; see [the source validation evidence](VALIDATION.md#complete-ue58-client-build-and-source-suites).

## Troubleshooting

| Symptom | Check/fix |
| --- | --- |
| Private bridge clone denied | Authenticate the GitHub account with repository access; `gh auth setup-git` configures Git's credential helper. |
| `ModelBuilder` import fails | Install the separate pinned MyoSuite dev checkout with the same `$python`; inspect `myosuite.__file__`. |
| Missing finger/model assets | Confirm `myo-sim==0.2.3` imports in that environment; rerun `pip check` and the import probe. |
| Wheel unsupported / CARLA DLL import failure | Use Windows x64 CPython 3.12 and its bundled cp312 win_amd64 wheel; install the x64 VC runtime/Unreal prerequisites. |
| NumPy ABI/import errors | Use this guide's isolated environment and constraints; the tested release uses NumPy 1.26.4. |
| Client/server version mismatch | Replace the client with the wheel from the selected server build; version equality still needs branch/engine provenance. |
| RPC refused or timeout | Check process/crash output, wait for map loading, verify host/port and private-network access; a launcher PID alone does not prove readiness. |
| Existing synchronous owner / frame gap | Stop other clients and restart from an exclusive asynchronous world; do not hide the error by allowing competing tickers. |
| Missing vehicle blueprint | Leave `--blueprint` unset for server discovery, or choose an actual four-wheel blueprint from the probe. |
| No 35 m straight spawn | Use Town10HD_Opt with unoccupied road spawn points; remove only vehicles belonging to your scenario. |
| Engine clone 404 | Complete Epic/GitHub linking and accept its invitation; verify fork access with `git ls-remote`. |
| UE build cannot find VS2022 / MSVC | Install VS2022 and the requested toolset/components; use its x64 Native Tools prompt, not an ordinary shell. |
| `py -3.12` finds no runtime | A uv-managed Python is listed by `py -0p`; select it with its `-V:` tag or use its full path. |
| `UbaSessionServer - Killed process ... Low on memory` repeats | Other processes hold most commit memory. Stop them or provide more commit (RAM/pagefile) before building; do not disable the protection while unrelated jobs are running. |
| Carla plugin errors on `PathTracingLensAOV` or `SCS_PathTracingLens*` | The engine predates the pinned `ue58-dev-carla` commit; update and rebuild it. |
| MSVC 19.44 internal compiler error C1001 in `DumpPackageToJson` or `uLangCore` | The second test PC's engine checkout carries local `OptimizeCode = CodeOptimization.Never` edits in those two `Build.cs` files for this. This was observed, not re-verified, here. |
| Missing `DLSSRRDenoiser` plugin | Check the actual `ue58-dev-carla` engine checkout; disabling the SDK does not add a missing engine plugin. |
| Cook/package failure | Verify content LFS objects, free disk, Town10 cook/default map, and the separate Zen server. Preserve the actual error; do not ignore cook failures. |
| Test reports failed cleanup | Inspect `cleanup_errors`, `vehicle_destroyed` and `settings_restored`; repair the server/scenario before repeating. |
| Cameras crash a null-RHI server | Restart with rendering enabled before creating image sensors. |

To use a trained driver after acceptance passes, fit the cockpit to its limb
model, provide a compatible CPU MyoSuite environment and muscle policy, and
use [examples/live_drive.py](examples/live_drive.py). See [README.md](README.md)
for its factory/policy API, calibration, coordinate conventions and limitations.

# Actual CARLA UE5.8 runtime status

Checked on **2026-10-23 (Europe/Berlin)**. No real CARLA server was started during
this investigation, and no live CARLA UE5.8 pass is claimed. Offline MuJoCo tests,
test doubles, an importable client, and skipped integration tests do not establish
that the bridge works against a running simulator.

## Published runtimes checked

The target is CARLA's **`ue58-dev` branch with Unreal Engine 5.8**. It must not be
confused with `ue5-dev`, CARLA 0.10.0, or the generic UE5 nightly:

| Source checked | Observed result |
| --- | --- |
| [Official branch download page](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/download.md) | UE5.8 / CARLA 1.0.0 release and Docker links remain marked TODO; the listed Linux nightly explicitly targets `ue5-dev` / UE5.5. |
| [GitHub releases](https://github.com/carla-simulator/carla/releases) | Published releases include 0.9.16 (UE4.26) and 0.10.0 (UE5.5); no 1.0.0 release was found. |
| [Official Docker tags](https://hub.docker.com/r/carlasim/carla/tags) | Tags API returned all 23 tags with no next page. No `ue58`, `dev`, nightly, or `1.0.0` tag was present. The UE5 image was `0.10.0`, uploaded in January 2025, for Linux amd64. `latest` is an older 0.9.x image. |
| [Latest `ue58-dev` simulator workflow](https://github.com/carla-simulator/carla/actions/runs/39761638188) | Failed while building the Unreal target: required `DLSSRRDenoiser` plugin missing. Package upload was skipped. The run had zero artifacts; the repository artifacts API also returned zero artifacts. |
| [R2 deployment branch](https://github.com/carla-simulator/carla/blob/ci-deploy-r2-ue58-dev/Util/Tools/Deploy.sh) | Deployment still names the generic `CARLA_UE5_Latest.tar.gz` and dated commit archives. No successful workflow/artifact providing a UE5.8 runtime was found for this branch or the inspected Windows branches. |

The public [generic UE5 nightly](https://downloads.carlasim.com/Linux/Dev/CARLA_UE5_Latest.tar.gz)
answered HTTP 200 to a HEAD request: **15,710,399,374 bytes**, last modified
**2026-09-02**. The official download page identifies it as UE5.5. It was not
downloaded or substituted for the requested UE5.8 build.

Representative HEAD probes returned **HTTP 404** on both
`downloads.carlasim.com` and the official Backblaze release origin:

```text
/Linux/Dev/CARLA_UE58_Latest.tar.gz
/Linux/Dev/CARLA_UE5_8_Latest.tar.gz
/Linux/Dev/CARLA_UE5.8_Latest.tar.gz
/Linux/Dev/CARLA_1.0.0_Latest.tar.gz
/Linux/Carla-1.0.0-Linux-Shipping.tar.gz
/Windows/Dev/CARLA_UE5_Latest.zip
/Windows/Dev/CARLA_UE58_Latest.zip
/Windows/Carla-1.0.0-Win64-Shipping.zip
```

These probes establish that those paths were unavailable, not that every
possible unpublished build is absent. Anonymous Backblaze bucket listing was
denied (HTTP 403), so it could not reveal unadvertised object names. The release
origin and naming conventions are documented in the upstream
[download sources reference](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Skills/setup/download-carla/references/sources.md).

## Available machines checked

- The local Windows 11 machine had no CARLA/Unreal process or listener on TCP
  2000/2001. No existing simulator or engine installation was found in the
  inspected project, download, and common engine directories.
- Its two local disks had approximately **60 GB and 66 GB free**. The official
  [Windows source-build guide](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/build_windows_ue5.md)
  says the engine alone needs about **225 GB**, before CARLA content and build
  outputs. A normal local source build does not fit the available space.
- A reachable Linux ARM64 machine had ample disk and RAM, but no existing
  CARLA/Unreal installation or listener on 2000/2001 was found. It does not provide
  a ready compatible host for the published Linux amd64 packages. The checked
  engine bootstrap selects x64 GitDependencies/toolchain executables and defaults
  to an x64 build host; CARLA's toolchain also contains x64-specific paths. Making
  that machine a native build host would require an additional ARM port, which
  was not attempted. NVIDIA describes the same host-architecture limitation in
  its [DGX Spark discussion](https://forums.developer.nvidia.com/t/anyone-got-unreal-engine-5-working-on-their-spark/382392/4).
- Local Docker Desktop was stopped; the remote Docker socket was unavailable to
  the current account. Starting Docker would not supply a missing UE5.8 image.

No large simulator download, engine build, system installation, or cloud resource
creation was performed during this investigation.

## What is needed to complete the live test

Provide either an existing **UE5.8 `ue58-dev` server build** that can run on the
local x64 machine, or the **host and RPC port of an existing running server**.
Also provide the client wheel produced by that exact build, compatible with the
Python version and operating system used to run the validation client. The
[official quick start](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Docs/start_quickstart.md)
requires matching client and server builds. A shared version string alone does
not prove a shared commit or engine version.

For a supplied Windows package, start its server from the extracted package root:

```powershell
.\CarlaUnreal.exe -nullrhi -nosound -carla-rpc-port=2000
```

For a supplied Linux package, from its extracted package root:

```sh
./CarlaUnreal.sh -nullrhi -nosound -carla-rpc-port=2000
```

`-nullrhi` is suitable for this physics/control validation, which must not spawn
cameras. The upstream [UE5.8 server guide](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Skills/ue58/run-carla-ue58-server/SKILL.md)
reports that spawning a camera with this flag can crash the server. A camera
validation instead needs a supported rendering configuration such as
`-RenderOffScreen` and adequate GPU resources.

From this package's root, in an environment with MyoSuite `dev` and the bridge
installed, install the supplied matching wheel and run:

```powershell
python -m pip install "C:\path\to\ue58-build\PythonAPI\carla\dist\matching-wheel.whl"
python examples/validate_carla.py --host 127.0.0.1 --port 2000 --build-ref ue58-dev --output artifacts/live-carla.json
```

Replace the wheel path and server address with actual values. Supply the server
build's source commit with `--build-commit ACTUAL_COMMIT_SHA` and keep its
engine/build metadata alongside the result. The JSON records client/server
versions and caller-supplied build identity; `provenance.source_verified` remains
false because RPC cannot independently attest the binary's branch. A
`--build-ref ue58-dev` label is a declaration, not independent proof of the
running binary's provenance. Record the actual command, result, and build identity
in [VALIDATION.md](VALIDATION.md) after running the test.

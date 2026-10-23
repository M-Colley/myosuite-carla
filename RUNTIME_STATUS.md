# CARLA runtime acquisition and UE5.8 status

Checked on **2026-10-23 (Europe/Berlin)**. The requested `ue58-dev` source has
been downloaded, and actual simulator validation passes on the available
official **CARLA 0.10.0 / UE5.5** Windows release. **No UE5.8 runtime pass is
claimed.** See [VALIDATION.md](VALIDATION.md) for the 170-test fresh-environment live result
and measured acceleration, braking and steering.

## Downloads performed

- Source: `D:/carla-ue58-dev`, branch `ue58-dev`, commit
  `defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`. Obtained with
  `git clone --depth 1 --branch ue58-dev --filter=blob:none` from the user's
  [CARLA repository link](https://github.com/carla-simulator/carla).
- Runtime: [official Windows CARLA 0.10.0 ZIP](https://downloads.carlasim.com/Windows/Carla-0.10.0-Win64-Shipping.zip),
  **10,203,028,286 bytes** compressed; 21,892,413,359 bytes expanded. Retained at
  `D:/carla-downloads/Carla-0.10.0-Win64-Shipping.zip`.
- Extracted package root:
  `D:/carla-downloads/Carla-0.10.0-Win64-Shipping/Carla-0.10.0-Win64-Shipping`.
  Its `CHANGELOG` explicitly identifies the UE5.5 migration. The matching
  CPython 3.12 Windows wheel from that package was installed into the isolated
  bridge environment `.venv-carla`; no client was substituted from PyPI.
- [Download manifest](evidence/carla-download-manifest.json) preserves paths,
  SHA-256 hashes, exact commands and the distinction between target source and
  supplementary runtime.

The repository's bundled [download procedure](https://github.com/carla-simulator/carla/blob/defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5/Skills/setup/download-carla/SKILL.md)
was used after its preflight passed:

```powershell
python Skills/setup/download-carla/scripts/download_carla.py release --version 0.10.0 --dest D:/carla-downloads --keep-archive
```

The download script printed a Linux-launcher detection warning after extraction
because it looks for `CarlaUE4.sh`; the Windows `CarlaUnreal.exe`, matching wheel
and complete extracted ZIP contents were present and successfully used.

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

- Before downloading, the local Windows 11 machine had no CARLA/Unreal process or listener on TCP
  2000/2001. No existing simulator or engine installation was found in the
  inspected project, download, and common engine directories.
- Initially its two local disks had approximately **60 GB and 66 GB free**. The official
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

The downloaded `ue58-dev` tree's build preflight confirms the engine and
`Content/Carla` assets are missing. Its bundled build guide budgets about
**400 GB** for a full build/package, versus about **61 GB** free on D: before
the supplementary runtime download. The Windows guide budgets **225 GB** for
the engine alone. The setup scripts were checked directly: they select the
correct `ue58-dev-carla` engine/content branches. Disk capacity prevents a
normal local source build; this is not an engine-access or branch-selection
problem. Saved [build preflight](evidence/carla-build-preflight.txt) and
[download preflight](evidence/carla-download-preflight.txt) retain the outputs.
The Linux-oriented preflight's D: device-class warning is not evidence that the
Windows drive is remote. No UE engine build or cloud resource was started.

## What is needed to complete the live test

Completing the target-specific test needs a runnable **UE5.8 `ue58-dev` server
build**, or sufficient x64 build storage for its engine, content and outputs,
plus the matching client wheel. The wheel must be compatible with the Python
version and operating system used to run the validation client. The
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

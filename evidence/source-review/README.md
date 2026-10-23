# Native CARLA review evidence

These records validate actual CARLA `ue58-dev` client source and production
transport classes. They do not start Unreal Engine or establish UE5.8 simulator
runtime compatibility. Source base: `defabe35d1ad47c4231d4f7fb1f4e46ac2e08ec5`.
See [the validation record](../../VALIDATION.md) for patch identities and scope.

- `future-atomic-{tsan,asan-ubsan}.log`: 19 production-header cases repeated
  ten times under each sanitizer configuration, 190 case executions per file.
- `stream-{tsan,asan-ubsan}.log`: 11 production streaming cases repeated
  twenty times, 220 case executions per file. Both the production subset and
  test executable are instrumented.
- `stream-windows-debug-repeat.log`: the same 11 cases, twenty repetitions
  under native Windows MSVC Debug. The accompanying CTest XML records its
  wrapper test; the log contains all individual GoogleTest results.
- `stream-getter-benchmark.json`: independently built baseline/fixed Release
  executables interleaved on one pinned CPU. This is getter cost, not simulator
  throughput. Benchmark source and procedure are in PR #9997.
- `client-baseline-failure.json`: the original full native run reached an RPC
  assertion and hung in cleanup; only the owned test process was terminated.
- `client-tests-remaining.xml`: the other 269 original native cases, with
  268 passing and one Windows wildcard-address TCP failure. The RPC test was
  explicitly excluded from this diagnostic run.
- `network-test-fixed-repeat.log`: both corrected native network cases run
  five times each, ten case executions. The accompanying GoogleTest XML
  contains the final repetition, as GoogleTest normally overwrites it.
- `episode-{tsan,asan-ubsan}.log`: seven genuine production-client RPC/TCP
  cases repeated twenty times, 140 case executions per file. The client and
  dependencies are instrumented. `episode-build-manifest.txt` records exact
  commands; the two `baseline-*.log` files reproduce original-source races.
- `python-client-version-{original,fixed}.log`: the original Git-tag lookup
  fails in the shallow checkout; the metadata-based test passes against the
  real production extension and installed wheel metadata. This focused check
  used the provisional wheel, before the final integrated client rebuild.
- `client-tests-integrated.xml` and `python-api-tests-integrated.xml`: the
  unfiltered final source suites, 306 and 92 cases respectively, with no
  failures or skips. The Python suite includes all three native wheel fixtures.
- `bridge-numpy{1,2}-tests-integrated.xml`: final clean-clone bridge runs with
  the rebuilt wheel, each 240 passing cases plus three explicit live skips.
- `native-wheel-bridge-smoke-numpy{1,2}.json`: actual muscle/contact inputs
  converted through the real native CARLA control/command bindings, without
  an Unreal connection. The `.py` file preserves the exact local check;
  its import-path assertions identify this machine's pinned clean clones.
- `build-manifest.json`: final source/dependency revisions, hashes, commands,
  tool versions, actual import paths and test summaries. Exploratory runs from
  the older local MyoSuite worktree are excluded from these final results.

The sanitizer logs use GCC 13 on Linux ARM64; native Windows tests use MSVC
19.51. Focused GitHub workflows additionally exercise Ubuntu x64. All timings
are machine-specific measurements, not general performance guarantees.

# Final test collection recovery

The test collector ended on 18 November 2026 at 09:04:37 UTC when the CARLA RPC client reported `resource unavailable try again` during the 35th attempt in seed 31. Seed 30 completed. Seed 32 had not started. Windows reported 3.13 GiB of available virtual memory at the subsequent check. The source of the RPC error was undetermined. The initial recording root, including the partial seed 31 trial, is retained. No final prediction metrics had been computed at this point. [Interrupted run](checks/test-interrupted-run.json) [Collector log](checks/test-interrupted-collection.txt)

The recovery retains seed 30 byte for byte. A new server process uses the same CARLA 1.0 package, map and launch flags. The collector repeats all 47 trials of seed 31 with the same seed and the unchanged stimuli and capture rules. Seed 32 uses its original plan. Each seed has one complete batch in the analysis directory. The interrupted seed contributes no trials to that analysis. The original model and acceptance checks remain fixed. A completed batch that fails coverage prevents acceptance. [Frozen selection](frozen-selection.json)

Extract `test.zip` into a directory named `test`. Extract `test-recovery.zip` into a separate directory named `test-recovery`. The commands assume the paths used in the main reproduction instructions. The assembly script verifies the original and recovered programme metadata for the common seed 31 prefix. Its output directory includes one copy of each complete seed and a manifest with every file checksum. [Interrupted data](data/interrupted-test) [Recovery data](data/recovered-test)

```powershell
python evidence/carla-1.0-vehicle-calibration-2026-11-17/assemble_test_view.py --interrupted artifacts/carla-1.0-calibration-2026-11-17/test --recovery artifacts/carla-1.0-calibration-2026-11-17/test-recovery --out artifacts/carla-1.0-calibration-2026-11-17/test-view
python examples/evaluate_vehicle_calibration.py --design artifacts/carla-1.0-calibration-2026-11-17/design-v2.json --data artifacts/carla-1.0-calibration-2026-11-17/development --test-data artifacts/carla-1.0-calibration-2026-11-17/test-view --out artifacts/carla-1.0-calibration-2026-11-17/analysis --stage test
```

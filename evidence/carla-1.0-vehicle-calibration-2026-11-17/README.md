# CARLA 1.0 vehicle calibration

The experiment identifies the Nissan Patrol surrogate against the official CARLA 1.0 Windows runtime. The map is `Carla/Maps/Town10HD_Opt`. The simulation step is 0.02 s. The runtime package, matching client wheel, source identity and installation checks are recorded in [the runtime validation evidence](../carla-1.0-myosuite-3.1-2026-11-17).

The [design](design.json) fixes the manoeuvres, exclusions, partitions, candidate models, selection criterion and acceptance checks. Commit `d8db34b10fd6420fb7769b46450f6efadd2f79d3` published this design before the first designed drive. Its SHA256 is `ec7f5837ec37247ec8698e62ffff20527516321d22400709c3a85753fa32c346`. Training uses seeds 10, 11 and 12. Validation uses seeds 20 and 21. Final evaluation uses seeds 30, 31 and 32. Every seed requests 47 trials across eight manoeuvre families.

The original design failed its coverage requirement before fitting. Seed 21 retained one of four steering hold trials. Two were required. All three rejected holds exceeded 3 degrees of body roll without a reported collision. The complete batch remains in the evidence. No model had been fitted and no validation prediction errors had been inspected.

The [amended design](design-v2.json) replaces the complete validation batch with seed 22. Its SHA256 is `29d8f669567a55716789b923ee798c40d59473db66e2cff301ef23d975245681`. The ordered validation reserves are 22, 23 and 24. Replacement depends on the original integrity and coverage rules. The first complete reserve batch satisfying those rules is used. Fitting stops if all reserves fail. Collector settings, trial counts, training seeds, candidate models, scoring and acceptance criteria remain fixed. Test seeds remain 30, 31 and 32. A test coverage failure prevents acceptance. Accepted seed 21 trials receive a descriptive evaluation after the selected candidate is published. This diagnostic cannot change selection or acceptance.

The evidence directory preserves exact file bytes through Git attributes. Initial commits normalized protocol line endings. The retained protocol files restore the exact bytes used for their recorded SHA256 checksums. Their parsed contents remain unchanged.

The collector retains every attempted recording. Collisions, absolute body roll above 3 degrees and incomplete trials are excluded under the fixed collection rule. Body roll can arise during ordinary cornering. Therefore, these exclusions do not establish curb contact. Each seed must retain at least half of its requested trials in every family. Steering stimuli use a linear input assumption and a nominal lateral acceleration ceiling of 3 m/s². Candidate gains remain conditional on the fixed nominal wheelbase of 3.075 m.

The fitter uses the three training batches. Eight predefined steering candidates share one longitudinal fit. This reuse avoids repeating an identical optimization. Commit `b16084f016987c7f0f0be9a23e2f2d9fe10c8473` added the reuse during development collection, before fitting or inspecting validation outcomes. Forward and reverse drive acceleration increase monotonically with throttle. Rollouts use predicted speed and zero road pitch. Recorded future pitch is unavailable during operational surrogate prediction and is excluded from fitting and evaluation.

The primary error is `0.5 * ((speed_RMSE / 1 m/s)^2 + (yaw_rate_RMSE / 5 deg/s)^2)`. Trial errors are averaged within each manoeuvre family. The eight family means have equal weight within a seed. Seed means also have equal weight. Validation selects the candidate with the lowest primary error. Selection requires improvement over the retained CARLA 0.10 preset.

Final evaluation compares the selected candidate with the retained preset on three fresh seed batches. The paired primary differences use a percentile bootstrap with 5,000 resamples and seed 20261117. The sampling unit is the seed batch. Acceptance requires the upper 95% interval bound below zero. Mean speed error, mean yaw rate error and mean final position error must each improve or remain equal. Each manoeuvre family must also satisfy `candidate_primary <= 1.25 * legacy_primary + 0.01`. Three batches provide pilot descriptive uncertainty. The experiment covers one vehicle and one map. Human controller parameters are evaluated separately.

## Reproduction

Install the dependencies and the matching CARLA client described in the runtime validation evidence. Start the server before collecting data. The following commands assume an empty `artifacts/carla-1.0-calibration-2026-11-17` data root with a copy of `design.json`. Each collector output directory must be empty. Analysis reports cannot be overwritten.

```powershell
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 10 11 12 20 21 --launch 10 --hand-brake 2 --reverse 3 --steer 12 --steer-fast 6 --steer-low 4 --steer-hold 4 --steer-doublet 6 --min-straight 100 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data artifacts/carla-1.0-calibration-2026-11-17/development
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 22 --launch 10 --hand-brake 2 --reverse 3 --steer 12 --steer-fast 6 --steer-low 4 --steer-hold 4 --steer-doublet 6 --min-straight 100 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data artifacts/carla-1.0-calibration-2026-11-17/validation-reserve-22
python examples/evaluate_vehicle_calibration.py --design artifacts/carla-1.0-calibration-2026-11-17/design-v2.json --data artifacts/carla-1.0-calibration-2026-11-17/development --supplement-data artifacts/carla-1.0-calibration-2026-11-17/validation-reserve-22 --test-data artifacts/carla-1.0-calibration-2026-11-17/test --out artifacts/carla-1.0-calibration-2026-11-17/analysis --stage fit
```

Publish the fit report and selected candidate before collecting the final test batches. The fitting command refuses to run if any final test seed directory already exists.

```powershell
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 30 31 32 --launch 10 --hand-brake 2 --reverse 3 --steer 12 --steer-fast 6 --steer-low 4 --steer-hold 4 --steer-doublet 6 --min-straight 100 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data artifacts/carla-1.0-calibration-2026-11-17/test
python examples/evaluate_vehicle_calibration.py --design artifacts/carla-1.0-calibration-2026-11-17/design-v2.json --data artifacts/carla-1.0-calibration-2026-11-17/development --test-data artifacts/carla-1.0-calibration-2026-11-17/test --out artifacts/carla-1.0-calibration-2026-11-17/analysis --stage test
```

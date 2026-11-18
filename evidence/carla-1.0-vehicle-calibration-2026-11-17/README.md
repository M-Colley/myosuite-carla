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

Install the `analysis` extra and the matching CARLA client described in the runtime validation evidence. Start the server before collecting data. The following commands assume a new `artifacts/carla-1.0-calibration-2026-11-17` data root with an exact copy of `design-v2.json`. Each collector output directory must be empty. Analysis reports cannot be overwritten.

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

The original final collection ended during seed 31 after a CARLA RPC error. The recovery retains the completed seed 30 and repeats the complete seed 31 with the same stimuli. It includes the planned seed 32. Use the recovery instructions to reproduce the evaluation from the archived recordings. [Recovery procedure and evidence](test-recovery.md)

## Recorded data

The repository stores the native recordings for every attempt, including rejected trials. The development archives contain the full original run and validation reserve 22. Extract each ZIP into a directory named after the ZIP stem. Place the directories in the data root used by the commands above. For analysis reproduction, extract the test archive after completing selection. The source distribution includes the reports, manifest and analysis code. The data archives are available in the repository. [Development data and checksums](data/development)

The published fit report preserves all eight validation evaluations. The frozen selection records the full parameter values and the SHA256 of that report. Commit `74b5889454852708f24c312a98b01dd8fd58b497` was pushed before collection started for seed 30. [Fit report](fit-report.json) [Frozen selection](frozen-selection.json)

Candidate C3 had the lowest validation primary error (0.20235). The legacy preset scored 0.47715. This corresponds to a 57.6% reduction across two validation batches and 79 retained trials. The selected exponent was 0.9711. The steering rate limits were 2.5 and 5.0 per second. The fitted lag increased by 0.01 s for each m/s of speed. Training used 125 trials and 1,079 s of data. The 5th and 95th percentiles of speed during steering were 1.34 and 11.07 m/s. Peak forward or reverse speed was 17.94 m/s. Peak reverse throttle was 0.572.

Seed 21 had a primary error of 0.21191 with C3 and 0.51281 with the legacy preset across 38 retained trials. The same scoring covered all eight observed manoeuvre families. Steering hold was represented by one trial. This diagnostic used the published selection and had no role in selection or promotion. [Diagnostic code](diagnose_seed21.py) [Diagnostic results](seed21-diagnostic.json)

The scripted following check exercised the selected vehicle through MyoSuite 3.1.0 and development commit `c93329ec64bc8b544d0b3427bba6766377c3528e`. Each environment completed two 2 s episodes using the 114 muscle body and a matching acceleration inverse. All episodes reached the full duration and released their vehicles. The check measures execution of the physical control loop with a scripted lead vehicle. [Verification script](verify_embodied_following.py) [Release report](embodied-following-0.json) [Development report](embodied-following-1.json)

## Final test results

C3 passed all 11 metric and family checks with three eligible test batches. They retained 42, 46 and 44 trials respectively (132 of 141 total, 1,116 s). Primary error fell from 0.51886 to 0.31086 (40.1%). The mean paired change was −0.20800 (95% percentile interval [−0.22406, −0.18333]). Speed RMSE fell from 0.537 to 0.464 m/s. Yaw rate RMSE fell from 2.834 to 1.214°/s. Final position error fell from 3.378 to 2.816 m. Launch primary error increased from 0.517 to 0.606 and stayed below the prespecified limit of 0.657. The other seven families improved. [Exact test report](test-report.json) [Paired scores](figures/validation-and-paired-test.svg) [Family results](figures/test-family-and-metrics.svg)

The position error at 3 s was 0.404 m for C3, 0.413 m for the legacy preset and 5.525 m for constant velocity extrapolation (means weighted by trial duration). The constant velocity baseline starts near rest and serves as a secondary comparison for these programmes. Test steering samples above 0.5 m/s and absolute steer above 0.02 had speed percentiles of 0.98 and 9.98 m/s (5th and 95th percentiles). Maximum observed vehicle speed was 21.29 m/s. The largest reverse throttle was 0.556. Three seed batches provide pilot uncertainty for the tested map and vehicle. The estimates apply to trials that met the fixed collection rules.

The SHA256 of the test report is `3bd8ba47750af17141d964d7dbc21deb55ad2003402d6a562a1d66048d7f3814`. The analysis directory assembled seed 30 from the interrupted run and seeds 31 and 32 from the recovery run. The file manifest records each source and verifies its exact copy. The final world was asynchronous with no vehicle or collision sensor. The recovery server was stopped after that check. [Assembly manifest](test-view-manifest.json) [World cleanup](checks/after-test-world.json)

Across development, validation, test and recovery, the four archives preserve 458 attempts, including the interrupted trial. Each ZIP has a member manifest and CRC verification. [Development](data/development) [Interrupted test](data/interrupted-test) [Recovered test](data/recovered-test)

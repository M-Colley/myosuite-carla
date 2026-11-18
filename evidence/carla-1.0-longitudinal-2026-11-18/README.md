# Targeted CARLA 1.0 longitudinal refinement

The previous calibration reduced overall error. However, two high speed launches had speed RMSE above 3 m/s and final position error above 26 m. The prospective experiment evaluates a small longitudinal correction against the frozen C3 model. The [design](design.json) was committed before the first designed drive in commit `5e8718d`. Its SHA256 is `82c46962c62ab14aeaf96355e5d50761c0cfd13a0b67ff1dfb4b5dcdd8c60598`.

The experiment uses the official CARLA 1.0 package and `vehicle.nissan.patrol`. Town12 provides the required road distance. The map is `Carla/Maps/Town12/Town12`. Spawn indices 236, 252, and 305 have at least 350 m of eligible road under a 600 m search limit. Eligible waypoints stay within 3 degrees of the initial yaw and 1 degree of road pitch. The simulation period is 0.02 s.

Each seed requests 12 complete trials. The balanced grid combines target speeds of 17, 19.5, and 22 m/s, throttle levels of 0.75 and 1, and service brake levels of 0.4 and 0.8. Each trial starts with 0.5 s at rest. It accelerates to the assigned speed, coasts for 1.5 s, and brakes to rest. Remaining road distance reserves stopping room. A trial that reaches a distance or time limit before completing its target and coast is excluded. Collisions, absolute body roll above 3 degrees, and incomplete trials are also excluded. Every attempt is preserved.

Training uses seeds 1100 and 1101. Validation uses seeds 1200 and 1201. Final testing uses seeds 1300, 1301, and 1302. Every seed must retain at least eight trials, two trials per target, one trial per target and throttle combination, and three trials per brake level. Coverage failure prevents selection or acceptance. No reserve seeds are available. A documented RPC interruption before prediction metrics are inspected permits a complete replay of the interrupted seed. Partial trials cannot contribute to the analysis view.

The fit uses SciPy bounded least squares on complete speed rollouts. Five parameters adjust positive drive gains at 15 and 20 m/s, quadratic drag, service braking, and throttle lag. Drive gains stay at one through 8 m/s. Positive gains preserve throttle monotonicity. Steering, reverse drive, rolling resistance, linear drag, hand brake deceleration, and interpolation knots retain their C3 values. Future measured speed and pitch are excluded. The regularization, bounds, optimizer settings, and single initial condition are fixed in the design.

Trial speed MSE has equal weight within each target. Target means have equal weight within each seed. Seed means also have equal weight. Validation requires lower primary error and no worse pooled 95th percentile trial speed RMSE. Final acceptance requires a negative upper bound of the paired seed percentile bootstrap interval. Mean speed RMSE, the pooled 95th percentile, maximum trial speed RMSE, and mean final position error must also remain equal or improve. Each target must satisfy `candidate MSE <= 1.1 * baseline MSE + 0.01`.

The resulting configuration remains explicit. This experiment evaluates forward longitudinal behavior on one highway map. Three final seeds provide pilot uncertainty. Reverse braking and low speed embodied control require separate retention checks before changing general presets.

## Reproduction

Install this checkout with its analysis dependencies and the matching CARLA client. Start the matching server on Town12. Run the collector and fitter from this checkout. Keep the original calibration evidence because it supplies the frozen C3 baseline. The fitting command requires the final test directory to remain uncollected.

```powershell
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 1100 1101 1200 1201 --launch 12 --launch-targets 17 19.5 22 --launch-throttles 0.75 1 --launch-brakes 0.4 0.8 --launch-coast-seconds 1.5 --hand-brake 0 --reverse 0 --steer 0 --steer-fast 0 --steer-low 0 --steer-hold 0 --steer-doublet 0 --min-straight 350 --straight-limit 600 --spawn-indices 236 252 305 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data artifacts/carla-1.0-longitudinal-2026-11-18/development
python examples/calibrate_longitudinal.py --design evidence/carla-1.0-longitudinal-2026-11-18/design.json --data artifacts/carla-1.0-longitudinal-2026-11-18/development --test-data artifacts/carla-1.0-longitudinal-2026-11-18/test --out artifacts/carla-1.0-longitudinal-2026-11-18/analysis --stage fit
```

The fitter writes `fit-report.json` and `frozen-selection.json`. Publish them before final collection. Final testing requires `selected: true`. Reports and source data cannot be overwritten. Source checksums normalize line endings to LF. Collector, baseline, protocol, and report checksums preserve their exact bytes.

```powershell
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 1300 1301 1302 --launch 12 --launch-targets 17 19.5 22 --launch-throttles 0.75 1 --launch-brakes 0.4 0.8 --launch-coast-seconds 1.5 --hand-brake 0 --reverse 0 --steer 0 --steer-fast 0 --steer-low 0 --steer-hold 0 --steer-doublet 0 --min-straight 350 --straight-limit 600 --spawn-indices 236 252 305 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data artifacts/carla-1.0-longitudinal-2026-11-18/test
python examples/calibrate_longitudinal.py --design evidence/carla-1.0-longitudinal-2026-11-18/design.json --data artifacts/carla-1.0-longitudinal-2026-11-18/development --test-data artifacts/carla-1.0-longitudinal-2026-11-18/test --out artifacts/carla-1.0-longitudinal-2026-11-18/analysis --stage test
```

Coverage failure writes an ineligible fit report and stops before optimization. An additional `--diagnostic` fit in a separate fresh output directory can retain exploratory fit diagnostics from complete accepted trajectories. Diagnostic fitting always leaves `selected: false`. Missing target support is reported explicitly. Diagnostic results cannot authorize final testing or change the fixed fit settings.

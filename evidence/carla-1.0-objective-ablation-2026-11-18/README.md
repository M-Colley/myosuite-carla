# Longitudinal fitting objective comparison

The comparison fits two objectives to the same 24 published Town12 training drives. Both fits start from C3 and vary the same five parameters within identical bounds. They use the same prior and trajectory weighting. The derivative objective minimizes one step acceleration error on the original moving samples. The full trajectory objective minimizes speed error across the simulated drive. Its objective includes standstill samples and subsequent transitions. Both sampling masks are part of the comparison. The earlier validation drives provide descriptive diagnostics. Commit `a11583b` published the fitted parameters before the moving probe collection. [Design](design.json) [Fit report](fit-report.json) [Frozen parameters](frozen-models.json)

The final trial grid crosses two entry speeds (17 and 20 m/s), two approaches (acceleration from rest or braking from 22 m/s) and two initial probe commands (0.75 throttle or 0.4 brake). Each of three fresh seed batches requests all eight conditions. The collector retains the approach and stop for every attempt. Models receive the same measured state at probe entry, the same preceding throttle history and the same subsequent controls. The primary measure is speed MSE over 3 s. C3 provides a fixed reference. [Full measurement design](design.json)

Each seed receives equal weight. The paired percentile bootstrap resamples the seed batches. Acceptance requires the upper interval bound below zero and the predefined coverage, position, tail error and condition limits. This study tests the fitting objectives within one parameterization. It uses one car and three routes in Town12.

## Results

All 24 final probes met the coverage checks. Each seed batch covered all eight conditions. Mean 3 s speed MSE was 0.05864 (m/s)² for the derivative fit and 0.09413 (m/s)² for the full trajectory fit. The increase was 0.03549 (m/s)² (95% paired seed batch percentile bootstrap interval [0.03522, 0.03568], 5,000 resamples). The error increased in every seed batch. Mean 3 s position error increased from 0.511 to 0.569 m. The 95th percentile of trial speed RMSE increased from 0.310 to 0.539 m/s. The maximum trial speed RMSE increased from 0.329 to 0.552 m/s. The full trajectory fit failed the prospective acceptance rule.

C3 produced a mean 3 s speed MSE of 0.56238 (m/s)² and a mean 3 s position error of 1.548 m. Both fitted models improved on C3 in this trial grid. C3 served as the descriptive reference. The primary comparison was between the two fitted objectives. The protocol varied the training loss and its prespecified sampling mask together, so the test cannot estimate their separate effects. The final evidence covers three seed batches on the recorded Town12 routes.

The live collector used the official CARLA 1.0 Windows package and matching Python client. It completed all three batches and restored asynchronous world settings. The final server probe counted zero vehicle actors and zero collision sensors. [Report](test-report.json) [Raw recordings](data/test/test.zip) [Archive checksums](data/test/manifest.json) [Collection log](checks/collection.txt) [World cleanup](checks/after-world.json)

## Reproduction

Use the CARLA 1.0 Windows server and matching client described in [the runtime record](../carla-1.0-myosuite-3.1-2026-11-17). Load `Town12` before collection. Extract the earlier [development archive](../carla-1.0-longitudinal-2026-11-18/data/development/development.zip) into `historical-development`. The archive includes the required seed folders. Use a fresh analysis directory.

```powershell
python examples/ablate_longitudinal.py --design evidence/carla-1.0-objective-ablation-2026-11-18/design.json --data historical-development --test-data moving-test --out moving-analysis --stage fit
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 2100 2101 2102 --launch 8 --probe-speeds 17 20 --probe-histories acceleration braking --probe-actions throttle brake --hand-brake 0 --reverse 0 --steer 0 --steer-fast 0 --steer-low 0 --steer-hold 0 --steer-doublet 0 --min-straight 350 --straight-limit 600 --spawn-indices 236 252 305 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data moving-test
python examples/ablate_longitudinal.py --design evidence/carla-1.0-objective-ablation-2026-11-18/design.json --data historical-development --test-data moving-test --out moving-analysis --stage test
```

Publish both fits before collecting the final data. For analysis reproduction from archives, extract the moving test data after fitting. The test stage requires the exact fit report and frozen parameter digest.

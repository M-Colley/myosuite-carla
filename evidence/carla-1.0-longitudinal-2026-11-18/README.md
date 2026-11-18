# High speed longitudinal calibration

The study tests the Nissan Patrol surrogate above the speed range of the original dynamics fit. Two launch trials in the earlier Town10 evaluation reached about 19 m/s and exceeded 26 m of final position error. Town12 provides the stopping distance required by the present protocol. The three selected routes each provide at least 600 m within the steering and grade limits used for collection. Client and server reported CARLA 1.0. The control step was 0.02 s. [Design](design.json) [Earlier results](../carla-1.0-vehicle-calibration-2026-11-17/test-report.json)

Each seed attempts twelve trials. The programmes cross three speed targets (17, 19.5 and 22 m/s), two throttle levels (0.75 and 1.0) and two service brake levels (0.4 and 0.8). Each trial starts from rest, accelerates to its requested speed, coasts for 1.5 s and brakes to rest. The collector retains all attempts and applies the fixed exclusion rules. The development run retained all 48 trials. The first 24 trials from seeds 1100 and 1101 were used for fitting. The 24 trials from seeds 1200 and 1201 were reserved for validation.

The fit minimizes errors over the speed trajectory. It adjusts five parameters and preserves the C3 steering and reverse drive models. The corrections to drive magnitude depend on speed and preserve monotonicity in throttle. The candidate was selected after validation mean speed RMSE decreased from 1.825 to 1.189 m/s and the 95th percentile decreased from 3.123 to 1.656 m/s. Two fitted corrections reached their lower bounds. Their numerical values are specific to this calibration. Commit `61f328a` published all fitted parameters before the final test collection. [Fit report](fit-report.json) [Frozen selection](frozen-selection.json)

The procedure compares the original C3 model with a correction fitted on new calibration data. A separate experiment with both estimators trained on the same recordings would isolate the effect of the fitting objective. The five parameter correction applies to the tested high speed forward conditions. C3 remains the general preset. The source distribution includes reports and scripts. Native trial archives are available in the repository.

## Reproduction

Use the official CARLA 1.0 Windows package and its bundled client. Load `Town12` before collecting the data. The following commands assume the bridge root and a new output directory.

```powershell
python examples/identify_vehicle.py --host 127.0.0.1 --port 2000 --timeout 60 --seeds 1100 1101 1200 1201 --launch 12 --launch-targets 17 19.5 22 --launch-throttles 0.75 1 --launch-brakes 0.4 0.8 --launch-coast-seconds 1.5 --hand-brake 0 --reverse 0 --steer 0 --steer-fast 0 --steer-low 0 --steer-hold 0 --steer-doublet 0 --min-straight 350 --straight-limit 600 --spawn-indices 236 252 305 --max-road-pitch 1 --max-lateral-acceleration 3 --steer-exponent 1 --data artifacts/carla-1.0-longitudinal-2026-11-18/development
python examples/calibrate_longitudinal.py --design evidence/carla-1.0-longitudinal-2026-11-18/design.json --data artifacts/carla-1.0-longitudinal-2026-11-18/development --test-data artifacts/carla-1.0-longitudinal-2026-11-18/test --out artifacts/carla-1.0-longitudinal-2026-11-18/analysis --stage fit
```

Publish the validation decision and parameters before collecting the final test batches. Use the same collector command with seeds 1300, 1301 and 1302 and the output directory `artifacts/carla-1.0-longitudinal-2026-11-18/test`. Use the analysis command with `--stage test`. To reproduce from native archives, extract each ZIP into a directory named after its stem. Complete selection before extracting the test archive.

## Final results

The correction passed all predefined checks on 36 final trials (three seed batches). All trials were retained. Primary speed MSE fell 66.1%. The mean paired change was −3.016 (m/s)² (95% seed batch percentile interval [−3.036, −3.003], 5,000 resamples). The three batches provide a pilot estimate for the tested vehicle, routes and control programmes. [Test report](test-report.json)

| Metric | C3 | Longitudinal correction |
| --- | ---: | ---: |
| Primary speed MSE ((m/s)²) | 4.560 | 1.544 |
| Mean trial speed RMSE (m/s) | 1.830 | 1.192 |
| 95th percentile trial speed RMSE (m/s) | 3.092 | 1.672 |
| Maximum trial speed RMSE (m/s) | 3.129 | 1.696 |
| Mean final position error (m) | 14.576 | 9.956 |

The profile is available for explicit use in the tested forward longitudinal conditions. The frozen C3 profile covers the general vehicle integration. The following code constructs a surrogate and a matching acceleration inverse from the accepted data file.

```python
import json
from pathlib import Path

from myosuite_carla.driver_models import AccelerationControlConfig
from myosuite_carla.vehicle_model import SurrogateVehicle, VehicleModelParams

path = Path("evidence/carla-1.0-longitudinal-2026-11-18/accepted-parameters.json")
params = VehicleModelParams.from_dict(json.loads(path.read_text(encoding="utf-8")))
vehicle = SurrogateVehicle(params)
inverse = AccelerationControlConfig(vehicle=params)
```

The archives contain all 84 attempted runs. CRC and member SHA256 checks verified both archives. The CARLA server returned to asynchronous mode and had no vehicle or collision sensor after collection. The experiment process was stopped. [Development archive](data/development) [Final archive](data/test) [Cleanup probe](checks/after-world.json)

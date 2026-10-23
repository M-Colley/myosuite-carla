from importlib.metadata import version
from pathlib import Path
import json
import math
import sys

import carla
import myosuite
import myosuite_carla
import numpy as np
from myosuite_carla import VehicleCommand
from myosuite_carla.validation import build_muscle_driver

assert Path(myosuite.__file__).resolve().is_relative_to(
    Path('C:/Users/Mark/Desktop/myosuite-carla/test-results/fresh-pc/myosuite').resolve())
assert Path(myosuite_carla.__file__).resolve().is_relative_to(
    Path('C:/Users/Mark/Desktop/myosuite-carla/src').resolve())

results = []
for control in ('throttle', 'brake', 'steering'):
    driver = build_muscle_driver(control)
    for _ in range(30):
        driver.advance(0.05)
    command = driver.mapper.sample(driver.data).command
    native = carla.VehicleControl(**command.as_dict())
    request = carla.command.ApplyVehicleControl(42, native)
    assert request.actor_id == 42
    for field, expected in command.as_dict().items():
        actual = getattr(request.control, field)
        if isinstance(expected, float):
            assert math.isclose(actual, expected, abs_tol=1e-6), (field, actual, expected)
        else:
            assert actual == expected
    channel = 'steer' if control == 'steering' else control
    assert abs(getattr(native, channel)) > 0.1
    assert driver.peaks['contact_newtons'] > 10
    assert driver.peaks['muscle_newtons'] > 100
    results.append({'mechanism': control, 'vehicle_command': command.as_dict(),
                    'native_control': str(native), 'peaks': driver.peaks})

for gear in (-(2**31), 2**31 - 1):
    native = carla.VehicleControl(**VehicleCommand(gear=gear).as_dict())
    assert native.gear == gear

report = {
    'status': 'passed',
    'scope': 'Offline real MyoSuite/MuJoCo muscle contact -> bridge mapping -> compiled CARLA VehicleControl and ApplyVehicleControl bindings',
    'unreal_server_connected': False,
    'module': carla.__file__,
    'myosuite_module': myosuite.__file__,
    'myosuite_carla_module': myosuite_carla.__file__,
    'client_version': carla.Client('127.0.0.1', 1).get_client_version(),
    'metadata_version': version('carla'),
    'versions': {name: version(name) for name in ('numpy', 'mujoco', 'myosuite', 'myo-sim', 'myosuite-carla')},
    'python': sys.version,
    'finite_muscle_results': all(np.isfinite(list(result['peaks'].values())).all() for result in results),
    'mechanisms': results,
}
output = Path(sys.argv[1])
output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps(report, indent=2, allow_nan=False))

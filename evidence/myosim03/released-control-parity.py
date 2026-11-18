import hashlib
import json
import sys

from myosuite_carla import DriverIntent, FullBodyDriverConfig, build_fullbody_driver

results = {}
for mode in ('default', 'fast'):
    cfg = None if mode == 'default' else FullBodyDriverConfig.fast()
    driver = build_fullbody_driver(cfg)
    driver.reset(seed=0)
    outputs = []
    for intent in (DriverIntent(), DriverIntent(steer=0.14, right_foot=0.03), DriverIntent()):
        for _ in range(20):
            driver.advance(intent)
            outputs.append(driver.mapper.sample(driver.data).values)
    results[mode] = {
        'actuators': int(driver.model.nu),
        'gaps': driver.initial_gaps,
        'pose_sha256': hashlib.sha256(driver.rest_qpos.tobytes()).hexdigest(),
        'trace_sha256': hashlib.sha256(json.dumps(outputs, sort_keys=True).encode()).hexdigest(),
    }
print(json.dumps(results))

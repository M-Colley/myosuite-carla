# Rate limit with a speed-dependent steering lag: design fixed before fitting

Written on 2026-11-04 before the fit was run and before any of the test drives
were recorded.

## Question

Two tests of Chaos's steering input rate limit kept the default. On seeds 14-17
the engine rates with the default lag (0.039 s) made one-period low-speed
sweeps 8% worse. On seeds 18-21 a jointly fitted constant lag of 0.030 s
improved every low-speed kind but made 8-14 m/s sweeps 39% worse. Fits split
by speed gave 0.026 s below 3 m/s and 0.038 s above. This run tests whether a
steer time constant that grows linearly with speed resolves both.

## Model

The surrogate's steer time constant is now `steer_lag + steer_lag_slope |u|`,
with `u` the speed before each frame. A zero slope (the default, and what
parameters saved earlier load as) reproduces the constant lag exactly; the
archived analysis of seeds 18-21 reran identically with this code. The slope
stands in for yaw dynamics that slow with speed, which the kinematic model
does not otherwise represent.

## Candidate

`carla_nissan_patrol()` with the engine's rate limit (rise 2.5, fall 5.0 per
second; the joint fit on seeds 0-17 gave 2.5 with interval 2.45-2.6) and with
`steer_lag` and `steer_lag_slope` fitted jointly. The gain table, exponent and
longitudinal parameters stay at the default.

## Fit (`fit.py`)

- Training: the steering segments (`steer*` kinds) of seeds 0, 3, 10, 12 and
  14-21. Seeds 18-21 were spent as the last test and now join training.
  Seeds 1, 2, 4, 11 and 13 are not used.
- Objective: as in the lateral fit, the pooled squared yaw-rate residual with
  the measured speed over frames above 0.5 m/s. The lag uses the measured speed
  before each frame.
- Search: a grid of `steer_lag` 0-0.06 s in steps of 0.002 s and
  `steer_lag_slope` 0-0.008 s per m/s in steps of 0.0002; the grid minimum is
  the candidate. The library does not allow negative slopes.
- Uncertainty: 1000 bootstrap resamples of training segments (seed 20261104);
  95% percentile intervals.
- Check of the fast path: at a point off the grid (lag 0.0305 s, slope 0.0021)
  its per-segment errors must match `evaluate_vehicle_model(...,
  measured_speed=True)` to 1e-9 deg/s.
- Prediction (descriptive): the slope's 95% interval excludes zero.

The fitted candidate is committed before the test drives are recorded.

## Test data

- Server: the CARLA 0.10.0 / UE5.5 package in
  `C:/Users/localadmin/Desktop/Carla-0.10.0-Win64-Shipping` (VERSION: CARLA
  `ada75f920642e18cace0a9f85ecf9d4077ddb531`), started with `-nullrhi -nosound
  -unattended -notrace -carla-rpc-port=2000` once at least 9 GB of commit
  memory is free, default map, client wheel 0.10.0.
- Vehicle: `vehicle.nissan.patrol`.
- Command:
  `examples/identify_vehicle.py --seeds 22 23 24 25 --launch 0 --hand-brake 0
  --reverse 0 --steer 4 --steer-fast 4 --steer-low 8 --steer-hold 8
  --steer-doublet 10 --data artifacts/steer-lag-speed`.
- Segments with a collision or a roll above 3 degrees are discarded, as
  before.

## Adoption rule (`analyse.py`)

Unchanged from the previous test (`0c0acf0`), on seeds 22-25. All errors are
duration-weighted means of per-segment RMSE from `evaluate_vehicle_model`;
measured-speed errors use `measured_speed=True`. The candidate replaces
`carla_nissan_patrol()` only if all of the following hold, each relative to
the current default:

- Control: the default's open-loop yaw-rate RMSE on the fresh `steer` sweeps is
  at most 2 x 0.337 deg/s.
- (a) Measured-speed yaw-rate RMSE on the pooled fresh `steer_low`,
  `steer_hold` and `steer_doublet` segments falls by at least 25%.
- (b) It falls in each of these three kinds separately.
- (c) For the fresh `steer` and `steer_fast` sweeps, it rises by at most 5% in
  each kind.
- (d) Open-loop yaw-rate RMSE on the pooled fresh segments of (a) does not rise.
- (e) On the original held-out seeds 1, 2 and 4, the open-loop yaw-rate RMSE
  and the 3 s position error of each manoeuvre group rise by at most 5%.
- (f) On the four muscle-driven recordings, open-loop yaw-rate RMSE and 3 s
  position error rise by at most 5%.

If adopted, the default takes the candidate's rates, lag and slope, and the
rest of `carla_nissan_patrol()` is unchanged.

To test its code, `analyse.py` was dry-run before this commit with a dummy
candidate chosen far from the expected fit (lag 0.05 s, slope 0.006 s per
m/s). Seeds 18-21 stood in for the fresh seeds and seed 0 for the
muscle-driven recordings. The rule above was written before that run and was
not changed. `fit.py` has not been run.

## Secondary analyses (descriptive, not part of the rule)

1. On seeds 22-25, by kind: the engine-rate model (default lag) and the
   constant-lag candidate of the previous test (`2d19c54`).
2. Seeds 11 and 13 (held out from every fit, inspected before) by kind.
3. The full-body `steer_left` and `steer_right` phases.

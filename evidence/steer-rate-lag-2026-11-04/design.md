# Joint steering rise rate and lag: design fixed before fitting

Written on 2026-11-04 before the fit was run and before any of the test drives
were recorded.

## Question

The rate limit test on seeds 14-17 used the engine's rates (rise 2.5, fall 5.0
per second) with the steering lag of the current default (0.039 s), which was
fitted without the limit. It cut the steering error of steps and doublets by
about 80% but made the one-period low-speed sweeps 8% worse, so the default
was kept. Post hoc, a rise rate of 2.75 improved every kind, and the lag looked
too long at 1-3 m/s but needed at speed. This run fits the rise rate and the
lag together and tests the result on fresh seeds.

## Candidate

`carla_nissan_patrol()` with `steer_rise_rate` and `steer_lag` fitted jointly
and `steer_fall_rate` fixed at 5.0 (the engine value, also the best fall rate
on seeds 14-17). The gain table, exponent and longitudinal parameters stay at
the default. A constant lag cannot represent a delay that changes with speed;
if that is the cause, the candidate should fail check (b) or (c) below.

## Fit (`fit.py`)

- Training: the steering segments (`steer*` kinds) of seeds 0, 3, 10, 12 and
  14-17, about 152 segments and 63,000 frames. Seeds 1, 2, 4, 11 and 13 are not
  used.
- Objective: as in the lateral fit, the pooled squared yaw-rate residual with
  the measured speed over frames above 0.5 m/s.
- Search: a grid of rise rates 1.0-5.0 in steps of 0.05 and lags 0-0.1 s in
  steps of 0.002 s; the grid minimum is the candidate.
- Uncertainty: 1000 bootstrap resamples of training segments (seed
  20261104), each taking its own grid minimum; 95% percentile intervals.
- Check of the fast path: at rise 2.5 and the default lag, its per-segment
  errors must match `evaluate_vehicle_model(..., measured_speed=True)` to
  1e-9 deg/s.
- Secondary (descriptive): the same fit on training segments whose mean
  steering speed is below 3 m/s and on those at or above it. Prediction: the
  lag fitted at speed is longer than the lag fitted below 3 m/s.

The fitted candidate is committed before the test drives are recorded.

## Test data

- Server: the CARLA 0.10.0 / UE5.5 package in
  `C:/Users/localadmin/Desktop/Carla-0.10.0-Win64-Shipping` (VERSION: CARLA
  `ada75f920642e18cace0a9f85ecf9d4077ddb531`), started with `-nullrhi -nosound
  -unattended -notrace -carla-rpc-port=2000`, default map, client wheel 0.10.0.
- Vehicle: `vehicle.nissan.patrol`.
- Command:
  `examples/identify_vehicle.py --seeds 18 19 20 21 --launch 0 --hand-brake 0
  --reverse 0 --steer 4 --steer-fast 4 --steer-low 8 --steer-hold 8
  --steer-doublet 10 --data artifacts/steer-rate-lag`.
- The 2-10 m/s (`steer`) and 8-14 m/s (`steer_fast`) sweeps are fresh tests of
  the regime where the lag was identified. They also serve as the session
  control.
- Segments with a collision or a roll above 3 degrees are discarded, as
  before.

## Adoption rule (`analyse.py`)

All errors are duration-weighted means of per-segment RMSE from
`evaluate_vehicle_model`; measured-speed errors use `measured_speed=True`.
The candidate replaces `carla_nissan_patrol()` only if all of the following
hold, each relative to the current default:

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

To test its code, `analyse.py` was dry-run before this commit with a dummy
candidate (rise 3.0, lag 0.03 s). Seeds 14, 15 and 16 and seed 4 stood in for
the fresh seeds, and seed 0 for the muscle-driven recordings. On those
stand-ins the dummy failed check (c). The rule above was written before that
run and was not changed. `fit.py` has not been run.

## Secondary analyses (descriptive, not part of the rule)

1. The engine-rate model of the previous test (rise 2.5, fall 5.0, default
   lag) on the fresh seeds, by kind: does its sweep result replicate?
2. Seeds 11 and 13 (held out from every fit, but inspected before) by kind.
3. The full-body `steer_left` and `steer_right` phases.

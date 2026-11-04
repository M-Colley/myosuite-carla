# Steering input rate limit: confirmatory design fixed before recording

Written on 2026-11-04 before any of these drives were recorded.

## Question

The post-hoc diagnosis of the wider steering identification found that Chaos's
steering input rate limit explained most of the surrogate's remaining
low-speed steering error. The engine source (`ChaosVehicleMovementComponent`,
UE5.8) applies `InterpInputValue` to the raw steering input before the curve.
The magnitude may grow at most 2.5 and shrink at most 5.0 per second. On seeds
10-13, adding it with no fitted parameter cut the measured-speed yaw-rate RMSE
by 57-58%. That finding came from data already inspected. This run tests it on
fresh seeds and in a regime it was not found in.

The surrogate now has `steer_rise_rate` and `steer_fall_rate`, applied to the
command before the steering lag. Zero (the default, and what parameters saved
earlier load as) leaves the command unlimited. The candidate is
`carla_nissan_patrol()` with rates 2.5 and 5.0 and every other parameter
unchanged. No parameter is fitted in this run.

## Data

- Server: the CARLA 0.10.0 / UE5.5 package in
  `C:/Users/localadmin/Desktop/Carla-0.10.0-Win64-Shipping` (VERSION: CARLA
  `ada75f920642e18cace0a9f85ecf9d4077ddb531`), started with `-nullrhi -nosound
  -unattended -notrace -carla-rpc-port=2000`, default map, client wheel 0.10.0.
- Vehicle: `vehicle.nissan.patrol`.
- Command:
  `examples/identify_vehicle.py --seeds 14 15 16 17 --launch 0 --hand-brake 0
  --reverse 0 --steer 3 --steer-low 8 --steer-hold 8 --steer-doublet 10 --data
  artifacts/steer-rate-limit`.
- `steer_low` and `steer_hold`: as in seeds 10-13 (0.8-3 m/s, up to full
  lock). They replicate the finding on fresh seeds.
- `steer_doublet` (new): at 3-10 m/s, a step to ±S held, the opposite step
  held equally long, then 1.5 s straight. S is chosen for a steady lateral
  acceleration of 0.5-3 m/s² (|S| 0.05-0.8); each half turns about 0.25 rad.
  This regime lies in the identified speed range. It exercises both rates and
  the sign change.
- `steer`: the original 2-10 m/s sweep, as a session control.
- All four seeds are held out. Segments with a collision or a roll above 3
  degrees are discarded, as before.

Before this run, commands only (no outcomes) were checked. The limit never
changes a command of training seeds 0 and 3 or held-out seeds 1, 2 and 4. It
changes 18 of 3470 frames of the muscle-driven recordings, in the full-body
`steer_left` and `steer_right` phases.

To test its code, `analyse.py` was dry-run with seeds 10-13 standing in for
the fresh seeds and training seed 0 standing in for the muscle-driven
recordings. That run, seen before this design was committed, put the best
rise rate at 2.75 and the best fall rate at 5.5 on seeds 10-13.

## Metrics

All errors are duration-weighted means of per-segment RMSE from
`evaluate_vehicle_model`. *Measured-speed* errors use `measured_speed=True`:
the logged speed replaces the modelled one, as in the lateral fit, so they
isolate the steering model. On seeds 10-13 (already inspected) the
measured-speed yaw-rate RMSE fell from 0.955 to 0.404 deg/s (training) and
0.997 to 0.431 (held out) with this metric. *Open-loop* errors are the usual
rollouts.

## Adoption rule

The candidate replaces `carla_nissan_patrol()` only if all of the following
hold, each relative to the current default:

- Control: the default's open-loop yaw-rate RMSE on the fresh `steer` sweeps is
  at most 2 x 0.337 deg/s. Otherwise the session is not treated as comparable
  and the default is kept.
- (a) Measured-speed yaw-rate RMSE on the pooled fresh `steer_low`,
  `steer_hold` and `steer_doublet` segments falls by at least 25%.
- (b) It falls in each of these three kinds separately.
- (c) Open-loop yaw-rate RMSE on the pooled fresh segments of (a) does not rise.
- (d) On the original held-out seeds 1, 2 and 4, the open-loop yaw-rate RMSE
  and the 3 s position error of each manoeuvre group rise by at most 1%. Their
  commands never reach the limit, so this is a check of the implementation.
- (e) On the four muscle-driven recordings used before, open-loop yaw-rate
  RMSE and 3 s position error rise by at most 5%.

The 25% threshold is under half the effect on seeds 10-13, because the
doublets at speed are expected to show a smaller effect.

## Secondary analyses (descriptive, not part of the rule)

1. **Rate profiles.** The measured-speed yaw-rate RMSE of the pooled fresh
   segments over the rise rate (0.5-6 per second in steps of 0.25, fall fixed
   at 5) and over the fall rate (1-12 in steps of 0.5 and unlimited, rise fixed
   at 2.5). Prediction: the best rise rate lies within 2.0-3.0 and the best
   fall rate within 4.0-6.0 or the profile is flat there.
2. **Full-body steering phases.** The `steer_left` and `steer_right` phases of
   the full-body run were inspected before. Prediction: the limit reduces the
   yaw-rate RMSE of `steer_right` (2.53 deg/s with the default), whose onset is
   the steepest command change in those data.
3. **Refit check.** Training seeds 0 and 3 never reach the limit. A refit with
   `--steer-rate-limit 2.5 5` must therefore reproduce the default's
   parameters. The candidate then equals a refit of the default with the limit.
4. Per-kind and per-band results, and the default's remaining error near full
   lock.

## Scope

The run uses CARLA 0.10.0 only. The UE5.5 engine defaults were not read; the
test is the empirical check for 0.10.0. CARLA 1.0's vehicle blueprints were
not inspected, so whether 1.0 uses the same rates stays open. The engine also
rate-limits throttle (rise 6, fall 10 per second), brake (6, 10) and hand
brake (12, 12). Those limits are out of scope here.

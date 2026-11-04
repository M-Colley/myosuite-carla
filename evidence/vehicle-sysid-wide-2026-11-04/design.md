# Wider steering identification: design fixed before recording

Written on 2026-11-04 before any of these drives were recorded.

## Question

The current surrogate (`carla_nissan_patrol()`, steering exponent 1.974) was
identified from steering up to |S| = 0.51 at 2.4-13.6 m/s. Speed knots 0, 1
and 2.5 m/s hold the gain identified at 4 m/s. The full-body steering phases
run at 1.1-2.2 m/s. This run asks whether the power-law steering curve holds
to full lock and at low speed, and whether adding such drives improves the fit
without degrading the original held-out errors.

## Data

- Server: the CARLA 0.10.0 / UE5.5 package in
  `C:/Users/localadmin/Desktop/Carla-0.10.0-Win64-Shipping` (VERSION: CARLA
  `ada75f920642e18cace0a9f85ecf9d4077ddb531`), started with `-nullrhi -nosound
  -unattended -notrace -carla-rpc-port=2000`, default map, client wheel 0.10.0.
- Vehicle: `vehicle.nissan.patrol`, as for seeds 0-4.
- Command:
  `examples/identify_vehicle.py --seeds 10 11 12 13 --launch 0 --hand-brake 0
  --reverse 0 --steer 3 --steer-low 10 --steer-hold 8 --data
  artifacts/vehicle-sysid-wide`.
- `steer_low`: one sinusoidal period at 0.8-3 m/s, amplitude 0.4-1.0,
  frequency 0.3-0.8 Hz, raised where needed to keep the heading excursion near
  0.6 rad. `steer_hold`: a step to ±0.2-1.0 held 0.6-2.5 s, then 1.5 s
  straight. `steer`: the original 2-10 m/s sweep, as a control for session
  differences.
- Split: seeds 10 and 12 join training; seeds 11 and 13 are held out.
- Segments with a collision or a roll above 3 degrees are discarded, as before.

## Analyses

Errors are duration-weighted means of per-segment open-loop RMSE from
`evaluate_vehicle_model`, as in the earlier entries.

1. **Control.** The current default on the held-out `steer` segments of seeds
   11 and 13. If its yaw-rate RMSE exceeds twice the original held-out steering
   value (2 x 0.337 deg/s), the new session is not treated as comparable: the
   data are reported but not pooled, and the default is not changed.
2. **Extrapolation test.** The current default and the retained linear fit on
   the held-out `steer_low` and `steer_hold` segments of seeds 11 and 13, by
   kind and by band (maximum |S| at most 0.51 or above; mean steering speed
   below 2.4 m/s or not). Descriptive; no parameter is changed by it.
3. **Steady-state curve.** Settled road-wheel angles `atan(L r / u)` from the
   second half of every hold (all four seeds), fitted as `g sgn(S) |S|^p`.
   CARLA 0.10.0's source gives p = 2 up to full lock. Descriptive.
4. **Refit.** Seeds 0, 3, 10 and 12 with the exponent identified, evaluated on
   seeds 1, 2, 4, 11 and 13 and on the four muscle-driven recordings used
   before.

## Adoption rule

The refit replaces `carla_nissan_patrol()` only if the control passes and all
of the following hold, each relative to the current default:

- (a) yaw-rate RMSE on the held-out `steer_low` and `steer_hold` segments of
  seeds 11 and 13 decreases;
- (b) overall yaw-rate RMSE on the original held-out seeds 1, 2 and 4 rises by
  at most 5%;
- (c) no 3 s position error rises by more than 5%, for each manoeuvre group of
  seeds 1, 2 and 4 and for the muscle-driven recordings overall.

The 5% tolerances allow a small trade-off in the original regime for a new
one. Otherwise the default is kept and the results are reported. A structural
change to the model (for example tyre slip or saturation) is out of scope for
this run.

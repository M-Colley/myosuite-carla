"""Shape of CARLA 0.10.0's steering response, from the retained sysid drives.

CARLA 1.0 forces Chaos's steering input curve to ``LinearFunction`` (upstream
commit 674a3a70e); UE5's default is ``SquaredFunction``. This estimates which
curve the 0.10.0 Nissan Patrol drives used. The effective road-wheel angle is
``delta = atan(L * yaw_rate / v)``; within a speed band, ``delta = g *
sign(s) * |s|**p`` is fitted to the steer command ``s`` over a grid of
exponents ``p``. Each steering phase is a sinusoid of one amplitude, so the
per-phase linear gain ``delta / s`` against the amplitude separates the
curves too: constant for a linear curve, proportional for a squared one.

Usage: python -I steer_input_curve.py evidence/ue55-vehicle-sysid.zip OUT.json
"""

import csv
import io
import json
import math
import sys
import zipfile

import numpy as np

WHEELBASE = 3.075  # vehicle_model.carla_nissan_patrol()
MIN_SPEED = 2.0
MIN_STEER = 0.03
EXPONENTS = np.round(np.arange(0.5, 2.51, 0.05), 2)
BANDS = ((2, 4), (4, 6), (6, 8), (8, 10), (10, 14))


def steer_phases(path):
    phases = []
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            if not name.endswith("frames.csv"):
                continue
            current, rows = None, []
            for row in csv.DictReader(io.TextIOWrapper(archive.open(name))):
                if row["phase"] != current:
                    if current and current.endswith("steer"):
                        phases.append((current, np.asarray(rows)))
                    current, rows = row["phase"], []
                rows.append((
                    float(row["command_steer"]),
                    float(row["speed_mps"]),
                    math.radians(float(row["wz_dps"])),
                ))
            if current and current.endswith("steer"):
                phases.append((current, np.asarray(rows)))
    return phases


def samples(phases, shift):
    """(steer, speed, delta, phase amplitude, phase index), steer leading by shift."""
    out = []
    for index, (_, a) in enumerate(phases):
        steer = a[: len(a) - shift, 0] if shift else a[:, 0]
        speed, yaw_rate = a[shift:, 1], a[shift:, 2]
        delta = np.arctan(WHEELBASE * yaw_rate / np.maximum(speed, 1e-6))
        keep = (speed > MIN_SPEED) & (np.abs(steer) > MIN_STEER)
        amplitude = np.max(np.abs(a[:, 0]))
        for row in zip(steer[keep], speed[keep], delta[keep]):
            out.append((*row, amplitude, index))
    return np.asarray(out)


def exponent_fit(steer, delta):
    """Least-squares gain for each exponent; (sse, gain) by exponent."""
    fits = {}
    for p in EXPONENTS:
        x = np.sign(steer) * np.abs(steer) ** p
        gain = float(x @ delta / (x @ x))
        residual = delta - gain * x
        fits[float(p)] = (float(residual @ residual), gain)
    return fits


def main(archive, output):
    phases = steer_phases(archive)
    # Yaw lags the command; pick the frame shift with the strongest correlation.
    shift = max(
        range(16),
        key=lambda k: abs(np.corrcoef(*samples(phases, k)[:, [0, 2]].T)[0, 1]),
    )
    data = samples(phases, shift)
    bands = []
    for low, high in BANDS:
        band = data[(data[:, 1] >= low) & (data[:, 1] < high)]
        if len(band) < 50:
            continue
        fits = exponent_fit(band[:, 0], band[:, 2])
        best = min(fits, key=lambda p: fits[p][0])
        gains = []
        for amplitude in np.unique(np.round(band[:, 3], 2)):
            chosen = np.round(band[:, 3], 2) == amplitude
            if chosen.sum() >= 20:
                s, d = band[chosen, 0], band[chosen, 2]
                gains.append([float(amplitude), float(s @ d / (s @ s))])
        bands.append({
            "speed_mps": [low, high],
            "samples": int(len(band)),
            "best_exponent": best,
            "squared_gain_rad": fits[2.0][1],
            "sse_ratio_linear_to_best": fits[1.0][0] / fits[best][0],
            "sse_ratio_squared_to_best": fits[2.0][0] / fits[best][0],
            "linear_gain_by_amplitude_rad": gains,
        })
    # Within single sweeps (one amplitude each), so the speed table and the
    # amplitude cannot stand in for the shape.
    within = []
    for index, (name, _) in enumerate(phases):
        rows = data[data[:, 4] == index]
        if len(rows) >= 40 and np.max(np.abs(rows[:, 0])) >= 0.2:
            fits = exponent_fit(rows[:, 0], rows[:, 2])
            within.append([name, min(fits, key=lambda p: fits[p][0])])
    exponents = [p for _, p in within]
    report = {
        "archive": "evidence/ue55-vehicle-sysid.zip",
        "carla": "0.10.0 (UE5.5), vehicle.nissan.patrol, Town10HD_Opt",
        "wheelbase_m": WHEELBASE,
        "steer_phases": len(phases),
        "yaw_shift_frames": shift,
        "bands": bands,
        "within_sweep_exponent": {
            "sweeps": len(within),
            "median": float(np.median(exponents)) if exponents else None,
            "quartiles": (
                [float(q) for q in np.percentile(exponents, [25, 75])]
                if exponents else None
            ),
            "by_phase": within,
        },
    }
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for band in bands:
        print(
            f"{band['speed_mps']} m/s n={band['samples']} p={band['best_exponent']:.2f}"
            f" SSE linear/best={band['sse_ratio_linear_to_best']:.1f}"
        )
    print("within-sweep exponent", report["within_sweep_exponent"]["median"],
          report["within_sweep_exponent"]["quartiles"], len(within))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

"""Plot frozen validation and test scores with equal family and seed weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

LEGACY = "CARLA 0.10 Patrol fit"
BLUE, ORANGE = "#2878B5", "#D86522"


def values(result: dict[str, Any], metric: str) -> np.ndarray:
    """Read the source scores without weighting by segment counts."""
    return np.asarray(
        [result["scores"][seed][metric] for seed in sorted(result["scores"], key=int)],
        dtype=float,
    )


def save(figure: Any, destination: Path, name: str) -> None:
    for extension in ("png", "svg"):
        figure.savefig(destination / f"{name}.{extension}", dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    matplotlib.use("Agg")
    fit = json.loads((args.analysis / "fit-report.json").read_text(encoding="utf-8"))
    test = json.loads((args.analysis / "test-report.json").read_text(encoding="utf-8"))
    selected = fit["selected"]
    if selected is None or test["selected"] != selected:
        raise ValueError(
            "Validation and test must identify the same selected candidate"
        )
    seeds = sorted(test["legacy"]["scores"], key=int)
    if len(seeds) != 3 or set(seeds) != set(test["candidate"]["scores"]):
        raise ValueError("The paired test pilot requires three complete seed batches")
    candidate_label = f"CARLA 1.0 candidate {selected}"
    args.out.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
    names = list(fit["candidates"])
    validation = [values(fit["legacy"], "primary").mean()] + [
        values(fit["candidates"][name]["validation"], "primary").mean()
        for name in names
    ]
    colours = [BLUE] + [ORANGE if name == selected else "#A9A9A9" for name in names]
    bars = axes[0].bar(range(len(validation)), validation, color=colours)
    axes[0].bar_label(bars, fmt="%.3g", padding=3, fontsize=8)
    axes[0].set_xticks(range(len(validation)), ["Legacy\n0.10"] + names)
    axes[0].set_ylabel("Primary error (normalized squared)")
    axes[0].set_title(f"Validation selection: {selected}")
    axes[0].set_ylim(bottom=0)
    left, right = (
        values(test["legacy"], "primary"),
        values(test["candidate"], "primary"),
    )
    x = np.arange(len(seeds))
    for index, (old, new) in enumerate(zip(left, right)):
        axes[1].plot([index - 0.1, index + 0.1], [old, new], color="#777777", lw=1)
    axes[1].scatter(x - 0.1, left, color=BLUE, label=LEGACY, s=45, zorder=3)
    axes[1].scatter(x + 0.1, right, color=ORANGE, label=candidate_label, s=45, zorder=3)
    axes[1].set_xticks(x, [f"Seed {seed}" for seed in seeds])
    axes[1].set_ylabel("Primary error (normalized squared)")
    axes[1].set_title("Test pilot: three seed batches")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    interval = test["paired_seed_primary_change"]
    low, high = interval["interval"]
    figure.suptitle("CARLA 1.0 vehicle calibration", fontsize=13)
    figure.supxlabel(
        "Primary scales: 1 m/s and 5 deg/s. Equal family and seed weights.\n"
        f"Paired primary change {interval['value']:.4g}; "
        f"95% seed bootstrap interval [{low:.4g}, {high:.4g}].",
        fontsize=9,
    )
    save(figure, args.out, "validation-and-paired-test")

    figure = plt.figure(figsize=(13, 8), layout="constrained")
    grid = figure.add_gridspec(2, 3)
    family_axis = figure.add_subplot(grid[0, :])
    families = list(test["legacy"]["scores"][seeds[0]]["families"])
    x = np.arange(len(families))
    for offset, role, colour, label in (
        (-0.18, "legacy", BLUE, LEGACY),
        (0.18, "candidate", ORANGE, candidate_label),
    ):
        means = [
            np.mean(
                [
                    test[role]["scores"][seed]["families"][family]["primary"]
                    for seed in seeds
                ]
            )
            for family in families
        ]
        family_axis.bar(x + offset, means, width=0.36, color=colour, label=label)
    family_axis.set_xticks(x, [name.replace("_", " ") for name in families])
    family_axis.set_ylabel("Primary error (normalized squared)")
    family_axis.set_title("Test primary error by manoeuvre family")
    family_axis.set_ylim(bottom=0)
    family_axis.legend(fontsize=9)
    family_axis.grid(axis="y", alpha=0.2)
    family_axis.set_axisbelow(True)
    for index, (metric, title, unit) in enumerate(
        (
            ("speed_rmse_mps", "Speed RMSE", "m/s"),
            ("yaw_rate_rmse_deg_s", "Yaw rate RMSE", "deg/s"),
            ("final_position_error_m", "Final position error", "m"),
        )
    ):
        axis = figure.add_subplot(grid[1, index])
        old, new = values(test["legacy"], metric), values(test["candidate"], metric)
        for pair in zip(old, new):
            axis.plot([0, 1], pair, color="#999999", alpha=0.7, lw=1)
        axis.scatter(np.zeros(3), old, color=BLUE, s=25, zorder=3)
        axis.scatter(np.ones(3), new, color=ORANGE, s=25, zorder=3)
        axis.plot([0, 1], [old.mean(), new.mean()], "kD-", ms=6, lw=2)
        axis.set_xticks([0, 1], ["Legacy 0.10", f"1.0 {selected}"])
        axis.set_ylabel(f"Mean error ({unit})")
        axis.set_title(title)
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("CARLA 1.0 test pilot: three seed batches", fontsize=13)
    figure.supxlabel(
        "Equal family and seed weights. Grey lines pair seed batches. "
        "Black diamonds show means across the three batches.\n"
        f"Decision: {test['decision'].replace('_', ' ')}.",
        fontsize=9,
    )
    save(figure, args.out, "test-family-and-metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

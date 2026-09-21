#!/usr/bin/env python3
"""Water formation against time for every run in the stoichiometry sweep.

    uv run python production/stoichiometry-3000K/plot_outputs.py
    uv run python production/stoichiometry-3000K/plot_outputs.py output -o water.png

Reads `species["H2O"]` straight out of each run's `log.jsonl`, which `run_one.py`
writes from the calculator's live topology -- the EVB's own answer about what is
bonded, not a geometric re-perception of it (the distinction `scripts/plot.py`
documents, and the reason this script never opens `traj.xyz`).

What the number *is* matters for reading the figure: it is the **instantaneous
population** of H2O at that frame, not a cumulative count of formation events.
A water that forms and is re-abstracted a picosecond later goes up and then back
down.  Distinguishing the two would need molecule identity tracked across frames,
which the log does not carry.

The runs are **still in progress and of unequal length**, so the two panels split
the difference rather than pretending otherwise: each seed is drawn to its own
last frame, and the seed-mean is drawn only out to the shortest seed of that
composition, where all five are still contributing.  Past that point a mean would
be measuring which jobs happened to survive.
"""

import json
import statistics
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# Categorical slots 1-5 of the validated default palette, in ladder order:
# oxygen-rich at one end, hydrogen-rich at the other.  Adjacent-pair CVD
# separation 9.1 (>= 8 target); the three mid slots sit under 3:1 contrast on
# white, which is why every series is also direct-labelled.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

# Neutral ink.  Text never wears the series color; the marker beside it carries
# the identity.
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def read_log(path: Path) -> tuple[dict, list[dict]]:
    """One run's header and frames.  A truncated last line is skipped.

    A job still writing -- or killed mid-write -- leaves a partial record, and
    every run in this sweep is currently in that state.
    """
    header, frames = {}, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record") == "header":
                header = record
            elif record.get("record") == "frame":
                frames.append(record)
    return header, frames


def ratio_key(ratio: str) -> float:
    """Sort key placing the ladder oxygen-rich -> hydrogen-rich."""
    h2, o2 = (float(part) for part in ratio.split(":"))
    return h2 / o2


def equivalence_ratio(species: dict[str, int]) -> float:
    """phi for H2/O2, from the frame-0 populations rather than the label.

    2 H2 + O2 -> 2 H2O is stoichiometric at H2:O2 = 2, so

        phi = (n_H2 / n_O2) / 2

    and the ladder reads 0.25 (lean) through 1 (stoichiometric) to 4 (rich).
    Taken from the counts the run actually started with, because `make_inputs.py`
    packs whole molecules -- a nominal ratio need not divide 100 exactly, and
    where it does not, phi follows what is in the box and not what is on the
    directory name.
    """
    o2 = species.get("O2", 0)
    if not o2:
        return float("inf")
    return species.get("H2", 0) / o2 / 2.0


def format_phi(phi: float) -> str:
    """phi to two decimals, trailing zeros trimmed: 0.25, 0.5, 1, 2, 4."""
    return f"{phi:.2f}".rstrip("0").rstrip(".")


def water_ceiling(species: dict[str, int]) -> int:
    """Most H2O the initial composition can make, via 2 H2 + O2 -> 2 H2O.

    Printed rather than drawn: at 22-50 it is an order of magnitude above
    anything reached so far, and a ceiling line would flatten the curves into
    the axis.
    """
    return min(species.get("H2", 0), 2 * species.get("O2", 0))


def step_sample(times: np.ndarray, counts: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Counts on `grid`, held at the last frame's value.

    A population is piecewise constant between frames, so interpolating it
    linearly would invent fractional molecules between two integers.
    """
    index = np.searchsorted(times, grid, side="right") - 1
    return counts[np.clip(index, 0, len(counts) - 1)]


def load_runs(directory: Path) -> dict[str, list[dict]]:
    """Every run under `directory`, grouped by composition and sorted by seed."""
    groups: dict[str, list[dict]] = {}
    for log in sorted(directory.glob("*/log.jsonl")):
        header, frames = read_log(log)
        if not frames:
            print(f"  (no frames yet: {log.parent.name})")
            continue

        config_path = log.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        ratio = config.get("ratio")
        if ratio is None:
            # r8_1_s3 -> 8:1.  Only reached if config.json went missing.
            head, _, _ = log.parent.name.rpartition("_")
            ratio = head.lstrip("r").replace("_", ":")

        times = np.array([f["time_fs"] for f in frames]) / 1000.0
        water = np.array([f["species"].get("H2O", 0) for f in frames])
        groups.setdefault(ratio, []).append(
            {
                "run": log.parent.name,
                "seed": config.get("seed"),
                "times": times,
                "water": water,
                "initial": frames[0]["species"],
                "ceiling": water_ceiling(frames[0]["species"]),
                "phi": equivalence_ratio(frames[0]["species"]),
                "temperature": statistics.mean(f["temperature_K"] for f in frames),
                "target_K": config.get("temperature", header.get("temperature_K")),
                "steps_done": frames[-1]["step"],
                "steps_planned": config.get("steps", header.get("steps")),
                "channels": sorted(
                    {c for f in frames for c in f.get("placeholder_channels", [])}
                ),
                "capped": sum(f.get("ncapped", 0) > 0 for f in frames),
            }
        )

    for runs in groups.values():
        runs.sort(key=lambda r: (r["seed"] is None, r["seed"], r["run"]))
    return dict(sorted(groups.items(), key=lambda item: ratio_key(item[0])))


def seed_mean(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Mean H2O over seeds, on the grid where every seed is still running."""
    horizon = min(r["times"][-1] for r in runs)
    grid = np.unique(np.concatenate([r["times"][r["times"] <= horizon] for r in runs]))
    stack = np.vstack([step_sample(r["times"], r["water"], grid) for r in runs])
    return grid, stack.mean(axis=0)


def plot(groups: dict[str, list[dict]], destination: Path) -> None:
    fig = plt.figure(figsize=(11, 7.2), dpi=200)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.35, 1.0], hspace=0.42)
    top = fig.add_subplot(outer[0])
    lower = outer[1].subgridspec(1, len(groups), wspace=0.12)

    colors = {ratio: PALETTE[i % len(PALETTE)] for i, ratio in enumerate(groups)}

    # --- seed means, one line per composition ------------------------------
    ends = []
    for ratio, runs in groups.items():
        grid, mean = seed_mean(runs)
        top.step(grid, mean, where="post", lw=2.0, color=colors[ratio], zorder=3)
        top.plot([grid[-1]], [mean[-1]], "o", ms=4, color=colors[ratio], zorder=4)
        ends.append([ratio, float(grid[-1]), float(mean[-1])])

    # Direct label at the live end of each line -- the three mid palette slots
    # are under 3:1 against white, so identity is never color alone.  Two
    # compositions can end at the same count, so the labels are pushed apart
    # vertically before they are drawn rather than overplotted on each other.
    span = max(e[2] for e in ends) or 1.0
    gap = 0.09 * span
    ends.sort(key=lambda e: e[2])
    for lower_end, upper_end in zip(ends, ends[1:]):
        shortfall = gap - (upper_end[2] - lower_end[2])
        if shortfall > 0:
            # Split the push so neither label ends up sitting on the other's
            # end marker, which one-sided spreading leaves behind.
            lower_end[2] -= shortfall / 2
            upper_end[2] += shortfall / 2
    for ratio, x, y in ends:
        top.annotate(
            ratio,
            xy=(x, y),
            xytext=(7, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=INK,
        )

    # phi alongside the number ratio: the composition is the swept variable, so
    # the legend carries both the mixture as packed and the combustion quantity
    # it corresponds to.  A seed disagreeing would mean the runs in one column
    # were packed differently, so it is asserted rather than averaged away.
    handles = []
    for ratio, runs in groups.items():
        phis = {round(r["phi"], 6) for r in runs}
        assert len(phis) == 1, f"{ratio}: seeds disagree on phi ({sorted(phis)})"
        handles.append(
            plt.Line2D(
                [],
                [],
                color=colors[ratio],
                lw=2.0,
                label=f"{ratio}  H2:O2   \u03c6 = {format_phi(runs[0]['phi'])}",
            )
        )
    top.legend(
        handles=handles,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=9,
        labelcolor=INK,
        ncol=len(groups),
        columnspacing=1.6,
        handlelength=1.6,
        borderaxespad=0.6,
    )
    top.set_title(
        "H2O population during H2/O2 combustion at "
        f"{groups[next(iter(groups))][0]['target_K']:.0f} K",
        fontsize=12,
        color=INK,
        loc="left",
        pad=28,
    )
    top.set_ylabel("H2O molecules (mean of runs)", fontsize=10, color=INK_MUTED)
    top.set_xlabel("time (ps)", fontsize=10, color=INK_MUTED)

    # --- one panel per composition, every seed drawn ----------------------
    ymax = max(r["water"].max() for runs in groups.values() for r in runs)
    # One x scale across the row: the panels are here to be compared, and a
    # per-panel scale would make a 30 ps run and a 10 ps one look alike.
    xmax = max(r["times"][-1] for runs in groups.values() for r in runs)
    shared = None
    for column, (ratio, runs) in enumerate(groups.items()):
        ax = fig.add_subplot(lower[column], sharey=shared, sharex=shared)
        shared = shared or ax
        for run in runs:
            ax.step(
                run["times"],
                run["water"],
                where="post",
                lw=1.0,
                color=colors[ratio],
                alpha=0.45,
                zorder=2,
            )
        grid, mean = seed_mean(runs)
        ax.step(grid, mean, where="post", lw=2.0, color=colors[ratio], zorder=3)
        ax.set_title(f"{ratio}  H2:O2", fontsize=10, color=INK, loc="left", pad=6)
        ax.text(
            0.04,
            0.96,
            f"{len(runs)} runs\nup to {runs[0]['ceiling']} H2O",
            transform=ax.transAxes,
            fontsize=8,
            color=INK_MUTED,
            va="top",
            linespacing=1.4,
        )
        ax.set_xlabel("time (ps)", fontsize=9, color=INK_MUTED)
        if column == 0:
            ax.set_ylabel("H2O molecules", fontsize=9, color=INK_MUTED)
        else:
            ax.tick_params(labelleft=False)
        ax.set_ylim(-0.25, max(ymax, 1) + 0.5)
        ax.set_xlim(0, xmax * 1.02)

    for ax in fig.get_axes():
        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlim(left=0)

    # fig.text(
    #     0.011,
    #     0.012,
    #     "Instantaneous population, not cumulative formation events.  Runs are "
    #     "in progress: runs are drawn to their own last frame, the bold mean "
    #     "only to the shortest run of each composition.",
    #     fontsize=8,
    #     color=INK_MUTED,
    # )
    fig.savefig(destination, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path(__file__).parent / "output",
        help="sweep output directory holding <run>/log.jsonl (default: ./output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="figure path (default: <directory>/water_formation.png)",
    )
    args = parser.parse_args()

    groups = load_runs(args.directory)
    if not groups:
        print(f"no runs with frames under {args.directory}")
        return 1

    print(
        f"{'run':<10}{'ps':>7}{'% done':>8}{'H2O now':>9}{'peak':>6}{'ceiling':>9}"
        f"{'T (K)':>8}{'capped':>8}"
    )
    for ratio, runs in groups.items():
        for run in runs:
            done = (
                100.0 * run["steps_done"] / run["steps_planned"]
                if run["steps_planned"]
                else float("nan")
            )
            print(
                f"{run['run']:<10}{run['times'][-1]:>7.1f}{done:>8.1f}"
                f"{run['water'][-1]:>9}{run['water'].max():>6}"
                f"{run['ceiling']:>9}{run['temperature']:>8.0f}{run['capped']:>8}"
            )

    print("\nwater formed by composition (at the shortest seed of each):")
    for ratio, runs in groups.items():
        grid, mean = seed_mean(runs)
        peaks = [int(r["water"].max()) for r in runs]
        spread = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
        print(
            f"  {ratio:<4} H2:O2  phi {format_phi(runs[0]['phi']):<4}  "
            f"mean {mean[-1]:.2f} H2O at {grid[-1]:.1f} ps   "
            f"peak per seed {peaks}  (sd {spread:.2f})"
        )

    # --- the diagnostics that say whether to believe it -------------------
    print("\nchecks:")

    horizon = min(min(r["times"][-1] for r in runs) for runs in groups.values())
    print(
        f"  every composition has all runs out to {horizon:.1f} ps; beyond that "
        "the mean is not plotted"
    )

    targets = {r["target_K"] for runs in groups.values() for r in runs}
    measured = statistics.mean(
        r["temperature"] for runs in groups.values() for r in runs
    )
    print(
        f"  mean temperature {measured:.0f} K against a "
        f"{'/'.join(f'{t:.0f}' for t in sorted(targets))} K target"
        + (
            ""
            # sweep.toml pins 3000 K and names the directory after it; config.json
            # is what actually ran, so a disagreement is the run, not the plot.
            if targets == {3000.0}
            else "  <- config.json disagrees with the sweep's 3000 K name"
        )
    )

    capped = sum(r["capped"] for runs in groups.values() for r in runs)
    print(
        f"  basis capped in {capped} frame(s)"
        + ("" if capped == 0 else "  <- states were dropped where this fired")
    )

    channels = sorted(
        {c for runs in groups.values() for r in runs for c in r["channels"]}
    )
    print(f"  unfitted couplings: {', '.join(channels) if channels else 'none'}")

    reached = [
        ratio for ratio, runs in groups.items() if any(r["water"].max() for r in runs)
    ]
    print(
        f"  compositions with any H2O so far: {', '.join(reached) if reached else 'none'}"
        "  <- counts are single molecules against ceilings of 22-50; this is the "
        "induction period, not a yield"
    )

    destination = args.output or args.directory / "water_formation.png"
    plot(groups, destination)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Every species' population against time for the stoichiometric 2:1 H2/O2 runs.

    uv run python production/stoichiometry-3000K/plot_species.py
    uv run python production/stoichiometry-3000K/plot_species.py output -o species.png

Reads `species` from each 2:1 run's `log.jsonl` through `plot_outputs.read_log`,
so restarted runs are stitched onto one clock the same way.  Each seed is drawn
thin to its own last frame; the bold line is the seed mean, drawn only out to
the shortest seed, where all of them are still contributing.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_outputs import read_log, step_sample

RATIO = "2:1"


def load_runs(directory: Path) -> list[dict]:
    """The 2:1 runs under `directory`, each as times (ps) and per-species counts."""
    runs = []
    for log in sorted(directory.glob("*/log.jsonl")):
        config_path = log.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        # r2_1_s3 -> 2:1.  The fallback is only reached if config.json went missing.
        fallback = log.parent.name.rpartition("_")[0].lstrip("r").replace("_", ":")
        if config.get("ratio", fallback) != RATIO:
            continue
        _, frames, _ = read_log(log)
        if not frames:
            continue
        times = np.array([f["time_fs"] for f in frames]) / 1000.0
        names = sorted({name for f in frames for name in f["species"]})
        counts = {
            name: np.array([f["species"].get(name, 0) for f in frames])
            for name in names
        }
        runs.append({"run": log.parent.name, "times": times, "counts": counts})
    return runs


def plot(runs: list[dict], destination: Path) -> None:
    species = sorted({name for r in runs for name in r["counts"]})
    horizon = min(r["times"][-1] for r in runs)
    grid = np.unique(np.concatenate([r["times"][r["times"] <= horizon] for r in runs]))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for i, name in enumerate(species):
        color = f"C{i}"
        stack = []
        for r in runs:
            counts = r["counts"].get(name, np.zeros_like(r["times"], dtype=int))
            ax.step(r["times"], counts, where="post", lw=0.6, alpha=0.25, color=color)
            stack.append(step_sample(r["times"], counts, grid))
        mean = np.mean(stack, axis=0)
        ax.step(grid, mean, where="post", lw=1.8, color=color, label=name)

    ax.set_xlabel("time (ps)")
    ax.set_ylabel("molecules")
    ax.set_title(f"Species populations, {RATIO} H2:O2 ({len(runs)} runs)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(destination)
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
        help="figure path (default: <directory>/species_2_1.png)",
    )
    args = parser.parse_args()

    runs = load_runs(args.directory)
    if not runs:
        print(f"no {RATIO} runs with frames under {args.directory}")
        return 1
    for r in runs:
        final = {name: int(c[-1]) for name, c in r["counts"].items() if c[-1]}
        print(f"{r['run']:<10}{r['times'][-1]:>7.1f} ps  {final}")

    destination = args.output or args.directory / "species_2_1.png"
    plot(runs, destination)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

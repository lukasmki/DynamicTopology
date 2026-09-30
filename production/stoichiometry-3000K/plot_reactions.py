#!/usr/bin/env python3
"""Molecule-to-molecule conversion counts for the stoichiometric 2:1 H2/O2 runs.

    uv run python production/stoichiometry-3000K/plot_reactions.py
    uv run python production/stoichiometry-3000K/plot_reactions.py output -o reactions.png

`log.jsonl` only carries species *populations*, which cannot say what turned into
what, so this reads the `connectivity` that `scripts/nvt.py` stamps on every
`traj.xyz` frame -- the calculator's live topology, not a geometric
re-perception.  A molecule is a connected component, identified by its set of
atom indices.

Between two consecutive frames, every molecule whose atom set does not survive
unchanged is a reactant, and every new atom set is a product.  Each
(reactant, product) pair that shares at least one atom is counted once as
reactant -> product.  So H2 + O2 -> H + HO2 counts H2 -> H, H2 -> HO2 and
O2 -> HO2.  An exchange that leaves the formula unchanged, such as
H + H2 -> H2 + H, lands on the diagonal.

Frames are written every `interval` steps, so a species that forms and reacts
away between two frames is never seen: its reactant goes straight to its
product in these counts.  A restarted run's repeated frame has identical
connectivity and counts nothing, and a frame cut off at the end of the file is
dropped.
"""

import json
import re
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.formula import Formula
from matplotlib.colors import LogNorm
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

RATIO = "2:1"
CMAP = "Blues"
CONNECTIVITY = re.compile(r'connectivity="_JSON (\[.*?\])"')


def frames(path: Path):
    """(symbols, connectivity string) for each complete frame of an extxyz file."""
    symbols = None
    with path.open() as handle:
        while True:
            count = handle.readline()
            if not count.strip():
                return
            natoms = int(count)
            comment = handle.readline()
            atoms = [handle.readline() for _ in range(natoms)]
            if not atoms[-1].endswith("\n"):
                return  # cut off mid-frame
            if symbols is None:
                symbols = [line.split(None, 1)[0] for line in atoms]
            yield symbols, CONNECTIVITY.search(comment).group(1)


def molecules(natoms: int, connectivity: str) -> list[frozenset[int]]:
    """Connected components of the bond list, as sets of atom indices."""
    bonds = np.array([b[:2] for b in json.loads(connectivity)], dtype=int).reshape(
        -1, 2
    )
    graph = coo_matrix(
        (np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])), shape=(natoms, natoms)
    )
    _, labels = connected_components(graph, directed=False)
    groups: dict[int, list[int]] = {}
    for atom, label in enumerate(labels):
        groups.setdefault(label, []).append(atom)
    return [frozenset(atoms) for atoms in groups.values()]


def count_conversions(path: Path) -> tuple[Counter, int]:
    """Counter of (reactant formula, product formula) over one trajectory."""
    conversions: Counter = Counter()
    formulas: dict[frozenset[int], str] = {}
    previous, before, nframes = None, None, 0

    def formula(molecule: frozenset[int]) -> str:
        if molecule not in formulas:
            # Same naming as `nvt.py`'s species counts (ASE's Hill formula).
            formulas[molecule] = Formula.from_list(
                [symbols[i] for i in sorted(molecule)]
            ).format("hill")
        return formulas[molecule]

    for symbols, connectivity in frames(path):
        nframes += 1
        if connectivity == previous:
            continue
        after = set(molecules(len(symbols), connectivity))
        if before is not None:
            products = after - before
            owner = {atom: p for p in products for atom in p}
            for reactant in before - after:
                for product in {owner[atom] for atom in reactant}:
                    conversions[formula(reactant), formula(product)] += 1
        previous, before = connectivity, after
    return conversions, nframes


def find_runs(directory: Path) -> list[Path]:
    """Trajectories of the 2:1 runs under `directory`."""
    runs = []
    for traj in sorted(directory.glob("*/traj.xyz")):
        config_path = traj.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        # r2_1_s3 -> 2:1.  The fallback is only reached if config.json went missing.
        fallback = traj.parent.name.rpartition("_")[0].lstrip("r").replace("_", ":")
        if config.get("ratio", fallback) == RATIO:
            runs.append(traj)
    return runs


def panel(ax, matrix, species, norm, label, title, fmt):
    """One species-by-species heatmap, each nonzero cell written out."""
    image = ax.imshow(
        np.ma.masked_equal(matrix, 0), origin="lower", cmap=CMAP, norm=norm
    )
    for y, x in zip(*np.nonzero(matrix)):
        value = matrix[y, x]
        ax.text(
            int(x),
            int(y),
            fmt(value),
            ha="center",
            va="center",
            fontsize=6,
            color="white" if norm(value) > 0.6 else "black",
        )
    ax.set_xticks(range(len(species)), species, rotation=45)
    ax.set_yticks(range(len(species)), species)
    ax.set_xlabel("reactant")
    ax.set_ylabel("product")
    ax.set_title(title, fontsize=10)
    plt.colorbar(image, ax=ax, label=label, shrink=0.8)


def fraction(value: float) -> str:
    return f"{value:.2f}" if value >= 0.005 else "<.01"


def plot(conversions: Counter, nruns: int, destination: Path) -> None:
    species = sorted({name for pair in conversions for name in pair})
    index = {name: i for i, name in enumerate(species)}
    matrix = np.zeros((len(species), len(species)))
    for (reactant, product), n in conversions.items():
        matrix[index[product], index[reactant]] = n

    # A row is one product: the fraction of its formations from each reactant.
    # A column is one reactant: the fraction of its conversions to each product.
    with np.errstate(invalid="ignore"):
        by_row = np.nan_to_num(matrix / matrix.sum(axis=1, keepdims=True))
        by_column = np.nan_to_num(matrix / matrix.sum(axis=0, keepdims=True))

    fig, axes = plt.subplots(1, 3, figsize=(19, 6), dpi=150)
    panel(
        axes[0],
        matrix,
        species,
        LogNorm(vmin=1, vmax=matrix.max()),
        "occurrences",
        "raw counts",
        lambda value: f"{value:.0f}",
    )
    panel(
        axes[1],
        by_row,
        species,
        plt.Normalize(0, 1),
        "fraction of row total",
        "normalized by row (product)",
        fraction,
    )
    panel(
        axes[2],
        by_column,
        species,
        plt.Normalize(0, 1),
        "fraction of column total",
        "normalized by column (reactant)",
        fraction,
    )
    fig.suptitle(f"Molecule conversions, {RATIO} H2:O2 ({nruns} runs)")
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
        help="sweep output directory holding <run>/traj.xyz (default: ./output)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="figure path (default: <directory>/reactions_2_1.png)",
    )
    args = parser.parse_args()

    runs = find_runs(args.directory)
    if not runs:
        print(f"no {RATIO} trajectories under {args.directory}")
        return 1

    total: Counter = Counter()
    with ProcessPoolExecutor() as pool:
        for traj, (conversions, nframes) in zip(
            runs, pool.map(count_conversions, runs)
        ):
            print(
                f"{traj.parent.name:<10}{nframes:>7} frames"
                f"{sum(conversions.values()):>8} conversions"
            )
            total += conversions

    print("\nmost frequent conversions:")
    for (reactant, product), n in total.most_common(15):
        print(f"  {reactant:>5} -> {product:<5}{n:>7}")

    destination = args.output or args.directory / "reactions_2_1.png"
    plot(total, len(runs), destination)
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

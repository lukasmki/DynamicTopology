#!/usr/bin/env python3
"""Plot energies, temperature and species counts from a trajectory.

The species panel reads each frame's stored `connectivity`, which
`scripts/nvt.py` now writes from the calculator's live topology.  It used to
call `ase2networkx(frame, True)` and re-perceive the bonding from geometry,
which was wrong twice over: molify's default cutoff is 1.2 * (r_cov + r_cov),
putting H-H at 0.7440 A against an 0.7445 A bond, so every H2 in the box was
counted as two free H atoms; and re-perceiving discards the EVB's own answer in
favour of a geometric guess about it.  `Topology.from_atoms` carries the
corrected `BOND_SCALE` and prefers stored connectivity, so it is the one path
worth using.
"""

from argparse import ArgumentParser
from ase import Atoms, io

import networkx as nx

import matplotlib.pyplot as plt

from DynamicTopology.core.topology import Topology


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-i", "--input", required=False, default="examples/nvt.xyz")
    parser.add_argument("-o", "--output", required=False)
    args = parser.parse_args()

    atoms: Atoms | list[Atoms] = io.read(args.input, index=":")
    if isinstance(atoms, Atoms):
        atoms: list[Atoms] = [atoms]

    TK = []
    PE = []
    KE = []
    SPECIES_COUNTS: list[dict[str, int]] = []

    for frame in atoms:
        frame: Atoms
        TK.append(frame.get_temperature())
        PE.append(frame.get_potential_energy())
        KE.append(frame.get_kinetic_energy())

        graph = Topology.from_atoms(frame).graph
        counts: dict[str, int] = {}
        for component in nx.connected_components(graph):
            formula = frame[sorted(component)].get_chemical_formula()
            counts[formula] = counts.get(formula, 0) + 1
        SPECIES_COUNTS.append(counts)

    steps = range(len(atoms))

    # plot the kinetic, potential and total energy
    TE = [ke + pe for ke, pe in zip(KE, PE)]
    fig, ax = plt.subplots(dpi=150)
    ax.plot(steps, KE, label="Kinetic")
    ax.plot(steps, PE, label="Potential")
    ax.plot(steps, TE, label="Total")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Energy [eV]")
    ax.legend()
    fig.tight_layout()

    # plot the temperature
    fig_temp, ax_temp = plt.subplots(dpi=150)
    ax_temp.plot(steps, TK)
    ax_temp.set_xlabel("Frame")
    ax_temp.set_ylabel("Temperature [K]")
    fig_temp.tight_layout()

    # plot the number of unique molecular species (from the graph) over time
    species = sorted({formula for counts in SPECIES_COUNTS for formula in counts})
    fig_species, ax_species = plt.subplots(dpi=150)
    for formula in species:
        ax_species.plot(
            steps, [counts.get(formula, 0) for counts in SPECIES_COUNTS], label=formula
        )
    ax_species.set_xlabel("Frame")
    ax_species.set_ylabel("Count")
    ax_species.legend()
    fig_species.tight_layout()

    # show the plots and if output is given, save them
    if args.output:
        fig.savefig(f"{args.output}_energy.png")
        fig_temp.savefig(f"{args.output}_temperature.png")
        fig_species.savefig(f"{args.output}_species.png")
    plt.show()


if __name__ == "__main__":
    main()

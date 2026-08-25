#!/usr/bin/env python3
"""Pack a box of H2 and O2 at a chosen composition.

Two ways to set the size, and for a composition sweep they are not
interchangeable:

  `--density`  mass density in kg/m^3.  The natural control for a real vessel.
  `--box`      cubic edge in Angstrom, i.e. fixed *number* density.

O2 is sixteen times heavier than H2, so holding mass density fixed while varying
the ratio changes the box a great deal: across H2:O2 = 1:2 to 8:1 at 100
molecules and 250 kg/m^3 the edge runs 24.4 A down to 15.2 A, a four-fold range
in volume.  Collision rates then move with composition, which confounds exactly
the variable a stoichiometry sweep is trying to isolate.  `--box` holds the
volume instead, so composition is the only thing that changes.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

from ase import Atoms, io
from molify import pack

from DynamicTopology.core import ReactionSet


def mass_density_for_edge(
    templates: list[Atoms], counts: list[int], edge: float
) -> float:
    """Mass density (kg/m^3) that packs `counts` of `templates` into `edge` A.

    `molify.pack` sizes its box from a mass density, so a fixed edge has to be
    converted to the density that produces it -- which depends on the
    composition, and is the whole reason the two options differ.
    """
    amu_to_kg = 1.66053906660e-27
    angstrom_to_m = 1e-10
    mass = sum(
        n * float(template.get_masses().sum()) for template, n in zip(templates, counts)
    )
    volume = (edge * angstrom_to_m) ** 3
    return mass * amu_to_kg / volume


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-x", "--ratio", help="H2 to O2 ratio", default="1:1")
    parser.add_argument(
        "-n", "--number", help="Total number of H2/O2 molecules", type=int, default=100
    )
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "-d",
        "--density",
        help="Target mass density in kg/m^3",
        type=float,
        default=None,
    )
    size.add_argument(
        "-b",
        "--box",
        type=float,
        default=None,
        help="Cubic box edge in Angstrom -- fixed number density. Use this for a "
        "composition sweep, so the box does not shrink as O2 is replaced by H2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="packing seed. `molify.pack` defaults to 42 and is deterministic, "
        "so this is stated rather than inherited: a box is then reproducible from "
        "the command that made it, not from a library default that could change.",
    )
    parser.add_argument("-o", "--output", default=sys.stdout)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.density is None and args.box is None:
        args.density = 30.0  # the long-standing default

    reaction_set = ReactionSet(args.rnet)

    # `get_molecules(formulas=...)` yields in the order asked, so `nums` below
    # lines up with the H2:O2 ratio rather than silently inverting it.
    mols = reaction_set.get_molecules(formulas=["H2", "O2"])
    atoms: list[list[Atoms]] = [[m.atoms] for m in mols]

    ratio = [float(r) for r in args.ratio.split(":")]
    nums = [int(args.number * x / sum(ratio)) for x in ratio]
    density = args.density

    # debug mixture
    if args.debug:
        mols = reaction_set.get_molecules()
        atoms = [[m.atoms] for m in mols]
        nums = [1] * len(atoms)
        density = 30.0

    if density is None:
        density = mass_density_for_edge([a[0] for a in atoms], nums, args.box)

    mixture = pack(atoms, nums, density, seed=args.seed)

    if isinstance(args.output, str):
        outfile = Path(args.output).resolve()
    else:
        outfile = args.output
    io.write(outfile, mixture, format="extxyz")

    print(
        f"{args.ratio:>5}  {nums[0]:>3} H2 + {nums[1]:>3} O2 = {len(mixture):>3} atoms"
        f"  box {mixture.cell.lengths()[0]:.3f} A  density {density:.2f} kg/m^3"
        f"  seed {args.seed}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()

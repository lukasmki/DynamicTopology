import sys
from argparse import ArgumentParser
from pathlib import Path

from ase import Atoms, io
from molify import pack

from DynamicTopology.core import ReactionSet


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-x", "--ratio", help="H2 to O2 ratio", default="1:1")
    parser.add_argument(
        "-n", "--number", help="Total nsumber of H2/O2 molecules", type=int, default=100
    )
    parser.add_argument(
        "-d", "--density", help="Target density in kg/m^3", type=float, default=30.0
    )
    parser.add_argument("-o", "--output", default=sys.stdout)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    reaction_set = ReactionSet(args.rnet)

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

    # print(atoms, nums, args.density)
    mixture = pack(atoms, nums, density)

    if isinstance(args.output, str):
        outfile = Path(args.output).resolve()
    else:
        outfile = args.output
    io.write(outfile, mixture, format="extxyz")


if __name__ == "__main__":
    main()

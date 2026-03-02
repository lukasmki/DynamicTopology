from molify import ase2networkx
from argparse import ArgumentParser
from pprint import pprint
from ase import Atoms, io

import matplotlib.pyplot as plt

from DynamicTopology.core import ReactionSet
from DynamicTopology.ase import DynamicTopology


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=False)
    args = parser.parse_args()

    atoms: Atoms | list[Atoms] = io.read(args.input, index=":")
    if isinstance(atoms, Atoms):
        atoms: list[Atoms] = [atoms]

    TK = []
    PE = []
    KE = []
    POP = []
    for frame in atoms:
        frame: Atoms
        TK.append(frame.get_temperature())
        PE.append(frame.get_potential_energy())
        KE.append(frame.get_kinetic_energy())

        graph = ase2networkx(frame, True)
        print(graph)


if __name__ == "__main__":
    main()

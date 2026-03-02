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

    reaction_set = ReactionSet(args.rnet)

    energies = []
    for frame in atoms:
        frame.calc = DynamicTopology(frame, reaction_set)
        energies.append(frame.get_potential_energy())
        pprint(frame.calc.results)

    if len(energies) > 1:
        plt.figure(dpi=150)
        plt.plot(energies)
        plt.ylabel("Energy [eV]")
        plt.xlabel("Frame")
        plt.show()


if __name__ == "__main__":
    main()

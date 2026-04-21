import logging
from argparse import ArgumentParser
from ase import Atoms, io
import cProfile
import pstats

from DynamicTopology.core import ReactionSet
from DynamicTopology.ase import DynamicTopology

logging.basicConfig(level=logging.DEBUG)


def main():
    atoms = io.read(args.input, index=0)
    assert isinstance(atoms, Atoms)

    profiler = cProfile.Profile()

    reaction_set = ReactionSet(args.rnet)
    atoms.calc = DynamicTopology(atoms, reaction_set)

    profiler.enable()
    energy = atoms.get_potential_energy()
    profiler.disable()
    print("energy", energy)

    stats = pstats.Stats(profiler)
    stats = stats.strip_dirs()
    stats = stats.sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-i", "--input", default="tests/data/mix-n100-d250.xyz")
    args = parser.parse_args()
    main()

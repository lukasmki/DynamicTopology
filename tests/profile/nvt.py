from ase.md import VelocityVerlet
from argparse import ArgumentParser
from ase import Atoms, io, units
import cProfile
import pstats

from DynamicTopology.core import ReactionSet
from DynamicTopology.ase import DynamicTopology


def status(atoms: Atoms, stepnum: int):
    PE = float(atoms.get_potential_energy())
    KE = float(atoms.get_kinetic_energy())
    TE: float = KE + PE
    T = float(atoms.get_temperature())
    print(f"{stepnum:4} {TE=:15.6f}, {PE=:15.6f}, {KE=:15.6f}, {T=:15.6f}")


def main():
    atoms = io.read(args.input, index=0)
    assert isinstance(atoms, Atoms)

    profiler = cProfile.Profile()

    reaction_set = ReactionSet(args.rnet)
    atoms.calc = DynamicTopology(atoms, reaction_set)

    dyn = VelocityVerlet(atoms, timestep=0.5 * units.fs)
    dyn.attach(status, 1, atoms, dyn.nsteps)

    profiler.enable()
    dyn.run(steps=5)
    profiler.disable()

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

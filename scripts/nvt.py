from ase.md.md import MolecularDynamics
from pathlib import Path
from argparse import ArgumentParser
from ase import Atoms, io, units

from ase.md.velocitydistribution import thermalize_momenta
from ase.md import Langevin

from DynamicTopology.core import ReactionSet
from DynamicTopology.ase import DynamicTopology, EVB


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "-o", "--output", type=Path, required=False, default=Path("nvt.xyz")
    )
    args = parser.parse_args()

    atoms: Atoms | list[Atoms] = io.read(args.input, index=-1)
    assert isinstance(atoms, Atoms)

    if not args.restart:
        thermalize_momenta(atoms, 2000)

    reaction_set = ReactionSet(args.rnet)
    atoms.calc = DynamicTopology(atoms, reaction_set)
    # atoms.calc = EVB(atoms, reaction_set)

    def status(atoms: Atoms, dyn: MolecularDynamics):
        PE = float(atoms.get_potential_energy())
        KE = float(atoms.get_kinetic_energy())
        TE: float = KE + PE
        T = float(atoms.get_temperature())
        print(f"{dyn.nsteps:4} {TE=:15.6f}, {PE=:15.6f}, {KE=:15.6f}, {T=:15.6f}")

    dyn = Langevin(
        atoms,
        timestep=1.0 * units.fs,
        temperature_K=2000,
        friction=0.01 / units.fs,
        fixcm=False,
    )
    dyn.attach(status, 1, atoms, dyn)
    dyn.attach(io.write, 5, args.output, atoms, append=True)
    dyn.run(steps=2000)

    # pprint(atoms.calc.results)


if __name__ == "__main__":
    main()

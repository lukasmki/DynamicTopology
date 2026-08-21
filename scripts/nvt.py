from ase.md.md import MolecularDynamics
from pathlib import Path
from argparse import ArgumentParser
from ase import Atoms, io, units

from ase.constraints import FixCom
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
    parser.add_argument("-n", "--steps", type=int, default=2000)
    args = parser.parse_args()

    atoms: Atoms | list[Atoms] = io.read(args.input, index=-1)
    assert isinstance(atoms, Atoms)

    if not args.restart:
        thermalize_momenta(atoms, 2000)

    reaction_set = ReactionSet(args.rnet)
    atoms.calc = DynamicTopology(atoms, reaction_set)
    # atoms.calc = EVB(atoms, reaction_set)

    # Previous total energy, so the jump across a topology change can be
    # reported.  A pivot-invariant basis means changing the carried topology is
    # bookkeeping and must not move the energy, so a large jump here is the
    # symptom to watch for.
    previous: list[float] = []

    def status(atoms: Atoms, dyn: MolecularDynamics):
        PE = float(atoms.get_potential_energy())
        KE = float(atoms.get_kinetic_energy())
        TE: float = KE + PE
        T = float(atoms.get_temperature())
        print(f"{dyn.nsteps:4} {TE=:15.6f}, {PE=:15.6f}, {KE=:15.6f}, {T=:15.6f}")

        diagnostics = getattr(atoms.calc, "diagnostics", None)
        if diagnostics is None:  # the EVB calculator holds fixed states
            return

        blocks = diagnostics["blocks"]
        capped = [i for i, b in enumerate(blocks) if b["capped"]]
        if capped:
            print(
                f"     WARN basis truncated in {len(capped)} block(s) "
                f"{capped[:5]}; the surface is seed-dependent where this fires"
            )

        if diagnostics["topology_changed"]:
            jump = f"{TE - previous[-1]:+.6f} eV" if previous else "n/a"
            reactive = [
                (i, b["nstates"], float(b["weights"].max()), b["gap"])
                for i, b in enumerate(blocks)
                if b["nstates"] > 1
            ]
            print(f"     TOPOLOGY CHANGED  dE_total = {jump}")
            for i, nstates, weight, gap in reactive[:5]:
                print(
                    f"       block {i}: {nstates} states, "
                    f"max c^2 = {weight:.3f}, gap = {gap:.4f} eV"
                )
            placeholders = sorted(
                {c for b in blocks for c in b["placeholder_channels"]}
            )
            if placeholders:
                print(f"       on unfitted couplings: {', '.join(placeholders)}")

        previous.append(TE)

    dyn = Langevin(
        atoms,
        timestep=0.5 * units.fs,
        temperature_K=2000,
        friction=0.01 / units.fs,
        fixcm=False,
    )
    atoms.set_constraint(FixCom())
    dyn.attach(status, 1, atoms, dyn)
    dyn.attach(io.write, 5, args.output, atoms, append=True)
    dyn.run(steps=args.steps)

    # pprint(atoms.calc.results)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit EVB couplings for every reaction in a dataset and write the .jsonl terms.

Each reaction's amplitude and width come from the stationary points already
stored in its `rxn_*.xyz` (reactant / transition state / product):

  amplitude  fitted by inverting the 2x2 secular equation at the transition
             state, which needs a reference energy on that frame.  Falls back
             to --amplitude when the frame carries none, so a dataset with
             geometries but no computed energies still gets correct widths.
  width      fitted from the endpoint geometries so the coupling is switched
             off at the reactant and product minima.  Geometry only.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

from ase import Atoms, io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.fit.dissociation import (
    DissociationFitError,
    fit_dissociation_energies,
    scale_factor,
)
from DynamicTopology.fit.coupling import (
    DEFAULT_EPS,
    CouplingFitError,
    fit_coupling,
)
from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.fit.dissociation import bonded_energy
from DynamicTopology.io.json import read_jsonl, write_jsonl


def diabatic_energy(
    reaction_set: ReactionSet, qforce: QForce, frame: Atoms, positions
) -> float:
    """Bonded energy of `frame`'s connectivity evaluated at `positions`."""
    atoms = frame.copy()
    atoms.calc = None
    atoms.positions = positions
    topology = Topology.from_atoms(atoms)
    topology.set_terms(reaction_set.get_terms(topology))
    return qforce(atoms.positions, atoms.pbc, atoms.cell, topology.term_dict)[0]


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--rnet", default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="amplitude (eV) to use where a transition state has no reference "
        "energy; without it such reactions are reported and skipped",
    )
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument(
        "--bonds",
        action="store_true",
        help="first rescale each molecule template's Morse well depths so "
        "its bonds carry its full atomization energy, leaving no constant "
        "shift. Do this before fitting couplings: the shift is what makes "
        "reactant and product disagree about the energy of a broken bond.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.rnet).resolve()
    manifest = json.loads(manifest_path.read_text())

    if args.bonds:
        print(f"{'molecule':<16}{'scale':>10}{'E_bonded':>12}{'E_reference':>13}")
        for entry in manifest["molecules"]:
            stem = manifest_path.parent / entry["path"]
            atoms = io.read(stem.with_suffix(".xyz"))
            terms = read_jsonl(stem.with_suffix(".jsonl"))
            try:
                fitted = fit_dissociation_energies(atoms, terms)
            except DissociationFitError as error:
                print(f"{stem.name:<16}{'-':>10}  FAILED: {error}")
                continue
            print(
                f"{stem.name:<16}{scale_factor(atoms, terms, fitted):>10.4f}"
                f"{bonded_energy(atoms, fitted):>12.5f}"
                f"{atoms.get_potential_energy():>13.5f}"
            )
            if not args.dry_run:
                write_jsonl(stem.with_suffix(".jsonl"), fitted, exist_ok=True)
        print()

    reaction_set = ReactionSet(manifest_path)
    qforce = QForce()

    print(f"{'reaction':<16}{'A (eV)':>12}{'a (1/A^2)':>12}  source")
    fitted = skipped = failed_amplitude = 0
    for entry in manifest["reactions"]:
        stem = manifest_path.parent / entry["path"]
        frames = io.read(stem.with_suffix(".xyz"), index=":")
        transition = frames[len(frames) // 2]

        try:
            if transition.calc is not None:
                energies = (
                    diabatic_energy(
                        reaction_set, qforce, frames[0], transition.positions
                    ),
                    diabatic_energy(
                        reaction_set, qforce, frames[-1], transition.positions
                    ),
                )
                terms = fit_coupling(frames, diabatic_energies=energies, eps=args.eps)
                source = "fitted from TS energy"
            elif args.amplitude is not None:
                terms = fit_coupling(frames, amplitude=args.amplitude, eps=args.eps)
                source = "width only (--amplitude)"
            else:
                print(
                    f"{stem.name:<16}{'-':>12}{'-':>12}  SKIPPED: no reference energy "
                    "on the transition state; run scripts/compute.py over the "
                    "reaction files, or pass --amplitude"
                )
                skipped += 1
                continue
        except CouplingFitError as error:
            if args.amplitude is None:
                print(f"{stem.name:<16}{'-':>12}{'-':>12}  FAILED: {error}")
                skipped += 1
                continue
            # The amplitude could not be fitted, but the width still can: it
            # depends only on geometry.  Falling back keeps the coupling
            # switched off at the endpoints, which is most of the benefit, and
            # leaves the amplitude provisional rather than dropping the channel.
            terms = fit_coupling(frames, amplitude=args.amplitude, eps=args.eps)
            source = f"width only (amplitude unfittable: {error.args[0][:60]}...)"
            failed_amplitude += 1

        kwargs = terms[0]["kwargs"]
        print(
            f"{stem.name:<16}{kwargs['A']:>12.4f}{kwargs['a']:>12.2f}  {source}"
        )
        if not args.dry_run:
            write_jsonl(stem.with_suffix(".jsonl"), terms, exist_ok=True)
        fitted += 1

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"\n{verb} {fitted} reaction term files "
        f"({fitted - failed_amplitude} with a fitted amplitude, "
        f"{failed_amplitude} width-only); {skipped} skipped"
    )
    if failed_amplitude:
        print(
            "\nAn unfittable amplitude means the force field's diabatic energies at "
            "the transition-state geometry lie below the reference barrier, so no "
            "real coupling reproduces it.  Those channels keep a provisional "
            "amplitude; the fix is a better diabatic force field, not a larger "
            "coupling."
        )
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())

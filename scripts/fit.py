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

Inverting the secular equation only has a real root where the reference barrier
lies below both diabats, which with plain two-parameter Morse it mostly does not:
the form is too deep at stretched geometries, and only 1 of the 19 HCombustion
channels is fittable as q-force hands them over.  --force-constants refits the
bonds first so that they are; see `fit.dissociation`.

Three routes, and they are not equivalent:

  --fit-mode shape  fits the Morse shape parameter `c`, which is O(dr**3) at
                    `r0`.  13 of 19 channels.
  --fit-mode k      buys the same depth by stiffening the bonds instead.  Tops
                    out at 17 of 19 and needs H2 at 12402 cm^-1 against an
                    experimental 4401 to get there; the count saturates, so a
                    cap of 100 buys what a cap of 16 does.
  --fit-mode both   the default.  Spends the free parameter first and the
                    minimum frequency for the rest: 18 of 19 at 1.41x.

**`--fit-mode shape` is not free, and this file said it was.**  `c` costs no
frequency *at `r0`*, which was the same thing as "at the minimum" only while the
bonded terms were the whole potential.  `fit_bond_lengths` ended that: `r0` is
now pulled 0.04-0.22 A inside the reference bond length so the Morse can lean
against the repulsion, and at that displacement the shape term is the largest
single contribution to the stiffness -- 63 eV/A**2 of H2's 115, 509 of O2's 793,
323 of HO's 480.  The surface it produced had an 11735 cm^-1 mode on it while
the `k`-scale reported 1.41x.

`w total` in the table below is the honest number and `--max-wavenumber` is what
prices it; see `fit.dissociation.DEFAULT_MAX_WAVENUMBER`.  The frequency table
is printed in cm^-1, where the cost is legible.
"""

from argparse import ArgumentParser
from pathlib import Path
import json
import sys

import numpy as np
from ase import Atoms, io

from DynamicTopology.core import ReactionSet
from DynamicTopology.fit.dissociation import (
    DEFAULT_CURVATURE_WEIGHT,
    DEFAULT_FREQUENCY_WEIGHT,
    DEFAULT_MARGIN,
    DEFAULT_MAX_SCALE,
    DEFAULT_MAX_SHAPE,
    DEFAULT_MAX_WAVENUMBER,
    DissociationFitError,
    bonded_energy,
    diabatic_energy,
    fit_dissociation_energies,
    fit_force_constants,
    frequency,
    install_templates,
    scale_factor,
    total_wavenumber,
)
from DynamicTopology.fit.coupling import (
    DEFAULT_EPS,
    CouplingFitError,
    fit_coupling,
)
from DynamicTopology.forcefield.qforce import QForce, SHAPE_DECAY
from DynamicTopology.io.json import read_jsonl, write_jsonl

# Masses used for the frequency report only, so a q-force force constant can be
# quoted as a wavenumber.  Nothing in the force field reads them.  `ase.data`
# is where `fit.dissociation` takes them from; these agree with it to the digits
# a wavenumber is printed to and are kept because the report also needs a
# fallback for an element that is not in a template.
MASSES: dict[str, float] = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}


def load_templates(
    manifest_path: Path, manifest: dict
) -> list[tuple[str, Atoms, list]]:
    """`(name, atoms, terms)` per molecule entry."""
    templates = []
    for entry in manifest["molecules"]:
        stem = manifest_path.parent / entry["path"]
        templates.append(
            (
                stem.name,
                io.read(stem.with_suffix(".xyz")),
                read_jsonl(stem.with_suffix(".jsonl")),
            )
        )
    return templates


def load_reactions(
    manifest_path: Path, manifest: dict
) -> list[tuple[str, list[Atoms]]]:
    """`(name, frames)` per reaction entry, frames in reactant/TS/product order."""
    return [
        (
            (manifest_path.parent / entry["path"]).name,
            io.read(
                (manifest_path.parent / entry["path"]).with_suffix(".xyz"), index=":"
            ),
        )
        for entry in manifest["reactions"]
    ]


def report_force_constants(fit, max_wavenumber: float = 0.0) -> None:
    """What the fit moved, and what it cost in wavenumbers.

    `w total` is the number the timestep is set by, so any row over
    `max_wavenumber` is flagged: those are the modes that would have to be
    integrated, and a fit that leaves one there has not bought a larger step
    however good its margins look.
    """
    if fit.mode in ("shape", "both"):
        print(
            f"{'bond':<22}{'c':>7}{'peak (eV)':>11}{'k scale':>9}"
            f"{'w before':>10}{'w total':>9}{'ratio':>7}  over cap"
        )
        k_scales = fit.k_scales or [1.0] * len(fit.variables)
        curvatures = fit.curvatures or [None] * len(fit.variables)
        for variable, shape, k_scale, curvature in zip(
            fit.variables, fit.scales, k_scales, curvatures
        ):
            masses = [MASSES.get(element, 1.0) for element in variable.elements]
            label = f"{variable.template} {'-'.join(variable.elements)}"
            # `D * c * s**3 * exp(-b s)` is maximal at `s = 3/b`, where it is
            # `(3/b)**3 * exp(-3) * c * D`.  `D` is re-solved by the inner fit,
            # so quote the bump against the depth the fitted terms ended up with
            # rather than the input one.
            depth = fit.depths.get((variable.template, variable.r0, variable.k), 0.0)
            peak = (3.0 / SHAPE_DECAY) ** 3 * np.exp(-3.0) * shape * depth
            before_w = frequency(variable.k, *masses)
            after_w = (
                total_wavenumber(curvature, *masses)
                if curvature is not None
                else frequency(variable.k * k_scale, *masses)
            )
            over = (
                f"  +{after_w - max_wavenumber:.0f}"
                if max_wavenumber > 0.0 and after_w > max_wavenumber
                else ""
            )
            print(
                f"{label:<22}{shape:>7.3f}{peak:>11.3f}"
                f"{k_scale:>9.3f}{before_w:>10.0f}{after_w:>9.0f}"
                f"{after_w / before_w:>7.2f}{over}"
            )
        # Measured on the total curvature, not on the k-scale.  The k-scale is
        # what the fit *chose*; the ratio below is what the molecule ends up
        # with, and with an untapered short-range repulsion in the force field
        # the two are not the same number.
        worst = max(
            (
                total_wavenumber(curvature, *[MASSES.get(e, 1.0) for e in v.elements])
                / frequency(v.k, *[MASSES.get(e, 1.0) for e in v.elements])
                for v, curvature in zip(fit.variables, curvatures)
                if curvature is not None
            ),
            default=max((max(v, 1.0 / v) for v in k_scales), default=1.0) ** 0.5,
        )
        print(f"\nworst frequency drift {worst:.2f}x")
        _report_timestep(fit, max_wavenumber)
        _report_margins(fit)
        return

    print(
        f"{'bond':<22}{'scale':>8}{'k before':>12}{'k after':>12}"
        f"{'w before':>11}{'w after':>10}{'ratio':>8}"
    )
    for variable, scale in zip(fit.variables, fit.scales):
        # The depth scale the inner solve chose moves `a = sqrt(k/2D)` too, but
        # the frequency is `sqrt(k/mu)` and depends on `k` alone.
        after = variable.k * scale
        masses = [MASSES.get(element, 1.0) for element in variable.elements]
        before_w = frequency(variable.k, *masses)
        after_w = frequency(after, *masses)
        label = f"{variable.template} {'-'.join(variable.elements)}"
        print(
            f"{label:<22}{scale:>8.3f}{variable.k:>12.0f}{after:>12.0f}"
            f"{before_w:>11.0f}{after_w:>10.0f}{after_w / before_w:>8.2f}"
        )
    # Reported in whichever direction moved furthest: a force constant that
    # dropped by half is as much of a change as one that doubled.
    worst = max((max(s, 1.0 / s) for s in fit.scales), default=1.0) ** 0.5
    print(f"\nworst frequency drift {worst:.2f}x")
    _report_margins(fit)


def _report_timestep(fit, max_wavenumber: float) -> None:
    """The worst stretching mode, and the timestep that follows from it.

    Velocity Verlet wants roughly 15 steps per vibrational period, so the
    fastest mode on the surface is what a production run's timestep has to be
    chosen against.  Printed here because it is the point of the cap and it is
    otherwise a calculation the reader has to do themselves.
    """
    masses = [
        [MASSES.get(element, 1.0) for element in variable.elements]
        for variable in fit.variables
    ]
    modes = [
        (total_wavenumber(curvature, *pair), variable)
        for curvature, pair, variable in zip(fit.curvatures, masses, fit.variables)
        if curvature is not None
    ]
    if not modes:
        return
    worst, variable = max(modes, key=lambda item: item[0])
    label = f"{variable.template} {'-'.join(variable.elements)}"
    # 1 cm^-1 is 29.9793 THz^-1; a period in fs is 33356.4 / nu.
    period = 33356.4 / worst if worst > 0 else float("inf")
    print(
        f"fastest mode {worst:.0f} cm^-1 ({label}), period {period:.2f} fs "
        f"-> {period / 15.0:.3f} fs at 15 steps/period"
    )
    if max_wavenumber > 0.0:
        over = sum(1 for value, _ in modes if value > max_wavenumber)
        print(f"{over} of {len(modes)} bond types over the {max_wavenumber:.0f} cap")


def _report_margins(fit) -> None:
    """Per-reaction feasibility, which every mode has to report the same way."""
    print(f"\n{'reaction':<16}{'margin before':>16}{'margin after':>14}  feasible")
    for name, after_margin in fit.margins.items():
        before_margin = fit.margins_before[name]
        print(
            f"{name:<16}{before_margin:>16.4f}{after_margin:>14.4f}"
            f"  {'yes' if after_margin > 0.0 else 'NO'}"
        )
    feasible = sum(value > 0.0 for value in fit.margins.values())
    print(f"\n{feasible} of {len(fit.margins)} channels fittable")


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--rnet", default="datasets/HCombustion/HCombustion.json")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS)
    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="amplitude (eV) to use where a transition state has no reference "
        "energy; without it such reactions are reported and skipped. A channel "
        "whose barrier cannot be inverted is decoupled (A = 0) rather than given "
        "this value.",
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
    parser.add_argument(
        "--force-constants",
        action="store_true",
        help="refit the Morse force constants as well as the depths, so the "
        "diabats rise above the reference barriers and the amplitudes become "
        "invertible. Implies --bonds. Paid for in vibrational frequencies; see "
        "--frequency-weight and --max-k-scale.",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["shape", "k", "both"],
        default="both",
        help="`shape` fits the Hulburt-Hirschfelder `c` per bond type and "
        "leaves every force constant, and so every vibrational frequency, "
        "exactly as q-force fitted it. `k` is the older route that buys the "
        "same depth by stiffening the bonds instead; it tops out at 17 of 19 "
        "channels and needs H2 at 12402 cm^-1 to get there.",
    )
    parser.add_argument(
        "--max-shape",
        type=float,
        default=DEFAULT_MAX_SHAPE,
        help="upper bound on `c`. 0 freezes the shape term, which is the "
        "vacuity check: plain Morse must make no progress.",
    )
    parser.add_argument(
        "--max-k-scale",
        type=float,
        default=DEFAULT_MAX_SCALE,
        help="hard bound on each force-constant scale. Its square root is the "
        "cap on frequency drift; 1.0 freezes the force constants entirely. "
        "Relative to the force constants in the .jsonl files as they stand, so "
        "re-running over already-fitted output compounds the bound.",
    )
    parser.add_argument(
        "--frequency-weight",
        type=float,
        default=DEFAULT_FREQUENCY_WEIGHT,
        help="how hard the fit is pulled back towards q-force's force constants",
    )
    parser.add_argument(
        "--max-wavenumber",
        type=float,
        default=DEFAULT_MAX_WAVENUMBER,
        help="stretching modes above this (cm^-1) cost the objective. This is "
        "the timestep expressed as a force-field property: ~15 steps per period "
        "means a dt of 0.5 fs needs everything under 4450.",
    )
    parser.add_argument(
        "--curvature-weight",
        type=float,
        default=DEFAULT_CURVATURE_WEIGHT,
        help="how much a mode over --max-wavenumber costs. 0 removes the cap "
        "entirely, which is the vacuity check for it.",
    )
    parser.add_argument(
        "--refit-manual",
        action="store_true",
        help="overwrite amplitudes marked `provenance: manual` in the existing "
        "term files. Without it they are kept and only their width is refitted.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=DEFAULT_MARGIN,
        help="how far below the reference barrier (eV) a diabat must sit before "
        "the channel counts as fittable",
    )
    args = parser.parse_args()

    manifest_path = Path(args.rnet).resolve()
    manifest = json.loads(manifest_path.read_text())

    molecule_stems = [
        manifest_path.parent / entry["path"] for entry in manifest["molecules"]
    ]

    # Built once and refitted in place.  The coupling fit below reads its
    # diabats from this object, so a --dry-run reports the couplings the bond
    # fit would actually produce instead of the ones already on disk.
    reaction_set = ReactionSet(manifest_path)

    if args.force_constants:
        templates = load_templates(manifest_path, manifest)
        fit = fit_force_constants(
            reaction_set,
            templates,
            load_reactions(manifest_path, manifest),
            margin=args.margin,
            frequency_weight=args.frequency_weight,
            max_scale=args.max_k_scale,
            mode=args.fit_mode,
            max_shape=args.max_shape,
            max_wavenumber=args.max_wavenumber,
            curvature_weight=args.curvature_weight,
        )
        report_force_constants(fit, args.max_wavenumber)
        if not args.dry_run:
            for (name, _, _), stem in zip(templates, molecule_stems):
                write_jsonl(stem.with_suffix(".jsonl"), fit.terms[name], exist_ok=True)
        print()

    elif args.bonds:
        print(f"{'molecule':<16}{'scale':>10}{'E_bonded':>12}{'E_reference':>13}")
        templates = load_templates(manifest_path, manifest)
        fitted_terms = []
        for (name, atoms, terms), stem in zip(templates, molecule_stems):
            try:
                fitted = fit_dissociation_energies(atoms, terms)
            except DissociationFitError as error:
                print(f"{name:<16}{'-':>10}  FAILED: {error}")
                fitted_terms.append(terms)
                continue
            print(
                f"{name:<16}{scale_factor(atoms, terms, fitted):>10.4f}"
                f"{bonded_energy(atoms, fitted):>12.5f}"
                f"{atoms.get_potential_energy():>13.5f}"
            )
            fitted_terms.append(fitted)
            if not args.dry_run:
                write_jsonl(stem.with_suffix(".jsonl"), fitted, exist_ok=True)
        install_templates(reaction_set, templates, fitted_terms)
        print()

    qforce = QForce()

    print(f"{'reaction':<16}{'A (eV)':>12}{'a (1/A^2)':>12}  source")
    fitted = skipped = decoupled = preserved = 0
    for entry in manifest["reactions"]:
        stem = manifest_path.parent / entry["path"]
        frames = io.read(stem.with_suffix(".xyz"), index=":")
        transition = frames[len(frames) // 2]

        # A hand-set amplitude is a decision, not a stale fit, so a rerun must
        # not silently discard it.  Only the width is recomputed -- it is pure
        # geometry, and the stored one is generally wrong anyway: a channel that
        # reached `manual` by way of the decoupled path carries the width for
        # `NOMINAL_AMPLITUDE`, not for the amplitude someone then wrote in.
        existing = stem.with_suffix(".jsonl")
        manual = None
        if not args.refit_manual and existing.exists():
            stored = read_jsonl(existing)
            if stored and stored[0].get("provenance") == "manual":
                manual = stored[0]["kwargs"]["A"]

        try:
            if manual is not None:
                terms = fit_coupling(frames, amplitude=manual, eps=args.eps)
                terms[0]["provenance"] = "manual"
                source = "kept (manual; --refit-manual to replace)"
                preserved += 1
            elif transition.calc is not None:
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
            # The barrier could not be inverted, so there is no amplitude to
            # write.  Decoupling says exactly that: A = 0 contributes no
            # stabilization, so `EVBBasis` never admits the state and the
            # Hamiltonian is not driven by a number nobody fitted.  The width
            # still comes from geometry and is kept, so filling the amplitude in
            # later needs no refit.
            amplitude = 0.0 if args.amplitude is None else args.amplitude
            terms = fit_coupling(frames, amplitude=amplitude, eps=args.eps)
            source = f"{terms[0]['provenance']} ({error.args[0][:52]}...)"
            decoupled += 1

        kwargs = terms[0]["kwargs"]
        print(f"{stem.name:<16}{kwargs['A']:>12.4f}{kwargs['a']:>12.2f}  {source}")
        if not args.dry_run:
            write_jsonl(stem.with_suffix(".jsonl"), terms, exist_ok=True)
        fitted += 1

    verb = "would write" if args.dry_run else "wrote"
    print(
        f"\n{verb} {fitted} reaction term files "
        f"({fitted - decoupled - preserved} with a fitted amplitude, "
        f"{decoupled} decoupled, {preserved} manual and kept); {skipped} skipped"
    )
    if decoupled:
        # Worth stating loudly, because it is invisible at run time: a decoupled
        # channel never enters a basis, so it never reaches
        # `Block.placeholder_channels` either.  This line is the only place the
        # count is reported.
        print(
            f"\n{decoupled} channel(s) carry A = 0 and are switched off entirely: the "
            "force field's diabatic energies at the transition-state geometry lie "
            "below the reference barrier, so no real coupling reproduces it. They "
            "will not appear in any EVB basis and will not be reported at run time. "
            "The fix is a better diabatic force field -- try --force-constants, or a "
            "looser --frequency-weight if it is already on."
        )
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())

"""Rescale Morse well depths so a template's bonds carry its atomization energy.

`QForce` writes a bond as

    E = D * (1 - exp(-a * dr))**2 - D,    a = sqrt(k / 2D)

which is zero at dissociation and -D at the minimum, so the well depths a
molecule's bonds carry *are* its atomization energy -- if they add up to it.
As fitted by q-force they do not: each bond is parameterized locally, and for
the HCombustion set the sum is off by -0.44 to +2.59 eV per template.

`ReactionSet` currently absorbs that difference into a constant `reference`
term.  That keeps the energy right at the equilibrium geometry but is wrong
everywhere else, and wrong in a way that matters here: the constant belongs to
the *template*, so a reactant diabat still carries the whole parent molecule's
shift at a geometry where its bond is nearly broken, while the dissociated
state has moved to the fragments' own templates and carries theirs instead.
The reference energy is therefore discontinuous across exactly the region an
EVB coupling has to describe, which is why a coupling fitted at the transition
state cannot reproduce a reference barrier.

Scaling the well depths instead puts the atomization energy where the
functional form already expects it.  The reference term then vanishes
identically, the dissociation limit is exact rather than offset, and the
reactant and product descriptions agree about the energy of a broken bond.

The scale factor is fixed by one condition per template -- its bonded energy at
its own reference geometry must equal its reference atomization energy -- so
there is nothing to choose and nothing to over-fit.  The cost is
transferability: an O-H depth that was one number across HO, H2O, HO2 and H2O2
becomes four.  Templates are stored and looked up independently, so this costs
nothing mechanically, but the depths are no longer a per-element-pair table.
"""

import logging

import numpy as np
from ase import Atoms, units
from scipy.optimize import brentq

from DynamicTopology.core.types import Term
from DynamicTopology.forcefield.qforce import QForce

logger: logging.Logger = logging.getLogger(__name__)

# Bracket for the scale factor.  Wide enough for any sane reparameterization;
# a root outside it means the reference energy and the force field disagree
# about the molecule, not that the bracket is too tight.
SCALE_BRACKET: tuple[float, float] = (0.05, 20.0)


class DissociationFitError(ValueError):
    """Raised when no scaling reproduces the reference atomization energy."""


def _scaled(terms: list[Term], scale: float) -> list[Term]:
    """Copy of `terms` with every Morse depth scaled and the shift zeroed.

    The zero shift is stated explicitly even when the input had no `reference`
    term.  ReactionSet synthesizes one from `E_atomization + sum(D)` for any
    template that does not carry it, and after this fit that formula no longer
    evaluates to zero -- the depths were solved against the *full* bonded
    energy, so the leftover is the angle and cross-term contribution at the
    reference geometry.  Writing the zero down keeps the loader from adding
    that leftover a second time.
    """
    out: list[Term] = []
    seen_reference = False
    for term in terms:
        if term["type"] == "bond":
            kwargs = dict(term["kwargs"])
            kwargs["D"] = kwargs["D"] * scale
            out.append({**term, "kwargs": kwargs})
        elif term["type"] == "reference":
            seen_reference = True
            out.append({**term, "kwargs": {**term["kwargs"], "E0": 0.0}})
        else:
            out.append(term)
    if not seen_reference:
        out.append({"type": "reference", "atoms": {"a1": 0}, "kwargs": {"E0": 0.0}})
    return out


def bonded_energy(atoms: Atoms, terms: list[Term]) -> float:
    """Bonded energy of `terms` at `atoms`' geometry, in eV."""
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    topology.set_terms(terms)
    qforce = QForce(bond_form="morse")
    return qforce(atoms.positions, atoms.pbc, atoms.cell, topology.term_dict)[0]


def fit_dissociation_energies(atoms: Atoms, terms: list[Term]) -> list[Term]:
    """Scale one template's Morse depths to match its atomization energy.

    Args:
        atoms: the molecule template, carrying its reference atomization energy
            (eV, referenced to free atoms, so exactly zero for a lone atom).
        terms: the template's force field terms.

    Returns:
        The terms with every bond's `D` scaled by a single factor and the
        constant `reference` shift set to zero.  Templates with no bonds are
        returned unchanged.
    """
    bonds = [t for t in terms if t["type"] == "bond"]
    if not bonds:
        # A free atom: its atomization energy is zero by definition and there is
        # no well depth to carry it.
        return list(terms)

    if atoms.calc is None:
        raise DissociationFitError(
            "template carries no reference atomization energy; run "
            "scripts/compute.py over the molecule files first"
        )
    target = atoms.get_potential_energy()

    def residual(scale: float) -> float:
        return bonded_energy(atoms, _scaled(terms, scale)) - target

    low, high = SCALE_BRACKET
    f_low, f_high = residual(low), residual(high)
    if f_low * f_high > 0.0:
        raise DissociationFitError(
            f"no scale factor in [{low}, {high}] reproduces the reference "
            f"atomization energy {target:+.4f} eV: the bonded energy runs from "
            f"{f_low + target:+.4f} to {f_high + target:+.4f} eV over that range. "
            "The template's geometry, its parameters, or its reference energy "
            "disagree with each other."
        )

    scale = brentq(residual, low, high, xtol=1e-12, rtol=1e-14)
    return _scaled(terms, scale)


def scale_factor(atoms: Atoms, terms: list[Term], fitted: list[Term]) -> float:
    """Ratio between fitted and original depths, for reporting."""
    original = sum(t["kwargs"]["D"] for t in terms if t["type"] == "bond")
    updated = sum(t["kwargs"]["D"] for t in fitted if t["type"] == "bond")
    return updated / original if original else 1.0

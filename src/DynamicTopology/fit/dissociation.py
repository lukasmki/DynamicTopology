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


Making the barriers reachable: the third Morse parameter
--------------------------------------------------------

Scaling `D` fixes the minimum and the dissociation limit and leaves the shape in
between untouched, which is where the remaining error lives.  With `D` pinned by
the atomization energy, `r0` by the geometry and `k` by the vibrational
frequency, two-parameter Morse has nothing left, and what it produces is 0.55 to
1.83 eV *too deep* at stretched geometries.  A reference barrier then lands below
a diabat, and `fit.coupling.fit_amplitude` -- which has a real root only below
*both* -- cannot fit the channel at all.  Eighteen of nineteen HCombustion
channels failed for that reason alone.

**The route that does not work.**  Since `a = sqrt(k / 2D)`, raising `k` steepens
the exponential and lifts the curve mid-range, and `mode="k"` still does exactly
that.  It buys the depth with the frequency, and the exchange rate is terrible:
reaching even 17 of 19 channels needs H2 at 12402 cm^-1 against an experimental
4401, and *no* force constant whatever reaches 18 -- the count saturates at 17,
so a cap of 100 buys precisely what a cap of 16 does.  At frequencies anyone
would defend, one channel in nineteen is fittable.  That is not a knob setting,
it is the functional form running out.

**The route that does.**  `compute_bond` carries a Hulburt-Hirschfelder shape
term,

    s = a * max(dr, 0),   E = D * [ (1 - exp(-a*dr))**2 - 1 + c * s**3 exp(-2s) ]

whose whole point is that it is `O(dr**3)` at the minimum: it leaves `D`, `r0`
and the curvature -- and therefore every vibrational frequency -- *exactly*
where q-force put them, while raising the stretched branch by up to
`0.168 * c * D`.  `c = 0` is plain Morse, which is what every term file
predating the parameter reads as.

`c` is bounded, and not arbitrarily: past `c = 1.308` the correction beats the
exponential and the dissociation curve turns over, putting a barrier on a
channel that has none and a bound state beyond it.  `DEFAULT_MAX_SHAPE` is that
limit.  See its comment for the derivation.

`fit_force_constants` fits one variable per distinct bond type in any of three
modes -- `shape` (`c` only, no frequency cost), `k` (the old route), or `both`
-- always with the `D` scale re-solved underneath by `fit_dissociation_energies`,
so the atomization energy stays exact by construction rather than becoming one
residual among many.  The objective is a hinge: once a margin is positive the
barrier is reproduced exactly by the amplitude, which is free per reaction, so
overshooting buys nothing.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
from scipy.optimize import brentq, minimize

from DynamicTopology.core.types import Term
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.qforce import QForce

logger: logging.Logger = logging.getLogger(__name__)

# Bracket for the scale factor.  Wide enough for any sane reparameterization;
# a root outside it means the reference energy and the force field disagree
# about the molecule, not that the bracket is too tight.
SCALE_BRACKET: tuple[float, float] = (0.05, 20.0)

# How far below the reference barrier the lower diabat has to sit before the
# channel counts as fitted.  Small on purpose: once the margin is positive the
# barrier is reproduced *exactly* by the coupling amplitude, which is free per
# reaction, so overshooting buys nothing and costs frequency.  It also keeps
# |A| small, and a large amplitude is a coupling that reaches geometries it
# should not.
DEFAULT_MARGIN: float = 0.02

# Weight on log(k-scale)**2 in the objective, i.e. how hard the fit is pulled
# back towards q-force's force constants.  The soft half of the trade.  Light,
# because the hard bound below is doing most of the work: the hinge stops
# pushing on its own as soon as a channel is feasible.
DEFAULT_FREQUENCY_WEIGHT: float = 0.005

# Hard bound on each k-scale; the cap on frequency drift is its square root.
# Set it to 1.0 to freeze the force constants, which in `mode="k"` is the
# vacuity check.
#
# Two -- 1.41x in wavenumbers -- is where the knee is, and the knee is sharp.
# Swept over HCombustion in the default `both` mode, with `c` doing as much as
# it can before any frequency is spent:
#
#     cap 1.0 (shape only)  13 / 19   1.00x
#     cap 1.5               15 / 19   1.22x
#     cap 1.7               16 / 19   1.30x
#     cap 1.85              16 / 19   1.36x
#     cap 2.0               18 / 19   1.41x   <- shipped
#
# For contrast, `mode="k"` with no shape term at all needs 2.00x to reach 12,
# 3.00x to reach 16, and saturates at 17 -- a cap of 100 buys exactly what a cap
# of 16 does.  The third Morse parameter is what turns "spend 3.31x and still
# fall short" into "spend 1.41x and stop".
#
# The one remaining channel is `rxn_08`, and no cap reaches it: its stored
# transition state is not a saddle but a *minimum*, 4.87 eV below its own
# reactant.  See `tests/test_reference_energies.py`.
#
# Note that this bounds the scale relative to whatever `k` the templates handed
# in already carry, not relative to q-force's original fit, so **running the fit
# twice over its own output compounds the bound**.  One pass is the intended
# use; the shipped parameters are one pass, from q-force's own values.
DEFAULT_MAX_SCALE: float = 2.0


# Stateless, and constructed once: the outer fit calls `bonded_energy`
# thousands of times.
_ACKS2 = ACKS2()


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
    """Total energy of `terms` at `atoms`' geometry, in eV.

    Bonded *and* nonbonded, because that is the sum `System.calculate` reports
    and therefore the sum a reference atomization energy has to be matched
    against.  Fitting the depths against the bonded part alone left every
    heteronuclear template overbound by exactly its own ACKS2 energy -- water at
    -12.3701 eV against a reference of -9.8735, 25% too deep -- with the error
    invisible on H2 and O2, which have no charge separation and so no nonbonded
    energy at all.

    The name is now a slight lie, kept because it is the vocabulary the rest of
    this module and `scripts/fit.py` are written in; `tests/test_reference_
    energies.py` is what pins the meaning.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    topology.set_terms(terms)
    qforce = QForce(bond_form="morse")
    energy = qforce(atoms.positions, atoms.pbc, atoms.cell, topology.term_dict)[0]
    return energy + nonbonded_energy(atoms, topology.term_dict)


def nonbonded_energy(atoms: Atoms, term_dict: dict) -> float:
    """ACKS2 energy of `term_dict` at `atoms`' geometry, in eV.

    Topology-independent in the sense that matters here: evaluated at each of
    the nineteen transition-state geometries under the reactant's and the
    product's parameter sets it gives the same number to every printed digit, so
    it can be added once outside the EVB Hamiltonian -- which is exactly what
    `System.calculate` does -- rather than sitting on the diagonal.
    """
    return _ACKS2(atoms.positions, atoms.pbc, atoms.cell, term_dict)[0]


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


# --------------------------------------------------------------------------
# Force constants
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class BondVariable:
    """One fitted force constant.

    Identified by the template it belongs to and its `(r0, k)` there, so the
    two O-H bonds of water -- which q-force gave identical parameters -- are
    one variable rather than two.  Not shared *across* templates: the fit wants
    O=O in O2 six times stiffer and O-O in H2O2 left alone, and
    `fit_dissociation_energies` already gave up transferability for `D`.
    """

    template: str
    r0: float
    k: float
    elements: tuple[str, str]


@dataclass
class ForceConstantFit:
    """Result of `fit_force_constants`."""

    # Template name -> its refitted term list, ready for `io.json.write_jsonl`.
    terms: dict[str, list[Term]] = field(default_factory=dict)
    # The fitted variables, in the order they were solved for.
    variables: list[BondVariable] = field(default_factory=list)
    # Multiplier applied to each variable's `k`, aligned with `variables`.
    scales: list[float] = field(default_factory=list)
    # Reaction name -> min(H_reactant, H_product) - E_reference at its
    # transition state.  Positive is fittable.
    margins: dict[str, float] = field(default_factory=dict)
    # The same, before anything was refitted.
    margins_before: dict[str, float] = field(default_factory=dict)
    # "shape" -> `scales` holds Hulburt-Hirschfelder `c` values; "k" -> force
    # constant scale factors.  The two are reported in different units and the
    # reader has no way to tell them apart from the numbers alone.
    mode: str = "shape"
    # Fitted Morse depth per `(template, r0, k)`, in eV, so a report can quote
    # the shape term's bump as an energy rather than as a bare coefficient.
    # Keyed on the *input* `(r0, k)` because that is what `BondVariable` carries;
    # the inner solve moves `D` but never `r0` or `k`.
    depths: dict[tuple[str, float, float], float] = field(default_factory=dict)
    # Force-constant scale per bond type; all 1.0 unless `mode == "both"`, where
    # `scales` holds `c` and the frequencies move as well.
    k_scales: list[float] = field(default_factory=list)


def bond_types(terms: list[Term]) -> list[tuple[float, float]]:
    """The distinct `(r0, k)` a template's bonds carry, in term order."""
    types: list[tuple[float, float]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        key = (term["kwargs"]["r0"], term["kwargs"]["k"])
        if key not in types:
            types.append(key)
    return types


def bond_depths(terms: list[Term]) -> list[float]:
    """Fitted Morse depth of each bond type, in eV, in `bond_types` order.

    Term files are in q-force's kJ/mol; everything a human reads is in eV.
    """
    depths: list[float] = []
    seen: list[tuple[float, float]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        key = (term["kwargs"]["r0"], term["kwargs"]["k"])
        if key in seen:
            continue
        seen.append(key)
        depths.append(term["kwargs"]["D"] * units.kJ / units.mol)
    return depths


def scale_force_constants(terms: list[Term], scales: list[float]) -> list[Term]:
    """Copy of `terms` with each bond type's `k` multiplied by its scale.

    `scales` is aligned with `bond_types(terms)`.  `D` is left alone; it is the
    inner solve's variable, not this one's.
    """
    types = bond_types(terms)
    out: list[Term] = []
    for term in terms:
        if term["type"] != "bond":
            out.append(term)
            continue
        kwargs = dict(term["kwargs"])
        index = types.index((kwargs["r0"], kwargs["k"]))
        kwargs["k"] = kwargs["k"] * scales[index]
        out.append({**term, "kwargs": kwargs})
    return out


# Upper bound on the Hulburt-Hirschfelder shape parameter, and not an arbitrary
# one: it is the largest `c` for which the bond still dissociates downhill.
#
# On the stretched branch the slope is
#
#     dE/dr  =  D * a * exp(-b*s) * [ 2(exp((b-1)s) - exp((b-2)s))
#                                       + c * s**2 * (3 - b*s) ]
#
# and the bracket is what can go negative, since `(3 - b*s) < 0` beyond
# `s = 3/b`.  Requiring it to stay non-negative everywhere gives, at
# `b = SHAPE_DECAY = 4`,
#
#     c  <=  min over s > 0.75 of  2(exp(3s) - exp(2s)) / (s**2 (4s - 3))  =  19.33
#
# Past that the correction wins over the exponential
# and the curve turns over: a barrier appears on a dissociation channel that has
# none, and beyond the barrier a *bound* state at long range that would trap two
# fragments that should have separated.  The fit will happily walk there --
# nothing in a margin objective knows what a dissociation curve is supposed to
# look like -- so the constraint has to be in the bound.
#
# The limit is generous rather than binding -- at `c = 19.3` the bump is 41% of
# the well depth -- which is the point: the bound is there to keep the curve
# physical, and the hinge objective plus the regularizer are what decide how
# much of it to use.
DEFAULT_MAX_SHAPE: float = 19.3


def set_shape_parameters(terms: list[Term], values: list[float]) -> list[Term]:
    """Copy of `terms` with each bond type's `c` set, in `bond_types` order.

    Unlike `scale_force_constants` this *sets* rather than scales, because `c`
    starts at zero -- a term file that predates the parameter reads as plain
    Morse -- and zero has no scale factor.
    """
    order = {pair: index for index, pair in enumerate(bond_types(terms))}
    out: list[Term] = []
    for term in terms:
        if term["type"] != "bond":
            out.append(term)
            continue
        kwargs = dict(term["kwargs"])
        kwargs["c"] = float(values[order[(kwargs["r0"], kwargs["k"])]])
        out.append({**term, "kwargs": kwargs})
    return out


def frequency(force_constant: float, mass_a: float, mass_b: float) -> float:
    """Harmonic wavenumber (cm^-1) of a diatomic with this force constant.

    `force_constant` is in q-force units (kJ/mol/nm**2) and the masses in amu.
    The report is in wavenumbers because that is where the cost of refitting `k`
    is legible: a factor of six in `k` reads as a factor of six, and as a factor
    of 2.5 in a number that can be compared against a spectrum.
    """
    # kJ/mol/nm^2 -> eV/A^2 -> J/m^2
    stiffness = force_constant * units.kJ / units.mol / units.nm**2
    stiffness_si = stiffness * units._e / 1e-20
    reduced_si = mass_a * mass_b / (mass_a + mass_b) * units._amu
    omega = np.sqrt(stiffness_si / reduced_si)  # rad/s
    return float(omega / (2.0 * np.pi * units._c * 100.0))


def diabatic_energy(reaction_set, qforce: QForce, frame: Atoms, positions) -> float:
    """Total energy of `frame`'s connectivity evaluated at `positions`.

    This is the diabat: the bonding topology of one endpoint, scored at the
    geometry of another.  Both of a reaction's endpoints evaluated at its
    transition state are what the coupling fit inverts, and what
    `fit_force_constants` is trying to push above the reference barrier.

    Includes the nonbonded term for the same reason `bonded_energy` does, and
    more sharply: the barrier it is compared against is a reference energy that
    contains everything, so a bonded-only diabat made `reaction_margins` --
    and therefore the entire force-constant objective -- an inequality between
    two different quantities.
    """
    from DynamicTopology.core.topology import Topology

    atoms = frame.copy()
    atoms.calc = None
    atoms.positions = positions
    topology = Topology.from_atoms(atoms)
    topology.set_terms(reaction_set.get_terms(topology))
    energy = qforce(atoms.positions, atoms.pbc, atoms.cell, topology.term_dict)[0]
    return energy + nonbonded_energy(atoms, topology.term_dict)


def install_templates(
    reaction_set,
    templates: list[tuple[str, Atoms, list[Term]]],
    fitted: list[list[Term]],
) -> None:
    """Swap refitted terms into a loaded `ReactionSet`, in memory.

    The outer solve evaluates hundreds of parameter sets and every one of them
    has to be scored through the real `ReactionSet.get_terms` path -- writing
    the files and reloading is both slow and a side effect on the dataset the
    fit has not yet decided to keep.  Only `kwargs` change, so the stored
    `Topology` graphs and hashes stay valid; the remapped-term cache does not,
    and is cleared.
    """
    from DynamicTopology.core.topology import Topology

    for (_, atoms, original), terms in zip(templates, fitted):
        key = Topology.from_terms(original, atoms).hash()
        reaction_set.data["molecules"][key].terms = terms
    reaction_set._term_cache.clear()


def reaction_margins(
    reaction_set, reactions: list[tuple[str, list[Atoms]]]
) -> dict[str, float]:
    """`min(H_reactant, H_product) - E_reference` at each transition state.

    Negative means no real coupling amplitude reproduces that barrier, because
    the EVB ground state of a two-level block lies below both diabats by
    construction.  Reactions whose transition state carries no reference energy
    are skipped -- there is nothing to be feasible against.
    """
    qforce = QForce()
    margins: dict[str, float] = {}
    for name, frames in reactions:
        transition = frames[len(frames) // 2]
        if transition.calc is None:
            continue
        reactant = diabatic_energy(
            reaction_set, qforce, frames[0], transition.positions
        )
        product = diabatic_energy(
            reaction_set, qforce, frames[-1], transition.positions
        )
        margins[name] = min(reactant, product) - transition.get_potential_energy()
    return margins


def fit_force_constants(
    reaction_set,
    templates: list[tuple[str, Atoms, list[Term]]],
    reactions: list[tuple[str, list[Atoms]]],
    margin: float = DEFAULT_MARGIN,
    frequency_weight: float = DEFAULT_FREQUENCY_WEIGHT,
    max_scale: float = DEFAULT_MAX_SCALE,
    mode: str = "both",
    max_shape: float = DEFAULT_MAX_SHAPE,
) -> ForceConstantFit:
    """Refit every template's force constants so the couplings become fittable.

    Outer variables are `log(k-scale)`, one per `BondVariable`; the inner solve
    is `fit_dissociation_energies`, which re-derives each template's `D` scale
    so its atomization energy is reproduced exactly whatever `k` the outer solve
    is trying.  The atomization condition is therefore a constraint the
    objective cannot trade away, not a term competing with the barriers.

    The objective is a hinge, not a least squares:

        J = sum_r max(0, margin - margin_r)**2 + frequency_weight * sum_b x_b**2

    A margin past the target is worth nothing -- the barrier is already
    reproduced exactly by the amplitude, which is free per reaction -- so the
    hinge stops pushing as soon as a channel is feasible and spends nothing more
    of the frequency on it.

    Args:
        reaction_set: a loaded `ReactionSet` for the same dataset.  **Mutated**:
            on return its templates carry the fitted terms, so the caller can
            fit couplings against them without writing files first.
        templates: `(name, atoms, terms)` per molecule, atoms carrying the
            reference atomization energy.
        reactions: `(name, frames)` per reaction, frames in reactant / TS /
            product order.
        margin: how far below the reference barrier a diabat must sit.
        frequency_weight: pull back towards the original force constants.
        max_scale: hard bound on each k-scale, in both directions.
    """
    variables: list[BondVariable] = []
    for name, atoms, terms in templates:
        symbols = {
            (term["kwargs"]["r0"], term["kwargs"]["k"]): tuple(
                atoms[index].symbol for index in term["atoms"].values()
            )
            for term in terms
            if term["type"] == "bond"
        }
        for r0, force_constant in bond_types(terms):
            variables.append(
                BondVariable(
                    template=name,
                    r0=r0,
                    k=force_constant,
                    elements=symbols[(r0, force_constant)],
                )
            )

    # Which slice of the variable vector belongs to which template, so a
    # template's scales can be handed to `scale_force_constants` in the order
    # `bond_types` produced them.
    spans: list[tuple[int, int]] = []
    start = 0
    for _, _, terms in templates:
        stop = start + len(bond_types(terms))
        spans.append((start, stop))
        start = stop

    if mode not in ("shape", "k", "both"):
        raise ValueError(f"mode must be 'shape', 'k' or 'both', got {mode!r}")

    # In "both" the vector is all the `c` values followed by all the log
    # k-scales, so the two halves keep their own natural bounds and a `spans`
    # slice indexes into either half with the same offsets.
    width = len(variables)

    def apply(terms: list[Term], x: np.ndarray, lo: int, hi: int) -> list[Term]:
        if mode == "shape":
            return set_shape_parameters(terms, list(x[lo:hi]))
        if mode == "k":
            return scale_force_constants(terms, list(np.exp(x[lo:hi])))
        return set_shape_parameters(
            scale_force_constants(terms, list(np.exp(x[width + lo : width + hi]))),
            list(x[lo:hi]),
        )

    def refit(x: np.ndarray) -> list[list[Term]]:
        return [
            fit_dissociation_energies(atoms, apply(terms, x, lo, hi))
            for (_, atoms, terms), (lo, hi) in zip(templates, spans)
        ]

    # Per-variable normalizer for the regularizer: the half-width of each
    # variable's box, so `used` below is "fraction of the freedom taken".
    if mode == "shape":
        scale_of = np.full(width, max(max_shape, 1e-12))
    elif mode == "k":
        scale_of = np.full(width, max(float(np.log(max_scale)), 1e-12))
    else:
        scale_of = np.concatenate(
            [
                np.full(width, max(max_shape, 1e-12)),
                np.full(width, max(float(np.log(max_scale)), 1e-12)),
            ]
        )

    def score(x: np.ndarray) -> float:
        try:
            install_templates(reaction_set, templates, refit(x))
        except DissociationFitError:
            # These force constants leave some template with no depth scale that
            # reproduces its atomization energy.  Not a failure of the fit, just
            # a point the outer solve should walk away from.
            return np.inf
        shortfall = [
            max(0.0, margin - value)
            for value in reaction_margins(reaction_set, reactions).values()
        ]
        # The regularizer is on the *fraction of the available box* each
        # variable uses, not on its raw value.  `c` and `log(k-scale)` live on
        # scales that differ by an order of magnitude, and `c`'s own bound moves
        # with `SHAPE_DECAY`, so penalizing raw magnitudes silently reweights
        # the objective whenever any of those change -- which it did: raising
        # the decay from 2 to 4 lifted the bound on `c` from 1.3 to 19.3 and
        # multiplied this penalty by ~200, crushing the fit for reasons that had
        # nothing to do with the physics.
        used = x / scale_of
        return float(
            np.dot(shortfall, shortfall) + frequency_weight * np.dot(used, used)
        )

    # Length of the search vector: `both` carries a `c` and a log k-scale per
    # bond type, the single-parameter modes carry one each.
    n_x = width * (2 if mode == "both" else 1)

    install_templates(reaction_set, templates, refit(np.zeros(n_x)))
    before = reaction_margins(reaction_set, reactions)

    frozen = {
        "shape": max_shape <= 0.0,
        "k": max_scale <= 1.0,
        "both": max_shape <= 0.0 and max_scale <= 1.0,
    }[mode]
    if frozen or not variables:
        # No freedom at all.  Say so by returning the unrefitted point rather
        # than handing a degenerate box to the optimizer; this is also the path
        # the vacuity check takes.
        solution = np.zeros(n_x)
    else:
        # `c` is bounded below by zero and above by `max_shape`; a k-scale is
        # bounded symmetrically in the log, so that halving and doubling are the
        # same distance from the starting point.
        bound = float(np.log(max_scale))
        if mode == "shape":
            box = [(0.0, max_shape)] * width
        elif mode == "k":
            box = [(-bound, bound)] * width
        else:
            box = [(0.0, max_shape)] * width + [(-bound, bound)] * width
        result = minimize(
            score,
            np.zeros(len(box)),
            method="Powell",
            bounds=box,
            options={"maxiter": 200, "xtol": 1e-3, "ftol": 1e-5},
        )
        solution = np.asarray(result.x, dtype=float)
        logger.info(
            "force-constant fit: %d evaluations, objective %.6g",
            result.nfev,
            result.fun,
        )

    fitted = refit(solution)
    install_templates(reaction_set, templates, fitted)
    return ForceConstantFit(
        terms={name: terms for (name, _, _), terms in zip(templates, fitted)},
        variables=variables,
        scales=[
            float(value)
            for value in (solution[:width] if mode != "k" else np.exp(solution))
        ],
        k_scales=(
            [float(value) for value in np.exp(solution[width:])]
            if mode == "both"
            else [1.0] * width
        ),
        mode=mode,
        margins=reaction_margins(reaction_set, reactions),
        margins_before=before,
        depths={
            (name, r0, k): depth
            for (name, _, original), terms in zip(templates, fitted)
            for (r0, k), depth in zip(bond_types(original), bond_depths(terms))
        },
    )

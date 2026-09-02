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

`c` is bounded, and not arbitrarily: past `c = 19.33` the correction beats the
exponential and the dissociation curve turns over, putting a barrier on a
channel that has none and a bound state beyond it.  `DEFAULT_MAX_SHAPE` is that
limit.  See its comment for the derivation.

`fit_force_constants` fits one variable per distinct bond type in any of three
modes -- `shape` (`c` only, no frequency cost), `k` (the old route), or `both`
-- always with `fit_template` re-solved underneath it, so neither the
atomization energy nor the geometry is something the objective can spend.  The
objective is a hinge: once a margin is positive the barrier is reproduced
exactly by the amplitude, which is free per reaction, so overshooting buys
nothing.


Where the minimum is: `r0` against a repulsion that is not zero there
---------------------------------------------------------------------

The paragraph above says `r0` is pinned by the geometry, and while the bonded
terms were the whole molecular potential that was true by construction: q-force
fitted `r0` to the geometry, so the bonded minimum sat on it and nothing had to
be solved.  `ZBL` ended that.  It is a real repulsion at bonding distances --
2.0 eV at the H2 bond length, 5.4 at O-H, 11.6 at O-O, with slopes to match --
and the *total* is what has a minimum, so the Morse has to lean into it.

Nothing made it.  `fit_dissociation_energies` matched each template's energy at
its stored QM geometry, exactly, to 1e-13 -- and nothing anywhere looked at the
gradient there.  The templates came out with the right energies at geometries
they were not at rest in, and relaxed away from them: H2 by 0.105 A, HO2's O-O
by 0.825, every stretching frequency 1.7 to 2.6x experiment.  H2's own minimum
landed *outside* the bond-perception radius, so a relaxed H2 re-perceived as two
free atoms.

`fit_bond_lengths` adds the missing condition -- one equation per bond type, the
total force along it vanishing at the reference geometry -- and `fit_template`
alternates it with the depth solve until both hold.  Both are determinate, so
neither is fitted and neither competes with the barriers.

It is also not always solvable, which is worth stating plainly: a Morse pulls at
most `D*a/2`, and `ZBL` pushes O-O in HO2 apart with 25.0 eV/A against a ceiling
of 8.3 at q-force's own force constants.  The ceiling rises with `D`, `k` and
`c`, so the geometry condition is really a constraint on the search box -- seven
of HCombustion's eight bond types come inside it at `c = 19.3` and the eighth at
`k`-scale 2 -- which is why the origin is no longer a feasible starting point
and `_feasible_start` exists.
"""

import logging
from dataclasses import dataclass, field

import numpy as np
from ase import Atoms, units
from ase.data import atomic_masses, atomic_numbers
from scipy.optimize import brentq, minimize

from DynamicTopology.core.types import Term
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.qforce import SHAPE_DECAY, QForce
from DynamicTopology.forcefield.zbl import ZBL

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
#     cap 1.1               13 / 19   1.05x
#     cap 1.2               13 / 19   1.10x
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
# **No nonbonded term moves this curve, and one of them provably cannot.**  A
# Lennard-Jones with per-state exclusions was added on the hypothesis that Pauli
# repulsion would lift the diabats at the transition states and make the channels
# fittable for free, and the numbers looked emphatic: 14 of 19 feasible with no
# refitting whatever.  All of it was artefact -- a diabat that had broken a bond
# called its two atoms different molecules while they sat at the bond length, and
# the whole-system sum charged them 727 to 1550 eV there.  Excluding those pairs
# properly returned the count to 1 of 19.
#
# The repulsion is now `ZBL`, which takes no topology at all, so it is the *same
# number* on every diabat of a block and cancels exactly out of every margin this
# module computes.  It cannot help here and it cannot hurt here, by construction.
# The overbinding this fit exists to repair is in the Morse form, and the bonded
# parameters are the only thing that can pay for it.
#
# **The knee is not a preference, and the lower caps are not usable.**  A
# decoupled channel is switched off entirely, so a cap that leaves six of them
# off does not merely fit fewer barriers -- it takes the EVB basis apart.  At cap
# 1.0 and at cap 1.5 the standard test geometries drop from three diabatic states
# to two, and nine tests across `test_evb_invariants`, `test_gradients` and
# `test_trajectory_io` fail their own vacuity guards: there is no longer a
# multi-state block to be pivot-invariant about, no channel inside the admission
# ramp, and no topology change along a trajectory.  Only cap 2.0 leaves a
# reactive model behind, which is why the 1.41x is spent.
#
# Re-running this fit from q-force's own force constants -- rather than from the
# twice-stiffened ones the dataset had drifted to -- reproduces the single-pass
# result exactly and changes only O2, from 3259 cm^-1 back to 2305.  That is what
# undoing the double application is worth; the rest of the drift is the fit
# genuinely wanting it.
#
# The one channel no cap reaches is `rxn_06`.  `rxn_08` -- whose stored
# transition state is not a saddle but a *minimum*, 4.87 eV below its own
# reactant; see `tests/test_reference_energies.py` -- was the holdout before the
# shape term existed and is fittable now, at cap 2.0, by 0.011 eV.
#
# Note that this bounds the scale relative to whatever `k` the templates handed
# in already carry, not relative to q-force's original fit, so **running the fit
# twice over its own output compounds the bound**.  One pass is the intended
# use; the shipped parameters are one pass, from q-force's own values.
DEFAULT_MAX_SCALE: float = 2.0


# Weight on the leftover force at each template's reference geometry, in
# 1/(eV/A)**2, i.e. how hard the objective insists that a molecule be at rest
# where its reference energy says it is.
#
# It is a penalty rather than a constraint, and that is a deliberate second
# attempt.  Treating it as a constraint -- `score` returning `inf` wherever
# `fit_bond_lengths` had no solution -- is exactly right on paper and
# catastrophic in practice: at `c = 0` and q-force's own force constants most
# of these bonds cannot cancel `ZBL` at any length, so the infeasible set is
# most of the box, and Powell line-searching across a plateau of infinities
# turned a 40 second fit into one that ran for half an hour without converging.
# A quadratic penalty puts the same pressure on a surface the optimizer can
# actually descend.
#
# 10.0 makes a 0.3 eV/A leftover force -- the worst any HCombustion template
# shows -- cost about as much as a 1 eV margin shortfall, so the geometry is
# worth roughly one channel.  That is the intended exchange rate: a template
# that cannot sit still is a worse defect than a barrier that cannot be fitted,
# but not by so much that the fit will spend every force constant it has to buy
# the last milli-eV per Angstrom.
DEFAULT_GEOMETRY_WEIGHT: float = 10.0


# Wavenumber (cm^-1) above which a stretching mode starts costing the
# objective.  This is the timestep, expressed as a property of the force field:
# velocity Verlet wants ~15 steps per vibrational period, so `dt` femtoseconds
# needs every mode under `33356 / (15 dt)`, which is 4450 at the 0.5 fs the
# production sweep is trying to reach.  4400 is that, rounded down to H2's own
# experimental stretch so the cap is a real number rather than a derived one.
#
# Nothing priced this before, and the result was a surface with an 11735 cm^-1
# mode on it -- a 2.84 fs period, which is what pinned the sweep at 0.05 fs.
# See `_bonded_curvature` for where the stiffness was coming from: not from the
# `k`-scale this fit reports, which is bounded at 1.41x, but from the shape term
# at a displaced `r0`.
DEFAULT_MAX_WAVENUMBER: float = 4400.0

# Weight on `max(0, nu - max_wavenumber)**2`, in 1/cm**-2.
#
# A hinge and not a bound, for the reason `DEFAULT_GEOMETRY_WEIGHT` records:
# a hard constraint here would return `inf` over most of the box at `c = 0`,
# and Powell line-searching a plateau of infinities is what turned a 40 second
# fit into an hour-long one last time.
#
# **The cap costs no channels at all, which was not the expected answer.**
# Swept over HCombustion from the `8f32706` reset in the default `both` mode at
# `--max-k-scale 3`, cap 4400.  The first table is the cap at the weight it was
# first guessed at, 1e-6, and it is the shape of a hinge too soft to bind:
#
#     cap        fastest mode   dt at 15 steps/period   channels   over cap
#     none            11697            0.190 fs           14/19      6 of 8
#     6000             6160            0.361              13/19      2 of 8
#     5000             5781            0.385              13/19      3 of 8
#     4400             5309            0.419              13/19      4 of 8
#     4000             5262            0.423              13/19      6 of 8
#
# It saturates around 5260 cm^-1 and stops responding to the cap, having bought
# 2.2x in timestep for one channel.  That reads like a floor and is not one: a
# scan of the whole `(k-scale, c)` box for water's O-H alone reaches 4337 cm^-1
# -- its repulsion's own curvature, 4342 -- across every `k`-scale at `c <= 5`.
# The fit was not failing to go lower, it was declining to.  Weight, at cap 4400:
#
#     weight     fastest mode   dt at 15 steps/period   channels   over cap
#     1e-6             5309            0.419              13/19      4 of 8
#     1e-5             4636            0.480              14/19      5 of 8
#     1e-4             4546            0.489              13/19      4 of 8
#     1e-3             4401            0.505              14/19      1 of 8
#     1e-2             4399            0.506              14/19      0 of 8
#     3e-2             4399            0.506              14/19      0 of 8
#     1e-1             4501            0.494              14/19      1 of 8
#
# 14 of 19 is what the *uncapped* fit gets.  So the whole 2.7x in timestep --
# 0.190 fs to 0.506 -- is bought for nothing, and the trade this hinge was
# written to manage turns out not to exist on this dataset.  What the cap
# actually does is stop the fit spending `c` in the region where `c` is
# expensive; there was another region, equally good for the margins, that it had
# no reason to prefer until now.
#
# 1e-2 is the middle of a plateau three decades wide, and the two weights that
# reach the cap on every bond type are inside it.  Higher is not better: at 1e-1
# the fit is back to one bond over, because the penalty starts distorting the
# search before it binds any harder.
DEFAULT_CURVATURE_WEIGHT: float = 1e-2

# Stateless, and constructed once: the outer fit calls `bonded_energy`
# thousands of times.
_ACKS2 = ACKS2()
_ZBL = ZBL()


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


# Memo for the nonbonded half of a template's energy and forces, keyed by the
# geometry and the ACKS2 parameters it was computed from.
#
# **Why this is safe, and why it matters.**  Both nonbonded terms are functions
# of the geometry alone -- `ZBL` reads only atomic numbers, and `ACKS2` reads
# the `atom` terms, which no part of this module fits.  The fit moves `D`, `r0`,
# `k` and `c`, every one of them bonded.  So across an entire
# `fit_force_constants` run, at a fixed template geometry, this pair of numbers
# never changes.
#
# It was being recomputed for every one of them: `fit_dissociation_energies`
# runs `brentq` to 1e-12, which is ~40 evaluations of `bonded_energy`, each
# solving the ACKS2 charge equilibration from scratch, and that happens once per
# round per template per objective evaluation.  Caching it is most of the
# difference between a fit that takes a minute and one that takes an hour.
#
# The key includes the ACKS2 parameters rather than trusting the argument
# above, so a caller that *did* fit them would miss the cache rather than read
# a stale number from it.
_NONBONDED_CACHE: dict[tuple, tuple[float, np.ndarray]] = {}


def _nonbonded_key(atoms: Atoms, term_dict: dict) -> tuple:
    acks2 = term_dict.get("atom", {})
    return (
        atoms.positions.tobytes(),
        atoms.numbers.tobytes(),
        atoms.cell.array.tobytes(),
        tuple(atoms.pbc),
        tuple(
            (name, np.asarray(value).tobytes())
            for name, value in sorted(acks2.get("kwargs", {}).items())
        ),
        np.asarray(acks2.get("atoms", ())).tobytes(),
    )


def _nonbonded(atoms: Atoms, term_dict: dict) -> tuple[float, np.ndarray]:
    """ACKS2 + ZBL energy and forces at `atoms`, memoized on the geometry."""
    key = _nonbonded_key(atoms, term_dict)
    hit = _NONBONDED_CACHE.get(key)
    if hit is None:
        energy, forces = _ACKS2(atoms.positions, atoms.pbc, atoms.cell, term_dict)
        zbl_energy, zbl_forces = _ZBL(
            atoms.positions, atoms.numbers, atoms.pbc, atoms.cell
        )
        hit = (float(energy + zbl_energy), forces + zbl_forces)
        _NONBONDED_CACHE[key] = hit
    return hit


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
    """ACKS2 plus whole-system ZBL, at `atoms`' geometry, in eV.

    Both are topology-independent, and in the strong sense: ACKS2 evaluated at
    each of the nineteen transition-state geometries under the reactant's and
    the product's parameter sets gives the same number to every printed digit,
    and `ZBL` does not consult the topology at all -- it reads atomic numbers
    off the `Atoms`.  So both are added once outside the EVB Hamiltonian, which
    is exactly what `System.calculate` does, rather than sitting on the diagonal.

    This is the sum a reference atomization energy has to be matched against.
    Fitting the Morse depths against the bonded part alone left every
    heteronuclear template overbound by exactly its own ACKS2 energy -- water at
    -12.3701 eV against a reference of -9.8735, 25% too deep -- and the error was
    invisible on H2 and O2, which have no charge separation.  `ZBL` closes that
    hole for the homonuclear templates too: it is nonzero on every bonded pair
    (+2.0 eV at the H2 bond length, +11.6 at O2's), so leaving it out here would
    reintroduce the same class of error on exactly the two templates the old
    version of this bug hid behind.
    """
    return _nonbonded(atoms, term_dict)[0]


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


# --------------------------------------------------------------------------
# Bond lengths
# --------------------------------------------------------------------------

# Half-width, in nm, of the window `fit_bond_lengths` searches around the `r0`
# a template arrived with.  It is a tripwire rather than a tuning parameter:
# the shifts this actually needs are 0.005 to 0.014 nm, so a solve that wants
# more than 0.03 has gone somewhere it should not, and the caller should be
# told rather than handed a molecule with a 3 Angstrom bond in it.
MAX_LENGTH_SHIFT: float = 0.03

# How many times `fit_template` alternates the `r0` solve with the `D` solve.
# Each is exact given the other and they couple only through `a = sqrt(k/2D)`,
# so the alternation contracts by about an order of magnitude a round.  Worst
# residual force left on any HCombustion template, in eV/A:
#
#     rounds   1        2        3        4
#              1.5e-3   1.2e-4   1.9e-5   1.9e-6
#
# Iterating the `r0` solve *within* a round instead makes it worse -- 3.0e-3,
# 5.6e-4, 1.1e-4, 2.0e-5 for the same work -- because the extra passes refine
# towards a fixed point of a stale `D`.  So `fit_bond_lengths` is one pass and
# the alternation is here, which is both twice as fast and ten times as
# accurate as looping in both places.
LENGTH_DEPTH_ROUNDS: int = 4


def _morse_stretch_force(r, D: float, r0, k: float, c: float):
    """`-dE/dr` of one Morse bond, in q-force units (kJ/mol/nm).

    Positive is the force pulling the two atoms *apart*, i.e. the sign a
    compressed bond carries.  This mirrors `QForce._bond_morse` exactly,
    including the one-sided shape term; it is written out a second time here
    because the fit needs the derivative of a single bond as a function of `r0`
    with everything else held still, and the force field only ever offers the
    assembled Cartesian forces of a whole system.

    `r` and `r0` broadcast, and the search below depends on it.  This is the
    innermost thing in `fit_force_constants` -- one call per grid point, per
    bond type, per round, per objective evaluation, and Powell's evaluation
    count runs to five figures -- so scanning a bond length has to be one array
    expression rather than four hundred scalar ones.  Left as a scalar loop it
    turned a fit that took seconds into one that took the better part of an
    hour.
    """
    dr = np.asarray(r, dtype=float) - np.asarray(r0, dtype=float)
    al = np.sqrt(k / (2 * D))
    exp_term = np.exp(-al * dr)
    de_dr = 2 * D * (1 - exp_term) * al * exp_term
    s = al * np.maximum(dr, 0.0)
    de_dr = de_dr + (
        D * c * al * s * s * (3.0 - SHAPE_DECAY * s) * np.exp(-SHAPE_DECAY * s)
    )
    return -de_dr


def bond_lengths(terms: list[Term], atoms: Atoms) -> list[list[float]]:
    """Actual bond distance in nm of every bond, grouped by `bond_types` order.

    Two bonds share a type when q-force gave them identical `(r0, k)`, which
    does not make them the same length -- H2O2's two O-H bonds are one type and
    happen to be symmetric, but nothing guarantees that in general.
    """
    types = bond_types(terms)
    lengths: list[list[float]] = [[] for _ in types]
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        lengths[index].append(float(atoms.get_distance(i, j)) / 10.0)
    return lengths


def set_bond_lengths(terms: list[Term], values: list[float]) -> list[Term]:
    """Copy of `terms` with each bond type's `r0` set, in `bond_types` order."""
    order = {pair: index for index, pair in enumerate(bond_types(terms))}
    out: list[Term] = []
    for term in terms:
        if term["type"] != "bond":
            out.append(term)
            continue
        kwargs = dict(term["kwargs"])
        kwargs["r0"] = float(values[order[(kwargs["r0"], kwargs["k"])]])
        out.append({**term, "kwargs": kwargs})
    return out


def stretch_forces(atoms: Atoms, terms: list[Term]) -> list[float]:
    """Net force along the bonds of each type, in eV/A, in `bond_types` order.

    For a bond `(i, j)` this is `0.5 * (F_i - F_j) . u_ij` summed over the
    bonds of the type -- the part of the total force that a change in that
    type's `r0` can move, and nothing else.  `F` is the *whole* force:
    every bonded term, ACKS2 and ZBL, through the same pipeline
    `System.calculate` uses, so no contribution is assumed away.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    topology.set_terms(terms)
    term_dict = topology.term_dict
    forces = QForce(bond_form="morse")(
        atoms.positions, atoms.pbc, atoms.cell, term_dict
    )[1]
    forces = forces + _nonbonded(atoms, term_dict)[1]

    types = bond_types(terms)
    out = [0.0] * len(types)
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        unit = atoms.positions[j] - atoms.positions[i]
        unit = unit / np.linalg.norm(unit)
        out[index] += 0.5 * float(np.dot(forces[j] - forces[i], unit))
    return out


def bond_curvatures(atoms: Atoms, terms: list[Term]) -> list[float]:
    """Second derivative along each bond type, in eV/A**2, in `bond_types` order.

    The *total*, not the Morse's own: measured by displacing the two atoms of
    each bond along their axis and differencing the assembled forces, so ZBL
    and ACKS2 are in it.  Averaged over the bonds of a type, which is what a
    per-type wavenumber can mean at all.

    This exists because `frequency` does not answer the question any more.  It
    converts a bonded force constant to a wavenumber, and while the bonded
    terms were the whole potential that was the frequency.  `ZBL`'s curvature
    at a bond length is not small next to a bond's -- 68.5 eV/A**2 at the O-H
    distance, which on its own is 4431 cm^-1 -- so a report built on the bonded
    `k` alone understates the real stiffness by about a factor of two, and the
    force-constant fit's headline "drift" number was understating it by that
    much.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    topology.set_terms(terms)
    term_dict = topology.term_dict
    qforce = QForce(bond_form="morse")

    def stretch(i: int, j: int, delta: float) -> float:
        """`-dE/dr` of the whole system with bond `(i, j)` stretched by `delta`."""
        moved = atoms.copy()
        unit = atoms.positions[j] - atoms.positions[i]
        unit = unit / np.linalg.norm(unit)
        moved.positions[j] = moved.positions[j] + delta * unit
        forces = qforce(moved.positions, moved.pbc, moved.cell, term_dict)[1]
        forces = forces + _nonbonded(moved, term_dict)[1]
        return float(np.dot(forces[j], unit))

    step = 1e-3
    types = bond_types(terms)
    totals = [0.0] * len(types)
    counts = [0] * len(types)
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        totals[index] += -(stretch(i, j, step) - stretch(i, j, -step)) / (2 * step)
        counts[index] += 1
    return [t / max(n, 1) for t, n in zip(totals, counts)]


def _bonded_curvature(kwargs: dict, r: float) -> float:
    """`d2E/dr2` of one Morse bond at separation `r`, in eV/A**2.

    Analytic, and written out here for the same reason `_morse_stretch_force`
    is: the objective needs the second derivative of a *single* bond as a
    function of that bond's parameters, thousands of times, and the force field
    only offers assembled Cartesian forces of a whole system.  Differentiating
    `QForce._bond_morse` twice,

        d2/dr2 [ D (1 - exp(-a dr))**2 ]  =  2 D a**2 exp(-a dr) (2 exp(-a dr) - 1)
        d2/dr2 [ D c s**3 exp(-b s) ]     =  D c a**2 (6 s - 6 b s**2 + b**2 s**3) exp(-b s)

    with `s = a max(dr, 0)`, so the shape term contributes nothing on the
    compressed branch -- where it is clamped -- and nothing at `dr = 0`, where
    it is cubic.

    **It contributes a great deal anywhere else**, which is the finding this
    function exists to price.  The claim that `c` is `O(dr**3)` at the minimum
    and therefore free of frequency was true while `r0` *was* the minimum.
    `fit_bond_lengths` ended that: `r0` is now pulled 0.04-0.22 A inside the
    reference bond length so the Morse can lean against the repulsion, and at
    that displacement this term is the largest single contribution to the
    stiffness -- 63 eV/A**2 of H2's 115, 509 of O2's 793, 323 of HO's 480.

    `r` in Angstrom; `kwargs` in q-force units, as stored.
    """
    D = kwargs["D"]
    k = kwargs["k"]
    c = kwargs.get("c", 0.0)
    dr = r / 10.0 - kwargs["r0"]  # nm
    al = np.sqrt(k / (2 * D))

    exp_term = np.exp(-al * dr)
    curvature = 2 * D * al * al * exp_term * (2 * exp_term - 1)

    s = al * max(dr, 0.0)
    curvature += (
        D
        * c
        * al
        * al
        * (6 * s - 6 * SHAPE_DECAY * s**2 + SHAPE_DECAY**2 * s**3)
        * np.exp(-SHAPE_DECAY * s)
    )
    # kJ/mol/nm**2 -> eV/A**2
    return float(curvature * units.kJ / units.mol / units.nm**2)


# Nonbonded curvature per bond type, memoized exactly like `_NONBONDED_CACHE`
# and for a stronger reason: it is a function of the geometry and the ACKS2
# parameters alone, and neither moves during a force-constant fit.  One
# measurement per template covers every objective evaluation.
_CURVATURE_CACHE: dict[tuple, list[float]] = {}


def nonbonded_curvatures(atoms: Atoms, terms: list[Term]) -> list[float]:
    """ACKS2 + `ZBL` second derivative along each bond type, in eV/A**2.

    The half of `bond_curvatures` the fit cannot change.  Split out because the
    other half is analytic and this one is not: measuring it costs two assembled
    force calls per bond, and the objective is evaluated four figures of times.

    Averaged over the bonds of a type, in `bond_types` order, which is what a
    per-type wavenumber can mean at all.
    """
    from DynamicTopology.core.topology import Topology

    topology = Topology.from_terms(terms, atoms)
    topology.set_terms(terms)
    term_dict = topology.term_dict

    types = bond_types(terms)
    grouping: list[tuple[int, int]] = []
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        grouping.append(
            (types.index((term["kwargs"]["r0"], term["kwargs"]["k"])), i, j)
        )

    # The `(r0, k)` a type is *named* by moves as the fit runs; which bonds are
    # grouped together does not.  Key on the grouping, so the cache cannot be
    # read across a genuinely different partition.
    key = (_nonbonded_key(atoms, term_dict), tuple(grouping))
    hit = _CURVATURE_CACHE.get(key)
    if hit is not None:
        return hit

    def stretch(i: int, j: int, delta: float) -> float:
        """`-dE_nonbonded/dr` with bond `(i, j)` stretched by `delta`."""
        moved = atoms.copy()
        unit = atoms.positions[j] - atoms.positions[i]
        unit = unit / np.linalg.norm(unit)
        moved.positions[j] = moved.positions[j] + delta * unit
        return float(np.dot(_nonbonded(moved, term_dict)[1][j], unit))

    step = 1e-3
    totals = [0.0] * len(types)
    counts = [0] * len(types)
    for index, i, j in grouping:
        totals[index] += -(stretch(i, j, step) - stretch(i, j, -step)) / (2 * step)
        counts[index] += 1

    hit = [t / max(n, 1) for t, n in zip(totals, counts)]
    _CURVATURE_CACHE[key] = hit
    return hit


def stretch_curvatures(
    atoms: Atoms, terms: list[Term], nonbonded: list[float] | None = None
) -> list[float]:
    """Per-bond-type stretch stiffness in eV/A**2: own Morse plus nonbonded.

    The cheap stand-in for `bond_curvatures` that the objective can afford --
    one analytic expression plus a cached measurement, against two assembled
    force calls per bond -- and deliberately *not* the same quantity.

    `bond_curvatures` differences the whole potential along "displace atom `j`
    along the `ij` axis", so it also picks up every angle, dihedral and *other*
    bond term that touches `j`.  This one takes the bond's own Morse and the
    nonbonded terms and stops.  Measured against it on the shipped parameters:

        template  bond   bond_curvatures   this   difference
        H2        H-H            115.118  115.120     0.001
        O2        O-O            792.475  792.506     0.031
        HO        O-H            477.132  477.172     0.039
        H2O       O-H            109.498  109.498     0.000
        HO2       O-O            104.347  104.335     0.011
        HO2       O-H             94.195   94.195     0.000
        H2O2      O-O            269.782  247.671    22.111
        H2O2      O-H             81.477   81.477     0.000

    One row differs and it is the one row that can: H2O2's O-O is the only bond
    here whose displaced atom carries both another bond and a dihedral.  Of the
    22.1, about 15.9 is the O-H Morse on the moved oxygen and 6.2 is the angle
    and dihedral terms.

    That is tolerable *for the use this has* and for no other.  The cap the
    objective applies binds on X-H stretches -- they are the fast modes, and
    they are exactly the rows that agree to 1e-3 -- while the row that differs
    is a 2967 cm^-1 heavy-atom mode, 8% wrong and nowhere near a cap of 4400.
    Anything that wants the real number should call `bond_curvatures`, which is
    what the report does.
    """
    if nonbonded is None:
        nonbonded = nonbonded_curvatures(atoms, terms)

    types = bond_types(terms)
    totals = [0.0] * len(types)
    counts = [0] * len(types)
    for term in terms:
        if term["type"] != "bond":
            continue
        i, j = list(term["atoms"].values())
        index = types.index((term["kwargs"]["r0"], term["kwargs"]["k"]))
        totals[index] += _bonded_curvature(term["kwargs"], atoms.get_distance(i, j))
        counts[index] += 1
    return [t / max(n, 1) + nb for t, n, nb in zip(totals, counts, nonbonded)]


def fit_bond_lengths(
    atoms: Atoms, terms: list[Term], strict: bool = True
) -> list[Term]:
    """Shift each bond type's `r0` so the *total* potential is flat along it.

    **Why this exists.**  `TestTemplateEnergies` pins each template's energy at
    its reference geometry and `fit_dissociation_energies` solves it to 1e-13.
    Nothing pinned the *gradient* there, and once `ZBL` was added it stopped
    being anywhere near zero: the repulsion is 2.0 eV at the H2 bond length and
    11.6 at O2's, with slopes to match, and `r0` was the only parameter that
    could have leaned against it -- so the minima simply moved outward.  H2 by
    0.105 A, HO2's O-O by 0.825, and every stretching frequency with them.

    The condition is one equation per bond type and it is determinate, not
    fitted: the net force along the type's bonds must vanish at the geometry the
    template's reference energy belongs to.  It is solved rather than optimized
    for the same reason the depth is -- a margin objective given a say in the
    geometry would trade the molecule against the barriers, and the geometry is
    data.

    **What absorbs what.**  Only the bond `r0` moves.  `bondbond` and
    `bondangle` carry reference lengths of their own and keep them: those terms
    are q-force's own cross-coupling fit, and shifting their reference changes
    what they mean rather than where they sit.  Their gradient at the reference
    geometry is part of what the bond `r0` is solved against, which is the
    consistent reading -- the condition is on the total, and the total is what
    the calculator computes.

    Solved by fixed point rather than by a root finder.  Everything except a
    type's own Morse is nearly constant in that type's `r0`, so subtracting the
    Morse's own contribution leaves an external force to cancel, and the `r0`
    that cancels it follows from a one-dimensional bracket on an analytic
    function.  The residual coupling -- through atoms two bonds share, and
    through the angle terms -- is what the iteration is for.
    """
    types = bond_types(terms)
    if not types:
        return list(terms)

    ev_per_qforce = units.kJ / units.mol / units.nm
    original = [r0 for r0, _ in types]
    current = list(original)

    # One pass.  The alternation with the depth solve lives in
    # `fit_template`; doing it in both places converges to the wrong place,
    # because the extra passes refine towards a fixed point of a stale `D`.
    # See `LENGTH_DEPTH_ROUNDS` for the numbers.
    working = set_bond_lengths(terms, current)
    params = {
        key: (t["kwargs"]["D"], t["kwargs"]["k"], t["kwargs"].get("c", 0.0))
        for t in working
        if t["type"] == "bond"
        for key in [(t["kwargs"]["r0"], t["kwargs"]["k"])]
    }
    total = stretch_forces(atoms, working)
    lengths = bond_lengths(working, atoms)

    for index, (r0, k) in enumerate(bond_types(working)):
        D, k_value, c = params[(r0, k)]
        # The type's own Morse contribution to `total[index]`, so that what
        # is left is the part no choice of `r0` can change.
        bond_r = np.asarray(lengths[index], dtype=float)
        own = (
            float(_morse_stretch_force(bond_r, D, r0, k_value, c).sum()) * ev_per_qforce
        )
        external = total[index] - own

        def residual(trial, _r=bond_r, _D=D, _k=k_value, _c=c, _e=external):
            """Total force along this bond type if its `r0` were `trial`.

            Vectorized over `trial`, so the grid below is a single call.
            """
            mine = _morse_stretch_force(
                _r, _D, np.asarray(trial, dtype=float)[..., None], _k, _c
            ).sum(-1)
            return mine * ev_per_qforce + _e

        # Shortening `r0` stretches the bond and so pulls harder -- but
        # only up to a point.  A Morse's pull peaks at `D*a/2` and falls
        # away again past the inflection, so an external push above that
        # ceiling cannot be cancelled at any bond length.  That is not a
        # bracketing failure to be widened around; it is the functional
        # form running out, and it is common enough here to be worth
        # naming: `ZBL` pushes O-O in HO2 apart with 25.0 eV/A and that
        # bond's Morse, as q-force parameterizes it, tops out at 8.3.
        #
        # The ceiling moves with `D`, `k` and `c`, all of which the outer
        # fit and the depth solve are free to raise, so infeasible here
        # means "not at these force constants" rather than "not at all":
        # at `c = 19.3` seven of HCombustion's eight bond types come
        # inside it, and the eighth does at `k`-scale 2.
        # The search window is `MAX_LENGTH_SHIFT` either side of the
        # length the template arrived with, so both branches below are
        # bounded by the same tripwire rather than by whatever bracket
        # happened to be tried.
        low = original[index] - MAX_LENGTH_SHIFT
        high = original[index] + MAX_LENGTH_SHIFT
        grid = np.linspace(low, high, 400)
        values = residual(grid)

        # Not a sign test on the endpoints.  The pull peaks part-way down
        # and falls off again past the inflection, so when a solution
        # exists there are *two* roots and both ends of the window can sit
        # on the same side of zero.  Bracket from the strongest pull
        # outward to `high`, which picks the root nearer the bond length:
        # the far one shortens `r0` past the inflection, where the
        # curvature has the wrong sign and the "minimum" is a maximum.
        peak = int(np.argmin(values))
        if values[peak] <= 0.0 <= values[-1]:
            current[index] = brentq(
                lambda x: float(residual(x)),
                grid[peak],
                high,
                xtol=1e-14,
                rtol=1e-15,
            )
        elif strict:
            raise DissociationFitError(
                f"no bond length within {MAX_LENGTH_SHIFT} nm cancels the "
                f"{external:+.3f} eV/A pushing the {types[index]} bonds "
                f"apart: this Morse pulls at most "
                f"{-(values[peak] - external):.3f} eV/A there"
            )
        else:
            # Best effort, for a baseline or a report: the length that
            # leaves the least force behind.  The caller is told nothing
            # here, which is why `strict` is the default.
            current[index] = float(grid[np.argmin(np.abs(values))])

    return set_bond_lengths(terms, current)


def fit_template(
    atoms: Atoms,
    terms: list[Term],
    strict: bool = True,
    rounds: int = LENGTH_DEPTH_ROUNDS,
) -> list[Term]:
    """Solve one template's `r0` and `D` together at its reference geometry.

    Two determinate conditions, not an optimization: the total energy equals the
    reference atomization energy (`fit_dissociation_energies`), and the total
    force along every bond vanishes (`fit_bond_lengths`).  They couple only
    through `a = sqrt(k / 2D)` -- a deeper well is a narrower one, which moves
    the gradient at a fixed geometry -- so alternating them converges rather
    than needing a joint solve.

    This is the inner solve of `fit_force_constants`: whatever `c` and `k` the
    outer search is trying, the geometry and the atomization energy are restored
    underneath it, so neither is something the margin objective can spend.
    """
    fitted = list(terms)
    for _ in range(rounds):
        fitted = fit_dissociation_energies(
            atoms, fit_bond_lengths(atoms, fitted, strict)
        )
    return fitted


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
    # Keyed on the *input* `(r0, k)` because that is what `BondVariable` carries.
    # The inner solve moves `D` and `r0`; `k` is the outer search's.
    depths: dict[tuple[str, float, float], float] = field(default_factory=dict)
    # Force-constant scale per bond type; all 1.0 unless `mode == "both"`, where
    # `scales` holds `c` and the frequencies move as well.
    k_scales: list[float] = field(default_factory=list)
    # Total second derivative along each bond type at its template's reference
    # geometry, in eV/A**2, aligned with `variables`.  Reported separately from
    # the fitted `k` because they are no longer the same quantity: `ZBL` adds
    # curvature at the bond length that the bonded parameters do not know
    # about, and it is roughly as large as the bond's own.  A report that
    # quotes only the fitted `k` understates the real stiffness by about 2x.
    curvatures: list[float] = field(default_factory=list)


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


def total_wavenumber(curvature: float, mass_a: float, mass_b: float) -> float:
    """Harmonic wavenumber (cm^-1) of a diatomic with this *total* curvature.

    `frequency` above answers the same question about a bonded force constant,
    which used to be the same number and is not any more.  The repulsion adds
    curvature at the bond length -- 68.5 eV/A**2 at O-H, on its own worth
    4431 cm^-1 -- and `fit_bond_lengths` displaces `r0` far enough that the
    shape term adds more again, so the bonded `k` understates the stiffness the
    integrator actually sees by a factor of two to four.

    This is the number the timestep is set by: a stable velocity-Verlet run
    wants ~15 steps per period, so a step of `dt` femtoseconds needs every mode
    under `33356 / (15 dt)` cm^-1 -- 4450 at 0.5 fs.

    `curvature` in eV/A**2, masses in amu.
    """
    return wavenumber(curvature, mass_a * mass_b / (mass_a + mass_b))


def wavenumber(curvature: float, reduced_mass: float) -> float:
    """`total_wavenumber` given the reduced mass directly, in amu.

    The form the objective uses, because `reduced_masses` below precomputes the
    reduction once for the whole search and there is nothing to be gained by
    undoing it and redoing it a few hundred thousand times.
    """
    stiffness = curvature * units._e / 1e-20
    if stiffness <= 0.0:  # a bond that is not at a minimum has no wavenumber
        return 0.0
    return float(
        np.sqrt(stiffness / (reduced_mass * units._amu)) / (2 * np.pi * units._c * 1e2)
    )


def reduced_masses(variables: list["BondVariable"]) -> np.ndarray:
    """Reduced mass in amu of each variable's bond, aligned with `variables`."""
    return np.array(
        [
            (lambda a, b: a * b / (a + b))(
                *(atomic_masses[atomic_numbers[e]] for e in variable.elements)
            )
            for variable in variables
        ]
    )


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
    energy += nonbonded_energy(atoms, topology.term_dict)

    return energy


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

    Only `kwargs` change, so no term list is rebuilt on the way in.  This used
    to route through `lj.with_exclusions`, and installing without it silently
    stripped the exclusion terms off every template: the global Lennard-Jones
    sum then stood uncancelled inside `nonbonded_energy` and every margin read
    feasible by hundreds of eV (rxn_14 by +2131) whatever the force constants
    were.  `ZBL` needs no exclusions, so there is nothing left to strip.
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
    geometry_weight: float = DEFAULT_GEOMETRY_WEIGHT,
    max_wavenumber: float = DEFAULT_MAX_WAVENUMBER,
    curvature_weight: float = DEFAULT_CURVATURE_WEIGHT,
) -> ForceConstantFit:
    """Refit every template's force constants so the couplings become fittable.

    Outer variables are `log(k-scale)`, one per `BondVariable`; the inner solve
    is `fit_dissociation_energies`, which re-derives each template's `D` scale
    so its atomization energy is reproduced exactly whatever `k` the outer solve
    is trying.  The atomization condition is therefore a constraint the
    objective cannot trade away, not a term competing with the barriers.

    The objective is a hinge, not a least squares:

        J = sum_r max(0, margin - margin_r)**2
          + geometry_weight  * sum_b (leftover force at the reference geometry)**2
          + curvature_weight * sum_b max(0, nu_b - max_wavenumber)**2
          + frequency_weight * sum_b (x_b / box_b)**2

    A margin past the target is worth nothing -- the barrier is already
    reproduced exactly by the amplitude, which is free per reaction -- so the
    hinge stops pushing as soon as a channel is feasible and spends nothing more
    on it.  The same shape is used for the wavenumber: a mode under the cap is
    free, and one over it pays.

    The third term is the timestep, and it was missing for as long as this fit
    existed.  Everything here was priced in `k`, whose bound is 1.41x in
    wavenumbers, while the stiffness the integrator sees came mostly from
    elsewhere -- the repulsion's own curvature and the shape term at a displaced
    `r0` -- and reached 11735 cm^-1 with the `k`-scale reporting 1.41x.  See
    `DEFAULT_MAX_WAVENUMBER` and `_bonded_curvature`.

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
        max_wavenumber: stretching modes above this cost the objective.
        curvature_weight: how much they cost.  Zero restores the old objective,
            which is the vacuity check for the cap.
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

    # Both constant across the entire search: the elements do not change, and
    # the nonbonded curvature is a function of the fixed template geometry.
    # Measured once here rather than per objective evaluation, which is the
    # difference between a hinge that costs nothing and one that doubles the
    # cost of the fit.
    masses = reduced_masses(variables)
    frozen_nonbonded = [
        nonbonded_curvatures(atoms, terms) for _, atoms, terms in templates
    ]

    def refit(x: np.ndarray, rounds: int = LENGTH_DEPTH_ROUNDS) -> list[list[Term]]:
        """Every template's geometry and depth re-solved at this point.

        Always best-effort on the geometry: where no bond length cancels the
        repulsion, `fit_bond_lengths` leaves the closest it can and `score`
        charges for what is left.  See `DEFAULT_GEOMETRY_WEIGHT` for why this
        is a penalty and not a constraint.
        """
        return [
            fit_template(atoms, apply(terms, x, lo, hi), strict=False, rounds=rounds)
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
            # Two rounds, not four.  The inner solve is most of this fit's
            # runtime and the objective only reads *margins* and a residual
            # force, neither of which a 1e-4 eV/A refinement moves.  What gets
            # written out is refit at full accuracy below.
            fitted = refit(x, rounds=2)
            install_templates(reaction_set, templates, fitted)
        except DissociationFitError:
            # These force constants leave some template with no depth scale that
            # reproduces its atomization energy.  Not a failure of the fit, just
            # a point the outer solve should walk away from.
            return np.inf
        leftover = np.array(
            [
                value
                for (_, atoms, _), terms in zip(templates, fitted)
                for value in stretch_forces(atoms, terms)
            ]
        )
        shortfall = [
            max(0.0, margin - value)
            for value in reaction_margins(reaction_set, reactions).values()
        ]
        # Stretching wavenumbers, in `variables` order.  `stretch_curvatures`
        # rather than `bond_curvatures`: same number on every mode the cap can
        # bind on, at a hundredth of the cost.  See its docstring for the one
        # row where they differ and why it does not matter here.
        overshoot = np.array(
            [
                max(0.0, wavenumber(curvature, mass) - max_wavenumber)
                for mass, curvature in zip(
                    masses,
                    (
                        value
                        for (_, atoms, _), terms, nonbonded in zip(
                            templates, fitted, frozen_nonbonded
                        )
                        for value in stretch_curvatures(atoms, terms, nonbonded)
                    ),
                )
            ]
        )
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
            np.dot(shortfall, shortfall)
            + geometry_weight * np.dot(leftover, leftover)
            + curvature_weight * np.dot(overshoot, overshoot)
            + frequency_weight * np.dot(used, used)
        )

    # Length of the search vector: `both` carries a `c` and a log k-scale per
    # bond type, the single-parameter modes carry one each.
    n_x = width * (2 if mode == "both" else 1)

    # The baseline is taken with the geometry solve in best-effort mode.  At
    # `c = 0` and `k`-scale 1 several bond types cannot cancel `ZBL` at any
    # length -- see `fit_bond_lengths` -- so the strict solve has no answer
    # there, and refusing to report a "before" column because the *starting*
    # point is infeasible would be reporting nothing at all.
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
        # **The search converges on its own; there is nothing here to cap.**
        # Worth writing down because it was twice diagnosed wrongly.  Swept over
        # `mode="shape"` from plain Morse, which is the slowest configuration:
        #
        #     budget         time    nfev   objective   channels
        #     maxfev=200     8.0s     200   135.18145   12 -> 14
        #     maxfev=400    16.0s     400     6.05919   12 -> 15
        #     maxfev=800    32.3s     800     5.72266   12 -> 14
        #     maxfev=2000   49.5s    1219     5.71545   12 -> 15
        #
        # The last row is the answer: `nfev` came in *under* its cap, so Powell
        # stopped on `xtol`/`ftol` at 1219 evaluations and about fifty seconds.
        # Everything past 400 is noise -- the objective moves 6.059 to 5.715 and
        # the channel count wanders between 14 and 15 while improving, because
        # the hinge, the regularizer and the geometry penalty will trade a
        # marginal channel against each other.
        #
        # `maxiter` was the wrong knob regardless: it counts Powell *sweeps*
        # while `maxfev` counts evaluations, and with `maxfev` defaulting to
        # `N * 1000` the sweep cap is never what binds.  Capping it at 30 and
        # then at 10 changed nothing, which is what sent this looking in the
        # wrong place to begin with.
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
    curvatures = [
        value
        for (_, atoms, _), terms in zip(templates, fitted)
        for value in bond_curvatures(atoms, terms)
    ]
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
        curvatures=curvatures,
        margins=reaction_margins(reaction_set, reactions),
        margins_before=before,
        depths={
            (name, r0, k): depth
            for (name, _, original), terms in zip(templates, fitted)
            for (r0, k), depth in zip(bond_types(original), bond_depths(terms))
        },
    )

"""Geometries that put a multi-state EVB basis on the table.

Several tests need a configuration where the basis is genuinely more than one
state -- pivot invariance is vacuous on a single state, the switch gradient is
unreachable unless a channel is mid-ramp, and a continuity test has nothing to
cross unless the basis size changes.  They used to build those by placing two
equilibrium templates a few Angstrom apart.

That stopped working once the couplings were fitted, and for the reason the
coupling fit exists: `fit.coupling.fit_width` pins the width so that `|V| <= eps`
at the reactant and product *minima*, because that is where the diabatic picture
is already correct and a coupling would be double-counting.  Two molecules at
their own equilibrium geometries therefore cannot mix, however close they are
placed -- the H2O + HO pair below stays at one state from 8 A down to contact,
and at 5000 K.  Before the fit they mixed only because fourteen channels carried
a 10 eV placeholder amplitude that never switched off.

So a multi-state basis now lives where it should: near a transition state.  The
dataset already stores one per reaction, and interpolating from it towards the
reactant sweeps the coupling from full strength down through the admission ramp
to nothing, which is every regime these tests need from a single deterministic
handle.
"""

from pathlib import Path

import numpy as np
from ase import Atoms, io


HCOMBUSTION = Path("datasets/HCombustion/HCombustion.json").resolve()

# Chosen over the other channels because it is the only one on the current
# surface whose basis reaches three states *and* has a visible admission ramp,
# which is what every multi-state test here needs.  Measured over the interval
# these constants come from:
#
#     t        E (eV)   states  min_switch     dE
#      0.100  -12.3162     2      1.0000     +0.0952
#      0.110  -12.2221     2      1.0000     +0.0941
#      0.120  -12.1327     3      0.0379     +0.0894
#      0.130  -12.0513     3      0.5810     +0.0814
#      0.140  -11.9808     3      1.0000     +0.0704
#      0.150  -11.9236     3      1.0000     +0.0572
#      0.160  -11.8814     3      1.0000     +0.0422
#
# The energy walks straight through the basis change without a step, which is
# the property `test_energy_conservation` and `TestCutoffContinuity` are for,
# and the three states persist past t = 0.25 so nothing here sits on a knife
# edge.
#
# **This was rxn_10, and before that rxn_16.**  It moves whenever the force
# field is refit, because which channels reach three states is a property of
# the fitted coupling amplitudes rather than of the geometry.  The wavenumber
# cap added to `fit.dissociation` took rxn_10 from three states with a clean
# ramp over t = -0.06 to 0.00 down to two states everywhere on its path -- the
# *channel* is still fitted and still feasible, its amplitude is simply no
# longer large enough to admit a third diabat at this geometry.  Surveying all
# nineteen channels either side of that refit, the only differences were
# rxn_10 (3 -> 2), rxn_05 (2 -> 1) and rxn_11 (1 -> 2); rxn_13 held at three.
#
# rxn_16, the original choice, moved for a different and worse reason: scanning
# its path crosses a *bond-perception* cutoff.  The O-O distance passes 1.716 A
# at t = 0.0865, the edge disappears, its Morse term goes with it, and the
# energy steps by 6.5 eV with the repulsion identical either side (16.856
# against 16.872 eV).  That defect is still there -- rxn_02, rxn_03, rxn_16 and
# rxn_17 all show 6.5 to 7.8 eV steps when scanned -- and it has nothing to do
# with any of the terms these tests measure, so the geometry moved rather than
# the bar.
REACTION = "rxn_13"

# Fraction of the way from the transition state towards the reactant.  Three
# states with every channel at full strength, and far enough past the ramp that
# a small change in the surface does not put it back inside one.
REACTION_PATH_TS = 0.16
# Squarely inside the admission ramp: three states, `min_switch` = 0.59, so the
# switch's own gradient is reachable and a pivot-invariance test has something
# to be invariant about.
#
# A stale value here does not fail loudly -- it makes every test that depends on
# a channel being mid-ramp pass vacuously -- which is why each of them asserts
# `min_switch` is strictly inside (0, 1) rather than trusting this number.
#
# **The ramp is about two hundredths of `t` wide, so this constant is fragile by
# nature.**  Rescanned twice now: once when `ZBL` acquired its taper, and again
# when `forcefield/lj.py` was switched back on and both datasets were refit
# against it.  The whole window on the current surface:
#
#     t       states   min_switch
#     0.0950       2     1.000000     <- the third state is not yet admitted
#     0.0975       3     0.000013
#     0.1000       3     0.007985
#     0.1025       3     0.051354
#     0.1050       3     0.153067
#     0.1075       3     0.320305
#     0.1100       3     0.537102   <- chosen, nearest a half-open switch
#     0.1125       3     0.760882
#     0.1150       3     0.930226
#     0.1175       3     0.997439
#     0.1200       3     1.000000
#
# The window is the same width as before (0.02 in `t`) and has moved bodily
# inward by about 0.027, which is what a refit does to it: the admission gate is
# an energy test, so it moves wherever the diabats move.
#
# Scanning `t` and taking the point nearest `min_switch` = 0.5 is the maintenance
# that follows a refit.  Scan at 0.0025 or finer: a 0.01 grid lands at most one
# point inside this window and can miss it altogether, at which point every test
# downstream reports that no geometry exercises the switch.
REACTION_PATH_RAMP = 0.11


def reaction_path(name: str, t: float, cell: float) -> Atoms:
    """Geometry `t` of the way from `name`'s transition state to its reactant.

    `t = 0` is the transition state and `t = 1` the reactant minimum; the scale
    is linear in Cartesian coordinates, so it is a handle on the geometry rather
    than a reaction coordinate in any physical sense.  Centred in a cubic box of
    edge `cell` with periodic boundaries off.
    """
    frames = io.read(HCOMBUSTION.parent / f"reactions/{name}.xyz", index=":")
    transition = frames[len(frames) // 2]

    atoms = transition.copy()
    atoms.calc = None  # the reference energy belongs to the frame, not to this
    # `molify.ase2networkx` prefers an explicit `info["connectivity"]` over
    # perceiving bonds from geometry, and the dataset's .xyz frames carry one.
    # Dropping it makes this geometry behave the way one from a trajectory does.
    # It also has to go for `TestPermutationInvariance` to mean anything:
    # `atoms[permutation]` reindexes the arrays but leaves `info` alone, so a
    # retained connectivity list would silently describe the wrong atoms.
    atoms.info.pop("connectivity", None)
    atoms.positions = transition.positions + t * (
        frames[0].positions - transition.positions
    )
    atoms.set_cell(np.eye(3) * cell)
    atoms.set_pbc(False)
    atoms.positions += cell / 2 - atoms.positions.mean(0)
    return atoms


def with_spectator(atoms: Atoms, template: Atoms, gap: float) -> Atoms:
    """`atoms` plus a copy of `template` placed `gap` Angstrom clear along x.

    The spectator shares no reaction with anything in `atoms`; the point is
    whether it changes the answer when it drifts across the bimolecular cutoff,
    which it must not.  `gap` is measured between the two fragments' x extents,
    so it is close to -- but not exactly -- the minimum interatomic separation;
    callers that care about that number should measure it.
    """
    spectator = template.copy()
    spectator.calc = None
    spectator.positions -= spectator.positions.min(0)
    spectator.positions[:, 1:] += atoms.positions.mean(0)[1:]
    spectator.positions[:, 0] += atoms.positions[:, 0].max() + gap

    combined = atoms.copy()
    combined += spectator
    combined.set_cell(atoms.cell)
    combined.set_pbc(False)
    return combined


# A second channel, used where a test needs the *carried topology* to change
# repeatedly rather than just to hold a multi-state basis.  A symmetric transfer
# is no good here: its two sides are degenerate, so a trajectory started on one
# tends to sit there, so the anti-vacuity guard downstream -- correctly -- says
# the test would be asserting nothing.  `rxn_14` recrosses four to five times in
# 300 steps at 1000 K on every seed tried *and* changes basis size on every
# seed, which is what that guard needs.  Surveyed across all nineteen channels
# at starts 0.1, 0.2 and 0.3, as `distinct basis sizes / topology switches` per
# seed, only these clear both:
#
#     rxn_14 at 0.2    2/4  2/5  2/5      <- chosen
#     rxn_12 at 0.3    3/1  3/3  2/3
#     rxn_17 at 0.1    2/1  3/1  3/1
#     rxn_13 at 0.2    2/1  2/1  2/1
#     rxn_19 at 0.2    2/1  2/1  2/1
#
# rxn_11 and rxn_16 vary their basis size on every seed but never switch
# topology, so they satisfy the guard while testing nothing -- which is the
# exact failure this survey exists to avoid.
#
# **This has now moved three times, and always for the same structural reason.**
# Was `rxn_02` at 0.1, which stopped switching when the couplings were refitted
# against `ZBL`: rxn_02's amplitude went from -72 eV -- an artefact of a
# Lennard-Jones wall standing between two atoms its transition state has 0.916 A
# apart -- to -2.28 eV, a real coupling and a much smaller one.  Then `rxn_04`
# at 0.2, which stopped switching when the wavenumber cap was added to
# `fit.dissociation`.  Which channels recross is a property of the fitted
# amplitudes, so any refit can move it, and re-running the survey above is the
# maintenance that follows a refit.
#
# **Then `rxn_14` at 0.2, which stopped when `ZBL` acquired its taper.**  That
# refit decoupled 7 of the 19 channels outright (their reference barriers now lie
# above a diabat, so no real amplitude reproduces them), and it changed the
# character of the ones that survive: re-running the survey, *no* channel on that
# surface recrossed at all.  Every candidate switched exactly once and stayed,
# and the count did not move with a longer run or a hotter one --
#
#     rxn_13 at 0.2, seeds 0/1/2, switches in  300 steps at 1000 K:  1  1  1
#                                              1200 steps at 1000 K:  1  1  1
#                                              1200 steps at 2000 K:  1  1  1
#
# -- so it was the barrier back, not the sampling.  `rxn_12` at 0.2 was chosen
# from that survey (3 switches pooled over three seeds / 3 distinct basis sizes).
#
# **And rescanned again when `forcefield/lj.py` was switched back on**, which put
# recrossing back.  Same protocol -- 13 fittable channels, starts 0.1/0.2/0.3,
# 300 steps at 1000 K, scored `switches pooled over three seeds / distinct basis
# sizes` -- and the rows that clear both floors:
#
#     rxn_12 at 0.1    7 / 3     <- chosen; per seed 3, 3, 1
#     rxn_10 at 0.2    4 / 3        per seed 0, 3, 1 -- one seed never switches
#     rxn_12 at 0.3    3 / 3
#     rxn_14 at 0.2    3 / 3
#     rxn_12 at 0.2    3 / 2        the previous choice
#     rxn_04 at 0.2    3 / 2
#
# Nine of the thirteen channels still never switch at any start.  What changed is
# that the best row now recrosses on two seeds of three rather than on none, so
# the constant moved along the path rather than to another channel.
#
# `rxn_04`, `rxn_14` and `rxn_18` raise `ValueError` at t = 0.1: the geometry
# perceives as a bridged species that is neither reactant nor product, which
# `ReactionSet.get_terms` correctly refuses.  That is a property of scanning a
# path, not of the surface, and those rows are simply unavailable.
SWITCHING_REACTION = "rxn_12"

# Where along `SWITCHING_REACTION`'s path to start a trajectory that has to
# cross a topology change.  Not the transition state itself: sitting exactly on
# it, the trajectory commits one way and stays.  A tenth of the way towards the
# reactant it switches on every seed tried, so the anti-vacuity guard downstream
# is not riding on a lucky random number, and on two seeds of three it recrosses;
# see the survey above.  Moved from 0.2 when the 12-6 came back and 0.1 became
# the better row on both scores.
SWITCHING_PATH_START = 0.1

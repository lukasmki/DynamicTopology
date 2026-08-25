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

# HO + H2O -> H2O + HO, a hydrogen transfer.  Chosen over the other channels
# because its basis holds three states over a wide interval either side of the
# transition state, so a test needing a multi-state block does not sit on a
# knife edge, and because the gate closes cleanly on the reactant side, which
# is where `REACTION_PATH_RAMP` comes from.
REACTION = "rxn_16"

# Fraction of the way from the transition state towards the reactant.  Zero is
# the transition state itself: three states, every channel at full strength.
REACTION_PATH_TS = 0.0
# Far enough along that one channel is partway through the admission ramp
# (min_switch ~ 0.28); the gate closes on it entirely by 0.275.
REACTION_PATH_RAMP = 0.225


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
# repeatedly rather than just to hold a multi-state basis.  `rxn_16` is a
# symmetric hydrogen transfer: its basis is wide (three states) but the two
# sides are degenerate, so a trajectory started there tends to sit in one
# topology.  `rxn_02` is not symmetric and switches four times in 300 steps at
# 1000 K, which is what an anti-vacuity guard on "did it actually cross one"
# needs.
SWITCHING_REACTION = "rxn_02"

# Where along `SWITCHING_REACTION`'s path to start a trajectory that has to
# cross a topology change.  Not the transition state itself: sitting exactly on
# it, the trajectory commits one way and stays, giving a single switch on four
# of five seeds.  A tenth of the way towards the reactant it recrosses -- at
# least twice on every seed tried, so the anti-vacuity guard downstream is not
# riding on a lucky random number.
SWITCHING_PATH_START = 0.1

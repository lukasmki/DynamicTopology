"""Regression guards for the force-call optimizations.

Every optimization here was justified by an argument that it changes no
physics, so what has to be pinned is exactly that.  The energies below are the
values the calculator produced *before* any of the optimization work, recorded
to full precision:

  - Screening reaction candidates on the reacting fragment rather than the whole
    block (`EVBBasis._channel_weight`) is exact because the molecules a reaction
    leaves alone contribute equally to both diabats and cancel from the gap.
  - Deriving the term cache key before hashing (`ReactionSet.get_terms_topology`)
    is exact because the Weisfeiler-Lehman hash is a function of the signature
    that key already contains.
  - Memoizing reaction lookups and template mappings
    (`ReactionSet._reaction_channels`) is exact because both depend only on a
    molecule's graph, never on the geometry.

None of those reorder a floating-point sum, so the assertions are for bit
equality rather than a tolerance.  A tolerance here would hide precisely the
failure mode that matters: a candidate wrongly screened out changes the energy
by electron-volts, not ulps.

One rejected approach is worth recording, because it looks correct and is not.
Pruning reaction *edges* on `|V| <= eps` before forming blocks seemed exact --
the admission stabilization is bounded above by `|V|`, so such an edge can never
be admitted -- and it was a 9x speedup.  But block membership is not an edge
property.  Two molecules joined only by a weak channel at the current topology
can be joined by a strong one *after* a different reaction fires: H2O and HO are
linked at the seed only by `HO + H2O -> H2O2 + H` (|V| ~ 1e-30), but once H2O
dissociates, `HO + H -> H2O` applies between them and is strong.  Pruning them
apart lost that channel and moved the energy by 3.41 eV.  `test_energy_matches_
reference` is what caught it.
"""

import numpy as np
import pytest
from ase import io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System


RSET_PATH = "datasets/HCombustion/HCombustion.json"

# (energy, energy_bonded, energy_nonbonded) in eV, and the number of blocks.
#
# Regenerated deliberately when the bond parameters were refitted against a
# corrected energy zero (`fit.dissociation` -- the depths had been solved so that
# `E_QForce = E_atomization` while the calculator reports `E_QForce + E_ACKS2`,
# double-counting the nonbonded term) and gained the Morse shape parameter `c`.
# The bonded parameters themselves moved, so the surface is *supposed* to move
# and no optimization argument applies -- unlike every other reason this
# fingerprint could shift, which is what it exists to catch.
#
# What did *not* move is worth more than what did.  The block counts are
# identical (87, 28), the nonbonded energies are identical to the last digit,
# and the total moved by only 0.011 eV across 200 atoms -- because the refit
# preserves every template's energy at its own geometry *exactly*, and this box
# is near equilibrium.  A refit that had actually broken something would not
# land within 2e-5 of the old number by accident.
#
# Regenerated a second time when Lennard-Jones repulsion and dispersion were
# added, twice more as that term was reshaped, and once more when it was taken
# out of the force field again and replaced by `ZBL` (`forcefield/zbl.py`).  The
# LJ era's numbers are in git; what they recorded, and what it is worth keeping,
# is that `energy_bonded` sat at ~-1.1e5 eV with the LJ sum at ~+1.1e5 -- the
# whole-system 12-6 counting every intramolecular pair at bond length, where it
# is enormous, and the `exclusion` terms inside the bonded total subtracting
# exactly those pairs.  Two halves of one cancelling pair, reported under
# different names, with only their sum physical.
#
# That cancellation is gone: `energy_bonded` is now -1227 eV and means what it
# says, because `ZBL` has no exclusions and nothing has to be cancelled.  The
# repulsion is +723 eV (d30) and +766 eV (d250) across 200 atoms, dominated by
# bonded pairs, and it is absorbed into the fitted Morse depths -- every
# template still reproduces its own reference energy to 1e-13.  See
# `test_reference_energies.py`.
#
# **What did not move is the point.**  The block counts (87, 28) and the ACKS2
# energies are identical to the last digit, as they must be: neither the
# repulsion nor the refit touches the block partition or the electrostatics.
# The totals moved by +3.78 eV (d30) and +42.7 eV (d250), and the dense box
# moving ten times as much is the term doing its job -- d250 is the box that
# used to interpenetrate.
#
# Regenerated once more when `fit.dissociation.fit_bond_lengths` was added --
# the condition that each template be at *rest* at its reference geometry, not
# merely at the right energy there.  That moved `energy_bonded` by -0.10 eV
# (d30) and -0.19 eV (d250), and moved the ACKS2 energies and the block counts
# by nothing whatever, which is the shape a bonded-parameter change is supposed
# to have.
#
# And again for `DEFAULT_MAX_WAVENUMBER`, the cap that made the fit pay for the
# curvature it was spending.  That refit is the largest change to the force
# field in this file's history *by the measure that mattered* -- the fastest
# stretching mode went from 11697 cm^-1 to 4399, and the timestep it admits from
# 0.19 fs to 0.51 -- and it moves the numbers here by one milli-electronvolt:
#
#     energy_bonded   -0.0010 eV (d30)   -0.0011 eV (d250)
#     ACKS2                 identical          identical
#     ZBL                   identical          identical
#     blocks                identical          identical
#
# That is not a coincidence and it is worth stating, because it is the whole
# reason the frequencies were free to be wrong for so long: the atomization
# condition pins each template's *energy* exactly, so a refit can move every
# curvature on the surface by a factor of three and leave a box's energy where
# it was.  Nothing in this file, or in `test_reference_energies.py`, could have
# noticed.  `TestTemplateFrequencies` is what notices now.
#
# And again when `ZBL` acquired its taper (`forcefield/zbl.py`) and both datasets
# were refit against it.  This is the largest move in the file's history, and the
# two boxes move by wildly different amounts -- which is the whole point of
# having two:
#
#     d30  (dilute)   -504.277 -> -507.901 eV      -3.62 eV
#     d250 (dense)    -460.463 -> -507.058 eV     -46.60 eV
#     ACKS2                 identical           identical
#     blocks                identical           identical
#
# Before the taper these two boxes differed by 43.8 eV; after it they differ by
# 0.84 eV.  Nearly the whole density dependence of this force field's energy was
# the untapered ZBL tail summed over intermolecular pairs -- 200 atoms squeezed
# into the d250 box paid 46.6 eV for contacts that a screened *nuclear*
# potential has no business charging for at 2-3 A.  It is the same term, on the
# same geometries, that made a 64-water box read +251 kbar.
#
# `energy_bonded` moves by +44.7 eV on both boxes and by nearly the same amount
# on each, because it is dominated by intramolecular pairs and the refit
# re-absorbed a slightly smaller ZBL into the Morse depths.  ACKS2 is untouched
# to the last digit, as it must be, and so is the block partition.
# And again when `forcefield/lj.py` was switched back on -- the 12-6, with
# `lj.switch` holding it off at bond lengths so that it needs no exclusions --
# and both datasets were refit against it.  **This is the entry where the two
# boxes are supposed to diverge, and they do:**
#
#     d30  (dilute)   -507.901 -> -508.150 eV      -0.25 eV
#     d250 (dense)    -507.058 -> -503.214 eV      +3.84 eV
#     ACKS2                 identical           identical
#     blocks                identical           identical
#
# The dense box goes *up* and the dilute one barely moves, which is what a
# restored intermolecular wall looks like and is the opposite of every previous
# entry here.  Read as a density dependence: the two boxes differed by 43.8 eV
# before the ZBL taper, 0.84 eV after it -- with no wall left, energy had almost
# stopped depending on density at all -- and **4.94 eV now**, of which the 12-6
# supplies 4.09 (17.95 eV on d30 against 22.05 on d250).  That number is the
# reason a 64-water box stopped collapsing; see
# `production/density-300K/README.md`.
#
# `energy_bonded` moves by -18.20 eV on *both* boxes, to five figures the same
# on each, which is the signature of a change absorbed intramolecularly: the
# refit took the new term's ~0.03-0.35 eV per bond back into the Morse depths,
# and every template still reproduces its own reference energy to 1e-13.  ACKS2
# is identical to the last digit and so is the block partition, as they must be
# -- the 12-6 touches neither the charge equilibration nor which molecules are
# near which.
#
# And again when `ACKS2` learned to sum its charge kernel over periodic images
# (`forcefield/ewald.py`).  **This is the first entry in this file where ACKS2
# is the thing that moves, and it is the only thing that moves:**
#
#     d30  (dilute, 45.49 A)   -508.14962 -> -508.14968 eV     -6.17e-05 eV
#     d250 (dense,  22.44 A)   -503.21391 -> -503.22271 eV     -8.80e-03 eV
#     energy_bonded                  identical          identical
#     blocks                         identical          identical
#
# Both boxes are `pbc = T T T`, so before this they were being handed an
# electrostatic energy that stopped at the nearest image -- the one term of the
# three nonbonded ones for which that is not merely an approximation but a
# different sum, since a charge kernel is not short ranged the way a tapered ZBL
# or a switched 12-6 is.  The correction is downward on both boxes and 143 times
# larger on the dense one.
#
# **No refit, and that is the point of the `energy_bonded` row.**  Unlike the
# ZBL taper and the 12-6 switch above -- both of which invalidated every
# `.jsonl` in both datasets, because `fit/dissociation.py` solves against
# `E_QForce + E_nonbonded` -- every dataset template carries `pbc="F F F"`, so
# the fit sees the open-boundary kernel, which is unchanged to the last bit.
# `energy_bonded` being *bit-identical* on both boxes is the evidence: a term
# that had reached the fit could not have left it so.
REFERENCE = {
    "tests/data/mix-n100-d30.xyz": (
        -508.1496794273134,
        -1201.0110654266937,
        0.041254295990371045,
        87,
    ),
    "tests/data/mix-n100-d250.xyz": (
        -503.22270719522555,
        -1201.104399596257,
        0.8476084009425316,
        28,
    ),
}


@pytest.fixture(scope="module")
def reaction_set():
    return ReactionSet(RSET_PATH)


@pytest.mark.parametrize("path", sorted(REFERENCE))
def test_energy_matches_reference(reaction_set, path):
    """The optimized force call reproduces the original numbers exactly."""
    atoms = io.read(path)
    results = System(atoms, Topology.from_atoms(atoms), reaction_set).calculate()

    energy, bonded, nonbonded, nblocks = REFERENCE[path]
    assert results["energy"] == energy, (
        f"total energy moved by {results['energy'] - energy:+.3e} eV; an "
        "optimization changed the physics"
    )
    assert results["energy_bonded"] == bonded
    assert results["energy_nonbonded"] == nonbonded
    assert len(results["blocks"]) == nblocks, (
        "the block partition changed, which changes which states can couple"
    )


@pytest.mark.parametrize("path", sorted(REFERENCE))
def test_forces_are_finite_and_nonzero(reaction_set, path):
    """Guards against a 'fast' path that quietly stops producing forces."""
    atoms = io.read(path)
    forces = System(atoms, Topology.from_atoms(atoms), reaction_set).calculate()[
        "forces"
    ]
    assert np.isfinite(forces).all()
    assert np.abs(forces).max() > 0.0


def test_repeated_calls_are_stable(reaction_set):
    """Caches are keyed per geometry, so re-evaluating must not drift.

    `EVBBasis` clears its caches at the top of every `build`, and the
    `ReactionSet` caches are geometry-independent by construction.  If either
    assumption breaks, the second call at an unchanged geometry disagrees with
    the first.
    """
    atoms = io.read("tests/data/mix-n100-d250.xyz")
    system = System(atoms, Topology.from_atoms(atoms), reaction_set)
    first = system.calculate()
    second = system.calculate()
    assert second["energy"] == first["energy"]
    assert np.array_equal(second["forces"], first["forces"])


# Box and spectator placement for the admitting-branch case below.  The gap is
# inside `System.bimol_cutoff` of 4.0 A on purpose, so the spectator is in the
# reaction network rather than merely nearby; measured, it makes no difference
# to the admitted count whether it sits at 2, 3 or 5 A, which is the screen
# behaving as advertised.
SPECTATOR_CELL = 24.0
SPECTATOR_GAP = 3.0


def _compare_screen_against_full(reaction_set, atoms, limit=400):
    """Return (n_admitted, n_rejected) after checking each against full evaluation."""
    from DynamicTopology.basis import state_key

    topology = Topology.from_atoms(atoms)
    system = System(atoms, topology, reaction_set)
    basis = system.basis
    basis.build(atoms, topology, system.bimol_cutoff)  # warm the caches

    network = reaction_set.get_network(topology, system.bimol_cutoff)
    admitted = rejected = 0
    for _, molecules in network.reaction_blocks():
        parent = Topology.from_molecules(molecules, remap=False)
        parent.attach_atoms(atoms)

        for reaction, mapping in basis._reactions(parent, system.bimol_cutoff):
            broken, formed = reaction.edge_changes(mapping)
            if not broken and not formed:
                continue
            child = reaction.apply(parent, mapping, share_atoms=True)
            if state_key(child) == state_key(parent):
                continue
            parent_energy, _, _ = basis._energy(parent, atoms)
            child_energy, _, _ = basis._energy(child, atoms)
            coupling, coupling_forces, coupling_virial = basis._coupling(
                atoms, reaction, mapping
            )

            screened, _, _ = basis._channel_weight(
                parent,
                mapping,
                (broken, formed),
                coupling,
                coupling_forces,
                coupling_virial,
                atoms,
            )
            exact = basis._switch(parent_energy, child_energy, coupling)
            # The switching weight, not just the admit/reject decision: the
            # cancellation claim is about the gap itself, and a screen that got
            # the gap slightly wrong would still round to the same boolean
            # everywhere except within `eps` of the threshold.
            assert screened == pytest.approx(exact, abs=1e-9), (
                f"screen and full evaluation disagree for {reaction.equation()}: "
                f"screened={screened}, exact={exact}"
            )
            admitted += screened > 0.0
            rejected += screened <= 0.0
            if admitted + rejected >= limit:
                return admitted, rejected
    return admitted, rejected


def test_screening_agrees_with_whole_state_evaluation(reaction_set):
    """`_channel_weight` must decide exactly what full evaluation would.

    The screen drops the spectator molecules from both diabats on the grounds
    that they cancel.  This checks that claim directly against the difference of
    the two full state energies.

    Both outcomes have to be covered.  At equilibrium geometries the gate
    rejects every channel -- correctly, nothing is reacting -- so a test run only
    there would pass against a `_channel_weight` that always returned zero.
    The stretched case below exists to exercise the admitting branch.
    """
    atoms = io.read("tests/data/mix-n100-d250.xyz")
    admitted, rejected = _compare_screen_against_full(reaction_set, atoms)
    assert rejected > 0, (
        "no channels were rejected; the test is not exercising the gate"
    )
    assert admitted == 0, (
        f"{admitted} channels admitted at equilibrium geometries, where the "
        "fitted coupling widths should quench every reaction"
    )


def test_screening_agrees_where_a_reaction_is_live(reaction_set):
    """The admitting branch of the screen, at a geometry that admits.

    Built from `geometry.REACTION` inside its admission ramp plus a spectator,
    rather than by deforming the mixture box.  That is both more robust and a
    better test of the claim: the screen's whole argument is that spectator
    molecules cancel between the two diabats, so the case worth running is one
    with an actual spectator in it, placed inside `bimol_cutoff` where it enters
    the reaction network.

    **This used to stretch one H-H bond to 1.8x and rely on that opening a
    channel.**  It stopped after the force-constant refit, and nothing about the
    box would bring it back: swept to 3.6x -- an H-H at 2.68 A, five times
    unbound -- and no channel admitted; nor did stretching an O-O to 2.87 A, nor
    walking an H2 up to an O2 down to a 0.85 A gap.  A deformation that happened
    to open a channel on one surface is not a property of the screen, and
    tying this test to one made it fail for a reason that had nothing to do with
    what it measures.
    """
    from geometry import REACTION, REACTION_PATH_RAMP, reaction_path, with_spectator

    spectator = io.read("datasets/HCombustion/molecules/mol_02.xyz")  # O2
    atoms = with_spectator(
        reaction_path(REACTION, REACTION_PATH_RAMP, SPECTATOR_CELL),
        spectator,
        SPECTATOR_GAP,
    )

    admitted, rejected = _compare_screen_against_full(reaction_set, atoms)
    assert admitted > 0, (
        "no channel was admitted at a geometry inside the admission ramp; the "
        "admitting branch of the screen is untested. Re-measure "
        "geometry.REACTION_PATH_RAMP -- it moves with every refit."
    )
    assert rejected > 0

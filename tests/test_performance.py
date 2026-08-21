"""Regression guards for the force-call optimizations.

Every optimization here was justified by an argument that it changes no
physics, so what has to be pinned is exactly that.  The energies below are the
values the calculator produced *before* any of the optimization work, recorded
to full precision:

  - Screening reaction candidates on the reacting fragment rather than the whole
    block (`EVBBasis._admits_reaction`) is exact because the molecules a reaction
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

# (energy, energy_bonded, energy_nonbonded) in eV, and the largest block, as
# produced before the optimization work.
REFERENCE = {
    "tests/data/mix-n100-d30.xyz": (
        -506.80083474578123,
        -506.8421507032254,
        0.04131595744415656,
        87,
    ),
    "tests/data/mix-n100-d250.xyz": (
        -505.9819399952007,
        -506.83834905014635,
        0.8564090549456548,
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
        parent_energy, _ = basis._energy(parent, atoms)

        for reaction, mapping in basis._reactions(parent, system.bimol_cutoff):
            broken, formed = reaction.edge_changes(mapping)
            if not broken and not formed:
                continue
            child = reaction.apply(parent, mapping, share_atoms=True)
            if state_key(child) == state_key(parent):
                continue
            child_energy, _ = basis._energy(child, atoms)
            coupling, _ = basis._coupling(atoms, reaction, mapping)

            screened = basis._admits_reaction(
                parent, mapping, (broken, formed), coupling, atoms
            )
            exact = basis._admits(parent_energy, child_energy, coupling)
            assert screened == exact, (
                f"screen and full evaluation disagree for {reaction.equation()}: "
                f"screened={screened}, exact={exact}"
            )
            admitted += screened
            rejected += not screened
            if admitted + rejected >= limit:
                return admitted, rejected
    return admitted, rejected


def test_screening_agrees_with_whole_state_evaluation(reaction_set):
    """`_admits_reaction` must decide exactly what full evaluation would.

    The screen drops the spectator molecules from both diabats on the grounds
    that they cancel.  This checks that claim directly against the difference of
    the two full state energies.

    Both outcomes have to be covered.  At equilibrium geometries the gate
    rejects every channel -- correctly, nothing is reacting -- so a test run only
    there would pass against an `_admits_reaction` that always returned False.
    The stretched case below exists to exercise the admitting branch.
    """
    atoms = io.read("tests/data/mix-n100-d250.xyz")
    admitted, rejected = _compare_screen_against_full(reaction_set, atoms)
    assert rejected > 0, "no channels were rejected; the test is not exercising the gate"
    assert admitted == 0, (
        f"{admitted} channels admitted at equilibrium geometries, where the "
        "fitted coupling widths should quench every reaction"
    )


def test_screening_agrees_where_a_reaction_is_live(reaction_set):
    """The admitting branch of the screen, on a stretched bond."""
    atoms = io.read("tests/data/mix-n100-d250.xyz")
    # Stretch one bond most of the way to dissociation so its channel switches on.
    atoms.positions[1] += (atoms.positions[1] - atoms.positions[0]) * 0.8

    admitted, rejected = _compare_screen_against_full(reaction_set, atoms)
    assert admitted > 0, (
        "no channel was admitted even with a bond stretched to 1.8x; the "
        "admitting branch of the screen is untested"
    )
    assert rejected > 0

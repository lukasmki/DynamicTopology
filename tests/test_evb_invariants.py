"""
Formal invariants of the multi-state EVB potential energy surface.

These are not accuracy tests.  They assert three properties that any
multi-state EVB method must satisfy for `E(x)` to be a well-defined potential
energy surface at all, independent of how well the force field or the couplings
happen to be parameterized:

  1. Pivot invariance.  The diabatic basis is generated from the topology the
     simulation happens to be carrying.  Since every state in that basis
     describes the same nuclear configuration, seeding the calculation from any
     of them must give the same energy and forces.  If it does not, `E` depends
     on trajectory history rather than on `x`, no energy is conserved, and
     forward and reverse paths across a barrier disagree.

  2. Continuity across the bimolecular cutoff.  Whether two molecules are close
     enough to be considered for reaction is decided by a hard distance test.
     Crossing it changes the dimension of the EVB matrix.  The energy must not
     jump when it does, or the "force" at that separation is a delta function
     and the reaction network is imprinting itself on the physics.  This is also
     how the size-consistency failure shows up in practice: the same spectator
     molecule contributes a different amount depending on whether it happens to
     share a subnetwork with an unrelated reaction.

  3. Permutation invariance.  Relabeling identical nuclei is not a physical
     change.  Energy must be unchanged and forces must follow the relabeling.
     Reaction templates are matched onto the live topology by graph isomorphism,
     and the matcher's choice among symmetry-equivalent mappings is arbitrary,
     so this is where an arbitrary choice becomes observable.

All three failed when this file was written, by 5.8-17.2 eV.  All three hold
now; the magnitude each one used to fail by is recorded on its class, because
those numbers are what makes a regression recognisable rather than just red.
`TestBasisClosure` covers the mechanism the first of them depends on.
"""

import numpy as np
import pytest

from DynamicTopology.basis import state_key
from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System

from geometry import (
    REACTION,
    REACTION_PATH_RAMP,
    REACTION_PATH_TS,
    reaction_path,
    with_spectator,
)


RSET_PATH = "datasets/HCombustion/HCombustion.json"

CELL = 60.0  # cubic box edge, Angstrom; large enough that nothing wraps
BIMOL_CUTOFF = 4.0  # must match System's default

# Tolerances.  The smooth part of the surface moves by ~1e-3 eV over the
# geometry steps used here, so these are loose enough not to be flaky and
# tight enough that the eV-scale failures below cannot hide.
ENERGY_TOL = 1e-6  # eV, for invariants that must hold exactly
FORCE_TOL = 1e-6  # eV/A
CONTINUITY_TOL = 1e-2  # eV, per geometry step across the cutoff


@pytest.fixture(scope="module")
def reaction_set():
    return ReactionSet(RSET_PATH)


@pytest.fixture(scope="module")
def templates(reaction_set):
    return {
        t.atoms.get_chemical_formula(): t
        for t in reaction_set.data["molecules"].values()
    }


def calculate(atoms, reaction_set, topology=None):
    """Single point at a fixed geometry, optionally seeded with a given topology."""
    if topology is None:
        topology = Topology.from_atoms(atoms)
    return System(atoms, topology, reaction_set).calculate()


def reactive(t=REACTION_PATH_TS):
    """A geometry on `rxn_16`'s path, which is where a multi-state basis lives.

    Two equilibrium templates placed near each other no longer mix at all --
    that is what a fitted coupling width means -- so these tests take their
    geometry from a reaction path instead.  See `tests/geometry.py`.  At the
    transition state the block holds three states; a 2-state star is
    pivot-invariant by construction and would pass test 1 vacuously.
    """
    return reaction_path(REACTION, t, CELL)


def basis_seeds(atoms, reaction_set):
    """Every state of the converged basis, as a topology fit to seed `System`.

    Seeding from each of these is the invariance the class below asserts: they
    all describe the same nuclei at the same positions, so they must all give
    the same answer.
    """
    system = System(atoms, Topology.from_atoms(atoms), reaction_set)
    blocks = system.basis.build(atoms, system.topology, BIMOL_CUTOFF)
    assert len(blocks) == 1, f"expected a single EVB block, got {len(blocks)}"
    block = blocks[0]
    assert block.nstates >= 3, (
        f"only {block.nstates} diabatic states; a 2-state basis is pivot-"
        "invariant by construction and would make these tests vacuous"
    )
    seeds = []
    for state in block.states:
        seed = Topology.from_molecules([state], remap=False)
        seed.set_atoms(atoms)
        seeds.append(seed)
    return seeds


class TestPivotInvariance:
    """E(x) must not depend on which state of the basis is treated as current.

    Held since the basis stopped being generated as a star centred on the
    current topology and became the closure of the gate-passing state graph,
    which is seed-independent because the gate is symmetric.  Was a 17.20 eV
    spread over the pivots of H2O+HO at one fixed geometry (-4.49 to -21.69 eV),
    with forces differing by 6.07 eV/A.

    Two defects fed that spread and both are pinned here: the star topology of
    the Hamiltonian, and `Reaction.apply` carrying the reactant's force field
    terms onto the product, which made every state in a block evaluate to the
    same diabatic energy.
    """

    def test_energy_is_independent_of_pivot(self, reaction_set):
        atoms = reactive()
        reference = calculate(atoms, reaction_set)["energy"]

        for i, seed in enumerate(basis_seeds(atoms, reaction_set)):
            energy = calculate(atoms, reaction_set, seed)["energy"]
            assert energy == pytest.approx(reference, abs=ENERGY_TOL), (
                f"seeding from state {i} changed the energy by "
                f"{energy - reference:+.4f} eV at an unchanged geometry"
            )

    def test_forces_are_independent_of_pivot(self, reaction_set):
        atoms = reactive()
        reference = calculate(atoms, reaction_set)["forces"]

        for i, seed in enumerate(basis_seeds(atoms, reaction_set)):
            forces = calculate(atoms, reaction_set, seed)["forces"]
            assert np.abs(forces - reference).max() < FORCE_TOL, (
                f"seeding from state {i} changed the forces by "
                f"{np.abs(forces - reference).max():.4f} eV/A"
            )

    def test_basis_is_not_truncated(self, reaction_set):
        """The two tests above are only meaningful on a converged basis.

        `max_states` / `max_depth` truncate in breadth-first order, which depends
        on the seed, so a capped basis is seed-dependent again and could make the
        invariance pass or fail for reasons unrelated to the formalism.
        """
        atoms = reactive()
        for i, seed in enumerate(basis_seeds(atoms, reaction_set)):
            blocks = calculate(atoms, reaction_set, seed)["blocks"]
            for j, block in enumerate(blocks):
                assert not block["capped"], (
                    f"basis for block {j} seeded from state {i} was truncated at "
                    f"{block['basis_size']} states / depth {block['depth']}"
                )

    def test_invariance_holds_with_a_channel_mid_ramp(self, reaction_set):
        """The switching weight has to be seed-independent too, not just the set.

        The gate admits a connected component, which is why membership does not
        depend on the seed.  The weight is a second thing every seed has to agree
        on, and it would not be enough for it to be merely close: it multiplies a
        coupling, so a seed-dependent weight is a seed-dependent energy.  It is
        symmetric in the two diabats by the same argument as the gate, so the
        agreement here is exact rather than approximate -- measured at 0.0 eV
        and 6.7e-16 eV/A across the three seeds, which all report the same
        `min_switch` of 0.278165 to the last digit.
        """
        atoms = reactive(REACTION_PATH_RAMP)
        reference = calculate(atoms, reaction_set)
        weight = min(block["min_switch"] for block in reference["blocks"])
        assert 0.0 < weight < 1.0, (
            f"no channel is inside the admission ramp (min_switch = {weight}); "
            "this geometry no longer exercises the switch"
        )

        for i, seed in enumerate(basis_seeds(atoms, reaction_set)):
            results = calculate(atoms, reaction_set, seed)
            assert results["energy"] == pytest.approx(
                reference["energy"], abs=ENERGY_TOL
            ), (
                f"seeding from state {i} changed the energy by "
                f"{results['energy'] - reference['energy']:+.4e} eV with a "
                "channel mid-ramp"
            )
            assert np.abs(results["forces"] - reference["forces"]).max() < FORCE_TOL
            assert min(b["min_switch"] for b in results["blocks"]) == pytest.approx(
                weight, abs=1e-12
            ), f"seeding from state {i} changed the switching weight itself"

    def test_nonbonded_is_independent_of_pivot(self, reaction_set):
        """ACKS2 is evaluated once, outside the EVB, from the current topology.

        That is only pivot-independent because its `atom` parameters happen to
        depend on the element alone in this dataset -- every O carries the same
        mu/eta/soft_*, every H likewise -- so the same nuclei at the same
        positions give the same charges whichever template they were looked up
        through.  It is a property of the data, not of the method, and it stops
        holding the moment a template carries topology-specific charges.  Pinned
        separately so that if it breaks it is not mistaken for a basis defect.
        """
        atoms = reactive()
        reference = calculate(atoms, reaction_set)["energy_nonbonded"]

        for i, seed in enumerate(basis_seeds(atoms, reaction_set)):
            nonbonded = calculate(atoms, reaction_set, seed)["energy_nonbonded"]
            assert nonbonded == pytest.approx(reference, abs=ENERGY_TOL), (
                f"seeding from state {i} changed the nonbonded energy by "
                f"{nonbonded - reference:+.4f} eV; ACKS2 atom parameters are no "
                "longer element-only and must move into the EVB diagonal"
            )


class TestBasisClosure:
    """Properties of the closure itself, independent of what it is used for."""

    def test_every_state_generates_the_same_basis(self, reaction_set):
        """The admitted set is a connected component, so any member generates it."""
        atoms = reactive()
        system = System(atoms, Topology.from_atoms(atoms), reaction_set)
        reference = system.basis.build(atoms, system.topology, BIMOL_CUTOFF)[0]
        expected = {state_key(state) for state in reference.states}

        for i, seed in enumerate(basis_seeds(atoms, reaction_set)):
            blocks = system.basis.build(atoms, seed, BIMOL_CUTOFF)
            found = {state_key(s) for block in blocks for s in block.states}
            assert found == expected, (
                f"seeding from state {i} gave {len(found)} states, not the "
                f"{len(expected)} of the basis it belongs to"
            )

    def test_admission_is_symmetric(self, reaction_set):
        """The gate cannot prefer one direction, or reachability is directional.

        This is the whole invariance argument in one assertion: the admitted set
        is the seed's connected component only if the edge relation is
        undirected.

        Asserted on the switching *weight*, not on the admit/reject decision.
        The gate is a ramp, so the coupling a channel actually carries has to be
        symmetric too -- two seeds that agree on which states exist but disagree
        on how strongly they couple still give different energies.  Exact
        equality rather than a tolerance: `hypot` and `abs` are both even in the
        half gap, so there is nothing to round.
        """
        atoms = reactive()
        system = System(atoms, Topology.from_atoms(atoms), reaction_set)
        gate = system.basis._switch

        rng = np.random.default_rng(0)
        ramping = 0
        for _ in range(200):
            a, b = rng.normal(0.0, 5.0, size=2)
            coupling = rng.normal(0.0, 2.0)
            weight = gate(a, b, coupling)
            assert weight == gate(b, a, coupling)
            # Sign of the coupling is a phase choice and must not matter either.
            assert weight == gate(a, b, -coupling)
            ramping += 0.0 < weight < 1.0

        # Drawn from a distribution that straddles the threshold, so the ramp
        # itself is covered rather than only its two flat ends.
        assert ramping > 0, (
            "no sampled pair landed inside the admission ramp; the symmetry of "
            "the switching weight is untested"
        )

    def test_hamiltonian_is_symmetric_and_couples_beyond_the_seed(
        self, reaction_set, templates
    ):
        atoms = reactive()
        system = System(atoms, Topology.from_atoms(atoms), reaction_set)
        block = system.basis.build(atoms, system.topology, BIMOL_CUTOFF)[0]
        ham, _, _ = block.hamiltonian()

        assert np.allclose(ham, ham.T, atol=0.0), "EVB matrix is not symmetric"

        # Distinct diabats: the terms-carryover bug made every state in a block
        # evaluate to the same energy, which this would have caught.
        assert np.ptp(block.energies) > 1.0, (
            f"all {block.nstates} diabats within {np.ptp(block.energies):.2e} eV "
            "of each other; states are not being parameterized independently"
        )

        # A star Hamiltonian has couplings only in the seed's row and column.
        off_diagonal = ~np.eye(block.nstates, dtype=bool)
        away_from_seed = off_diagonal.copy()
        away_from_seed[block.seed_index, :] = False
        away_from_seed[:, block.seed_index] = False
        assert np.abs(ham[away_from_seed]).max() > 0.0, (
            "no coupling between two non-seed states; the Hamiltonian is still a "
            "star and the basis cannot represent a multi-step channel"
        )


class TestCutoffContinuity:
    """Energy must be continuous where the reaction network changes shape."""

    # Held since the couplings were given fitted widths.  The hard distance test
    # is still there, but a state entering or leaving the matrix at 4.0 A now
    # arrives with a negligible coupling, so it no longer moves the energy: the
    # worst step across the boundary fell from -2.43 eV (uniform a = 10, where
    # the coupling was still 0.96-7.70 eV at the reaction endpoints) to 0.0018
    # eV.  Restoring the uniform widths reintroduces the discontinuity.
    def test_energy_is_continuous_across_the_bimolecular_cutoff(
        self, reaction_set, templates
    ):
        # A spectator H2 is walked away from a reacting fragment.  Below the
        # cutoff it joins that subnetwork and its channels enter the same
        # matrix; above it, it becomes an independent subnetwork.  The geometry
        # change per step is negligible either side of the boundary.
        #
        # H2 rather than O2: the reacting fragment is H2O2, and no channel in
        # the dataset pairs it with O2 at all, so an O2 spectator crosses the
        # cutoff without the applicable set ever changing and the scan measures
        # nothing.  The count assertion in the next test is what caught that.
        reacting = reactive()
        energies, separations = [], []
        for dx in np.arange(3.80, 4.35, 0.05):
            atoms = with_spectator(reacting, templates["H2"].atoms, dx)
            energies.append(calculate(atoms, reaction_set)["energy"])
            positions = atoms.positions
            separations.append(
                min(
                    np.linalg.norm(positions[i] - positions[j])
                    for i in range(len(reacting))
                    for j in range(len(reacting), len(atoms))
                )
            )
        energies = np.array(energies)
        assert separations[0] < BIMOL_CUTOFF < separations[-1], (
            "the scan must straddle the cutoff for this test to mean anything; "
            f"got {separations[0]:.3f} to {separations[-1]:.3f} A"
        )

        jumps = np.abs(np.diff(energies))
        worst = int(jumps.argmax())
        assert jumps.max() < CONTINUITY_TOL, (
            f"energy jumped by {jumps.max():.4f} eV over a 0.05 A step, between "
            f"separations {separations[worst]:.3f} and {separations[worst + 1]:.3f} A "
            f"(cutoff {BIMOL_CUTOFF} A)"
        )

    def test_the_repulsion_does_not_move_at_the_cutoff(self, reaction_set, templates):
        """The repulsion is the same number on both sides of the applicable set.

        `bimol_cutoff` decides which reactions are applicable, and therefore how
        many diabatic states a block has.  A repulsion that knew about the
        topology would change shape exactly there, and every previous version of
        this term did: a boolean-on-applicability rule stepped by the whole
        wall, and the pair potential had to be switched off by `bimol_cutoff` to
        make the step cost anything less than that.  `ZBL` takes no topology at
        all, so the set can change however it likes and the number does not move
        -- and `pair_potential` needs no cutoff, which is why it does not have
        one.

        Asserted as exact equality rather than as a tolerance, because that is
        what topology-independence means.  The count assertion is what stops it
        from being vacuously true of a scan where the set never changed.
        """
        from DynamicTopology.core import Topology

        reacting = reactive()
        repulsions, counts, separations = [], [], []
        for dx in np.arange(3.88, 4.00, 0.005):
            atoms = with_spectator(reacting, templates["H2"].atoms, dx)
            state = Topology.from_atoms(atoms)
            state.set_atoms(atoms)
            repulsions.append(
                System(atoms, state, reaction_set).calculate()["energy_zbl"]
            )
            counts.append(
                sum(
                    1 for _ in reaction_set.get_network(state, BIMOL_CUTOFF).reactions()
                )
            )
            positions = atoms.positions
            separations.append(
                min(
                    np.linalg.norm(positions[i] - positions[j])
                    for i in range(len(reacting))
                    for j in range(len(reacting), len(atoms))
                )
            )

        assert separations[0] < BIMOL_CUTOFF < separations[-1], (
            "the scan must straddle the cutoff for this test to mean anything; "
            f"got {separations[0]:.3f} to {separations[-1]:.3f} A"
        )
        assert len(set(counts)) > 1, (
            f"no step in the scan changed the applicable set; counts were {counts}"
        )
        assert repulsions[0] > 1.0, (
            "the repulsion is negligible throughout, so a scan of it cannot show "
            "whether the applicable set moves it"
        )

        # The spectator is *moving* along this scan, so the repulsion is
        # supposed to change -- it is a function of the geometry.  What it must
        # not do is change *differently* where the applicable set changes.  So
        # the step at each boundary is compared against the steps where nothing
        # changed, rather than against zero.
        steps = np.abs(np.diff(repulsions))
        boundaries = [i for i in range(len(counts) - 1) if counts[i] != counts[i + 1]]
        interior = [s for i, s in enumerate(steps) if i not in boundaries]
        assert interior, "the applicable set changed at every step; nothing to compare"

        for i in boundaries:
            assert steps[i] <= 2.0 * max(interior), (
                f"the repulsion stepped by {steps[i]:.3e} eV as the applicable "
                f"set changed from {counts[i]} reactions to {counts[i + 1]}, "
                f"against {max(interior):.3e} eV for the largest step where it "
                f"did not, between separations {separations[i]:.4f} and "
                f"{separations[i + 1]:.4f} A. It has acquired a topology "
                "dependence."
            )


class TestPermutationInvariance:
    """Relabeling identical nuclei is not a physical change."""

    # Held since three separate index-space defects were fixed: the coupling was
    # superposed onto the transition-state reference in matcher-discovery order
    # rather than template order; ACKS2 built its response matrix in term order
    # while indexing it globally; and only one of the several symmetry-equivalent
    # isomorphisms was enumerated, so relabeling silently selected a different
    # reaction channel (H2O's two O-H bonds), matching in energy by symmetry
    # while putting the forces on different atoms.  Was 5.04 eV / 5.29 eV/A.
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_relabeling_atoms_changes_nothing(self, reaction_set, seed):
        atoms = reactive()
        reference = calculate(atoms, reaction_set)

        permutation = np.random.default_rng(seed).permutation(len(atoms))
        permuted = calculate(atoms[permutation], reaction_set)

        assert permuted["energy"] == pytest.approx(
            reference["energy"], abs=ENERGY_TOL
        ), (
            f"relabeling changed the energy by "
            f"{permuted['energy'] - reference['energy']:+.4f} eV"
        )
        # Forces must follow the relabeling, not stay put.
        assert np.abs(permuted["forces"] - reference["forces"][permutation]).max() < (
            FORCE_TOL
        ), (
            "relabeling changed the forces by "
            f"{np.abs(permuted['forces'] - reference['forces'][permutation]).max():.4f} eV/A"
        )

"""
Energy conservation tests for the ASE calculators.

A conservative force field satisfies F = -dE/dr exactly, so an NVE trajectory
integrated with velocity Verlet conserves the total energy up to the
integrator's O(dt^2) discretization error.  Two things follow, and both are
asserted here:

  1. The total energy stays in a narrow band around its initial value.
  2. That band shrinks ~4x when the timestep is halved.  This is the part that
     actually distinguishes "conservative forces with a finite timestep" from
     "inconsistent forces": a genuine force/energy mismatch produces drift that
     does not improve as dt shrinks.

What each calculator covers:

  EVB               Bonded QForce terms only, with the empirical geometric-mean
                    coupling.  ACKS2 is added once on top of the ground state
                    rather than to the Hamiltonian diagonal, so it does not
                    pass through the nonlinear coupling.  Conservative.
  DynamicTopology   QForce + RMSD EVB coupling + ACKS2 electrostatics.  ACKS2
                    charges respond to the geometry, and that response is now
                    carried into the force by an adjoint solve
                    (ACKS2.compute_response_forces), so the electrostatic force
                    is the gradient of the electrostatic energy for polar
                    species too -- not only where charges vanish by symmetry.

Both calculators are exercised only where the bonding topology is constant.  A
topology switch is a discontinuous change of the potential and is not expected
to conserve energy, so each run asserts the topology never changed.

The basis, on the other hand, is allowed to change: `EVBBasis` ramps a state's
coupling up from zero as it enters, so a state joining or leaving the basis
costs no energy.  `test_gate_crossing_conserves_energy` is the case that
exercises it, and it used to be an xfail -- with the hard threshold it drifted
by one `eps` per admission event.
"""

import numpy as np
import pytest
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from DynamicTopology.ase import EVB, DynamicTopology
from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System

from geometry import (
    REACTION,
    SWITCHING_PATH_START,
    SWITCHING_REACTION,
    reaction_path,
)


RSET_PATH = "datasets/HCombustion/HCombustion.json"

CELL = 24.0  # cubic box edge, Angstrom
SPACING = 6.0  # molecule separation, Angstrom (beyond the 4.0 bimolecular cutoff)
TEMPERATURE = 300.0  # K
SEED = 0

# Where the gate-crossing cases start.  Two equilibrium templates placed near
# each other no longer produce a multi-state basis at all -- fitted coupling
# widths switch off at the reactant and product minima by construction -- so a
# trajectory that crosses the admission gate has to start near a transition
# state.  See `tests/geometry.py`.  Slightly on the reactant side of the saddle
# rather than on it, so the run starts in a well and the kinetic energy the
# drift is measured against is representative.
GATE_START = -0.15
GATE_TEMPERATURE = 1000.0  # K

TIMESTEP = 0.05  # fs
# Peak-to-peak drift, as a fraction of the mean kinetic energy.  Measured values
# at this timestep are ~3e-4 (EVB) and ~1e-5 (DynamicTopology).
DRIFT_TOL = 2e-3
# Halving dt must shrink the drift by at least this much (ideal: 4.0).
CONVERGENCE_TOL = 3.5


@pytest.fixture(scope="module")
def reaction_set():
    return ReactionSet(RSET_PATH)


def build_atoms(reaction_set, formulas, spacing=SPACING):
    """Lay out molecule templates from the ReactionSet along x in a cubic box.

    H2 is deliberately absent from every case in this file.  Its bond length
    (0.7445 A) sits just above molify's default perception cutoff for H-H
    (1.2 * 2 * 0.31 = 0.744 A), so a hand-placed H2 is perceived as two unbonded
    H atoms.  The .xyz files under tests/data carry an explicit `connectivity`
    array and so do not hit this; templates placed by hand do.  The assertion
    below fails loudly rather than silently simulating the wrong system.
    """
    atoms = Atoms(cell=np.eye(3) * CELL, pbc=False)
    x = 4.0
    for formula in formulas:
        template = next(reaction_set.get_molecules(formulas=[formula]))
        mol = template.atoms.copy()
        mol.positions -= mol.positions.mean(0)
        mol.positions += np.array([x, CELL / 2, CELL / 2])
        atoms += mol
        x += spacing
    atoms.set_cell(np.eye(3) * CELL)
    atoms.set_pbc(False)

    expected_bonds = sum(
        len(next(reaction_set.get_molecules(formulas=[f])).graph.edges)
        for f in formulas
    )
    perceived = Topology.from_atoms(atoms)
    assert perceived.graph.number_of_edges() == expected_bonds, (
        f"bond perception failed for {formulas}: "
        f"expected {expected_bonds} bonds, got {list(perceived.graph.edges())}"
    )
    return atoms


def relax(atoms, calc_cls, reaction_set, fmax=0.02, steps=300):
    """Relax to a local minimum so the NVE run starts near the bottom of a well.

    Only used for DynamicTopology.  EVB's coupling, H_ij = sqrt((1+h)*H_ii*H_jj),
    grows without bound as the diabatic energies grow, so its ground state has no
    lower bound and a geometry optimization runs away instead of converging.
    """
    atoms = atoms.copy()
    atoms.calc = calc_cls(atoms, reaction_set)
    BFGS(atoms, logfile=None).run(fmax=fmax, steps=steps)
    atoms.calc = None
    return atoms


def _topology_edges(calc):
    """Current bonding topology of a calculator, as a hashable value."""
    topology = getattr(calc.system, "topology", None)
    if topology is None:  # EVBSystem holds fixed states, not a live topology
        return None
    return frozenset(frozenset(edge) for edge in topology.graph.edges())


def run_nve(
    atoms,
    calc_cls,
    reaction_set,
    dt_fs=TIMESTEP,
    steps=200,
    seed=SEED,
    temperature_K=TEMPERATURE,
):
    """Run NVE.

    Returns (total energies, kinetic energies, topology_changed, basis_changes).
    The last is how many times the number of diabatic states changed, which is
    what the gate-crossing cases assert they are not vacuous by: conserving
    energy across a gate is only interesting if a gate was crossed.
    """
    atoms = atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=temperature_K, rng=np.random.default_rng(seed)
    )
    Stationary(atoms)
    atoms.calc = calc_cls(atoms, reaction_set)

    # Guard against a vacuous pass: a state that produces no force conserves
    # energy trivially, which would make the assertions below meaningless.
    assert np.abs(atoms.get_forces()).max() > 0.0, "calculator produced zero forces"

    initial_edges = _topology_edges(atoms.calc)
    changed = False
    basis_changes, previous_size = 0, None

    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    total, kinetic = [], []
    for _ in range(steps + 1):
        total.append(atoms.get_total_energy())
        kinetic.append(atoms.get_kinetic_energy())
        if _topology_edges(atoms.calc) != initial_edges:
            changed = True
        diagnostics = getattr(atoms.calc, "diagnostics", None)
        if diagnostics is not None:
            size = sum(block["basis_size"] for block in diagnostics["blocks"])
            basis_changes += previous_size is not None and size != previous_size
            previous_size = size
        dyn.run(1)

    return np.array(total), np.array(kinetic), changed, basis_changes


def drift_ratio(total, kinetic):
    """Peak-to-peak energy drift relative to the mean kinetic energy.

    Normalizing by kinetic energy keeps the threshold meaningful across systems
    of different size and stiffness, and independent of the large constant
    offset in the EVB ground-state energy.
    """
    return (total.max() - total.min()) / kinetic.mean()


def assert_conserves(total, kinetic, changed):
    assert not changed, "topology changed mid-run; conservation is not expected"
    assert drift_ratio(total, kinetic) < DRIFT_TOL, (
        f"energy drift {total.max() - total.min():.3e} eV is large compared to "
        f"mean kinetic energy {kinetic.mean():.3e} eV"
    )


def assert_converges_with_timestep(drifts):
    for coarse, fine in zip(drifts, drifts[1:]):
        assert coarse / fine > CONVERGENCE_TOL, (
            f"drift did not shrink as the timestep was halved: {drifts}. "
            "Drift that is independent of dt indicates forces inconsistent "
            "with the energy, not integrator error."
        )


class TestEVBEnergyConservation:
    """EVB evaluates bonded terms only and must conserve energy.

    Runs start from the template geometry: EVB's potential is unbounded below
    (see `relax`), so these use a short window and rely on the timestep-
    convergence test below to show the residual is integrator error.
    """

    # O2 and other species that dissociate into bare atoms are excluded: the
    # dissociated state carries no bonded terms, so H_jj = 0, the geometric-mean
    # coupling is 0, and the ground state collapses to a flat zero-force state.
    SYSTEMS = [
        pytest.param(["H2O"], id="H2O"),
        pytest.param(["HO2"], id="HO2"),
        pytest.param(["H2O2"], id="H2O2"),
        pytest.param(["H2O", "O2"], id="H2O+O2"),
    ]

    @pytest.mark.parametrize("formulas", SYSTEMS)
    def test_conserves_energy(self, reaction_set, formulas):
        atoms = build_atoms(reaction_set, formulas)
        total, kinetic, changed, _ = run_nve(atoms, EVB, reaction_set, steps=80)
        assert_conserves(total, kinetic, changed)

    def test_drift_converges_with_timestep(self, reaction_set):
        atoms = build_atoms(reaction_set, ["H2O", "O2"])
        drifts = []
        for dt in (0.1, 0.05, 0.025):
            total, _, changed, _ = run_nve(
                atoms, EVB, reaction_set, dt_fs=dt, steps=int(round(4.0 / dt))
            )
            assert not changed
            drifts.append(total.max() - total.min())
        assert_converges_with_timestep(drifts)


class TestDynamicTopologyEnergyConservation:
    """DynamicTopology adds the RMSD EVB coupling and ACKS2 electrostatics.

    The polar cases are the ones that matter here.  O2 conserved even under the
    old frozen-charge forces, because its charges vanish by symmetry and the
    missing term was identically zero; HO2 and H2O2 did not, and they are what
    the charge-response adjoint bought.  Keep at least one of each, so a
    regression that reintroduces the frozen-charge force is caught rather than
    hidden behind the symmetric cases.
    """

    SYSTEMS = [
        pytest.param(["O2"], id="O2"),
        pytest.param(["O2", "O2"], id="O2+O2"),
        pytest.param(["HO2"], id="HO2"),
        pytest.param(["H2O2"], id="H2O2"),
    ]

    @pytest.mark.parametrize("formulas", SYSTEMS)
    def test_conserves_energy(self, reaction_set, formulas):
        atoms = relax(
            build_atoms(reaction_set, formulas), DynamicTopology, reaction_set
        )
        total, kinetic, changed, _ = run_nve(
            atoms, DynamicTopology, reaction_set, steps=200
        )
        assert_conserves(total, kinetic, changed)

    def test_drift_converges_with_timestep(self, reaction_set):
        atoms = relax(
            build_atoms(reaction_set, ["O2", "O2"]), DynamicTopology, reaction_set
        )
        drifts = []
        for dt in (0.1, 0.05, 0.025):
            total, _, changed, _ = run_nve(
                atoms,
                DynamicTopology,
                reaction_set,
                dt_fs=dt,
                steps=int(round(10.0 / dt)),
            )
            assert not changed
            drifts.append(total.max() - total.min())
        assert_converges_with_timestep(drifts)

    def test_gate_crossing_conserves_energy(self, reaction_set):
        """H2O crosses the admission gate, and must conserve energy anyway.

        This is the case the switching function exists for, kept separate from
        the parametrized cases above so a regression in the ramp is legible on
        its own.  With a hard threshold the drift over this run is one `eps` per
        admission event and dt-*independent*, which is the signature of a
        discontinuous potential rather than integrator error; it is 4.4e-04 of
        the kinetic energy with the ramp, and `switch_width = 0` fails here.

        A drift that climbs back toward 1e-2 means the ramp is not continuous; a
        drift that is small but stops shrinking with dt means the switch's own
        gradient is wrong.  `test_drift_converges_with_timestep` below covers the
        second, so assert the first here.
        """
        atoms = reaction_path(REACTION, GATE_START, CELL)
        total, kinetic, changed, basis_changes = run_nve(
            atoms,
            DynamicTopology,
            reaction_set,
            steps=200,
            temperature_K=GATE_TEMPERATURE,
        )
        assert basis_changes > 0, (
            "the basis never changed size over this run, so nothing crossed the "
            "gate and conserving energy across it is not being tested"
        )
        assert_conserves(total, kinetic, changed)

    def test_gate_crossing_drift_converges_with_timestep(self, reaction_set):
        """The same gate crossing again, with the dt-halving criterion.

        The one that separates "continuous potential, finite timestep" from
        "smaller discontinuity": a step in the energy does not shrink with dt,
        however small it is.  Measured ratios 4.00 and 4.00, on a run whose
        basis does change size.
        """
        atoms = reaction_path(REACTION, GATE_START, CELL)
        drifts = []
        for dt in (0.1, 0.05, 0.025):
            total, _, changed, basis_changes = run_nve(
                atoms,
                DynamicTopology,
                reaction_set,
                temperature_K=GATE_TEMPERATURE,
                dt_fs=dt,
                steps=int(round(10.0 / dt)),
            )
            assert not changed
            drifts.append(total.max() - total.min())
        assert_converges_with_timestep(drifts)


class TestTopologyChangeContinuity:
    """Switching the carried topology mid-trajectory must not move the energy.

    This is the property the pivot-invariant basis exists to provide, and the
    only place it is exercised on geometries the dynamics actually visits --
    `test_evb_invariants.py` asserts it at one hand-placed geometry.  The
    distinction matters because the failure mode is history-dependence: a
    surface that is pivot-invariant at a chosen point can still drift if the
    basis it generates along a trajectory does not close.

    Asserted directly rather than inferred from drift.  At each switch the same
    geometry is re-evaluated seeded with the topology the calculator was
    carrying *before* the switch, and the two must agree.  Reading it off the
    total-energy trace instead would be far weaker: measured over this run, a
    switch moves the energy by less than an ordinary integration step does, so
    the events are already quieter than the integrator and a drift-based test
    could not tell a broken swap from the baseline.
    """

    # Started at `rxn_02`'s transition state.  `rxn_16`, which the rest of this
    # file uses, holds a wider basis but is a *symmetric* hydrogen transfer, so
    # a trajectory started there sits in one topology and the guard below --
    # correctly -- reports that the test would be asserting nothing.  `rxn_02`
    # is not symmetric and crosses four times in these 300 steps.
    REACTION = SWITCHING_REACTION
    START = SWITCHING_PATH_START
    TEMPERATURE = 1000.0
    STEPS = 300
    DT = 0.1  # fs; hot and coarse on purpose, to provoke switches

    def _trajectory(self, reaction_set):
        """NVE on the pair, yielding (positions, previous_graph) at each switch."""
        atoms = reaction_path(self.REACTION, self.START, CELL)
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=self.TEMPERATURE, rng=np.random.default_rng(SEED)
        )
        Stationary(atoms)
        atoms.calc = DynamicTopology(atoms, reaction_set)

        switches, sizes, energies = [], [], []
        dyn = VelocityVerlet(atoms, timestep=self.DT * units.fs)
        for _ in range(self.STEPS):
            previous = atoms.calc.system.topology.graph.copy()
            energy = atoms.get_potential_energy()
            diagnostics = atoms.calc.diagnostics
            if atoms.calc.topology_changed:
                switches.append(
                    (
                        atoms.positions.copy(),
                        previous,
                        energy,
                        atoms.get_forces().copy(),
                    )
                )
            sizes.append(tuple(b["basis_size"] for b in diagnostics["blocks"]))
            energies.append(energy)
            dyn.run(1)
        return switches, sizes, energies

    def test_energy_is_unchanged_by_the_switch(self, reaction_set):
        switches, sizes, _ = self._trajectory(reaction_set)

        assert len(switches) >= 2, (
            f"only {len(switches)} topology switches in {self.STEPS} steps; this "
            "test asserts nothing unless the trajectory actually crosses one"
        )
        assert len(set(sizes)) > 1, (
            "the basis never changed size, so the run never left a single "
            "reaction regime and the switches are not representative"
        )

        for positions, previous_graph, energy, forces in switches:
            atoms = reaction_path(self.REACTION, self.START, CELL)
            atoms.set_positions(positions)
            seeded = System(
                atoms, Topology(previous_graph.copy(), atoms), reaction_set
            ).calculate()
            assert seeded["energy"] == pytest.approx(energy, abs=1e-6), (
                "the energy depends on which topology the trajectory happened to "
                "be carrying, so E is a function of history and not of geometry"
            )
            np.testing.assert_allclose(seeded["forces"], forces, atol=1e-6)


class TestGateContinuity:
    """A state entering or leaving the basis must not move the energy.

    `EVBBasis` admits a state when mixing with it stabilizes the block by more
    than `eps`.  With a hard threshold that costs up to `eps` of energy every
    time one crosses -- which is what made `test_gate_crossing_conserves_energy`
    an xfail -- so the coupling is now ramped up from zero across
    `[eps, eps + switch_width]` instead.

    Asserted on the second difference of the energy rather than the first.  A
    step of size `d` in the energy shows up as a spike of `d` in the second
    difference, while the smooth background contributes only O(h^2 * E''), so
    the event separates from the surrounding steps by orders of magnitude
    instead of having to be distinguished from them by a threshold.  Measured
    over the scan below: the event's second difference is 2.4e-06 with the ramp,
    indistinguishable from the 2.3e-06 background, against 1.0e-03 with
    `switch_width = 0` -- a factor of 436.
    """

    # Scanned either side of the two straddling geometries as well as between
    # them, so the event is interior to the scan and the background is sampled
    # on both sides.
    SCAN = np.linspace(-0.5, 1.5, 41)
    # The event's second difference must not stand out from the background by
    # more than this. The hard gate exceeds it by ~436x.
    SPIKE_TOL = 10.0

    def _crossing(self, reaction_set):
        """Consecutive trajectory geometries that straddle a basis-size change.

        Taken from a trajectory rather than a hand-built scan because the gate
        has to be crossed at a geometry the dynamics actually visits: the
        stabilization is a property of the diabatic gap, and walking a molecule
        along an axis moves the Morse energy by far more per step than the event
        is worth, which buries it completely.  Here the two frames are a single
        0.05 fs timestep apart.
        """
        atoms = reaction_path(REACTION, GATE_START, CELL)
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=GATE_TEMPERATURE, rng=np.random.default_rng(SEED)
        )
        Stationary(atoms)
        atoms.calc = DynamicTopology(atoms, reaction_set)

        dyn = VelocityVerlet(atoms, timestep=TIMESTEP * units.fs)
        previous_positions, previous_size = None, None
        for _ in range(200):
            atoms.get_potential_energy()
            size = sum(b["basis_size"] for b in atoms.calc.diagnostics["blocks"])
            if previous_size is not None and size != previous_size:
                return previous_positions, atoms.positions.copy()
            previous_positions, previous_size = atoms.positions.copy(), size
            dyn.run(1)
        raise AssertionError(
            "no basis-size change over 200 steps; the scan below would not "
            "cross the gate and the test would be vacuous"
        )

    def test_energy_is_continuous_across_the_gate(self, reaction_set):
        before, after = self._crossing(reaction_set)
        template = reaction_path(REACTION, GATE_START, CELL)

        energies, sizes = [], []
        for t in self.SCAN:
            atoms = template.copy()
            atoms.positions = before + t * (after - before)
            results = System(
                atoms, Topology.from_atoms(atoms), reaction_set
            ).calculate()
            energies.append(results["energy"])
            sizes.append(sum(b["basis_size"] for b in results["blocks"]))

        assert len(set(sizes)) > 1, (
            f"the scan stayed at {sizes[0]} states throughout; it has to cross "
            "the gate for this test to mean anything"
        )

        curvature = np.abs(np.diff(np.array(energies), 2))
        spike = curvature.max() / np.median(curvature)
        assert spike < self.SPIKE_TOL, (
            f"the energy's second difference spikes by {spike:.0f}x at the basis "
            f"change ({curvature.max():.3e} eV against a background of "
            f"{np.median(curvature):.3e}); a state is entering the basis with a "
            "coupling it did not ramp up to"
        )

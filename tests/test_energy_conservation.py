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
                    coupling.  Conservative.
  DynamicTopology   QForce + RMSD EVB coupling + ACKS2 electrostatics.  ACKS2
                    forces freeze the charges (dQ/dr = 0, see TestACKS2Gradients
                    in test_gradients.py), so the electrostatic force is not the
                    gradient of the electrostatic energy whenever the charge
                    response is significant.  It vanishes by symmetry for
                    homonuclear species, so the conserving cases here are
                    O2-only; the polar counterexample is pinned by an xfail
                    below.

Both calculators are exercised only where the bonding topology is constant.  A
topology switch is a discontinuous change of the potential and is not expected
to conserve energy, so each run asserts the topology never changed.
"""

import numpy as np
import pytest
from ase import Atoms, units
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.verlet import VelocityVerlet
from ase.optimize import BFGS

from DynamicTopology.ase import EVB, DynamicTopology
from DynamicTopology.core import ReactionSet, Topology


RSET_PATH = "datasets/HCombustion/HCombustion.json"

CELL = 24.0  # cubic box edge, Angstrom
SPACING = 6.0  # molecule separation, Angstrom (beyond the 4.0 bimolecular cutoff)
TEMPERATURE = 300.0  # K
SEED = 0

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


def run_nve(atoms, calc_cls, reaction_set, dt_fs=TIMESTEP, steps=200, seed=SEED):
    """Run NVE and return (total energies, kinetic energies, topology_changed)."""
    atoms = atoms.copy()
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=TEMPERATURE, rng=np.random.default_rng(seed)
    )
    Stationary(atoms)
    atoms.calc = calc_cls(atoms, reaction_set)

    # Guard against a vacuous pass: a state that produces no force conserves
    # energy trivially, which would make the assertions below meaningless.
    assert np.abs(atoms.get_forces()).max() > 0.0, "calculator produced zero forces"

    initial_edges = _topology_edges(atoms.calc)
    changed = False

    dyn = VelocityVerlet(atoms, timestep=dt_fs * units.fs)
    total, kinetic = [], []
    for _ in range(steps + 1):
        total.append(atoms.get_total_energy())
        kinetic.append(atoms.get_kinetic_energy())
        if _topology_edges(atoms.calc) != initial_edges:
            changed = True
        dyn.run(1)

    return np.array(total), np.array(kinetic), changed


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
        total, kinetic, changed = run_nve(atoms, EVB, reaction_set, steps=80)
        assert_conserves(total, kinetic, changed)

    def test_drift_converges_with_timestep(self, reaction_set):
        atoms = build_atoms(reaction_set, ["H2O", "O2"])
        drifts = []
        for dt in (0.1, 0.05, 0.025):
            total, _, changed = run_nve(
                atoms, EVB, reaction_set, dt_fs=dt, steps=int(round(4.0 / dt))
            )
            assert not changed
            drifts.append(total.max() - total.min())
        assert_converges_with_timestep(drifts)


class TestDynamicTopologyEnergyConservation:
    """DynamicTopology adds the RMSD EVB coupling and ACKS2 electrostatics.

    Limited to homonuclear systems, where ACKS2 charges vanish by symmetry and
    the frozen-charge force approximation is therefore exact.
    """

    SYSTEMS = [
        pytest.param(["O2"], id="O2"),
        pytest.param(["O2", "O2"], id="O2+O2"),
    ]

    @pytest.mark.parametrize("formulas", SYSTEMS)
    def test_conserves_energy(self, reaction_set, formulas):
        atoms = relax(build_atoms(reaction_set, formulas), DynamicTopology, reaction_set)
        total, kinetic, changed = run_nve(atoms, DynamicTopology, reaction_set, steps=200)
        assert_conserves(total, kinetic, changed)

    def test_drift_converges_with_timestep(self, reaction_set):
        atoms = relax(build_atoms(reaction_set, ["O2", "O2"]), DynamicTopology, reaction_set)
        drifts = []
        for dt in (0.1, 0.05, 0.025):
            total, _, changed = run_nve(
                atoms, DynamicTopology, reaction_set, dt_fs=dt, steps=int(round(10.0 / dt))
            )
            assert not changed
            drifts.append(total.max() - total.min())
        assert_converges_with_timestep(drifts)

    @pytest.mark.parametrize(
        "formulas",
        [pytest.param(["H2O"], id="H2O"), pytest.param(["HO2"], id="HO2")],
    )
    @pytest.mark.xfail(
        strict=True,
        reason="ACKS2 forces freeze the charges (dQ/dr = 0), so for polar species "
        "the electrostatic force is not the gradient of the electrostatic energy "
        "and NVE energy is not conserved. The drift is ~25-400% of the kinetic "
        "energy and does not shrink with the timestep. Drop this xfail once the "
        "charge-response term is implemented.",
    )
    def test_polar_species_do_not_conserve_energy(self, reaction_set, formulas):
        atoms = relax(build_atoms(reaction_set, formulas), DynamicTopology, reaction_set)
        total, kinetic, _ = run_nve(atoms, DynamicTopology, reaction_set, steps=200)
        # Only the drift criterion is asserted: these runs may also switch
        # topology, which is a separate reason not to expect conservation.
        assert drift_ratio(total, kinetic) < DRIFT_TOL

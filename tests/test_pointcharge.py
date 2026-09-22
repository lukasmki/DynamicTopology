"""Fixed point-charge electrostatics through the whole force call.

`test_gradients.py` and `test_stress.py` check `PointCharge` one evaluation at a
time.  What they cannot see is the part that is new in kind: the charges are a
property of the diabatic state, so they sit on the EVB diagonal, and every block
sees every other block through its weight-averaged charges.  `System.calculate`
iterates the blocks to self-consistency, and the forces are the gradient of the
energy only once it has.  The tests here are the ones that would notice if it
had not.

The dataset is `datasets/Water-fixed-pc`, activated under `params.use` for the
duration of this module so that it cannot collide with the other datasets the
suite loads into the same process.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.forcefield.params import ForceFieldParams, use
from DynamicTopology.forcefield.pointcharge import PointCharge
from DynamicTopology.system import System

from test_gradients import finite_difference_forces
from test_stress import assert_symmetric, finite_difference_virial

MANIFEST = Path("datasets/Water-fixed-pc/Water.json").resolve()
ZUNDEL = Path("datasets/Water-fixed-pc/reactions/h3o-h2o-transfer.xyz")
WATER = Path("datasets/Water-fixed-pc/molecules/h2o.xyz")
PARAMS = ForceFieldParams.from_dict(json.loads(MANIFEST.read_text())["global_params"])


@pytest.fixture(scope="module")
def reaction_set():
    with use(PARAMS):
        yield ReactionSet(MANIFEST)


@pytest.fixture(autouse=True)
def _pointcharge_active(reaction_set):
    with use(PARAMS):
        yield


def zundel(offset=(0.0, 0.0, 0.0), proton=0.0) -> Atoms:
    """The hop's transition-state frame, its shared proton moved `proton` A along O-O.

    At the symmetric Zundel the block is 50/50; +/-0.1 A takes it to 33/67, so
    within that range both states carry real weight and it is the charges that
    decide how much.
    """
    atoms = io.read(ZUNDEL, index=1)
    atoms.calc = None
    atoms.positions[1, 0] += proton
    atoms.positions += offset
    return atoms


def box(*parts: Atoms, cell=20.0, pbc=False) -> Atoms:
    """`parts` in one cell, each keeping the connectivity its frame states.

    Carried over explicitly because `Atoms.__iadd__` drops `info`, and a
    transition-state frame perceived from distances bridges the shared proton
    into an H5O2 that no template describes.
    """
    atoms = Atoms()
    connectivity = []
    for part in parts:
        offset = len(atoms)
        connectivity += [
            [i + offset, j + offset, order] for i, j, order in part.info["connectivity"]
        ]
        atoms += part
    atoms.info["connectivity"] = connectivity
    atoms.set_cell(np.eye(3) * cell)
    atoms.pbc = pbc
    # A deterministic jiggle, so no force component vanishes by symmetry.
    atoms.positions += np.random.default_rng(7).normal(0.0, 0.02, atoms.positions.shape)
    return atoms


def water(offset) -> Atoms:
    atoms = io.read(WATER)
    atoms.calc = None
    atoms.positions += offset
    return atoms


def calculate(atoms, reaction_set):
    return System(atoms, Topology.from_atoms(atoms), reaction_set).calculate()


def multi_state(results):
    return [block for block in results["blocks"] if block["nstates"] > 1]


class TestSelfConsistency:
    def test_one_mixed_block_is_converged_in_one_sweep(self, reaction_set):
        """Its environment is spectators, whose charges nothing can move."""
        atoms = box(zundel(proton=0.05), water([0.0, 4.5, 0.0]))
        results = calculate(atoms, reaction_set)
        assert len(multi_state(results)) == 1
        assert results["electrostatics_sweeps"] == 1

    def test_two_mixed_blocks_are_iterated(self, reaction_set):
        """Two charged Zundels 7 A apart: each one's ground state moves the other's.

        Beyond the bimolecular cutoff, so they are separate blocks, and 14.4/7 =
        2 eV of Coulomb between them, so the coupling is not a rounding error.
        """
        atoms = box(zundel(proton=0.05), zundel((0.0, 7.0, 0.0), proton=-0.03))
        results = calculate(atoms, reaction_set)
        assert len(multi_state(results)) == 2
        assert results["electrostatics_sweeps"] > 2

    def test_a_block_feels_the_other_blocks_state(self, reaction_set):
        """Move only the second complex's proton; the first's weights must follow.

        With ACKS2 the charges are the same on every state, so the first block's
        diagonal cannot depend on which state the second one is in.  Here it is
        the point: the +1 moves with the proton.
        """
        a = calculate(
            box(zundel(proton=0.05), zundel((0.0, 7.0, 0.0), proton=-0.1)), reaction_set
        )
        b = calculate(
            box(zundel(proton=0.05), zundel((0.0, 7.0, 0.0), proton=0.1)), reaction_set
        )
        wa, wb = multi_state(a)[0]["weights"], multi_state(b)[0]["weights"]
        assert abs(wa[0] - wb[0]) > 1e-4


class TestGradients:
    """F = -dE/dr and W = dE/de through `System`, which is exact only at convergence."""

    def test_forces_one_mixed_block(self, reaction_set):
        atoms = box(zundel(proton=0.05), water([0.0, 4.5, 0.0]))
        self._check_forces(atoms, reaction_set)

    def test_forces_two_mixed_blocks(self, reaction_set):
        """The case the iteration is for.

        Measured with `SCF_MAX_SWEEPS` forced to 1: the forces are off by
        0.058 eV/A, against 4.3e-8 once converged (four sweeps), because the
        first block's weights move by 3e-3 after it has already been
        diagonalized.
        """
        atoms = box(zundel(proton=0.05), zundel((0.0, 7.0, 0.0), proton=-0.03))
        self._check_forces(atoms, reaction_set)

    def test_forces_and_virial_periodic(self, reaction_set):
        """A charged cell: Ewald, the background, and a mixed block together."""
        atoms = box(zundel((2.0, 3.0, 3.0), proton=0.05), cell=10.0, pbc=True)
        results = calculate(atoms, reaction_set)
        assert len(multi_state(results)) == 1
        self._check_forces(atoms, reaction_set)

        def energy_fn(positions, cell):
            perturbed = atoms.copy()
            perturbed.positions = positions
            perturbed.set_cell(cell)
            return calculate(perturbed, reaction_set)["energy"]

        w_fd = finite_difference_virial(
            energy_fn, atoms.positions, np.asarray(atoms.cell), delta=1e-6
        )
        assert_symmetric(results["virial"], atol=1e-6)
        np.testing.assert_allclose(results["virial"], w_fd, atol=1e-4, rtol=1e-4)

    def _check_forces(self, atoms, reaction_set):
        results = calculate(atoms, reaction_set)

        def energy_fn(positions):
            perturbed = atoms.copy()
            perturbed.positions = positions
            return calculate(perturbed, reaction_set)["energy"]

        f_fd = finite_difference_forces(energy_fn, atoms.positions, delta=1e-5)
        np.testing.assert_allclose(results["forces"], f_fd, atol=1e-6, rtol=1e-5)


class TestTheCharges:
    def test_every_template_carries_its_formal_charge(self, reaction_set):
        formal = {"H": 0, "O": 0, "HO": -1, "H2O": 0, "H3O": 1}
        for molecule in reaction_set.data.molecules.values():
            q = sum(t["kwargs"]["q"] for t in molecule.terms if t["type"] == "charge")
            formula = molecule.atoms.get_chemical_formula()
            assert q == pytest.approx(formal[formula], abs=1e-12), formula

    def test_the_excess_charge_follows_the_proton(self, reaction_set):
        """Reactant frame and product frame: the +1 is on a different oxygen."""
        frames = io.read(ZUNDEL, index=":")

        def charge_near(atoms, oxygen):
            topology = Topology.from_atoms(atoms)
            topology.set_terms(reaction_set.get_terms(topology))
            (molecule,) = [m for m in topology.molecules() if oxygen in m.graph]
            return sum(
                t["kwargs"]["q"]
                for t in reaction_set.get_terms(molecule)
                if t["type"] == "charge"
            )

        assert charge_near(frames[0], 0) == pytest.approx(1.0)
        assert charge_near(frames[0], 4) == pytest.approx(0.0)
        assert charge_near(frames[-1], 0) == pytest.approx(0.0)
        assert charge_near(frames[-1], 4) == pytest.approx(1.0)


class TestTheExclusionUnderEwald:
    def test_a_neutral_molecule_has_no_self_image_energy(self, reaction_set):
        """An isolated water, every pair excluded, in a periodic box.

        What survives is the molecule's interaction with its own periodic
        images, a dipole-dipole sum that falls as 1/L^3 -- measured at -1.9 meV
        in a 12.43 A box.  Excluding the whole periodic `K_ij`, images and all,
        leaves `CCOUL/2 * K_self * sum q_i^2` instead, which is -0.91 eV here
        and falls only as 1/L; that is the error this rules out.
        """
        atoms = water([6.0, 6.0, 6.0])
        atoms.set_cell(np.eye(3) * 12.43)
        atoms.pbc = True
        topology = Topology.from_atoms(atoms)
        term_dict = topology.set_terms(reaction_set.get_terms(topology))
        energy, _, _ = PointCharge()(atoms.positions, atoms.pbc, atoms.cell, term_dict)
        assert abs(energy) < 5e-3

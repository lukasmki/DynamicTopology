"""The iterative ACKS2 charge solve (`global_params.charge_solver = "iterative"`).

PME for the reciprocal half of the kernel, a real-space cutoff for the rest,
and projected conjugate gradients for the environment's system.  An
approximation to the direct solve rather than a reordering of it, so these
hold it to the direct numbers within tolerances well under anything MD
resolves, and its own forces to its own energy.
"""

import dataclasses

import numpy as np
import pytest

from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.ewald import Ewald, EwaldOperatorSetup
from DynamicTopology.forcefield.neighbors import Geometry
from DynamicTopology.forcefield.params import active, use
from DynamicTopology.forcefield.pme import PMESetup
from DynamicTopology.forcefield.pointcharge import geometry

from test_gradients import (
    POS_ZUNDEL,
    ZUNDEL_A,
    ZUNDEL_B,
    acks2_at_weights,
    fake_block,
    finite_difference_forces,
    zundel_state,
    zundel_system,
)

PBC = np.ones(3, dtype=bool)
CELL = np.eye(3) * 9.0


def iterative():
    return use(dataclasses.replace(active(), charge_solver="iterative"))


def test_pme_is_the_reciprocal_sum():
    """PME's potential against the reciprocal sum done directly, at one `kappa`."""
    rng = np.random.default_rng(0)
    pos = rng.uniform(0.0, 9.0, size=(30, 3))
    q = rng.normal(size=30)
    q -= q.mean()
    ewald = Ewald(CELL, accuracy=1e-12)
    vecs, rij = geometry(pos, PBC, CELL)
    kernel = ewald.bind(pos, vecs, rij)
    exact = kernel.cos @ (ewald.weight * (kernel.cos.T @ q)) + kernel.sin @ (
        ewald.weight * (kernel.sin.T @ q)
    )
    phi = PMESetup(CELL, ewald.kappa, 1e-12).bind(pos).potential(q)
    assert np.abs(phi - exact).max() < 1e-7 * np.abs(exact).max()


def test_the_operator_is_the_matrix():
    """`EwaldOperator` applied, against the dense `EwaldKernel` it stands for.

    Different splittings of one kernel, so they agree to the lattice sums'
    accuracy, entry for entry, and so do their contractions.
    """
    rng = np.random.default_rng(1)
    pos = rng.uniform(0.0, 9.0, size=(40, 3))
    vecs, rij = geometry(pos, PBC, CELL)
    dense = Ewald(CELL).bind(pos, vecs, rij)
    operator = EwaldOperatorSetup(CELL).bind(pos, Geometry(pos, PBC, CELL))
    K = dense.matrix()
    q = rng.normal(size=40)
    np.testing.assert_allclose(operator.matvec(q), K @ q, atol=1e-7 * np.abs(K @ q).max())
    np.testing.assert_allclose(operator.columns([3, 7]), K[:, [3, 7]], atol=1e-7)
    factor = rng.normal(size=(40, 2))
    signs = np.array([1.0, -1.0])
    expected = dense.contract((factor * signs) @ factor.T)
    got = operator.contract(None, factor, signs)
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b, atol=1e-6 * np.abs(b).max())


def test_one_topology_matches_the_direct_solve():
    td = zundel_system(ZUNDEL_A)
    direct = ACKS2()(POS_ZUNDEL, PBC, CELL, td)
    with iterative():
        solver = ACKS2()
        got = solver(POS_ZUNDEL, PBC, CELL, td)
        assert solver.K is None, "the iterative path was not taken"
    # 3 eV here; the lattice sums agree to ~5e-8 of it in a cell this small.
    assert got[0] == pytest.approx(direct[0], abs=1e-6)
    np.testing.assert_allclose(got[1], direct[1], atol=1e-6)
    np.testing.assert_allclose(got[2], direct[2], atol=5e-6)


def test_a_block_matches_the_direct_solve():
    """The block path: the environment's response, the corrections, `evaluate`."""
    seed = zundel_system(ZUNDEL_A)
    blocks = [fake_block([zundel_state(ZUNDEL_A), zundel_state(ZUNDEL_B)])]
    weights = {0: np.array([0.6, 0.4])}

    def run():
        acks2, result = acks2_at_weights(POS_ZUNDEL, PBC, CELL, seed, blocks, weights)
        return acks2, acks2.corrections(0), result

    _, corrections, (energy, forces, virial) = run()
    with iterative():
        solver, got_corrections, (got_energy, got_forces, got_virial) = run()
        assert solver.K is None
    np.testing.assert_allclose(got_corrections, corrections, atol=1e-6)
    assert got_energy == pytest.approx(energy, abs=1e-6)
    np.testing.assert_allclose(got_forces, forces, atol=1e-6)
    np.testing.assert_allclose(got_virial, virial, atol=5e-6)


def test_its_forces_are_its_own_gradient():
    """The iterative solve's forces against its own energy, to the residual."""
    td = zundel_system(ZUNDEL_A)
    with iterative(), use(dataclasses.replace(active(), solver_tolerance=1e-12)):
        _, forces, _ = ACKS2()(POS_ZUNDEL, PBC, CELL, td)
        numeric = finite_difference_forces(
            lambda p: ACKS2()(p, PBC, CELL, td)[0], POS_ZUNDEL
        )
    np.testing.assert_allclose(forces, numeric, atol=1e-5)


def test_a_molecule_kept_explicit_takes_the_direct_path():
    """A stretched bond leaves its molecule's potentials explicit, which the
    projected solve does not handle; that call is the direct one, exactly."""
    td = zundel_system(ZUNDEL_A)
    positions = POS_ZUNDEL.copy()
    positions[9] += [0.0, 3.5, 3.5]  # O-H 5.8 A
    direct = ACKS2()(positions, PBC, CELL, td)
    with iterative():
        solver = ACKS2()
        got = solver(positions, PBC, CELL, td)
        assert solver.K is not None, "the call did not fall back"
    assert got[0] == direct[0]
    np.testing.assert_array_equal(got[1], direct[1])

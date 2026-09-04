"""
Stress tests for force field implementations.

The companion to `test_gradients.py`: where that file checks `F = -dE/dr`
against central differences in the *positions*, this one checks the virial
`W = dE/d(strain)` against central differences in the *cell*, with the atoms
scaled affinely along with it.

**Why the virial exists at all.**  Every ASE barostat asks the calculator for
`stress`, and until this was added `DynamicTopology.implemented_properties` held
only `energy` and `forces`, so an NPT run raised `PropertyNotImplementedError`
before taking a step.  `production/density-300K` is the simulation that needs it.

**Why it is cheap to compute.**  All three force fields build the full
minimum-image displacement matrix `vecs` and express every energy as a function
of those vectors alone.  A homogeneous strain `e` maps `v -> (I + e) v`, so

    W_ab = sum_pairs  v_a * (dE/dv)_b

and `dE/dv` is exactly the per-pair gradient each `compute_*` already forms on
its way to scattering the forces.  No new derivatives are involved anywhere --
which is the thing these tests are really checking, since a wrong contraction
would still look like a plausible 3x3.

Two properties are asserted throughout, and the second catches errors the first
cannot:

  * the virial matches the finite-difference strain derivative, and
  * it is **symmetric**.  Any potential invariant under rigid rotation has a
    symmetric stress, so an asymmetric result means a term was contracted as
    `dv_a v_b` in one place and `v_a dv_b` in another -- a transpose bug that a
    trace-only check (i.e. anything that only looks at the pressure) would pass.
"""

import numpy as np
import pytest

from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.zbl import ZBL

from DynamicTopology.forcefield.coupling import EVBCoupling

from test_gradients import (
    make_term,
    POS_2,
    POS_3,
    POS_4,
    POS_H2O2,
)
from geometry import REACTION, REACTION_PATH_RAMP, reaction_path


# Finite difference step, dimensionless (it is a strain, not a length).
DELTA = 1e-6

# Stress tests must be periodic: with `pbc` false there is no cell for the
# energy to depend on.  A 9 A cube around the ~1-4 A test geometries is small
# enough that the minimum-image branch is genuinely exercised and large enough
# that no pair sits near the L/2 seam, where an infinitesimal strain would flip
# which image is nearest and the energy would not be differentiable.
PBC = np.ones(3, dtype=bool)
CELL = np.eye(3) * 9.0


def strained(pos, cell, e):
    """`pos` and `cell` under the homogeneous strain `e`.

    Positions are rows, so the column-vector map `r -> (I + e) r` is written as
    a right-multiplication by the transpose.  The cell's rows are its lattice
    vectors and transform the same way, which is what makes the deformation
    homogeneous: fractional coordinates are unchanged, so no atom crosses a
    boundary and the minimum-image assignment is fixed across the difference.
    """
    m = np.eye(3) + e
    return pos @ m.T, np.asarray(cell) @ m.T


def finite_difference_virial(energy_fn, pos, cell, delta=DELTA):
    """`dE/de_ab` by central differences over the six independent components.

    Only symmetric strains are applied.  The antisymmetric part is an
    infinitesimal rotation, which every one of these potentials is invariant
    under, so it carries no information and differencing it would return noise
    divided by `2 * delta`.
    """
    w = np.zeros((3, 3))
    for a in range(3):
        for b in range(a, 3):
            e = np.zeros((3, 3))
            # Split off-diagonals across both slots so the strain stays
            # symmetric.  Both W_ab and W_ba then pick up half the perturbation
            # and, W being symmetric, the difference returns W_ab whole -- the
            # same normalisation the diagonal gets, where the two halves land in
            # the one slot.  No extra factor either way.
            e[a, b] += 0.5
            e[b, a] += 0.5
            pos_p, cell_p = strained(pos, cell, delta * e)
            pos_m, cell_m = strained(pos, cell, -delta * e)
            w[a, b] = (energy_fn(pos_p, cell_p) - energy_fn(pos_m, cell_m)) / (
                2 * delta
            )
            w[b, a] = w[a, b]
    return w


def assert_symmetric(w, atol=1e-8):
    """Rotational invariance of the potential implies a symmetric virial."""
    np.testing.assert_allclose(
        w,
        w.T,
        atol=atol,
        err_msg="virial is not symmetric -- check for a transposed contraction",
    )


# ---------------------------------------------------------------------------
# ZBL
# ---------------------------------------------------------------------------


class TestZBLStress:
    zbl = ZBL()

    @pytest.mark.parametrize(
        "positions", [POS_2, POS_3, POS_4, POS_H2O2], ids=["n2", "n3", "n4", "h2o2"]
    )
    def test_all_pairs(self, positions):
        numbers = np.array([8 if i % 2 == 0 else 1 for i in range(len(positions))])

        def energy_fn(p, c):
            return self.zbl(p, numbers, PBC, c)[0]

        _, _, w = self.zbl(positions, numbers, PBC, CELL)
        w_fd = finite_difference_virial(energy_fn, positions, CELL)
        assert_symmetric(w)
        np.testing.assert_allclose(w, w_fd, atol=1e-5, rtol=1e-5)

    def test_dense_periodic_box(self):
        """A box tight enough that the minimum-image branch does real work."""
        rng = np.random.default_rng(11)
        cell = np.eye(3) * 6.0
        positions = rng.uniform(0.0, 6.0, (10, 3))
        numbers = rng.choice([1, 8], size=10)

        def energy_fn(p, c):
            return self.zbl(p, numbers, PBC, c)[0]

        _, _, w = self.zbl(positions, numbers, PBC, cell)
        w_fd = finite_difference_virial(energy_fn, positions, cell)
        assert_symmetric(w)
        np.testing.assert_allclose(w, w_fd, atol=1e-5, rtol=1e-4)

    def test_repulsion_is_a_positive_pressure(self):
        """ZBL is purely repulsive, so it can only ever push the box outward.

        A sign error in the virial is otherwise invisible: it would still match
        a finite difference taken with the same wrong sign convention somewhere
        upstream, and would still be symmetric.  This pins the convention
        against the physics instead.  `W = dE/de` positive means compressing the
        box (negative strain) lowers the energy -- so for a repulsion `W` must
        be negative, i.e. the pressure `-tr(W)/3V` is positive.
        """
        numbers = np.array([1, 1])
        positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        _, _, w = self.zbl(positions, numbers, PBC, CELL)
        pressure = -np.trace(w) / (3 * np.linalg.det(CELL))
        assert pressure > 0.0


# ---------------------------------------------------------------------------
# ACKS2
# ---------------------------------------------------------------------------


class TestACKS2Stress:
    """The charge response is the part that can go wrong here.

    `compute_coulomb` alone is not the gradient of its own energy -- the charges
    solve a geometry-dependent linear system, so straining the cell moves them.
    `test_call` is the test that covers it: it re-solves the charges at every
    displaced geometry, exactly as a trajectory does.  `test_coulomb_frozen`
    checks only the explicit half and by construction cannot see a missing
    response term, which is the same trap `test_gradients.py` documents for the
    forces.
    """

    acks2 = ACKS2()

    _TERM_DICT = {
        "atom": {
            "atoms": np.array([[0], [1], [2], [3]]),
            "kwargs": {
                "mu": np.array([8.12, 8.12, 1.88, 1.88]),
                "eta": np.array([3.74, 3.74, 7.28, 7.28]),
                "soft_amp": np.array([3.88, 3.88, 2.10, 2.10]),
                "soft_decay": np.array([0.44, 0.44, 0.27, 0.27]),
            },
        }
    }

    def _vecs(self, pos, cell):
        vecs = pos[:, None, :] - pos[None, :, :]
        f = vecs @ np.linalg.inv(cell)
        vecs = vecs - (PBC * np.floor(f + 0.5)) @ cell
        return vecs, np.sqrt(np.sum(vecs * vecs, -1))

    def test_coulomb_frozen(self):
        """The explicit half of the strain derivative, at fixed charges."""
        vecs_ref, rij_ref = self._vecs(POS_H2O2, CELL)
        Q = self.acks2.compute_charges(rij_ref, self._TERM_DICT["atom"]["kwargs"])

        def energy_fn(p, c):
            v, r = self._vecs(p, c)
            return self.acks2.compute_coulomb(Q, r, v)[0]

        _, _, w = self.acks2.compute_coulomb(Q, rij_ref, vecs_ref)
        w_fd = finite_difference_virial(energy_fn, POS_H2O2, CELL)
        assert_symmetric(w)
        np.testing.assert_allclose(w, w_fd, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("positions", [POS_H2O2, POS_4], ids=["h2o2", "n4"])
    def test_call(self, positions):
        """The whole term, with the charges free to re-solve under the strain."""

        def energy_fn(p, c):
            return ACKS2()(p, PBC, c, self._TERM_DICT)[0]

        _, _, w = ACKS2()(positions, PBC, CELL, self._TERM_DICT)
        w_fd = finite_difference_virial(energy_fn, positions, CELL)
        assert_symmetric(w)
        np.testing.assert_allclose(w, w_fd, atol=1e-6, rtol=1e-4)

    def test_response_is_not_negligible(self):
        """Guard the guard: the frozen-charge virial must be visibly wrong.

        If the response contribution happened to be tiny for this geometry then
        `test_call` would pass with the response term deleted, and would be
        testing nothing.  It is not tiny -- this asserts that.
        """
        vecs, rij = self._vecs(POS_H2O2, CELL)
        Q, u, A = self.acks2.solve_charges(rij, self._TERM_DICT["atom"]["kwargs"])
        _, _, w_explicit = self.acks2.compute_coulomb(Q, rij, vecs)
        _, w_response = self.acks2.compute_response_forces(
            Q, u, A, rij, vecs, self._TERM_DICT["atom"]["kwargs"]
        )
        assert np.abs(w_response).max() > 0.01 * np.abs(w_explicit).max()


# ---------------------------------------------------------------------------
# QForce
# ---------------------------------------------------------------------------


class TestQForceStress:
    """One case per term method, mirroring `TestQForceGradients`.

    The parameters and geometries are the same ones the force tests use, so a
    term whose force is right and whose stress is wrong is isolated to the
    virial contraction rather than to the derivative underneath it.

    `CLAUDE.md`'s rule that every new or edited `compute_*` needs a case in
    `test_gradients.py` extends here: a term with a force case and no stress
    case is half tested, and the half that is missing is the one an NPT run
    depends on.
    """

    qf = QForce()

    def _check(self, pos, term_dict, qf=None, atol=1e-4, rtol=1e-4):
        qf = qf or self.qf

        def energy_fn(p, c):
            return qf(p, PBC, c, term_dict)[0]

        _, _, w = qf(pos, PBC, CELL, term_dict)
        w_fd = finite_difference_virial(energy_fn, pos, CELL)
        assert_symmetric(w, atol=1e-10)
        np.testing.assert_allclose(
            w,
            w_fd,
            atol=atol,
            rtol=rtol,
            err_msg=f"Virial mismatch for term type: {list(term_dict)}",
        )

    def test_bond(self):
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0])
        self._check(POS_2, td)

    @pytest.mark.parametrize("c", [1.0, 4.0, -1.5])
    def test_bond_morse_shape(self, c):
        """The shape term, walked along the stretched branch and back.

        `POS_2` sits near `r0`, where the Hulburt-Hirschfelder correction and
        its first two derivatives all vanish -- the same blind spot the force
        test documents.  A virial checked only there would pass with the shape
        term's contribution dropped entirely.
        """
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0], c=[c])
        al = np.sqrt(251200.0 / (2 * 436.0))
        for offset in (-1.0, 0.0, 0.5, 1.5, 3.0):
            pos = np.array([[0.0, 0.0, 0.0], [0.07772 + offset / al, 0.0, 0.0]])
            self._check(pos, td)

    def test_bond_harmonic(self):
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0])
        self._check(POS_2, td, qf=QForce(bond_form="harmonic"))

    def test_reference_carries_no_stress(self):
        """A constant shift moves no atom and stores no stress.

        Worth asserting rather than assuming: `compute_reference` is the one
        term whose virial is structurally zero, so a non-zero result here would
        mean the accumulator is picking up something that is not a gradient.
        """
        td = make_term("reference", [[0]], E0=[-1234.5])
        _, _, w = self.qf(POS_3, PBC, CELL, td)
        np.testing.assert_allclose(w, np.zeros((3, 3)), atol=0.0)

    def test_exclusion(self):
        td = make_term("exclusion", [[0, 1]], sigma=[0.25], eps=[0.5])
        self._check(POS_3, td)

    def test_angle(self):
        td = make_term("angle", [[0, 1, 2]], theta0=[1.911], k=[500.0])
        self._check(POS_3, td)

    def test_bondbond(self):
        td = make_term(
            "bondbond", [[1, 0, 2, 0]], r1_0=[0.085], r2_0=[0.120], k=[100.0]
        )
        self._check(POS_3, td)

    def test_bondangle(self):
        td = make_term(
            "bondangle", [[0, 1, 2, 0, 1]], theta0=[1.911], r0=[0.100], k=[500.0]
        )
        self._check(POS_3, td)

    def test_angleangle(self):
        td = make_term(
            "angleangle",
            [[2, 0, 1, 0, 1, 3]],
            theta1_0=[1.782],
            theta2_0=[1.782],
            k=[43.0],
        )
        self._check(POS_H2O2, td)

    def test_periodicdihedral(self):
        td = make_term(
            "periodicdihedral", [[0, 1, 2, 3]], phi0=[3.14159], n=[2.0], k=[1.0]
        )
        self._check(POS_4, td)

    def test_dihedralbond(self):
        td = make_term(
            "dihedralbond",
            [[0, 1, 2, 3, 1, 2]],
            phi0=[3.14159],
            n=[2.0],
            k=[5.0],
            r0=[0.140],
        )
        self._check(POS_4, td)

    def test_dihedralangle(self):
        td = make_term(
            "dihedralangle",
            [[0, 1, 2, 3, 0, 1, 2]],
            phi0=[3.14159],
            n=[2.0],
            k=[1.0],
            theta0=[2.094],
        )
        self._check(POS_4, td)

    def test_dihedralangleangle(self):
        td = make_term(
            "dihedralangleangle",
            [[0, 1, 2, 3]],
            phi0=[3.14159],
            n=[2.0],
            k=[1.0],
            theta0_1=[2.094],
            theta0_2=[2.094],
        )
        self._check(POS_4, td)

    def test_the_unit_conversion(self):
        """The virial converts with the energy factor and no length factor.

        `__call__` works in nm and multiplies the forces by
        `units.kJ/units.mol/units.nm`.  The virial is `v (x) dE/dv` with `v`
        already in nm, so it is an energy and takes `units.kJ/units.mol` alone.
        Dividing by `units.nm` as well -- the natural copy-paste error -- would
        leave every pressure a factor of ten small, which no finite-difference
        test written in the same wrong units could catch.  This pins it against
        an independent quantity: for a pure Morse bond the trace of the virial
        must equal `r * dE/dr` in eV, computed here from the ASE-unit energy
        curve rather than from anything inside `QForce`.
        """
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0])
        r = 0.11  # nm, well up the repulsive wall so dE/dr is large
        pos = np.array([[0.0, 0.0, 0.0], [r * 10.0, 0.0, 0.0]])
        _, _, w = self.qf(pos, PBC, CELL, td)

        h = 1e-6

        def energy_at(d):
            return self.qf(np.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]]), PBC, CELL, td)[0]

        # dE/dr in eV/Angstrom, times r in Angstrom -> eV
        de_dr = (energy_at(r * 10.0 + h) - energy_at(r * 10.0 - h)) / (2 * h)
        assert np.trace(w) == pytest.approx(de_dr * r * 10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# The EVB coupling
# ---------------------------------------------------------------------------


class TestEVBCouplingStress:
    coupling = EVBCoupling()

    def test_rmsd(self):
        """The RMSD-Gaussian coupling's virial.

        The same caveat the force test carries applies here: the analytic
        gradient holds `Superpose3D`'s optimal alignment fixed while a finite
        difference re-optimises it.  That is exact for rotation and translation,
        which the RMSD is invariant under -- but *not* for the scale factor `S`,
        which a strain genuinely moves.  The tolerance below is looser than the
        per-term ones for that reason, and it is the reason this term is checked
        on its own rather than only inside the assembled surface.
        """
        pos = POS_3.copy()
        ref = pos + np.array(
            [
                [0.40, 0.10, 0.20],
                [-0.30, 0.20, -0.10],
                [0.20, -0.40, 0.30],
            ]
        )
        ensemble = ref[np.newaxis]
        td = make_term("rmsd", [[0, 1, 2]], A=[-10.0], a=[10.0])

        def energy_fn(p, c):
            return self.coupling(p, PBC, c, ensemble, td)[0]

        _, _, w = self.coupling(pos, PBC, CELL, ensemble, td)
        w_fd = finite_difference_virial(energy_fn, pos, CELL, delta=1e-5)
        assert_symmetric(w, atol=1e-9)
        np.testing.assert_allclose(w, w_fd, atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# The assembled surface
# ---------------------------------------------------------------------------

RSET_PATH = "datasets/HCombustion/HCombustion.json"
SYSTEM_CELL = 60.0


class TestSystemStress:
    """Finite differences through the whole force call.

    The counterpart of `TestSystemGradients`, and it exists for the same reason:
    every test above checks one contraction in isolation, which leaves the
    assembly unchecked -- the EVB ground state, the Hellmann-Feynman
    contraction against `vham`, and the strain gradient of the switching weight
    a channel carries as it enters the basis.

    That last one is why the geometry is `REACTION_PATH_RAMP` rather than
    anything convenient: it puts a channel strictly inside the admission ramp,
    where `_channel_weight` returns a non-trivial `dweight_virial`.  Nothing
    else in this file would notice if that term were dropped.
    """

    @pytest.fixture(scope="class")
    def reaction_set(self):
        from DynamicTopology.core import ReactionSet

        return ReactionSet(RSET_PATH)

    def _calculate(self, atoms, reaction_set):
        from DynamicTopology.core import Topology
        from DynamicTopology.system import System

        return System(atoms, Topology.from_atoms(atoms), reaction_set).calculate()

    def test_virial_inside_the_admission_ramp(self, reaction_set):
        atoms = reaction_path(REACTION, REACTION_PATH_RAMP, SYSTEM_CELL)
        results = self._calculate(atoms, reaction_set)

        weight = min(block["min_switch"] for block in results["blocks"])
        assert 0.0 < weight < 1.0, (
            f"no channel is inside the admission ramp (min_switch = {weight}); "
            "the switch's strain gradient is not being exercised and this test "
            "is vacuous"
        )

        def energy_fn(positions, cell):
            perturbed = atoms.copy()
            perturbed.positions = positions
            perturbed.set_cell(cell)
            return self._calculate(perturbed, reaction_set)["energy"]

        w_fd = finite_difference_virial(
            energy_fn, atoms.positions, np.asarray(atoms.cell), delta=1e-6
        )
        assert_symmetric(results["virial"], atol=1e-6)
        np.testing.assert_allclose(results["virial"], w_fd, atol=1e-4, rtol=1e-4)

    def test_stress_reaches_the_calculator(self, reaction_set):
        """`atoms.get_stress()` must work -- this is the whole point.

        Every ASE barostat calls it, and before the virial existed it raised
        `PropertyNotImplementedError`.  Checked against the virial the system
        returns, divided by the volume, in ASE's Voigt order.
        """
        from ase.stress import full_3x3_to_voigt_6_stress

        from DynamicTopology.ase import DynamicTopology

        atoms = reaction_path(REACTION, REACTION_PATH_RAMP, SYSTEM_CELL)
        atoms.set_pbc(True)
        atoms.calc = DynamicTopology(atoms, reaction_set)

        stress = atoms.get_stress()
        assert stress.shape == (6,)
        assert np.all(np.isfinite(stress))

        expected = full_3x3_to_voigt_6_stress(
            atoms.calc.system.calculate()["virial"] / atoms.get_volume()
        )
        np.testing.assert_allclose(stress, expected, rtol=1e-8, atol=1e-12)

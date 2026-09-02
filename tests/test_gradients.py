"""
Gradient tests for force field implementations.

Each test verifies that analytical forces (F = -dE/dr) are consistent with
forces computed by central finite differences of the energy.
"""

import numpy as np
import pytest
from ase import units

from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.coupling import EVBCoupling
from DynamicTopology.forcefield.lj import LennardJones
from DynamicTopology.forcefield.zbl import ZBL
from DynamicTopology.forcefield import zbl as zbl_module

from geometry import REACTION, REACTION_PATH_RAMP, reaction_path


# Finite difference step (Angstrom)
DELTA = 1e-4

PBC = np.zeros(3, dtype=bool)
CELL = np.eye(3) * 30.0


def finite_difference_forces(energy_fn, pos, delta=DELTA):
    """Central finite difference: F_ia = -(E(r + h*e_ia) - E(r - h*e_ia)) / (2h)."""
    f_fd = np.zeros_like(pos)
    for i in range(pos.shape[0]):
        for d in range(3):
            pos_p = pos.copy()
            pos_p[i, d] += delta
            pos_m = pos.copy()
            pos_m[i, d] -= delta
            f_fd[i, d] = -(energy_fn(pos_p) - energy_fn(pos_m)) / (2 * delta)
    return f_fd


def make_term(term_type, atoms_rows, **kwargs):
    """Build a term_dict in the format expected by QForce/ACKS2 __call__."""
    return {
        term_type: {
            "atoms": np.array(atoms_rows, dtype=int),
            "kwargs": {
                k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in kwargs.items()
            },
        }
    }


# ---------------------------------------------------------------------------
# Reference geometries (Angstrom)
# ---------------------------------------------------------------------------

# Diatomic (H-H slightly off equilibrium r0=0.777 Å)
POS_2 = np.array(
    [
        [0.00, 0.00, 0.00],
        [0.85, 0.10, 0.00],
    ]
)

# Triatomic: H-O-H-like (perturbed from equilibrium)
POS_3 = np.array(
    [
        [1.00, 0.10, 0.05],
        [0.00, 0.00, 0.00],
        [-0.10, 1.00, -0.05],
    ]
)

# 4-atom chain with a nonzero dihedral angle
POS_4 = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [2.5, 1.2, 0.0],
        [3.5, 1.2, 0.9],
    ]
)

# H2O2-like geometry (4 atoms, non-planar)
POS_H2O2 = np.array(
    [
        [-0.705, 0.142, 0.10],
        [0.705, -0.142, 0.00],
        [-1.051, -0.759, 0.40],
        [1.051, 0.759, -0.40],
    ]
)


# ---------------------------------------------------------------------------
# QForce gradient tests
# ---------------------------------------------------------------------------


class TestQForceGradients:
    """Verify that QForce analytical forces match central finite differences.

    Term dict format (from Topology.set_terms):
        { term_type: { "atoms": np.ndarray (n_terms, n_cols),
                       "kwargs": { param: np.ndarray (n_terms,) } } }

    Positions are in Angstrom; QForce converts to nm internally.
    Returned forces are in eV/Å.
    """

    qf = QForce()

    def _check(self, pos, term_dict, atol=1e-3, rtol=1e-3):
        def energy_fn(p):
            return self.qf(p, PBC, CELL, term_dict)[0]

        _, f_analytical = self.qf(pos, PBC, CELL, term_dict)
        f_fd = finite_difference_forces(energy_fn, pos)
        np.testing.assert_allclose(
            f_analytical,
            f_fd,
            atol=atol,
            rtol=rtol,
            err_msg=f"Force mismatch for term type: {list(term_dict)}",
        )

    def test_bond(self):
        """Morse bond potential between two atoms (the default form)."""
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0])
        self._check(POS_2, td)

    @pytest.mark.parametrize("c", [1.0, 4.0, -1.5])
    def test_bond_morse_shape(self, c):
        """The Hulburt-Hirschfelder term `c` on the stretched branch.

        `POS_2` sits near `r0`, which is exactly where the correction and its
        first two derivatives vanish -- a gradient checked only there would pass
        against an implementation that computed the term wrongly, or not at all.
        So this walks out along the stretch, through the peak of `s**3 exp(-2s)`
        at `s = 1.5` and past it, and back onto the compressed branch where the
        term is clamped off.
        """
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0], c=[c])
        al = np.sqrt(251200.0 / (2 * 436.0))
        for s in (-1.0, 0.0, 0.5, 1.5, 3.0, 6.0):
            pos = np.array([[0.0, 0.0, 0.0], [0.07772 + s / al, 0.0, 0.0]])
            self._check(pos, td)

    def test_bond_morse_shape_is_off_by_default(self):
        """A term file with no `c` must read as exactly the old Morse."""
        kwargs = dict(r0=[0.07772], k=[251200.0], D=[436.0])
        plain = make_term("bond", [[0, 1]], **kwargs)
        zeroed = make_term("bond", [[0, 1]], c=[0.0], **kwargs)
        qf = QForce()
        for offset in (-0.02, 0.0, 0.05, 0.2):
            pos = np.array([[0.0, 0.0, 0.0], [0.07772 + offset, 0.0, 0.0]])
            assert qf(pos, PBC, CELL, plain)[0] == pytest.approx(
                qf(pos, PBC, CELL, zeroed)[0], abs=1e-12
            )

    def test_bond_harmonic(self):
        """Harmonic bond potential, the alternative form selected on QForce."""
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0])
        qf = QForce(bond_form="harmonic")

        def energy_fn(p):
            return qf(p, PBC, CELL, td)[0]

        _, f_analytical = qf(POS_2, PBC, CELL, td)
        np.testing.assert_allclose(
            f_analytical,
            finite_difference_forces(energy_fn, POS_2),
            atol=1e-3,
            rtol=1e-3,
            err_msg="Force mismatch for harmonic bond",
        )

    def test_bond_forms_differ(self):
        """Guard against the two bond forms silently being the same function.

        Morse must be bounded by its dissociation asymptote where harmonic is
        not; that difference is the whole reason the reactive path needs Morse.
        """
        td = make_term("bond", [[0, 1]], r0=[0.07772], k=[251200.0], D=[436.0])
        stretched = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        e_morse = QForce(bond_form="morse")(stretched, PBC, CELL, td)[0]
        e_harm = QForce(bond_form="harmonic")(stretched, PBC, CELL, td)[0]
        assert e_morse < e_harm, (
            f"Morse ({e_morse:.3f} eV) should saturate well below harmonic "
            f"({e_harm:.3f} eV) at a badly stretched bond"
        )
        # Morse is bounded above by its asymptote, which sits at +D over the well.
        assert e_morse <= 0.0

    def test_reference(self):
        """Constant per-molecule reference shift contributes energy but no force."""
        td = make_term("reference", [[0]], E0=[-1234.5])
        energy, forces = self.qf(POS_3, PBC, CELL, td)
        assert energy == pytest.approx(-1234.5 * (units.kJ / units.mol))
        np.testing.assert_allclose(forces, np.zeros_like(POS_3), atol=0.0)

    def test_angle(self):
        """Harmonic angle (in cosine)."""
        td = make_term("angle", [[0, 1, 2]], theta0=[1.911], k=[500.0])
        self._check(POS_3, td)

    def test_bondbond(self):
        """Bond-bond cross term (atoms: a0-a1 and a2-a3).
        Atom indices can be reused: bond1=(1,0), bond2=(2,0) sharing atom 0.
        Equilibrium values r1_0, r2_0 are shifted from the actual bond lengths
        to ensure nonzero forces.
        """
        td = make_term(
            "bondbond", [[1, 0, 2, 0]], r1_0=[0.085], r2_0=[0.120], k=[100.0]
        )
        self._check(POS_3, td)

    def test_bondangle(self):
        """Bond-angle cross term: angle (a0-a1-a2) × bond (a3-a4)."""
        td = make_term(
            "bondangle", [[0, 1, 2, 0, 1]], theta0=[1.911], r0=[0.100], k=[500.0]
        )
        self._check(POS_3, td)

    def test_angleangle(self):
        """Angle-angle cross term: angle1 (a0-a1-a2) × angle2 (a3-a4-a5)."""
        td = make_term(
            "angleangle",
            [[2, 0, 1, 0, 1, 3]],
            theta1_0=[1.782],
            theta2_0=[1.782],
            k=[43.0],
        )
        self._check(POS_H2O2, td)

    def test_periodicdihedral(self):
        """Periodic dihedral: k * (1 + cos(n*phi - phi0))."""
        td = make_term(
            "periodicdihedral", [[0, 1, 2, 3]], phi0=[3.14159], n=[2.0], k=[1.0]
        )
        self._check(POS_4, td)

    def test_dihedralbond(self):
        """Dihedral-bond cross term: dihedral (a0-a3) × bond (a4-a5)."""
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
        """Dihedral-angle cross term: dihedral (a0-a3) × angle (a4-a5-a6)."""
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
        """Dihedral-angle-angle cross term sharing 4 atoms.

        Dihedral uses atoms (0,1,2,3); angle1 uses (0,1,2); angle2 uses (1,2,3).
        """
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


# ---------------------------------------------------------------------------
# ACKS2 gradient tests
# ---------------------------------------------------------------------------


class TestACKS2Gradients:
    """Verify ACKS2 forces against central finite differences.

    Two different things are checked here and the distinction matters.
    `test_coulomb_forces` freezes Q on both sides, so it tests only the Coulomb
    force formula in isolation -- by construction it cannot see whether the
    charge response is handled, and for a long time it passed while
    `ACKS2.__call__` was not conservative at all.  `test_call_forces` is the one
    that covers the real calculator: it perturbs the geometry and lets the
    charges re-solve, exactly as they do along a trajectory.
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

    def test_coulomb_forces(self):
        """Test compute_coulomb gradients with frozen charges.

        Charges Q are solved once at the reference geometry and held fixed for
        both the analytical forces and the finite-difference energy perturbations.
        This isolates the Coulomb force formula from the charge-response (dQ/dr)
        contribution, which `test_call_forces` covers instead.
        """
        indices = self._TERM_DICT["atom"]["atoms"][:, 0]
        params = self._TERM_DICT["atom"]["kwargs"]
        sub = np.ix_(indices, indices)

        # Solve charges at reference geometry
        vecs_ref = (POS_H2O2[:, None, :] - POS_H2O2[None, :, :])[sub]
        rij_ref = np.sqrt(np.sum(vecs_ref * vecs_ref, -1))
        Q = self.acks2.compute_charges(rij_ref, params)

        def energy_fn(p):
            v = (p[:, None, :] - p[None, :, :])[sub]
            r = np.sqrt(np.sum(v * v, -1))
            e, _ = self.acks2.compute_coulomb(Q, r, v)
            return e

        _, f_analytical = self.acks2.compute_coulomb(Q, rij_ref, vecs_ref)
        f_fd = finite_difference_forces(energy_fn, POS_H2O2)
        np.testing.assert_allclose(f_analytical, f_fd, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "positions", [POS_2, POS_3, POS_4, POS_H2O2], ids=["n2", "n3", "n4", "h2o2"]
    )
    def test_call_forces(self, positions):
        """The full calculator must be conservative, charge response included.

        Unlike `test_coulomb_forces`, the charges are *not* frozen: every
        finite-difference displacement re-solves them, which is what happens
        between MD steps.  The gap between the two is the dQ/dr term, and it is
        large -- dropping it moves the H2O force by ~1 eV/A and makes NVE energy
        drift by hundreds of percent (see test_energy_conservation.py).
        """
        params = self._TERM_DICT["atom"]["kwargs"]
        term_dict = {
            "atom": {
                "atoms": np.array([[i] for i in range(len(positions))]),
                "kwargs": {k: v[: len(positions)] for k, v in params.items()},
            }
        }
        acks2 = ACKS2()

        def energy_fn(p):
            # A fresh instance per call: the cache keys on positions, and
            # reusing one here would be testing the cache, not the gradient.
            return ACKS2()(p, PBC, CELL, term_dict)[0]

        _, f_analytical = acks2(positions, PBC, CELL, term_dict)
        f_fd = finite_difference_forces(energy_fn, positions)
        np.testing.assert_allclose(f_analytical, f_fd, atol=1e-6, rtol=1e-5)

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_invariant_under_atom_relabeling(self, seed):
        """Relabeling atoms must not change the energy, only permute the forces.

        The per-atom parameters arrive in term order while positions and forces
        are in global order.  When the two coincide -- as in every other test
        here -- an index-space mix-up is invisible, so this drives them apart on
        purpose.  Both the parameters and the term's atom indices are permuted
        together, so the physical system is identical throughout.
        """
        permutation = np.random.default_rng(seed).permutation(len(POS_H2O2))
        params = self._TERM_DICT["atom"]["kwargs"]

        reference_e, reference_f = ACKS2()(POS_H2O2, PBC, CELL, self._TERM_DICT)

        # Same molecule, atoms listed in a different order.
        permuted_term_dict = {
            "atom": {
                "atoms": np.array([[i] for i in permutation]),
                "kwargs": {k: v[permutation] for k, v in params.items()},
            }
        }
        energy, forces = ACKS2()(POS_H2O2, PBC, CELL, permuted_term_dict)

        assert energy == pytest.approx(reference_e, abs=1e-10)
        np.testing.assert_allclose(forces, reference_f, atol=1e-10)


# ---------------------------------------------------------------------------
# ZBL gradient tests
# ---------------------------------------------------------------------------


class TestZBLGradients:
    """The screened-nuclear repulsion that opposes ACKS2's contact funnel.

    Checked across the whole range it is asked to work over, because it is asked
    to work over an unusually wide one: it must be a hard wall at 0.3 A, a few eV
    at bond lengths, and a fraction of that in the van der Waals region, and the
    screening function is a sum of four exponentials whose derivative is easy to
    get subtly wrong in a way that only shows up at one end.

    `ZBL` is the only force field here that does not take a `term_dict`; it reads
    atomic numbers straight off the `Atoms`.  That is deliberate -- see
    `forcefield/zbl.py` -- so these tests pass `numbers` directly and there is no
    term-order/global-order case to check.
    """

    zbl = ZBL()

    @pytest.mark.parametrize("z1, z2", [(1, 1), (8, 1), (8, 8)], ids=["hh", "oh", "oo"])
    @pytest.mark.parametrize(
        "r",
        [0.3, 0.6, 0.777, 0.96, 1.21, 3.0],
        ids=["core", "fusion", "hh_bond", "oh_bond", "oo_bond", "vdw"],
    )
    def test_pair_potential(self, z1, z2, r):
        """`du/dr` against central differences, decade by decade."""
        r = np.array([float(r)])
        z_1, z_2 = np.array([float(z1)]), np.array([float(z2)])
        _, analytic = zbl_module.pair_potential(r, z_1, z_2)

        h = 1e-7
        numeric = (
            zbl_module.pair_potential(r + h, z_1, z_2)[0]
            - zbl_module.pair_potential(r - h, z_1, z_2)[0]
        ) / (2 * h)
        assert analytic == pytest.approx(numeric, rel=1e-5)

    @pytest.mark.parametrize(
        "positions", [POS_2, POS_3, POS_4, POS_H2O2], ids=["n2", "n3", "n4", "h2o2"]
    )
    def test_all_pairs(self, positions):
        numbers = np.array([8 if i % 2 == 0 else 1 for i in range(len(positions))])

        def energy_fn(p):
            return self.zbl(p, numbers, PBC, CELL)[0]

        _, f_analytical = self.zbl(positions, numbers, PBC, CELL)
        f_numerical = finite_difference_forces(energy_fn, positions)
        np.testing.assert_allclose(f_analytical, f_numerical, atol=1e-5, rtol=1e-5)

    def test_forces_under_periodic_boundaries(self):
        """The minimum-image path has its own branch and its own gradient."""
        rng = np.random.default_rng(11)
        positions = rng.uniform(0.0, 6.0, (10, 3))
        numbers = rng.choice([1, 8], size=10)
        cell = np.eye(3) * 6.0
        pbc = np.ones(3, dtype=bool)

        def energy_fn(p):
            return self.zbl(p, numbers, pbc, cell)[0]

        _, f_analytical = self.zbl(positions, numbers, pbc, cell)
        f_numerical = finite_difference_forces(energy_fn, positions, delta=1e-5)
        np.testing.assert_allclose(f_analytical, f_numerical, atol=1e-4, rtol=1e-4)

    def test_it_is_repulsive_and_monotone_everywhere(self):
        """Positive, decreasing, and with a restoring force at every separation.

        A capped or tapered wall satisfies the first and fails the third exactly
        where it matters -- the force goes to zero inside the plateau and atoms
        drift through it.  This is the property that makes the term work as a
        guard rather than merely as a large number.
        """
        r = np.geomspace(0.05, 8.0, 400)
        for z1, z2 in [(1.0, 1.0), (8.0, 1.0), (8.0, 8.0)]:
            u, du_dr = zbl_module.pair_potential(
                r, np.full_like(r, z1), np.full_like(r, z2)
            )
            assert np.all(u > 0.0)
            assert np.all(np.diff(u) < 0.0)
            assert np.all(du_dr < 0.0)

    def test_it_beats_the_acks2_funnel_at_contact(self):
        """The whole point of the term, as a number.

        ACKS2 pulls an O-H pair downhill monotonically to -4.03 eV at contact
        with no repulsive branch of its own; a 200-atom box run against it alone
        reached 0.60 A intermolecular contacts.  At that separation this term has
        to be worth substantially more than kT = 0.26 eV against it.
        """
        r = np.array([0.6])
        u = zbl_module.pair_potential(r, np.array([8.0]), np.array([1.0]))[0]
        assert u[0] > 15.0  # measured: 20.9 eV, against ACKS2's -2.19 eV there


# ---------------------------------------------------------------------------
# EVBCoupling gradient tests
# ---------------------------------------------------------------------------


class TestLennardJonesGradients:
    """The Pauli/dispersion term, in both of the forms it is evaluated in.

    It is computed twice by design -- once over all pairs by `LennardJones` and
    once per near-neighbour pair by `QForce.compute_exclusion`, which subtracts
    it again -- and the two must be the same function of the geometry to the
    last bit or an isolated template stops reproducing its own energy.  So each
    is checked against finite differences here, and `test_the_two_forms_cancel`
    checks them against each other.

    `POS_2` sits at 0.856 A, well inside sigma, so these run on the steep
    repulsive branch where a sign error in `du/dr` cannot hide.
    """

    lj = LennardJones()
    qf = QForce()

    # q-force units: sigma in nm, eps in kJ/mol.  O and the hydrogen values
    # derived from its van der Waals radius, as the dataset carries them.
    SIGMA = [0.296, 0.196]
    EPS = [0.71128, 0.184]

    def _atom_terms(self, n):
        return make_term(
            "lennardjones",
            [[i] for i in range(n)],
            sigma=[self.SIGMA[i % 2] for i in range(n)],
            eps=[self.EPS[i % 2] for i in range(n)],
        )

    @pytest.mark.parametrize(
        "positions", [POS_2, POS_3, POS_4, POS_H2O2], ids=["n2", "n3", "n4", "h2o2"]
    )
    def test_all_pairs(self, positions):
        td = self._atom_terms(len(positions))

        def energy_fn(p):
            return self.lj(p, PBC, CELL, td)[0]

        _, f_analytical = self.lj(positions, PBC, CELL, td)
        f_fd = finite_difference_forces(energy_fn, positions)
        np.testing.assert_allclose(f_analytical, f_fd, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("positions", [POS_2, POS_3, POS_4], ids=["n2", "n3", "n4"])
    def test_exclusion(self, positions):
        """The negative pair term `QForce` subtracts for intramolecular pairs."""
        n = len(positions)
        rows, sigma, eps = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                rows.append([i, j])
                sigma.append(np.sqrt(self.SIGMA[i % 2] * self.SIGMA[j % 2]))
                eps.append(np.sqrt(self.EPS[i % 2] * self.EPS[j % 2]))
        td = make_term("exclusion", rows, sigma=sigma, eps=eps)

        def energy_fn(p):
            return self.qf(p, PBC, CELL, td)[0]

        _, f_analytical = self.qf(positions, PBC, CELL, td)
        f_fd = finite_difference_forces(energy_fn, positions)
        np.testing.assert_allclose(f_analytical, f_fd, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize(
        "positions", [POS_2, POS_3, POS_4, POS_H2O2], ids=["n2", "n3", "n4", "h2o2"]
    )
    def test_the_two_forms_cancel(self, positions):
        """One small molecule: every pair is a near neighbour, so the sum is zero.

        This is the identity the whole decomposition rests on -- the global sum
        minus the per-molecule exclusions -- and it is asserted to machine
        precision rather than to a tolerance, because nothing here is an
        approximation: the same `pair_potential` is evaluated twice with
        opposite signs.
        """
        n = len(positions)
        rows, sigma, eps = [], [], []
        for i in range(n):
            for j in range(i + 1, n):
                rows.append([i, j])
                sigma.append(np.sqrt(self.SIGMA[i % 2] * self.SIGMA[j % 2]))
                eps.append(np.sqrt(self.EPS[i % 2] * self.EPS[j % 2]))

        global_energy, global_forces = self.lj(
            positions, PBC, CELL, self._atom_terms(n)
        )
        exclusion_energy, exclusion_forces = self.qf(
            positions, PBC, CELL, make_term("exclusion", rows, sigma=sigma, eps=eps)
        )
        assert abs(global_energy + exclusion_energy) < 1e-9 * max(
            1.0, abs(global_energy)
        ), f"{global_energy} does not cancel {exclusion_energy}"
        np.testing.assert_allclose(
            global_forces, -exclusion_forces, atol=1e-9, rtol=1e-9
        )

    @pytest.mark.parametrize("separation", [0.744, 0.3, 0.2, 0.024])
    def test_a_compressed_bond_stays_within_precision(self, separation):
        """The decomposition must survive a bond being crushed.

        `E = sum_all_pairs - sum_near_pairs` evaluates every *bonded*
        pair in both sums, at a separation where plain 12-6 is enormous.  At
        equilibrium that is merely ugly.  On a hot trajectory it destroyed the
        3000 K probe: an H2 bond compressed towards contact put 1e8 eV into both
        sums by step 130, 1e10 by step 140, and 1e21 by step 143 -- at which
        point their difference, the physical energy of order 1e2, came back
        quantized to 2**22 eV, the forces went with it, and the box heated to
        1e16 K.  The potential energy alone looked fine the whole way down,
        which is why this is asserted on the magnitude rather than on the total.

        `pair_potential`'s linear continuation is what bounds it.  Both halves
        go through that function, so the cancellation stays exact as well --
        which is the other half of the assertion, and the half that broke when
        `compute_exclusion` still open-coded the form.
        """
        pos = np.array([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]])
        sigma = np.sqrt(self.SIGMA[0] * self.SIGMA[1])
        eps = np.sqrt(self.EPS[0] * self.EPS[1])

        total, forces = self.lj(pos, PBC, CELL, self._atom_terms(2))
        cancel, cancel_forces = self.qf(
            pos,
            PBC,
            CELL,
            make_term("exclusion", [[0, 1]], sigma=[sigma], eps=[eps]),
        )

        # Bounded: unlinearized 12-6 reaches 8e7 eV at 0.3 A and 1e21 at 0.024.
        assert abs(total) < 1e5, (
            f"a bond at {separation} A contributes {total:.3e} eV to the "
            "whole-system sum; the difference of two such numbers has no "
            "precision left and the forces derived from it are noise"
        )
        assert abs(total + cancel) < 1e-6, (
            f"at {separation} A the global sum ({total:.6e}) and the exclusion "
            f"({cancel:.6e}) no longer cancel"
        )
        np.testing.assert_allclose(forces, -cancel_forces, atol=1e-6, rtol=1e-9)


@pytest.fixture(scope="module")
def reaction_set():
    from DynamicTopology.core import ReactionSet

    return ReactionSet(RSET_PATH)


class TestEVBCouplingGradients:
    """Verify EVBCoupling RMSD forces against central finite differences.

    Note: The coupling forces are derived holding the optimal rigid-body
    alignment (R, T, S from Superpose3D) fixed at the reference geometry.
    Finite differences implicitly re-optimize the alignment at each perturbed
    geometry.  A failing test may indicate either an approximation in the
    analytical gradient or a bug in the drmsd formula.
    """

    coupling = EVBCoupling()

    def test_rmsd(self):
        """RMSD-Gaussian coupling: A * exp(-a * rmsd^2)."""
        pos = POS_3.copy()
        # Reference ensemble displaced enough that rmsd > 0 (avoids rmsd → 0 singularity)
        ref = pos + np.array(
            [
                [0.40, 0.10, 0.20],
                [-0.30, 0.20, -0.10],
                [0.20, -0.40, 0.30],
            ]
        )
        ensemble = ref[np.newaxis]  # shape (1, n_atoms, 3)
        td = make_term("rmsd", [[0, 1, 2]], A=[-10.0], a=[10.0])

        def energy_fn(p):
            return self.coupling(p, PBC, CELL, ensemble, td)[0]

        _, f_analytical = self.coupling(pos, PBC, CELL, ensemble, td)
        f_fd = finite_difference_forces(energy_fn, pos)
        np.testing.assert_allclose(f_analytical, f_fd, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# The assembled surface
# ---------------------------------------------------------------------------

RSET_PATH = "datasets/HCombustion/HCombustion.json"
SYSTEM_CELL = 60.0  # cubic box edge, Angstrom; large enough that nothing wraps


class TestSystemGradients:
    """Finite differences through the whole force call, not one term at a time.

    Everything above tests a `compute_*` method in isolation, which leaves the
    assembly untested: the EVB ground state, the Hellmann-Feynman contraction,
    and -- the reason this class exists -- the switching weight `EVBBasis` puts
    on a coupling as a state enters the basis.  That weight depends on the
    geometry through the diabatic gap, so it carries a force of its own, and
    nothing else here would notice if it were dropped.
    """

    @pytest.fixture(scope="class")
    def reaction_set(self):
        from DynamicTopology.core import ReactionSet

        return ReactionSet(RSET_PATH)

    def _calculate(self, atoms, reaction_set):
        from DynamicTopology.core import Topology
        from DynamicTopology.system import System

        return System(atoms, Topology.from_atoms(atoms), reaction_set).calculate()

    def test_forces_inside_the_admission_ramp(self, reaction_set):
        """F = -dE/dr with a channel switching on.

        The tolerance is set by what the missing term actually costs: with
        `_channel_weight` stubbed to return no gradient the error is 6.0e-03,
        while the correct forces agree to 1.7e-08.  Anything in between would
        make this pass on a broken implementation.
        """
        atoms = reaction_path(REACTION, REACTION_PATH_RAMP, SYSTEM_CELL)
        results = self._calculate(atoms, reaction_set)

        weight = min(block["min_switch"] for block in results["blocks"])
        assert 0.0 < weight < 1.0, (
            f"no channel is inside the admission ramp (min_switch = {weight}); "
            "the switch's gradient is not being exercised and this test is "
            "vacuous"
        )

        def energy_fn(positions):
            perturbed = atoms.copy()
            perturbed.positions = positions
            return self._calculate(perturbed, reaction_set)["energy"]

        f_fd = finite_difference_forces(energy_fn, atoms.positions, delta=1e-5)
        np.testing.assert_allclose(results["forces"], f_fd, atol=1e-7, rtol=1e-5)

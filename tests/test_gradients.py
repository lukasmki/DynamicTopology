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
    """Verify ACKS2 Coulomb forces against central finite differences.

    Note: ACKS2 forces are computed under a frozen-charge approximation.
    Charges Q are re-solved at each MD step, but the analytical gradient
    treats Q as fixed (dQ/dr = 0).  Finite differences re-solve Q at each
    perturbed geometry, so they include the charge-response contribution.
    A failing test indicates that the charge-response term is non-negligible
    for the chosen geometry and parameters.
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
        contribution that would appear if charges were re-solved at each step.
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

        reference_e, reference_f = ACKS2()(
            POS_H2O2, PBC, CELL, self._TERM_DICT
        )

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
# EVBCoupling gradient tests
# ---------------------------------------------------------------------------


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

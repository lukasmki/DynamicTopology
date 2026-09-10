"""The periodic charge kernel: is it summing the right lattice sum?

`test_gradients.py` and `test_stress.py` check the Ewald kernel the way they
check everything else -- analytic derivatives against finite differences of the
same code -- and that is exactly the check a wrong lattice sum survives.  A
kernel that converges beautifully to the wrong number differentiates
beautifully too.  So this file pins the *value*:

  * against the NaCl Madelung constant, which is known to arbitrary precision
    and which no truncated pair sum reproduces (`MinimumImage` gets 1.456
    against 1.7476, off by 17%),
  * against itself under a changed splitting parameter, which the exact sum is
    independent of and a mis-split one is not, and
  * against the open-boundary kernel in the large-cell limit, where the leading
    difference is the interaction with the periodic images of the molecule's own
    dipole and must fall off as 1/L^3.

The derivative tests here are narrower than the ones in `test_gradients.py`:
they contract the kernel against a fixed weight matrix rather than going through
ACKS2, which isolates the reciprocal-space force and virial from the charge
response.  The reciprocal virial is the piece with no pair-sum analogue -- it
comes from `k -> (I - e) k`, the reciprocal vectors straining *inversely* to the
positions -- so it is the one term in this module that a `v_a v_b` contraction
lifted from the real-space code would get wrong while still looking plausible.
"""

import numpy as np
import pytest

from DynamicTopology.forcefield.ewald import Ewald, MinimumImage

from test_gradients import finite_difference_forces
from test_stress import finite_difference_virial


# https://en.wikipedia.org/wiki/Madelung_constant, NaCl structure.
MADELUNG_NACL = 1.7475645946331821

# Nearest-neighbour distance in the NaCl test crystal.  Any value comfortably
# above ~1.5 A works: the kernel is erf(2 r)/r, and erf(2 * 2.82) differs from 1
# by 1e-16, so at these separations the screened kernel *is* the bare Coulomb
# one and the Madelung constant is a legitimate reference for it.
NACL_D = 2.82


def nacl():
    """The 8-ion conventional cell of rock salt, and its minimum-image geometry."""
    idx = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    pos = idx * NACL_D
    q = np.where(idx.sum(1) % 2 == 0, 1.0, -1.0)
    cell = np.eye(3) * (2 * NACL_D)
    return pos, q, cell


def geometry(pos, cell, pbc=True):
    """Minimum-image displacements and distances, as `ACKS2.__call__` builds them."""
    vecs = pos[:, None, :] - pos[None, :, :]
    if pbc:
        f = vecs @ np.linalg.inv(cell)
        vecs = vecs - np.floor(f + 0.5) @ cell
    return vecs, np.sqrt(np.sum(vecs * vecs, -1))


def madelung(K, q):
    """The Madelung constant implied by a kernel matrix.

    `E = 1/2 sum_ij q_i q_j K_ij` is the energy of the whole cell, so it counts
    every interaction twice over the `N` ions of the standard per-ion
    definition `E_i = -M / d`; hence the factor of two.
    """
    return -2 * (0.5 * q @ K @ q) * NACL_D / len(q)


class TestMadelung:
    """The value test.  Everything else in this file is self-consistency."""

    @pytest.mark.parametrize("accuracy", [1e-6, 1e-8, 1e-10])
    def test_rock_salt(self, accuracy):
        pos, q, cell = nacl()
        vecs, rij = geometry(pos, cell)
        K = Ewald(cell, accuracy=accuracy).bind(pos, vecs, rij).matrix()
        assert madelung(K, q) == pytest.approx(MADELUNG_NACL, abs=10 * accuracy)

    def test_minimum_image_does_not_get_it(self):
        """Guard the guard: the reference must be one the old kernel fails.

        Without this, `test_rock_salt` could be passing on a lattice sum that
        never left the unit cell.  It is off by 17%, which is the size of the
        thing this module was added to compute.
        """
        pos, q, cell = nacl()
        vecs, rij = geometry(pos, cell)
        assert madelung(MinimumImage(rij, vecs).matrix(), q) == pytest.approx(
            1.4560, abs=1e-3
        )

    def test_self_image_term_is_what_makes_it_work(self):
        """The diagonal is not decoration.

        `K_ii` -- an ion's interaction with its own images, which has no
        counterpart in the open-boundary kernel and which `ACKS2.build_system`
        has to add to the hardness rather than overwrite -- carries 80% of the
        rock-salt energy: without it the Madelung constant reads 0.350 against
        1.7476.  Dropping it leaves a sum that converges perfectly to that.
        """
        pos, q, cell = nacl()
        vecs, rij = geometry(pos, cell)
        K = Ewald(cell).bind(pos, vecs, rij).matrix()
        without = K.copy()
        without[np.diag_indices_from(without)] = 0.0
        assert madelung(without, q) == pytest.approx(0.3502, abs=1e-3)


class TestConvergence:
    def test_independent_of_the_splitting(self):
        """`kappa` is a computational parameter, not a physical one.

        The exact sum does not depend on where the real/reciprocal split falls,
        so the only thing separating two accuracies is the truncation each one
        was asked for.  A term dropped from one half but not restored in the
        other -- the classic Ewald bug -- shows up here as a `kappa`-dependent
        energy, and nowhere else.
        """
        rng = np.random.default_rng(0)
        cell = np.eye(3) * 9.0
        pos = rng.uniform(0, 9.0, size=(5, 3))
        q = rng.normal(size=5)
        q -= q.mean()
        vecs, rij = geometry(pos, cell)

        energies = []
        for accuracy in (1e-6, 1e-8, 1e-10, 1e-12):
            setup = Ewald(cell, accuracy=accuracy)
            energies.append(0.5 * q @ setup.bind(pos, vecs, rij).matrix() @ q)
        # The splittings genuinely differ -- 0.83 to 1.17 -- so the agreement
        # below is not two runs of the same sum.
        assert Ewald(cell, accuracy=1e-6).kappa != Ewald(cell, accuracy=1e-12).kappa
        np.testing.assert_allclose(energies, energies[-1], atol=1e-7)

    @pytest.mark.parametrize("size", [20.0, 40.0, 80.0])
    def test_approaches_the_open_boundary_kernel(self, size):
        """A neutral molecule alone in a big cell is nearly in vacuum.

        Nearly, not exactly: what is left is the molecule interacting with the
        images of its own dipole, which decays as 1/L^3.  Asserting the *rate*
        rather than just smallness is what distinguishes a correct periodic sum
        from one that is merely converging to the vacuum answer -- a kernel that
        forgot the images entirely would pass a smallness test perfectly.
        """
        pos = np.array([[0.0, 0.0, 0.0], [0.97, 0.0, 0.0]])
        q = np.array([0.4, -0.4])
        cell = np.eye(3) * size
        vecs, rij = geometry(pos, cell)

        periodic = (
            0.5 * q @ Ewald(cell, accuracy=1e-10).bind(pos, vecs, rij).matrix() @ q
        )
        vacuum = 0.5 * q @ MinimumImage(rij, vecs).matrix() @ q
        # 3.16e-2 eV*A/q^2 at 20 A, falling by 8 for every doubling.
        assert (periodic - vacuum) * size**3 == pytest.approx(-0.3164, rel=2e-2)


class TestKernelDerivatives:
    """`contract` against finite differences, with no ACKS2 in the way.

    The weight matrix stands in for the two ACKS2 builds -- `q_i q_j` for the
    energy and `lam_i q_j + lam_j q_i` for the charge response.  Both are built
    from vectors that sum to zero, and so is this one: the `k = 0` term of the
    lattice sum is dropped on exactly that assumption (see `ewald.py`), so a
    weight without it would be testing a sum the code does not claim to compute.
    """

    CELL = np.eye(3) * 7.0

    def system(self):
        rng = np.random.default_rng(3)
        pos = rng.uniform(0.0, 7.0, size=(6, 3))
        a = rng.normal(size=6)
        b = rng.normal(size=6)
        a -= a.mean()
        b -= b.mean()
        W = 0.5 * (np.outer(a, b) + np.outer(b, a))
        return pos, W

    def scalar(self, pos, cell, W):
        vecs, rij = geometry(pos, cell)
        return np.sum(W * Ewald(cell).bind(pos, vecs, rij).matrix())

    def test_forces(self):
        pos, W = self.system()
        vecs, rij = geometry(pos, self.CELL)
        dS_dr, _ = Ewald(self.CELL).bind(pos, vecs, rij).contract(W)

        fd = finite_difference_forces(lambda p: self.scalar(p, self.CELL, W), pos)
        np.testing.assert_allclose(-dS_dr, fd, atol=1e-6, rtol=1e-5)

    def test_virial(self):
        pos, W = self.system()
        vecs, rij = geometry(pos, self.CELL)
        _, dS_de = Ewald(self.CELL).bind(pos, vecs, rij).contract(W)

        fd = finite_difference_virial(lambda p, c: self.scalar(p, c, W), pos, self.CELL)
        np.testing.assert_allclose(dS_de, fd, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(dS_de, dS_de.T, atol=1e-12)

    def test_reciprocal_virial_is_not_negligible(self):
        """The real-space contraction alone must be visibly wrong.

        `test_virial` would pass on a copy of the pair-sum virial if the
        reciprocal part happened to be small here.  It is not: it is the term
        that carries the strain dependence of the reciprocal vectors, and it is
        comparable to the pair part.
        """
        pos, W = self.system()
        vecs, rij = geometry(pos, self.CELL)
        setup = Ewald(self.CELL)
        _, full = setup.bind(pos, vecs, rij).contract(W)
        _, pairs = MinimumImage(rij, vecs).contract(W)
        assert np.abs(full - pairs).max() > 0.05 * np.abs(full).max()


def test_kernel_is_symmetric():
    """`ACKS2` solves with `A` untransposed and reuses it for the adjoint; both
    steps assume the kernel it was built from is symmetric."""
    pos, q, cell = nacl()
    vecs, rij = geometry(pos, cell)
    K = Ewald(cell).bind(pos, vecs, rij).matrix()
    np.testing.assert_allclose(K, K.T, atol=1e-12)

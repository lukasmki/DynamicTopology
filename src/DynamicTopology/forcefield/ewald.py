"""The ACKS2 charge kernel, with and without the periodic lattice sum.

ACKS2 needs the smeared-charge kernel

    g(r) = erf(gamma * r) / r

in two places: as the Coulomb block of each diabatic state's linear system, and
in the derivative of the solved functional, `1/2 <dA/dr, Z>`.  Under periodic
boundary conditions both are a lattice sum over every image rather than a
nearest-image pair term, so this module puts the two behind one interface and
`acks2.py` never branches on `pbc`:

    kernel.matrix()    -> `K`, the (n, n) kernel matrix
    kernel.contract(W) -> `(dS/dr_i, dS/de_ab)` for `S = sum_ij W_ij K_ij`

`contract` also takes `factor`, an `(n, r)` array with `W = factor factor^T`,
when the caller has one: the reciprocal half then costs `n r` per k rather
than `n^2`, and ACKS2's `W` is the mean charges plus one column per state.

`contract` is what makes the abstraction worth having.  Every caller that
differentiates the kernel -- ACKS2 against the weight-averaged second moment of
its states' charges, `PointCharge` against its own, the admission gate against
the difference of two states -- is of exactly that form and differs only in the
symmetric weight matrix `W`, so the periodic derivatives are written once
instead of once per caller.  Under `MinimumImage`
that contraction is the pair sum the old inline code did; under `Ewald` it also
carries a reciprocal-space term that is *not* a sum over pair separations, and
which is why the kernel had to become an object rather than a matrix.

**The splitting.**  Writing `erf(a r) = 1 - erfc(a r)` twice turns the periodic
sum of `g` into a standard Ewald sum of `1/r` minus a short-ranged remainder,
and the two real-space pieces merge:

    K_ij = sum_n' [erf(gamma |r_ij + n|) - erf(kappa |r_ij + n|)] / |r_ij + n|
         + (4 pi / V) sum_{k != 0} exp(-k^2 / 4 kappa^2) / k^2 * cos(k . r_ij)
         - delta_ij * 2 kappa / sqrt(pi)

The prime excludes `n = 0` when `i == j`: an atom does not interact with itself,
but it *does* interact with its own images, and that is the diagonal `K_ii`
below -- a term with no counterpart in the open-boundary kernel, carried on the
diagonal of the ACKS2 matrix alongside the atomic hardness.

**The `k = 0` term is dropped and its neutralizing background put back.**  The
`k = 0` term of the reciprocal sum is the divergent one; regularizing it against
a neutralizing background leaves a constant `-pi / (V kappa^2)` in every entry
of `K`, carried as `background` and added to every entry.  Adding any constant
to every entry of `K` changes the energy by that constant times
`(sum_i q_i)^2` and every ACKS2 charge row by the same constant times
`sum_i q_i` -- which the molecules' charge multipliers absorb, so the solved
charges never see it.  For a neutral contraction it drops out of the energy, the
forces and the virial as well, and it was once omitted on those grounds.  It is
carried because not every contraction is neutral: a cell holding an H3O+ is
charged, and a point-charge exclusion or the admission gate's difference of two
states contracts `K` against a weight whose entries do not sum to zero -- each
needs the entries themselves, which without it drifted with `kappa`.

**Only the Coulomb block is summed over images.**  ACKS2's other
geometry-dependent block, the bond softness, decays as `exp(-r / tau)` with
`tau ~ 0.3 A`; at the nearest-image cutoff of a 9 A cell that is `e^-15`, so it
stays a nearest-image pair term and no periodic treatment is needed.
"""

import numpy as np
from scipy.linalg.blas import dsyrk
from scipy.sparse import csr_matrix
from scipy.special import erf

from DynamicTopology.forcefield.params import ForceFieldParams, resolve


# `GAMMA` and `ACCURACY` are now fields of `params.ForceFieldParams` --
# `gamma` and `accuracy` -- resolved in `MinimumImage`/`Ewald.__init__`
# (per force call and per cell respectively); see that module for why.


def _screened(rij, r, alpha):
    """`erf(alpha r) / r` and its radial derivative, on every pair at once.

    `r` is `rij` shifted off zero so the diagonal divides safely.  Both results
    are zeroed there explicitly: the kernel vanishes with its numerator anyway,
    but the derivative does not -- it tends to `2 alpha / sqrt(pi) / eps`, which
    is `4e15` rather than something a stray multiply would forgive.
    """
    diag = np.diag_indices_from(rij)
    k = erf(alpha * rij) / r
    dk = (2 * alpha / np.sqrt(np.pi)) * np.exp(-((alpha * rij) ** 2)) / r - k / r
    k[diag] = 0.0
    dk[diag] = 0.0
    return k, dk


def _weights(factor, signs=None):
    """`W = F diag(signs) F^T`, for a kernel that needs the weight itself."""
    signed = factor if signs is None else factor * np.asarray(signs)
    return signed @ factor.T


def contract_pairs(coeff, vecs, r):
    """`dS/dr_i` and `dS/de_ab` for a pair sum `S = sum_ij W_ij f(r_ij)`.

    `coeff` is `W * f'(r)`, already multiplied out by the caller, which is what
    lets the softness block in `acks2.py` share this with the Coulomb kernels.

    The factor of two on the force is the pair double count: entries `(i, j)`
    and `(j, i)` both move with `r_i`.  The virial has no such factor because
    `v_a v_b` is even under the swap and so has nothing to cancel against --
    the same asymmetry `acks2.py` documents at length, kept in one place here.
    """
    c = coeff / r
    dS_dr = 2.0 * np.einsum("ij,ija->ia", c, vecs)
    # `sum_ij c_ij v_a v_b` as one (3, n^2) x (n^2, 3) product: a fourth of the
    # time of the three-operand einsum it replaces.
    flat = vecs.reshape(-1, 3)
    dS_de = (c.reshape(-1, 1) * flat).T @ flat
    return dS_dr, dS_de


class MinimumImage:
    """`erf(gamma r) / r` over the nearest image of each pair and nothing else.

    The open-boundary kernel, and the fallback for a partially periodic cell:
    Ewald in fewer than three dimensions is a different summation altogether,
    so a slab keeps the behaviour it had rather than silently getting a 3D sum.

    `K_ii` is zero -- with no images there is nothing for an atom to interact
    with but the other atoms.
    """

    def __init__(self, rij, vecs=None, params: ForceFieldParams | None = None):
        self.rij = rij
        self.vecs = vecs
        self.r = rij + np.finfo(np.float64).eps
        self._derivative = None
        # Captured per kernel, which is per force call: the geometry this object
        # wraps and the smearing width it sums with are the same generation.
        self.gamma = resolve(params).gamma

    def matrix(self):
        # `_screened` forms the derivative on its way to the kernel; keeping it
        # spares `derivative` a second pass of `erf` and `exp` over every pair.
        k, dk = _screened(self.rij, self.r, self.gamma)
        if self._derivative is None:
            self._derivative = dk
        return k

    def derivative(self):
        """`dK_ij/dr_ij`, memoized for the same reason `matrix` is.

        More than one weight matrix can be contracted against one geometry --
        the electrostatics and the admission gate's gradient both do -- so the
        kernel's own derivative is the same array each time.
        """
        if self._derivative is None:
            self._derivative = _screened(self.rij, self.r, self.gamma)[1]
        return self._derivative

    def columns(self, index) -> np.ndarray:
        """`K[:, index]`."""
        return self.matrix()[:, index]

    def contract(self, W, factor=None, signs=None):
        if W is None:
            W = _weights(factor, signs)
        return contract_pairs(W * self.derivative(), self.vecs, self.r)


class Ewald:
    """Cell-dependent setup for the periodic kernel; `bind` attaches a geometry.

    Split in two because the expensive part -- choosing `kappa` and enumerating
    the reciprocal vectors -- depends only on the cell, while the trigonometric
    tables depend on the positions.  `ACKS2` keeps one of these per cell.

    **Why the reciprocal vectors are chosen by integer index and not by `|k|`.**
    The obvious selection, `|k| <= k_max`, is discontinuous in the cell: a
    strain of 1e-6 moves a whole degenerate shell of a cubic lattice across the
    cutoff at once, and the energy jumps.  Selecting an *integer* ellipsoid
    whose semi-axes come from `ceil(k_max / |b_i|)` instead makes the set
    piecewise constant in the cell, and the pieces meet where the vectors that
    join or leave weigh `accuracy` apiece.  The ellipsoid contains the sphere,
    so this only ever sums more than asked for.
    """

    def __init__(self, cell, accuracy=None, params: ForceFieldParams | None = None):
        ff = resolve(params)
        # `accuracy` stays an explicit argument because `tests/test_ewald.py`
        # sweeps it to show the sum is independent of the splitting; `None`
        # takes the dataset's.
        accuracy = ff.accuracy if accuracy is None else accuracy
        self.gamma = ff.gamma
        self.cell = np.asarray(cell, dtype=float)
        self.volume = abs(np.linalg.det(self.cell))
        if self.volume <= 0.0:
            raise ValueError("Ewald summation needs a cell with nonzero volume")

        # Rows of `recip` are the reciprocal lattice vectors b_j, so that
        # a_i . b_j = 2 pi delta_ij and k = m . recip for an integer triple m.
        recip = 2 * np.pi * np.linalg.inv(self.cell).T
        blen = np.linalg.norm(recip, axis=1)

        # The real-space sum is truncated at the nearest image, so the cutoff
        # is half the *perpendicular* width of the cell -- 2 pi / |b_i|, which
        # for a skewed cell is shorter than any lattice vector.
        span = np.sqrt(-np.log(accuracy))
        cutoff = 0.5 * (2 * np.pi / blen).min()

        # kappa is the smallest splitting that leaves the nearest image enough:
        # smallest, because every reciprocal vector is paid for on every atom
        # pair, so the reciprocal sum is the expensive half here.  Capping it at
        # `gamma` covers the cell so small that no splitting converges in one
        # image; there the real-space term vanishes identically and the whole
        # kernel is summed in reciprocal space, more slowly but correctly.
        self.kappa = min(span / cutoff, self.gamma)

        kmax = 2 * self.kappa * span
        nmax = np.maximum(np.ceil(kmax / blen).astype(int), 1)
        axes = [np.arange(-n, n + 1) for n in nmax]
        m = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

        # Half of reciprocal space, doubled in `weight` below: every quantity
        # here is even in k, so k and -k contribute identically.  The half-space
        # test (first nonzero component positive) drops k = 0 along with it.
        half = (m[:, 0] > 0) | (
            (m[:, 0] == 0) & ((m[:, 1] > 0) | ((m[:, 1] == 0) & (m[:, 2] > 0)))
        )
        inside = np.sum((m / nmax) ** 2, axis=1) <= 1.0
        self.kvecs = m[half & inside] @ recip
        self.k2 = np.sum(self.kvecs**2, axis=1)

        # 4 pi / (V k^2) is the Fourier transform of 1/r, the Gaussian is the
        # k-space image of the screening, and the 2 is the half-space fold.
        self.weight = (
            (8 * np.pi / self.volume) * np.exp(-self.k2 / (4 * self.kappa**2)) / self.k2
        )

        # Removes the k = 0 image of an atom's own screening charge, which the
        # reciprocal sum includes and which is not a real interaction.
        self.self_term = -2 * self.kappa / np.sqrt(np.pi)

        # The neutralizing background, which is what the dropped `k = 0` term
        # leaves behind.  A constant on *every* entry of `K`, diagonal included.
        #
        # **It contracts to zero against any neutral weight, and that is why it
        # was absent.**  `sum_ij q_i q_j c = c (sum_i q_i)**2`, so with the
        # sum-zero constraint the original ACKS2 imposed it changed no energy,
        # no force and no charge that code ever computed.  What it does
        # change is an individual `K_ij`, which without it is not a well-defined
        # number at all: it drifts with the splitting parameter, by 0.0054 on an
        # O-H pair between `accuracy` 1e-4 and 1e-12 while the neutral
        # contraction holds to ten digits.
        #
        # That became reachable when the intramolecular exclusion arrived.  The
        # screened Coulomb energy contracts the kernel against `S_ij q_i q_j`,
        # and a screen is not neutral -- the excluded pairs of a water carry
        # `sum_{excl} q_i q_j != 0` -- so the exclusion picked up whatever
        # `kappa` the cell happened to produce.  It showed up first in the
        # *virial*, at 0.5% of a 7-atom box, because `kappa` is a function of
        # the cell and a strain moves it; the energy was quietly wrong by the
        # same mechanism without moving.
        self.background = -np.pi / (self.kappa**2 * self.volume)

    def bind(self, pos, vecs, rij):
        return EwaldKernel(self, pos, vecs, rij)


class EwaldKernel:
    """The periodic kernel at one geometry.

    `pos` is in ACKS2's term order, the same order as `vecs` and `rij`; only
    differences of positions ever enter, and `cos(k . r)` is invariant under a
    lattice translation, so neither the origin nor whether an atom has been
    wrapped into the cell matters.
    """

    def __init__(self, setup, pos, vecs, rij):
        self.setup = setup
        self.vecs = vecs
        self.rij = rij
        self.r = rij + np.finfo(np.float64).eps
        self._matrix = None
        self._derivative = None

        # The structure factor, factorized: cos(k . r_ij) = c_i c_j + s_i s_j
        # turns every k-sum below into a matrix product over atoms, which is
        # what keeps the cost at one dgemm rather than n^2 trigonometric calls.
        kr = pos @ setup.kvecs.T
        self.cos = np.cos(kr)
        self.sin = np.sin(kr)

    def matrix(self):
        """`K_ij`, the kernel summed over every image.  Memoized: `ACKS2` asks
        for it once to build its linear system and once more for `dE/dQ`."""
        if self._matrix is None:
            setup = self.setup
            # Both screenings' derivatives come out of the same two passes, and
            # `contract` wants exactly their difference; keeping it here spares
            # it a second `erf` and `exp` over every pair.
            k_gamma, dk_gamma = _screened(self.rij, self.r, setup.gamma)
            k_kappa, dk_kappa = _screened(self.rij, self.r, setup.kappa)
            short = k_gamma - k_kappa
            if self._derivative is None:
                self._derivative = dk_gamma - dk_kappa
            del k_gamma, k_kappa, dk_gamma, dk_kappa
            # The reciprocal half is `A A^T` with `A = [c sqrt(w), s sqrt(w)]`:
            # one symmetric rank-k update, half the flops of the two products
            # it replaces.  `dsyrk` fills one triangle; `A.T` with `trans=1` is
            # the Fortran-ordered operand it wants, without a copy.
            root = np.sqrt(setup.weight)
            A = np.hstack([self.cos * root, self.sin * root])
            upper = dsyrk(1.0, A.T, trans=1)
            K = np.triu(upper)
            K += np.triu(upper, 1).T
            K += short
            del upper, short, A
            K[np.diag_indices_from(K)] += setup.self_term
            K += setup.background
            self._matrix = K
        return self._matrix

    def columns(self, index) -> np.ndarray:
        """`K[:, index]`."""
        return self.matrix()[:, index]

    def contract(self, W, factor=None, signs=None):
        """`dS/dr_i` and `dS/de_ab` for `S = sum_ij W_ij K_ij`, `W` symmetric.

        `factor`, if given, is `F` with `W = F diag(signs) F^T` (signs all `+1`
        by default), used for the reciprocal half; the real-space half is a
        pair sum and needs `W` itself, which is formed from `factor` if `W` is
        `None`.

        The reciprocal half is where this stops being a pair sum.  Its virial
        comes from the two ways a strain reaches it: the `1/V` prefactor, and
        the reciprocal vectors themselves, which transform *inversely* to the
        positions -- `k -> (I - e) k`, so `d(k^2)/de_ab = -2 k_a k_b`.  The
        structure factor is untouched, since `k . r` is invariant, which is why
        no analogue of the real-space `v_a v_b` term appears.
        """
        setup = self.setup
        if W is None:
            W = _weights(factor, signs)

        if self._derivative is None:
            # Geometry only, and `ACKS2` contracts the kernel twice per force
            # call -- once for the explicit Coulomb force and once for the
            # charge response -- against the same positions, so the real-space
            # derivative is memoized alongside `matrix`.
            self._derivative = (
                _screened(self.rij, self.r, setup.gamma)[1]
                - _screened(self.rij, self.r, setup.kappa)[1]
            )
        dS_dr, dS_de = contract_pairs(W * self._derivative, self.vecs, self.r)

        if factor is None:
            Wc = W @ self.cos
            Ws = W @ self.sin
        else:
            signed = factor if signs is None else factor * np.asarray(signs)
            Wc = signed @ (factor.T @ self.cos)
            Ws = signed @ (factor.T @ self.sin)
        # Per-k contribution to S, from  sum_ij W_ij cos(k . r_ij).
        Sk = setup.weight * (
            np.einsum("ik,ik->k", self.cos, Wc) + np.einsum("ik,ik->k", self.sin, Ws)
        )

        # d/dr_i sum_ij W_ij cos(k . r_ij) = -2 k [s_i (Wc)_i - c_i (Ws)_i]
        amp = self.sin * Wc - self.cos * Ws
        dS_dr = dS_dr - 2.0 * ((amp * setup.weight) @ setup.kvecs)

        coeff = 2.0 * Sk * (1.0 / (4 * setup.kappa**2) + 1.0 / setup.k2)
        dS_de = (
            dS_de
            + np.einsum("k,ka,kb->ab", coeff, setup.kvecs, setup.kvecs)
            - np.eye(3) * Sk.sum()
        )

        # The background reaches the strain through the volume alone -- it has
        # no position dependence, hence no `dS_dr` term.  `V -> V (1 + tr e)`,
        # so `d(-pi / (kappa**2 V))/de_ab = +pi / (kappa**2 V) delta_ab`, which
        # is `-background` on the diagonal.  `kappa` is a function of the cell
        # too and is deliberately not differentiated: the completed kernel is
        # independent of it to the reciprocal sum's own accuracy, which is the
        # property the background was added to restore.
        dS_de = dS_de - np.eye(3) * (setup.background * W.sum())
        return dS_dr, dS_de


class EwaldOperatorSetup:
    """The periodic kernel as an operator: a real-space cutoff and PME.

    `Ewald` chooses the smallest splitting the nearest image allows and sums
    the reciprocal half directly, which is right for the dense matrix the
    direct charge solve factors.  An iterative solve needs only products with
    the kernel, so the balance moves: the real-space half is cut off at
    `real_space_cutoff` (or half the cell, if that is shorter) and kept as a
    sparse matrix over the pairs inside it, and `kappa` is whatever that
    cutoff needs for `accuracy`; the reciprocal half, which then reaches
    further, goes to `pme.PME`.  Neither ever forms an `N x N` array.
    """

    def __init__(self, cell, params: ForceFieldParams | None = None):
        from DynamicTopology.forcefield.pme import PMESetup

        ff = resolve(params)
        self.gamma = ff.gamma
        self.cell = np.asarray(cell, dtype=float)
        self.volume = abs(np.linalg.det(self.cell))
        recip = 2 * np.pi * np.linalg.inv(self.cell).T
        half_width = 0.5 * (2 * np.pi / np.linalg.norm(recip, axis=1)).min()
        self.cutoff = min(ff.real_space_cutoff, half_width)
        span = np.sqrt(-np.log(ff.accuracy))
        self.kappa = min(span / self.cutoff, self.gamma)
        self.pme = PMESetup(self.cell, self.kappa, ff.accuracy)
        self.self_term = -2 * self.kappa / np.sqrt(np.pi)
        self.background = -np.pi / (self.kappa**2 * self.volume)
        # `K_rec_ii`, the same for every atom: the influence summed over the
        # whole mesh, B-spline correction removed.  For the preconditioner.
        self.diagonal = float(
            np.sum(self.pme.multiplicity * self.pme.G)
            + self.self_term
            + self.background
        )

    def bind(self, pos, geometry):
        """The operator at `pos`; `geometry` is a `neighbors.Geometry` of `pos`."""
        return EwaldOperator(self, pos, geometry)


class EwaldOperator:
    """The kernel at one geometry, as products rather than a matrix.

    The real-space half is read off `geometry`'s neighbour list inside
    `setup.cutoff` -- a cell list, not the `N x N` minimum image -- so building
    the operator is `O(N)` in the atoms and `O(N log N)` with the PME mesh.
    """

    def __init__(self, setup, pos, geometry):
        self.setup = setup
        self.geometry = geometry
        n = geometry.natoms
        self.n = n
        pairs = geometry.pairs(setup.cutoff)
        i, j, r = pairs.i, pairs.j, pairs.r
        k_gamma = erf(setup.gamma * r) / r
        k_kappa = erf(setup.kappa * r) / r
        self.i, self.j = i, j
        self.v, self.r = pairs.v, r
        self.short = k_gamma - k_kappa
        self.dshort = (
            (2 * setup.gamma / np.sqrt(np.pi)) * np.exp(-((setup.gamma * r) ** 2)) / r
            - k_gamma / r
        ) - (
            (2 * setup.kappa / np.sqrt(np.pi)) * np.exp(-((setup.kappa * r) ** 2)) / r
            - k_kappa / r
        )
        self.real = csr_matrix(
            (np.concatenate([self.short, self.short]), (np.concatenate([i, j]), np.concatenate([j, i]))),
            shape=(n, n),
        )
        self.pme = setup.pme.bind(pos)
        self._columns: dict[int, np.ndarray] = {}

    def matrix(self):
        raise RuntimeError(
            "the iterative charge solver's kernel is an operator; it has no matrix"
        )

    def matvec(self, X: np.ndarray) -> np.ndarray:
        """`K @ X`, for `X` a vector or `(n, r)` columns."""
        setup = self.setup
        out = self.real @ X + self.pme.potential(X) + setup.self_term * X
        return out + setup.background * X.sum(axis=0)

    def columns(self, index) -> np.ndarray:
        """`K[:, index]`, one PME pass per column not asked for before.

        Kept for the life of the operator, i.e. one force call: the admission
        gate asks for the same few atoms' columns channel after channel.
        """
        index = np.atleast_1d(index)
        missing = [a for a in dict.fromkeys(index.tolist()) if a not in self._columns]
        if missing:
            E = np.zeros((self.n, len(missing)))
            E[missing, np.arange(len(missing))] = 1.0
            for a, column in zip(missing, self.matvec(E).T):
                self._columns[a] = column
        return np.stack([self._columns[a] for a in index.tolist()], axis=1)

    def contract(self, W, factor=None, signs=None):
        """`dS/dr_i`, `dS/de_ab` for `S = sum_ij W_ij K_ij`, `W = F diag(s) F^T`.

        Needs `factor`: the reciprocal half is one PME pass per column, and the
        real-space half reads `W` on the pairs inside the cutoff only.
        """
        if factor is None:
            raise ValueError("EwaldOperator.contract needs the weight as a factor")
        setup = self.setup
        signs = np.ones(factor.shape[1]) if signs is None else np.asarray(signs)
        signed = factor * signs
        i, j = self.i, self.j
        w_pairs = np.einsum("pc,pc->p", signed[i], factor[j])
        # Each unordered pair is both (i, j) and (j, i) of the sum.
        coeff = 2.0 * w_pairs * self.dshort / self.r
        v = self.v
        grad = coeff[:, None] * v
        dS_dr = np.zeros((self.n, 3))
        for a in range(3):
            dS_dr[:, a] = np.bincount(i, grad[:, a], self.n) - np.bincount(
                j, grad[:, a], self.n
            )
        dS_de = grad.T @ v
        _, rec_r, rec_e = self.pme.contract(factor, signs)
        total_weight = float(np.sum(signs * factor.sum(axis=0) ** 2))
        dS_de = dS_de + rec_e - np.eye(3) * (setup.background * total_weight)
        return dS_dr + rec_r, dS_de

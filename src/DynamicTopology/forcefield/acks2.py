"""Fragment ACKS2: charge equilibration per diabatic state.

Each diabatic state `s` minimizes its own ACKS2 functional over the charges `q`
and the Kohn-Sham potentials `u` of every atom in the system,

    F_s(q, u) = mu.q + 1/2 q^T (K + 2 diag(eta)) q
              - u^T (q - q0_s) - 1/2 u^T L_{X_s} u

in the units the bare kernel implies (`E = CCOUL * F`).
Two things make it a function of the bonding, and they are what the
single-solve ACKS2 this replaced could not express:

  - **`q0_s`, the reference charges**, are the template's `q0` per atom (zero
    where a template states none).  They carry each molecule's formal charge, so
    an H3O+ holds its +1 and a Grotthuss hop moves it.
  - **The softness `X_s` acts only within a molecule of state `s`.**  Its rows sum
    to zero per molecule, so `q = q0 + ...` keeps every molecule at exactly its
    formal charge: charge moves between molecules only by changing the bonding,
    which is the EVB's job, never by the solve.  That removes the 0.145 e the old
    global softness moved from H2 to O2 at a 2 A contact.

**The energy is the minimum itself, relative to the isolated molecules.**  A
state's electrostatic energy is `F_s*` minus, for each of its molecules, the
same functional minimized for that molecule alone in open boundaries.  So a lone
template scores exactly zero here -- its gas-phase energy belongs to its bonded
terms -- and what remains is intermolecular electrostatics, the polarization of
each molecule by the others (its intramolecular half included), and every image
interaction.  Both pieces are stationary in their own variables, so **the forces
need no charge response**: `dE/dr = 1/2 x^T (dA/dr) x` at the solved `x` for
each, with no adjoint solve.

**That reference is also what makes `coulombexclusion` unnecessary here**, and
the kernel inside the solve is the full one.  The exclusion existed to keep a
molecule's own Coulomb energy off its bonded terms; subtracting the isolated
minimum does that exactly, without removing the intramolecular kernel from the
equilibration.  So an isolated water's charges are the ones the `atom`
parameters were fitted to (`q_H = +0.304`), and the intramolecular energy a
molecule gains or pays when its neighbours polarize it stays in the energy --
the 0.07 eV of the water dimer's hydrogen bond the screen used to remove.
Under Ewald nothing is removed from `K_ij`, so the image-part artifact of
masking a periodic kernel (`pointcharge.py`) cannot arise either.

**Solving every state costs one solve, not one per state.**  A block is a union
of whole molecules, and both topology-dependent pieces are intramolecular,
so the states of a block differ only in their own block's rows.  The atoms
outside every multi-state block -- the environment -- form one fixed system,
solved once; its polarization response to each block is `n_b` more solves
(`G`), and a state's solve is then a small dense system over its own block with
the environment folded in as a reaction field (`Sigma = K_be G`) and a
potential (`p = K_be y`).  The environment's response to each state is exact:
`z_e = y - G q_b`.

**The Kohn-Sham potentials are eliminated before anything is factored.**  They
enter only within a molecule, so each molecule's `u` is solved for its charges
in closed form (`_Piece._eliminate`) and the system every solve sees is over
`[q, lambda_q]`: `n + m` unknowns instead of `2n + 2m`, an eighth of the LU.
A molecule whose softness is nearly disconnected -- a bond stretched past 2-4
A -- keeps its `u` as unknowns, since eliminating it would add entries of
order `1/X` to the hardness (`SOFTNESS_FLOOR`).

**The environment's system is LU-factored, or solved iteratively.**  The
direct solve (`global_params.charge_solver = "direct"`, the default) factors
it against the dense kernel.  The iterative one keeps the kernel as an
operator -- a real-space cutoff and particle-mesh Ewald,
`ewald.EwaldOperator` -- and solves by conjugate gradients projected onto each
molecule's charge sum (`_EnvironmentSolver`); a call with a molecule kept
explicit in the environment falls back to the direct solve.

**Two or more multi-state blocks see each other through their means**, exactly
as `PointCharge` does (a Hartree product), with the environment-screened
coupling `M = K_bb' - Sigma_bb'`; `System.calculate` sweeps until no weight moves.
With one multi-state block -- an ion and its hop partners, everything else
spectating -- the first pass is final and exact.

**Forces are one kernel contraction.**  Every derivative is `1/2 <dA/dr, Z>`
with `Z` the weight-averaged second moment of the full solution: the mean charges
`qbar`, plus each block's charge covariance over its states pushed through the
environment's response.  The kernel part is a single `kernel.contract(W)` for the
whole system; the intramolecular softness and the isolated references are sparse
pair sums, per state within a block.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from ase import units
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import erf
from scipy.sparse.csgraph import connected_components

from DynamicTopology.forcefield.neighbors import as_geometry, pair_gradients
from DynamicTopology.forcefield.params import active
from DynamicTopology.forcefield.pointcharge import KernelCache, direct_kernel

logger = logging.getLogger(__name__)

# The per-atom parameters of an `atom` term; `q0` is optional and zero if absent.
PARAMETERS: tuple[str, ...] = ("mu", "eta", "soft_amp", "soft_decay")

# A molecule's `u` is eliminated from the charge solve only if its softest mode
# is at least this stiff, which bounds the entries elimination adds to the
# hardness at `1 / SOFTNESS_FLOOR`.  A water's is about 0.5; a molecule falls
# below it once a bond stretches past 2-4 A, by element.  See `_Piece._eliminate`.
SOFTNESS_FLOOR: float = 1e-3


def _rows(indices: np.ndarray, wanted: np.ndarray, what: str) -> np.ndarray:
    """Row of `indices` holding each entry of `wanted`; every one must be there."""
    order = np.argsort(indices, kind="stable")
    at = np.searchsorted(indices[order], wanted)
    at = np.minimum(at, len(indices) - 1)
    rows = order[at]
    missing = indices[rows] != wanted
    if np.any(missing):
        raise KeyError(
            f"{int(np.sum(missing))} atom(s) carry no `{what}` term "
            f"(first: {wanted[missing][:5].tolist()})"
        )
    return rows


def atom_parameters(term_dict: dict, atoms: np.ndarray) -> dict[str, np.ndarray]:
    """`mu`, `eta`, `soft_amp`, `soft_decay` and `q0` for `atoms` (global indices)."""
    block = term_dict.get("atom")
    if block is None:
        raise KeyError("No atom parameters set")
    rows = _rows(block["atoms"][:, 0], atoms, "atom")
    kwargs = block["kwargs"]
    out = {}
    for name in PARAMETERS:
        out[name] = np.asarray(kwargs[name], dtype=float)[rows]
    q0 = kwargs.get("q0")
    out["q0"] = (
        np.zeros(len(atoms)) if q0 is None else np.asarray(q0, dtype=float)[rows]
    )
    return out


def molecule_labels(term_dict: dict, atoms: np.ndarray) -> np.ndarray:
    """Connected component of each of `atoms` (global, ascending) under the bonds."""
    n = len(atoms)
    bonds = term_dict.get("bond")
    if bonds is None or n == 0:
        return np.arange(n)
    edges = bonds["atoms"][:, :2]
    i = np.minimum(np.searchsorted(atoms, edges[:, 0]), n - 1)
    j = np.minimum(np.searchsorted(atoms, edges[:, 1]), n - 1)
    keep = (atoms[i] == edges[:, 0]) & (atoms[j] == edges[:, 1])
    graph = coo_matrix((np.ones(int(np.sum(keep))), (i[keep], j[keep])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def molecule_pairs(mol: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every pair `i < j` within one molecule, in row-major order.

    The pairs `np.nonzero(np.triu(mol[:, None] == mol[None, :], 1))` lists, in
    the same order, without the `n x n` comparison: one stack of molecules per
    size, each its own `triu_indices`.
    """
    n = len(mol)
    if n == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    order = np.argsort(mol, kind="stable")
    counts = np.bincount(mol)
    starts = np.cumsum(counts) - counts
    pi, pj = [np.zeros(0, dtype=int)], [np.zeros(0, dtype=int)]
    for k in np.unique(counts):
        if k < 2:
            continue
        # Each row a molecule's atoms, ascending: `order` is a stable sort.
        members = order[starts[np.flatnonzero(counts == k)][:, None] + np.arange(k)]
        a, b = np.triu_indices(k, 1)
        pi.append(members[:, a].ravel())
        pj.append(members[:, b].ravel())
    pi, pj = np.concatenate(pi), np.concatenate(pj)
    row_major = np.lexsort((pj, pi))
    return pi[row_major], pj[row_major]


class _Piece:
    """One bonding pattern over a set of atoms: what its functional is built from.

    `atoms` are solver indices (ascending), `glob` the same atoms' global
    indices.  Pairs are stored once, `i < j`, and only within a molecule: those
    are the only pairs the softness and the isolated-molecule reference act on,
    and they are measured on `geometry` (a `neighbors.Geometry` in solver
    order) pair by pair, so nothing here is `n x n` but the dense systems of
    the direct solve.
    """

    def __init__(self, atoms, glob, term_dict, geometry, gamma, memo=None):
        self.atoms = atoms
        self.n = n = len(atoms)
        params = atom_parameters(term_dict, glob)
        self.mu = params["mu"]
        self.eta = params["eta"]
        self.q0 = params["q0"]
        amp, decay = params["soft_amp"], params["soft_decay"]
        # Per-molecule results shared between the pieces of one `bind`, keyed
        # on what they are a function of: the molecule's atoms and their
        # parameters, at the one geometry a `bind` sees.  The states of a block
        # share all but a molecule or two, and each used to redo every one.
        self.memo = memo
        if memo is not None:
            self._record = np.column_stack(
                [glob.astype(float), self.mu, self.eta, self.q0, amp, decay]
            )
        self.mol = molecule_labels(term_dict, glob)
        self.nmol = int(self.mol.max()) + 1 if n else 0

        self.pi, self.pj = molecule_pairs(self.mol)
        r = geometry.between(atoms[self.pi], atoms[self.pj])[1]
        self.g, self.dg = direct_kernel(r, gamma)
        tau = 0.5 * (decay[self.pi] + decay[self.pj])
        self.x = amp[self.pi] * amp[self.pj] * np.exp(-r / tau)
        self.dx = -self.x / tau
        self._response = None
        self._explicit = None

    def _softness(self) -> np.ndarray:
        """`X` over this piece, dense, zero off-molecule and on the diagonal."""
        X = np.zeros((self.n, self.n))
        X[self.pi, self.pj] = self.x
        X[self.pj, self.pi] = self.x
        return X

    def _stacked(self, members: np.ndarray, values: np.ndarray) -> np.ndarray:
        """A pair quantity over molecules `members` (G, k), as `(G, k, k)`.

        `values` is per pair (`self.pi`, `self.pj`); the result is what
        indexing the symmetric `n x n` matrix of them, zero off-molecule and on
        the diagonal, at `members[:, :, None], members[:, None, :]` gives --
        without the matrix.
        """
        G, k = members.shape
        out = np.zeros((G, k, k))
        if G == 0 or k < 2:
            return out
        row = np.full(self.nmol, -1)
        row[self.mol[members[:, 0]]] = np.arange(G)
        rank = np.zeros(self.n, dtype=int)
        rank[members] = np.arange(k)
        g = row[self.mol[self.pi]]
        mine = g >= 0
        g, a, c = g[mine], rank[self.pi[mine]], rank[self.pj[mine]]
        out[g, a, c] = values[mine]
        out[g, c, a] = values[mine]
        return out

    def _split(self, members: np.ndarray, kind: str):
        """`members` (G, k) as (keys, the rows the memo does not hold yet).

        Without a memo every row is missing and the keys are `None`.
        """
        if self.memo is None:
            return [None] * len(members), np.arange(len(members))
        # One byte string per molecule, from one view of the stacked records.
        records = np.ascontiguousarray(self._record[members]).reshape(len(members), -1)
        width = records.shape[1] * records.itemsize
        raw = records.view(np.dtype((np.void, width))).ravel().tolist()
        keys = [(kind, key) for key in raw]
        memo = self.memo
        missing = np.array(
            [i for i, key in enumerate(keys) if key not in memo], dtype=int
        )
        return keys, missing

    def _molecules(self):
        """The members of each molecule, one `(G, k)` stack per molecule size `k`."""
        order = np.argsort(self.mol, kind="stable")
        counts = np.bincount(self.mol, minlength=self.nmol)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for k in np.unique(counts):
            mols = np.flatnonzero(counts == k)
            yield order[starts[mols][:, None] + np.arange(k)]

    def _eliminate(self) -> None:
        """Split the molecules into those whose `u` is eliminated and the rest.

        The Kohn-Sham rows of `full_system` are `L u - lambda_u = q - q0`, with
        `L = X - diag(X 1)` the softness Laplacian of each molecule, and
        `sum u = 0` per molecule.  Solved per molecule, `u = Gamma (q - q0)`
        with `Gamma` the `uu` block of `[[L, -1], [-1^T, 0]]^{-1}`: negative
        semidefinite, so eliminating `u` adds `-Gamma` to the hardness.

        `Gamma` is `L`'s pseudo-inverse, so a molecule whose softness is nearly
        disconnected -- a bond the topology still holds stretched to 10 A, its
        `X` near `e^-40` -- would put entries of `1/X` beside a hardness of
        order ten and cancel away the precision of every other mode of that
        molecule.  Those molecules keep `u` and `lambda_u` as unknowns, exactly
        as `full_system` has them, where `u` stays of order one.  Every other
        molecule -- every molecule near equilibrium -- is eliminated.
        """
        n = self.n
        explicit = np.zeros(n, dtype=bool)
        rows, cols, values = [np.zeros(0, int)], [np.zeros(0, int)], [np.zeros(0)]
        for members in self._molecules():
            k = members.shape[1]
            if k == 1:
                continue  # `u = 0`: nothing to eliminate
            keys, missing = self._split(members, "gamma")
            if len(missing):
                fresh = members[missing]
                Xg = self._stacked(fresh, self.x)
                at = np.arange(k)
                L = Xg.copy()
                L[:, at, at] -= Xg.sum(axis=2)
                # `L` is negative semidefinite with one zero mode; the next is
                # the softest the molecule has.
                soft = -np.linalg.eigvalsh(L)[:, -2]
                ok = soft >= SOFTNESS_FLOOR
                B = np.zeros((int(np.sum(ok)), k + 1, k + 1))
                B[:, :k, :k] = L[ok]
                B[:, k, :k] = B[:, :k, k] = -1.0
                inverse = np.linalg.inv(B)[:, :k, :k]
                blocks = iter(inverse)
                computed = [next(blocks) if good else None for good in ok]
                if self.memo is not None:
                    for i, value in zip(missing, computed):
                        self.memo[keys[i]] = value
            if self.memo is not None:
                computed = [self.memo[key] for key in keys]
            ok = np.array([value is not None for value in computed])
            explicit[members[~ok].ravel()] = True
            members = members[ok]
            block = np.array([value for value in computed if value is not None])
            block = block.reshape(len(members), k, k)
            rows.append(np.broadcast_to(members[:, :, None], block.shape).ravel())
            cols.append(np.broadcast_to(members[:, None, :], block.shape).ravel())
            values.append(block.ravel())
        self._response = csr_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n, n),
        )
        self._explicit = np.flatnonzero(explicit)
        mols, self._explicit_mol = np.unique(self.mol[self._explicit], return_inverse=True)
        self._nexplicit_mol = len(mols)
        if len(self._explicit):
            X = csr_matrix(
                (np.concatenate([self.x, self.x]), (np.concatenate([self.pi, self.pj]), np.concatenate([self.pj, self.pi]))),
                shape=(n, n),
            )
            self._softness_explicit = X[self._explicit][:, self._explicit].toarray()
        else:
            self._softness_explicit = np.zeros((0, 0))

    def potentials(self, x: np.ndarray, response: bool = False) -> np.ndarray:
        """The Kohn-Sham potentials `u` at a solution `x` of `system`.

        With `response`, `x` is instead the response of the solution to a
        change in `b` (vector or columns), and so is the `u` returned.
        """
        if self._response is None:
            self._eliminate()
        n, m = self.n, self.nmol
        u = self._response @ x[:n]
        u[self._explicit] += x[n + m : n + m + len(self._explicit)]
        return u if response else u - self._response @ self.q0

    def system(self, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """`A x = b` over this piece with the Kohn-Sham potentials eliminated.

        `x = [q, lambda_q, u_e, lambda_e]`: `n + m` unknowns plus the `u` and
        the Kohn-Sham constraint of each molecule kept explicit (`_eliminate`),
        where `full_system` has `2n + 2m` -- the same minimum, an eighth of the
        LU.  Returns `(A, b, c)` with the functional's minimum
        `-1/2 (b.x + c)`.  `A` is symmetric; `potentials` recovers `u`.
        """
        if self._response is None:
            self._eliminate()
        n, m = self.n, self.nmol
        G, e = self._response, self._explicit
        ne, me = len(e), self._nexplicit_mol
        size = n + m + ne + me
        A = np.zeros((size, size))
        b = np.zeros(size)
        A[:n, :n] = K
        if n:
            A[:n, :n] -= G.toarray()
        at = np.arange(n)
        A[at, at] += 2.0 * self.eta
        A[n + self.mol, at] = A[at, n + self.mol] = -1.0
        Gq0 = G @ self.q0
        b[:n] = -self.mu - Gq0
        b[n : n + m] = -np.bincount(self.mol, self.q0, m)
        if ne:
            u = n + m + np.arange(ne)
            lam = n + m + ne + self._explicit_mol
            Xe = self._softness_explicit
            A[np.ix_(u, u)] = Xe - np.diag(Xe.sum(axis=1))
            A[e, u] = A[u, e] = -1.0
            A[lam, u] = A[u, lam] = -1.0
            b[u] = -self.q0[e]
        return A, b, float(self.q0 @ Gq0)

    def rhs(self) -> tuple[np.ndarray, float]:
        """`system`'s `b` and `c`, for a piece with no molecule kept explicit."""
        if self._response is None:
            self._eliminate()
        n, m = self.n, self.nmol
        b = np.zeros(n + m)
        Gq0 = self._response @ self.q0
        b[:n] = -self.mu - Gq0
        b[n:] = -np.bincount(self.mol, self.q0, m)
        return b, float(self.q0 @ Gq0)

    def full_system(self, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`A x = b` over this piece with `K` its bare kernel block, unreduced.

        `x = [q, u, lambda_q, lambda_u]`: one charge constraint and one
        Kohn-Sham constraint per molecule.  `A` is symmetric.
        """
        n, m = self.n, self.nmol
        A = np.zeros((2 * n + 2 * m, 2 * n + 2 * m))
        b = np.zeros(2 * n + 2 * m)
        A[:n, :n] = K
        at = np.arange(n)
        A[at, at] += 2.0 * self.eta
        X = self._softness()
        A[n : 2 * n, n : 2 * n] = X - np.diag(X.sum(axis=1))
        A[at, n + at] = A[n + at, at] = -1.0
        A[2 * n + self.mol, at] = A[at, 2 * n + self.mol] = -1.0
        A[2 * n + m + self.mol, n + at] = A[n + at, 2 * n + m + self.mol] = -1.0
        b[:n] = -self.mu
        b[n : 2 * n] = -self.q0
        b[2 * n : 2 * n + m] = -np.bincount(self.mol, self.q0, m)
        return A, b

    def isolated(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Each molecule's own minimum, alone and in open boundaries, summed.

        Returns `(F, q, u)`.  Block diagonal, so it is solved one stack of
        equal-sized molecules at a time rather than as one dense system.
        """
        n = self.n
        q, u = np.zeros(n), np.zeros(n)
        if n == 0:
            return 0.0, q, u
        total = 0.0
        for members in self._molecules():  # (G, k)
            if self.memo is not None:
                keys, missing = self._split(members, "isolated")
                k = members.shape[1]
                if len(missing):
                    F, qs, us = self._isolated_stack(members[missing])
                    for i, f, qm, um in zip(missing, F, qs, us):
                        self.memo[keys[i]] = (f, qm, um)
                # Summed molecule by molecule, so that a molecule's share does
                # not depend on which others it was solved alongside.
                for row, key in zip(members, keys):
                    f, qm, um = self.memo[key]
                    total += f
                    q[row] = qm
                    u[row] = um
                continue
            G, k = members.shape
            dim = 2 * k + 2
            A = np.zeros((G, dim, dim))
            b = np.zeros((G, dim))
            at = np.arange(k)
            A[:, :k, :k] = self._stacked(members, self.g)
            A[:, at, at] += 2.0 * self.eta[members]
            Xg = self._stacked(members, self.x)
            A[:, k : 2 * k, k : 2 * k] = Xg
            A[:, k + at, k + at] -= Xg.sum(axis=2)
            A[:, at, k + at] = A[:, k + at, at] = -1.0
            A[:, 2 * k, :k] = A[:, :k, 2 * k] = -1.0
            A[:, 2 * k + 1, k : 2 * k] = A[:, k : 2 * k, 2 * k + 1] = -1.0
            b[:, :k] = -self.mu[members]
            b[:, k : 2 * k] = -self.q0[members]
            b[:, 2 * k] = -self.q0[members].sum(axis=1)
            x = np.linalg.solve(A, b[:, :, None])[:, :, 0]
            total += -0.5 * float(np.sum(b * x))
            q[members] = x[:, :k]
            u[members] = x[:, k : 2 * k]
        return total, q, u

    def _isolated_stack(self, members):
        """`isolated`'s solve for one stack of equal-sized molecules, per molecule."""
        G, k = members.shape
        dim = 2 * k + 2
        A = np.zeros((G, dim, dim))
        b = np.zeros((G, dim))
        at = np.arange(k)
        A[:, :k, :k] = self._stacked(members, self.g)
        A[:, at, at] += 2.0 * self.eta[members]
        Xg = self._stacked(members, self.x)
        A[:, k : 2 * k, k : 2 * k] = Xg
        A[:, k + at, k + at] -= Xg.sum(axis=2)
        A[:, at, k + at] = A[:, k + at, at] = -1.0
        A[:, 2 * k, :k] = A[:, :k, 2 * k] = -1.0
        A[:, 2 * k + 1, k : 2 * k] = A[:, k : 2 * k, 2 * k + 1] = -1.0
        b[:, :k] = -self.mu[members]
        b[:, k : 2 * k] = -self.q0[members]
        b[:, 2 * k] = -self.q0[members].sum(axis=1)
        x = np.linalg.solve(A, b[:, :, None])[:, :, 0]
        F = -0.5 * np.sum(b * x, axis=1)
        return F, x[:, :k], x[:, k : 2 * k]

    def pair_derivative(self, du2, q_iso, u_iso) -> np.ndarray:
        """`d(F - F_iso)/dr` on each intramolecular pair, charges held fixed.

        `du2` is `E[(u_i - u_j)^2]` over the pairs, for the softness
        `-1/2 sum_pairs X (u_i - u_j)^2`.  The kernel part of `F` is not here --
        it is the one contraction `evaluate` does for the whole system -- while
        the isolated reference's kernel acts only within each molecule, so it is.
        """
        diu = u_iso[self.pi] - u_iso[self.pj]
        iso = q_iso[self.pi] * q_iso[self.pj] * self.dg - 0.5 * diu * diu * self.dx
        return -0.5 * du2 * self.dx - iso


class _EnvironmentSolver:
    """The environment's reduced system, solved by projected conjugate gradients.

    The system `system` assembles is `H q - S^T lambda = b_q`, `-S q = b_l`,
    with `H = K + 2 diag(eta) - Gamma` and `S` the molecules' charge sums.
    `H` is positive definite on charges that keep every molecule's sum, so
    writing `q = q_p + z` with `S z = 0` leaves `P H z = P (b_q - H q_p)`, `P`
    removing each molecule's mean: symmetric positive definite on that
    subspace, which is what conjugate gradients needs.  `lambda` is the
    per-molecule mean of the residual `H q - b_q`, which is constant within
    each molecule at the solution.

    Preconditioned molecule by molecule, with each molecule's own block of `H`
    solved exactly on its sum-zero subspace: the kernel there is the bare
    `erf(gamma r) / r` plus the periodic diagonal, near enough for a
    preconditioner.  The intramolecular coupling is the strongest, so this
    leaves only the slow intermolecular polarization to the iteration.

    The kernel is only ever applied (`EwaldOperator.matvec`), never formed.
    """

    MAX_ITERATIONS = 2000

    def __init__(self, acks2, env: np.ndarray, piece: _Piece):
        self.acks2 = acks2
        self.kernel = acks2.kernel
        self.env = env
        self.piece = piece
        self.n_all = len(acks2.act)
        self.full = len(env) == self.n_all
        self.n, self.m = piece.n, piece.nmol
        self.mol = piece.mol
        self.counts = np.bincount(self.mol, minlength=self.m).astype(float)
        self.sums = csr_matrix(
            (np.ones(self.n), (self.mol, np.arange(self.n))), shape=(self.m, self.n)
        )
        self.gamma_matrix = piece._response
        self.eta2 = 2.0 * piece.eta
        self.tolerance = active().solver_tolerance
        self.iterations = 0
        self._blocks = self._preconditioner()

    # -- operators --------------------------------------------------------

    def H(self, Z: np.ndarray) -> np.ndarray:
        if self.full:
            KZ = self.kernel.matvec(Z)
        else:
            X = np.zeros((self.n_all, Z.shape[1]))
            X[self.env] = Z
            KZ = self.kernel.matvec(X)[self.env]
        return KZ + self.eta2[:, None] * Z - self.gamma_matrix @ Z

    def mean(self, V: np.ndarray) -> np.ndarray:
        return (self.sums @ V) / self.counts[:, None]

    def project(self, V: np.ndarray) -> np.ndarray:
        return V - self.mean(V)[self.mol]

    def _preconditioner(self):
        piece, kernel = self.piece, self.kernel
        diagonal = kernel.setup.diagonal
        gamma = kernel.setup.gamma
        blocks = []
        for members in piece._molecules():
            G, k = members.shape
            if k == 1:
                continue  # a lone atom's charge is fixed by its sum
            atoms = piece.atoms[members]
            _, r = self.acks2.geometry.between(
                np.repeat(atoms, k, axis=1), np.tile(atoms, (1, k))
            )
            r = r.reshape(G, k, k)
            off = ~np.eye(k, dtype=bool)
            H = np.where(off, erf(gamma * np.where(off, r, 1.0)) / np.where(off, r, 1.0), 0.0)
            H = H + diagonal
            H[:, np.arange(k), np.arange(k)] += self.eta2[members]
            Gm = np.zeros((G, k, k))
            for a in range(k):
                for c in range(k):
                    Gm[:, a, c] = np.asarray(
                        self.gamma_matrix[members[:, a], members[:, c]]
                    ).ravel()
            H = H - Gm
            # An orthonormal basis of the sum-zero subspace.
            values, vectors = np.linalg.eigh(np.eye(k) - 1.0 / k)
            N = vectors[:, values > 0.5]
            inner = np.linalg.inv(np.einsum("ai,gab,bj->gij", N, H, N))
            blocks.append((members, np.einsum("ai,gij,bj->gab", N, inner, N)))
        return blocks

    def precondition(self, R: np.ndarray) -> np.ndarray:
        out = np.zeros_like(R)
        for members, P in self._blocks:
            out[members] = np.einsum("gab,gbr->gar", P, R[members])
        return out

    def _cg(self, rhs: np.ndarray, x0: np.ndarray) -> np.ndarray:
        """`P H z = rhs` on the sum-zero subspace, columnwise."""
        x = x0.copy()
        r = rhs - self.project(self.H(x))
        z = self.precondition(r)
        p = z.copy()
        rz = np.sum(r * z, axis=0)
        target = self.tolerance * np.maximum(np.linalg.norm(rhs, axis=0), 1e-300)
        for iteration in range(self.MAX_ITERATIONS):
            if np.all(np.linalg.norm(r, axis=0) <= target):
                self.iterations += iteration
                return x
            Ap = self.project(self.H(p))
            curvature = np.sum(p * Ap, axis=0)
            alpha = np.where(curvature > 0, rz / np.where(curvature > 0, curvature, 1.0), 0.0)
            x += alpha * p
            r -= alpha * Ap
            z = self.precondition(r)
            rz_new = np.sum(r * z, axis=0)
            beta = np.where(rz > 0, rz_new / np.where(rz > 0, rz, 1.0), 0.0)
            p = z + beta * p
            rz = rz_new
        logger.warning(
            "ACKS2 iterative solve stopped at %d iterations, residual %.2e of %.2e",
            self.MAX_ITERATIONS,
            float(np.max(np.linalg.norm(r, axis=0) / target * self.tolerance)),
            self.tolerance,
        )
        self.iterations += self.MAX_ITERATIONS
        return x

    def solve(self, b: np.ndarray, columns: np.ndarray | None = None) -> np.ndarray:
        """`system`'s solution `[q, lambda]`: for its own `b`, or for `columns`.

        `columns` (n, r) are right-hand sides in the charge rows with zero
        constraint rows -- the environment's response to a block, `G`.
        """
        n = self.n
        if n == 0:
            return np.zeros(len(b)) if columns is None else np.zeros((len(b), columns.shape[1]))
        if columns is None:
            b_q = b[:n, None]
            # Any charges with the molecules' sums: `S q0` is what `b_l` asks.
            start = self.piece.q0[:, None]
            warm = self.acks2._warm.get(self.env.tobytes())
        else:
            b_q = columns
            start = np.zeros_like(columns)
            warm = None
        x0 = warm if warm is not None and warm.shape == b_q.shape else np.zeros_like(b_q)
        z = self._cg(self.project(b_q - self.H(start)), self.project(x0))
        q = start + z
        lam = self.mean(self.H(q) - b_q)
        if columns is None:
            self.acks2._warm = {self.env.tobytes(): z}
            return np.concatenate([q[:, 0], lam[:, 0]])
        return np.vstack([q, lam])


@dataclass
class _Block:
    """A multi-state EVB block as the charge solve sees it.  Solver indices."""

    atoms: np.ndarray  # (nb,)
    pieces: list[_Piece]
    iso: list[tuple[float, np.ndarray, np.ndarray]]
    lus: list = field(default_factory=list)
    rhs: list[np.ndarray] = field(default_factory=list)
    consts: list[float] = field(default_factory=list)  # `c` of each reduced system
    Keb: np.ndarray | None = None  # (ne, nb) bare kernel, environment to block
    columns: np.ndarray | None = None  # (n, nb) `K[:, atoms]`, iterative solver only
    Gq: np.ndarray | None = None  # (ne, nb) environment charge response
    Gu: np.ndarray | None = None  # (ne, nb) environment potential response
    weights: np.ndarray | None = None
    mean: np.ndarray | None = None  # (nb,) weight-averaged charges
    z: list[np.ndarray] = field(default_factory=list)
    ltilde: np.ndarray | None = None  # (nstates,) effective functional, F units


class ACKS2:
    """Fragment ACKS2 behind the block protocol `System.calculate` drives.

    `prepare` at the geometry, `bind` to the blocks, then `corrections` /
    `update` per multi-state block until self-consistent, then `evaluate`.
    `__call__` is the one-topology case every other caller wants.

    Solver indices are the atoms carrying an `atom` term, ascending; `act` maps
    them to global indices and `local` back.
    """

    # The corrections of one block depend on the others' mean charges.
    self_consistent: bool = True

    @property
    def CCOUL(self) -> float:
        """The Coulomb constant in eV*Angstrom, from the active parameters."""
        return active().ccoul

    def __init__(self):
        self.kernels = KernelCache()
        # The iterative solver's last solution, to start the next from.
        self._warm: dict = {}
        self.Q = None
        self.act = None
        self.local = None
        self.shape = None

    # -- the geometry ---------------------------------------------------------

    def prepare(self, pos, pbc, cell, term_dict: dict, displacements=None) -> None:
        """Build the kernel at this geometry and hold the seed topology's terms.

        `displacements` is the force call's `neighbors.Geometry` over every
        atom, or `geometry(pos, pbc, cell)`, if the caller already has it.
        """
        block = term_dict.get("atom")
        if block is None:
            raise KeyError("No atom parameters set")
        self.shape = pos.shape
        self.act = np.unique(block["atoms"][:, 0])
        self.local = np.full(len(pos), -1, dtype=int)
        self.local[self.act] = np.arange(len(self.act))
        self.geometry = as_geometry(displacements, pos, pbc, cell).subset(self.act)
        # The iterative solver keeps the kernel as an operator and `K` unset;
        # see `_EnvironmentSolver`.
        self.operator = bool(np.all(pbc)) and active().charge_solver == "iterative"
        self._where = (pos[self.act], pbc, cell)
        self.kernel = self.kernels.get(
            pos[self.act], self.geometry, pbc, cell, operator=self.operator
        )
        self.K = None if self.operator else self.kernel.matrix()
        self.seed = term_dict
        self.gamma = active().gamma
        self.blocks: dict[int, _Block] = {}

    def _direct(self, reason: str) -> None:
        """Take this call on the dense kernel and the direct solve."""
        logger.info("ACKS2: direct charge solve for this call: %s", reason)
        pos, pbc, cell = self._where
        self.operator = False
        self.kernel = self.kernels.get(pos, self.geometry, pbc, cell)
        self.K = self.kernel.matrix()

    def _piece(self, atoms: np.ndarray, term_dict: dict, memo=None) -> _Piece:
        return _Piece(atoms, self.act[atoms], term_dict, self.geometry, self.gamma, memo)

    # -- the blocks -----------------------------------------------------------

    def bind(self, blocks) -> None:
        """Factor the environment and every multi-state block's states.

        The environment is every atom outside a multi-state block, in the seed
        topology -- which is what each single-state block's one state is.
        """
        n = len(self.act)
        self.blocks = {}
        in_block = np.zeros(n, dtype=bool)
        # Per-molecule results for every state of every block; see `_Piece`.
        memo: dict = {}
        for k, block in enumerate(blocks):
            if block.nstates < 2:
                continue
            glob = np.array(sorted(block.states[0].graph.nodes()), dtype=int)
            atoms = self.local[glob]
            if np.any(atoms < 0):
                raise KeyError("An EVB block holds an atom with no `atom` term.")
            in_block[atoms] = True
            pieces = [
                self._piece(atoms, state.term_dict, memo) for state in block.states
            ]
            weights = np.zeros(block.nstates)
            weights[block.seed_index] = 1.0
            self.blocks[k] = _Block(
                atoms, pieces, [p.isolated() for p in pieces], weights=weights
            )

        env = np.flatnonzero(~in_block)
        ne = len(env)
        self.env = env
        self.env_piece = self._piece(env, self.seed)
        if self.operator:
            if self.env_piece._response is None:
                self.env_piece._eliminate()  # settles which are kept explicit
            if len(self.env_piece._explicit):
                self._direct("a molecule of the environment is kept explicit")
        if self.operator:
            b, c = self.env_piece.rhs()
            self.solver = _EnvironmentSolver(self, env, self.env_piece)
            self.y = self.solver.solve(b)
        else:
            A, b, c = self.env_piece.system(self.K[np.ix_(env, env)])
            self.lu_env = lu_factor(A) if len(b) else None
            self.y = lu_solve(self.lu_env, b) if len(b) else b
        self.y_u = self.env_piece.potentials(self.y)
        self.const_env = -0.5 * (float(b @ self.y) + c)
        self.iso_env = self.env_piece.isolated()

        for block in self.blocks.values():
            nb = len(block.atoms)
            if self.operator:
                block.columns = self.kernel.columns(block.atoms)
                block.Keb = block.columns[env]
            else:
                block.Keb = self.K[np.ix_(env, block.atoms)]
            if ne and self.operator:
                G = self.solver.solve(np.zeros(len(b)), block.Keb)
            elif ne:
                C = np.zeros((len(b), nb))
                C[:ne] = block.Keb
                G = lu_solve(self.lu_env, C)
            else:
                G = np.zeros((0, nb))
            block.Gq = G[:ne]
            block.Gu = self.env_piece.potentials(G, response=True)
            # The environment as the block sees it: a potential and a reaction
            # field, the same for every state.
            p = block.Keb.T @ self.y[:ne]
            sigma = block.Keb.T @ block.Gq
            if self.operator:
                Kbb = block.columns[block.atoms]
            else:
                Kbb = self.K[np.ix_(block.atoms, block.atoms)]
            for piece in block.pieces:
                A, rhs, c = piece.system(Kbb)
                A[:nb, :nb] -= sigma
                rhs[:nb] -= p
                block.lus.append(lu_factor(A))
                block.rhs.append(rhs)
                block.consts.append(c)
            seed = int(np.argmax(block.weights))
            block.mean = lu_solve(block.lus[seed], block.rhs[seed])[:nb]
            block.z = [None] * len(block.pieces)
            block.ltilde = np.zeros(len(block.pieces))

    def _coupling(self, a: _Block, b: _Block) -> np.ndarray:
        """`K_ab - Sigma_ab`: two blocks' charges, screened by the environment."""
        if self.K is None:
            return b.columns[a.atoms] - a.Keb.T @ b.Gq
        return self.K[np.ix_(a.atoms, b.atoms)] - a.Keb.T @ b.Gq

    def corrections(self, index: int) -> np.ndarray:
        """Each state's electrostatic energy, in eV, given the other blocks' means.

        That is `d E / d w_s`: the state's own minimum in the environment's
        potential and reaction field and in the other blocks' mean charges,
        less its molecules' isolated minima.  A single-state block has nothing
        to decide and reports zero; its energy is part of the environment's.
        """
        block = self.blocks.get(index)
        if block is None:
            return np.zeros(1)
        nb = len(block.atoms)
        h = np.zeros(nb)
        for other_index, other in self.blocks.items():
            if other_index != index:
                h += self._coupling(block, other) @ other.mean
        out = np.empty(len(block.pieces))
        states = zip(block.lus, block.rhs, block.consts, block.iso)
        for s, (lu, rhs, const, iso) in enumerate(states):
            c = rhs.copy()
            c[:nb] -= h
            z = lu_solve(lu, c)
            value = -0.5 * (float(c @ z) + const)
            block.z[s] = z
            block.ltilde[s] = value - float(h @ z[:nb])
            out[s] = value - iso[0]
        return self.CCOUL * out * units.eV

    def update(self, index: int, weights: np.ndarray) -> None:
        """Adopt block `index`'s ground-state weights and its mean charges."""
        block = self.blocks[index]
        nb = len(block.atoms)
        block.weights = np.asarray(weights, dtype=float)
        block.mean = block.weights @ np.array([z[:nb] for z in block.z])

    # -- evaluation -----------------------------------------------------------

    def evaluate(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Energy, forces and virial of the whole system at the current weights.

        The weights and every solved `x` are held fixed under the derivative:
        the first is Hellmann-Feynman, the second is the functional being
        stationary in its own variables.  Both are exact once `System` has swept
        the blocks to self-consistency.
        """
        ccoul = self.CCOUL
        n, env, ne = len(self.act), self.env, len(self.env)
        # `System` has always solved every state by now; a direct caller of
        # `bind` then `evaluate` may not have, and gets the seed weights.
        for index, block in self.blocks.items():
            if any(z is None for z in block.z):
                self.corrections(index)
        blocks = list(self.blocks.values())

        F = self.const_env - self.iso_env[0]
        for a, block in enumerate(blocks):
            F += float(block.weights @ (block.ltilde - [iso[0] for iso in block.iso]))
            for other in blocks[a + 1 :]:
                F += float(block.mean @ self._coupling(block, other) @ other.mean)

        # Mean charges and potentials, and each block's charge covariance.
        qbar, ubar = np.zeros(n), np.zeros(n)
        q_env, u_env = self.y[:ne].copy(), self.y_u.copy()
        states = []
        for block in blocks:
            nb = len(block.atoms)
            Zq = np.array([z[:nb] for z in block.z])
            Zu = np.array([p.potentials(z) for p, z in zip(block.pieces, block.z)])
            dq = Zq - block.mean
            cov = dq.T @ (block.weights[:, None] * dq)
            states.append((Zq, Zu, cov))
            q_env -= block.Gq @ block.mean
            u_env -= block.Gu @ block.mean
            qbar[block.atoms] = block.mean
            ubar[block.atoms] = block.weights @ Zu
        qbar[env], ubar[env] = q_env, u_env

        # The bare kernel over every pair: W = E[q q^T] / 2.  A covariance is
        # a weighted sum of outer products, so `W = F F^T` with one column for
        # the mean and one per state of each block -- which the kernel's
        # reciprocal half contracts at a cost of `n` per column rather than `n^2`.
        columns = [qbar[:, None]]
        for block, (Zq, _, _) in zip(blocks, states):
            R = np.zeros((n, len(block.atoms)))
            R[env] = -block.Gq
            R[block.atoms] = np.eye(len(block.atoms))
            root = np.sqrt(np.maximum(block.weights, 0.0))  # a -1e-17 is a 0
            columns.append(R @ ((Zq - block.mean).T * root))
        factor = np.sqrt(0.5) * np.hstack(columns)
        if self.K is None:
            dS_dr, dS_de = self.kernel.contract(None, factor)
        else:
            W = factor @ factor.T
            dS_dr, dS_de = self.kernel.contract(W, factor)
        forces = -dS_dr
        virial = dS_de.copy()

        # The intramolecular pair terms: the environment's softness with its
        # fluctuation under the blocks' states, each block's state by state.
        piece = self.env_piece
        pi, pj = piece.pi, piece.pj
        du = ubar[env[pi]] - ubar[env[pj]]
        du2 = du * du
        for block, (_, _, cov) in zip(blocks, states):
            d = block.Gu[pi] - block.Gu[pj]
            du2 = du2 + np.einsum("pa,ab,pb->p", d, cov, d)
        pair_i = [env[pi]]
        pair_j = [env[pj]]
        pair_c = [piece.pair_derivative(du2, *self.iso_env[1:])]
        for block, (_, Zu, _) in zip(blocks, states):
            for w, piece, u, iso in zip(block.weights, block.pieces, Zu, block.iso):
                if w == 0.0:
                    continue
                pi, pj = piece.pi, piece.pj
                du = u[pi] - u[pj]
                pair_i.append(block.atoms[pi])
                pair_j.append(block.atoms[pj])
                pair_c.append(w * piece.pair_derivative(du * du, *iso[1:]))
        pair_i, pair_j = np.concatenate(pair_i), np.concatenate(pair_j)
        f_pair, w_pair = pair_gradients(
            pair_i,
            pair_j,
            np.concatenate(pair_c),
            *self.geometry.between(pair_i, pair_j),
            n,
        )
        forces += f_pair
        virial += w_pair

        self.Q = qbar
        out = np.zeros(self.shape)
        out[self.act] = ccoul * forces * units.eV / units.Angstrom
        return ccoul * F * units.eV, out, ccoul * virial * units.eV

    def linear_system(self, pos, pbc, cell, term_dict: dict):
        """`A x = b` for one topology over every atom, as `__call__` solves it.

        `x = [q, u, lambda_q, lambda_u]` in solver order (`self.act`), one pair of
        multipliers per molecule.  For a caller probing the equilibration itself
        -- a polarizability, a response -- rather than the energy.
        """
        self.prepare(pos, pbc, cell, term_dict)
        if self.K is None:
            self._direct("`linear_system` returns the matrix")
        return self._piece(np.arange(len(self.act)), term_dict).full_system(self.K)

    def __call__(
        self, pos, pbc, cell, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """One fixed bonding pattern: its own reference charges and molecules."""
        self.prepare(pos, pbc, cell, term_dict)
        self.bind([])
        return self.evaluate()

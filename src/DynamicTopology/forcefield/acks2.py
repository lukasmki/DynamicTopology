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
LU-factored once; its polarization response to each block is `n_b`
back-substitutions (`G`), and a state's solve is then a small dense system over
its own block with the environment folded in as a reaction field
(`Sigma = K_be G`) and a potential (`p = K_be y`).  The environment's response to
each state is exact: `z_e = y - G q_b`.

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

from dataclasses import dataclass, field

import numpy as np
from ase import units
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from DynamicTopology.forcefield.params import active
from DynamicTopology.forcefield.pointcharge import (
    KernelCache,
    direct_kernel,
    geometry,
    pair_gradients,
)

# The per-atom parameters of an `atom` term; `q0` is optional and zero if absent.
PARAMETERS: tuple[str, ...] = ("mu", "eta", "soft_amp", "soft_decay")


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


class _Piece:
    """One bonding pattern over a set of atoms: what its functional is built from.

    `atoms` are solver indices (ascending), `glob` the same atoms' global
    indices.  Pairs are stored once, `i < j`, and only within a molecule: those
    are the only pairs the softness and the isolated-molecule reference act on.
    """

    def __init__(self, atoms, glob, term_dict, rij, gamma):
        self.atoms = atoms
        self.n = n = len(atoms)
        params = atom_parameters(term_dict, glob)
        self.mu = params["mu"]
        self.eta = params["eta"]
        self.q0 = params["q0"]
        amp, decay = params["soft_amp"], params["soft_decay"]
        self.mol = molecule_labels(term_dict, glob)
        self.nmol = int(self.mol.max()) + 1 if n else 0

        same = np.triu(self.mol[:, None] == self.mol[None, :], 1)
        self.pi, self.pj = np.nonzero(same)
        r = rij[atoms[self.pi], atoms[self.pj]]
        self.g, self.dg = direct_kernel(r, gamma)
        tau = 0.5 * (decay[self.pi] + decay[self.pj])
        self.x = amp[self.pi] * amp[self.pj] * np.exp(-r / tau)
        self.dx = -self.x / tau

    def _softness(self) -> np.ndarray:
        """`X` over this piece, dense, zero off-molecule and on the diagonal."""
        X = np.zeros((self.n, self.n))
        X[self.pi, self.pj] = self.x
        X[self.pj, self.pi] = self.x
        return X

    def system(self, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`A x = b` over this piece with `K` its bare kernel block.

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
        K = np.zeros((n, n))
        K[self.pi, self.pj] = self.g
        K[self.pj, self.pi] = self.g
        X = self._softness()

        order = np.argsort(self.mol, kind="stable")
        counts = np.bincount(self.mol, minlength=self.nmol)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        total = 0.0
        for k in np.unique(counts):
            mols = np.flatnonzero(counts == k)
            members = order[starts[mols][:, None] + np.arange(k)]  # (G, k)
            sub = (members[:, :, None], members[:, None, :])
            G = len(mols)
            dim = 2 * k + 2
            A = np.zeros((G, dim, dim))
            b = np.zeros((G, dim))
            at = np.arange(k)
            A[:, :k, :k] = K[sub]
            A[:, at, at] += 2.0 * self.eta[members]
            Xg = X[sub]
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


@dataclass
class _Block:
    """A multi-state EVB block as the charge solve sees it.  Solver indices."""

    atoms: np.ndarray  # (nb,)
    pieces: list[_Piece]
    iso: list[tuple[float, np.ndarray, np.ndarray]]
    lus: list = field(default_factory=list)
    rhs: list[np.ndarray] = field(default_factory=list)
    Keb: np.ndarray | None = None  # (ne, nb) bare kernel, environment to block
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
        self.Q = None
        self.act = None
        self.local = None
        self.shape = None

    # -- the geometry ---------------------------------------------------------

    def prepare(self, pos, pbc, cell, term_dict: dict) -> None:
        """Build the kernel at this geometry and hold the seed topology's terms."""
        block = term_dict.get("atom")
        if block is None:
            raise KeyError("No atom parameters set")
        self.shape = pos.shape
        self.act = np.unique(block["atoms"][:, 0])
        self.local = np.full(len(pos), -1, dtype=int)
        self.local[self.act] = np.arange(len(self.act))
        self.vecs, self.rij = geometry(pos[self.act], pbc, cell)
        self.kernel = self.kernels.get(pos[self.act], self.vecs, self.rij, pbc, cell)
        self.K = self.kernel.matrix()
        self.seed = term_dict
        self.gamma = active().gamma
        self.blocks: dict[int, _Block] = {}

    def _piece(self, atoms: np.ndarray, term_dict: dict) -> _Piece:
        return _Piece(atoms, self.act[atoms], term_dict, self.rij, self.gamma)

    # -- the blocks -----------------------------------------------------------

    def bind(self, blocks) -> None:
        """Factor the environment and every multi-state block's states.

        The environment is every atom outside a multi-state block, in the seed
        topology -- which is what each single-state block's one state is.
        """
        n = len(self.act)
        self.blocks = {}
        in_block = np.zeros(n, dtype=bool)
        for k, block in enumerate(blocks):
            if block.nstates < 2:
                continue
            glob = np.array(sorted(block.states[0].graph.nodes()), dtype=int)
            atoms = self.local[glob]
            if np.any(atoms < 0):
                raise KeyError("An EVB block holds an atom with no `atom` term.")
            in_block[atoms] = True
            pieces = [self._piece(atoms, state.term_dict) for state in block.states]
            weights = np.zeros(block.nstates)
            weights[block.seed_index] = 1.0
            self.blocks[k] = _Block(
                atoms, pieces, [p.isolated() for p in pieces], weights=weights
            )

        env = np.flatnonzero(~in_block)
        ne = len(env)
        self.env = env
        self.env_piece = self._piece(env, self.seed)
        A, b = self.env_piece.system(self.K[np.ix_(env, env)])
        self.lu_env = lu_factor(A) if len(b) else None
        self.y = lu_solve(self.lu_env, b) if len(b) else b
        self.const_env = -0.5 * float(b @ self.y)
        self.iso_env = self.env_piece.isolated()

        for block in self.blocks.values():
            nb = len(block.atoms)
            block.Keb = self.K[np.ix_(env, block.atoms)]
            if ne:
                C = np.zeros((len(b), nb))
                C[:ne] = block.Keb
                G = lu_solve(self.lu_env, C)
            else:
                G = np.zeros((0, nb))
            block.Gq, block.Gu = G[:ne], G[ne : 2 * ne]
            # The environment as the block sees it: a potential and a reaction
            # field, the same for every state.
            p = block.Keb.T @ self.y[:ne]
            sigma = block.Keb.T @ block.Gq
            Kbb = self.K[np.ix_(block.atoms, block.atoms)]
            for piece in block.pieces:
                A, rhs = piece.system(Kbb)
                A[:nb, :nb] -= sigma
                rhs[:nb] -= p
                block.lus.append(lu_factor(A))
                block.rhs.append(rhs)
            seed = int(np.argmax(block.weights))
            block.mean = lu_solve(block.lus[seed], block.rhs[seed])[:nb]
            block.z = [None] * len(block.pieces)
            block.ltilde = np.zeros(len(block.pieces))

    def _coupling(self, a: _Block, b: _Block) -> np.ndarray:
        """`K_ab - Sigma_ab`: two blocks' charges, screened by the environment."""
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
        for s, (lu, rhs, iso) in enumerate(zip(block.lus, block.rhs, block.iso)):
            c = rhs.copy()
            c[:nb] -= h
            z = lu_solve(lu, c)
            value = -0.5 * float(c @ z)
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
        q_env, u_env = self.y[:ne].copy(), self.y[ne : 2 * ne].copy()
        states = []
        for block in blocks:
            nb = len(block.atoms)
            Zq = np.array([z[:nb] for z in block.z])
            Zu = np.array([z[nb : 2 * nb] for z in block.z])
            dq = Zq - block.mean
            cov = dq.T @ (block.weights[:, None] * dq)
            states.append((Zq, Zu, cov))
            q_env -= block.Gq @ block.mean
            u_env -= block.Gu @ block.mean
            qbar[block.atoms] = block.mean
            ubar[block.atoms] = block.weights @ Zu
        qbar[env], ubar[env] = q_env, u_env

        # The bare kernel over every pair: W = E[q q^T] / 2.
        W = np.outer(qbar, qbar)
        for block, (_, _, cov) in zip(blocks, states):
            R = np.zeros((n, len(block.atoms)))
            R[env] = -block.Gq
            R[block.atoms] = np.eye(len(block.atoms))
            W += R @ cov @ R.T
        W *= 0.5
        dS_dr, dS_de = self.kernel.contract(W)
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
        f_pair, w_pair = pair_gradients(
            np.concatenate(pair_i),
            np.concatenate(pair_j),
            np.concatenate(pair_c),
            self.vecs,
            self.rij,
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
        return self._piece(np.arange(len(self.act)), term_dict).system(self.K)

    def __call__(
        self, pos, pbc, cell, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """One fixed bonding pattern: its own reference charges and molecules."""
        self.prepare(pos, pbc, cell, term_dict)
        self.bind([])
        return self.evaluate()

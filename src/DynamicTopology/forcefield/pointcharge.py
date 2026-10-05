"""Fixed point-charge electrostatics, set per molecule topology.

Every template carries its own charges
(`{"type": "charge", "atoms": {"p0": i}, "kwargs": {"q": ...}}`), so the charge
distribution is a function of the bonding, and a Grotthuss hop carries the
excess charge with the proton.  It is fragment ACKS2's zero-softness limit
(`forcefield/acks2.py`): the same per-state structure with the charges fixed at
the reference `q0` instead of solved for, so no linear system at all.

**What that costs is that electrostatics becomes state-dependent**, so it moves
onto the EVB diagonal, and it does so in a way no other term here does: it is
not local to a block.  Every block's charges act on every other block's atoms,
and which charges a multi-state block carries depends on its ground state.  The
blocks are therefore coupled, and the coupling is handled as a Hartree product:
each block sees the others through their *weight-averaged* charges

    qbar_B = sum_s w_s q_s,     w_s = (c_s)^2 of block B's ground state,

and its diagonal for state `s` is

    H_ss += E_intra(q_s) + q_s . V_B,    V_B = CCOUL * K[B, not B] . qbar

`E_intra` is the block's own Coulomb energy at state `s`'s charges with state
`s`'s exclusions removed.  The total energy is the expectation of the product
state, `sum_B lambda_B - E_inter`, where `E_inter` is the block-block Coulomb
energy of the averaged charges that `sum_B lambda_B` has counted twice.
Minimizing that functional over each block's eigenvector in turn is block
coordinate descent on a quadratic form, so it converges monotonically; at
convergence it is stationary in every eigenvector, which is what makes the
Hellmann-Feynman forces exact.  **With at most one multi-state block the first
pass is already converged** -- the environment is then a set of single-state
blocks whose charges nothing can move -- and that is the normal case in MD: an
ion and its hop partners in one block, every other molecule a spectator.

**At fixed weights the whole Coulomb energy is one kernel contraction.**  The
energy is `sum_ij W_ij K_ij` minus the exclusions, with

    W_ij = CCOUL/2 * sum_s w_s q_si q_sj     i, j in the same block
    W_ij = CCOUL/2 * qbar_i qbar_j           otherwise

so `evaluate` returns the energy, forces and virial of the whole system from a
single `kernel.contract(W)` plus a pair sum over the excluded pairs.  The charges
are fixed, so there is no response term and no linear solve anywhere in this
module.  That makes it cheaper than ACKS2, though not by much: 6.8 ms against
9.7 ms on an 82-atom periodic box, because building the Ewald kernel is the
larger cost and both terms pay it.

**The exclusion removes the direct pair and not its images.**  Under Ewald,
`K_ij` for an excluded pair is the direct `erf(gamma r)/r` *plus* the
interaction of `i` with every periodic image of `j`.  Only the first is what the
bonded terms own; removing the whole `K_ij`, as the old ACKS2 screen did, also
removes the image part while keeping each atom's own `K_ii`, which leaves
`CCOUL/2 * K_self * sum_i q_i**2` per molecule behind -- -0.91 eV per water in a
12.43 A box, scaling as 1/L, i.e. a pressure.  So the correction here is always
the open-boundary pair kernel, `MinimumImage`, at the minimum-image separation;
under open boundaries the two are the same thing.

The kernel is `ACKS2`'s own `erf(gamma r)/r` (`forcefield/ewald.py`), so both
boundary conditions come for free and the core is regularized at the same
`gamma`.  At every distance a charge pair is not excluded at, `erf(2 r)` is 1 to
better than 1e-3 past 1.35 A, so this is point-charge electrostatics everywhere
it matters.

**Under `charge_solver = "iterative"` the kernel is an operator**, the
real-space half over the force call's neighbour list and the reciprocal half by
PME (`ewald.EwaldOperator`), and no `N x N` array is formed: every product
taken with `K` is with a charge vector or a multi-state block's few columns, and
`evaluate` contracts `W` as a factor -- the mean charges plus one column per
state of each multi-state block, as `ACKS2` does.  The same energy to the
lattice sums' accuracy (~1e-8 relative), at `O(N log N)`.

**A charged system under full periodicity is neutralized by a uniform
background**, which `Ewald.background` carries explicitly.  A reaction never
changes the total charge, so the background energy is the same number on every
diabatic state and does not affect which is lower.

Admission into the EVB basis (`basis.py`) screens on the bonded gap plus the
electrostatic one, `forcefield.electrostatics.ElectrostaticGap`, which under
this term is exactly the difference of the two states' diagonal corrections
when one block is multi-state -- so a hop the environment favours, or
autoionization in a polar environment, is not screened as though in vacuum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import units
from scipy.special import erf

from DynamicTopology.forcefield.ewald import Ewald, EwaldOperatorSetup, MinimumImage
from DynamicTopology.forcefield.neighbors import as_geometry, pair_gradients
from DynamicTopology.forcefield.params import active

_NO_PAIRS: np.ndarray = np.zeros((0, 2), dtype=int)


def template_charges(term_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """`(indices, q)` from a term dict's `charge` terms."""
    params = term_dict.get("charge")
    if params is None:
        raise KeyError(
            "No `charge` terms set.  A dataset with `electrostatics = "
            '"pointcharge"` must give every atom of every template a charge term.'
        )
    return params["atoms"][:, 0], np.asarray(params["kwargs"]["q"], dtype=float)


def exclusion_pairs(term_dict: dict) -> np.ndarray:
    """The `coulombexclusion` pairs of a term dict, `(m, 2)`, or none."""
    params = term_dict.get("coulombexclusion")
    return _NO_PAIRS if params is None else params["atoms"][:, :2]


def geometry(pos, pbc, cell) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-image displacements `r_i - r_j`, `(n, n, 3)`, and their lengths.

    Every pair, which only a term without a cutoff needs; the others read
    `neighbors.Geometry`, which builds this only when one of them asks.
    """
    pbc = np.asarray(pbc, dtype=bool)
    vecs = pos[:, None, :] - pos[None, :, :]
    if np.any(pbc):
        cell = np.asarray(cell, dtype=float)
        F = vecs @ np.linalg.inv(cell)
        vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell
    return vecs, np.sqrt(np.sum(vecs * vecs, -1))


class KernelCache:
    """The charge kernel for a set of boundary conditions, Ewald setup cached per cell.

    Ewald needs all three directions periodic; a slab or a wire keeps the
    nearest-image kernel.  The cell-dependent half of the Ewald setup -- the
    splitting parameter and the reciprocal vectors -- is rebuilt only when the
    cell changes, so a fixed cell builds it once and an NPT trajectory once per
    step.
    """

    def __init__(self):
        self.ewald = None
        self.key = None
        self.operator = None
        self.operator_key = None

    def get(self, pos, geometry, pbc, cell, operator: bool = False):
        """The kernel at this geometry; `operator` asks for products only.

        `geometry` is a `neighbors.Geometry` of `pos`.  `operator` takes
        `ewald.EwaldOperatorSetup` -- a real-space cutoff over a neighbour list
        and PME, no `N x N` array -- under full periodicity; the other two are
        dense, and build the geometry's dense arrays.
        """
        if not np.all(pbc):
            vecs, rij = geometry.dense()
            return MinimumImage(rij, vecs)
        cell = np.asarray(cell, dtype=float)
        key = cell.tobytes()
        if operator:
            if self.operator is None or key != self.operator_key:
                self.operator = EwaldOperatorSetup(cell)
                self.operator_key = key
            return self.operator.bind(pos, geometry)
        if self.ewald is None or key != self.key:
            self.ewald = Ewald(cell)
            self.key = key
        return self.ewald.bind(pos, *geometry.dense())


def direct_kernel(r: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """`erf(gamma r) / r` and its radial derivative, on a list of pair distances.

    The pair form of `ewald._screened`, for the excluded pairs only.  Every
    distance here is between two distinct atoms, so there is no diagonal to
    guard.
    """
    g = erf(gamma * r) / r
    dg = (2 * gamma / np.sqrt(np.pi)) * np.exp(-((gamma * r) ** 2)) / r - g / r
    return g, dg


@dataclass
class _Block:
    """One EVB block as the electrostatics sees it.  Local indices throughout."""

    atoms: np.ndarray  # (nb,) global indices
    charges: np.ndarray  # (nstates, nb)
    pairs: list[np.ndarray]  # per state, (m, 2) block-local
    weights: np.ndarray  # (nstates,)
    intra: np.ndarray | None = None  # (nstates,), eV; computed on first use
    # `K[:, atoms]`, (natoms, nb), under the operator kernel; on first use.
    columns: np.ndarray | None = None


class PointCharge:
    """Fixed per-template charges, on the EVB diagonal.  See the module docstring.

    The interface is the one `System.calculate` drives both electrostatic terms
    through: `prepare` at the geometry, `bind` to the blocks, then `corrections`
    / `update` per block until self-consistent, then `evaluate`.  `__call__` is
    the single-topology case every other caller wants -- the fitter, `evb.py`.

    Unlike `ACKS2` everything is in global atom order: there is no linear system
    whose rows have to line up with a parameter vector, so there is no term
    order to confuse it with.

    **The kernel is a matrix or an operator.**  Under full periodicity with
    `charge_solver = "iterative"` it is `ewald.EwaldOperator` -- the real-space
    half over a neighbour list, the reciprocal half by PME -- as for ACKS2,
    though nothing here is solved: every product this class takes with `K` is
    a product with the charges, or with a block's few columns, so it never
    needs `K` itself, and the whole term is `O(N log N)`.  `K` is then `None`
    and every product goes through `_columns` and `kernel.matvec`.  Otherwise
    `K` is the dense matrix, as it always was.
    """

    # The corrections depend on the other blocks' weights, so `System` has to
    # iterate.  `ACKS2`'s do not.
    self_consistent: bool = True

    @property
    def CCOUL(self) -> float:
        """The Coulomb constant in eV*Angstrom, shared with `ACKS2`."""
        return active().ccoul

    def __init__(self):
        self.kernels = KernelCache()
        self.natoms = 0
        self.act = None
        self.local = None
        self.geometry = None
        self.kernel = None
        self.K = None
        self.qbar = None
        self.phi = None
        self.blocks: list[_Block] = []

    # -- the geometry ---------------------------------------------------------

    def prepare(self, pos, pbc, cell, term_dict: dict, displacements=None) -> None:
        """Build the kernel at this geometry and take the seed topology's charges.

        The seed's charges are the starting environment for every block, and
        remain the charges of any atom no block claims.  `displacements` is the
        force call's `neighbors.Geometry`, or `geometry(pos, pbc, cell)`, if the
        caller already has it.
        """
        self.geometry = as_geometry(displacements, pos, pbc, cell)

        self.natoms = len(pos)
        # Every atom takes part, in global order; `act`/`local` are the index
        # maps `ACKS2` needs and the admission gate reads off either term.
        self.act = np.arange(self.natoms)
        self.local = self.act
        operator = bool(np.all(pbc)) and active().charge_solver == "iterative"
        self.kernel = self.kernels.get(pos, self.geometry, pbc, cell, operator=operator)
        self.K = None if operator else self.kernel.matrix()

        indices, q = template_charges(term_dict)
        self.qbar = self._scatter(indices, q, "the current topology")
        self.phi = self.CCOUL * self._apply(self.qbar)
        self.blocks = []

    def _apply(self, x: np.ndarray) -> np.ndarray:
        """`K @ x`."""
        return self.K @ x if self.K is not None else self.kernel.matvec(x)

    def _columns(self, block: _Block) -> np.ndarray:
        """`K[:, block.atoms]`, kept on the block under the operator kernel."""
        if self.K is not None:
            return self.K[:, block.atoms]
        if block.columns is None:
            block.columns = self.kernel.columns(block.atoms)
        return block.columns

    def _own(self, block: _Block) -> np.ndarray:
        """`K[b, b]` for the block's atoms `b`."""
        if self.K is not None:
            return self.K[np.ix_(block.atoms, block.atoms)]
        return self._columns(block)[block.atoms]

    def _scatter(self, indices, q, what: str) -> np.ndarray:
        """`q` on `indices`, as a system-length vector; every atom must be named."""
        out = np.full(self.natoms, np.nan)
        out[indices] = q
        missing = np.flatnonzero(np.isnan(out))
        if len(missing):
            raise KeyError(
                f"{len(missing)} atom(s) of {what} carry no `charge` term "
                f"(first: {missing[:5].tolist()}); every template needs one per atom."
            )
        return out

    # -- the blocks -----------------------------------------------------------

    def _add_block(self, atoms, charges, pairs, seed: int) -> None:
        weights = np.zeros(len(charges))
        weights[seed] = 1.0
        self.blocks.append(_Block(atoms, charges, pairs, weights))

    def bind(self, blocks) -> None:
        """Read each block's per-state charges and exclusions off its states' terms.

        `blocks` are `basis.Block`s.  Every state of a block spans the same
        atoms -- a reaction rewires bonds, it does not add or remove atoms -- so
        one index array serves all of them.  The environment is reset to each
        block's seed state, which is what the seed topology `prepare` read
        already says; it is restated so the two cannot silently disagree.
        """
        local = np.full(self.natoms, -1, dtype=int)
        for block in blocks:
            atoms = np.array(sorted(block.states[0].graph.nodes()), dtype=int)
            local[atoms] = np.arange(len(atoms))
            charges = np.full((block.nstates, len(atoms)), np.nan)
            pairs = []
            for s, state in enumerate(block.states):
                indices, q = template_charges(state.term_dict)
                charges[s, local[indices]] = q
                pairs.append(local[exclusion_pairs(state.term_dict)])
            if np.any(np.isnan(charges)):
                raise KeyError(
                    "A diabatic state carries no `charge` term for some of its "
                    "atoms; every template needs one per atom."
                )
            local[atoms] = -1
            self._add_block(atoms, charges, pairs, block.seed_index)
            self.qbar[atoms] = charges[block.seed_index]
        self.phi = self.CCOUL * self._apply(self.qbar)

    def _intra(self, block: _Block) -> np.ndarray:
        """Each state's Coulomb energy within the block, exclusions removed, in eV."""
        if block.intra is None:
            b = block.atoms
            Kbb = self._own(block)
            gamma = active().gamma
            intra = np.empty(len(block.charges))
            for s, (q, pairs) in enumerate(zip(block.charges, block.pairs)):
                energy = 0.5 * float(q @ Kbb @ q)
                if len(pairs):
                    i, j = pairs[:, 0], pairs[:, 1]
                    g, _ = direct_kernel(self.geometry.between(b[i], b[j])[1], gamma)
                    energy -= float(np.sum(q[i] * q[j] * g))
                intra[s] = self.CCOUL * energy * units.eV
            block.intra = intra
        return block.intra

    def corrections(self, index: int) -> np.ndarray:
        """What each state of block `index` adds to its diagonal, in eV.

        Its own Coulomb energy plus its charges in the potential of every other
        block's current averaged charges.  The potential of the block's own
        averaged charges is taken back out of `phi`, since a state interacts with
        itself through `_intra` and not through the average.
        """
        block = self.blocks[index]
        if self.K is None and len(block.charges) == 1:
            # A single-state block has nothing to decide, and `System` reads
            # this only to report the block's energy; under the operator its
            # `K[b, b]` would cost a PME pass per atom, and a liquid's blocks
            # span the box.  Reported as zero, as `ACKS2` reports it: the
            # block's electrostatics is part of the whole system's, in
            # `evaluate`.
            return np.zeros(1)
        b = block.atoms
        own = self.CCOUL * (self._own(block) @ self.qbar[b])
        environment = (self.phi[b] - own) * units.eV
        return self._intra(block) + block.charges @ environment

    def update(self, index: int, weights: np.ndarray) -> None:
        """Adopt block `index`'s new ground-state weights as its averaged charges."""
        block = self.blocks[index]
        block.weights = np.asarray(weights, dtype=float)
        b = block.atoms
        new = block.weights @ block.charges
        self.phi += self.CCOUL * (self._columns(block) @ (new - self.qbar[b]))
        self.qbar[b] = new

    # -- evaluation -----------------------------------------------------------

    def _exclusions(self):
        """The excluded pairs of every block's states, and `CCOUL w q_i q_j` on each."""
        pair_i, pair_j, pair_c = [], [], []
        for block in self.blocks:
            b = block.atoms
            for w, q, pairs in zip(block.weights, block.charges, block.pairs):
                if w == 0.0 or len(pairs) == 0:
                    continue
                i, j = pairs[:, 0], pairs[:, 1]
                pair_i.append(b[i])
                pair_j.append(b[j])
                pair_c.append(self.CCOUL * w * q[i] * q[j])
        return pair_i, pair_j, pair_c

    def _kernel_part(self) -> tuple[float, np.ndarray, np.ndarray]:
        """`S = sum_ij W_ij K_ij` and its derivatives, from the dense `W`.

        `W_ij = CCOUL/2 qbar_i qbar_j`, except within a multi-state block, whose
        states are alternatives rather than a mixture: `CCOUL/2 sum_s w_s q_si
        q_sj` there, which is not `qbar qbar^T`.
        """
        ccoul = self.CCOUL
        W = 0.5 * ccoul * np.outer(self.qbar, self.qbar)
        for block in self.blocks:
            b = block.atoms
            if len(block.charges) > 1:
                W[np.ix_(b, b)] = (
                    0.5
                    * ccoul
                    * np.einsum(
                        "s,si,sj->ij", block.weights, block.charges, block.charges
                    )
                )
        energy = float(np.sum(W * self.K))
        dS_dr, dS_de = self.kernel.contract(W)
        return energy, dS_dr, dS_de

    def _kernel_part_factored(self) -> tuple[float, np.ndarray, np.ndarray]:
        """`_kernel_part` through the operator, with `W` as a factor.

        `sum_s w_s q_s q_s^T = qbar qbar^T + sum_s w_s (q_s - qbar)(q_s - qbar)^T`
        when the weights sum to one, so `W = F F^T` with one column for the
        mean charges and one per state of each multi-state block, as `ACKS2`
        factors its own -- and `S = sum_c f_c^T K f_c` is one product.
        """
        columns = [self.qbar[:, None]]
        for block in self.blocks:
            if len(block.charges) < 2:
                continue
            mean = block.weights @ block.charges
            root = np.sqrt(np.maximum(block.weights, 0.0))
            C = np.zeros((self.natoms, len(block.charges)))
            C[block.atoms] = ((block.charges - mean) * root[:, None]).T
            columns.append(C)
        factor = np.sqrt(0.5 * self.CCOUL) * np.hstack(columns)
        energy = float(np.sum(factor * self.kernel.matvec(factor)))
        dS_dr, dS_de = self.kernel.contract(None, factor)
        return energy, dS_dr, dS_de

    def evaluate(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Energy, forces and virial of the whole system at the current weights.

        The weights are held fixed under the derivative, which is the
        Hellmann-Feynman statement and is exact once `System` has iterated the
        blocks to self-consistency.
        """
        if self.K is None:
            energy, dS_dr, dS_de = self._kernel_part_factored()
        else:
            energy, dS_dr, dS_de = self._kernel_part()
        forces = -dS_dr
        virial = dS_de

        pair_i, pair_j, pair_c = self._exclusions()
        if pair_i:
            i = np.concatenate(pair_i)
            j = np.concatenate(pair_j)
            c = np.concatenate(pair_c)
            v, r = self.geometry.between(i, j)
            g, dg = direct_kernel(r, active().gamma)
            # The exclusion is subtracted, so its radial derivative is `-c g'`.
            energy -= float(np.sum(c * g))
            f_pair, w_pair = pair_gradients(i, j, -c * dg, v, r, self.natoms)
            forces = forces + f_pair
            virial = virial + w_pair

        return (
            energy * units.eV,
            forces * units.eV / units.Angstrom,
            virial * units.eV,
        )

    def __call__(
        self, pos, pbc, cell, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """One fixed bonding pattern: its own charges, its own exclusions."""
        self.prepare(pos, pbc, cell, term_dict)
        atoms = np.arange(self.natoms)
        self._add_block(
            atoms, self.qbar[None, :].copy(), [exclusion_pairs(term_dict)], 0
        )
        return self.evaluate()

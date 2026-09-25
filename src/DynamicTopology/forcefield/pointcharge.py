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

from DynamicTopology.forcefield.ewald import Ewald, MinimumImage
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
    """Minimum-image displacements `r_i - r_j`, `(n, n, 3)`, and their lengths."""
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

    def get(self, pos, vecs, rij, pbc, cell):
        if not np.all(pbc):
            return MinimumImage(rij, vecs)
        cell = np.asarray(cell, dtype=float)
        key = cell.tobytes()
        if self.ewald is None or key != self.key:
            self.ewald = Ewald(cell)
            self.key = key
        return self.ewald.bind(pos, vecs, rij)


def pair_gradients(
    i: np.ndarray, j: np.ndarray, dE_dr: np.ndarray, vecs: np.ndarray, rij: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Forces and virial of a sum over the pairs `(i[p], j[p])`, one term each.

    `dE_dr[p]` is the radial derivative of pair `p`'s term.  `vecs` is `r_i -
    r_j`, so `dE/dr_i = dE_dr * v / r` and minus that on `j`; the virial is
    `sum_p dE_dr v_a v_b / r`, the same per-pair gradient against strain.
    """
    forces = np.zeros((len(rij), 3))
    if len(i) == 0:
        return forces, np.zeros((3, 3))
    v = vecs[i, j]
    coeff = dE_dr / rij[i, j]
    grad = coeff[:, None] * v
    np.add.at(forces, i, -grad)
    np.add.at(forces, j, grad)
    return forces, np.einsum("p,pa,pb->ab", coeff, v, v)


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


class PointCharge:
    """Fixed per-template charges, on the EVB diagonal.  See the module docstring.

    The interface is the one `System.calculate` drives both electrostatic terms
    through: `prepare` at the geometry, `bind` to the blocks, then `corrections`
    / `update` per block until self-consistent, then `evaluate`.  `__call__` is
    the single-topology case every other caller wants -- the fitter, `evb.py`.

    Unlike `ACKS2` everything is in global atom order: there is no linear system
    whose rows have to line up with a parameter vector, so there is no term
    order to confuse it with.
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
        self.vecs = None
        self.rij = None
        self.kernel = None
        self.K = None
        self.qbar = None
        self.phi = None
        self.blocks: list[_Block] = []

    # -- the geometry ---------------------------------------------------------

    def prepare(self, pos, pbc, cell, term_dict: dict) -> None:
        """Build the kernel at this geometry and take the seed topology's charges.

        The seed's charges are the starting environment for every block, and
        remain the charges of any atom no block claims.
        """
        vecs, rij = geometry(pos, pbc, cell)

        self.natoms = len(pos)
        # Every atom takes part, in global order; `act`/`local` are the index
        # maps `ACKS2` needs and the admission gate reads off either term.
        self.act = np.arange(self.natoms)
        self.local = self.act
        self.vecs = vecs
        self.rij = rij
        self.kernel = self.kernels.get(pos, vecs, rij, pbc, cell)
        self.K = self.kernel.matrix()

        indices, q = template_charges(term_dict)
        self.qbar = self._scatter(indices, q, "the current topology")
        self.phi = self.CCOUL * (self.K @ self.qbar)
        self.blocks = []

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
        self.phi = self.CCOUL * (self.K @ self.qbar)

    def _intra(self, block: _Block) -> np.ndarray:
        """Each state's Coulomb energy within the block, exclusions removed, in eV."""
        if block.intra is None:
            b = block.atoms
            Kbb = self.K[np.ix_(b, b)]
            gamma = active().gamma
            intra = np.empty(len(block.charges))
            for s, (q, pairs) in enumerate(zip(block.charges, block.pairs)):
                energy = 0.5 * float(q @ Kbb @ q)
                if len(pairs):
                    i, j = pairs[:, 0], pairs[:, 1]
                    g, _ = direct_kernel(self.rij[b[i], b[j]], gamma)
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
        b = block.atoms
        own = self.CCOUL * (self.K[np.ix_(b, b)] @ self.qbar[b])
        environment = (self.phi[b] - own) * units.eV
        return self._intra(block) + block.charges @ environment

    def update(self, index: int, weights: np.ndarray) -> None:
        """Adopt block `index`'s new ground-state weights as its averaged charges."""
        block = self.blocks[index]
        block.weights = np.asarray(weights, dtype=float)
        b = block.atoms
        new = block.weights @ block.charges
        self.phi += self.CCOUL * (self.K[:, b] @ (new - self.qbar[b]))
        self.qbar[b] = new

    # -- evaluation -----------------------------------------------------------

    def evaluate(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Energy, forces and virial of the whole system at the current weights.

        The weights are held fixed under the derivative, which is the
        Hellmann-Feynman statement and is exact once `System` has iterated the
        blocks to self-consistency.
        """
        ccoul = self.CCOUL
        W = 0.5 * ccoul * np.outer(self.qbar, self.qbar)
        pair_i, pair_j, pair_c = [], [], []
        for block in self.blocks:
            b = block.atoms
            if len(block.charges) > 1:
                # Within a block the states are alternatives, not a mixture:
                # sum_s w_s q_s q_s^T, which is not qbar qbar^T.
                W[np.ix_(b, b)] = (
                    0.5
                    * ccoul
                    * np.einsum(
                        "s,si,sj->ij", block.weights, block.charges, block.charges
                    )
                )
            for w, q, pairs in zip(block.weights, block.charges, block.pairs):
                if w == 0.0 or len(pairs) == 0:
                    continue
                i, j = pairs[:, 0], pairs[:, 1]
                pair_i.append(b[i])
                pair_j.append(b[j])
                pair_c.append(ccoul * w * q[i] * q[j])

        energy = float(np.sum(W * self.K))
        dS_dr, dS_de = self.kernel.contract(W)
        forces = -dS_dr
        virial = dS_de

        if pair_i:
            i = np.concatenate(pair_i)
            j = np.concatenate(pair_j)
            c = np.concatenate(pair_c)
            g, dg = direct_kernel(self.rij[i, j], active().gamma)
            # The exclusion is subtracted, so its radial derivative is `-c g'`.
            energy -= float(np.sum(c * g))
            f_pair, w_pair = pair_gradients(i, j, -c * dg, self.vecs, self.rij)
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

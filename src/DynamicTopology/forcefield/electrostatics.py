"""Which electrostatic term a dataset uses: `params.ForceFieldParams.electrostatics`.

Resolved at call time, like every other global parameter, because the fitter
builds its force fields at import -- before any manifest has said which one it
was fitted with.
"""

from __future__ import annotations

import numpy as np
from ase import units

from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.params import ForceFieldParams, active, resolve
from DynamicTopology.forcefield.pointcharge import (
    PointCharge,
    direct_kernel,
    pair_gradients,
)

_CLASSES = {"acks2": ACKS2, "pointcharge": PointCharge}


class Electrostatics:
    """One instance of each term, handing out whichever the active set names.

    Each instance carries a per-cell Ewald cache, so switching between them must
    not rebuild it; keeping one of each is what `System`, `EVBSystem` and the
    fitter all want.
    """

    def __init__(self):
        self._instances: dict[str, object] = {}

    def get(self, params: ForceFieldParams | None = None):
        name = resolve(params).electrostatics
        instance = self._instances.get(name)
        if instance is None:
            instance = self._instances[name] = _CLASSES[name]()
        return instance

    def __call__(self, pos, pbc, cell, term_dict: dict):
        return self.get()(pos, pbc, cell, term_dict)


def fixed_charges(terms) -> dict[int, float]:
    """`{global index: charge}` a term list fixes: `charge.q`, else `atom.q0`.

    The charges the admission gate screens with.  Under `pointcharge` they are
    the charges the diagonal uses; under ACKS2 they are the reference charges
    the equilibration starts from, which carry each molecule's formal charge and
    none of its polarization.
    """
    out: dict[int, float] = {}
    for term in terms:
        if term["type"] == "charge":
            out[next(iter(term["atoms"].values()))] = float(term["kwargs"]["q"])
        elif term["type"] == "atom":
            out.setdefault(
                next(iter(term["atoms"].values())), float(term["kwargs"].get("q0", 0.0))
            )
    return out


class ElectrostaticGap:
    """What a reaction changes in the fixed-charge Coulomb energy, for the gate.

    `EVBBasis` admits a neighbouring diabat on the stabilization of the two-level
    block it forms with its parent, which depends on the gap between their
    diagonals.  Without this the gap is the bonded one alone, so a hop that the
    surrounding charges make favourable is screened as though it were in vacuum.
    With it, the gap carries

        dE = E(child) - E(parent),   E(X) = CCOUL [1/2 q_X^T K q_X - sum_excl q_i q_j g_ij]

    over the whole system at fixed charges.  The two states differ only on the
    reacting fragment `M`, so the difference is local: the fragment's charges in
    the potential of everything else plus its own interaction, `O(|M| N)`.
    Being a difference of one function of the state, it is antisymmetric under
    exchanging the two, which is what the closure's seed-invariance argument
    needs.  (Atoms outside the block are held at the seed topology's charges,
    so another block's pivot does reach this one's gate.)

    The charges are `fixed_charges`: exact for `pointcharge`, the reference
    charges under ACKS2 -- a smooth surrogate for the gate, while the diagonal
    stays the full solve.
    """

    def __init__(self, reaction_set):
        self.reaction_set = reaction_set
        self.ff = None

    def _charged(self) -> bool:
        """Does any template carry a nonzero charge?  If not, the gap is zero.

        HCombustion's reference charges are all zero, and the fragment lookups
        this saves are most of what the gate would otherwise cost there.
        Re-derived when `ReactionSet.load` replaces the data.
        """
        data = self.reaction_set.data
        if getattr(self, "_charged_for", None) is not data:
            self._charged_for = data
            self._charged_value = any(
                value != 0.0
                for template in data.molecules.values()
                for value in fixed_charges(template.terms).values()
            )
        return self._charged_value

    def bind(self, ff, term_dict: dict) -> None:
        """Take a prepared electrostatic term's kernel and the seed's charges."""
        self.ff = ff
        self.q_seed = np.zeros(len(ff.act))
        for block in ("atom", "charge"):
            params = term_dict.get(block)
            if params is None:
                continue
            key = "q" if block == "charge" else "q0"
            values = params["kwargs"].get(key)
            if values is None:
                continue
            local = ff.local[params["atoms"][:, 0]]
            keep = local >= 0
            self.q_seed[local[keep]] = np.asarray(values, dtype=float)[keep]
        self.K = ff.kernel.matrix()
        self.enabled = self._charged()
        self._parents: dict = {}
        self._fragment_cache: dict = {}

    def _charges(self, topology) -> tuple[dict[int, float], np.ndarray]:
        terms = topology.terms or self.reaction_set.get_terms(topology)
        charges = fixed_charges(terms)
        pairs = [
            tuple(term["atoms"].values())[:2]
            for term in terms
            if term["type"] == "coulombexclusion"
        ]
        return charges, np.array(pairs, dtype=int).reshape(-1, 2)

    def _graph_charges(self, graph, atoms) -> tuple[np.ndarray, np.ndarray]:
        """A fragment graph's charges over `atoms` and its exclusions, memoized.

        Keyed on the bonding alone, which fixes the templates; the cache lives
        for one `bind`, i.e. one force call.
        """
        from DynamicTopology.basis import state_key
        from DynamicTopology.core.topology import Topology

        key = state_key(Topology(graph))
        cached = self._fragment_cache.get(key)
        if cached is None:
            charges, pairs = self._charges(Topology(graph.copy()))
            q = np.array([charges.get(a, 0.0) for a in atoms])
            cached = (q, np.searchsorted(atoms, pairs))
            self._fragment_cache[key] = cached
        return cached

    def _parent(self, parent) -> np.ndarray:
        from DynamicTopology.basis import state_key

        key = state_key(parent)
        q = self._parents.get(key)
        if q is None:
            q = self.q_seed.copy()
            charges, _ = self._charges(parent)
            for atom, value in charges.items():
                q[self.ff.local[atom]] = value
            self._parents[key] = q
        return q

    def _fragments(self, before, after):
        """The fragment's solver indices and, per side, its charges and exclusions.

        The exclusions come back as indices into the fragment.
        """
        atoms = np.array(sorted(before.nodes()), dtype=int)
        M = self.ff.local[atoms]
        return M, [self._graph_charges(graph, atoms) for graph in (before, after)]

    def energy(self, parent, before, after) -> float:
        """`E(child) - E(parent)` in eV; `before`/`after` are the fragment graphs."""
        if self.ff is None or not self.enabled:
            return 0.0
        M, sides = self._fragments(before, after)
        if not any(np.any(q) for q, _ in sides):
            return 0.0
        q_parent = self._parent(parent)
        rest = np.ones(len(q_parent), dtype=bool)
        rest[M] = False
        phi = self.K[np.ix_(M, np.flatnonzero(rest))] @ q_parent[rest]
        K_MM = self.K[np.ix_(M, M)]
        gamma = active().gamma
        energies = []
        for q, pairs in sides:
            e = float(q @ phi) + 0.5 * float(q @ K_MM @ q)
            if len(pairs):
                i, j = pairs[:, 0], pairs[:, 1]
                g, _ = direct_kernel(self.ff.rij[M[i], M[j]], gamma)
                e -= float(np.sum(q[i] * q[j] * g))
            energies.append(e)
        return active().ccoul * (energies[1] - energies[0]) * units.eV

    def gradients(self, parent, before, after) -> tuple[np.ndarray, np.ndarray]:
        """Forces (`-d dE/dr`, global order) and virial of `energy`.

        Scalar zeros where `energy` is identically zero, which broadcast.
        """
        if self.ff is None or not self.enabled:
            return 0.0, np.zeros((3, 3))
        M, sides = self._fragments(before, after)
        if not any(np.any(q) for q, _ in sides):
            return 0.0, np.zeros((3, 3))
        forces = np.zeros((len(self.ff.local), 3))
        q_parent = self._parent(parent)
        full = []
        for q, _ in sides:
            Q = q_parent.copy()
            Q[M] = q
            full.append(Q)
        W = 0.5 * (np.outer(full[1], full[1]) - np.outer(full[0], full[0]))
        dS_dr, dS_de = self.ff.kernel.contract(W)
        f_local, virial = -dS_dr, dS_de.copy()
        gamma = active().gamma
        # E(X) carries `-q_i q_j g` per excluded pair, and dE = E(child) - E(parent).
        for sign, Q, (_, pairs) in zip((-1.0, 1.0), full, sides):
            if not len(pairs):
                continue
            i, j = M[pairs[:, 0]], M[pairs[:, 1]]
            _, dg = direct_kernel(self.ff.rij[i, j], gamma)
            f_pair, w_pair = pair_gradients(
                i, j, -sign * Q[i] * Q[j] * dg, self.ff.vecs, self.ff.rij
            )
            f_local += f_pair
            virial += w_pair
        scale = active().ccoul * units.eV
        forces[self.ff.act] = scale * f_local / units.Angstrom
        return forces, scale * virial

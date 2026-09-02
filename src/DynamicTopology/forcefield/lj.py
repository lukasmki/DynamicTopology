"""Lennard-Jones repulsion and dispersion.  **Not in the active force field.**

Kept whole, and kept tested, but nothing in `System` or `EVBSystem` calls it.
The repulsion those need is `forcefield/zbl.py`; this module is here because it
is a complete and correct 12-6 with q-force's parameters and conventions, and
because the reason it was retired is worth being able to reproduce.

**Why it was retired.**  Not because a 12-6 is wrong, but because q-force's
parameters make it enormous at the separations reactive chemistry visits.  With
`sigma_H = 1.96 A` and `r**-12` hardness, a pair at a *bond length* is deep
inside the core:

    pair  r (A)  what it is              12-6      ZBL
    H-H   0.777  the H2 bond length    504 eV    2.0 eV
    O-H   0.960  the O-H bond length   930 eV    5.4 eV
    O-O   1.210  the O2 bond length   1348 eV   11.6 eV

A fixed-topology force field never notices, because those pairs are always
excluded.  A reactive one does: a diabatic state that has *broken* a bond calls
its two atoms different molecules while they sit at the bond length, and pays
hundreds of eV for it.  Every attempt to strip that penalty made the repulsion
differ between diabatic states, and each failed differently -- stripping it from
the unreacted parent as well let the box fuse at 0.60 A; gating it on the
coupling left a 19 eV plateau across rxn_16's reaction path where a wall was
half-removed; and transition states with close non-bonded contacts came back
with EVB amplitudes of -72 to -108 eV.  `ZBL` is 100 to 1000 times smaller at
exactly these separations, which is what lets it be applied to every pair with
no exclusions at all -- and therefore identically on every diabat, which is the
property that makes all four of those failure modes impossible rather than
fixed.

The functions that carried those four attempts -- `reactive_exclusion_weights`,
`fixed_basis_weights`, `exclusion_correction`, `lost_exclusion_correction` --
are gone.  What remains is the Lennard-Jones itself.

**The decomposition, for when this is used.**  LJ depends on the topology only
through exclusions, and exclusions are local to a molecule, so

    E_LJ(state) = sum_{i<j, far apart along the bonds} u_ij
                = sum_{i<j} u_ij  -  sum_{i<j, near along the bonds} u_ij

The first sum is the same for every diabatic state -- it does not know what is
bonded to what -- and is what `LennardJones.__call__` evaluates.  The second is
carried by `QForce.compute_exclusion` terms, derived by `with_exclusions` rather
than shipped in the `.jsonl` so that sigma and epsilon are stated exactly once.
An isolated small molecule therefore gets *exactly* zero: every pair is within
`EXCLUSION_DEPTH` bonds, so the two sums cancel term by term.

Units are q-force's -- sigma in nm, epsilon in kJ/mol, as the XML emits them --
converted to ASE units (eV, Angstrom) at the end of `__call__`, exactly as
`QForce` does.
"""

import networkx as nx
import numpy as np
from ase import units


# How far along the bond graph the Lennard-Jones interaction is switched off, in
# bonds.  3 excludes 1-2, 1-3 and 1-4 pairs, so the term acts between atoms four
# or more bonds apart and between atoms in different molecules -- GROMACS's
# `nrexcl = 3`, and q-force's own convention.
#
# On HCombustion this is indistinguishable from excluding whole molecules, since
# no template has graph diameter above 3 (H2O2's H...H is the one 1-4 pair, at
# 2.593 A and -0.0012 eV).  It bites on anything larger.
EXCLUSION_DEPTH: int = 3


# Where 12-6 stops being evaluated as `r**-12` and continues as its own tangent,
# as a fraction of sigma.  See `pair_potential`.
#
# 0.4 sigma is 0.78 A for an H-H pair and 1.18 A for O-O -- inside the Morse
# core of a real bond and far inside any intermolecular contact, so the switch
# never engages on a pair whose repulsion is doing physical work.  The wall
# there is already 451 eV (H-H) and 1750 eV (O-O); nothing intermolecular
# arrives at such a separation with the term switched on.
CORE_FRACTION: float = 0.4


def pair_potential(
    r: np.ndarray, sigma: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """12-6 and its radial derivative, `(u, du/dr)`.

    Unit-agnostic: `u` comes back in whatever `eps` is given in and `du/dr` in
    that per whatever `sigma` and `r` are given in.  The whole-system sum and
    `QForce.compute_exclusion` work in q-force's nm; the correction works in
    Angstrom.  Every caller goes through here so the forms can never drift apart
    -- and they must not, because the near-neighbour half of the sum is
    cancelled by `exclusion` terms built from this same form.

    **Below `CORE_FRACTION * sigma` the potential continues along its own
    tangent instead of as `r**-12`, and this is not cosmetic.**  The
    decomposition `E = sum_all_pairs - sum_near_pairs` evaluates every *bonded*
    pair in both sums, at a separation where 12-6 is enormous.  At equilibrium
    that is merely ugly -- 843 eV cancelling to zero on an H2 template, with
    thirteen digits to spare.  On a hot trajectory it is fatal: an H2 bond
    compressed to 0.3 A puts 1e8 eV into both sums, 0.2 A puts in 1e10, and
    0.024 A puts in 1e21.  Measured on the 3000 K probe, the two sums reached
    1e21 by step 143 and their difference -- the physical energy, of order 1e2
    -- came back quantized to 2**22 eV.  The forces went with it and the box
    heated to 1e16 K.

    Linear continuation bounds what a compressed bond can contribute (about 4500
    eV at 0.3 A rather than 8e7, and 5700 eV at 0.024 A rather than 1e21) while
    staying C1, so the cancellation keeps its precision and the gradient stays
    the gradient.  Because both halves come through here, the switch cancels
    exactly on any pair that is excluded, which is every pair it can reach.
    """
    core = CORE_FRACTION * sigma
    # Evaluate the true form at max(r, core) -- so the branch below never sees
    # the divergence at all -- then extrapolate back along the tangent.
    r_eval = np.maximum(r, core)
    sr6 = (sigma / r_eval) ** 6
    sr12 = sr6 * sr6
    u = 4.0 * eps * (sr12 - sr6)
    du_dr = -(24.0 * eps / r_eval) * (2.0 * sr12 - sr6)
    # `du_dr` needs no branch: evaluated at `core` it is already the tangent's
    # constant slope, which is exactly the derivative of the linear piece.
    u = np.where(r < core, u + du_dr * (r - core), u)
    return u, du_dr


def _near_pairs(graph: nx.Graph, depth: int = EXCLUSION_DEPTH) -> set[tuple[int, int]]:
    """Node pairs within `depth` bonds of each other in `graph`, as `(i, j)`, i < j."""
    pairs: set[tuple[int, int]] = set()
    for node in graph:
        for other, distance in nx.single_source_shortest_path_length(
            graph, node, cutoff=depth
        ).items():
            if distance == 0 or other <= node:
                continue
            pairs.add((node, other))
    return pairs


def exclusion_terms(terms: list[dict]) -> list[dict]:
    """Terms that cancel the whole-system sum between near-neighbour atoms.

    `LennardJones` sums 12-6 over every pair in the system, bonded pairs
    included, because that sum is the same for every diabatic state and can
    therefore be taken once outside the EVB.  What is topology-dependent is
    which pairs should not have been counted, and that is the pairs within
    `EXCLUSION_DEPTH` bonds of each other -- a per-molecule quantity, and so
    expressible as ordinary terms that ride the template remapping and memoize
    per molecule.

    Derived rather than shipped in the `.jsonl` so that sigma and epsilon are
    stated exactly once, in the `lennardjones` terms.  A radius edited in the
    dataset then cannot leave a stale exclusion behind it, which would show up
    only as a template no longer reproducing its own energy.

    The combining rule is geometric in both parameters, matching q-force's
    `A=sqrt(A1*A2); B=sqrt(B1*B2)` and `LennardJones.__call__`.  The pair values
    are worked out once here rather than twice from different code, because the
    cancellation has to be exact and not merely close.
    """
    parameters = {
        next(iter(term["atoms"].values())): term["kwargs"]
        for term in terms
        if term["type"] == "lennardjones"
    }
    if not parameters:
        return []

    # The same construction `Topology.from_terms` uses, so the graph here is the
    # one the rest of the code will see.
    graph = nx.Graph()
    graph.add_nodes_from(parameters)
    graph.add_edges_from(
        tuple(term["atoms"].values()) for term in terms if term["type"] == "bond"
    )

    exclusions: list[dict] = []
    for i, j in sorted(_near_pairs(graph)):
        sigma = np.sqrt(parameters[i]["sigma"] * parameters[j]["sigma"])
        eps = np.sqrt(parameters[i]["eps"] * parameters[j]["eps"])
        if sigma == 0.0 or eps == 0.0:
            continue  # the whole-system term contributes exactly zero
        exclusions.append(
            {
                "type": "exclusion",
                "atoms": {"p1": i, "p2": j},
                "kwargs": {"sigma": float(sigma), "eps": float(eps)},
            }
        )
    return exclusions


def with_exclusions(terms: list[dict]) -> list[dict]:
    """`terms` plus its exclusions, unless it already states them.

    Anything that evaluates a term list against a reference energy has to go
    through here.  A raw `.jsonl` carries `lennardjones` parameters but no
    exclusions -- those are derived when the dataset loads -- so scoring one
    directly charges it the whole-system sum with nothing cancelling it: H2 came
    out at +842 eV against a reference of -4.67.
    """
    if any(term["type"] == "exclusion" for term in terms):
        return list(terms)
    return list(terms) + exclusion_terms(terms)


def lj_parameters(terms: list[dict], natoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-atom `(sigma, eps)` in ASE units, from a term list.

    q-force states them in nm and kJ/mol; everything outside this module works
    in Angstrom and eV.  They are per *element*, so any topology of a given
    system yields the same arrays -- which is what lets the whole-system sum be
    taken once, outside the EVB, without becoming a pivot dependence.
    """
    sigma = np.zeros(natoms)
    eps = np.zeros(natoms)
    for term in terms:
        if term["type"] != "lennardjones":
            continue
        index = next(iter(term["atoms"].values()))
        sigma[index] = term["kwargs"]["sigma"] * 10.0
        eps[index] = term["kwargs"]["eps"] * units.kJ / units.mol
    return sigma, eps


class LennardJones:
    """12-6 with geometric combining, over all pairs.

    Two index spaces meet here, as in `ACKS2`, and confusing them is silent:
    the per-atom parameters arrive in *term order* -- the order the
    `lennardjones` terms were collected -- while `pos` and the returned forces
    are in *global* atom order.  `indices[k]` is the global index of the k-th
    term.  Everything is built in term order and the forces are scattered back
    at the very end.

    Combining is geometric in both parameters, matching q-force's
    `A=sqrt(A1*A2); B=sqrt(B1*B2)` -- *not* Lorentz-Berthelot.  Using the
    arithmetic mean for sigma here would silently disagree with
    `compute_exclusion`, and the disagreement would show up only as templates
    no longer reproducing their own energies.
    """

    def __call__(
        self, pos: np.ndarray, pbc: np.ndarray, cell: np.ndarray, term_dict: dict
    ) -> tuple[float, np.ndarray]:
        params = term_dict.get("lennardjones")
        if params is None:
            raise KeyError(
                "No `lennardjones` parameters set. Every atom needs a sigma and "
                "an epsilon, or the system has no Pauli repulsion and molecules "
                "will interpenetrate."
            )
        indices = params["atoms"][:, 0]
        sigma = params["kwargs"]["sigma"]
        eps = params["kwargs"]["eps"]

        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell

        # Both axes in term order, so the parameter vectors line up with them.
        sub = np.ix_(indices, indices)
        vecs = vecs[sub] / 10.0  # angstrom to nm
        rij = np.sqrt(np.sum(vecs * vecs, -1))

        diag = np.diag_indices(len(indices))
        r = rij.copy()
        r[diag] = 1.0  # excluded below; only keeps the division finite

        sig = np.sqrt(sigma[:, None] * sigma[None, :])
        epsilon = np.sqrt(eps[:, None] * eps[None, :])

        e, du_dr = pair_potential(r, sig, epsilon)
        e[diag] = 0.0
        du_dr[diag] = 0.0
        e_tot = 0.5 * float(np.sum(e)) * units.kJ / units.mol
        # The 0.5 above cancels because both (i, j) and (j, i) contribute to
        # dE/d(pos_i), which is the same convention `ACKS2.compute_coulomb` uses.
        nij = vecs / r[:, :, None]
        f_tot = -np.sum(du_dr[:, :, None] * nij, axis=1)
        f_tot = f_tot * units.kJ / units.mol / units.nm

        forces = np.zeros_like(pos)
        forces[indices] = f_tot
        return e_tot, forces

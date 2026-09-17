"""Intramolecular exclusions for the three whole-system nonbonded terms.

`LennardJones`, `ZBL` and `ACKS2` are each summed over *every* pair in the
system with no reference to the bond graph, because a sum that does not depend
on the topology is the same number on every diabatic state and can therefore be
evaluated once, outside the EVB Hamiltonian.  That property is the reason those
three terms are where they are, and nothing here changes it: the whole-system
sums stay exactly as they were.

What this module adds is the *correction* -- which pairs should not have been
counted.  That is the pairs within `lj.EXCLUSION_DEPTH` bonds of each other, and
it is a per-molecule quantity, which is what makes it expressible as a term and
therefore evaluable per diabatic state alongside the bonded ones.  This is the
decomposition `QForce.compute_exclusion` was already written for; it had been
dormant because `ReactionSet.load` stopped deriving the terms.

**Why the exclusions are wanted.**  Without them a template's gas-phase geometry
is set by the bonded terms *and* by whatever the three nonbonded sums happen to
contribute at bond lengths, and `fit/dissociation.py` has to absorb the
difference into the Morse depths.  For H3O+ that absorption failed outright: the
fitted depth came out at 5.76 eV against a 5.21 eV tapered-ZBL step per O-H, so
the true minimum of the isolated cation sat at 1.60 A and the symmetric C3v
structure was a *saddle*.  With the intramolecular nonbonded removed, the
gas-phase minimum of a small template is the q-force potential's own minimum and
nothing has to cancel anything.

**Why Coulomb takes a different route.**  `exclusion` and `zblexclusion` are
additive pair corrections, so `QForce` evaluates them per diabatic state
alongside the bonded terms and nothing about the architecture changes.  Coulomb
is not additive.  ACKS2's charges come from a linear solve whose matrix
*contains* the kernel, so the exclusion cannot simply be subtracted from the
energy and left out of the equilibration.

The obvious repair is to mask the kernel inside the solve.  It is consistent --
a screened kernel is simply another kernel in the sense `forcefield/ewald.py`
means, with energy, forces and virial all derived from it -- and it is wrong
here for a reason that has nothing to do with consistency: it makes the charges
a function of the bond graph.  `System.calculate` evaluates ACKS2 once, outside
the EVB Hamiltonian, precisely because it is the same number for every diabatic
state; topology-dependent charges evaluated once anyway make the total energy
depend on which state happens to be the pivot, measured at **0.88 eV** on
`tests/test_evb_invariants.py`.  Solving them per state instead would be
correct and would multiply the most expensive thing in a force call by the basis
size.

**What is done instead.**  The charges are solved once from the *unmasked*
kernel, and the exclusion is applied afterwards, in two places that are the same
quantity seen from two directions:

  - On each diabatic state's diagonal, as `-CCOUL sum_{excl} Q_i Q_j K_ij` --
    one lookup per excluded pair into a kernel matrix that has already been
    built (`ACKS2.exclusion_energy`).  This is what lets the exclusion decide
    which bonding pattern is lower, which is the whole point of putting it on a
    diagonal.
  - Once on the whole system, as a screen on the Coulomb *energy* functional.
    The screen is `1 - sum_s w_s M_s`, `w_s` the ground-state weights the EVB
    diagonalization just produced, so a pair bonded in every state of its block
    is removed outright and one bonded in some of them is removed in proportion.
    That is not an interpolation: `sum_s w_s d(correction_s)/dr` is linear in
    the masks and the kernel contraction is linear in its weight, so the whole
    Hellmann-Feynman sum collapses into a single weight matrix.  One contraction
    and one adjoint solve serve the entire system, which is the same cost the
    unexcluded term always had.

`ACKS2.prepare`/`ACKS2.compute` is the split that makes the ordering possible,
and `ACKS2.compute_response_forces` carries the one asymmetry that has to be got
right: `dE/dQ` is screened because the energy is, while the `-lam^T (dA/dr) x`
weight is not, because `A` was never screened.

Two consequences worth knowing about.  The charges are exactly what they were
before exclusions existed -- an isolated water still comes out at `q_H =
+0.30399` -- so nothing about a molecule's dipole moves.  And the exclusion is
the first thing in this codebase to contract the periodic charge kernel against
a *non-neutral* weight, which is what exposed the missing `k = 0` background in
`forcefield/ewald.py`; an individual `K_ij` was not a well-defined number until
that was added.

**What it does cost is the intramolecular half of the polarization response,
and that is a real 0.07 eV of the hydrogen bond.**  The water dimer well goes
from 0.1677 eV at 2.85 A to 0.1024 eV at 2.91 A -- measured with the datasets
held fixed, so this is the mechanism and not a refit -- decomposed at 2.85 A as

    dimerization gain, unscreened ACKS2      -0.2063 eV
      of which intramolecular self-energy    -0.0691 eV   <- removed here
    q_H, monomer -> dimer                +0.30399 -> +0.31264

The molecules polarize each other, the charges grow, and the *intramolecular*
Coulomb energy falls along with the intermolecular one.  Booking that gain
inside the molecule is exactly what the exclusion exists to stop -- the bonded
terms own the intramolecular energy, and a template whose gas-phase minimum
depends on its neighbours is the thing this module was written to remove -- but
the binding it took with it has to come back somewhere else.  `eta` is the lever
(`datasets/Water/README.md`): the intermolecular term scales as `q**2`, so
softening the hardness recovers the depth without reintroducing the coupling.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from DynamicTopology.forcefield.lj import EXCLUSION_DEPTH, _near_pairs

# Whether to emit `coulombexclusion` terms.  On: see the module docstring for
# how they are applied, which is not by masking the charge solve.  Turning this
# off drops the Coulomb exclusion and keeps the other two, which is a dataset
# the whole pipeline still evaluates consistently -- but not one that can be
# fitted: the H3O+ template is unfittable without it, because ACKS2 scores the
# cation as a neutral H3O and the -4.8 eV of spurious binding that produces has
# nothing left to cancel it once ZBL's intramolecular +15.45 eV is excluded.
EXCLUDE_COULOMB: bool = True


def bond_graph(terms: list[dict], natoms: int | None = None) -> nx.Graph:
    """The bond graph a term list describes.

    The same construction `Topology.from_terms` uses, so the graph here is the
    one the rest of the code will see.  Nodes come from the `atom` terms rather
    than from the bonds, so an isolated atom -- `h` and `o` in the Water set --
    is still a node and still yields no pairs.
    """
    graph = nx.Graph()
    for term in terms:
        if term["type"] in ("atom", "lennardjones"):
            graph.add_node(next(iter(term["atoms"].values())))
    if natoms is not None:
        graph.add_nodes_from(range(natoms))
    graph.add_edges_from(
        tuple(term["atoms"].values()) for term in terms if term["type"] == "bond"
    )
    return graph


def exclusion_terms(
    terms: list[dict],
    numbers: np.ndarray | None = None,
    depth: int = EXCLUSION_DEPTH,
) -> list[dict]:
    """Exclusion terms for every pair within `depth` bonds, for all three sums.

    `numbers` supplies the atomic numbers the `zblexclusion` terms need; ZBL
    reads them from the `Atoms` rather than from a template, so they have to be
    handed in here.  Omitting it drops the ZBL exclusions and keeps the rest,
    which is what a caller scoring only the Lennard-Jones decomposition wants.

    A pair contributes a `lennardjones` exclusion only if both atoms carry a
    nonzero sigma and epsilon: with `sigma_H = 0` the whole-system term is
    identically zero on that pair, so there is nothing to cancel and a term
    would be an expensive no-op.  ZBL and Coulomb have no such escape -- ZBL has
    no free parameters and the charge kernel is element-independent -- so every
    pair gets one of each.
    """
    lj_params = {
        next(iter(term["atoms"].values())): term["kwargs"]
        for term in terms
        if term["type"] == "lennardjones"
    }
    graph = bond_graph(terms)
    out: list[dict] = []
    for i, j in sorted(_near_pairs(graph, depth)):
        atoms = {"p1": i, "p2": j}

        if i in lj_params and j in lj_params:
            sigma = np.sqrt(lj_params[i]["sigma"] * lj_params[j]["sigma"])
            eps = np.sqrt(lj_params[i]["eps"] * lj_params[j]["eps"])
            if sigma != 0.0 and eps != 0.0:
                out.append(
                    {
                        "type": "exclusion",
                        "atoms": atoms,
                        "kwargs": {"sigma": float(sigma), "eps": float(eps)},
                    }
                )

        if numbers is not None:
            out.append(
                {
                    "type": "zblexclusion",
                    "atoms": atoms,
                    "kwargs": {
                        "z1": float(numbers[i]),
                        "z2": float(numbers[j]),
                    },
                }
            )

        if EXCLUDE_COULOMB:
            out.append({"type": "coulombexclusion", "atoms": atoms, "kwargs": {}})
    return out


def with_exclusions(
    terms: list[dict],
    numbers: np.ndarray | None = None,
    depth: int = EXCLUSION_DEPTH,
) -> list[dict]:
    """`terms` plus its exclusions, unless it already states them.

    Anything that evaluates a term list against a reference energy has to go
    through here, for the reason `lj.with_exclusions` gives: a raw `.jsonl`
    carries nonbonded *parameters* but no exclusions, so scoring one directly
    charges it the whole-system sums with nothing cancelling them.
    """
    if any(term["type"].endswith("exclusion") for term in terms):
        return list(terms)
    return list(terms) + exclusion_terms(terms, numbers, depth)

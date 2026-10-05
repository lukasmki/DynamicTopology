"""Intramolecular exclusions for the three whole-system nonbonded terms.

`LennardJones`, `ZBL` and `ACKS2` are each summed over *every* pair in the
system with no reference to the bond graph, because a sum that does not depend
on the topology is the same number on every diabatic state and can therefore be
evaluated once, outside the EVB Hamiltonian.  That property is the reason those
three terms are where they are, and nothing here changes it: the whole-system
sums stay exactly as they were.  (`LennardJones` keeps it only while the
templates agree on each atom's sigma and epsilon; where they do not, the part
that differs goes on the diagonal -- see `forcefield/lj.py`.)

What this module adds is the *correction* -- which pairs should not have been
counted.  That is the pairs within `params.exclusion_depth` bonds of each other, and
it is a per-molecule quantity, which is what makes it expressible as a term and
therefore evaluable per diabatic state alongside the bonded ones.  This is the
decomposition `QForce.compute_exclusion` was already written for; it had been
dormant because `ReactionSet.load` stopped deriving the terms.

**Why the exclusions are wanted.**  Without them a template's gas-phase geometry
is set by the bonded terms *and* by whatever the three nonbonded sums happen to
contribute at bond lengths, and fast-forces' `refine` has to absorb the
difference into the Morse depths.  For H3O+ that absorption failed outright: the
fitted depth came out at 5.76 eV against a 5.21 eV tapered-ZBL step per O-H, so
the true minimum of the isolated cation sat at 1.60 A and the symmetric C3v
structure was a *saddle*.  With the intramolecular nonbonded removed, the
gas-phase minimum of a small template is the q-force potential's own minimum and
nothing has to cancel anything.

**Coulomb is the exception, and under ACKS2 it needs no exclusion at all.**
`exclusion` and `zblexclusion` are additive pair corrections, so `QForce`
evaluates them per diabatic state alongside the bonded terms.  Coulomb is not
additive under charge equilibration -- the charges come from a solve whose
matrix *contains* the kernel -- and the route this module used to lay out for it
(charges from the unmasked kernel, the exclusion applied afterwards as a screen
on the energy) left an energy that was not the one the charges minimize.  That
cost 0.07 eV of the water dimer's hydrogen bond and let ACKS2's intermolecular
charge transfer go unopposed at an H2 + O2 contact.

Fragment ACKS2 (`forcefield/acks2.py`) solves the charges per diabatic state and
reports each state's minimum *relative to its molecules' isolated minima*.  That
reference is the molecule's own gas-phase electrostatics, so subtracting it
keeps them off the bonded terms exactly -- the property the exclusion was for --
while the kernel inside the solve stays whole.  ACKS2 therefore ignores
`coulombexclusion` terms.

`PointCharge` still reads them: with fixed charges the Coulomb energy *is*
additive, and an excluded pair's direct `erf(gamma r)/r` is simply subtracted,
per state, leaving its periodic images in place.  For the templates shipped here
every intramolecular pair is within `exclusion_depth`, and the two routes then
give the same energy -- which `tests/test_acks2_fragment.py` holds them to.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from DynamicTopology.forcefield.lj import near_pairs
from DynamicTopology.forcefield.params import ForceFieldParams, resolve

# `EXCLUDE_COULOMB` is now `params.ForceFieldParams.exclude_coulomb`, and the
# depth these terms are derived to is `exclusion_depth`; see that module for
# why both belong to the dataset rather than to the installed source tree.


def bond_graph(terms: list[dict], natoms: int | None = None) -> nx.Graph:
    """The bond graph a term list describes.

    The same construction `Topology.from_terms` uses, so the graph here is the
    one the rest of the code will see.  Nodes come from the `atom` terms rather
    than from the bonds, so an isolated atom -- `h` and `o` in the Water set --
    is still a node and still yields no pairs.
    """
    graph = nx.Graph()
    for term in terms:
        if term["type"] in ("atom", "charge", "lennardjones"):
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
    depth: int | None = None,
    params: ForceFieldParams | None = None,
    graph: nx.Graph | None = None,
) -> list[dict]:
    """Exclusion terms for every pair within `depth` bonds, for all three sums.

    The pairs come from `graph` when it is given and from the `bond` terms
    otherwise.  A fitter wants the first: it needs the exclusions of a topology
    whose bond *parameters* are what it is solving for, so there are no bond
    terms yet to read the graph off.

    `depth` defaults to the active `exclusion_depth` and `params` to the active
    set, both resolved in the body: this module is imported long before any
    manifest is read, so a default argument would pin the wrong dataset's depth.

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
    ff = resolve(params)
    depth = ff.exclusion_depth if depth is None else depth
    lj_params = {
        next(iter(term["atoms"].values())): term["kwargs"]
        for term in terms
        if term["type"] == "lennardjones"
    }
    if graph is None:
        graph = bond_graph(terms)
    out: list[dict] = []
    for i, j in sorted(near_pairs(graph, depth)):
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

        if ff.exclude_coulomb:
            out.append({"type": "coulombexclusion", "atoms": atoms, "kwargs": {}})
    return out


def with_exclusions(
    terms: list[dict],
    numbers: np.ndarray | None = None,
    depth: int | None = None,
    params: ForceFieldParams | None = None,
) -> list[dict]:
    """`terms` plus its exclusions, unless it already states them.

    Anything that evaluates a term list against a reference energy has to go
    through here: a raw `.jsonl` carries nonbonded *parameters* but no
    exclusions, so scoring one directly charges it the whole-system sums with
    nothing cancelling them -- H2 came out at +842 eV against a reference of
    -4.67 the one time it was tried.  `evaluate.term_dict` does this for you.
    """
    if any(term["type"].endswith("exclusion") for term in terms):
        return list(terms)
    return list(terms) + exclusion_terms(terms, numbers, depth, params)

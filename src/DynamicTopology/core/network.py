"""
Instantaneous reaction network
"""

import logging

from typing import Self, Iterable
import networkx as nx
from .topology import Topology
from .reaction import Reaction

from .types import Term

logger: logging.Logger = logging.getLogger(__name__)


class ReactionNetwork:
    def __init__(self, graph: nx.MultiGraph, terms: list[Term] | None = None):
        self.graph: nx.MultiGraph = graph
        self.term_dict = {}
        self.terms = self.set_terms(terms) if terms else []

    def set_terms(self, terms: list[Term]):
        self.terms = terms

        # structure for later
        self.term_dict = {}
        for term in terms:
            term_type = term["type"]
            if term_type in self.term_dict:
                self.term_dict[term_type]["atoms"].append(tuple(term["atoms"].values()))
                for k, v in term["kwargs"].items():
                    self.term_dict[term_type]["kwargs"][k].append(v)
            else:
                self.term_dict[term_type] = {
                    "atoms": [tuple(term["atoms"].values())],
                    "kwargs": {},
                }
                for k, v in term["kwargs"].items():
                    self.term_dict[term_type]["kwargs"][k] = [v]
        return terms

    def reactions(self) -> Iterable[tuple[int, int, dict]]:
        return self.graph.edges(data=True)

    def subnetworks(self) -> Iterable[Self]:
        for subidx in nx.connected_components(self.graph):
            subgraph = self.graph.subgraph(subidx)
            yield ReactionNetwork(subgraph)

    def reaction_blocks(self) -> list[tuple[list[dict], list[Topology]]]:
        """Group reactions into independent EVB problems, plus the spectators.

        Two reactions belong to the same problem only if they act on overlapping
        atoms.  Reactions on disjoint atoms commute: the state in which both
        have occurred is the product of the two single-reaction states, so their
        blocks factorize and their energies add.  Putting them in one matrix
        instead makes the ground state depend on how many unrelated channels
        happen to be enumerated nearby -- a star of N degenerate leaves is
        stabilized by sqrt(N)*|V| -- which is not size consistent.

        Note this is a different partition from `subnetworks()`, which splits on
        *molecular proximity*: any two molecules within the bimolecular cutoff
        land in one component, dragging their unrelated reactions along with
        them.  Partitioning on shared atoms is what the formalism actually
        requires.

        Returns (reactions, molecules) per block, followed by one
        ([], [molecule]) entry for each molecule no reaction touches.
        """
        reactions: list[dict] = []
        endpoints: list[tuple[int, int]] = []
        for imol, jmol, rxn_data in self.graph.edges(data=True):
            reactions.append(rxn_data)
            endpoints.append((imol, jmol))

        # Union-find over reactions that share atoms.
        parent = list(range(len(reactions)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[rj] = ri

        atom_sets = [frozenset(r["mapping"].values()) for r in reactions]
        for i in range(len(reactions)):
            for j in range(i + 1, len(reactions)):
                if atom_sets[i] & atom_sets[j]:
                    union(i, j)

        molecules: dict[int, Topology] = dict(self.graph.nodes(data="molecule"))

        grouped: dict[int, list[int]] = {}
        for i in range(len(reactions)):
            grouped.setdefault(find(i), []).append(i)

        blocks: list[tuple[list[dict], list[Topology]]] = []
        claimed: set[int] = set()
        for members in grouped.values():
            mol_ids: list[int] = sorted(
                {m for i in members for m in endpoints[i]}
            )
            claimed.update(mol_ids)
            blocks.append(
                ([reactions[i] for i in members], [molecules[m] for m in mol_ids])
            )

        for mol_id, molecule in molecules.items():
            if mol_id not in claimed:
                blocks.append(([], [molecule]))

        return blocks

    def states(self, thresh=1e-6) -> list[list[tuple[dict | None, Topology]]]:
        """Diabatic states for each independent EVB problem (N_blocks, N_states).

        State 0 of each block is the block's current topology; the rest are that
        topology with one reaction applied.
        """
        states: list[list[tuple[dict | None, Topology]]] = []
        for reactions, molecules in self.reaction_blocks():
            S0: Topology = Topology.from_molecules(molecules, remap=False)

            # Several symmetry-equivalent mappings can lead to the same product
            # topology.  They are one diabatic state reached by equivalent
            # routes, not several states, so keep the most strongly coupled
            # representative: that is the route whose geometry is closest to the
            # transition state, and choosing on the coupling rather than on
            # enumeration order is what makes the choice independent of how the
            # atoms happen to be labelled.
            best: dict[frozenset, tuple[float, dict, Topology]] = {}
            for rxn_data in reactions:
                coupling = rxn_data.get("coupling_energy")
                if coupling is not None and abs(coupling) < thresh:
                    logger.info(
                        "Dropping negligibly coupled reaction %s",
                        rxn_data["reaction"],
                    )
                    continue

                rxn: Reaction = rxn_data["reaction"]
                Si: Topology = rxn.apply(S0, rxn_data["mapping"])
                key = frozenset(frozenset(edge) for edge in Si.graph.edges())
                strength = abs(coupling) if coupling is not None else 0.0
                if key not in best or strength > best[key][0]:
                    best[key] = (strength, rxn_data, Si)

            block_states: list[tuple[dict | None, Topology]] = [(None, S0)]
            block_states.extend(
                (rxn_data, Si) for _, rxn_data, Si in best.values()
            )

            states.append(block_states)
        return states

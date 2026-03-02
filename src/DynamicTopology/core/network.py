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

    def states(self, thresh=1e-6) -> list[list[tuple[dict | None, Topology]]]:
        """Returns a list of states for each Reaction subnetwork (N_subnets, N_states)"""
        states: list[list[tuple[dict | None, Topology]]] = []
        for network in self.subnetworks():
            molecules: list[Topology] = [
                m for _, m in network.graph.nodes(data="molecule")
            ]
            S0: Topology = Topology.from_molecules(molecules, remap=False)

            subnet_states: list[tuple[dict | None, Topology]] = [(None, S0)]
            for i, edge in enumerate(network.graph.edges(data=True)):
                imol, jmol, rxn_data = edge
                if "coupling" in rxn_data and abs(rxn_data["coupling"]) < thresh:
                    logger.info("Removing reaction", rxn_data["reaction"])
                    continue

                rxn: Reaction = rxn_data["reaction"]
                Si: Topology = rxn.apply(S0, rxn_data["mapping"])
                subnet_states.append((rxn_data, Si))

            states.append(subnet_states)
        return states

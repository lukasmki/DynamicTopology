from ase import Atoms

from copy import deepcopy
import networkx as nx
from typing import Self, Any
from .topology import Topology

from .types import Term


class Reaction:
    def __init__(
        self,
        reactants: Topology,
        products: Topology,
        atoms: Atoms | list[Atoms] | None = None,
        terms: list[Term] | None = None,
    ):
        self.reactants: Topology = reactants
        self.products: Topology = products
        self.atoms: list[Atoms] = atoms if isinstance(atoms, list) else [atoms]
        self.term_dict: dict[str, Any] = {}
        self.terms: list[Term] = self.set_terms(terms) if terms else []

    def __repr__(self):
        return f"Reaction({self.equation()})"

    def hash(self) -> tuple[str, str]:
        return self.reactants.hash(), self.products.hash()

    def reverse(self) -> Self:
        return self.__class__(self.products, self.reactants, self.atoms, self.terms)

    def set_terms(self, terms: list[Term]) -> list[Term]:
        self.terms: list[Term] = terms

        # structure into lists for later
        self.term_dict: dict[str, Any] = {}
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

    def get_mapping(self, reactants: Topology) -> dict:
        matcher = nx.isomorphism.GraphMatcher(
            self.reactants.graph,
            reactants.graph,
            node_match=lambda a, b: a["atomic_number"] == b["atomic_number"],
        ).match()
        mol_map = next(matcher)
        return mol_map

    def apply(self, topology: Topology, mapping: dict[int, int]):
        """Applies this reaction to topology and returns a new topology"""
        new_topology = topology.copy()
        R = nx.relabel_nodes(self.reactants.graph, mapping)
        P = nx.relabel_nodes(self.products.graph, mapping)
        broken = R.edges - P.edges
        formed = P.edges - R.edges
        new_topology.graph.remove_edges_from(broken)
        new_topology.graph.add_edges_from(formed)
        return new_topology

    def copy(self) -> Self:
        reactants: Topology = self.reactants.copy()
        products: Topology = self.products.copy()
        atoms: list[Atoms] = deepcopy(self.atoms)
        terms: list[Term] = deepcopy(self.terms)
        return self.__class__(reactants, products, atoms, terms)

    @classmethod
    def from_atoms(cls, atoms: list[Atoms], terms: list[Term] | None = None):
        R, TS, P = atoms[0], atoms[1:-1], atoms[-1]
        R_topo = Topology.from_atoms(R)
        P_topo = Topology.from_atoms(P)
        return cls(R_topo, P_topo, TS, terms)

    def equation(self) -> str:
        r: str = " + ".join(
            [
                a.get_chemical_formula()
                for _, a in self.reactants.molecules(return_atoms=True)  # ty: ignore
            ]
        )
        p: str = " + ".join(
            [
                a.get_chemical_formula()
                for _, a in self.products.molecules(return_atoms=True)  # ty: ignore
            ]
        )
        return f"{r} -> {p}"

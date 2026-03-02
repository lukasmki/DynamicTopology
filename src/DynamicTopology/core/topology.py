from typing import Self, Iterable, Any
from copy import deepcopy
import networkx as nx
from ase import Atoms
from molify import ase2networkx

from .types import Term


class Topology:
    def __init__(
        self,
        graph: nx.Graph,
        atoms: Atoms | None = None,
        terms: list[Term] | None = None,
    ):
        self.graph: nx.Graph = graph
        self.atoms: Atoms = self.set_atoms(atoms) if atoms else None
        self.term_dict: dict[str, Any] = {}
        self.terms: list[Term] = self.set_terms(terms) if terms else []

    def __repr__(self):
        return f"Topology(graph={str(self.graph)}, {self.atoms=})"

    def hash(self) -> str:
        return nx.weisfeiler_lehman_graph_hash(self.graph, node_attr="atomic_number")

    def molecules(
        self, return_atoms=False
    ) -> Iterable[tuple[Self, Atoms]] | Iterable[Self]:
        for subidx in nx.connected_components(self.graph):
            # yield subgraphs with global node indices
            # subgraphs connectivity can't be changed but attrs can
            subgraph = self.graph.subgraph(subidx)
            if return_atoms:
                sub_atoms = self.atoms[list(subidx)] if self.atoms else None
                yield Topology(graph=subgraph), sub_atoms
            else:
                yield Topology(graph=subgraph)

    def set_atoms(self, atoms: Atoms) -> Atoms:
        self.atoms = atoms
        self.graph.add_nodes_from(
            [(a.index, {"atomic_number": a.number, "symbol": a.symbol}) for a in atoms]
        )
        return atoms

    def set_terms(self, terms: list[Term]) -> list[Term]:
        self.terms = terms
        # structure into lists for later
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

    def copy(self) -> Self:
        graph: nx.Graph = deepcopy(self.graph)
        atoms = self.atoms.copy() if self.atoms else None
        terms = deepcopy(self.terms) if self.terms else []
        return self.__class__(graph, atoms, terms)

    @classmethod
    def from_atoms(cls, atoms: Atoms):
        graph = ase2networkx(atoms, False)
        for _, node_data in graph.nodes(data=True):
            node_data.pop("position", None)
            node_data.pop("charge", None)
        return cls(graph, atoms)

    @classmethod
    def from_molecules(cls, molecules: list[Self], remap=True) -> Self:
        """
        If remap is false, returns only a merger of the graphs. it will attempt to
        perform a disjoint union of the molecules and will throw an error if there are index overlaps.
        """
        copy_atoms: bool = all(m.atoms is not None for m in molecules)
        copy_terms: bool = any(len(m.terms) > 0 for m in molecules)
        graph: nx.Graph = molecules[0].graph.copy()
        if remap:
            # reindex to be first molecule in topology
            mapping = {n: i for i, n in enumerate(graph)}
            graph: nx.Graph = nx.relabel_nodes(graph, mapping)
            atoms: Atoms | None = molecules[0].atoms.copy() if copy_atoms else None
            terms: list[Term] = []
            if copy_terms:
                terms: list[Term] = deepcopy(molecules[0].terms)
                for term in terms:
                    for k, v in term["atoms"].items():
                        term["atoms"][k] = mapping[v]

            for mol in molecules[1:]:
                n: int = len(graph)
                mapping: dict[int, int] = {m: n + i for i, m in enumerate(mol.graph)}
                graph: nx.Graph = nx.union(graph, nx.relabel_nodes(mol.graph, mapping))

                if copy_atoms:  # remap atoms to new atoms object
                    atoms += mol.atoms.copy()

                if copy_terms:  # remap terms to new global indices
                    mol_terms: list[Term] = deepcopy(mol.terms)
                    for term in mol_terms:
                        for k, v in term["atoms"].items():
                            term["atoms"][k] = mapping[v]
                    terms.extend(mol_terms)

            return cls(graph, atoms, terms)
        else:
            for mol in molecules[1:]:
                graph: nx.Graph = nx.union(graph, mol.graph)
            return cls(graph, None, None)

    @classmethod
    def from_terms(cls, terms: list[Term], atoms: Atoms | None = None) -> Self:
        bonds: list[tuple[int, int]] = []
        for term in terms:
            if term["type"] == "bond":
                bonds.append(tuple(term["atoms"].values()))
        graph = nx.Graph(bonds)
        return cls(graph, atoms, terms)

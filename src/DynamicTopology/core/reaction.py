import numpy as np
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
        atoms: Atoms | list[Atoms],
        terms: list[Term] | None = None,
    ):
        self.reactants: Topology = reactants
        self.products: Topology = products
        self.atoms: list[Atoms] = atoms if isinstance(atoms, list) else [atoms]
        self.term_dict: dict[str, Any] = {}
        self.terms: list[Term] = self.set_terms(terms) if terms else []
        # (broken, formed) in *template* indices, filled on first use.  The
        # templates are immutable once loaded, so the diff is a property of the
        # reaction and not of the topology it is applied to.
        self._template_changes: tuple[list, list] | None = None

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

        # convert to numpy
        for term_type, term_data in self.term_dict.items():
            term_data["atoms"] = np.array(term_data["atoms"])
            for arg, vals in term_data["kwargs"].items():
                term_data["kwargs"][arg] = np.array(vals)

        return terms

    def _matcher(self, reactants: Topology):
        return nx.isomorphism.GraphMatcher(
            self.reactants.graph,
            reactants.graph,
            node_match=lambda a, b: a["atomic_number"] == b["atomic_number"],
        )

    def get_mapping(self, reactants: Topology) -> dict:
        """One mapping of the template onto `reactants`.

        Kept for callers that only need some valid correspondence.  Prefer
        `get_mappings` anywhere the choice is observable -- see its docstring.
        """
        return next(self._matcher(reactants).isomorphisms_iter())

    def get_mappings(self, reactants: Topology) -> list[dict]:
        """Every distinct way the template maps onto `reactants`.

        Symmetry-equivalent atoms admit several isomorphisms: H2O has two ways
        to be the reactant of `H2O -> HO + H`, one per O-H bond, and
        `O2 + H2 -> HO2 + H` has four.  Taking only the first makes the choice
        depend on the order nodes happen to be stored in, which breaks
        permutation invariance of the surface -- the energy may coincide by
        symmetry while the forces land on different atoms -- and silently
        discards the channel that is geometrically active whenever the arbitrary
        pick is the wrong one.

        Returned in a deterministic order.  Channels leading to the same product
        topology are still distinct here because they differ in which atoms
        correspond to which transition-state reference positions, and so carry
        different couplings; collapsing them is the caller's job, once those
        couplings are known.
        """
        return list(self._matcher(reactants).isomorphisms_iter())

    def edge_changes(self, mapping: dict[int, int]) -> tuple[set, set]:
        """(broken, formed) bonds this reaction makes, in live indices.

        Depends only on the templates and the mapping, never on the topology
        being acted upon, so a caller screening many candidates can rewire a
        small fragment with this instead of copying a whole block through
        `apply`.

        Because of that, the *template* diff is computed once and only the
        relabelling is per call.  Taking the difference through
        `nx.relabel_nodes` instead built two whole graphs per candidate channel,
        which on a 64-water box is 3888 pairs of graph copies per force call --
        more than the couplings they were being screened for.  Nodes absent from
        `mapping` keep their template index, as `relabel_nodes` left them.
        """
        if self._template_changes is None:
            reactant_edges = {
                (u, v) if u <= v else (v, u) for u, v in self.reactants.graph.edges()
            }
            product_edges = {
                (u, v) if u <= v else (v, u) for u, v in self.products.graph.edges()
            }
            self._template_changes = (
                sorted(reactant_edges - product_edges),
                sorted(product_edges - reactant_edges),
            )

        broken, formed = self._template_changes
        get = mapping.get
        return (
            {(get(u, u), get(v, v)) for u, v in broken},
            {(get(u, u), get(v, v)) for u, v in formed},
        )

    def apply(
        self, topology: Topology, mapping: dict[int, int], share_atoms: bool = False
    ):
        """Applies this reaction to topology and returns a new topology.

        `share_atoms` is forwarded to `Topology.copy`: the product describes the
        same nuclei at the same positions, so a caller enumerating many products
        at one fixed geometry can avoid copying the `Atoms` each time.
        """
        new_topology = topology.copy(share_atoms=share_atoms)
        broken, formed = self.edge_changes(mapping)
        new_topology.graph.remove_edges_from(broken)
        new_topology.graph.add_edges_from(formed)
        # The product is bonded differently from the reactant, so the reactant's
        # terms do not describe it.  `Topology.copy` carries terms across, and a
        # caller that looks them up lazily ("fetch only if empty") would then
        # silently evaluate the reactant's energy at the product's topology --
        # every state in a block coming out at the same energy.
        new_topology.set_terms([])
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

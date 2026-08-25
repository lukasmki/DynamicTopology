import numpy as np
from typing import Self, Iterable, Any
from copy import deepcopy
import networkx as nx
from ase import Atoms
from molify import ase2networkx

from .types import Term


# Bond perception cutoff as a multiple of the summed covalent radii.
#
# `molify`'s default of 1.2 puts the H-H cutoff at 1.2 * 2 * 0.31 = 0.7440 A,
# and equilibrium H2 is 0.7445 A -- outside it by 5e-4 A.  Every H2 at or beyond
# its own bond length therefore perceived as two free atoms: the 100-molecule
# H2/O2 box in tests/data came out with 50 bonds and 150 fragments instead of
# 100 and 100, and a plain .xyz of H2 scored 0.0 eV against a reference of
# -4.6701.  It stayed hidden because every file in the repo carries an explicit
# `connectivity` list, which perception prefers.
#
# 1.25 already recovers that box exactly and 1.35 still fuses nothing, so 1.3 is
# the middle of the plateau rather than either edge of it.
BOND_SCALE: float = 1.3


class Topology:
    def __init__(
        self,
        graph: nx.Graph,
        atoms: Atoms | None = None,
        terms: list[Term] | None = None,
    ):
        self.graph: nx.Graph = graph
        self._hash: str | None = None
        self._molecules: list[frozenset] | None = None
        self.atoms: Atoms | None = self.set_atoms(atoms) if atoms else None
        self.terms: list[Term] = terms if terms is not None else []
        self.term_dict: dict[str, Any] = (
            self.set_terms(terms) if terms is not None else {}
        )

    def __repr__(self):
        return f"Topology(graph={str(self.graph)}, {self.atoms=})"

    def hash(self) -> str:
        if self._hash is None:
            self._hash = nx.weisfeiler_lehman_graph_hash(
                self.graph, node_attr="atomic_number"
            )
        return self._hash

    def molecules(
        self, return_atoms=False
    ) -> Iterable[tuple[Self, Atoms]] | Iterable[Self]:
        if self._molecules is None:
            self._molecules = list(nx.connected_components(self.graph))
        for subidx in self._molecules:
            # yield subgraphs with global node indices
            # subgraphs connectivity can't be changed but attrs can
            subgraph = self.graph.subgraph(subidx)
            if return_atoms:
                sub_atoms = self.atoms[list(subidx)] if self.atoms else None
                yield Topology(graph=subgraph), sub_atoms
            else:
                yield Topology(graph=subgraph)

    def set_atoms(self, atoms: Atoms) -> Atoms:
        """Attach `atoms` and make the graph span all of them.

        Every atom becomes a node, so this is the whole-system operation: it
        says "this topology describes exactly these atoms".  For a topology that
        covers only part of a system -- a molecule, or one EVB block -- use
        `attach_atoms` instead.
        """
        self._hash = None
        self._molecules = None
        self.atoms = atoms
        self.graph.add_nodes_from(
            [(a.index, {"atomic_number": a.number, "symbol": a.symbol}) for a in atoms]
        )
        return atoms

    def attach_atoms(self, atoms: Atoms) -> Atoms:
        """Attach `atoms` as the coordinate source without changing the nodes.

        Node indices are global throughout the codebase, so a subgraph spanning
        part of a system still indexes into the *full* `Atoms` -- that is what
        makes `molecules()` and the term remapping work.  Giving such a subgraph
        its coordinates therefore must not touch the node set, or the block
        silently grows to the whole system.  `set_atoms` does exactly that and is
        the right call only when the graph is meant to span every atom.
        """
        self._hash = None
        self._molecules = None
        self.atoms = atoms
        return atoms

    def set_terms(self, terms: list[Term]) -> dict[str, Any]:
        self.terms = terms
        # structure into arrays for later
        term_dict = {}
        for term in terms:
            term_type = term["type"]
            if term_type in term_dict:
                term_dict[term_type]["atoms"].append(tuple(term["atoms"].values()))
                for k, v in term["kwargs"].items():
                    term_dict[term_type]["kwargs"][k].append(v)
            else:
                term_dict[term_type] = {
                    "atoms": [tuple(term["atoms"].values())],
                    "kwargs": {},
                }
                for k, v in term["kwargs"].items():
                    term_dict[term_type]["kwargs"][k] = [v]

        # convert to numpy
        for term_type, term_data in term_dict.items():
            term_data["atoms"] = np.array(term_data["atoms"])
            count = len(term_data["atoms"])
            for arg, vals in term_data["kwargs"].items():
                # Every term of a type must state the same parameters, because
                # this layout is column-oriented: a key that only some terms
                # carry produces a short column silently misaligned against the
                # others.  It became reachable when `bond` gained the optional
                # Hulburt-Hirschfelder `c` -- merging a refitted template with
                # one that predates it puts both kinds in a single `term_dict`.
                # Caught here rather than in the force field, where it surfaces
                # as an unattributable broadcasting error.
                if len(vals) != count:
                    raise ValueError(
                        f"{count} `{term_type}` terms but only {len(vals)} state "
                        f"`{arg}`. The term list mixes templates that disagree "
                        f"about which parameters a `{term_type}` has; refit them "
                        "together, or give the ones that omit it an explicit "
                        "default."
                    )
                term_data["kwargs"][arg] = np.array(vals)

        self.term_dict = term_dict
        return term_dict

    def copy(self, share_atoms: bool = False) -> "Topology":
        """Copy the graph and terms.

        `share_atoms` hands the copy the same `Atoms` object instead of a
        duplicate.  Safe -- and much cheaper -- whenever the geometry is held
        fixed while many topologies over it are enumerated, which is the case
        throughout a single force evaluation.
        """
        graph: nx.Graph = self.graph.copy()
        terms = list(self.terms) if self.terms else []
        if self.atoms is None:
            return self.__class__(graph, None, terms)
        if share_atoms:
            # Bypass __init__'s set_atoms: it would add every atom as a node,
            # which turns a block-restricted graph into a whole-system one.
            new = self.__class__(graph, None, terms)
            new.attach_atoms(self.atoms)
            return new
        return self.__class__(graph, self.atoms.copy(), terms)

    @classmethod
    def from_atoms(cls, atoms: Atoms, scale: float = BOND_SCALE) -> "Topology":
        """Perceive bonds from geometry, unless `atoms` states its own.

        `ase2networkx` prefers an explicit `info["connectivity"]` and only falls
        back to distance cutoffs, so `scale` matters exactly for the frames that
        carry no connectivity -- a trajectory, or any plain `.xyz`.

        Periodicity is read off `atoms` rather than assumed.  It used to be
        passed positionally as `False`, which silently split every molecule
        straddling a boundary in a packed box into two fragments.
        """
        graph = ase2networkx(atoms, pbc=bool(atoms.pbc.any()), scale=scale)
        for _, node_data in graph.nodes(data=True):
            node_data.pop("position", None)
            node_data.pop("charge", None)
        return cls(graph, atoms)

    @classmethod
    def from_molecules(
        cls, molecules: list["Topology"], remap=True, copy=False
    ) -> "Topology":
        """
        If remap is false, returns only a merger of the graphs. it will attempt to
        perform a disjoint union of the molecules and will raise an error if there are index overlaps.
        """
        copy_atoms: bool = copy and all(m.atoms is not None for m in molecules)
        copy_terms: bool = copy and any(len(m.terms) > 0 for m in molecules)
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
    def from_terms(cls, terms: list[Term], atoms: Atoms | None = None) -> "Topology":
        bonds: list[tuple[int, int]] = []
        for term in terms:
            if term["type"] == "bond":
                bonds.append(tuple(term["atoms"].values()))
        graph = nx.Graph(bonds)
        return cls(graph, atoms, terms)

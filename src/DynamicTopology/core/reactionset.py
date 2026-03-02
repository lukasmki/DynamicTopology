import json
import sys
import numpy as np
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from ase import Atoms, io
from ase.io.formats import ioformats

from .reaction import Reaction
from .topology import Topology
from .network import ReactionNetwork

from .types import Term


class ReactionSet:
    default_data = {
        "meta": {
            "molecules": {},
            "reactions": {},
        },
        "molecules": {},
        "reactions": {},
    }

    def __init__(self, path: str | Path | None = None):
        self.data = self.default_data.copy()
        if path:
            self.load(path)

    def get_molecule(self, molecule: Topology) -> Topology:
        """Returns a copy of the molecule in the database"""
        return self.data["molecules"][molecule.hash()].copy()

    def get_molecules(
        self, ids: list[int] | None = None, formulas: list[str] | None = None
    ) -> Iterable[Topology]:
        """Returns a copy of the molecules in the database"""
        if ids:
            for molid in ids:
                mol_hash = self.data["meta"]["molecules"][molid]
                yield self.data["molecules"][mol_hash].copy()
        elif formulas:
            for formula in formulas:
                mol_hash = self.data["meta"]["molecules"][formula.upper()]
                yield self.data["molecules"][mol_hash].copy()
        else:
            for mol in self.data["molecules"].values():
                yield mol.copy()

    def get_reactions(
        self,
        reactants: Topology | None = None,
        products: Topology | None = None,
        reaction: Reaction | None = None,
    ):
        """Returns a copy of the reaction in the database"""
        if reaction is not None:
            for rxn in self.data["reactions"].get(reaction.reactants.hash(), []):
                if rxn.products.hash() == reaction.products.hash():
                    yield rxn.copy()
        elif (reactants is not None) and (products is not None):
            reaction: Reaction = Reaction(reactants, products)
            for rxn in self.data["reactions"].get(reaction.reactants.hash(), []):
                if rxn.products.hash() == reaction.products.hash():
                    yield rxn.copy()
        elif reactants:
            for rxn in self.data["reactions"].get(reactants.hash(), []):
                yield rxn.copy()
        elif products:
            for rxn in self.data["reactions"].get(products.hash(), []):
                yield rxn.reverse().copy()
        else:
            for rxn_list in self.data["reactions"].values():
                for rxn in rxn_list:
                    yield rxn.copy()

    def add_molecule(self, molid: int, molecule: Topology) -> None:
        molecule_hash = molecule.hash()
        if molecule_hash in self.data["molecules"]:
            print(
                f"WARN: Overwriting molecule {molecule} - {molecule_hash}",
                file=sys.stderr,
            )
        self.data["molecules"][molecule_hash] = molecule
        self.data["meta"]["molecules"].update(
            {
                molid: molecule_hash,
                molecule.atoms.get_chemical_formula(): molecule_hash,
            }
        )

    def add_reaction(self, rxnid: int, reaction: Reaction) -> None:
        reactant_hash, product_hash = reaction.hash()
        if reactant_hash in self.data["reactions"]:
            self.data["reactions"][reactant_hash].append(reaction)
        else:
            self.data["reactions"][reactant_hash] = [reaction]

        if product_hash in self.data["reactions"]:
            self.data["reactions"][product_hash].append(reaction.reverse())
        else:
            self.data["reactions"][product_hash] = [reaction.reverse()]

        self.data["meta"]["reactions"].update(
            {
                rxnid: (reactant_hash, product_hash),
                reaction.equation(): (reactant_hash, product_hash),
            }
        )

    def get_terms(
        self, parametrizable: Topology | ReactionNetwork
    ) -> list[dict[str, Any]]:
        if isinstance(parametrizable, Topology):
            return self.get_terms_topology(parametrizable)
        elif isinstance(parametrizable, ReactionNetwork):
            return self.get_terms_network(parametrizable)
        else:
            raise ValueError(f"Unknown parametrizable object {type(parametrizable)}")

    def get_terms_network(self, network: ReactionNetwork) -> list[Term]:
        all_terms: list[Term] = []
        for subnet in network.subnetworks():
            if subnet.terms:
                all_terms.extend(subnet.terms)
                continue
            subnet.reactions()
        raise NotImplementedError()

    def get_terms_topology(self, topology: Topology) -> list[Term]:
        """Get force field terms to load into topology"""
        all_terms = []
        for mol in topology.molecules():
            assert isinstance(mol, Topology)

            if mol.terms:  # if terms already set
                all_terms.extend(mol.terms)
                continue

            mol_data: Topology = self.get_molecule(mol)
            if mol_data is None:
                raise ValueError(f"Molecule {mol} not found in ReactionSet")

            # map force field template to global indices
            matcher = nx.isomorphism.GraphMatcher(
                mol_data.graph,
                mol.graph,
                node_match=lambda a, b: a["atomic_number"] == b["atomic_number"],
            ).match()
            mol_map = next(matcher)

            # reindex terms
            for term in mol_data.terms:
                for k, v in term["atoms"].items():
                    term["atoms"][k] = mol_map[v]
            all_terms.extend(mol_data.terms)
            del matcher
        return all_terms

    def get_network(self, topology: Topology, bimol_cutoff=4.0) -> ReactionNetwork:
        graph = nx.MultiGraph()

        for i, (imol, iatoms) in enumerate(topology.molecules(return_atoms=True)):
            graph.add_node(i, molecule=imol)
            imol_data: list[Reaction] = list(self.get_reactions(reactants=imol))

            # reindex reaction to global indices
            for rxn in imol_data:
                mol_map = rxn.get_mapping(imol)
                graph.add_edge(i, i, reaction=rxn, mapping=mol_map)
                # print(i, i, rxn, mol_map)

            for j, (jmol, jatoms) in enumerate(topology.molecules(return_atoms=True)):
                if j >= i:
                    continue
                # neighbor check
                dv = iatoms.positions[:, None, :] - jatoms.positions[None, :, :]
                if topology.atoms.cell:
                    df = dv @ np.linalg.inv(topology.atoms.cell)
                    dv = (
                        dv
                        - (topology.atoms.pbc * np.floor(df + 0.5))
                        @ topology.atoms.cell
                    )
                rmin = np.sqrt(np.sum(dv * dv, -1).min())
                if rmin > bimol_cutoff:
                    continue

                ijmol = Topology.from_molecules([imol, jmol], False)
                ijmol_data: list[Reaction] = list(self.get_reactions(reactants=ijmol))

                # reindex reactions to global indices
                for rxn in ijmol_data:
                    mol_map = rxn.get_mapping(ijmol)
                    graph.add_edge(i, j, reaction=rxn, mapping=mol_map)

        return ReactionNetwork(graph)

    def load(self, path: str | Path) -> None:
        """Load ReactionSet data from .json or .h5 archive"""
        if isinstance(path, str):
            path = Path(path).resolve()
        if path.suffix == ".json":
            # load from json manifest
            with open(path, "r") as fp:
                data = json.load(fp)

            # NOTE: should use the metadata to handle
            # merge conflicts with multiple reaction sets

            for mol in data["molecules"]:
                # load molecule and associated force field
                data_path: Path = path.parent / mol["path"]

                if data_path.suffix in ioformats:
                    atoms: Atoms | list[Atoms] = io.read(data_path)
                else:  # try to read as xyz
                    atoms: Atoms | list[Atoms] = io.read(data_path.with_suffix(".xyz"))
                with open(data_path.with_suffix(".jsonl"), "r") as fp:
                    terms: list[Term] = [json.loads(term) for term in fp.readlines()]

                assert isinstance(atoms, Atoms)
                molecule = Topology.from_terms(terms, atoms)
                self.add_molecule(mol["id"], molecule)

            for rxn in data["reactions"]:
                # load reactions and associated diabatic coupling force field
                data_path: Path = path.parent / rxn["path"]

                if data_path.suffix in ioformats:
                    atoms: Atoms | list[Atoms] = io.read(data_path, index=":")
                else:  # try to read as xyz
                    atoms: Atoms | list[Atoms] = io.read(
                        data_path.with_suffix(".xyz"), index=":"
                    )

                with open(data_path.with_suffix(".jsonl"), "r") as fp:
                    terms: list[Term] = [json.loads(term) for term in fp.readlines()]

                reaction = Reaction.from_atoms(atoms, terms)
                self.add_reaction(rxn["id"], reaction)

        elif path.suffix == ".h5":
            # load from h5
            raise NotImplementedError(".h5 io not implemented")
        else:
            raise ValueError("Reaction set path should point to one of: .json, .h5")

    def save(self, path: str | Path) -> None:
        """Save as .h5 archive"""
        pass

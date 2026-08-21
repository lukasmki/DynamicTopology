from copy import deepcopy
import json
import sys
import numpy as np
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
from ase import Atoms, io, units
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
        self.data = deepcopy(self.default_data)
        self._term_cache: dict[tuple[str, frozenset], list] = {}
        self._bimol_hash_cache: dict[frozenset, str] = {}
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

    @staticmethod
    def _reference_term(atoms: Atoms, terms: list[Term]) -> Term | None:
        """Constant shift putting this template on the reference energy scale.

        Diabatic states are compared by their absolute energies, so every
        bonding topology has to be measured from the same zero.  The dataset
        supplies that zero: `scripts/compute.py` writes an atomization energy
        per template (eV, referenced to free atoms, hence exactly 0 for a free
        atom).  Morse bonds already account for -sum(D) of it at the minimum,
        so the part the force field cannot reproduce is the residual

            E0 = E_atomization + sum(D)

        which is what gets stored.  Morse therefore carries the physics and the
        shift only corrects for q-force having fitted each bond locally rather
        than to the molecule's total atomization energy -- a residual of a few
        tenths of an eV for most templates here, but +2.59 eV for H2O2, which
        is worth knowing about rather than silently absorbing.

        Returns None for a template with no reference energy, so datasets whose
        energies have not been computed yet keep loading unchanged (they simply
        keep the old, uncalibrated behaviour).
        """
        if atoms.calc is None:
            return None
        try:
            e_atomization = atoms.get_potential_energy()  # eV
        except (RuntimeError, AttributeError):
            return None

        sum_d = sum(  # kJ/mol
            term["kwargs"]["D"] for term in terms if term["type"] == "bond"
        )
        e0 = e_atomization / (units.kJ / units.mol) + sum_d  # kJ/mol

        # Anchored on atom 0 purely so the term has an index to be remapped
        # through when the template is matched onto the live system; the energy
        # belongs to the molecule as a whole and contributes no force.
        return {"type": "reference", "atoms": {"a1": 0}, "kwargs": {"E0": e0}}

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

            mol_hash = mol.hash()
            # The stored value is a term list already remapped onto specific
            # global indices, so the key must determine it uniquely.  The WL
            # hash is a graph invariant over atomic numbers and cannot say which
            # node carries which element, and the node set does not either, so
            # the two together under-specify the value.  Pinning the per-node
            # element and the edge set makes the key exact.  Hardening: no
            # collision has been demonstrated against the narrower key.
            cache_key = (
                mol_hash,
                frozenset(
                    (node, data.get("atomic_number"))
                    for node, data in mol.graph.nodes(data=True)
                ),
                frozenset(frozenset(edge) for edge in mol.graph.edges()),
            )

            if cache_key in self._term_cache:
                all_terms.extend(self._term_cache[cache_key])
                continue

            mol_data: Topology = self.data["molecules"].get(mol_hash)
            if mol_data is None:
                raise ValueError(f"Molecule {mol} not found in ReactionSet")

            # map force field template to global indices
            matcher = nx.isomorphism.GraphMatcher(
                mol_data.graph,
                mol.graph,
                node_match=lambda a, b: a["atomic_number"] == b["atomic_number"],
            ).match()
            mol_map = next(matcher)
            del matcher

            # build new term dicts with remapped indices — do not mutate mol_data
            remapped: list[Term] = [
                {
                    "type": term["type"],
                    "atoms": {k: mol_map[v] for k, v in term["atoms"].items()},
                    "kwargs": term["kwargs"],
                }
                for term in mol_data.terms
            ]

            self._term_cache[cache_key] = remapped
            all_terms.extend(remapped)
        return all_terms

    def get_network(self, topology: Topology, bimol_cutoff=4.0) -> ReactionNetwork:
        graph = nx.MultiGraph()

        # Pre-collect once to avoid O(N²) Atoms slicing inside the inner loop
        mol_list: list[tuple[Topology, any]] = list(
            topology.molecules(return_atoms=True)
        )

        cell = topology.atoms.cell
        pbc = topology.atoms.pbc
        inv_cell = np.linalg.inv(cell) if np.any(pbc) else None

        for i, (imol, iatoms) in enumerate(mol_list):
            graph.add_node(i, molecule=imol)
            imol_data: list[Reaction] = list(self.get_reactions(reactants=imol))

            # reindex reaction to global indices
            for rxn in imol_data:
                for mol_map in rxn.get_mappings(imol):
                    graph.add_edge(i, i, reaction=rxn, mapping=mol_map)

            for j, (jmol, jatoms) in enumerate(mol_list[:i]):
                # neighbor check
                dv = iatoms.positions[:, None, :] - jatoms.positions[None, :, :]
                if inv_cell is not None:
                    df = dv @ inv_cell
                    dv = dv - (pbc * np.floor(df + 0.5)) @ cell
                rmin = np.sqrt(np.sum(dv * dv, -1).min())
                if rmin > bimol_cutoff:
                    continue

                # cache combined topology hash to avoid repeated nx.union + WL hash
                pair_key = frozenset({imol.hash(), jmol.hash()})
                if pair_key not in self._bimol_hash_cache:
                    ijmol = Topology.from_molecules([imol, jmol], False)
                    self._bimol_hash_cache[pair_key] = ijmol.hash()
                combined_hash = self._bimol_hash_cache[pair_key]

                ijmol_data: list[Reaction] = list(
                    self.data["reactions"].get(combined_hash, [])
                )

                if not ijmol_data:
                    continue

                # need the combined topology object for get_mapping
                ijmol = Topology.from_molecules([imol, jmol], False)

                # reindex reactions to global indices
                for rxn in ijmol_data:
                    for mol_map in rxn.get_mappings(ijmol):
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
                # A template that already states its shift keeps it.  Templates
                # whose Morse depths have been fitted to carry the atomization
                # energy state it as zero, and synthesizing another one here
                # would count the same energy twice.
                if not any(term["type"] == "reference" for term in terms):
                    reference = self._reference_term(atoms, terms)
                    if reference is not None:
                        terms = terms + [reference]
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

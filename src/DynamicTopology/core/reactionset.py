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
        self._term_cache: dict[tuple[tuple, tuple], list] = {}
        self._bimol_hash_cache: dict[frozenset, str] = {}
        # Keyed by molecule signature rather than held on the Topology, because
        # `Topology._hash` only ever caches within one object and these objects
        # are rebuilt from scratch on every force call.
        self._hash_cache: dict[tuple[tuple, tuple], str] = {}
        self._channel_cache: dict[tuple[tuple, tuple], list] = {}
        if path:
            self.load(path)

    def _unknown_molecule_message(self, mol: Topology) -> str:
        """Say which species was perceived, and what it is nearest to.

        The bare repr of a `Topology` names a node and edge count and nothing
        else, which is not enough to tell a missing template from a
        misperceived geometry -- and the common cause is the latter.  A
        transition state is the sharpest example: `rxn_04`'s own stored TS frame
        perceives as H-O-H-O, a hydrogen sitting 1.16 A from one oxygen and 1.16
        A from the other, which is not a molecule any database can hold because
        it is not a molecule.  Naming the nearest known species by how many
        bonds separate them is what distinguishes "you are one bond from water"
        from "this element combination is genuinely absent".
        """
        elements = [
            mol.graph.nodes[node]["atomic_number"] for node in mol.graph.nodes()
        ]
        formula = Atoms(numbers=elements).get_chemical_formula()
        bonds = sorted(sorted(edge) for edge in mol.graph.edges())

        # Nearest by edge-set difference among the templates with the same
        # multiset of elements; anything else is not a rearrangement of this.
        neighbours: list[tuple[int, str]] = []
        for template in self.data["molecules"].values():
            template_elements = [
                template.graph.nodes[node]["atomic_number"]
                for node in template.graph.nodes()
            ]
            if sorted(template_elements) != sorted(elements):
                continue
            neighbours.append(
                (
                    abs(template.graph.number_of_edges() - mol.graph.number_of_edges()),
                    f"{Atoms(numbers=template_elements).get_chemical_formula()} "
                    f"with bonds "
                    f"{sorted(sorted(e) for e in template.graph.edges())}",
                )
            )
        nearest = (
            min(neighbours)[1]
            if neighbours
            else "nothing in the database has this element composition"
        )

        return (
            f"perceived the molecule {formula} with bonds {bonds}, which is not "
            f"in the ReactionSet. Nearest known species: {nearest}. A geometry "
            "part-way through a reaction perceives as a bridged species that is "
            "neither the reactant nor the product and has no diabatic template "
            "by construction -- if this frame is a transition state, score it "
            "under an endpoint's connectivity instead (see "
            "`fit.dissociation.diabatic_energy`). Otherwise the species is "
            "genuinely missing and belongs in the dataset manifest."
        )

    @staticmethod
    def _molecule_signature(molecule: Topology) -> tuple[tuple, tuple]:
        """Exact identity of a molecule as it sits in the live system.

        Which node carries which element, and which nodes are bonded.  Two
        molecules with the same signature are the same molecule on the same
        atoms, so anything derived purely from the graph -- its hash, its
        template mapping, its remapped terms -- is shared between them.
        """
        # Sorted tuples rather than frozensets: the signature is derived
        # thousands of times per force call and tuple construction is markedly
        # cheaper, while sorting makes it just as canonical.
        return (
            tuple(
                sorted(
                    (node, data.get("atomic_number"))
                    for node, data in molecule.graph.nodes(data=True)
                )
            ),
            tuple(sorted(tuple(sorted(edge)) for edge in molecule.graph.edges())),
        )

    def hash_molecule(self, molecule: Topology) -> str:
        """Weisfeiler-Lehman hash of `molecule`, memoized across force calls."""
        signature = self._molecule_signature(molecule)
        cached = self._hash_cache.get(signature)
        if cached is None:
            cached = molecule.hash()
            self._hash_cache[signature] = cached
        return cached

    def _reaction_channels(
        self, topology: Topology, combined_hash: str | None = None
    ) -> list[tuple[Reaction, dict]]:
        """Every (reaction, index mapping) applicable to `topology`.

        Memoized by molecule signature.  Both halves of this depend only on the
        graph and never on the geometry: which reactions match a molecule, and
        how their templates map onto its indices.  A molecular dynamics run
        recomputes it every step for topologies that mostly do not change, and
        the isomorphism search is not cheap -- `get_mappings` plus the reaction
        lookup were over half the cost of building the network.

        The stored `Reaction` objects are shared rather than copied.  Callers
        treat them as read-only (`apply` copies the topology it is given, and
        everything else here reads `atoms`/`term_dict`/`hash`), and copying them
        meant a `deepcopy` of the transition-state ensemble on every lookup.
        """
        signature = self._molecule_signature(topology)
        cached = self._channel_cache.get(signature)
        if cached is not None:
            return cached

        if combined_hash is None:
            combined_hash = self.hash_molecule(topology)
        channels = [
            (reaction, mapping)
            for reaction in self.data["reactions"].get(combined_hash, [])
            for mapping in reaction.get_mappings(topology)
        ]
        self._channel_cache[signature] = channels
        return channels

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

            # The stored value is a term list already remapped onto specific
            # global indices, so the key must determine it uniquely: which node
            # carries which element, and which nodes are bonded.  That pair is
            # already exact, and the Weisfeiler-Lehman hash is a function of it,
            # so including the hash in the key would add nothing -- while
            # computing it costs a full graph traversal.  Deriving the key first
            # and hashing only on a miss took ~26.5k WL hashes per force call
            # down to a handful; it was 36% of the runtime on a 200-atom box.
            cache_key = self._molecule_signature(mol)

            cached = self._term_cache.get(cache_key)
            if cached is not None:
                all_terms.extend(cached)
                continue

            mol_hash = self.hash_molecule(mol)
            mol_data: Topology = self.data["molecules"].get(mol_hash)
            if mol_data is None:
                raise ValueError(self._unknown_molecule_message(mol))

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

            # reindex reaction to global indices
            for rxn, mol_map in self._reaction_channels(imol):
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
                pair_key = frozenset(
                    {self.hash_molecule(imol), self.hash_molecule(jmol)}
                )
                if pair_key not in self._bimol_hash_cache:
                    ijmol = Topology.from_molecules([imol, jmol], False)
                    self._bimol_hash_cache[pair_key] = self.hash_molecule(ijmol)
                combined_hash = self._bimol_hash_cache[pair_key]

                if not self.data["reactions"].get(combined_hash):
                    continue

                # need the combined topology object to map templates onto it
                ijmol = Topology.from_molecules([imol, jmol], False)

                # reindex reactions to global indices
                for rxn, mol_map in self._reaction_channels(ijmol, combined_hash):
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

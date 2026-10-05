from dataclasses import dataclass, field
import json
import sys
import numpy as np
from pathlib import Path
from typing import Iterable

import networkx as nx
from ase import Atoms, io
from ase.io.formats import ioformats

from .reaction import Reaction
from .topology import Topology
from .network import ReactionNetwork

from .types import Term
from ..io.json import read_jsonl
from ..forcefield.exclusions import with_exclusions
from ..forcefield.params import ForceFieldParams, activate

# Atom pairs the bimolecular pair scan holds at once.  At 24 bytes a pair (a
# 3-vector of displacements) this caps its scratch near 25 MB, so the scan stays
# vectorized on a system of any size instead of trading a Python loop for an
# allocation that does not fit.
PAIR_SCRATCH: int = 1 << 20


def _reference_term(atoms: Atoms, terms: list[Term]) -> Term | None:
    """Constant shift putting this template on the reference energy scale.

    Diabatic states are compared by their absolute energies, so every bonding
    topology has to be measured from the same zero.  The dataset supplies that
    zero: `fast-forces label` writes an atomization energy per template (eV,
    referenced to free atoms, hence exactly 0 for a free atom).  Morse bonds
    already account for -sum(D) of it at the minimum, so the part the force
    field cannot reproduce is the residual

        E0 = E_atomization + sum(D)

    which is what gets stored.  Morse therefore carries the physics and the
    shift only corrects for q-force having fitted each bond locally rather than
    to the molecule's total atomization energy -- a residual of a few tenths of
    an eV for most templates here, but +2.59 eV for H2O2, which is worth knowing
    about rather than silently absorbing.

    Returns None for a template with no reference energy, so datasets whose
    energies have not been computed yet keep loading unchanged (they simply keep
    the old, uncalibrated behaviour).
    """
    if atoms.calc is None:
        return None
    try:
        e_atomization = atoms.get_potential_energy()  # eV
    except (RuntimeError, AttributeError):
        return None

    sum_d = sum(  # eV
        term["kwargs"]["D"] for term in terms if term["type"] == "bond"
    )
    e0 = e_atomization + sum_d  # eV

    # Anchored on atom 0 purely so the term has an index to be remapped through
    # when the template is matched onto the live system; the energy belongs to
    # the molecule as a whole and contributes no force.
    return {"type": "reference", "atoms": {"a1": 0}, "kwargs": {"E0": e0}}


@dataclass
class ReactionSetData:
    """Everything a dataset manifest states, parsed and keyed by WL hash.

    The database proper: what was loaded, not how it is queried.  Every lookup
    here is a plain dict probe on a Weisfeiler-Lehman hash and returns the
    stored object by reference
    """

    molecules: dict[str, Topology] = field(default_factory=dict)
    # Both directions of every reaction: forward under the reactant hash and
    # reversed under the product's, which is what lets `channels` be a single
    # probe rather than a scan.
    reactions: dict[str, list[Reaction]] = field(default_factory=dict)
    # Manifest `id` and chemical formula -> molecule hash.  Two maps rather than
    # the one mixed dict this replaced: an integer key colliding with a formula
    # was indistinguishable there, and neither could be typed.
    ids: dict[int, str] = field(default_factory=dict)
    formulas: dict[str, str] = field(default_factory=dict)
    # The global force field parameters this set was fitted at.  Defaults until
    # a manifest says otherwise; `from_manifest` replaces it and activates it
    # before a single template is read, since deriving the exclusions and
    # solving the reference shifts both depend on them.
    params: ForceFieldParams = field(default_factory=ForceFieldParams)
    source: Path | None = None

    def add_molecule(self, molid: int, molecule: Topology) -> None:
        molecule_hash = molecule.hash()
        if molecule_hash in self.molecules:
            print(
                f"WARN: Overwriting molecule {molecule} - {molecule_hash}",
                file=sys.stderr,
            )
        self.molecules[molecule_hash] = molecule
        self.ids[molid] = molecule_hash
        # Case-folded on the way in as well as on the way out.  Storing the
        # formula verbatim and looking it up upper-cased agreed for H/O species
        # and raised `KeyError` for any two-letter element ("ClH" -> "CLH").
        self.formulas[molecule.atoms.get_chemical_formula().upper()] = molecule_hash

    def add_reaction(self, reaction: Reaction) -> None:
        reactant_hash, product_hash = reaction.hash()
        self.reactions.setdefault(reactant_hash, []).append(reaction)
        self.reactions.setdefault(product_hash, []).append(reaction.reverse())

    def channels(self, molecule_hash: str) -> list[Reaction]:
        """Every stored reaction this molecule hash is the reactant side of."""
        return self.reactions.get(molecule_hash, [])

    def by_id(self, molid: int) -> Topology:
        """The template the manifest gave this `id`."""
        return self.molecules[self.ids[molid]]

    def by_formula(self, formula: str) -> Topology:
        """The template with this chemical formula, case-insensitively."""
        return self.molecules[self.formulas[formula.upper()]]

    @property
    def n_reactions(self) -> int:
        """Stored reactions, counting each one twice -- once per direction."""
        return sum(len(rxns) for rxns in self.reactions.values())

    @classmethod
    def from_manifest(cls, path: str | Path) -> "ReactionSetData":
        """Parse a dataset manifest and everything it points at.

        `path` names a JSON manifest listing molecule and reaction entries by
        extensionless path; each is paired with a `.xyz` (geometry) and a
        `.jsonl` (parameters, one term per line).  An entry's `smiles` is
        annotation for a reader -- identity here is the WL hash of the `.xyz`'s
        bond graph, so nothing below reads it.
        """
        path = Path(path).resolve()
        if path.suffix != ".json":
            raise ValueError(
                f"Reaction set path should point to a .json manifest, got '{path.suffix}'"
            )

        with open(path, "r") as fp:
            manifest = json.load(fp)

        # NOTE: should use the metadata to handle
        # merge conflicts with multiple reaction sets

        # **Before anything else.**  `global_params` states the constants this
        # dataset's `.jsonl` files were fitted at -- the two taper radii, the
        # exclusion depth, the Morse asymptote -- and the loops below derive
        # exclusions and solve reference shifts against exactly those.
        # Activating them afterwards would fit the templates on one surface and
        # evaluate them on another, which is the silent invalidation
        # `forcefield/params.py` exists to close.  A manifest that omits the key
        # gets the defaults, unchanged.
        params = ForceFieldParams.from_dict(
            manifest.get("global_params"), source=str(path)
        )
        activate(params, source=str(path))
        data = cls(params=params, source=path)

        for mol in manifest["molecules"]:
            # load molecule and associated force field
            data_path: Path = path.parent / mol["path"]

            if data_path.suffix in ioformats:
                atoms: Atoms | list[Atoms] = io.read(data_path)
            else:  # try to read as xyz
                atoms: Atoms | list[Atoms] = io.read(data_path.with_suffix(".xyz"))
            terms: list[Term] = read_jsonl(data_path.with_suffix(".jsonl"))

            assert isinstance(atoms, Atoms)
            # Intramolecular nonbonded exclusions, derived here rather than
            # stored in the `.jsonl`.  They are a function of the bond graph and
            # of this dataset's `exclusion_depth`, so deriving them keeps them
            # correct when either changes, and keeps the term files to the
            # parameters a fit actually produces.  A dataset shipping its own
            # exclusions explicitly is left alone.
            terms = with_exclusions(terms, atoms.get_atomic_numbers(), params=params)
            # A template that already states its shift keeps it -- fast-forces
            # writes one for every template it fits, and `refine` states zero
            # for depths rescaled to the atomization energy -- and
            # synthesizing another one here would count the same energy twice.
            if not any(term["type"] == "reference" for term in terms):
                reference = _reference_term(atoms, terms)
                if reference is not None:
                    terms = terms + [reference]
            data.add_molecule(mol["id"], Topology.from_terms(terms, atoms))

        for rxn in manifest["reactions"]:
            # load reactions and associated diabatic coupling force field
            data_path: Path = path.parent / rxn["path"]

            if data_path.suffix in ioformats:
                atoms: Atoms | list[Atoms] = io.read(data_path, index=":")
            else:  # try to read as xyz
                atoms: Atoms | list[Atoms] = io.read(
                    data_path.with_suffix(".xyz"), index=":"
                )

            terms: list[Term] = read_jsonl(data_path.with_suffix(".jsonl"))

            data.add_reaction(Reaction.from_atoms(atoms, terms))

        return data


class ReactionSet:
    """The query engine over a loaded dataset.

    Loads a manifest into `data` (a `ReactionSetData`), activating its
    `global_params`, and maps the stored templates onto a live system's atom
    indices: `get_terms` for a topology's parameters and `get_network` for the
    reactions available to it.  Results are cached by molecule signature.
    """

    def __init__(self, path: str | Path | None = None):
        self.data = ReactionSetData()
        self._reset_caches()
        if path:
            self.load(path)

    def _reset_caches(self) -> None:
        """Drop everything derived from `self.data`.

        Called on every `load`, since a set that is reloaded onto a different
        dataset would otherwise answer from the previous one's templates.
        """
        self._term_cache: dict[tuple[tuple, tuple], list] = {}
        # `_term_cache`'s lists, vectorized as `Topology.set_terms` would; see
        # `assign_terms`.
        self._vector_cache: dict[tuple[tuple, tuple], dict] = {}
        self._bimol_hash_cache: dict[frozenset, str] = {}
        # Keyed by molecule signature rather than held on the Topology, because
        # `Topology._hash` only ever caches within one object and these objects
        # are rebuilt from scratch on every force call.
        self._hash_cache: dict[tuple[tuple, tuple], str] = {}
        self._channel_cache: dict[tuple[tuple, tuple], list] = {}

    @property
    def params(self) -> ForceFieldParams:
        """The global force field parameters this set was fitted at."""
        return self.data.params

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
        for template in self.data.molecules.values():
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
            "fast-forces' `refine.diabatic_energy`). Otherwise the species is "
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
        #
        # Nearly every molecule here is an induced subgraph *view* of a block,
        # and reading one through networkx's filtered views costs a filter test
        # per node and per adjacency entry -- 1.7 s of a 4.6 s force call on a
        # hot 3000 K box.  The same two sorted tuples come straight out of the
        # underlying graph's dicts: the output is sorted, so the order they are
        # visited in cannot show, and each edge is taken once as `u <= v`,
        # which is what `tuple(sorted(edge))` makes of it -- a self-loop
        # included once, as `edges()` lists it.  Anything else -- an
        # edge-filtered view -- takes the general path below.
        graph = molecule.graph
        base = getattr(graph, "_graph", None)
        if base is None:
            base, members = graph, graph._node
        elif getattr(graph, "_EDGE_OK", None) is nx.filters.no_filter and hasattr(
            graph._NODE_OK, "nodes"
        ):
            members = graph._NODE_OK.nodes.intersection(base._node)
        else:
            base = None
        if base is not None:
            node_data, adjacency = base._node, base._adj
            return (
                tuple(sorted((n, node_data[n].get("atomic_number")) for n in members)),
                tuple(
                    sorted(
                        (u, v)
                        for u in members
                        for v in adjacency[u]
                        if u <= v and v in members
                    )
                ),
            )
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
            for reaction in self.data.channels(combined_hash)
            for mapping in reaction.get_mappings(topology)
        ]
        self._channel_cache[signature] = channels
        return channels

    def get_molecule(self, molecule: Topology) -> Topology:
        """Returns a copy of the molecule in the database"""
        return self.data.molecules[molecule.hash()].copy()

    def get_molecules(
        self, ids: list[int] | None = None, formulas: list[str] | None = None
    ) -> Iterable[Topology]:
        """Returns a copy of the molecules in the database"""
        if ids:
            for molid in ids:
                yield self.data.by_id(molid).copy()
        elif formulas:
            for formula in formulas:
                yield self.data.by_formula(formula).copy()
        else:
            for mol in self.data.molecules.values():
                yield mol.copy()

    def get_reactions(
        self,
        reactants: Topology | None = None,
        products: Topology | None = None,
        reaction: Reaction | None = None,
    ):
        """Returns a copy of the reaction in the database"""
        if reaction is None and (reactants is not None) and (products is not None):
            reaction = Reaction(reactants, products)

        if reaction is not None:
            for rxn in self.data.channels(reaction.reactants.hash()):
                if rxn.products.hash() == reaction.products.hash():
                    yield rxn.copy()
        elif reactants:
            for rxn in self.data.channels(reactants.hash()):
                yield rxn.copy()
        elif products:
            for rxn in self.data.channels(products.hash()):
                yield rxn.reverse().copy()
        else:
            for rxn_list in self.data.reactions.values():
                for rxn in rxn_list:
                    yield rxn.copy()

    def add_molecule(self, molid: int, molecule: Topology) -> None:
        self.data.add_molecule(molid, molecule)

    def add_reaction(self, reaction: Reaction) -> None:
        self.data.add_reaction(reaction)

    def set_template_terms(self, molecule_hash: str, terms: list[Term]) -> None:
        """Install a freshly fitted parameter set onto a stored template.

        The fitter refits `kwargs` in place rather than rewriting the `.jsonl`
        files and reloading, so the stored graphs and their hashes stay valid
        and only the remapped-term cache goes stale.  Clearing it is the half
        that is easy to forget, which is why this is one call.
        """
        self.data.molecules[molecule_hash].terms = terms
        self._term_cache.clear()
        self._vector_cache.clear()

    def get_terms(self, topology: Topology) -> list[Term]:
        """Force field terms for `topology`, remapped onto its global indices."""
        return self.get_terms_topology(topology)

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
            mol_data: Topology | None = self.data.molecules.get(mol_hash)
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

    @staticmethod
    def _molecule_separations(
        positions: np.ndarray, starts: np.ndarray, cell, pbc
    ) -> np.ndarray:
        """(M, M) minimum interatomic distance between every pair of molecules.

        `positions` is ordered molecule by molecule and `starts` holds each
        molecule's first row, so the (N, N) matrix of squared minimum-image
        separations reduces to the (M, M) answer with two `minimum.reduceat`
        passes and no Python-level pair loop.

        This is the same quantity the bimolecular cutoff was always testing,
        computed the other way round: per pair it is one small numpy expression
        whose cost is nearly all fixed overhead, and at a hundred molecules
        there are five thousand such pairs -- a third of the force call.  Rows
        are chunked so the scratch displacement array stays bounded whatever
        the system size.
        """
        natoms = len(positions)
        nmol = len(starts)
        cell = np.asarray(cell)
        inv_cell = np.linalg.inv(cell) if np.any(pbc) else None
        ends = np.append(starts[1:], natoms)

        out = np.empty((nmol, nmol))
        max_rows = max(1, PAIR_SCRATCH // max(natoms, 1))
        lo = 0
        while lo < nmol:
            hi = lo + 1
            while hi < nmol and ends[hi] - starts[lo] <= max_rows:
                hi += 1
            a0, a1 = starts[lo], ends[hi - 1]

            dv = positions[a0:a1, None, :] - positions[None, :, :]
            if inv_cell is not None:
                dv = dv - (pbc * np.floor(dv @ inv_cell + 0.5)) @ cell
            distsq = np.sum(dv * dv, -1)

            columns = np.minimum.reduceat(distsq, starts, axis=1)
            out[lo:hi] = np.minimum.reduceat(columns, starts[lo:hi] - a0, axis=0)
            lo = hi

        return np.sqrt(out)

    def _pair_channels(self, isig, jsig, imol, jmol) -> list[tuple[Reaction, dict]]:
        """The channels of two molecules together, by their merged signature.

        `imol` and `jmol` are the molecules, or zero-argument callables that
        build them: they are needed only on a cache miss.
        """
        combined_signature = (
            tuple(sorted(isig[0] + jsig[0])),
            tuple(sorted(isig[1] + jsig[1])),
        )
        channels = self._channel_cache.get(combined_signature)
        if channels is None:
            imol = imol() if callable(imol) else imol
            jmol = jmol() if callable(jmol) else jmol
            # cache the combined hash to avoid a repeated WL hash
            pair_key = frozenset({self.hash_molecule(imol), self.hash_molecule(jmol)})
            ijmol = Topology.from_molecules([imol, jmol], False)
            combined_hash = self._bimol_hash_cache.get(pair_key)
            if combined_hash is None:
                combined_hash = self.hash_molecule(ijmol)
                self._bimol_hash_cache[pair_key] = combined_hash
            channels = self._reaction_channels(ijmol, combined_hash)
        return channels

    def state_channels(
        self, topology: Topology, bimol_cutoff: float, distsq: np.ndarray, row: dict
    ) -> list[tuple[Reaction, dict]] | None:
        """`get_network(topology, bimol_cutoff).reactions()`, without the network.

        The same `(reaction, mapping)` pairs in the same order, which matters:
        `EVBBasis._close` admits states in the order their channels are found,
        and where a block reaches its state cap that order decides which ones.
        `MultiGraph.edges()` lists, for each molecule `n` in turn, the self-loops
        `get_network` added for it and then its edges to every later molecule
        `k > n`, ascending -- the adjacency order `get_network`'s insertion
        leaves -- so this walks the molecules in that order instead of building
        the graph, which with its per-molecule subgraph views was most of a
        state's cost.

        `distsq` holds squared minimum-image distances between atoms, row
        `row[node]` per atom: the caller builds it once per block, where
        `get_network` redid the minimum-image arithmetic for every state.
        Returns `None` for a topology over a graph view, which takes the
        general path.
        """
        graph = topology.graph
        if getattr(graph, "_graph", None) is not None:
            return None
        components, nodes, signatures = self._components(topology)

        def molecule(i):
            return lambda: Topology(graph=graph.subgraph(components[i]))

        order = [row[n] for group in nodes for n in group]
        starts = np.cumsum([0] + [len(group) for group in nodes[:-1]])
        block = distsq[np.ix_(order, order)]
        reduced = np.minimum.reduceat(np.minimum.reduceat(block, starts, axis=1), starts, axis=0)
        separation = np.sqrt(reduced)

        # The pairs `get_network` keeps, `(n, k)` with `k > n`, row-major: the
        # order its adjacency is walked in.  `not (separation > cutoff)` is its
        # own test, spelled out.
        close = np.triu(~(separation.T > bimol_cutoff), 1)
        partners = [[] for _ in nodes]
        for n, k in zip(*np.nonzero(close)):
            partners[n].append(int(k))

        found = []
        for n in range(len(nodes)):
            single = self._channel_cache.get(signatures[n])
            if single is None:
                single = self._reaction_channels(molecule(n)())
            found.extend(single)
            for k in partners[n]:
                found.extend(
                    self._pair_channels(
                        signatures[k], signatures[n], molecule(k), molecule(n)
                    )
                )
        return found

    @staticmethod
    def _components(topology: Topology) -> tuple[list, list, list]:
        """A plain-graph topology's molecules: node sets, sorted nodes, signatures.

        In `topology.molecules()` order, with each signature the one
        `_molecule_signature` gives that molecule.  Memoized on the topology
        against the identity of its `_molecules` list, which `Topology` drops
        whenever its graph or atoms change.
        """
        graph = topology.graph
        components = topology._molecules
        if components is None:
            components = topology._molecules = list(nx.connected_components(graph))
        memo = getattr(topology, "_signatures", None)
        if memo is not None and memo[0] is components:
            return components, memo[1], memo[2]
        node_data, adjacency = graph._node, graph._adj
        nodes, signatures = [], []
        for members in components:
            ordered = sorted(members)
            nodes.append(ordered)
            # `_molecule_signature`'s fast path, on the plain graph.
            signatures.append(
                (
                    tuple((n, node_data[n].get("atomic_number")) for n in ordered),
                    tuple(
                        sorted(
                            (u, v)
                            for u in members
                            for v in adjacency[u]
                            if u <= v and v in members
                        )
                    ),
                )
            )
        topology._signatures = (components, nodes, signatures)
        return components, nodes, signatures

    def assign_terms(self, topology: Topology) -> dict:
        """`topology.set_terms(self.get_terms(topology))`, assembled per molecule.

        Terms are per molecule and `set_terms` groups them by type in the order
        they come, so its arrays are each molecule's own, concatenated in
        molecule order -- and each molecule's are cached by signature, vectorized
        once, where `set_terms` re-vectorized a whole block term by term for
        every admitted state.  Every type, key and column comes out in the
        order `set_terms` gives it, so the arrays are the same.  Anything
        unusual -- a graph view, molecules that disagree about a type's
        parameters -- takes `set_terms`, which also raises for the latter.
        """
        graph = topology.graph
        if getattr(graph, "_graph", None) is not None:
            return topology.set_terms(self.get_terms(topology))
        components, _, signatures = self._components(topology)
        terms: list = []
        pieces: list[dict] = []
        for members, signature in zip(components, signatures):
            listed = self._term_cache.get(signature)
            if listed is None:
                listed = self.get_terms_topology(
                    Topology(graph=graph.subgraph(members))
                )
            terms.extend(listed)
            vector = self._vector_cache.get(signature)
            if vector is None:
                vector = Topology(nx.Graph()).set_terms(listed)
                self._vector_cache[signature] = vector
            pieces.append(vector)

        term_dict: dict = {}
        for piece in pieces:
            for term_type, block in piece.items():
                term_dict.setdefault(term_type, []).append(block)
        merged = {}
        for term_type, blocks in term_dict.items():
            # Keyed by name, as `set_terms` appends them, so only a molecule
            # stating a different set of parameters is a disagreement.
            keys = list(blocks[0]["kwargs"])
            if any(set(block["kwargs"]) != set(keys) for block in blocks[1:]):
                return topology.set_terms(terms)
            merged[term_type] = {
                "atoms": np.concatenate([block["atoms"] for block in blocks]),
                "kwargs": {
                    key: np.concatenate([block["kwargs"][key] for block in blocks])
                    for key in keys
                },
            }
        topology.terms = terms
        topology.term_dict = merged
        return merged

    @staticmethod
    def squared_separations(positions: np.ndarray, cell, pbc) -> np.ndarray:
        """`(n, n)` squared minimum-image distances, as `_molecule_separations`
        computes them before reducing over molecules."""
        cell = np.asarray(cell)
        dv = positions[:, None, :] - positions[None, :, :]
        if np.any(pbc):
            dv = dv - (pbc * np.floor(dv @ np.linalg.inv(cell) + 0.5)) @ cell
        return np.sum(dv * dv, -1)

    def get_network(self, topology: Topology, bimol_cutoff=4.0) -> ReactionNetwork:
        graph = nx.MultiGraph()

        mol_list: list[Topology] = list(topology.molecules())
        # Molecule-major atom order, so the pair scan can reduce over blocks.
        # Taking the positions straight from the system also avoids the ASE
        # `Atoms` slice per molecule that `molecules(return_atoms=True)` builds
        # and that nothing here wanted but the coordinates.
        nodes = [sorted(mol.graph.nodes()) for mol in mol_list]
        order = [node for group in nodes for node in group]
        starts = np.cumsum([0] + [len(group) for group in nodes[:-1]])
        signatures = [self._molecule_signature(mol) for mol in mol_list]

        separation = self._molecule_separations(
            topology.atoms.positions[order],
            starts,
            topology.atoms.cell,
            topology.atoms.pbc,
        )

        for i, imol in enumerate(mol_list):
            graph.add_node(i, molecule=imol)

            # reindex reaction to global indices
            for rxn, mol_map in self._reaction_channels(imol):
                graph.add_edge(i, i, reaction=rxn, mapping=mol_map)

            for j in range(i):
                # neighbor check
                if separation[i, j] > bimol_cutoff:
                    continue

                # Two disjoint molecules union to a topology whose signature is
                # exactly the merge of theirs, so the channel cache can be
                # probed without building the combined topology at all.  On a
                # settled trajectory that is the whole of this branch: building
                # it cost an `nx.union_all` and a fresh signature for every
                # neighbouring pair on every force call, for an answer already
                # in the cache.
                channels = self._pair_channels(
                    signatures[i], signatures[j], imol, mol_list[j]
                )

                # reindex reactions to global indices
                for rxn, mol_map in channels:
                    graph.add_edge(i, j, reaction=rxn, mapping=mol_map)

        return ReactionNetwork(graph)

    def load(self, path: str | Path) -> None:
        """Load ReactionSet data from a .json manifest"""
        self.data = ReactionSetData.from_manifest(path)
        self._reset_caches()

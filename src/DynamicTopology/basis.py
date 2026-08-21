"""Pivot-invariant diabatic basis for the multi-state EVB Hamiltonian.

The basis a multi-state EVB calculation spans has to be a property of the
nuclear coordinates.  Generating it as "the current topology plus everything one
reaction away from it" is not: for three or more states the set reachable from
one topology is not the set reachable from another, so `E` depends on what the
trajectory happened to be carrying rather than on `x`.  No energy is conserved,
forward and reverse paths across a barrier disagree, and the star shape it
produces also pins the ground-state weight of the pivot at exactly 1/2, which is
what made the old `c**2 > 0.9` swap test unreachable.

This module builds the basis by closure under a *geometric* criterion instead.
Starting anywhere, a neighbouring topology is admitted when mixing with it
actually lowers the energy:

    dH   = (H_child - H_parent) / 2
    stab = sqrt(dH**2 + V**2) - |dH|        admit when stab > eps

which is the stabilization of the lower root of the corresponding two-level
block.  Testing `|V| > eps` instead would be wrong in both directions: the
stabilization is quadratic in `V` when the diabats are well separated (so a large
coupling between distant diabats moves nothing) and linear when they are
degenerate (so a small coupling between crossing diabats moves everything).

**Why the result does not depend on where you start.**  The criterion is
symmetric under exchanging parent and child -- `dH` changes sign, `dH**2` and
`|dH|` do not -- so the admitted set is exactly the connected component of the
gate-passing state graph that contains the seed.  Every state in that component
generates the same component.  That is the invariance, and it makes the topology
carried between MD steps bookkeeping rather than a physical event.

The one thing that breaks it is truncation: breadth-first order depends on the
seed, so a basis cut short by `max_states` or `max_depth` is seed-dependent
again.  Those are safety valves, not parameters to tune, and a `Block` reports
`capped` whenever one of them actually refused a state that passed the gate.
`tests/test_evb_invariants.py` asserts they did not fire.

The closure is run per *block* -- a maximal set of reactions sharing atoms, from
`ReactionNetwork.reaction_blocks()`.  Atom-disjoint reactions commute, so their
blocks factorize and their energies add; putting them in one matrix instead makes
the ground state depend on how many unrelated channels happen to be enumerated
nearby.  Because `ReactionSet.get_network` only ever looks at the molecules of
the topology it is given, running it on a block's topology restricts the search
to that block for free.
"""

import logging
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from ase import Atoms

from DynamicTopology.core.reaction import Reaction
from DynamicTopology.core.reactionset import ReactionSet
from DynamicTopology.core.topology import Topology

logger: logging.Logger = logging.getLogger(__name__)

# Amplitude the HCombustion dataset ships for channels whose coupling has never
# been fitted.  Used only to report which states entered a basis on a
# placeholder; see `_is_placeholder`.
PLACEHOLDER_AMPLITUDE: float = -10.0

# A state is identified by which atoms it spans and how they are bonded.  The
# node set is part of the key because an edge set alone does not distinguish two
# blocks that are both fully dissociated -- each has no edges at all.
StateKey = tuple[frozenset, frozenset]


def state_key(topology: Topology) -> StateKey:
    return (
        frozenset(topology.graph.nodes()),
        frozenset(frozenset(edge) for edge in topology.graph.edges()),
    )


def _canonical(key: StateKey) -> tuple:
    """Sortable form of a state key, so basis ordering is reproducible."""
    nodes, edges = key
    return (tuple(sorted(nodes)), tuple(sorted(tuple(sorted(e)) for e in edges)))


def _is_placeholder(reaction: Reaction) -> bool:
    """Whether this reaction's coupling amplitude was never fitted.

    Prefers the explicit `provenance` key `scripts/fit.py` writes; falls back to
    recognising the shipped placeholder amplitude for term files predating it.
    """
    for term in reaction.terms:
        if term["type"] != "rmsd":
            continue
        provenance = term.get("provenance")
        if provenance is not None:
            return provenance != "fitted"
        amplitude = term["kwargs"].get("A")
        return amplitude is not None and abs(amplitude - PLACEHOLDER_AMPLITUDE) < 1e-9
    return False


@dataclass
class Block:
    """One independent EVB problem: a closed basis and its Hamiltonian.

    `states` is in a canonical order that does not depend on the seed, so two
    calculations that converge to the same basis produce identical blocks.
    `seed_index` locates the topology the caller came in with.
    """

    states: list[Topology]
    energies: np.ndarray  # (n,)
    forces: np.ndarray  # (n, natoms, 3)
    couplings: np.ndarray  # (n, n), zero diagonal
    coupling_forces: np.ndarray  # (n, n, natoms, 3)
    seed_index: int
    depth: int
    capped: bool
    placeholder_channels: list[str] = field(default_factory=list)

    @property
    def nstates(self) -> int:
        return len(self.states)

    def hamiltonian(self) -> tuple[np.ndarray, np.ndarray]:
        """The EVB matrix and its gradient, diagonals filled in."""
        diag = np.diag_indices(self.nstates)
        ham = self.couplings.copy()
        ham[diag] = self.energies
        fham = self.coupling_forces.copy()
        fham[diag] = self.forces
        return ham, fham


class EVBBasis:
    """Builds pivot-invariant diabatic bases for a fixed geometry.

    Caches are per geometry: `build` clears them, so one instance can be reused
    across force evaluations without going stale.
    """

    def __init__(
        self,
        reaction_set: ReactionSet,
        bonded_ff,
        coupling_ff,
        eps: float = 1e-3,
        max_states: int = 32,
        max_depth: int | None = None,
    ):
        self.reaction_set = reaction_set
        self.bonded_ff = bonded_ff
        self.coupling_ff = coupling_ff
        self.eps = eps
        self.max_states = max_states
        # Breadth-first depth is always below the basis size, so `max_states` is
        # the cap that binds and `max_depth` is off by default.  A small depth
        # limit truncates well before the closure converges -- at 4 it cut the
        # 7-state H2O+HO basis to 5 from some seeds and to 7 from others, which
        # is precisely the seed-dependence this module exists to remove.  Set it
        # only to deliberately bound multi-step chains.
        self.max_depth = max_states if max_depth is None else max_depth

        self._energy_cache: dict[StateKey, tuple[float, np.ndarray]] = {}
        self._reaction_cache: dict[StateKey, list[tuple[Reaction, dict]]] = {}
        self._coupling_cache: dict[tuple, tuple[float, np.ndarray]] = {}
        self._molecule_cache: dict[tuple, float] = {}

    # -- pieces the closure needs ------------------------------------------

    def _reactions(
        self, state: Topology, bimol_cutoff: float
    ) -> list[tuple[Reaction, dict]]:
        """Every reaction applicable to `state`, with its index mapping.

        `get_network` walks the molecules of the topology it is handed, so
        passing a block's topology keeps the search inside that block.
        """
        key = state_key(state)
        cached = self._reaction_cache.get(key)
        if cached is not None:
            return cached

        network = self.reaction_set.get_network(state, bimol_cutoff)
        found = [
            (data["reaction"], data["mapping"])
            for _, _, data in network.reactions()
        ]
        self._reaction_cache[key] = found
        return found

    def _molecule_energy(self, molecule: Topology, atoms: Atoms) -> float:
        """Bonded energy of a single molecule, memoized for this geometry.

        Keyed by molecule signature, so the molecules a reaction leaves alone are
        evaluated once for the whole block rather than once per candidate.
        """
        signature = self.reaction_set._molecule_signature(molecule)
        cached = self._molecule_cache.get(signature)
        if cached is not None:
            return cached

        terms = self.reaction_set.get_terms(molecule)
        nodes = sorted(molecule.graph.nodes())
        local = {node: i for i, node in enumerate(nodes)}
        term_dict = Topology(nx.Graph()).set_terms(
            [
                {
                    "type": term["type"],
                    "atoms": {k: local[v] for k, v in term["atoms"].items()},
                    "kwargs": term["kwargs"],
                }
                for term in terms
            ]
        )
        energy, _ = self.bonded_ff(
            atoms.positions[nodes], atoms.pbc, atoms.cell, term_dict
        )
        self._molecule_cache[signature] = energy
        return energy

    def _local_energy(self, graph: nx.Graph, atoms: Atoms) -> float:
        """Bonded energy of the molecules in `graph`, summed over components."""
        return sum(
            self._molecule_energy(Topology(graph.subgraph(nodes)), atoms)
            for nodes in nx.connected_components(graph)
        )

    def _admits_reaction(
        self, parent: Topology, mapping: dict, changes: tuple[set, set],
        coupling: float, atoms: Atoms,
    ) -> bool:
        """Would applying `reaction` to `parent` produce an admissible state?

        Decided without building the product.  The gate needs only the gap
        between the two diabats, and every molecule the reaction leaves alone
        contributes the same energy to both, so it cancels from the difference
        exactly -- screening on the reacting fragment is not an approximation.

        It is also what keeps the closure affordable.  A reaction template spans
        whole molecules, so the fragment is two molecules at most, while a block
        in a dense box can hold most of the system: 63 of 100 molecules on a
        200-atom box, where building and evaluating each whole candidate took
        1.3 s to produce a single state.  The great majority of candidates are
        rejected, and none of them now costs more than the reaction touches.
        """
        broken, formed = changes
        before = parent.graph.subgraph(frozenset(mapping.values()))
        after = before.copy()
        after.remove_edges_from(broken)
        after.add_edges_from(formed)
        return self._admits(
            self._local_energy(before, atoms),
            self._local_energy(after, atoms),
            coupling,
        )

    def _energy(self, state: Topology, atoms: Atoms) -> tuple[float, np.ndarray]:
        """Diabatic energy and forces of one state at the current geometry.

        Evaluated over the state's own atoms rather than the whole system.
        `QForce.__call__` forms a full N x N x 3 displacement array per call, so
        a three-atom state measured against a 200-atom box does some four
        thousand times the work it needs.  Restricting it here keeps every
        `compute_*` method untouched.

        Called only for states the closure actually admits -- candidates are
        screened with `_local_energy`, which is why this can afford to evaluate
        the block in one lump.

        Two index spaces meet, as in ACKS2: terms arrive in global indices, the
        sliced positions are in block order, and the forces are scattered back to
        global order before returning.
        """
        key = state_key(state)
        cached = self._energy_cache.get(key)
        if cached is not None:
            return cached

        if len(state.terms) == 0:
            state.set_terms(self.reaction_set.get_terms(state))

        nodes = sorted(state.graph.nodes())
        local = {node: i for i, node in enumerate(nodes)}
        term_dict = Topology(nx.Graph()).set_terms(
            [
                {
                    "type": term["type"],
                    "atoms": {k: local[v] for k, v in term["atoms"].items()},
                    "kwargs": term["kwargs"],
                }
                for term in state.terms
            ]
        )

        energy, local_forces = self.bonded_ff(
            atoms.positions[nodes], atoms.pbc, atoms.cell, term_dict
        )
        forces = np.zeros_like(atoms.positions)
        forces[nodes] = local_forces

        result = (energy, forces)
        self._energy_cache[key] = result
        return result

    def _coupling(
        self, atoms: Atoms, reaction: Reaction, mapping: dict
    ) -> tuple[float, np.ndarray]:
        """Off-diagonal coupling for one reaction channel, in global order.

        Row k of the live geometry is compared against row k of the stored
        transition-state reference, so the live indices must be ordered by
        *template* index.  `mapping` is keyed by template index but iterates in
        the isomorphism matcher's discovery order, so `mapping.values()` is
        generally a permutation of the wanted order and would superpose atoms
        onto the wrong reference positions.
        """
        cache_key = (reaction.hash(), tuple(sorted(mapping.items())))
        cached = self._coupling_cache.get(cache_key)
        if cached is not None:
            return cached

        order = [mapping[k] for k in sorted(mapping)]
        ensemble = np.stack([frame.positions for frame in reaction.atoms])
        energy, local_forces = self.coupling_ff(
            atoms.positions[order],
            atoms.pbc,
            atoms.cell,
            ensemble,
            reaction.term_dict,
        )

        forces = np.zeros_like(atoms.positions)
        forces[order, :] = local_forces
        result = (float(energy), forces)
        self._coupling_cache[cache_key] = result
        return result

    def _admits(self, parent_energy: float, child_energy: float, coupling: float) -> bool:
        """Does mixing these two diabats lower the energy by more than `eps`?"""
        half_gap = 0.5 * (child_energy - parent_energy)
        stabilization = float(np.hypot(half_gap, coupling)) - abs(half_gap)
        return stabilization > self.eps

    # -- the closure --------------------------------------------------------

    def build(
        self, atoms: Atoms, seed: Topology, bimol_cutoff: float = 4.0
    ) -> list[Block]:
        """Closed diabatic bases for every independent block of the system."""
        self._energy_cache.clear()
        self._reaction_cache.clear()
        self._coupling_cache.clear()
        self._molecule_cache.clear()

        network = self.reaction_set.get_network(seed, bimol_cutoff)
        blocks: list[Block] = []
        for reactions, molecules in network.reaction_blocks():
            block_seed = Topology.from_molecules(molecules, remap=False)
            block_seed.attach_atoms(atoms)
            # The reactions applicable to a block's seed are exactly the ones
            # `reaction_blocks` grouped it from -- a block's molecules are the
            # endpoints of its reaction edges -- so re-deriving them with another
            # `get_network` call per block is pure duplicated work.  Only the
            # states the closure discovers need enumerating from scratch.
            self._reaction_cache[state_key(block_seed)] = [
                (data["reaction"], data["mapping"]) for data in reactions
            ]
            blocks.append(self._close(atoms, block_seed, bimol_cutoff))
        return blocks

    def _close(self, atoms: Atoms, seed: Topology, bimol_cutoff: float) -> Block:
        seed_key = state_key(seed)
        admitted: dict[StateKey, Topology] = {seed_key: seed}
        depths: dict[StateKey, int] = {seed_key: 0}
        # Undirected: several symmetry-equivalent mappings can reach the same
        # product topology.  They are one diabatic state arrived at by
        # equivalent routes, so keep the most strongly coupled representative --
        # choosing on the coupling rather than on enumeration order is what makes
        # the choice independent of how the atoms happen to be labelled.
        edges: dict[frozenset, tuple[float, np.ndarray]] = {}
        placeholders: dict[str, None] = {}
        capped = False

        frontier: list[tuple[Topology, int]] = [(seed, 0)]
        while frontier:
            parent, depth = frontier.pop(0)
            parent_key = state_key(parent)

            for reaction, mapping in self._reactions(parent, bimol_cutoff):
                broken, formed = reaction.edge_changes(mapping)
                if not broken and not formed:
                    continue  # a no-op template, e.g. H + H -> H + H

                coupling, coupling_forces = self._coupling(atoms, reaction, mapping)
                if not self._admits_reaction(
                    parent, mapping, (broken, formed), coupling, atoms
                ):
                    continue

                # Only now is the product worth building: `apply` copies the
                # whole block graph, and so does deriving its state key.
                child = reaction.apply(parent, mapping, share_atoms=True)
                child_key = state_key(child)
                if child_key == parent_key:
                    continue

                pair = frozenset((parent_key, child_key))
                previous = edges.get(pair)
                if previous is None or abs(coupling) > abs(previous[0]):
                    edges[pair] = (coupling, coupling_forces)
                if _is_placeholder(reaction):
                    placeholders[reaction.equation()] = None

                if child_key in admitted:
                    continue
                # Expansion continues past `max_depth` so that `capped` is exact:
                # it fires only where a state that passed the gate was actually
                # refused, never merely because a frontier state ran out of
                # neighbours.
                if len(admitted) >= self.max_states or depth + 1 > self.max_depth:
                    capped = True
                    continue
                admitted[child_key] = child
                depths[child_key] = depth + 1
                frontier.append((child, depth + 1))

        if capped:
            logger.warning(
                "EVB basis truncated at %d states / depth %d; the basis is "
                "seed-dependent where this fires",
                len(admitted),
                self.max_depth,
            )

        # Canonical order, so the block does not remember which state seeded it.
        order = sorted(admitted, key=_canonical)
        index = {key: i for i, key in enumerate(order)}
        states = [admitted[key] for key in order]

        nstates = len(states)
        energies = np.zeros(nstates)
        forces = np.zeros((nstates,) + atoms.positions.shape)
        for i, state in enumerate(states):
            energies[i], forces[i] = self._energy(state, atoms)

        couplings = np.zeros((nstates, nstates))
        coupling_forces = np.zeros((nstates, nstates) + atoms.positions.shape)
        for pair, (value, gradient) in edges.items():
            keys = tuple(pair)
            if len(keys) != 2 or any(key not in index for key in keys):
                continue  # an edge to a state the caps refused
            i, j = index[keys[0]], index[keys[1]]
            couplings[i, j] = couplings[j, i] = value
            coupling_forces[i, j] = coupling_forces[j, i] = gradient

        return Block(
            states=states,
            energies=energies,
            forces=forces,
            couplings=couplings,
            coupling_forces=coupling_forces,
            seed_index=index[seed_key],
            depth=max(depths.values()),
            capped=capped,
            placeholder_channels=sorted(placeholders),
        )

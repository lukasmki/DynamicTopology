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

**Why the gate is a ramp and not a step.**  The ground state of a two-level
block is exactly `min(diabats) - stab`, so dropping a state costs precisely
`stab` and a hard threshold moves the energy discontinuously by up to `eps` every
time one crosses it.  That is not a small effect where it matters: it made `eps`
an accuracy knob on energy conservation rather than on basis size, and an NVE run
of a single H2O drifted by one `eps` per admission event, dt-independently.  So
the coupling is scaled by a switching function that rises from zero at `stab =
eps` to one at `stab = eps + switch_width`.  A state arrives decoupled,
contributing nothing, and gains its coupling smoothly; `Block.min_switch` reports
how far into a ramp a block currently sits.

The ramp runs *upward* from `eps`, so the admitted set is exactly what the hard
gate admitted -- widening downward would enlarge every basis, and `max_states`
already binds on a hot box.  What it costs is that the energy the truncation
discards is bounded by `eps + switch_width` rather than by `eps`.
`switch_width = 0` restores the step exactly.

Two discontinuities the switch does *not* remove, both recorded here because they
look like the same bug and are not:

  - `stab` has a kink at `dH = 0` (`|dH|` does), and the switch's slope amplifies
    it by `1/switch_width`.  It can only bite where a pair is near-degenerate
    *and* inside the ramp, which needs `|V| <= eps + switch_width` -- an
    essentially uncoupled degeneracy.  Softening `|dH|` to `hypot(dH, delta)`
    removes it if it is ever measured firing.
  - A state whose diabatic energy is *below* the current ground state changes it
    by `H_parent - H_child` on entry, not by `stab`.  That jump comes from the
    diagonal, and scaling `V` cannot touch it.  The gate excludes such a state
    whenever `|V| < sqrt(2*|dH|*eps)`, so it is reachable in principle.

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


def _smoothstep(t: float) -> float:
    """Quintic ramp on [0, 1], flat to second order at both ends.

    C2 rather than the cheaper C1 cubic: the first derivative is what the forces
    are, so a discontinuous *second* derivative is the coarsest thing that still
    leaves the force smooth.
    """
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def _smoothstep_slope(t: float) -> float:
    """d/dt of `_smoothstep`.  Zero at t = 0 and t = 1, which is the point."""
    return 30.0 * t * t * (t - 1.0) * (t - 1.0)


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
    virials: np.ndarray  # (n, 3, 3)
    coupling_virials: np.ndarray  # (n, n, 3, 3)
    seed_index: int
    depth: int
    capped: bool
    # Smallest switching weight on any coupling in this block, so a caller can
    # see whether the surface is currently inside a ramp.  1.0 means every
    # channel is at full strength, which is also what a single-state block
    # reports: there is nothing to switch.
    min_switch: float = 1.0
    placeholder_channels: list[str] = field(default_factory=list)

    @property
    def nstates(self) -> int:
        return len(self.states)

    def hamiltonian(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The EVB matrix and its two gradients, diagonals filled in.

        `vham` is the strain gradient in exactly the sense `fham` is the
        position gradient, so `System.calculate` contracts both with the same
        ground-state eigenvector.  One sign difference is deliberate and lives
        in the terms themselves: `fham` holds *forces* (`-dE/dr`) while `vham`
        holds virials (`+dE/de`).
        """
        diag = np.diag_indices(self.nstates)
        ham = self.couplings.copy()
        ham[diag] = self.energies
        fham = self.coupling_forces.copy()
        fham[diag] = self.forces
        vham = self.coupling_virials.copy()
        vham[diag] = self.virials
        return ham, fham, vham


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
        max_states: int = 64,
        max_depth: int | None = None,
        switch_width: float | None = None,
    ):
        self.reaction_set = reaction_set
        self.bonded_ff = bonded_ff
        self.coupling_ff = coupling_ff
        self.eps = eps
        # Width of the admission ramp, defaulting to one `eps`: a channel is
        # decoupled at `stab = eps` and at full strength at `stab = 2*eps`.
        # Zero collapses the ramp back to the hard threshold, which is how the
        # discontinuity it exists to remove can be reproduced on demand.
        self.switch_width = eps if switch_width is None else switch_width
        self.max_states = max_states
        # Breadth-first depth is always below the basis size, so `max_states` is
        # the cap that binds and `max_depth` is off by default.  A small depth
        # limit truncates well before the closure converges -- at 4 it cut the
        # 7-state H2O+HO basis to 5 from some seeds and to 7 from others, which
        # is precisely the seed-dependence this module exists to remove.  Set it
        # only to deliberately bound multi-step chains.
        self.max_depth = max_states if max_depth is None else max_depth

        self._bimol_cutoff: float = 4.0

        self._energy_cache: dict[StateKey, tuple[float, np.ndarray]] = {}
        self._reaction_cache: dict[StateKey, list[tuple[Reaction, dict]]] = {}
        self._coupling_cache: dict[tuple, tuple[float, np.ndarray]] = {}
        # (energy, nodes, forces over those nodes).  The forces are the ones the
        # bonded force field already returned and this used to discard; keeping
        # them costs no extra evaluation and is what the switch's gradient needs.
        self._molecule_cache: dict[tuple, tuple[float, list[int], np.ndarray]] = {}
        self._ensemble_cache: dict[Reaction, np.ndarray] = {}
        self._inv_cell_cache: np.ndarray | None = None

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
            (data["reaction"], data["mapping"]) for _, _, data in network.reactions()
        ]
        self._reaction_cache[key] = found
        return found

    def _molecule_terms(
        self, molecule: Topology, atoms: Atoms
    ) -> tuple[float, list[int], np.ndarray]:
        """Bonded energy and forces of a single molecule, memoized for this geometry.

        Keyed by molecule signature, so the molecules a reaction leaves alone are
        evaluated once for the whole block rather than once per candidate.  The
        forces are returned in the molecule's own node order, not scattered:
        almost every caller only wants the energy, and scattering into a
        system-sized array for each of the thousands of screening calls per step
        would cost more than the evaluation.
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
        energy, local_forces, local_virial = self.bonded_ff(
            atoms.positions[nodes], atoms.pbc, atoms.cell, term_dict
        )
        result = (energy, nodes, local_forces, local_virial)
        self._molecule_cache[signature] = result
        return result

    def _molecule_energy(self, molecule: Topology, atoms: Atoms) -> float:
        return self._molecule_terms(molecule, atoms)[0]

    def _local_energy(self, graph: nx.Graph, atoms: Atoms) -> float:
        """Bonded energy of the molecules in `graph`.

        Bonded only, because that is now the whole of what differs between two
        diabats.  Both nonbonded terms -- `ACKS2` and `ZBL` -- are functions of
        the geometry and the elements alone, so they contribute equally to every
        state of a block and cancel from the gap this screens on.  That is the
        property the previous Lennard-Jones design did not have, and paying for
        it here is what the Lennard-Jones correction that used to sit on this
        line was doing.
        """
        return sum(
            self._molecule_energy(Topology(graph.subgraph(nodes)), atoms)
            for nodes in nx.connected_components(graph)
        )

    def _local_gradients(
        self, graph: nx.Graph, atoms: Atoms
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bonded forces (global atom order) and virial of `graph`'s molecules.

        The counterpart of `_local_energy`, and taken only on the paths that
        need a gradient of the gap -- i.e. only for a channel sitting strictly
        inside the admission ramp.  Everything it reads is already in the
        molecule cache from the corresponding `_local_energy` call, so returning
        the virial alongside the forces costs nothing: both were computed by the
        one `bonded_ff` call the cache holds.
        """
        forces = np.zeros_like(atoms.positions)
        virial = np.zeros((3, 3))
        for nodes in nx.connected_components(graph):
            _, mol_nodes, local_forces, local_virial = self._molecule_terms(
                Topology(graph.subgraph(nodes)), atoms
            )
            forces[mol_nodes] += local_forces
            virial += local_virial
        return forces, virial

    def _channel_weight(
        self,
        parent: Topology,
        mapping: dict,
        changes: tuple[set, set],
        coupling: float,
        coupling_forces: np.ndarray,
        coupling_virial: np.ndarray,
        atoms: Atoms,
    ) -> tuple[float, np.ndarray | None]:
        """How strongly this channel couples, and the gradient of that weight.

        Returns `(weight, d weight / d r)`, with the gradient `None` wherever the
        weight is exactly 0 or 1 -- the switching function is flat at both ends,
        so outside the ramp there is no extra force term and no need for the
        fragment forces that computing one would cost.  A weight of 0 means the
        channel is not admitted at all.

        Decided without building the product.  The gate needs only the gap
        between the two diabats, and every molecule the reaction leaves alone
        contributes the same energy to both, so it cancels from the difference
        exactly -- screening on the reacting fragment is not an approximation,
        and neither is differentiating it.

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

        energy_before = self._local_energy(before, atoms)
        energy_after = self._local_energy(after, atoms)
        weight = self._switch(energy_before, energy_after, coupling)
        if weight <= 0.0 or weight >= 1.0:
            return weight, None, None

        # d(weight)/dr = S'(t)/width * d(stab)/dr, with
        #   stab = hypot(h, V) - |h|,  h = (E_after - E_before) / 2
        # so the chain rule needs the gap's gradient as well as the coupling's.
        # Forces are -dE/dr, hence the sign flips below.
        half_gap = 0.5 * (energy_after - energy_before)
        hyp = float(np.hypot(half_gap, coupling))
        stabilization = hyp - abs(half_gap)
        slope = (
            _smoothstep_slope((stabilization - self.eps) / self.switch_width)
            / self.switch_width
        )

        forces_before, virial_before = self._local_gradients(before, atoms)
        forces_after, virial_after = self._local_gradients(after, atoms)
        dgap = 0.5 * (forces_before - forces_after)
        dstabilization = (half_gap / hyp - np.sign(half_gap)) * dgap + (
            coupling / hyp
        ) * -coupling_forces

        # The same chain rule against strain rather than position.  The two sign
        # flips of the force convention are absent here because virials are
        # already `+dE/de`: `d(half_gap)/de` is `+0.5 * (W_after - W_before)`
        # where the force version is `0.5 * (F_before - F_after)`, and the
        # coupling enters as `+coupling_virial` where it entered as
        # `-coupling_forces`.
        dgap_strain = 0.5 * (virial_after - virial_before)
        dstabilization_strain = (half_gap / hyp - np.sign(half_gap)) * dgap_strain + (
            coupling / hyp
        ) * coupling_virial
        return weight, slope * dstabilization, slope * dstabilization_strain

    def _energy(
        self, state: Topology, atoms: Atoms
    ) -> tuple[float, np.ndarray, np.ndarray]:
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

        energy, local_forces, virial = self.bonded_ff(
            atoms.positions[nodes], atoms.pbc, atoms.cell, term_dict
        )
        forces = np.zeros_like(atoms.positions)
        forces[nodes] = local_forces

        # The virial needs no scatter: it is a 3x3 sum over the state's own
        # pairs, so the restriction to `nodes` that makes this call cheap does
        # not change it.
        result = (energy, forces, virial)
        self._energy_cache[key] = result
        return result

    def _ensemble(self, reaction: Reaction) -> np.ndarray:
        """The reaction's stored transition-state frames as one (E, n, 3) array.

        Stacked once per reaction template per process rather than once per
        channel: the frames are immutable template data, and restacking them is
        pure allocation on a path entered thousands of times per force call.
        """
        cached = self._ensemble_cache.get(reaction)
        if cached is None:
            cached = np.stack([frame.positions for frame in reaction.atoms])
            self._ensemble_cache[reaction] = cached
        return cached

    def _channel_couplings(
        self, atoms: Atoms, channels: list[tuple[Reaction, dict]]
    ) -> list[tuple[float, np.ndarray, list[int], np.ndarray]]:
        """Couplings for many channels at once, as `(energy, forces, order, virial)`.

        The forces are in *channel* order -- row k belongs to atom `order[k]` --
        and are deliberately not scattered into a system-sized array.  Almost
        every channel is rejected on `abs(energy) <= eps` without its gradient
        ever being read, and allocating a (natoms, 3) zero array per channel to
        hold a seven-row answer cost more than computing it.

        Channels are grouped by reaction template because that is what makes the
        batch well shaped: one template fixes the fragment size and the
        transition-state ensemble, so every channel of a template superposes the
        same number of atoms onto the same reference and `_kabsch` can take them
        in one stack.

        Row k of the live geometry is compared against row k of the stored
        reference, so the live indices must be ordered by *template* index.
        `mapping` is keyed by template index but iterates in the isomorphism
        matcher's discovery order, so `mapping.values()` is generally a
        permutation of the wanted order and would superpose atoms onto the wrong
        reference positions.
        """
        results: list[tuple[float, np.ndarray, list[int], np.ndarray] | None]
        results = [None] * len(channels)

        # Keyed on the reaction *object* and the live indices it acts on.  The
        # template fixes which key of `mapping` each position of `order` came
        # from, so the pair identifies the channel exactly, and identity is what
        # groups the batch anyway -- deriving a key from `reaction.hash()` and a
        # sorted copy of `mapping.items()` instead sorted every mapping twice
        # per channel, on a path entered a few thousand times per force call.
        groups: dict[Reaction, list[int]] = {}
        orders: list[list[int]] = []
        keys: list[tuple] = []
        for i, (reaction, mapping) in enumerate(channels):
            order = [mapping[k] for k in sorted(mapping)]
            orders.append(order)
            key = (id(reaction), tuple(order))
            keys.append(key)
            cached = self._coupling_cache.get(key)
            if cached is not None:
                results[i] = cached
            else:
                groups.setdefault(reaction, []).append(i)

        if groups:
            inv_cell = self._inv_cell(atoms)
            positions = atoms.positions
            for reaction, members in groups.items():
                energies, forces, virials = self.coupling_ff(
                    positions[[orders[i] for i in members]],
                    atoms.pbc,
                    atoms.cell,
                    self._ensemble(reaction),
                    reaction.term_dict,
                    inv_cell=inv_cell,
                )
                for k, i in enumerate(members):
                    result = (float(energies[k]), forces[k], orders[i], virials[k])
                    self._coupling_cache[keys[i]] = result
                    results[i] = result

        return results  # type: ignore[return-value]

    def _inv_cell(self, atoms: Atoms) -> np.ndarray | None:
        """`inv(cell)` for the current geometry, or None for an open system."""
        if not np.any(atoms.pbc):
            return None
        if self._inv_cell_cache is None:
            self._inv_cell_cache = np.linalg.inv(atoms.cell)
        return self._inv_cell_cache

    def _coupling(
        self, atoms: Atoms, reaction: Reaction, mapping: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Off-diagonal coupling for one reaction channel, in global order.

        A single-channel view of `_channel_couplings` that pays the scatter into
        a system-sized force array.  The closure does not use it -- it works in
        channel order and scatters only what it admits -- but it is the honest
        statement of what a channel's coupling is, and what a caller comparing
        one channel against a whole-state evaluation wants.
        """
        energy, local_forces, order, virial = self._channel_couplings(
            atoms, [(reaction, mapping)]
        )[0]
        forces = np.zeros_like(atoms.positions)
        forces[order, :] = local_forces
        return energy, forces, virial

    def _switch(
        self, parent_energy: float, child_energy: float, coupling: float
    ) -> float:
        """Weight in [0, 1] this channel's coupling carries in the Hamiltonian.

        Zero where mixing the two diabats lowers the energy by less than `eps`,
        one where it lowers it by more than `eps + switch_width`, and a quintic
        ramp between -- so a state joins the basis decoupled and gains its
        coupling smoothly instead of arriving at full strength.

        Symmetric under exchanging the two diabats and under the sign of the
        coupling, exactly rather than to a tolerance: `hypot` and `abs` are both
        even in the half gap.  That symmetry is the whole invariance argument --
        the admitted set is the seed's connected component only if the edge
        relation is undirected.
        """
        half_gap = 0.5 * (child_energy - parent_energy)
        stabilization = float(np.hypot(half_gap, coupling)) - abs(half_gap)
        if stabilization <= self.eps:
            return 0.0
        if self.switch_width <= 0.0 or stabilization >= self.eps + self.switch_width:
            return 1.0
        return _smoothstep((stabilization - self.eps) / self.switch_width)

    def _admits(
        self, parent_energy: float, child_energy: float, coupling: float
    ) -> bool:
        """Does mixing these two diabats lower the energy by more than `eps`?"""
        return self._switch(parent_energy, child_energy, coupling) > 0.0

    # -- the closure --------------------------------------------------------

    def build(
        self, atoms: Atoms, seed: Topology, bimol_cutoff: float = 4.0
    ) -> list[Block]:
        """Closed diabatic bases for every independent block of the system."""
        self._energy_cache.clear()
        self._reaction_cache.clear()
        self._coupling_cache.clear()
        self._molecule_cache.clear()
        self._inv_cell_cache = None
        self._bimol_cutoff = bimol_cutoff

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
        edges: dict[frozenset, tuple[float, np.ndarray, np.ndarray, float]] = {}
        placeholders: dict[str, None] = {}
        capped = False

        frontier: list[tuple[Topology, int]] = [(seed, 0)]
        while frontier:
            parent, depth = frontier.pop(0)
            parent_key = state_key(parent)

            # Every channel of this parent at once.  The couplings are the
            # dominant cost of the closure and each one is a superposition of a
            # handful of atoms onto a template, so evaluating them one at a time
            # spends nearly all of its time in fixed per-call overhead; see
            # `forcefield.coupling._kabsch`.
            channels = []
            for reaction, mapping in self._reactions(parent, bimol_cutoff):
                broken, formed = reaction.edge_changes(mapping)
                if not broken and not formed:
                    continue  # a no-op template, e.g. H + H -> H + H
                channels.append((reaction, mapping, broken, formed))

            couplings = self._channel_couplings(
                atoms, [(reaction, mapping) for reaction, mapping, _, _ in channels]
            )

            for (reaction, mapping, broken, formed), (
                coupling,
                local_coupling_forces,
                order,
                coupling_virial,
            ) in zip(channels, couplings):
                # `stab = hypot(h, V) - |h| <= |V|` for every gap `h`, by the
                # triangle inequality, so a channel whose raw coupling is
                # already below `eps` cannot clear the gate whatever the gap
                # turns out to be.  Skipping it here is exact, not a screen:
                # `_switch` would return 0.0 and the `weight <= 0.0` test below
                # would drop the channel anyway.  It is worth testing because
                # the gap is the expensive half -- `_channel_weight` rewires the
                # fragment and evaluates the bonded energy of both sides -- and
                # in a condensed phase almost every enumerated channel fails
                # here: 100% of 3888 on a 64-water box, 94% of 584 on the
                # H2/O2 mixture, where it is 40% of the force call.
                if abs(coupling) <= self.eps:
                    continue

                # Past the gate, and only here, is a system-sized gradient worth
                # building: `_channel_weight` and the Hamiltonian both want the
                # coupling's forces in global atom order.
                coupling_forces = np.zeros_like(atoms.positions)
                coupling_forces[order, :] = local_coupling_forces

                weight, dweight, dweight_virial = self._channel_weight(
                    parent,
                    mapping,
                    (broken, formed),
                    coupling,
                    coupling_forces,
                    coupling_virial,
                    atoms,
                )
                if weight <= 0.0:
                    continue

                # V_eff = weight * V, so the force picks up the switch's own
                # gradient.  Dropping that term would leave the forces
                # inconsistent with the energy wherever a channel is ramping.
                value = weight * coupling
                gradient = weight * coupling_forces
                virial_gradient = weight * coupling_virial
                if dweight is not None:
                    gradient = gradient - coupling * dweight
                    # `W_eff = d(weight*V)/de = weight*W_V + V*d(weight)/de`.
                    # The sign is `+` where the force line above is `-`, for the
                    # same reason as in `_channel_weight`: a force is `-dE/dr`
                    # and a virial is `+dE/de`.
                    virial_gradient = virial_gradient + coupling * dweight_virial

                # Only now is the product worth building: `apply` copies the
                # whole block graph, and so does deriving its state key.
                child = reaction.apply(parent, mapping, share_atoms=True)
                child_key = state_key(child)
                if child_key == parent_key:
                    continue

                pair = frozenset((parent_key, child_key))
                previous = edges.get(pair)
                # Compared on the switched coupling, since that is what enters
                # the matrix: a route with the larger raw coupling but a smaller
                # weight is the weaker one.
                if previous is None or abs(value) > abs(previous[0]):
                    edges[pair] = (value, gradient, virial_gradient, weight)
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
        virials = np.zeros((nstates, 3, 3))
        for i, state in enumerate(states):
            energies[i], forces[i], virials[i] = self._energy(state, atoms)

        couplings = np.zeros((nstates, nstates))
        coupling_forces = np.zeros((nstates, nstates) + atoms.positions.shape)
        coupling_virials = np.zeros((nstates, nstates, 3, 3))
        min_switch = 1.0
        for pair, (value, gradient, virial_gradient, weight) in edges.items():
            keys = tuple(pair)
            if len(keys) != 2 or any(key not in index for key in keys):
                continue  # an edge to a state the caps refused
            i, j = index[keys[0]], index[keys[1]]
            couplings[i, j] = couplings[j, i] = value
            coupling_forces[i, j] = coupling_forces[j, i] = gradient
            coupling_virials[i, j] = coupling_virials[j, i] = virial_gradient
            min_switch = min(min_switch, weight)

        return Block(
            states=states,
            energies=energies,
            forces=forces,
            virials=virials,
            couplings=couplings,
            coupling_forces=coupling_forces,
            coupling_virials=coupling_virials,
            seed_index=index[seed_key],
            depth=max(depths.values()),
            capped=capped,
            min_switch=min_switch,
            placeholder_channels=sorted(placeholders),
        )

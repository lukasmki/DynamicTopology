"""System class definition"""

from DynamicTopology.utils import log_debug

import logging
from typing import Any

import numpy as np
from ase import Atoms

from DynamicTopology.basis import Block, EVBBasis
from DynamicTopology.core.reactionset import ReactionSet
from DynamicTopology.core.topology import Topology
from DynamicTopology.forcefield.coupling import EVBCoupling
from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.forcefield.electrostatics import Electrostatics, ElectrostaticGap
from DynamicTopology.forcefield.lj import LennardJones
from DynamicTopology.forcefield.neighbors import Geometry, NeighborList
from DynamicTopology.forcefield.zbl import ZBL

logger: logging.Logger = logging.getLogger(__name__)

# How much more ground-state weight a state needs before it takes over as the
# topology carried to the next step.  The basis is seed-independent, so the
# pivot no longer affects the energy and this is only here to stop the label
# thrashing between near-degenerate states in the trajectory log.
PIVOT_HYSTERESIS: float = 0.1

# Convergence of the block sweep, on the largest change in any ground-state
# weight between two sweeps, and the most sweeps before giving up.  Only
# `pointcharge` electrostatics, or a 12-6 whose templates disagree on two
# blocks at once, ever needs more than one, and only when two or more blocks
# are multi-state at once; see `forcefield/pointcharge.py`.  The
# Hellmann-Feynman forces are exact at convergence, so the tolerance is an
# energy-conservation knob and is set well below anything an integrator sees.
SCF_TOLERANCE: float = 1e-10
SCF_MAX_SWEEPS: int = 100


class System:
    """One reactive force call, independent of ASE.

    Holds the geometry, the current topology and the reaction set.
    `calculate` builds the EVB basis, diagonalizes each block and returns a
    dict with `energy` (eV), `forces` (eV/A), `virial` (eV), the next
    `topology`, the energy broken into its parts and a per-block `blocks`
    summary.  `update` swaps in new atoms or a new topology between calls.
    """

    def __init__(
        self,
        atoms: Atoms,
        topology: Topology,
        reaction_set: ReactionSet,
        bimol_cutoff: float = 4.0,
        evb: dict[str, Any] | None = None,
    ):
        self.atoms: Atoms = atoms
        self.topology: Topology = topology
        self.reaction_set: ReactionSet = reaction_set
        self.bimol_cutoff: int | float = bimol_cutoff

        self.bonded_ff = QForce()
        # `ACKS2` or `PointCharge`, whichever the dataset's `global_params`
        # names; resolved per call through `nonbonded_ff`.
        self.electrostatics = Electrostatics()
        self.zbl_ff = ZBL()
        self.lj_ff = LennardJones()
        # The pair terms' neighbour list, kept from one call to the next for
        # as long as `global_params.neighbor_skin` allows; see `neighbors.py`.
        self.neighbors = NeighborList()
        self.coupling = EVBCoupling()
        # The electrostatic half of the admission gate's gap; see `basis.py`.
        self.gap = ElectrostaticGap(reaction_set)
        # `evb` passes `EVBBasis`'s own knobs -- `eps`, `switch_width`,
        # `max_states`, `max_depth` -- straight through, defaults untouched when
        # it is None.  They were reachable only by mutating the basis after
        # construction, which meant a production run could not pin them and a
        # `config.json` could not record them.  They change the potential energy
        # surface, so anything that varies them has to say so.
        self.basis = EVBBasis(
            reaction_set, self.bonded_ff, self.coupling, gap=self.gap, **(evb or {})
        )

    @property
    def nonbonded_ff(self):
        """The electrostatic term the active parameters name."""
        return self.electrostatics.get()

    def __repr__(self) -> str:
        return f"System( {repr(self.atoms)}, {repr(self.topology)} )"

    def update(self, atoms: Atoms | None = None, topology: Topology | None = None):
        if atoms is not None:
            self.atoms = atoms
        if topology is not None:
            self.topology = topology

    @staticmethod
    def _pivot(weights: np.ndarray, incumbent: int) -> int:
        """Which state's topology to carry into the next step.

        The dominant diabat, with a margin against the state already held.  The
        old rule asked for a weight above 0.9, which a star Hamiltonian with
        near-degenerate diagonals can never produce -- its ground state is
        (|0> - |u>)/sqrt(2), pinning the pivot weight at exactly 1/2 however
        strong the coupling.  With every off-diagonal filled the weights are
        free to concentrate, so the dominant state is a meaningful choice again.
        """
        best = int(np.argmax(weights))
        if best == incumbent:
            return incumbent
        if weights[best] - weights[incumbent] > PIVOT_HYSTERESIS:
            return best
        return incumbent

    def calculate(self) -> dict[str, Any]:
        pos = self.atoms.positions
        pbc = self.atoms.pbc
        cell = self.atoms.cell

        # The electrostatic kernel first: the admission gate screens on the
        # electrostatic gap as well as the bonded one, so the closure needs it.
        nonbonded = self.nonbonded_ff
        self.reaction_set.assign_terms(self.topology)
        # Every pair term's minimum-image geometry, shared: one neighbour list,
        # at `neighbor_radius` plus the skin and cut down for each term, and the
        # dense `N x N` displacements only if a term with no cutoff asks.
        displacements = Geometry(pos, pbc, np.asarray(cell), neighbors=self.neighbors)
        nonbonded.prepare(
            pos, pbc, cell, self.topology.term_dict, displacements=displacements
        )
        self.gap.bind(nonbonded, self.topology.term_dict)
        self.lj_ff.prepare(
            pos, pbc, cell, self.topology.term_dict, displacements=displacements
        )

        # Close the diabatic basis around the current geometry.  The result does
        # not depend on which topology is passed as the seed; see basis.py.
        evb_blocks: list[Block] = self.basis.build(
            self.atoms, self.topology, self.bimol_cutoff
        )
        log_debug(logger, f"Closed {len(evb_blocks)} EVB blocks")

        # The charges of every diabatic state, before anything is diagonalized.
        # Both electrostatic terms give each state a scalar on its block's
        # diagonal -- its whole electrostatic energy -- and both are coupled to
        # the other blocks through their weight-averaged charges:
        #
        #   - `ACKS2` solves each state's own equilibration: its reference
        #     charges, its molecules, its exclusions.  The atoms outside every
        #     multi-state block are factored once and folded into each state's
        #     small solve, so the environment's polarization answers each state
        #     exactly.  See `forcefield/acks2.py`.
        #   - `PointCharge` carries each state's template charges.  See
        #     `forcefield/pointcharge.py`.
        nonbonded.bind(evb_blocks)
        # The 12-6 the same way, for the same reason: each state takes its sigma
        # and epsilon from its own templates.  On a block whose states agree --
        # every block, while the templates carry per-element values -- its
        # corrections are zero and it stays off the diagonal entirely.
        self.lj_ff.bind(evb_blocks)

        # Diagonalize every multi-state block, and repeat until no block's
        # ground state moves.  Each block sees the others through their
        # weight-averaged charges; with one multi-state block its environment is
        # a set of single-state blocks nothing can move, so one sweep is final,
        # and the loop only iterates when two multi-state blocks see each other.
        multi = [i for i, block in enumerate(evb_blocks) if block.nstates > 1]
        hamiltonians = {i: evb_blocks[i].hamiltonian() for i in multi}
        solutions: dict[int, tuple] = {}
        sweeps = 0
        while True:
            sweeps += 1
            # Nothing to compare against on the first sweep.
            change = 0.0 if sweeps > 1 else np.inf
            for i in multi:
                block = evb_blocks[i]
                ham0, _, _ = hamiltonians[i]
                corrections = nonbonded.corrections(i) + self.lj_ff.corrections(i)
                # The correction goes on the diagonal and nowhere else, so it
                # does what it is here to do -- decide which bonding pattern is
                # lower -- while its *gradient* stays out of `fham`.  The
                # Hellmann-Feynman sum `sum_s w_s d(correction_s)/dr` is linear
                # in the weights and collapses into the single
                # `nonbonded.evaluate()` call below.
                ham = ham0.copy()
                ham[np.diag_indices(block.nstates)] += corrections
                eigval, eigvec = np.linalg.eigh(ham)
                statevec = eigvec[:, 0]
                weights = statevec * statevec
                previous = solutions.get(i)
                if previous is not None:
                    change = max(change, float(np.max(np.abs(weights - previous[3]))))
                solutions[i] = (ham, eigval, statevec, weights, corrections)
                nonbonded.update(i, weights)
                self.lj_ff.update(i, weights)
            self_consistent = nonbonded.self_consistent or self.lj_ff.self_consistent
            if not self_consistent or len(multi) <= 1:
                break
            if change < SCF_TOLERANCE:
                break
            if sweeps >= SCF_MAX_SWEEPS:
                logger.warning(
                    "EVB nonbonded terms not self-consistent after %d sweeps "
                    "(largest weight change %.2e); forces are not exact",
                    sweeps,
                    change,
                )
                break

        energy = 0.0
        forces = np.zeros_like(pos)
        # dE/d(strain), a single 3x3 for the whole system.  `ase.py` divides by
        # the cell volume to get the stress; keeping it as a virial here means
        # a non-periodic system (zero volume) is still well defined.
        virial = np.zeros((3, 3))
        final_states: list[Topology] = []
        blocks: list[dict[str, Any]] = []

        for i, block in enumerate(evb_blocks):
            log_debug(logger, f"block {i}, nstates = {block.nstates}")

            if block.nstates == 1:
                corrections = nonbonded.corrections(i) + self.lj_ff.corrections(i)
                energy += block.energies[0]
                forces += block.forces[0]
                virial += block.virials[0]
                final_states.append(block.states[0])
                weights = np.ones(1)
                pivot = 0
                gap = np.inf
                block_energy = float(block.energies[0] + corrections[0])
            else:
                ham, eigval, statevec, weights, corrections = solutions[i]
                _, fham, vham = hamiltonians[i]

                energy_gs = np.einsum("i,ij,j->", statevec, ham, statevec)
                forces_gs = np.einsum("i,ijnd,j->nd", statevec, fham, statevec)
                # The same Hellmann-Feynman contraction against strain.  It is
                # legitimate for the same reason the forces are: the eigenvector
                # is stationary, so its own derivative contributes nothing at
                # first order and only the matrix's explicit dependence survives.
                virial_gs = np.einsum("i,ijab,j->ab", statevec, vham, statevec)

                # `energy_gs` carries the corrections; the electrostatic term
                # below carries them too.  Taking them off here is what keeps
                # `energy_bonded` the bonded energy and stops them being counted
                # twice.  The forces need no such subtraction: `fham` never had
                # the corrections in it.
                energy += energy_gs - float(weights @ corrections)
                forces += forces_gs
                virial += virial_gs

                pivot = self._pivot(weights, block.seed_index)
                final_states.append(block.states[pivot])
                gap = float(eigval[1] - eigval[0])
                block_energy = float(energy_gs)

            blocks.append(
                {
                    "nstates": block.nstates,
                    "weights": weights,
                    "gap": gap,
                    "pivot": pivot,
                    "energy": block_energy,
                    "basis_size": block.nstates,
                    "depth": block.depth,
                    "capped": block.capped,
                    "min_switch": block.min_switch,
                    "placeholder_channels": block.placeholder_channels,
                }
            )

        energy_bonded = energy

        # Electrostatics at the ground-state weights just found: the whole
        # term, blocks and environment, with `sum_s w_s d(correction_s)/dr`
        # written as one weight matrix so it is one kernel contraction.  The
        # per-state charges are solved inside the Hamiltonian, which is what
        # keeps the result independent of the pivot.
        en_nb, fr_nb, w_nb = nonbonded.evaluate()
        energy += en_nb
        forces += fr_nb
        virial += w_nb

        # ZBL over *every* pair, bonded ones included and nothing excluded.
        # This is the term that opposes ACKS2's contact funnel; see
        # `forcefield/zbl.py` for why it takes no topology.  Being the same
        # number for every diabatic state of every block, adding it once here
        # shifts each diagonal equally, which shifts the ground-state eigenvalue
        # by exactly that constant and leaves the eigenvectors alone.
        en_zbl, fr_zbl, w_zbl = self.zbl_ff(
            pos, self.atoms.numbers, pbc, cell, displacements=displacements
        )
        energy += en_zbl
        forces += fr_zbl
        virial += w_zbl

        # 12-6 over every pair, switched on exactly where ZBL switches off, and
        # -- like ZBL and unlike every earlier attempt at this term -- with no
        # exclusions.  It can be applied to bonded pairs because `lj.switch` has
        # already taken it to zero there, which is the whole reason the four
        # failure modes in `forcefield/lj.py`'s docstring cannot recur: they all
        # descended from a broken bond paying hundreds of eV for a pair at the
        # bond length, and the term is worth ~1e-3 eV there now.  While the
        # templates agree on every atom's parameters it is the same number for
        # every diabatic state, so it shifts every EVB diagonal equally and
        # leaves the eigenvectors alone, exactly as ZBL does.  Where they do not,
        # the state-dependent part went on the diagonal above, and this is the
        # whole sum at the ground-state weights.
        #
        # This is the term that supplies the intermolecular wall and the
        # dispersion.  ZBL's taper removed the first and the model never had the
        # second; see `production/density-300K/README.md`.
        en_lj, fr_lj, w_lj = self.lj_ff.evaluate()
        energy += en_lj
        forces += fr_lj
        virial += w_lj

        # combine block topologies
        new_topo = Topology.from_molecules(final_states, remap=False)
        new_topo.set_atoms(self.atoms)

        results: dict[str, Any] = {
            "energy": energy,
            "forces": forces,
            "virial": virial,
            "topology": new_topo,
            "energy_bonded": energy_bonded,
            "energy_nonbonded": en_nb,
            "energy_zbl": en_zbl,
            "energy_lj": en_lj,
            "blocks": blocks,
            "electrostatics_sweeps": sweeps,
        }

        return results

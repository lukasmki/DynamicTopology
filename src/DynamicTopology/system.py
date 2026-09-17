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
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.lj import LennardJones
from DynamicTopology.forcefield.zbl import ZBL

logger: logging.Logger = logging.getLogger(__name__)

# How much more ground-state weight a state needs before it takes over as the
# topology carried to the next step.  The basis is seed-independent, so the
# pivot no longer affects the energy and this is only here to stop the label
# thrashing between near-degenerate states in the trajectory log.
PIVOT_HYSTERESIS: float = 0.1


# An empty `(0, 2)` pair list, so a state with no exclusions costs no allocation.
_NO_PAIRS: np.ndarray = np.zeros((0, 2), dtype=int)


def _exclusion_pairs(state: Topology) -> np.ndarray:
    """The pairs this diabatic state's templates take out of the charge kernel.

    Global atom indices, straight off the state's own term dict -- `EVBBasis`
    has already given every state in a block its terms in order to evaluate its
    bonded energy, so this is a dictionary lookup rather than another pass over
    the reaction set.  A dataset that ships no `coulombexclusion` terms yields
    nothing here and the electrostatics are unscreened, exactly as before they
    existed.
    """
    params = state.term_dict.get("coulombexclusion")
    return _NO_PAIRS if params is None else params["atoms"][:, :2]


class System:
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
        self.nonbonded_ff = ACKS2()
        self.zbl_ff = ZBL()
        self.lj_ff = LennardJones()
        self.coupling = EVBCoupling()
        # `evb` passes `EVBBasis`'s own knobs -- `eps`, `switch_width`,
        # `max_states`, `max_depth` -- straight through, defaults untouched when
        # it is None.  They were reachable only by mutating the basis after
        # construction, which meant a production run could not pin them and a
        # `config.json` could not record them.  They change the potential energy
        # surface, so anything that varies them has to say so.
        self.basis = EVBBasis(
            reaction_set, self.bonded_ff, self.coupling, **(evb or {})
        )

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

        # Close the diabatic basis around the current geometry.  The result does
        # not depend on which topology is passed as the seed; see basis.py.
        evb_blocks: list[Block] = self.basis.build(
            self.atoms, self.topology, self.bimol_cutoff
        )
        log_debug(logger, f"Closed {len(evb_blocks)} EVB blocks")

        # The charges, before anything is diagonalized.  `ACKS2.prepare` solves
        # the *unscreened* equilibration, which depends on the coordinates and
        # the elements alone, so it is the same problem for every diabatic state
        # of every block and is solved once here.  What is not the same for
        # every state is which pairs are intramolecular, and that correction
        # needs the charges -- hence this order.  See `forcefield/exclusions.py`.
        terms = self.reaction_set.get_terms(self.topology)
        self.topology.set_terms(terms)
        self.nonbonded_ff.prepare(pos, pbc, cell, self.topology.term_dict)

        energy = 0.0
        forces = np.zeros_like(pos)
        # dE/d(strain), a single 3x3 for the whole system.  `ase.py` divides by
        # the cell volume to get the stress; keeping it as a virial here means
        # a non-periodic system (zero volume) is still well defined.
        virial = np.zeros((3, 3))
        final_states: list[Topology] = []
        blocks: list[dict[str, Any]] = []
        # One `(pairs, weight)` entry per diabatic state, accumulated across
        # every block and handed to `ACKS2.screen_matrix` below.
        weighted_pairs: list[tuple[np.ndarray, float]] = []

        for i, block in enumerate(evb_blocks):
            log_debug(logger, f"block {i}, nstates = {block.nstates}")

            # What each state of this block would have the charge kernel stop
            # counting, and what that is worth at the charges already solved.
            # A lookup per excluded pair, which is the whole reason this can sit
            # on a diagonal the charge solve itself could never afford to.
            pairs = [_exclusion_pairs(state) for state in block.states]
            corrections = np.array(
                [self.nonbonded_ff.exclusion_energy(p) for p in pairs]
            )

            if block.nstates == 1:
                energy += block.energies[0]
                forces += block.forces[0]
                virial += block.virials[0]
                final_states.append(block.states[0])
                weights = np.ones(1)
                pivot = 0
                gap = np.inf
                block_energy = float(block.energies[0] + corrections[0])
            else:
                ham, fham, vham = block.hamiltonian()
                # The exclusion correction goes on the diagonal and nowhere
                # else, so it does what it is here to do -- decide which bonding
                # pattern is lower -- while its *gradient* stays out of `fham`.
                # Putting it there would cost one kernel contraction and one
                # adjoint solve per state; leaving it out costs nothing, because
                # the Hellmann-Feynman sum `sum_s w_s d(correction_s)/dr` is
                # linear in the masks and collapses into the single screened
                # `ACKS2.compute` call below.
                ham[np.diag_indices(block.nstates)] += corrections
                eigval, eigvec = np.linalg.eigh(ham)
                statevec = eigvec[:, 0]

                energy_gs = np.einsum("i,ij,j->", statevec, ham, statevec)
                forces_gs = np.einsum("i,ijnd,j->nd", statevec, fham, statevec)
                # The same Hellmann-Feynman contraction against strain.  It is
                # legitimate for the same reason the forces are: the eigenvector
                # is stationary, so its own derivative contributes nothing at
                # first order and only the matrix's explicit dependence survives.
                virial_gs = np.einsum("i,ijab,j->ab", statevec, vham, statevec)

                weights = statevec * statevec
                # `energy_gs` carries the corrections; the electrostatic term
                # below carries them too, in the screen.  Taking them off here
                # is what keeps `energy_bonded` the bonded energy and stops the
                # exclusion being counted twice.  The forces need no such
                # subtraction: `fham` never had the corrections in it.
                energy += energy_gs - float(weights @ corrections)
                forces += forces_gs
                virial += virial_gs

                pivot = self._pivot(weights, block.seed_index)
                final_states.append(block.states[pivot])
                gap = float(eigval[1] - eigval[0])
                block_energy = float(energy_gs)

            weighted_pairs.extend(zip(pairs, (float(w) for w in weights)))

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

        # Electrostatics, at the charges solved before the loop and screened by
        # what the ground state just decided is bonded.  The screen is
        # `1 - sum_s w_s M_s`: a pair intramolecular in every state of its block
        # drops out entirely, one intramolecular in only some of them drops out
        # by however much ground state those states hold.  That is not an
        # interpolation -- it is `sum_s w_s d(correction_s)/dr` written as one
        # weight matrix, which the linearity of the kernel contraction permits,
        # so the exclusion costs no extra contraction and no extra solve.
        #
        # The charges themselves are never screened.  Screening them is what
        # makes them topology-dependent, and a topology-dependent charge solve
        # outside the EVB Hamiltonian is exactly the pivot dependence
        # `tests/test_evb_invariants.py` measures.
        screen = self.nonbonded_ff.screen_matrix(weighted_pairs)
        en_nb, fr_nb, w_nb = self.nonbonded_ff.compute(screen)
        energy += en_nb
        forces += fr_nb
        virial += w_nb

        # ZBL over *every* pair, bonded ones included and nothing excluded.
        # This is the term that opposes ACKS2's contact funnel; see
        # `forcefield/zbl.py` for why it takes no topology.  Being the same
        # number for every diabatic state of every block, adding it once here
        # shifts each diagonal equally, which shifts the ground-state eigenvalue
        # by exactly that constant and leaves the eigenvectors alone.
        en_zbl, fr_zbl, w_zbl = self.zbl_ff(pos, self.atoms.numbers, pbc, cell)
        energy += en_zbl
        forces += fr_zbl
        virial += w_zbl

        # 12-6 over every pair, switched on exactly where ZBL switches off, and
        # -- like ZBL and unlike every earlier attempt at this term -- with no
        # exclusions.  It can be applied to bonded pairs because `lj.switch` has
        # already taken it to zero there, which is the whole reason the four
        # failure modes in `forcefield/lj.py`'s docstring cannot recur: they all
        # descended from a broken bond paying hundreds of eV for a pair at the
        # bond length, and the term is worth ~1e-3 eV there now.  Being the same
        # number for every diabatic state, it shifts every EVB diagonal equally
        # and leaves the eigenvectors alone, exactly as ZBL does.
        #
        # This is the term that supplies the intermolecular wall and the
        # dispersion.  ZBL's taper removed the first and the model never had the
        # second; see `production/density-300K/README.md`.
        en_lj, fr_lj, w_lj = self.lj_ff(pos, pbc, cell, self.topology.term_dict)
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
        }

        return results

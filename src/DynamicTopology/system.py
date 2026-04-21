"""System class definition"""

from DynamicTopology.utils import log_debug

import logging
from typing import Any

import numpy as np
from ase import Atoms

from DynamicTopology.core.reaction import Reaction
from DynamicTopology.core.reactionset import ReactionSet
from DynamicTopology.core.topology import Topology
from DynamicTopology.forcefield.coupling import EVBCoupling
from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.forcefield.acks2 import ACKS2

logger: logging.Logger = logging.getLogger(__name__)


class System:
    def __init__(
        self,
        atoms: Atoms,
        topology: Topology,
        reaction_set: ReactionSet,
        bimol_cutoff: float = 4.0,
    ):
        self.atoms: Atoms = atoms
        self.topology: Topology = topology
        self.reaction_set: ReactionSet = reaction_set
        self.bimol_cutoff: int | float = bimol_cutoff

        self.bonded_ff = QForce()
        self.nonbonded_ff = ACKS2()
        self.coupling = EVBCoupling()

    def __repr__(self) -> str:
        return f"System( {repr(self.atoms)}, {repr(self.topology)} )"

    def update(self, atoms: Atoms | None = None, topology: Topology | None = None):
        if atoms is not None:
            self.atoms = atoms
        if topology is not None:
            self.topology = topology

    def calculate_state(
        self, pos, pbc, cell, topology: Topology
    ) -> tuple[float, np.ndarray]:
        if len(topology.terms) == 0:
            terms = self.reaction_set.get_terms(topology)
            topology.set_terms(terms)
        energy, forces = self.bonded_ff(pos, pbc, cell, topology.term_dict)
        return energy, forces

    def calculate(self) -> dict[str, Any]:
        pos = self.atoms.positions
        pbc = self.atoms.pbc
        cell = self.atoms.cell

        # compute instantaneous reaction network for current topology
        network = self.reaction_set.get_network(self.topology, self.bimol_cutoff)
        log_debug(logger, "Computed instantaneous reacton network")

        # compute couplings
        for rxn in network.reactions():
            imol, jmol, rxn_data = rxn
            reaction: Reaction = rxn_data["reaction"]
            mapping: dict = rxn_data["mapping"]

            reactant = pos[list(mapping.values())]
            ensemble = np.stack([a.positions for a in reaction.atoms])
            energy, forces = self.coupling(
                reactant, pbc, cell, ensemble, reaction.term_dict
            )
            # save for later
            rxn_data["coupling_energy"] = energy
            tmp = np.zeros_like(pos)
            tmp[list(mapping.values()), :] = forces
            rxn_data["coupling_forces"] = tmp

        # enumerate states
        energy = 0.0
        forces = np.zeros_like(pos)

        log_debug(logger, "Local EVB subnets:")
        final_states = []
        for i, subnet in enumerate(network.states()):
            nstates = len(subnet)
            log_debug(logger, f"subnet {i}, nstates = {nstates}")

            # if no reactions
            if nstates == 1:
                rxn_data, topo = subnet[0]
                en, fr = self.calculate_state(pos, pbc, cell, topo)
                energy += en
                forces += fr
                continue

            # compute all states
            state: np.ndarray = np.empty(nstates, dtype=Topology)
            state_energy = np.zeros(nstates)
            state_forces = np.zeros((nstates,) + pos.shape)
            coupling_energy = np.zeros(nstates)
            coupling_forces = np.zeros((nstates,) + pos.shape)
            for j, (rxn_data, topo) in enumerate(subnet):
                # log_debug(logger, f"  state {j}, topo = {topo}")
                en, fr = self.calculate_state(pos, pbc, cell, topo)
                state_energy[j] = en
                state_forces[j] = fr
                if rxn_data:
                    coupling_energy[j] = rxn_data["coupling_energy"]
                    coupling_forces[j] = rxn_data["coupling_forces"]
                state[j] = topo

            # build evb hamiltonian
            ham = np.zeros((nstates, nstates))
            ham[np.diag_indices(nstates)] = state_energy
            ham[0, 1:] = coupling_energy[1:]
            ham[1:, 0] = coupling_energy[1:]

            # and matrix of gradients
            fham = np.zeros(
                (
                    nstates,
                    nstates,
                )
                + pos.shape
            )
            fham[np.diag_indices(nstates)] = state_forces
            fham[0, 1:] = coupling_forces[1:]
            fham[1:, 0] = coupling_forces[1:]

            # compute ground state energy
            eigval, eigvec = np.linalg.eigh(ham)
            statevec = eigvec[:, 0]

            energy_gs = np.einsum("i,ij,j->", statevec.T, ham, statevec)
            forces_gs = np.einsum("i,ijnd,j->nd", statevec.T, fham, statevec)

            energy += energy_gs
            forces += forces_gs

            # compute eigenvectors with numpy for state change
            statevecsq = statevec * statevec
            # log_debug(
            #     logger, f"statevec {statevecsq}, statevec sum {np.sum(statevecsq)}"
            # )
            # print(statevecsq)
            wi = np.where((statevecsq > 0.9))
            # choose the primary state for reaction subnet
            if len(state[wi]) == 0:
                final_states.append(state[0])
            else:
                final_states.append(state[wi][0])

        # compute topology independent nonbonded
        terms = self.reaction_set.get_terms(self.topology)
        self.topology.set_terms(terms)
        en_nb, fr_nb = self.nonbonded_ff(pos, pbc, cell, self.topology.term_dict)
        # energy += en_nb
        # forces += fr_nb

        # combine subnet topologies
        new_topo: Topology = Topology.from_molecules(final_states, remap=False)
        assert isinstance(new_topo, Topology)
        new_topo.set_atoms(self.atoms)

        results: dict[str, Any] = {
            "energy": energy,
            "forces": forces,
            "topology": new_topo,
        }

        return results

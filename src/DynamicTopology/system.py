"""System class definition"""

import logging
from typing import Any

import numpy as np
import torch as t
from ase import Atoms, units
from torch import Tensor

from DynamicTopology.core.reaction import Reaction
from DynamicTopology.core.reactionset import ReactionSet
from DynamicTopology.core.topology import Topology
from DynamicTopology.forcefield.coupling import EVBCoupling
from DynamicTopology.forcefield.qforce import QForce

logger: logging.Logger = logging.getLogger(name=__name__)


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
        self.coupling = EVBCoupling()

    def __repr__(self) -> str:
        return f"System( {repr(self.atoms)}, {repr(self.topology)} )"

    def update(self, atoms: Atoms | None = None, topology: Topology | None = None):
        if atoms is not None:
            self.atoms = atoms
        if topology is not None:
            self.topology = topology

    def convert_to_tensors(self, term_dict: dict):
        for term_type, param_dict in term_dict.items():
            term_dict[term_type]["atoms"] = t.tensor(param_dict["atoms"])
            for name, value in param_dict["kwargs"].items():
                term_dict[term_type]["kwargs"][name] = t.tensor(value)

    def calculate_state(
        self, pos: Tensor, pbc: Tensor, cell: Tensor, topology: Topology
    ) -> Tensor:
        if len(topology.terms) == 0:
            terms = self.reaction_set.get_terms(topology)
            topology.set_terms(terms)
        self.convert_to_tensors(topology.term_dict)
        en = self.bonded_ff(pos, pbc, cell, topology.term_dict) * units.kJ / units.mol
        return en

    def calculate(self) -> dict[str, Any]:
        # load the atoms as tensors
        pos = t.tensor(self.atoms.positions)
        pbc = t.tensor(self.atoms.pbc)
        cell = t.tensor(np.array(self.atoms.cell))
        pos.requires_grad_(True)

        # compute instantaneous reaction network for current topology
        network = self.reaction_set.get_network(self.topology, self.bimol_cutoff)

        # get parameters and compute couplings
        for rxn in network.reactions():
            imol, jmol, rxn_data = rxn
            reaction: Reaction = rxn_data["reaction"]
            mapping: dict = rxn_data["mapping"]
            self.convert_to_tensors(reaction.term_dict)
            reactant = pos[list(mapping.values())]
            ensemble = t.stack([t.tensor(a.positions) for a in reaction.atoms])
            e = (
                self.coupling(reactant, pbc, cell, ensemble, reaction.term_dict)
                * units.kJ
                / units.mol
            )
            rxn_data["coupling"] = e

        # enumerate states
        energy: Tensor = t.tensor(0.0)
        logger.info(msg="Local EVB networks:")
        states = network.states()
        final_states = []
        for i, subnet in enumerate(states):
            logger.info(msg=f"subnet {i}, nstates = {len(subnet)}")

            if len(subnet) == 1:
                rxn_data, topo = subnet[0]
                en = self.calculate_state(pos, pbc, cell, topo)
                energy += en
                continue

            # build evb hamiltonian
            S: np.ndarray = np.empty(len(subnet), dtype=Topology)
            ham = t.zeros((len(subnet), len(subnet)))
            for j, state in enumerate(subnet):
                rxn_data: dict = state[0]
                topo: Topology = state[1]
                ham[j, j] = self.calculate_state(pos, pbc, cell, topo)
                if rxn_data:
                    ham[0, j] = rxn_data["coupling"]
                    ham[j, 0] = ham[0, j]
                S[j] = topo

            # compute ground state energy using numerically stable eigvalsh
            energy += t.linalg.eigvalsh(ham)[0]

            # compute eigenvectors with numpy for state change
            val, vec = np.linalg.eigh(ham.detach().cpu().numpy())
            statew = vec[:, 0] * vec[:, 0]
            wi = np.where((statew > 0.9))

            # choose the primary state for reaction subnet
            if len(S[wi]) == 0:
                final_states.append(S[0])
            else:
                final_states.append(S[wi][0])

        # combine subnet topologies
        new_topo: Topology = Topology.from_molecules(final_states, remap=False)
        assert isinstance(new_topo, Topology)
        new_topo.set_atoms(self.atoms)

        forces: Tensor = -t.autograd.grad(energy, pos, t.ones_like(energy))[0]
        results: dict[str, Any] = {
            "energy": energy,
            "forces": forces,
            "topology": new_topo,
        }

        return results

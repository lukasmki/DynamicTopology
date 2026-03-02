from DynamicTopology.forcefield.qforce import QForce
from typing import Any
from DynamicTopology.core import Topology
from ase import Atoms

import torch as t
import numpy as np


class EVBSystem:
    def __init__(self, atoms: Atoms, states: list[Topology], hardness: float = 0.95):
        """High level API for running multi-state EVB calculations

        Args:
            atoms (Atoms): ASE atoms object
            states (list[Topology]): list of topologies corresponding to different bonding states
            hardness (float (0, 1]): coupling potential hardness, h, where

        :math:`H_{12} = \sqrt{(1+h) * H_1 * H_2}`
        """
        self.atoms: Atoms = atoms
        self.states: list[Topology] = states

        self.hardness: float = hardness
        self.bonded_ff = QForce()

    def update(self, atoms: Atoms | None = None):
        if atoms is not None:
            assert len(atoms) == len(self.atoms), "Atoms should be the same length"
            self.atoms = atoms

    def calculate(self) -> dict[str, Any]:
        """Run the calculation

        Returns:
            results (dict[str, Any]): energy, forces, statevec EVB state vector
        """
        pos = t.tensor(self.atoms.positions)
        pbc = t.tensor(self.atoms.pbc)
        cell = t.tensor(np.array(self.atoms.cell))
        pos.requires_grad_(True)

        # EVB hamiltonian
        ham = t.zeros((len(self.states), len(self.states)))

        # fill diagonal
        for i, istate in enumerate(self.states):
            if not istate.term_dict:
                # print(f"State {i} has no terms")
                continue
            ham[i, i] = self.bonded_ff(pos, pbc, cell, istate.term_dict)

        # fill off-diagonal
        for i, istate in enumerate(self.states):
            for j, jstate in enumerate(self.states[i + 1 :]):
                # compute state coupling
                ham[i, j] = t.sqrt((1 + self.hardness) * ham[i, i] * ham[j, j])
                ham[j, i] = ham[i, j]

        # compute energy/forces with torch
        energy: t.Tensor = t.linalg.eigvalsh(ham)[0]
        forces: t.Tensor = -t.autograd.grad(energy, pos, t.ones_like(energy))[0]

        # compute state vector with numpy
        val, vec = np.linalg.eigh(ham.detach().cpu().numpy())
        statevec = vec[:, 0] * vec[:, 0]

        results: dict[str, Any] = {
            "energy": energy,
            "forces": forces,
            "statevec": statevec,
        }
        return results

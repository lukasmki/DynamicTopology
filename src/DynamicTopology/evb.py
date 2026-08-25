from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.qforce import QForce
from typing import Any
from DynamicTopology.core import Topology
from ase import Atoms

import numpy as np


class EVBSystem:
    def __init__(self, atoms: Atoms, states: list[Topology], hardness: float = 0.95):
        r"""High level API for running multi-state EVB calculations

        Args:
            atoms (Atoms): ASE atoms object
            states (list[Topology]): list of topologies corresponding to different bonding states
            hardness (float (0, 1]): coupling potential hardness, h, where

        :math:`H_{12} = \sqrt{(1+h) * H_1 * H_2}`
        """
        self.atoms: Atoms = atoms
        self.states: list[Topology] = states

        self.hardness: float = hardness
        self.nonbonded_ff = ACKS2()
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
        pos = self.atoms.positions
        pbc = self.atoms.pbc
        cell = self.atoms.cell

        # EVB hamiltonian
        nstates = len(self.states)
        ham = np.zeros((nstates, nstates))
        state_forces = np.zeros((nstates,) + pos.shape)

        # Fill the diagonal with bonded energies only.  Electrostatics are
        # topology-independent, so they are the same number for every state;
        # adding them here would still change the result, because the coupling
        # sqrt((1+h)*H_ii*H_jj) is nonlinear in the diagonal and a common shift
        # does not pass through it.  They are added once, outside the
        # Hamiltonian, below -- the same way System.calculate does it.
        for i, istate in enumerate(self.states):
            if not istate.term_dict:
                continue
            en, fr = self.bonded_ff(pos, pbc, cell, istate.term_dict)
            ham[i, i] = en
            state_forces[i] = fr

        # fill off-diagonal
        fham = np.zeros((nstates, nstates) + pos.shape)
        fham[np.diag_indices(nstates)] = state_forces

        for i in range(nstates):
            for j in range(i + 1, nstates):
                hii, hjj = ham[i, i], ham[j, j]
                hprod = (1 + self.hardness) * hii * hjj
                hij = np.sqrt(np.abs(hprod))
                ham[i, j] = hij
                ham[j, i] = hij

                # gradient of H_ij = sqrt((1+h)*H_ii*H_jj) via chain rule
                # H_ij = sign * sqrt(|hprod|), so the chain rule carries the
                # sign of hprod onto *both* terms, not just the first.  Inert
                # while H_ii and H_jj share a sign, wrong when they do not.
                if hii != 0.0 and hjj != 0.0:
                    fij = np.sign(hprod) * (
                        (hij / (2 * hii)) * state_forces[i]
                        + (hij / (2 * hjj)) * state_forces[j]
                    )
                else:
                    fij = np.zeros_like(pos)
                fham[i, j] = fij
                fham[j, i] = fij

        # compute ground state energy and forces
        eigval, eigvec = np.linalg.eigh(ham)
        statevec = eigvec[:, 0]
        statevecsq = statevec * statevec

        energy = np.einsum("i,ij,j->", statevec, ham, statevec)
        forces = np.einsum("i,ijnd,j->nd", statevec, fham, statevec)

        # Topology-independent electrostatics, added once on top of the ground
        # state.  Every state carries the same `atom` terms, so state 0's
        # term_dict is representative; a state with no terms at all (a fully
        # dissociated topology) would carry none, hence the search.
        en_nb, fr_nb = 0.0, np.zeros_like(pos)
        for state in self.states:
            if state.term_dict:
                en_nb, fr_nb = self.nonbonded_ff(pos, pbc, cell, state.term_dict)
                break

        results: dict[str, Any] = {
            "energy": energy + en_nb,
            "forces": forces + fr_nb,
            "energy_bonded": energy,
            "energy_nonbonded": en_nb,
            "statevec": statevecsq,
        }
        return results

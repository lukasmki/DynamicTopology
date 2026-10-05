from DynamicTopology.forcefield.electrostatics import Electrostatics
from DynamicTopology.forcefield.lj import LennardJones, global_params
from DynamicTopology.forcefield.zbl import ZBL
from DynamicTopology.forcefield.qforce import QForce
from typing import Any
from DynamicTopology.core import Topology
from ase import Atoms

import numpy as np


def _lj_terms(indices: np.ndarray, sigma: np.ndarray, eps: np.ndarray) -> dict:
    """A `term_dict` holding only `lennardjones` rows, one per `indices`."""
    return {
        "lennardjones": {
            "atoms": np.asarray(indices, dtype=int)[:, None],
            "kwargs": {"sigma": np.asarray(sigma), "eps": np.asarray(eps)},
        }
    }


def _reference_lj(states: list[Topology], natoms: int) -> dict | None:
    """Each atom's 12-6 parameters from the first state that states them.

    The states of `EVBSystem` need not span the system: `ase.EVB` enumerates them
    per subnet, so an H2O + O2 box has H2O's states over three atoms and O2's
    over the other two.  The reference therefore has to be assembled atom by
    atom.  When the first state with terms covers every atom any state does,
    its own `term_dict` is the reference, term order and all.
    """
    with_terms = [s.term_dict for s in states if s.term_dict]
    if not with_terms or "lennardjones" not in with_terms[0]:
        return None
    first = with_terms[0]
    covered = set(first["lennardjones"]["atoms"][:, 0].tolist())
    sigma, eps = global_params(first, natoms)
    for term_dict in with_terms[1:]:
        params = term_dict.get("lennardjones")
        if params is None:
            continue
        for k, i in enumerate(params["atoms"][:, 0].tolist()):
            if i not in covered:
                covered.add(i)
                sigma[i] = params["kwargs"]["sigma"][k]
                eps[i] = params["kwargs"]["eps"][k]
    if covered == set(first["lennardjones"]["atoms"][:, 0].tolist()):
        return first
    indices = np.array(sorted(covered))
    return _lj_terms(indices, sigma[indices], eps[indices])


def _own_lj(term_dict: dict, reference: dict | None) -> dict | None:
    """The reference with this state's own parameters on its atoms, if they differ.

    `None` when they agree on every atom the state covers, which is every
    state while the templates carry per-element values.
    """
    params = term_dict.get("lennardjones")
    if reference is None or params is None:
        return None
    ref = reference["lennardjones"]
    indices = ref["atoms"][:, 0]
    sigma = np.array(ref["kwargs"]["sigma"], dtype=float)
    eps = np.array(ref["kwargs"]["eps"], dtype=float)
    position = {int(i): k for k, i in enumerate(indices)}
    for k, i in enumerate(params["atoms"][:, 0].tolist()):
        sigma[position[i]] = params["kwargs"]["sigma"][k]
        eps[position[i]] = params["kwargs"]["eps"][k]
    if np.array_equal(sigma, ref["kwargs"]["sigma"]) and np.array_equal(
        eps, ref["kwargs"]["eps"]
    ):
        return None
    return _lj_terms(indices, sigma, eps)


class EVBSystem:
    def __init__(
        self,
        atoms: Atoms,
        states: list[Topology],
        hardness: float = 0.95,
        reaction_set=None,
    ):
        r"""High level API for running multi-state EVB calculations

        Args:
            atoms (Atoms): ASE atoms object
            states (list[Topology]): list of topologies corresponding to different bonding states
            hardness (float (0, 1]): coupling potential hardness, h, where
            reaction_set (ReactionSet | None): the database the states came
                from.  Unused by the energy -- kept because callers pass it and
                because it identifies where a hand-built state list came from.
                It was once needed to strip a Pauli wall off dissociated states;
                the repulsion is now `ZBL`, which is small enough at bond
                lengths that nothing has to be stripped.  See
                `forcefield/zbl.py`.

        :math:`H_{12} = \sqrt{(1+h) * H_1 * H_2}`
        """
        self.atoms: Atoms = atoms
        self.states: list[Topology] = states
        self.reaction_set = reaction_set

        self.hardness: float = hardness
        self.electrostatics = Electrostatics()
        self.zbl_ff = ZBL()
        self.lj_ff = LennardJones()
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

        # `ZBL` and `LennardJones` are topology-independent, so they are added
        # once, outside the Hamiltonian, below: adding them here would still
        # change the result, because the coupling sqrt((1+h)*H_ii*H_jj) is
        # nonlinear in the diagonal and a common shift does not pass through it.
        # The Lennard-Jones sum used to have to go *on* the diagonal instead,
        # because the pairs it should not have counted were cancelled by
        # `exclusion` terms that were already there, and splitting a cancelling
        # pair across the two sides is ruinous here: it left a water molecule's
        # diagonal at -1833 eV against a bonded energy of -9.87, and a coupling
        # of sqrt(1.95 * 1833 * 1829) = 2560 eV followed.  Now that `lj.switch`
        # has taken the term to zero at bond lengths there are no exclusions to
        # be split from.
        #
        # That holds while every state carries the same per-atom parameters.  A
        # state whose templates give some atom a different sigma or epsilon has
        # a different 12-6, and only its *difference* from a reference state
        # goes on the diagonal -- so the common part still stays out of the
        # nonlinear coupling, and with agreeing templates every difference is
        # exactly zero and nothing changes.
        #
        # The electrostatics are not topology-independent under either term --
        # each state carries its own charges -- so they go on the diagonal.
        nonbonded = self.electrostatics.get()
        state_nb = np.zeros(nstates)
        state_lj = np.zeros(nstates)

        # The switched 12-6 at reference parameters, and each state charged its
        # difference from it -- only if its parameters differ at all.
        reference = _reference_lj(self.states, len(pos))
        en_lj, fr_lj = 0.0, np.zeros_like(pos)
        if reference is not None:
            en_lj, fr_lj, _ = self.lj_ff(pos, pbc, cell, reference)

        for i, istate in enumerate(self.states):
            if not istate.term_dict:
                continue
            # `EVBSystem` reports no stress -- it holds a fixed state list and
            # is the simple alternative to `System` -- so the virial is dropped.
            en, fr, _ = self.bonded_ff(pos, pbc, cell, istate.term_dict)
            en_q, fr_q, _ = nonbonded(pos, pbc, cell, istate.term_dict)
            en, fr = en + en_q, fr + fr_q
            state_nb[i] = en_q
            own = _own_lj(istate.term_dict, reference)
            if own is not None:
                en_s, fr_s, _ = self.lj_ff(pos, pbc, cell, own)
                state_lj[i] = en_s - en_lj
                en, fr = en + state_lj[i], fr + (fr_s - fr_lj)
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

        # The electrostatics are already on the diagonal; reported as the
        # ground state's share.  So is the 12-6's state-dependent part.
        en_nb = float(statevecsq @ state_nb)
        en_lj_states = float(statevecsq @ state_lj)

        # `EVBSystem` holds a fixed state list and reports no stress; the
        # virials are discarded here rather than threaded through.
        en_zbl, fr_zbl, _ = self.zbl_ff(pos, self.atoms.numbers, pbc, cell)

        # `energy` already contains the electrostatics and the 12-6's
        # differences from the reference state.
        energy_bonded = energy - en_nb - en_lj_states
        results: dict[str, Any] = {
            "energy": energy_bonded + en_nb + en_zbl + en_lj + en_lj_states,
            "forces": forces + fr_zbl + fr_lj,
            "energy_bonded": energy_bonded,
            "energy_nonbonded": en_nb,
            "energy_zbl": en_zbl,
            "energy_lj": en_lj + en_lj_states,
            "statevec": statevecsq,
        }
        return results

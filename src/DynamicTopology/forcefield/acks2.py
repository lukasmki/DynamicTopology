from ase import units
from scipy.special import erf
import numpy as np


class ACKS2:
    CCOUL = 14.4  # eV

    def __init__(self):
        self.Q = None
        self.pos_hash = None

    def compute_charges(self, rij, params, indices):
        natoms = len(indices)
        neqns = 2 * natoms + 2

        A = np.zeros((neqns, neqns))
        b = np.zeros(neqns)

        # interaction
        A[:natoms, :natoms] = erf(2 * rij) / (rij + np.finfo(np.float64).eps)

        # softness
        amp, decay = params["soft_amp"], params["soft_decay"]
        X0 = amp[:, None] * amp[None, :]
        tau = 0.5 * (decay[indices, None] + decay[None, indices])
        bsoft = X0 * np.exp(-rij / tau)
        bsoft[indices, indices] = 0.0
        bsoft[indices, indices] = -1 * np.sum(bsoft, axis=1)
        A[natoms : 2 * natoms, natoms : 2 * natoms] = bsoft

        # coupling
        A[:natoms, natoms : 2 * natoms] = -np.eye(natoms)
        A[natoms : 2 * natoms, :natoms] = -np.eye(natoms)

        # diagonal
        A[indices, indices] = 2.0 * params["eta"]
        b[:natoms] = -params["mu"]

        # Constraints
        # KS coeffs
        A[-1, indices + natoms] = -1
        A[indices + natoms, -1] = -1
        b[-1] = 0.0

        # total charge
        A[-2, indices] = -1
        A[indices, -2] = -1
        b[-2] = 0.0

        # solve
        x = np.linalg.solve(A, b)
        Q = x[:natoms]

        return Q

    def compute_coulomb(self, Q, rij, vecs, indices):
        qiqj = Q[:, None] * Q[None, :]
        qiqj[np.diag_indices(len(Q))] = 0.0
        r = rij + np.finfo(np.float64).eps
        kernel = erf(2 * rij) / r
        e = self.CCOUL * qiqj * kernel
        e[indices, indices] = 0.0  # zero the diagonal
        e_tot = 0.5 * np.sum(e) * units.eV

        # forces: F_i = CCOUL * sum_j qi*qj * (pos_i-pos_j)/rij^3
        # The energy 0.5-factor cancels because both e[i,j] and e[j,i] contribute to dE/d(pos_i)
        dkernel_dr = (4 / np.sqrt(np.pi)) * np.exp(-4 * rij**2) / r - kernel / r
        nij = vecs[indices] / (r[:, :, None])
        f = -nij * self.CCOUL * qiqj[:, :, None] * dkernel_dr[:, :, None]
        f_tot = np.sum(f, 1) * units.eV / units.Angstrom
        return e_tot, f_tot

    def __call__(self, pos, pbc, cell, term_dict: dict) -> tuple[float, np.ndarray]:
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell
        atom_params = term_dict.get("atom")
        if atom_params is None:
            raise KeyError("No atom parameters set")
        indices = atom_params["atoms"][:, 0]
        params = atom_params["kwargs"]
        rij = np.sqrt(np.sum(vecs[indices] * vecs[indices], -1))

        pos_hash = hash(pos.tobytes())
        if self.Q is None or pos_hash != self.pos_hash:
            self.Q = self.compute_charges(rij, params, indices)
            self.pos_hash = pos_hash

        e_tot, f_tot = self.compute_coulomb(self.Q, rij, vecs, indices)
        return e_tot, f_tot

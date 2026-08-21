from ase import units
from scipy.special import erf
import numpy as np


class ACKS2:
    """Charge-equilibration electrostatics.

    Two index spaces meet here and must not be confused.  The per-atom
    parameters (`mu`, `eta`, `soft_amp`, `soft_decay`) arrive in *term order* --
    the order the `atom` terms were collected -- while `pos` and the returned
    forces are in *global* atom order.  `indices[k]` is the global index of the
    k-th term.  Everything below is built in term order, and the forces are
    scattered back to global order only at the very end.

    Getting this wrong is silent: term order coincides with global order
    whenever the atom terms happen to be collected in index order, which is the
    common case, so the error only appears once a molecule is matched onto the
    live system in a different order.
    """

    CCOUL = 14.4  # eV

    def __init__(self):
        self.Q = None
        self.state_hash = None

    def compute_charges(self, rij, params):
        """Solve the ACKS2 linear system.  All arguments are in term order."""
        natoms = rij.shape[0]
        neqns = 2 * natoms + 2
        diag = np.diag_indices(natoms)
        atom = np.arange(natoms)

        A = np.zeros((neqns, neqns))
        b = np.zeros(neqns)

        # interaction
        A[:natoms, :natoms] = erf(2 * rij) / (rij + np.finfo(np.float64).eps)

        # softness
        amp, decay = params["soft_amp"], params["soft_decay"]
        X0 = amp[:, None] * amp[None, :]
        tau = 0.5 * (decay[:, None] + decay[None, :])
        bsoft = X0 * np.exp(-rij / tau)
        bsoft[diag] = 0.0
        bsoft[diag] = -1 * np.sum(bsoft, axis=1)
        A[natoms : 2 * natoms, natoms : 2 * natoms] = bsoft

        # coupling
        A[:natoms, natoms : 2 * natoms] = -np.eye(natoms)
        A[natoms : 2 * natoms, :natoms] = -np.eye(natoms)

        # diagonal
        A[atom, atom] = 2.0 * params["eta"]
        b[:natoms] = -params["mu"]

        # Constraints
        # KS coeffs
        A[-1, atom + natoms] = -1
        A[atom + natoms, -1] = -1
        b[-1] = 0.0

        # total charge
        A[-2, atom] = -1
        A[atom, -2] = -1
        b[-2] = 0.0

        # solve
        x = np.linalg.solve(A, b)
        Q = x[:natoms]

        return Q

    def compute_coulomb(self, Q, rij, vecs):
        """Coulomb energy and forces.  All arguments and results in term order.

        The charges are held fixed: there is no dQ/dr response term, so these
        forces are not the exact gradient of this energy whenever the charge
        response is significant.  See the xfails in test_energy_conservation.py.
        """
        diag = np.diag_indices(len(Q))
        qiqj = Q[:, None] * Q[None, :]
        qiqj[diag] = 0.0
        r = rij + np.finfo(np.float64).eps
        kernel = erf(2 * rij) / r
        e = self.CCOUL * qiqj * kernel
        e[diag] = 0.0  # zero the diagonal
        e_tot = 0.5 * np.sum(e) * units.eV

        # forces: F_i = CCOUL * sum_j qi*qj * (pos_i-pos_j)/rij^3
        # The energy 0.5-factor cancels because both e[i,j] and e[j,i] contribute to dE/d(pos_i)
        dkernel_dr = (4 / np.sqrt(np.pi)) * np.exp(-4 * rij**2) / r - kernel / r
        nij = vecs / (r[:, :, None])
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

        # Both axes in term order, so the parameter vectors line up with them.
        sub = np.ix_(indices, indices)
        vecs = vecs[sub]
        rij = np.sqrt(np.sum(vecs * vecs, -1))

        # Cache on the parameters as well as the geometry: the same positions
        # with a different set of atom terms is a different problem.
        state_hash = hash((pos.tobytes(), indices.tobytes()))
        if self.Q is None or state_hash != self.state_hash:
            self.Q = self.compute_charges(rij, params)
            self.state_hash = state_hash

        e_tot, f_tot = self.compute_coulomb(self.Q, rij, vecs)

        # scatter term-ordered forces back to global atom order
        forces = np.zeros_like(pos)
        forces[indices] = f_tot
        return e_tot, forces

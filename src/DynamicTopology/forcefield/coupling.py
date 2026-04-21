import numpy as np
from typing import Callable
from superpose3d import Superpose3D


class EVBCoupling:
    def __init__(self):
        pass

    def __call__(
        self,
        pos: np.ndarray,
        pbc: np.ndarray,
        cell: np.ndarray,
        ensemble: np.ndarray,
        term_dict: dict,
    ) -> tuple[float, np.ndarray]:
        if np.any(pbc):  # unwrap coordinates
            frac = pos @ np.linalg.inv(cell)
            diffs = np.diff(frac, axis=0)
            shift = diffs.round()
            frac[1:] = frac[0] + np.cumsum(diffs - shift, 0)
            pos = frac @ cell

        e, f = 0.0, np.zeros_like(pos)
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            de, df = fn(pos, ensemble, param_dict["atoms"], **param_dict["kwargs"])
            e += de
            f += df
        return e, f

    def compute_rmsd(self, pos, ensemble, atoms, A, a):
        rmsd = np.zeros(len(ensemble))
        drmsd = np.zeros((len(ensemble),) + pos.shape)
        for i in range(len(ensemble)):
            _, R, T, S = Superpose3D(ensemble[i], pos)
            ppos = S * np.einsum("ij,jk->ik", pos, R.T) + T[None, :]

            distsq = np.sum(np.square(ensemble[i] - ppos), -1)
            rmsd[i] = np.sqrt(np.mean(distsq))
            # d(rmsd)/d(pos) = (1/(n*rmsd)) * (ppos - ensemble) @ (S*R)
            drmsd[i] = (
                (1.0 / (rmsd[i] + np.finfo(np.float64).eps))
                / len(distsq)
                * (ppos - ensemble[i])
                @ (S * R)
            )
        e = A * np.exp(-a * rmsd**2)
        f = -1 * e * -a * 2 * rmsd * drmsd
        return np.mean(e), np.mean(f, 0)

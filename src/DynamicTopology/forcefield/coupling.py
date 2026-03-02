from typing import Callable

import torch as t
import torch.nn as nn
from torchpose3d import Superpose3D


class EVBCoupling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        pos: t.Tensor,
        pbc: t.Tensor,
        cell: t.Tensor,
        ensemble: t.Tensor,
        term_dict: dict,
    ):
        if t.any(pbc):  # unwrap coordinates
            frac = pos @ t.inverse(cell)
            diffs = t.diff(frac, dim=0)
            shift = diffs.round()
            frac[1:] = frac[0] + t.cumsum(diffs - shift, 0)
            pos = frac @ cell

        e = t.tensor(0.0)
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            e += fn(pos, ensemble, param_dict["atoms"], **param_dict["kwargs"])
        return e

    def compute_rmsd(self, pos, ensemble, atoms, A, a):
        r = t.zeros(ensemble.size(0))
        for i in range(ensemble.size(0)):
            results = Superpose3D(pos, ensemble[i])
            r[i] = results["RMSD"]
        e = A * t.exp(-a * r**2)
        return t.sum(e)

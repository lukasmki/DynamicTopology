from typing import Callable
import torch as t
import torch.nn as nn


class ACKS2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos: t.Tensor, pbc: t.Tensor, cell: t.Tensor, term_dict: dict):
        # compute all distance vectors
        # vecs[1, 0] - vector from atom_0 to atom_1
        vecs = pos[:, None, :] - pos[None, :, :]
        if t.any(pbc):
            F = vecs @ t.inverse(cell)
            vecs = vecs - (pbc * t.floor(F + 0.5)) @ cell

        # compute terms
        e = t.tensor(0.0)
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            e += fn(vecs, param_dict["atoms"], **param_dict["kwargs"])
        return e

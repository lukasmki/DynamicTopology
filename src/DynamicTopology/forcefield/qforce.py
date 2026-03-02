import torch as t
import torch.nn as nn

from typing import Callable


class QForce(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos: t.Tensor, pbc: t.Tensor, cell: t.Tensor, term_dict: dict):
        # compute all distance vectors
        # vecs[1, 0] - vector from atom_0 to atom_1
        vecs = pos[:, None, :] - pos[None, :, :]
        if t.any(pbc):
            F = vecs @ t.inverse(cell)
            vecs = vecs - (pbc * t.floor(F + 0.5)) @ cell

        # qforce units
        vecs /= 10  # angstrom to nm

        # compute terms
        e = t.tensor(0.0)
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            e += fn(vecs, param_dict["atoms"], **param_dict["kwargs"])
        return e

    def compute_bond(self, vecs, atoms, D, r0, k):
        """Morse potential"""
        v = vecs[atoms[:, 1], atoms[:, 0]]
        r = t.sqrt(t.sum(v * v, -1))
        dr = r - r0
        al = t.sqrt(k / (2 * D))
        e = D * (1 - t.exp(-al * dr)) ** 2 - D
        return t.sum(e)

    def compute_angle(self, vecs, atoms, theta0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        ra = t.sqrt(t.sum(va * va, -1, True))
        rb = t.sqrt(t.sum(vb * vb, -1, True))
        na = va / ra
        nb = vb / rb
        costheta = t.sum(na * nb, -1)
        e = k * t.square(costheta - t.cos(theta0))
        return t.sum(e)

    def compute_bondbond(self, vecs, atoms, r1_0, r2_0, k):
        v1 = vecs[atoms[:, 0], atoms[:, 1]]
        v2 = vecs[atoms[:, 2], atoms[:, 3]]
        r1 = t.sqrt(t.sum(v1 * v1, -1))
        r2 = t.sqrt(t.sum(v2 * v2, -1))
        e = t.clamp_min(k * (r1 - r1_0) * (r2 - r2_0), -10)
        return t.sum(e)

    def compute_bondangle(self, vecs, atoms, theta0, r0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        ra = t.sqrt(t.sum(va * va, -1, True))
        rb = t.sqrt(t.sum(vb * vb, -1, True))
        costheta = t.sum((va / ra) * (vb / rb), -1)
        vc = vecs[atoms[:, 3], atoms[:, 4]]
        rc = t.sqrt(t.sum(vc * vc, -1))
        e = t.clamp_min(k * (rc - r0) * (costheta - t.cos(theta0)), -20)
        return t.sum(e)

    def compute_angleangle(self, vecs, atoms, theta1_0, theta2_0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 4]]
        vd = vecs[atoms[:, 5], atoms[:, 4]]
        ra = t.sqrt(t.sum(va * va, -1, True))
        rb = t.sqrt(t.sum(vb * vb, -1, True))
        rc = t.sqrt(t.sum(vc * vc, -1, True))
        rd = t.sqrt(t.sum(vd * vd, -1, True))
        ct1 = t.sum((va / ra) * (vb / rb), -1)
        ct2 = t.sum((vc / rc) * (vd / rd), -1)
        e = k * (ct1 - t.cos(theta1_0)) * (ct2 - t.cos(theta2_0))
        return t.sum(e)

    def compute_periodicdihedral(self, vecs, atoms, phi0, n, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        rb = t.sqrt(t.sum(vb * vb, -1, True))
        axis = vb / rb
        u = va - t.sum(va * axis, -1, True) * axis
        v = vc - t.sum(vc * axis, -1, True) * axis
        phi = t.arctan2(t.sum(t.cross(axis, u, -1) * v, -1), t.sum(u * v, -1))
        e = k * t.pow(phi - phi0, n)
        return t.sum(e)

    # def compute_dihedralbond(self, vecs, atoms, r0, phi0, n, k):
    #     raise NotImplementedError()

    # def compute_dihedralangle(self, vecs, atoms, theta0, phi0, n, k):
    #     raise NotImplementedError()

    # def compute_dihedralangleangle(self, vecs, atoms, theta0_1, theta0_2, phi_0, n, k):
    #     raise NotImplementedError()

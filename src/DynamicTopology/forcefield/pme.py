"""Smooth particle-mesh Ewald for the reciprocal half of the ACKS2 kernel.

`ewald.Ewald` sums the reciprocal half of the kernel,

    K_rec_ij = (4 pi / V) sum_{k != 0} exp(-k^2 / 4 kappa^2) / k^2 cos(k . r_ij),

over every reciprocal vector for every atom, and `EwaldKernel.matrix` forms it
as a dense `N x N` matrix.  That is what the direct charge solve needs, and
what makes a large box cost `O(N^2)` memory and `O(N^2 N_k)` time.  An
iterative solve needs only `K_rec @ q`, which PME computes on a mesh in
`O(N p^3 + M log M)` (Essmann et al., J. Chem. Phys. 103, 8577 (1995)):

  - spread each charge onto the `p^3` mesh points around it with cardinal
    B-spline weights of order `p`,
  - FFT, multiply by `G(m) |b(m)|^2` -- the kernel's Fourier transform and the
    B-splines' Euler-exponential correction -- and transform back,
  - read each atom's potential off the mesh with the same weights.

Every operation is linear and the spreading and the gathering are each
other's transpose, so the operator is symmetric, as `K_rec` is.  It is an
approximation to `K_rec` -- one whose error falls with the mesh spacing and
the spline order -- not the same numbers: see `PMESetup` for how the mesh is
chosen and `tests/test_pme.py` for what it reproduces.

**Forces** are the derivative of `S = sum_c f_c^T K_rec f_c` over charge sets
`f_c`: `dS/dr_i = 2 f_ci grad phi_c(r_i)`, the gradient taken through the
spline weights.  **The virial** is the same expression `ewald.EwaldKernel`
uses, with each mesh wavevector's share of `S` in place of each reciprocal
vector's: the mesh deforms with the cell, so the spline weights do not move
under a strain and only `1/V` and the `k` vectors do.
"""

from __future__ import annotations

import numpy as np

# Even, because an odd order's Euler-spline factor vanishes at the Nyquist
# frequency.  8 is the order at which the mesh below reaches the direct sum's
# accuracy without a mesh much finer than the atoms.
ORDER: int = 8


def _splines(w: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """`M_p(w + j)` and its derivative for `j = 0..p-1`, `w` in `[0, 1)`.

    Built level by level from `M_2(x) = 1 - |x - 1|` with
    `M_n(x) = (x M_{n-1}(x) + (n - x) M_{n-1}(x - 1)) / (n - 1)`;
    `dM_p(x)/dx = M_{p-1}(x) - M_{p-1}(x - 1)`.
    """
    x = w[:, None] + np.arange(order)[None, :]  # (N, p)
    # level[s] holds M_n(x - s) for every shift s the recursion still needs.
    level = [np.clip(1.0 - np.abs(x - s - 1.0), 0.0, None) for s in range(order - 1)]
    previous = None
    for n in range(3, order + 1):
        previous = level
        level = [
            ((x - s) * previous[s] + (n - (x - s)) * previous[s + 1]) / (n - 1)
            for s in range(order - n + 1)
        ]
    values = level[0]
    derivative = previous[0] - previous[1]
    return values, derivative


def _euler(mesh: int, order: int) -> np.ndarray:
    """`|b(m)|^2` along one axis, `m = 0..mesh-1`."""
    m = np.arange(mesh)
    k = np.arange(order - 1)
    weights = _splines(np.zeros(1), order)[0][0, 1:]  # M_p(1..p-1)
    total = np.exp(2j * np.pi * np.outer(m, k) / mesh) @ weights
    return 1.0 / np.abs(total) ** 2


class PMESetup:
    """Cell-dependent half of PME: the mesh and its influence function.

    The mesh resolves the kernel's Fourier transform out to the wavevector at
    which `exp(-k^2 / 4 kappa^2)` falls to `accuracy`, `k_max = 2 kappa
    sqrt(-ln accuracy)`, with `oversampling` points per shortest wavelength
    there; at order 8 and 1.5x that reproduces the direct reciprocal sum to
    about `1e-6` of its size on the water boxes.
    """

    def __init__(self, cell, kappa: float, accuracy: float, oversampling: float = 1.5):
        self.cell = np.asarray(cell, dtype=float)
        self.inv_cell = np.linalg.inv(self.cell)
        self.volume = abs(np.linalg.det(self.cell))
        self.kappa = kappa
        self.order = ORDER
        recip = 2 * np.pi * self.inv_cell.T  # rows b_a
        blen = np.linalg.norm(recip, axis=1)
        kmax = 2 * kappa * np.sqrt(-np.log(accuracy))
        mesh = np.ceil(oversampling * 2 * kmax / blen).astype(int)
        mesh = np.maximum(mesh, 2 * self.order)
        self.mesh = tuple(int(m + (m % 2)) for m in mesh)  # even, for rfft

        K1, K2, K3 = self.mesh
        m1 = np.fft.fftfreq(K1, 1.0 / K1)
        m2 = np.fft.fftfreq(K2, 1.0 / K2)
        m3 = np.arange(K3 // 2 + 1)
        M = np.stack(np.meshgrid(m1, m2, m3, indexing="ij"), axis=-1)
        k = M @ recip
        k2 = np.sum(k * k, axis=-1)
        k2[0, 0, 0] = 1.0
        G = (4 * np.pi / self.volume) * np.exp(-k2 / (4 * kappa**2)) / k2
        G[0, 0, 0] = 0.0
        B = (
            _euler(K1, self.order)[:, None, None]
            * _euler(K2, self.order)[None, :, None]
            * _euler(K3, self.order)[None, None, : K3 // 2 + 1]
        )
        self.influence = G * B
        self.G = G
        self.k = k
        self.k2 = k2
        # Each half-spectrum point stands for itself and its mirror, except
        # the planes that are their own mirror.
        multiplicity = np.full(K3 // 2 + 1, 2.0)
        multiplicity[0] = 1.0
        if K3 % 2 == 0:
            multiplicity[-1] = 1.0
        self.multiplicity = multiplicity[None, None, :]

    def bind(self, pos: np.ndarray) -> "PME":
        return PME(self, pos)


class PME:
    """PME at one geometry: the spline weights of every atom."""

    def __init__(self, setup: PMESetup, pos: np.ndarray):
        self.setup = setup
        p = setup.order
        mesh = np.array(setup.mesh)
        u = (pos @ setup.inv_cell) * mesh  # (N, 3) in mesh units
        base = np.floor(u).astype(int)
        w = u - base
        self.n = len(pos)
        self.theta, self.dtheta, self.index = [], [], []
        for a in range(3):
            values, derivative = _splines(w[:, a], p)
            self.theta.append(values)
            self.dtheta.append(derivative)
            self.index.append((base[:, a, None] - np.arange(p)[None, :]) % mesh[a])
        # Flat mesh index and weight of each atom's p^3 points, (N, p^3).
        K2, K3 = setup.mesh[1], setup.mesh[2]
        ix, iy, iz = self.index
        self.flat = (
            ix[:, :, None, None] * (K2 * K3) + iy[:, None, :, None] * K3 + iz[:, None, None, :]
        ).reshape(self.n, -1)
        tx, ty, tz = self.theta
        self.weight = (tx[:, :, None, None] * ty[:, None, :, None] * tz[:, None, None, :]).reshape(
            self.n, -1
        )

    def _spread(self, q: np.ndarray) -> np.ndarray:
        size = int(np.prod(self.setup.mesh))
        grid = np.bincount(
            self.flat.ravel(), weights=(self.weight * q[:, None]).ravel(), minlength=size
        )
        return grid.reshape(self.setup.mesh)

    def _convolve(self, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`theta = N_mesh * ifft(influence * fft(grid))`, and `fft(grid)`."""
        transformed = np.fft.rfftn(grid)
        theta = np.fft.irfftn(transformed * self.setup.influence, s=self.setup.mesh)
        return theta * np.prod(self.setup.mesh), transformed

    def potential(self, Q: np.ndarray) -> np.ndarray:
        """`K_rec @ Q`, for `Q` a vector or `(N, r)` columns."""
        single = Q.ndim == 1
        Q = Q[:, None] if single else Q
        out = np.empty_like(Q, dtype=float)
        for c in range(Q.shape[1]):
            theta, _ = self._convolve(self._spread(Q[:, c]))
            out[:, c] = np.sum(theta.ravel()[self.flat] * self.weight, axis=1)
        return out[:, 0] if single else out

    def contract(self, factor: np.ndarray, signs=None):
        """`S = sum_c s_c f_c^T K_rec f_c`, `dS/dr_i` and `dS/de_ab`."""
        setup = self.setup
        mesh = np.array(setup.mesh)
        signs = np.ones(factor.shape[1]) if signs is None else np.asarray(signs)
        total = 0.0
        dS_dr = np.zeros((self.n, 3))
        dS_de = np.zeros((3, 3))
        tx, ty, tz = self.theta
        dx, dy, dz = self.dtheta
        p = setup.order
        for c in range(factor.shape[1]):
            f = factor[:, c]
            theta, transformed = self._convolve(self._spread(f))
            local = theta.ravel()[self.flat].reshape(self.n, p, p, p)
            # d phi / d u_a through each axis' spline derivative.
            gx = np.einsum("nijk,ni,nj,nk->n", local, dx, ty, tz)
            gy = np.einsum("nijk,ni,nj,nk->n", local, tx, dy, tz)
            gz = np.einsum("nijk,ni,nj,nk->n", local, tx, ty, dz)
            du = np.stack([gx, gy, gz], axis=1) * mesh  # per unit fractional coordinate
            grad = du @ setup.inv_cell.T  # d phi / d r
            dS_dr += signs[c] * 2.0 * f[:, None] * grad
            Sm = setup.multiplicity * setup.influence * np.abs(transformed) ** 2
            S = float(np.sum(Sm))
            total += signs[c] * S
            coeff = 2.0 * Sm * (1.0 / (4 * setup.kappa**2) + 1.0 / setup.k2)
            coeff[0, 0, 0] = 0.0
            dS_de += signs[c] * (
                np.einsum("xyz,xyza,xyzb->ab", coeff, setup.k, setup.k) - np.eye(3) * S
            )
        return total, dS_dr, dS_de

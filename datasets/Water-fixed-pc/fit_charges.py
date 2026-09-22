#!/usr/bin/env python3
"""Fit this dataset's fixed point charges and write them into `molecules/*.jsonl`.

Each template's charges are a least-squares fit to its own electrostatic
potential (Merz-Kollman), at the dataset's level of theory and at the template's
own geometry, with the total charge constrained to the species' formal charge:

    H2O 0,  H3O+ +1,  OH- -1,  H 0,  O 0.

The potential is sampled on four shells at 1.4, 1.6, 1.8 and 2.0 times each
atom's Merz-Kollman radius (H 1.2 A, O 1.4 A), at 5 points per A^2, keeping only
points outside every other atom's shell.  Symmetry-equivalent hydrogens are
averaged afterwards, which moves them by less than 1e-4 e.

These are gas-phase charges.  Every liquid fixed-charge water model carries a
larger dipole than this -- SPC/E 2.35 D against 1.85 D in the gas -- to stand in
for the polarization this model does not have; see the dataset README.

    uv run python datasets/Water-fixed-pc/fit_charges.py            # print only
    uv run python datasets/Water-fixed-pc/fit_charges.py --write    # and write

`--write` replaces any `charge` terms in each `.jsonl` and drops its ACKS2 `atom`
terms, which nothing reads under `electrostatics = "pointcharge"`.
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase import io
from pyscf import dft, gto

ROOT = Path(__file__).parent.resolve()
XC = "wb97x_v"
BASIS = "aug-cc-pvtz"
BOHR = 0.52917721092  # Angstrom

# (template, formal charge).  Closed-shell ions, as `make_water.py` builds them.
SPECIES = [("h2o", 0), ("h3o", 1), ("h1o", -1)]
FREE_ATOMS = ["h", "o"]

MK_RADII = {"H": 1.2, "O": 1.4}  # Angstrom
SHELLS = (1.4, 1.6, 1.8, 2.0)
DENSITY = 5.0  # points per A^2


def sphere(n: int) -> np.ndarray:
    """`n` near-uniform unit vectors on a Fibonacci spiral."""
    k = np.arange(n) + 0.5
    z = 1.0 - 2.0 * k / n
    phi = np.pi * (1.0 + 5**0.5) * k
    rho = np.sqrt(1.0 - z * z)
    return np.stack([rho * np.cos(phi), rho * np.sin(phi), z], axis=1)


def mk_points(symbols: list[str], positions: np.ndarray) -> np.ndarray:
    """Merz-Kollman sampling points, in Angstrom."""
    radii = np.array([MK_RADII[s] for s in symbols])
    points = []
    for scale in SHELLS:
        for centre, radius in zip(positions, radii):
            r = scale * radius
            shell = centre + r * sphere(int(DENSITY * 4 * np.pi * r * r))
            d = np.linalg.norm(shell[:, None, :] - positions[None], axis=-1)
            points.append(shell[np.all(d >= scale * radii[None] - 1e-9, axis=1)])
    return np.concatenate(points)


def esp(mol: gto.Mole, dm: np.ndarray, points_bohr: np.ndarray) -> np.ndarray:
    """Total electrostatic potential at `points_bohr`, in Hartree/e."""
    coords = mol.atom_coords()
    d = np.linalg.norm(points_bohr[:, None] - coords[None], axis=-1)
    nuclear = (mol.atom_charges()[None] / d).sum(axis=1)
    electronic = np.empty(len(points_bohr))
    for start in range(0, len(points_bohr), 256):
        chunk = points_bohr[start : start + 256]
        ints = mol.intor("int1e_grids", grids=chunk)
        electronic[start : start + 256] = np.einsum("gij,ij->g", ints, dm)
    return nuclear - electronic


def fit(symbols, positions, V, points, total: int) -> np.ndarray:
    """Least-squares charges reproducing `V` on `points`, summing to `total`."""
    inv = 1.0 / np.linalg.norm(points[:, None] - positions[None], axis=-1)
    n = len(symbols)
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = inv.T @ inv
    A[:n, n] = A[n, :n] = 1.0
    b = np.concatenate([inv.T @ V, [total]])
    return np.linalg.solve(A, b)[:n]


def equivalent_average(symbols: list[str], q: np.ndarray) -> np.ndarray:
    """Every template here has one heavy atom, so all its hydrogens are equivalent."""
    q = q.copy()
    hydrogens = [i for i, s in enumerate(symbols) if s == "H"]
    q[hydrogens] = q[hydrogens].mean()
    return q


def charges_for(name: str, total: int) -> tuple[np.ndarray, list[str], float]:
    atoms = io.read(ROOT / "molecules" / f"{name}.xyz")
    symbols = atoms.get_chemical_symbols()
    mol = gto.M(
        atom=[(s, tuple(p)) for s, p in zip(symbols, atoms.positions)],
        unit="Angstrom",
        basis=BASIS,
        charge=total,
        spin=0,
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.nlc = "VV10"
    mf.grids.level = 4
    mf.nlcgrids.level = 1
    mf.conv_tol = 1e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {name}")
    dm = mf.make_rdm1()

    points = mk_points(symbols, atoms.positions)
    V = esp(mol, dm, points / BOHR)
    q = fit(symbols, atoms.positions / BOHR, V, points / BOHR, total)
    q = equivalent_average(symbols, q)

    inv = 1.0 / np.linalg.norm(
        points[:, None] / BOHR - atoms.positions[None] / BOHR, axis=-1
    )
    rrms = np.sqrt(np.mean((inv @ q - V) ** 2) / np.mean(V**2))
    return q, symbols, float(rrms)


def write_charges(name: str, q: np.ndarray) -> None:
    path = ROOT / "molecules" / f"{name}.jsonl"
    terms = [json.loads(line) for line in path.read_text().splitlines() if line]
    kept = [t for t in terms if t["type"] not in ("atom", "charge")]
    charges = [
        {"type": "charge", "atoms": {"p0": i}, "kwargs": {"q": float(qi)}}
        for i, qi in enumerate(q)
    ]
    path.write_text("\n".join(json.dumps(t) for t in charges + kept) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    results = {}
    for name, total in SPECIES:
        q, symbols, rrms = charges_for(name, total)
        results[name] = q
        atoms = io.read(ROOT / "molecules" / f"{name}.xyz")
        line = "  ".join(f"{s} {qi:+.4f}" for s, qi in zip(symbols, q))
        extra = ""
        if total == 0:
            dipole = np.linalg.norm(q @ atoms.positions) / 0.20819434  # e*A -> D
            extra = f"  dipole {dipole:.3f} D"
        print(f"{name:4s} (q = {total:+d})  {line}   RRMS {rrms:.3f}{extra}")
    for name in FREE_ATOMS:
        results[name] = np.zeros(1)

    if args.write:
        for name, q in results.items():
            write_charges(name, q)


if __name__ == "__main__":
    main()

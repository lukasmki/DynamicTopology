#!/usr/bin/env python3
"""Build the Water dataset's geometries: `molecules/*.xyz` and `reactions/*.xyz`.

Geometries only.  The `energy=` field each frame needs is filled in afterwards
by `scripts/compute.py`, and the `.jsonl` terms by `scripts/fit.py`; see the
header of `datasets/Water/README.md` for the full recipe.

Protonation states
------------------
A template is keyed by the Weisfeiler-Lehman hash over atomic numbers, so the
force field cannot tell OH- from the OH radical, or H3O+ from the (unbound)
H3O radical.  This is a bulk-water set, where an OH is always hydroxide and an
H3O is always hydronium, so both templates are built as the **ions**: `h1o` is
OH- and `h3o` is H3O+.  That is what makes the autoionization energetics come
out right, and it is why `h1o` here is not HCombustion's `mol_03`.

`h`, `o` and `h2o` are the neutral species and are byte-identical to
HCombustion's `mol_08`, `mol_07` and `mol_04`.

Electron bookkeeping
--------------------
Every channel below conserves electrons across the templates it connects
(H3O+ 10 + H2O 10 = H2O 10 + H3O+ 10; OH- 10 + H2O 10 = H2O 10 + OH-;
2 x H2O 20 = H3O+ 10 + OH- 10), so computing each species at its own charge
against *neutral* free atoms leaves the atomic references cancelling exactly.
No further correction is needed, and `compute.py -c` may be run per file.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms, io

from pathlib import Path

ROOT = Path(__file__).parent.resolve()
METHOD = "wb97x_v/cc-pvtz"

# Equilibrium bond lengths (Angstrom) and angles (degrees).
R_OH_WATER = 0.9584
R_OH_HYDRONIUM = 0.976
R_OH_HYDROXIDE = 0.964
ANG_WATER = 104.45
ANG_HYDRONIUM = 111.8


def _frame(u: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal frame with `u` as its polar axis."""
    u = u / np.linalg.norm(u)
    seed = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(seed, u)) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    e1 = seed - np.dot(seed, u) * u
    e1 /= np.linalg.norm(e1)
    return u, e1, np.cross(u, e1)


def branches(
    origin: np.ndarray,
    u: np.ndarray,
    alpha: float,
    azimuths: list[float],
    r: float,
) -> list[np.ndarray]:
    """Hydrogens at polar angle `alpha` from `u`, at the given azimuths.

    `u` points from `origin` towards the hydrogen bond, so `alpha` is measured
    against the bond being made or broken and every geometry below is stated in
    the internal coordinates the reaction is actually described by.
    """
    u, e1, e2 = _frame(np.asarray(u, dtype=float))
    a = np.radians(alpha)
    return [
        origin
        + r
        * (
            np.cos(a) * u
            + np.sin(a) * (np.cos(np.radians(phi)) * e1 + np.sin(np.radians(phi)) * e2)
        )
        for phi in azimuths
    ]


# Two O-H bonds at `alpha` from the shared-proton axis close an H-O-H angle of
# `theta` when their azimuths differ by this much.  Inverting the spherical law
# of cosines here keeps the pyramidal H3O+ at its real 111.8 degrees instead of
# the 136 that a naive +/-90 split would give.
def azimuth_split(alpha: float, theta: float) -> float:
    a, t = np.radians(alpha), np.radians(theta)
    cos_dphi = (np.cos(t) - np.cos(a) ** 2) / np.sin(a) ** 2
    return float(np.degrees(np.arccos(np.clip(cos_dphi, -1.0, 1.0))))


HYDRONIUM_SPLIT = azimuth_split(ANG_HYDRONIUM, ANG_HYDRONIUM)

# An acceptor water straddles the hydrogen bond: its two O-H sit at +/-90
# degrees of azimuth, which closes `theta` only at a polar angle of
# `180 - theta/2`.  Anything less than 90 leans the hydrogens back towards the
# proton they are supposed to be pointing away from.
ACCEPTOR_ALPHA = 180.0 - ANG_WATER / 2


def write(path: Path, frames: list[Atoms]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    for atoms in frames:
        atoms.set_initial_charges(np.zeros(len(atoms)))
        atoms.info.setdefault("method", METHOD)
    io.write(path, frames, format="extxyz")
    print(f"wrote {path.relative_to(ROOT)}  ({len(frames)} frame(s))")


def make(
    symbols: str,
    positions: list,
    bonds: list[tuple[int, int]],
    charge: int,
    spin: int,
) -> Atoms:
    atoms = Atoms(symbols, positions=np.asarray(positions, dtype=float))
    atoms.info.update(
        connectivity=[[i, j, None] for i, j in bonds], charge=charge, spin=spin
    )
    return atoms


# --------------------------------------------------------------------------
# Molecules
# --------------------------------------------------------------------------


def molecules() -> None:
    # Free atoms.  Atomization energy is 0 by construction, and `compute.py`
    # writes exactly that, so these carry no bonded terms at all.
    write(ROOT / "molecules/h.xyz", [make("H", [[0, 0, 0]], [], 0, 1)])
    write(ROOT / "molecules/o.xyz", [make("O", [[0, 0, 0]], [], 0, 2)])

    # OH-, C_inf_v.  Ordered O first to match `h2o`/`h3o`.
    write(
        ROOT / "molecules/h1o.xyz",
        [make("OH", [[0, 0, 0], [0, 0, R_OH_HYDROXIDE]], [(0, 1)], -1, 0)],
    )

    # H2O, C2v.  Identical geometry to HCombustion's mol_04.
    half = np.radians(ANG_WATER / 2)
    write(
        ROOT / "molecules/h2o.xyz",
        [
            make(
                "OH2",
                [
                    [0, 0, 0],
                    [R_OH_WATER * np.sin(half), 0, R_OH_WATER * np.cos(half)],
                    [-R_OH_WATER * np.sin(half), 0, R_OH_WATER * np.cos(half)],
                ],
                [(0, 1), (0, 2)],
                0,
                0,
            )
        ],
    )

    # H3O+, C3v pyramidal.  Three O-H at the polar angle that closes 111.8
    # degrees between every pair.
    beta = np.degrees(
        np.arccos(np.sqrt((np.cos(np.radians(ANG_HYDRONIUM)) + 0.5) / 1.5))
    )
    origin = np.zeros(3)
    hs = branches(origin, [0, 0, 1], beta, [0.0, 120.0, 240.0], R_OH_HYDRONIUM)
    write(
        ROOT / "molecules/h3o.xyz",
        [make("OH3", [origin, *hs], [(0, 1), (0, 2), (0, 3)], 1, 0)],
    )


# --------------------------------------------------------------------------
# Reactions
# --------------------------------------------------------------------------
# Every reaction is laid out with the two oxygens on the x axis and the
# transferring proton `Hs` on that axis between them, so a frame is fully
# specified by the O-O distance, where `Hs` sits along it, and the polar angles
# of the spectator hydrogens.  Atom order is fixed across all three frames of a
# reaction, as `Reaction.from_atoms` and the RMSD coupling both require.


def dimer(
    d_oo: float,
    d_ahs: float,
    a_donor: list[tuple[float, list[float]]],
    a_acceptor: list[tuple[float, list[float]]],
    r_donor: float,
    r_acceptor: float,
) -> list[np.ndarray]:
    """`[Oa, Hs, *donor spectators, Ob, *acceptor spectators]`.

    `a_donor` / `a_acceptor` are `(polar angle, azimuths)` pairs measured from
    each oxygen's direction towards `Hs`.
    """
    oa = np.zeros(3)
    ob = np.array([d_oo, 0.0, 0.0])
    hs = np.array([d_ahs, 0.0, 0.0])
    donor = [
        p
        for alpha, azimuths in a_donor
        for p in branches(oa, [1, 0, 0], alpha, azimuths, r_donor)
    ]
    acceptor = [
        p
        for alpha, azimuths in a_acceptor
        for p in branches(ob, [-1, 0, 0], alpha, azimuths, r_acceptor)
    ]
    return [oa, hs, *donor, ob, *acceptor]


def grotthuss() -> None:
    """H3O+ + H2O -> H2O + H3O+ (Zundel proton hop).

    Atoms: O0, Hs1, Ha2, Ha3, O4, Hb5, Hb6.  Degenerate, so the product frame
    is the reactant's mirror image about the O-O midpoint.
    """
    pyramidal = [(ANG_HYDRONIUM, [HYDRONIUM_SPLIT / 2, -HYDRONIUM_SPLIT / 2])]
    # The acceptor water presents its lone pair to Hs.
    lone_pair = [(ACCEPTOR_ALPHA, [90.0, 270.0])]

    # The reactant is held at 2.75 A rather than at the 2.4 A where the
    # symmetric Zundel is the gas-phase global minimum: at contact there is no
    # barrier at all, so a reactant frame placed there is not a minimum, and the
    # width fit -- which asks where the coupling must switch off -- has nothing
    # to switch off against.  Compressing the O-O is part of the coordinate.
    reactant = dimer(2.75, 1.00, pyramidal, lone_pair, R_OH_HYDRONIUM, R_OH_WATER)
    # Zundel: O-O contracted, proton exactly midway, both oxygens equivalent.
    ts = dimer(2.45, 1.225, pyramidal, pyramidal, 0.97, 0.97)
    product = [p * np.array([-1.0, 1.0, 1.0]) + np.array([2.75, 0.0, 0.0]) for p in reactant]
    # Mirroring swaps the roles but not the indices: O0 keeps index 0 and is now
    # the acceptor.  Reorder so the *positions* follow the mirror while the
    # labels stay put.
    product = [
        product[4], product[1], product[5], product[6], product[0], product[2], product[3]
    ]

    write(
        ROOT / "reactions/h3o-h2o-transfer.xyz",
        [
            make("OHHHOHH", reactant, [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6)], 1, 0),
            make("OHHHOHH", ts, [(0, 1), (0, 2), (0, 3), (4, 5), (4, 6)], 1, 0),
            make("OHHHOHH", product, [(0, 2), (0, 3), (4, 1), (4, 5), (4, 6)], 1, 0),
        ],
    )


def hydroxide() -> None:
    """OH- + H2O -> H2O + OH- (hydroxide proton hop through H3O2-).

    Atoms: O0, Hs1, Ha2, O3, Hb4.  O0 starts as the water, O3 as the hydroxide.
    """
    water = [(ANG_WATER, [0.0])]
    # Hydroxide's own H sits near-perpendicular to the near-linear O-H...O.
    terminal = [(105.0, [0.0])]

    # As for the Zundel above, held wide of the symmetric H3O2- minimum.
    reactant = dimer(2.70, 0.99, water, terminal, R_OH_WATER, R_OH_HYDROXIDE)
    ts = dimer(2.45, 1.225, terminal, terminal, 0.965, 0.965)
    product = [p * np.array([-1.0, 1.0, 1.0]) + np.array([2.70, 0.0, 0.0]) for p in reactant]
    product = [product[3], product[1], product[4], product[0], product[2]]

    write(
        ROOT / "reactions/h2o-oh-transfer.xyz",
        [
            make("OHHOH", reactant, [(0, 1), (0, 2), (3, 4)], -1, 0),
            make("OHHOH", ts, [(0, 1), (0, 2), (3, 4)], -1, 0),
            make("OHHOH", product, [(0, 2), (3, 1), (3, 4)], -1, 0),
        ],
    )


def autoionization() -> None:
    """H2O + H2O -> H3O+ + OH- (contact ion pair).

    Atoms: O0, Hs1, Ha2, O3, Hb4, Hb5.  Not degenerate: the reactant is the
    water dimer at its 2.91 A equilibrium, the product a contact ion pair at
    2.45 A.  The barrier is late (Hammond), so the transition state is placed
    past the midpoint towards the product.
    """
    donor = [(ANG_WATER, [0.0])]
    acceptor_water = [(ACCEPTOR_ALPHA, [90.0, 270.0])]
    pyramidal = [(ANG_HYDRONIUM, [HYDRONIUM_SPLIT / 2, -HYDRONIUM_SPLIT / 2])]
    hydroxide_terminal = [(105.0, [0.0])]

    reactant = dimer(2.91, 0.975, donor, acceptor_water, R_OH_WATER, R_OH_WATER)
    ts = dimer(2.50, 1.21, hydroxide_terminal, pyramidal, 0.97, 0.98)
    product = dimer(2.45, 1.48, hydroxide_terminal, pyramidal, R_OH_HYDROXIDE, R_OH_HYDRONIUM)

    write(
        ROOT / "reactions/h2o-autoionization.xyz",
        [
            make("OHHOHH", reactant, [(0, 1), (0, 2), (3, 4), (3, 5)], 0, 0),
            make("OHHOHH", ts, [(0, 1), (0, 2), (3, 4), (3, 5)], 0, 0),
            make("OHHOHH", product, [(0, 2), (3, 1), (3, 4), (3, 5)], 0, 0),
        ],
    )


if __name__ == "__main__":
    molecules()
    grotthuss()
    hydroxide()
    autoionization()

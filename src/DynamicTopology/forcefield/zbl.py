"""Universal screened-nuclear repulsion, over every pair in the system.

**What this term is for.**  ACKS2 has no repulsive branch.  Its Coulomb kernel
`erf(2r)/r` is finite at contact rather than divergent -- it tends to 2.257, so
the charges saturate and there is no runaway in the solve itself -- but the
interaction it produces between two atoms of different electronegativity is a
smooth, monotone, ~4 eV attractive funnel all the way to zero separation:

    r (A)   2.0    1.5    1.2   0.96    0.6    0.2   0.05
    Q_H     0.110  0.212  0.259 0.286  0.317  0.347  0.353
    E (eV) -0.09  -0.43  -0.80 -1.21  -2.19  -3.71  -4.03

No minimum, no wall.  At 3000 K, kT is 0.26 eV.  A 200-atom H2/O2 box run
against ACKS2 alone reached 0.60 A intermolecular contacts with 55 pairs inside
1.2 A, reading -46.6 eV of nonbonded energy -- 0.85 eV per pair, which is this
table.  This is the term that opposes it.

**Why it is topology-independent, and why that is the whole design.**  The
obvious repulsion to reach for is the Lennard-Jones already in the templates,
excluded between bonded atoms the way any fixed-topology force field excludes
it.  That was tried four times.  It fails because q-force's 12-6 is enormous at
the separations reactive chemistry actually visits -- 504 eV at the H2 bond
length, 930 eV at O-H, 1348 eV at O-O -- so a diabatic state that has *broken* a
bond pays hundreds of eV for a pair that is merely close.  Every attempt to
strip that penalty made the repulsion differ between diabatic states, and each
one failed in its own way: stripping it from the parent as well let the box fuse
at 0.60 A; gating it on the coupling left a 19 eV plateau across rxn_16's
reaction path where a wall was half-removed; and transition states with close
non-bonded contacts came back with EVB amplitudes of -72 to -108 eV.

This term takes no topology at all.  Its only parameter is the atomic number,
which it reads from `numbers` rather than from a term list, so there is no
template, no remapping and no per-state path by which it could acquire a state
dependence.  That is deliberate and it is structural: a term identical across
every diabatic state adds the same constant to every EVB diagonal, and a common
shift of the diagonal moves `np.linalg.eigh`'s eigenvalue by exactly that
constant while leaving the eigenvectors untouched.  So this term *cannot*
produce a plateau, a spurious coupling amplitude, a pivot dependence, or a
discontinuity at the bimolecular cutoff -- not "does not", cannot.  Each of the
four previous failures lived in the degree of freedom this removes.

The same property means it cancels exactly out of every energy *difference*,
including the diabatic margins `fit/dissociation.py` scores channels on.  It
buys stability and it buys nothing at all for fittability; that work belongs to
the bonded fit.

**Form.**  The ZBL universal potential (Ziegler, Biersack and Littmark), which
is what reactive potentials -- ReaxFF, Tersoff/ZBL -- use for exactly this job.
Screened Coulomb between bare nuclei, fitted across the periodic table, with no
free parameters beyond Z.  Values it takes at the separations that matter here,
against the 12-6 it replaces:

    pair  r (A)  what it is                12-6      ZBL
    H-H   0.777  the H2 bond length      504 eV    2.0 eV
    O-H   0.960  the O-H bond length     930 eV    5.4 eV
    O-O   1.210  the O2 bond length     1348 eV   11.6 eV
    O-H   0.600  the observed fusion    2.6e5 eV  20.9 eV

It is applied to bonded pairs too, since it knows nothing about bonds.  Those
values are absorbed by the fitted Morse depths -- `fit/dissociation.py` solves
against `E_QForce + E_nonbonded`, and this term is part of `E_nonbonded` -- so a
template still reproduces its own reference atomization energy.  O2 is the
stress case: +11.6 eV and +36.6 eV/A at its own bond length.

**No cutoff and no taper.**  ZBL decays exponentially without help (O-O: 0.42 eV
at 2.5 A, 0.026 eV at 4 A, 0.001 eV at 6 A), and every cutoff this codebase has
added has cost more than it bought.  The consequence at van der Waals range is
worth stating: this is about 3x the 12-6's own repulsion at 2.5 A (0.42 against
0.14 eV for O-O), and the shallow dispersion well the Lennard-Jones provided --
-7 meV for O-O -- is gone.  That well is a fortieth of kT at 3000 K.

Angstrom and eV throughout, unlike `QForce` and `LennardJones`, because the ZBL
constants are stated in those units and converting them would put a unit slip
between this module and its own literature.
"""

import numpy as np


# Screening length prefactor, `0.8854 * a_0`, in Angstrom.
SCREENING_LENGTH: float = 0.46850

# Coulomb constant in eV*Angstrom, matching `ACKS2.CCOUL` to the digits ASE uses.
CCOUL: float = 14.399645

# The universal screening function `phi(x) = sum_k C[k] * exp(-B[k] * x)`.
PHI_C: tuple[float, ...] = (0.18175, 0.50986, 0.28022, 0.02817)
PHI_B: tuple[float, ...] = (3.19980, 0.94229, 0.40290, 0.20162)


def pair_potential(
    r: np.ndarray, z1: np.ndarray, z2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ZBL and its radial derivative, `(u, du/dr)`, in eV and eV/Angstrom.

    `r` must be strictly positive; callers holding a full distance matrix should
    substitute anything on the diagonal and zero the result there, as
    `ZBL.__call__` does.

    Not unit-agnostic, unlike `lj.pair_potential`: `SCREENING_LENGTH` is in
    Angstrom and `CCOUL` in eV*Angstrom, so `r` has to be in Angstrom and the
    result comes back in eV.  There is only one caller and it works in ASE
    units, so there is nothing for this to drift against.
    """
    a = SCREENING_LENGTH / (z1**0.23 + z2**0.23)
    x = r / a

    phi = np.zeros_like(r)
    dphi_dx = np.zeros_like(r)
    for c, b in zip(PHI_C, PHI_B):
        term = c * np.exp(-b * x)
        phi = phi + term
        dphi_dx = dphi_dx - b * term

    k = CCOUL * z1 * z2
    u = k * phi / r
    # d/dr [ k phi(r/a) / r ] = k ( phi'(x)/a / r  -  phi(x) / r^2 )
    du_dr = k * (dphi_dx / (a * r) - phi / r**2)
    return u, du_dr


class ZBL:
    """ZBL repulsion summed over every pair, with no exclusions.

    Deliberately *not* wired like `ACKS2` and `LennardJones`.  Those read their
    per-atom parameters out of `term_dict`, which means they carry the term-order
    versus global-order distinction and, more importantly, that a state could in
    principle hand them a different parameter set.  This one takes `numbers`
    straight from the `Atoms`, so the only thing it can be a function of is the
    geometry and the elements -- see the module docstring for why that matters
    more than the consistency.
    """

    def __call__(
        self,
        pos: np.ndarray,
        numbers: np.ndarray,
        pbc: np.ndarray,
        cell: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            f = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(f + 0.5)) @ cell

        rij = np.sqrt(np.sum(vecs * vecs, -1))
        diag = np.diag_indices(len(pos))
        r = rij.copy()
        r[diag] = 1.0  # excluded below; only keeps the division finite

        z = np.asarray(numbers, dtype=float)
        u, du_dr = pair_potential(r, z[:, None], z[None, :])
        u[diag] = 0.0
        du_dr[diag] = 0.0

        # The 0.5 cancels for the forces because both (i, j) and (j, i)
        # contribute to dE/d(pos_i) -- the convention `ACKS2.compute_coulomb`
        # and `LennardJones.__call__` both use.
        energy = 0.5 * float(np.sum(u))
        nij = vecs / r[:, :, None]
        forces = -np.sum(du_dr[:, :, None] * nij, axis=1)
        return energy, forces

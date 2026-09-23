import numpy as np
from ase import units
from typing import Callable

from DynamicTopology.forcefield.params import active

# `BOND_ASYMPTOTE` is now a field of `params.ForceFieldParams`,
# `bond_asymptote`.  It changes what fast-forces' `refine` solves against, so it
# belongs to the dataset fitted at it and not to whichever source tree is
# installed: a checkout whose asymptote had moved evaluated every template up to
# 73 meV high and said nothing.  The diabatic argument for lifting the
# dissociated limit off zero moved there with the numbers.
#
# `bond_asymptote` is in eV, like every parameter this class reads.  A bond's
# own `h` overrides it.

# Floors on the two cross terms that are products of two displacements and so
# unbounded below.  q-force states them as -10 and -20 kJ/mol; they are the same
# energies here, in eV.
CLIP_BONDBOND: float = -10.0 * units.kJ / units.mol
CLIP_BONDANGLE: float = -20.0 * units.kJ / units.mol


class QForce:
    """Bonded force field, in ASE units throughout: Angstrom, eV, radians.

    Parameters arrive in those units too -- a `.jsonl` stores them so, and
    `io/units.py` converts q-force's nm and kJ/mol once, when fast-forces
    imports its XML -- so nothing here rescales.

    `bond_form` selects the bond functional form:

      "morse"      D*(1 - exp(-a*dr))**2 - D,  a = sqrt(k/2D)
      "harmonic"   0.5*k*dr**2

    Morse is the default and is required for reactive work.  It is bounded
    above by its dissociation asymptote, so a product state whose newly formed
    bond is still several Angstrom long costs at most D rather than the
    unbounded 0.5*k*dr**2 that the harmonic form charges; and its well depth
    places bound and dissociated topologies on a physically ordered scale
    instead of leaving bond breaking free.  The harmonic form is retained for
    non-reactive use and for comparison.
    """

    def __init__(self, bond_form: str = "morse"):
        if bond_form not in ("morse", "harmonic"):
            raise ValueError(
                f"bond_form must be 'morse' or 'harmonic', got {bond_form!r}"
            )
        self.bond_form: str = bond_form

    def __call__(
        self, pos: np.ndarray, pbc: np.ndarray, cell: np.ndarray, term_dict: dict
    ) -> tuple[float, np.ndarray, np.ndarray]:
        # compute all distance vectors
        # vecs[1, 0] - vector from atom_0 to atom_1
        vecs = pos[:, None, :] - pos[None, :, :]
        if np.any(pbc):
            F = vecs @ np.linalg.inv(cell)
            vecs = vecs - (pbc * np.floor(F + 0.5)) @ cell

        # compute terms
        e = 0.0
        f = np.zeros_like(pos)
        w = np.zeros((3, 3))
        for term_type, param_dict in term_dict.items():
            fn: Callable | None = getattr(self, f"compute_{term_type}", None)
            if fn is None:
                continue
            de, df, dw = fn(vecs, param_dict["atoms"], **param_dict["kwargs"])
            e += de
            f += df
            w += dw
        return e, f, w

    def _accumulate_forces(self, f, atoms_col, grad):
        """
        Scatter force gradients into the global force array.

        grad shape: (n_terms, 3)
        atoms_col: integer column index into `atoms` selecting which atom receives `grad`.
        The sign convention follows F = -dE/dr, but since vecs[j,i] = pos_i - pos_j,
        dE/d(pos_i) contributions are passed in directly and negated for pos_j.
        """
        np.add.at(f, atoms_col, grad)

    @staticmethod
    def _virial(*pairs):
        """Sum of `v (x) dE/dv` over a term's internal displacement vectors.

        Every term here is a function of displacement vectors only, so under a
        homogeneous strain `v -> (I + e) v` and `dE/de_ab = sum v_a (dE/dv)_b`.
        Each `compute_*` already forms `dE/dv` on its way to scattering the
        forces -- the sign convention being that `dE/dv` is the gradient with
        respect to the vector as `vecs` defines it, so it is scattered as `-dE/dv`
        onto the atom the vector points *to*.  Passing the same arrays here costs
        one outer product per vector and needs no new derivatives.

        The result is an energy, in eV: `v` is in Angstrom and `dE/dv` in eV/A.
        """
        w = np.zeros((3, 3))
        for v, dE_dv in pairs:
            w += np.einsum("na,nb->ab", v, dE_dv)
        return w

    def compute_bond(self, vecs, atoms, D, r0, k, h=None):
        if self.bond_form == "morse":
            return self._bond_morse(vecs, atoms, D, r0, k, h)
        return self._bond_harmonic(vecs, atoms, D, r0, k)

    def _bond_morse(self, vecs, atoms, D, r0, k, h=None):
        """Morse with a per-bond asymptote on the stretched branch.

            Dw = D + h  (dr > 0),   Dw = D  (dr <= 0),   a = sqrt(k / 2Dw)
            E  = Dw * (1 - exp(-a*dr))**2 - D

        The `-D` offset puts the minimum at `-D`, so a topology's energy carries
        the depth of the bonds it contains and breaking a bond costs `+D` rather
        than nothing.  The *dissociated* limit is `h` above zero, not at it, and
        the well the exponential climbs is `D + h` deep.  `h` is in eV like `D`;
        omitted, it is `bond_asymptote` for every bond.

        **Why `h` is per bond.**  Plain Morse is exact at the minimum and at
        dissociation and has nothing left over in between: `D` is pinned by the
        atomization energy, `r0` by the geometry and `k` by the vibrational
        frequency, and the form comes out too deep at stretched geometries.  At
        fixed `k`, `Dw (1 - exp(-a dr))**2` increases monotonically in `Dw` at
        every `dr > 0`, towards the harmonic `k dr**2 / 2`, so raising `h` lifts
        the whole stretched branch while the curvature at `dr = 0` stays
        `2 Dw a**2 = k` -- no fitted frequency moves -- and the curve stays a
        Morse, monotone out to its limit whatever `h` is.  That is what
        fast-forces' `refine.fit_force_constants` fits.  It replaced a
        Hulburt-Hirschfelder shape term `c s**3 exp(-b s)`, which did the same
        job with two parameters per bond, needed a monotonicity bound
        `c <= c_max(b)` to keep the curve from turning over, and fitted ~40x
        slower to the same 19 of 19 HCombustion channels.

        **Stretched branch only.**  The asymptote is a statement about
        dissociation, so it has no business on the compressed side -- and it
        does not arrive there harmlessly.  A deeper well at fixed `k` is a
        *narrower* exponent, which softens compression: `exp(-a dr)` for
        `dr < 0` grows more slowly, so a molecule squeezed head-on deforms
        further.  Applied to both branches it cost 0.03 A of intermolecular
        closest approach in `test_collapse.py`, against a bar with 0.03 A of
        margin left.

        The join is C2, so the forces never see it: both branches are `-D` at
        `dr = 0` with zero slope, and the curvature there is `k` whatever `Dw`
        is.  Only the third derivative jumps.
        """
        if h is None:
            # Resolved here rather than in the signature so that a dataset
            # loaded after import gets its own asymptote rather than the one
            # that was current when this module was first imported.
            h = active().bond_asymptote
        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3)  vec from atom0->atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        dr = r - r0
        Dw = np.where(dr > 0.0, D + h, D)  # (n,)
        al = np.sqrt(k / (2 * Dw))  # (n,)  1/A
        exp_term = np.exp(-al * dr)  # (n,)
        e = Dw * (1 - exp_term) ** 2 - D
        # dE/dr  =  2*Dw*(1 - exp)*al*exp
        de_dr = 2 * Dw * (1 - exp_term) * al * exp_term  # (n,)

        e_tot = np.sum(e)
        # dr/dv = v/r,  v = pos_atom1 - pos_atom0
        dv = (de_dr / r)[:, None] * v  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)
        np.add.at(f, atoms[:, 1], -dv)
        return e_tot, f, self._virial((v, dv))

    def _bond_harmonic(self, vecs, atoms, D, r0, k):
        """Harmonic potential, E = 0.5*k*dr**2.  `D` is unused."""
        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3)  vec from atom0->atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        dr = r - r0
        e = 0.5 * k * dr * dr
        e_tot = np.sum(e)

        de_dr = k * dr  # (n,)
        dv = (de_dr / r)[:, None] * v  # (n, 3)  force direction

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)  # atom0:  v points away from atom1
        np.add.at(f, atoms[:, 1], -dv)  # atom1
        return e_tot, f, self._virial((v, dv))

    def compute_reference(self, vecs, atoms, E0):
        """Constant per-molecule reference energy (the EVB alpha shift).

        Geometry-independent, so it contributes no force.  It exists to put
        different bonding topologies on a common absolute energy scale: without
        it the diabatic energies are each measured from their own minimum and
        are not comparable, which makes every EVB eigenvalue meaningless.

        Set by ReactionSet at load time as the residual between the template's
        reference atomization energy and the depth its Morse bonds already
        supply, so it is small and Morse carries the physics.
        """
        # Geometry-independent, so it is strain-independent too: a constant
        # shift moves no atom and stores no stress.
        return np.sum(E0), np.zeros((vecs.shape[0], 3)), np.zeros((3, 3))

    def compute_exclusion(self, vecs, atoms, sigma, eps):
        """Cancels the global Lennard-Jones term between near neighbours.

        `ReactionSet.load` derives these terms for every template through
        `forcefield/exclusions.py`, one per pair within `exclusion_depth` bonds
        whose combined sigma and epsilon are nonzero, so this method is on
        every diabat's diagonal.

        `forcefield/lj.py` sums 12-6 over *every* pair in the system, including
        pairs that are bonded to each other, because that sum is the same for
        every diabatic state and can therefore be evaluated once outside the EVB.
        What is topology-dependent is which pairs should not have been counted,
        and that is the pairs within `params.exclusion_depth` bonds of each other --
        a per-molecule quantity, which is what makes it expressible as a term.

        The functional form must match `LennardJones` exactly, combining rule
        included, or an isolated template stops reproducing its own energy.
        `sigma` and `eps` are therefore the already-combined pair values, worked
        out once when the template is loaded rather than twice from different
        code -- and the form itself comes from `lj.pair_potential` for the same
        reason.  It was open-coded here once, and the copy silently stopped
        matching the moment `pair_potential` gained its short-range linear
        continuation: the two halves of the cancellation disagreed by 1609 eV on
        an H2 template.
        """
        from DynamicTopology.forcefield.lj import pair_potential

        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3)  vec from atom0->atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        # The same function of the same numbers as the other half of the
        # cancellation, `LennardJones.__call__`.
        u, du_dr = pair_potential(r, sigma, eps)
        e_tot = -np.sum(u)

        # e = -u, so de_dr = -du/dr
        de_dr = -du_dr
        dv = (de_dr / r)[:, None] * v  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)
        np.add.at(f, atoms[:, 1], -dv)
        return e_tot, f, self._virial((v, dv))

    def compute_zblexclusion(self, vecs, atoms, z1, z2):
        """Cancels the global ZBL repulsion between near neighbours.

        The counterpart of `compute_exclusion` for `forcefield/zbl.py`, and it
        exists for the same reason: that term is summed over every pair in the
        system so that it is the same number on every diabatic state, and what
        is topology-dependent is which pairs should not have been counted.

        This is the exclusion that matters most.  Tapered ZBL is 5.2-5.5 eV at
        an O-H bond length and falls to zero by 1.6 A, so leaving it inside a
        template makes the Morse depth cancel a 5 eV step rather than describe a
        bond.  H3O+ is where that cancellation failed -- fitted depth 5.76 eV
        against a 5.21 eV step, leaving the isolated cation's true minimum at
        1.60 A.  See `forcefield/exclusions.py`.
        """
        from DynamicTopology.forcefield.zbl import pair_potential

        v = vecs[atoms[:, 1], atoms[:, 0]]  # (n, 3), atom0 -> atom1
        r = np.sqrt(np.sum(v * v, -1))  # (n,)
        u, du_dr = pair_potential(r, z1, z2)  # eV, eV/Angstrom

        e_tot = -float(np.sum(u))
        de_dr = -du_dr
        dv = (de_dr / r)[:, None] * v  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos)
        np.add.at(f, atoms[:, 0], dv)
        np.add.at(f, atoms[:, 1], -dv)
        return e_tot, f, self._virial((v, dv))

    def compute_angle(self, vecs, atoms, theta0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]  # (n, 3)
        vb = vecs[atoms[:, 2], atoms[:, 1]]  # (n, 3)
        ra = np.sqrt(np.sum(va * va, -1, keepdims=True))  # (n,1)
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        na = va / ra  # unit vectors
        nb = vb / rb
        costheta = np.sum(na * nb, -1)  # (n,)
        e = 0.5 * k * np.square(costheta - np.cos(theta0))
        e_tot = np.sum(e)

        # dE/d(cos) = k*(cos - cos0)
        dE_dcos = k * (costheta - np.cos(theta0))  # (n,)

        # d(cos)/d(va) = (nb - cos*na) / |va|
        # d(cos)/d(vb) = (na - cos*nb) / |vb|
        costheta_k = costheta[:, None]
        dcos_dva = (nb - costheta_k * na) / ra  # (n, 3)
        dcos_dvb = (na - costheta_k * nb) / rb

        # chain rule: dE/d(va) = dE/d(cos) * dcos/d(va)
        dE_dva = dE_dcos[:, None] * dcos_dva  # (n, 3)
        dE_dvb = dE_dcos[:, None] * dcos_dvb

        # va = pos[atom0] - pos[atom1]  =>  dE/d(pos_a0)=+dE_dva, dE/d(pos_a1)=-dE_dva
        # vb = pos[atom2] - pos[atom1]  =>  dE/d(pos_a2)=+dE_dvb, dE/d(pos_a1)-= dE_dvb
        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dE_dva)
        np.add.at(f, atoms[:, 1], dE_dva + dE_dvb)
        np.add.at(f, atoms[:, 2], -dE_dvb)
        return e_tot, f, self._virial((va, dE_dva), (vb, dE_dvb))

    def compute_bondbond(self, vecs, atoms, r1_0, r2_0, k):
        v1 = vecs[atoms[:, 0], atoms[:, 1]]  # (n, 3)
        v2 = vecs[atoms[:, 2], atoms[:, 3]]
        r1 = np.sqrt(np.sum(v1 * v1, -1))  # (n,)
        r2 = np.sqrt(np.sum(v2 * v2, -1))
        raw = k * (r1 - r1_0) * (r2 - r2_0)
        e = np.clip(raw, CLIP_BONDBOND, None)
        e_tot = np.sum(e)

        # gradient only where not clipped
        mask = (raw > CLIP_BONDBOND).astype(float)[:, None]
        # dE/d(r1) = k*(r2-r2_0),  dE/d(r2) = k*(r1-r1_0)
        dE_dr1 = (k * (r2 - r2_0))[:, None] * mask  # (n,1)
        dE_dr2 = (k * (r1 - r1_0))[:, None] * mask

        # dr/dv = v/r
        dv1 = dE_dr1 * v1 / r1[:, None]
        dv2 = dE_dr2 * v2 / r2[:, None]

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dv1)
        np.add.at(f, atoms[:, 1], dv1)
        np.add.at(f, atoms[:, 2], -dv2)
        np.add.at(f, atoms[:, 3], dv2)
        return e_tot, f, self._virial((v1, dv1), (v2, dv2))

    def compute_bondangle(self, vecs, atoms, theta0, r0, k):
        # angle part (atoms 0,1,2)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        ra = np.sqrt(np.sum(va * va, -1, keepdims=True))
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        na = va / ra
        nb = vb / rb
        costheta = np.sum(na * nb, -1)  # (n,)
        dcos = costheta - np.cos(theta0)

        # bond part (atoms 3,4)
        vc = vecs[atoms[:, 3], atoms[:, 4]]
        rc = np.sqrt(np.sum(vc * vc, -1))  # (n,)
        dr = rc - r0

        raw = k * dr * dcos
        e = np.clip(raw, CLIP_BONDANGLE, None)
        e_tot = np.sum(e)

        mask = (raw > CLIP_BONDANGLE).astype(float)[:, None]

        # dE/d(cos) = k * dr
        dE_dcos = (k * dr)[:, None] * mask
        costheta_k = costheta[:, None]
        dcos_dva = (nb - costheta_k * na) / ra
        dcos_dvb = (na - costheta_k * nb) / rb

        dE_dva = dE_dcos * dcos_dva
        dE_dvb = dE_dcos * dcos_dvb

        # dE/d(rc) = k * dcos
        dE_drc = (k * dcos)[:, None] * mask
        dE_dvc = dE_drc * vc / rc[:, None]  # (n, 3)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # angle vectors: va = pos[a0]-pos[a1], vb = pos[a2]-pos[a1]
        np.add.at(f, atoms[:, 0], -dE_dva)
        np.add.at(f, atoms[:, 1], dE_dva + dE_dvb)
        np.add.at(f, atoms[:, 2], -dE_dvb)
        # bond vector: vc = pos[a3]-pos[a4]
        np.add.at(f, atoms[:, 3], -dE_dvc)
        np.add.at(f, atoms[:, 4], dE_dvc)
        return e_tot, f, self._virial((va, dE_dva), (vb, dE_dvb), (vc, dE_dvc))

    def compute_angleangle(self, vecs, atoms, theta1_0, theta2_0, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 4]]
        vd = vecs[atoms[:, 5], atoms[:, 4]]
        ra = np.sqrt(np.sum(va * va, -1, keepdims=True))
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        rc = np.sqrt(np.sum(vc * vc, -1, keepdims=True))
        rd = np.sqrt(np.sum(vd * vd, -1, keepdims=True))
        na, nb = va / ra, vb / rb
        nc, nd = vc / rc, vd / rd
        ct1 = np.sum(na * nb, -1)  # (n,)
        ct2 = np.sum(nc * nd, -1)
        dct1 = ct1 - np.cos(theta1_0)
        dct2 = ct2 - np.cos(theta2_0)
        e = k * dct1 * dct2
        e_tot = np.sum(e)

        # dE/d(ct1) = k * dct2,  dE/d(ct2) = k * dct1
        dE_dct1 = (k * dct2)[:, None]
        dE_dct2 = (k * dct1)[:, None]

        ct1_k = ct1[:, None]
        ct2_k = ct2[:, None]
        dct1_dva = (nb - ct1_k * na) / ra
        dct1_dvb = (na - ct1_k * nb) / rb
        dct2_dvc = (nd - ct2_k * nc) / rc
        dct2_dvd = (nc - ct2_k * nd) / rd

        dE_dva = dE_dct1 * dct1_dva
        dE_dvb = dE_dct1 * dct1_dvb
        dE_dvc = dE_dct2 * dct2_dvc
        dE_dvd = dE_dct2 * dct2_dvd

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # angle 1: va=pos[0]-pos[1], vb=pos[2]-pos[1]
        np.add.at(f, atoms[:, 0], -dE_dva)
        np.add.at(f, atoms[:, 1], dE_dva + dE_dvb)
        np.add.at(f, atoms[:, 2], -dE_dvb)
        # angle 2: vc=pos[3]-pos[4], vd=pos[5]-pos[4]
        np.add.at(f, atoms[:, 3], -dE_dvc)
        np.add.at(f, atoms[:, 4], dE_dvc + dE_dvd)
        np.add.at(f, atoms[:, 5], -dE_dvd)
        return (
            e_tot,
            f,
            self._virial((va, dE_dva), (vb, dE_dvb), (vc, dE_dvc), (vd, dE_dvd)),
        )

    def _dihedral_phi_and_grads(self, va, vb, vc):
        """
        Returns (phi, dphi/d(pos_a0..3), dphi/d(va, vb, vc)) for a batch of
        dihedrals -- position-level gradients for the forces, vector-level ones
        for the virial.

        Vector convention:
            va = vecs[a0, a1] = pos_a0 - pos_a1
            vb = vecs[a2, a1] = pos_a2 - pos_a1  (central bond)
            vc = vecs[a3, a2] = pos_a3 - pos_a2

        phi = atan2(S, C),  S = (axis x u)·v,  C = u·v
        axis = vb/|vb|,  u = P·va,  v = P·vc  (P = I - axis⊗axis, perp projection)

        Analytic vec-level gradients (derived via chain rule through u,v):
            dphi/dva  = (C*(v x axis) - S*v) / (S²+C²)
            dphi/dvc  = (C*(axis x u) - S*u) / (S²+C²)
            dphi/dvb  = -(va·axis/|vb|)*dphi/dva - (vc·axis/|vb|)*dphi/dvc

        Position-level chain rule (va=-pos_a1, vb=pos_a2-pos_a1, vc=-pos_a2):
            dphi/d(pos_a0) =  dphi/dva
            dphi/d(pos_a1) = -dphi/dva - dphi/dvb
            dphi/d(pos_a2) =  dphi/dvb - dphi/dvc
            dphi/d(pos_a3) =  dphi/dvc
        """
        rb = np.sqrt(np.sum(vb * vb, -1, keepdims=True))
        axis = vb / rb
        u = va - np.sum(va * axis, -1, keepdims=True) * axis  # va perp to axis
        v = vc - np.sum(vc * axis, -1, keepdims=True) * axis  # vc perp to axis
        axu = np.cross(axis, u, -1)

        S = np.sum(axu * v, -1)  # sin-like
        C = np.sum(u * v, -1)  # cos-like
        r2 = np.maximum(S**2 + C**2, 1e-30)
        phi = np.arctan2(S, C)

        # vec-level gradients via chain rule through u and v
        # dphi/dva: dS/dva = v×axis,  dC/dva = v
        dphi_dva = (C[:, None] * np.cross(v, axis, -1) - S[:, None] * v) / r2[:, None]
        # dphi/dvc: dS/dvc = axu,  dC/dvc = u
        dphi_dvc = (C[:, None] * axu - S[:, None] * u) / r2[:, None]
        # dphi/dvb: chain rule through axis = vb/|vb| (verified numerically)
        va_dot = np.sum(va * axis, -1, keepdims=True)
        vc_dot = np.sum(vc * axis, -1, keepdims=True)
        dphi_dvb = -(va_dot / rb) * dphi_dva - (vc_dot / rb) * dphi_dvc

        # position-level gradients via chain rule:
        #   va = pos_a0 - pos_a1,  vb = pos_a2 - pos_a1,  vc = pos_a3 - pos_a2
        dphi_dpos0 = dphi_dva
        dphi_dpos1 = -dphi_dva - dphi_dvb
        dphi_dpos2 = dphi_dvb - dphi_dvc
        dphi_dpos3 = dphi_dvc

        # The vec-level gradients are returned alongside the position-level
        # ones because the virial needs them and they would otherwise have to be
        # rebuilt from the position gradients, which is not possible: the map
        # from three vectors to four positions is not invertible (it drops the
        # centre of mass).  The forces use the position form, the virial the
        # vector form, and both come from the one derivation above.
        return (
            phi,
            dphi_dpos0,
            dphi_dpos1,
            dphi_dpos2,
            dphi_dpos3,
            dphi_dva,
            dphi_dvb,
            dphi_dvc,
        )

    def compute_periodicdihedral(self, vecs, atoms, phi0, n, k):
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]

        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )

        e = k * (1 + np.cos(n * phi - phi0))
        e_tot = np.sum(e)

        # dE/dphi = -k * n * sin(n*phi - phi0)
        dE_dphi = (-k * n * np.sin(n * phi - phi0))[:, None]  # (n,1)

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # F = -dE/d(pos) = -dE/dphi * dphi/d(pos)
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        return (
            e_tot,
            f,
            self._virial((va, dE_dphi * dva), (vb, dE_dphi * dvb), (vc, dE_dphi * dvc)),
        )

    def compute_dihedralbond(self, vecs, atoms, phi0, n, k, r0):
        # dihedral part (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )
        cos_term = 1 + np.cos(n * phi - phi0)  # (n,)

        # bond part (atoms 4,5)
        vd = vecs[atoms[:, 4], atoms[:, 5]]
        rd = np.sqrt(np.sum(vd * vd, -1))  # (n,)
        dr = rd - r0

        e = k * dr * cos_term
        e_tot = np.sum(e)

        # dE/dphi  = k * dr * (-n * sin(n*phi-phi0))
        dE_dphi = (k * dr * (-n * np.sin(n * phi - phi0)))[:, None]
        # dE/d(rd) = k * cos_term
        dE_drd = (k * cos_term)[:, None]
        dE_dvd = dE_drd * vd / rd[:, None]

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        np.add.at(f, atoms[:, 4], -dE_dvd)  # vd = pos[a4]-pos[a5]
        np.add.at(f, atoms[:, 5], dE_dvd)
        return (
            e_tot,
            f,
            self._virial(
                (va, dE_dphi * dva),
                (vb, dE_dphi * dvb),
                (vc, dE_dphi * dvc),
                (vd, dE_dvd),
            ),
        )

    def compute_dihedralangle(self, vecs, atoms, phi0, n, k, theta0):
        # dihedral (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )
        cos_term = 1 + np.cos(n * phi - phi0)

        # angle (atoms 4,5,6)
        vp = vecs[atoms[:, 4], atoms[:, 5]]
        vq = vecs[atoms[:, 6], atoms[:, 5]]
        rp = np.sqrt(np.sum(vp * vp, -1, keepdims=True))
        rq = np.sqrt(np.sum(vq * vq, -1, keepdims=True))
        np_ = vp / rp
        nq = vq / rq
        costheta = np.sum(np_ * nq, -1)  # (n,)
        dcos = costheta - np.cos(theta0)

        e = k * dcos * cos_term
        e_tot = np.sum(e)

        dE_dphi = (k * dcos * (-n * np.sin(n * phi - phi0)))[:, None]
        dE_dcos = (k * cos_term)[:, None]

        costheta_k = costheta[:, None]
        dcos_dvp = (nq - costheta_k * np_) / rp
        dcos_dvq = (np_ - costheta_k * nq) / rq

        dE_dvp = dE_dcos * dcos_dvp
        dE_dvq = dE_dcos * dcos_dvq

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        np.add.at(f, atoms[:, 4], -dE_dvp)
        np.add.at(f, atoms[:, 5], dE_dvp + dE_dvq)
        np.add.at(f, atoms[:, 6], -dE_dvq)
        return (
            e_tot,
            f,
            self._virial(
                (va, dE_dphi * dva),
                (vb, dE_dphi * dvb),
                (vc, dE_dphi * dvc),
                (vp, dE_dvp),
                (vq, dE_dvq),
            ),
        )

    def compute_dihedralangleangle(self, vecs, atoms, phi0, n, k, theta0_1, theta0_2):
        # dihedral (atoms 0..3)
        va = vecs[atoms[:, 0], atoms[:, 1]]
        vb = vecs[atoms[:, 2], atoms[:, 1]]
        vc = vecs[atoms[:, 3], atoms[:, 2]]
        phi, dp0, dp1, dp2, dp3, dva, dvb, dvc = self._dihedral_phi_and_grads(
            va, vb, vc
        )
        cos_term = 1 + np.cos(n * phi - phi0)

        # angle 1 (atoms 0,1,2) - same vectors as dihedral start
        va1 = vecs[atoms[:, 0], atoms[:, 1]]
        vb1 = vecs[atoms[:, 2], atoms[:, 1]]
        ra1 = np.sqrt(np.sum(va1 * va1, -1, keepdims=True))
        rb1 = np.sqrt(np.sum(vb1 * vb1, -1, keepdims=True))
        na1, nb1 = va1 / ra1, vb1 / rb1
        costheta1 = np.sum(na1 * nb1, -1)
        dcos1 = costheta1 - np.cos(theta0_1)

        # angle 2 (atoms 1,2,3)
        va2 = vecs[atoms[:, 1], atoms[:, 2]]
        vb2 = vecs[atoms[:, 3], atoms[:, 2]]
        ra2 = np.sqrt(np.sum(va2 * va2, -1, keepdims=True))
        rb2 = np.sqrt(np.sum(vb2 * vb2, -1, keepdims=True))
        na2, nb2 = va2 / ra2, vb2 / rb2
        costheta2 = np.sum(na2 * nb2, -1)
        dcos2 = costheta2 - np.cos(theta0_2)

        e = k * dcos1 * dcos2 * cos_term
        e_tot = np.sum(e)

        dE_dphi = (k * dcos1 * dcos2 * (-n * np.sin(n * phi - phi0)))[:, None]
        dE_dcos1 = (k * dcos2 * cos_term)[:, None]
        dE_dcos2 = (k * dcos1 * cos_term)[:, None]

        ct1k = costheta1[:, None]
        dcos1_dva1 = (nb1 - ct1k * na1) / ra1
        dcos1_dvb1 = (na1 - ct1k * nb1) / rb1

        ct2k = costheta2[:, None]
        dcos2_dva2 = (nb2 - ct2k * na2) / ra2
        dcos2_dvb2 = (na2 - ct2k * nb2) / rb2

        # angle1: va1=pos[0]-pos[1], vb1=pos[2]-pos[1]
        # angle vector va1=pos[0]-pos[1]: F_pos0 += -dE/dva1, F_pos1 += +dE/dva1 ... chain rule:
        # E_ang1 contrib: dE/d(pos_0) = dE_dcos1 * dcos1_dva1 (since va1=pos0-pos1, d/dpos0=+I)
        # F = -dE/d(pos), so F_pos0 -= dE_dcos1 * dcos1_dva1
        dE_ang1_pos0 = dE_dcos1 * dcos1_dva1
        dE_ang1_pos1 = -dE_dcos1 * (dcos1_dva1 + dcos1_dvb1)
        dE_ang1_pos2 = dE_dcos1 * dcos1_dvb1

        # angle2: va2=pos[1]-pos[2], vb2=pos[3]-pos[2]
        dE_ang2_pos1 = dE_dcos2 * dcos2_dva2
        dE_ang2_pos2 = -dE_dcos2 * (dcos2_dva2 + dcos2_dvb2)
        dE_ang2_pos3 = dE_dcos2 * dcos2_dvb2

        n_atoms = vecs.shape[0]
        f = np.zeros((n_atoms, 3))
        # dihedral forces
        np.add.at(f, atoms[:, 0], -dE_dphi * dp0)
        np.add.at(f, atoms[:, 1], -dE_dphi * dp1)
        np.add.at(f, atoms[:, 2], -dE_dphi * dp2)
        np.add.at(f, atoms[:, 3], -dE_dphi * dp3)
        # angle1 forces
        np.add.at(f, atoms[:, 0], -dE_ang1_pos0)
        np.add.at(f, atoms[:, 1], -dE_ang1_pos1)
        np.add.at(f, atoms[:, 2], -dE_ang1_pos2)
        # angle2 forces
        np.add.at(f, atoms[:, 1], -dE_ang2_pos1)
        np.add.at(f, atoms[:, 2], -dE_ang2_pos2)
        np.add.at(f, atoms[:, 3], -dE_ang2_pos3)
        # The two angles reuse the dihedral's own atoms but not its vectors:
        # angle 1 shares va/vb, while angle 2 is built from va2 = vecs[a1, a2]
        # and vb2 = vecs[a3, a2], neither of which is va, vb or vc.  Each
        # displacement vector is listed once with its own total gradient.
        w = self._virial(
            (va, dE_dphi * dva),
            (vb, dE_dphi * dvb),
            (vc, dE_dphi * dvc),
            (va1, dE_dcos1 * dcos1_dva1),
            (vb1, dE_dcos1 * dcos1_dvb1),
            (va2, dE_dcos2 * dcos2_dva2),
            (vb2, dE_dcos2 * dcos2_dvb2),
        )
        return e_tot, f, w

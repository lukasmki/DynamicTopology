"""Does the Water dataset actually bind a hydrogen bond?

Nothing in this suite used to ask.  `test_gradients` and `test_stress` check
every force against its own energy, `test_collapse` checks that molecules do not
interpenetrate, and `test_reference_energies` checks each template against its
own atomization energy -- all of which hold exactly as well for a liquid that
boils as for one that does not.  The hydrogen bond, the water dimer well and the
liquid density lived only as prose tables in `datasets/Water/README.md` and in
`forcefield/lj.py`'s docstring, so `TAPER_RADIUS`, `SWITCH_RADIUS` and `sigma_H`
could all be moved without a single test noticing.

That is not hypothetical: the dataset shipped for some time with `q_H = +0.224`
and a 1.258 D gas-phase dipole -- roughly half of what any working water model
carries -- which put the dimer well at -0.072 eV against a CCSD(T) reference of
-0.218 eV and produced a liquid whose O-O first peak sat at 3.05-3.15 A instead
of 2.8 A.  Every test in the suite passed throughout.

So this file pins the three numbers that decide whether the liquid is a liquid,
and it pins them against external references rather than against the code:

  the dimer well      -0.218 eV at 2.91 A, CCSD(T)
  the monomer dipole  1.855 D, gas-phase experiment
  the polarizability  1.45 A^3 isotropic, experiment

None of the three is expected to be hit exactly, and the last one *cannot* be --
see `TestPolarizability`.  The bounds below are wide enough to be a description
of the model rather than a transcription of its current output, and each one
records what it was measured at.
"""

import math

import numpy as np
import pytest
from ase import Atoms

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet
from DynamicTopology.core.topology import Topology
from DynamicTopology.forcefield.acks2 import ACKS2

RSET_PATH = "datasets/Water/Water.json"

# Experimental monomer geometry, held rigid throughout.  Both monomers are built
# from it, so every bonded term cancels out of an interaction energy exactly and
# what these tests measure is the nonbonded surface alone.
R_OH = 0.9572
THETA = math.radians(104.52)

# Isolated-molecule box, large enough that nothing sees its own image.  `pbc` is
# off, so `ACKS2` takes the `MinimumImage` kernel and this is a vacuum number.
CELL = 30.0

# Conversion from e*Angstrom to Debye, and 1/(4 pi eps0) in eV*Angstrom/e^2 --
# the same constant `ACKS2.CCOUL` carries, repeated here because the
# polarizability below converts with it rather than scaling an energy by it.
DEBYE = 4.803205
CCOUL = 14.4


@pytest.fixture(scope="module")
def reaction_set() -> ReactionSet:
    rset = ReactionSet()
    rset.load(RSET_PATH)
    return rset


def monomer() -> np.ndarray:
    """`[O, H, H]` in the xy plane, with the first O-H along +x."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [R_OH, 0.0, 0.0],
            [R_OH * math.cos(THETA), R_OH * math.sin(THETA), 0.0],
        ]
    )


def dimer(d_oo: float, tilt: float = 110.0) -> np.ndarray:
    """Cs water dimer, `[Od, Hb, Hd, Oa, Ha, Ha]`, O-O along x.

    The donor points its first O-H straight down the O-O axis, so the hydrogen
    bond is linear; the acceptor's bisector is tilted `tilt` degrees off that
    axis with its molecular plane perpendicular to the donor's.  That is the
    standard Cs arrangement, and `tilt` is held fixed rather than optimised so
    that the curve below is a deterministic function of one variable.

    Built here rather than taken from `datasets/Water/make_water.py`, whose
    `dimer()` lays out *reaction* frames -- the bridging proton on the O-O axis
    between the two oxygens, at an arbitrary fraction of the way across.  That
    is the right parameterisation for a proton transfer and the wrong one for a
    hydrogen bond, where the proton stays on its donor.
    """
    half = THETA / 2
    donor = monomer()
    t = math.radians(tilt)
    bisector = np.array([-math.cos(t), 0.0, math.sin(t)])
    perp = np.array([0.0, 1.0, 0.0])
    acceptor = np.array(
        [
            [0.0, 0.0, 0.0],
            R_OH * (math.cos(half) * bisector + math.sin(half) * perp),
            R_OH * (math.cos(half) * bisector - math.sin(half) * perp),
        ]
    )
    return np.vstack([donor, acceptor + np.array([d_oo, 0.0, 0.0])])


def energy(positions: np.ndarray, reaction_set: ReactionSet) -> tuple[float, int]:
    """Total energy through the calculator, and the largest EVB basis size."""
    atoms = Atoms(
        "OHH" * (len(positions) // 3),
        positions=positions,
        cell=[CELL] * 3,
        pbc=False,
    )
    atoms.calc = DynamicTopology(atoms, reaction_set)
    total = atoms.get_potential_energy()
    nstates = max(b["nstates"] for b in atoms.calc.diagnostics["blocks"])
    return total, nstates


def charges(reaction_set: ReactionSet, positions: np.ndarray) -> np.ndarray:
    """ACKS2 charges on an isolated water, in atom order."""
    atoms = Atoms("OHH", positions=positions, cell=[CELL] * 3, pbc=False)
    topology = Topology.from_atoms(atoms)
    topology.set_terms(reaction_set.get_terms(topology))
    acks2 = ACKS2()
    acks2(atoms.positions, atoms.pbc, atoms.cell.array, topology.term_dict)
    return acks2.Q


# Reference well depth and position for the water dimer, CCSD(T)/CBS.
REFERENCE_DEPTH = 0.218
REFERENCE_R_OO = 2.91

# O-O separations the curve is sampled at.  Wide enough either side of the
# minimum that a well which has slid out of the bracket fails on position rather
# than quietly reporting the endpoint as its minimum.
# 2.91 is on the grid on purpose: it is where the reference minimum is and where
# the *previous* parameters put theirs, so a dataset that has slid back to the
# underbound surface fails on position as well as on depth.
SEPARATIONS = (2.6, 2.7, 2.75, 2.8, 2.85, 2.9, 2.91, 3.0, 3.1, 3.3, 3.6, 4.0)


class TestHydrogenBond:
    """The water dimer well, through the full calculator.

    Measured at `eta_O = 2.8084`, `eta_H = 5.4635`, `sigma_H = eps_H = 0` and
    `sigma_O = 0.305` nm (3.05 A), against the same scan on the parameters that preceded
    them:

        R_OO     before      after      (eV, this geometry)
        2.60    +0.1100    -0.0168
        2.70    -0.0143    -0.1305
        2.80    -0.0603    -0.1651
        2.85    -0.0685    -0.1677   <- after
        2.91    -0.0719    -0.1643   <- before
        3.10    -0.0662    -0.1370
        3.60    -0.0466    -0.0736

    The well is 2.3x deeper and its minimum has moved in from 2.91 to 2.85 A,
    which is the direction the liquid needs: the O-O first peak was at 3.07 A
    against an experimental 2.80 A and is now at 2.82 A.  Note that the *position* bracket below deliberately
    excludes the reference 2.91 A rather than centring on it -- thermal disorder
    puts a liquid's first peak outside its pair minimum, so a dimer that already
    sits at 2.91 A produces a liquid past 3 A, which is the failure this file was
    written for.

    **-0.18 eV against a -0.218 eV reference is the intended answer, not a
    residual deficit**, and going deeper would be wrong for a different reason
    than it looks.  See `TestPolarizability`: this force field has no many-body
    response at all, so a dimer tuned to the gas-phase pair energy leaves the
    liquid underbound, exactly as a non-polarizable model tuned that way does.
    SPC/E carries a 2.35 D dipole against a gas-phase 1.855 D for this reason.
    The bracket below therefore sits *around* the current value rather than
    reaching down to the reference.
    """

    def test_the_well_is_deep_enough_to_hold_a_liquid(self, reaction_set):
        monomer_energy, _ = energy(monomer(), reaction_set)
        curve = {
            d: energy(dimer(d), reaction_set)[0] - 2 * monomer_energy
            for d in SEPARATIONS
        }
        depth = -min(curve.values())

        assert 0.15 < depth < 0.25, (
            f"the water dimer well is {depth:.4f} eV against a CCSD(T) reference "
            f"of {REFERENCE_DEPTH} eV; outside 0.15-0.25 eV the liquid is either "
            "under-cohesive (the 3.05 A O-O peak this file was written for) or "
            "over-bound to the point of freezing"
        )

    def test_the_well_sits_at_a_hydrogen_bond_distance(self, reaction_set):
        monomer_energy, _ = energy(monomer(), reaction_set)
        curve = {
            d: energy(dimer(d), reaction_set)[0] - 2 * monomer_energy
            for d in SEPARATIONS
        }
        r_min = min(curve, key=curve.get)

        assert 2.70 <= r_min <= 2.90, (
            f"the dimer minimum is at {r_min:.2f} A; it is required to sit inside "
            f"the experimental {REFERENCE_R_OO} A rather than on it, because the "
            "liquid's O-O peak lands outside the dimer's minimum and has to reach "
            "2.8 A. A minimum at or beyond 2.91 A is the signature of a 12-6 wall "
            "standing where the hydrogen bond should be -- that is what put the "
            "liquid's first peak at 3.05-3.15 A"
        )

    def test_the_curve_is_a_single_well(self, reaction_set):
        """No step, no second minimum -- and no EVB channel opening mid-curve.

        Two waters at their own equilibrium geometries cannot mix: fast-forces' `coupling`
        pins each width so the coupling is off at the reactant minimum.  If this
        ever reports more than one state the curve above stops being a pure
        nonbonded measurement and the brackets there mean something else.
        """
        monomer_energy, _ = energy(monomer(), reaction_set)
        values, states = [], []
        for d in SEPARATIONS:
            total, nstates = energy(dimer(d), reaction_set)
            values.append(total - 2 * monomer_energy)
            states.append(nstates)

        assert set(states) == {1}, (
            f"the dimer scan admitted an EVB channel (basis sizes {states}); "
            "these are two equilibrium monomers and should stay diabatic"
        )

        r_min = int(np.argmin(values))
        assert r_min not in (0, len(values) - 1), "the minimum is at an endpoint"
        inward, outward = values[: r_min + 1], values[r_min:]
        assert all(np.diff(inward) < 0), f"the approach is not monotone: {inward}"
        assert all(np.diff(outward) > 0), f"the tail is not monotone: {outward}"


class TestMonomerCharges:
    """What `ACKS2` puts on an isolated water.

    The dipole is the thing to watch rather than the charges themselves: the
    electrostatic part of a hydrogen bond goes as `mu^2`, so the 1.258 D this
    dataset used to produce was not 32% away from the 1.855 D experiment, it was
    a factor of 2.2 away from the binding energy.

    Measured here at 1.711 D with `q_H = +0.3040`.  For scale, the fixed-charge
    models: SPC/E `q_H = +0.4238` (2.35 D), TIP3P `+0.417`, TIP4P/2005 `+0.5564`.
    Those all sit *above* the gas-phase experiment deliberately, which is the
    same compensation `TestHydrogenBond` describes; this model sits below it and
    makes up the difference in the short-range kernel, which is why the dimer
    lands where it does with a smaller dipole than SPC/E's.
    """

    def test_the_charges_are_large_enough_to_hydrogen_bond(self, reaction_set):
        q = charges(reaction_set, monomer())

        assert q[0] < 0 < q[1], f"water is polarised the wrong way round: {q}"
        assert 0.28 < q[1] < 0.34, (
            f"q_H is {q[1]:+.4f}; below +0.28 the hydrogen bond is carried by "
            "charges roughly half the size of any working water model's and the "
            "liquid loses its tetrahedral structure"
        )
        assert abs(q.sum()) < 1e-10, f"the box is not neutral: {q.sum():.3e}"

    def test_the_gas_phase_dipole(self, reaction_set):
        positions = monomer()
        q = charges(reaction_set, positions)
        mu = np.linalg.norm((q[:, None] * positions).sum(axis=0)) * DEBYE

        assert 1.55 < mu < 1.95, (
            f"the gas-phase dipole is {mu:.3f} D against an experimental 1.855 D; "
            "the hydrogen bond goes as mu^2, so a 30% error here is a factor of "
            "two in the binding"
        )


class TestPolarizability:
    """The model's electronic response -- and the ceiling on it.

    An external potential enters the ACKS2 stationarity condition on exactly the
    same row as the electronegativity, so a uniform field is applied by shifting
    `b[:natoms]` and re-solving; no new code path is involved and the result is
    the exact linear response of the charges the force field uses.

    Measured, against experiment:

                         in-plane   out-of-plane   dipole axis   isotropic
        this model          1.373       0.000          0.482        0.618
        experiment          1.53        1.42           1.47         1.45

    **The out-of-plane response is exactly zero and no parameter can lift it.**
    Charge equilibration moves charge *between atoms*, so the induced dipole is
    confined to the span of the interatomic vectors; water is planar, so there is
    nowhere for charge to go perpendicular to it.  Lifting it needs a degree of
    freedom that is not a charge -- atomic dipoles, or a Drude particle.

    That is the whole reason there is no cooperativity in this model.  A water in
    the liquid carries a 1.117 D dipole against 1.111 D for the same geometries
    isolated: a +0.6% condensed-phase enhancement where real water shows +40 to
    +60%, and it stays at +0.6-0.8% under every parameter variation tried,
    including `soft_amp x5`.  Hydrogen-bonded chains and rings do not reinforce
    each other here; there are only independent pairs.

    These assertions exist so that the zero is *pinned as known* rather than
    rediscovered, and so that anyone who does add atomic dipoles finds a test
    that fails loudly and tells them why.
    """

    @staticmethod
    def polarizability(reaction_set: ReactionSet, axis: int) -> float:
        """Induced dipole per unit field along `axis`, in A^3."""
        positions = monomer()
        atoms = Atoms("OHH", positions=positions, cell=[CELL] * 3, pbc=False)
        topology = Topology.from_atoms(atoms)
        topology.set_terms(reaction_set.get_terms(topology))
        params = topology.term_dict["atom"]["kwargs"]

        vecs = positions[:, None, :] - positions[None, :, :]
        rij = np.sqrt((vecs * vecs).sum(-1))
        acks2 = ACKS2()
        A, b = acks2.build_system(rij, params)

        # A uniform field `F` along `axis` is the external potential
        # `phi_i = -F * r_i.e`, which adds to `dE/dQ_i`; `b[:n]` is `-mu`, so it
        # becomes `-(mu + phi)`.  Small enough to be linear, large enough to stay
        # well clear of the solve's conditioning.
        field = 1e-4
        natoms = len(positions)
        shifted = b.copy()
        shifted[:natoms] = b[:natoms] + field * positions[:, axis]

        q0 = np.linalg.solve(A, b)[:natoms]
        q1 = np.linalg.solve(A, shifted)[:natoms]
        induced = ((q1 - q0)[:, None] * positions).sum(axis=0)[axis]
        return induced / field * CCOUL

    def test_the_in_plane_response_is_the_right_order(self, reaction_set):
        alpha = self.polarizability(reaction_set, 0)
        assert 0.9 < alpha < 1.9, (
            f"in-plane polarizability is {alpha:.3f} A^3 against an experimental "
            "1.53; this is the one component charge equilibration can get right"
        )

    def test_the_out_of_plane_response_is_identically_zero(self, reaction_set):
        """Pinned as a known structural limit -- see the class docstring.

        If this ever fails, the model has gained a non-charge degree of freedom
        and the hydrogen-bond brackets in `TestHydrogenBond` were tuned to
        compensate for its absence.  Retune them; do not relax this.
        """
        alpha = self.polarizability(reaction_set, 2)
        assert abs(alpha) < 1e-9, (
            f"out-of-plane polarizability is {alpha:.3e} A^3, not zero -- water "
            "is planar and charge equilibration cannot polarise off the plane, "
            "so something has added a degree of freedom that is not a charge"
        )

    def test_the_isotropic_response_is_undersized_and_known_to_be(self, reaction_set):
        alpha = np.mean([self.polarizability(reaction_set, i) for i in range(3)])
        assert 0.4 < alpha < 0.9, (
            f"isotropic polarizability is {alpha:.3f} A^3 against an experimental "
            "1.45; a third of it is missing because the out-of-plane component is "
            "structurally zero, and this bracket records that rather than hiding it"
        )

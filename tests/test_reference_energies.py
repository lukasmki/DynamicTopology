"""Does the force field reproduce the energies it was fitted to?

Every other test in this suite checks the code against itself: analytic forces
against finite differences, caches against uncached lookups, energies against
their own previous values.  All of that can hold perfectly while the number the
calculator reports is simply wrong, and it did -- the fit solves
`E_QForce(x0) = E_atomization` while `System.calculate` reports
`E_QForce + E_ACKS2`, so the nonbonded energy was added on top of a bonded term
that had already absorbed it.  Water came out at -12.3701 eV against a reference
atomization energy of -9.8735: 25% overbound, through 104 passing tests.

So this file compares against the dataset instead of against the code.  The
reference energies live on the `.xyz` frames themselves (written by
`scripts/compute.py`), which makes the comparison free and leaves no room to
disagree about what the target is.

Two failure modes are deliberately kept apart, because they have nothing to do
with each other and would otherwise be reported as one number:

  `stored`     connectivity taken from `info["connectivity"]`, as the dataset
               ships it.  Isolates the *energy* question.
  `perceived`  connectivity perceived from the geometry, as any plain `.xyz`
               through `scripts/singlepoint.py` gets it.  Isolates the
               *topology* question -- notably that H2's equilibrium bond length
               of 0.7445 A sits just outside the 1.2 * 2 * r_cov(H) = 0.7440 A
               cutoff, so an unmodified perception loses every H2 in the box.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, io, units

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.forcefield.lj import LennardJones, with_exclusions
from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.io.json import read_jsonl

RSET_PATH = Path("datasets/HCombustion/HCombustion.json").resolve()

# Isolated-molecule box.  Large enough that nothing sees its own image, and
# `pbc` is off, so this is a vacuum calculation to match the reference.
CELL = 30.0

# Templates whose atoms are not all the same element, and which therefore carry
# a nonzero ACKS2 energy.  Named rather than derived so that a template quietly
# losing its charges cannot make this file pass by making the question vacuous.
#
# `ZBL` is not restricted this way: it is nonzero on *every* template, H2 and O2
# included, because it does not care what is bonded to what and a bonded pair is
# a close pair.  That is what closes the hole this list describes -- the
# double-count used to be invisible on the homonuclear templates because they
# had no nonbonded energy at all.
HETERONUCLEAR = {"mol_03", "mol_04", "mol_05", "mol_06"}


def _entries(kind: str) -> list[Path]:
    manifest = json.loads(RSET_PATH.read_text())
    return [RSET_PATH.parent / entry["path"] for entry in manifest[kind]]


@pytest.fixture(scope="module")
def reaction_set() -> ReactionSet:
    return ReactionSet(RSET_PATH)


def isolate(frame: Atoms, perceived: bool) -> Atoms:
    """`frame` centred in a vacuum box, with or without its stored bonds."""
    atoms = frame.copy()
    atoms.calc = None  # the reference energy belongs to the frame, not to this
    if perceived:
        atoms.info.pop("connectivity", None)
    atoms.set_cell(np.eye(3) * CELL)
    atoms.set_pbc(False)
    atoms.positions += CELL / 2 - atoms.positions.mean(0)
    return atoms


def evaluate(atoms: Atoms, reaction_set: ReactionSet) -> tuple[float, dict]:
    """Total energy and diagnostics through the public calculator path."""
    atoms.calc = DynamicTopology(atoms, reaction_set)
    return atoms.get_potential_energy(), atoms.calc.diagnostics


class TestTemplateEnergies:
    """A molecule template, at its own geometry, must read its own energy.

    This is the one place in the project where the answer is exact rather than
    approximate: `fit_dissociation_energies` solves a depth scale precisely so
    that this equality holds, and `brentq` converges it to 1e-12.  Any residual
    here is not force field error, it is the fit and the calculator disagreeing
    about what they are computing.
    """

    @pytest.mark.parametrize("perceived", [False, True], ids=["stored", "perceived"])
    def test_template_energy_matches_its_reference(self, reaction_set, perceived):
        errors = {}
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            reference = frame.get_potential_energy()
            energy, _ = evaluate(isolate(frame, perceived), reaction_set)
            if abs(energy - reference) > 1e-6:
                errors[stem.name] = (
                    f"{frame.get_chemical_formula()}: "
                    f"reference {reference:+.4f} eV, calculated {energy:+.4f} eV, "
                    f"error {energy - reference:+.4f} eV"
                )
        assert not errors, "templates do not reproduce their own energies:\n  " + (
            "\n  ".join(f"{k}  {v}" for k, v in sorted(errors.items()))
        )

    def test_the_nonbonded_term_is_not_zero(self, reaction_set):
        """Anti-vacuity: the test above must not pass by ACKS2 vanishing.

        The double count is invisible on H2 and O2 because a homonuclear
        diatomic has no charge separation and so no nonbonded energy at all.
        If the heteronuclear templates ever join them -- a charge parameter lost
        in a refactor, say -- every assertion above would start passing for a
        reason that has nothing to do with the fit being right.
        """
        for stem in _entries("molecules"):
            if stem.name not in HETERONUCLEAR:
                continue
            frame = io.read(stem.with_suffix(".xyz"))
            _, diagnostics = evaluate(isolate(frame, False), reaction_set)
            assert abs(diagnostics["energy_nonbonded"]) > 0.1, (
                f"{stem.name} ({frame.get_chemical_formula()}) carries a "
                "negligible nonbonded energy, so it can no longer detect the "
                "bonded and nonbonded terms disagreeing about the energy zero"
            )

    def test_the_repulsion_is_not_zero_on_any_template(self, reaction_set):
        """Anti-vacuity for `ZBL`, on every template rather than some.

        `ZBL` is what the depth fit now has to absorb, and it is substantial:
        +2.3 eV on H2 at its own bond length, +11.6 on O2.  If it silently went
        to zero, every template above would still reproduce its own energy --
        the fit would simply have absorbed nothing -- and the file would pass
        while the box had no repulsion in it at all.

        Unrestricted by element, unlike the ACKS2 check above, because that is
        the point of the term: a homonuclear diatomic has no charge separation
        and so no ACKS2 energy, and used to carry no nonbonded energy of any
        kind.
        """
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            if len(frame) == 1:
                continue  # a single atom has no pairs
            _, diagnostics = evaluate(isolate(frame, False), reaction_set)
            assert diagnostics["energy_zbl"] > 0.5, (
                f"{stem.name} ({frame.get_chemical_formula()}) carries "
                f"{diagnostics['energy_zbl']:+.4f} eV of repulsion; the term "
                "the fit is absorbing has gone away"
            )


class TestTheLennardJones:
    """`forcefield/lj.py` is back in the active force field, and on new terms.

    It supplies the intermolecular wall and the dispersion, which the model had
    neither of once `zbl.taper` cut the screened-nuclear form off at 1.5 A.
    What makes that possible is `lj.switch`: the term is now ~1e-2 eV at a bond
    length instead of ~1e3, so it can be applied to every pair with **no
    exclusions**, which is what makes it identical on every diabatic state and
    puts it structurally where `ZBL` already is.  The four failure modes in that
    module's docstring all descended from a broken bond paying hundreds of eV
    for an exclusion it had lost; with nothing excluded there is nothing to
    lose.

    So the decomposition identity below is no longer what the calculator relies
    on -- `ReactionSet.load` derives no exclusions and `with_exclusions` builds
    them only here.  It is kept because it is the sharpest available check that
    `LennardJones.__call__` and `QForce.compute_exclusion` still evaluate the
    same function of the same numbers, switch included, and because a dataset
    shipping explicit `exclusion` terms would still be honoured.
    """

    # What `lj.switch` has to hold the term under at a bond length.  Not a
    # tolerance on a converged quantity: it is the design claim.  Unswitched,
    # these templates carry 400-1400 eV per bond, which is the number that made
    # the term unusable in a reactive model.
    BONDED_CEILING: float = 1.0

    def test_the_switch_keeps_it_off_at_bond_lengths(self, reaction_set):
        """No isolated template may carry more than `BONDED_CEILING` of 12-6.

        This is the load-bearing property of the whole re-enablement and it is
        the one that a change to `lj.SWITCH_RADIUS` would break silently.  An
        isolated template has only bonded and 1-3 pairs, so whatever it carries
        here is pure contamination -- it is absorbed by the fitted Morse depths
        the way `ZBL`'s +2 to +11.6 eV is, but only while it stays this size.

        Measured at `SWITCH_RADIUS = 0.22` nm: 0.061 eV for water, and the
        worst template in either dataset is the O2 bond at 0.35 eV.  At the
        complementary radius 0.15 nm -- the tidy choice, matching `zbl.taper` --
        water alone reads **+20.6 eV**, which is what this pins against.
        """
        errors = {}
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            atoms = isolate(frame, False)
            topology = Topology.from_atoms(atoms)
            topology.set_terms(reaction_set.get_terms(topology))
            if "lennardjones" not in topology.term_dict:
                continue
            total, _, _ = LennardJones()(
                atoms.positions, atoms.pbc, atoms.cell, topology.term_dict
            )
            if abs(total) > self.BONDED_CEILING:
                errors[stem.name] = f"{frame.get_chemical_formula()}: {total:+.4f} eV"
        assert not errors, (
            "the switched 12-6 is not off at bond lengths; `lj.SWITCH_RADIUS` "
            f"has moved inward or `lj.switch` has stopped biting (ceiling "
            f"{self.BONDED_CEILING} eV):\n  "
            + "\n  ".join(f"{k}  {v}" for k, v in sorted(errors.items()))
        )

    def test_the_sum_cancels_on_an_isolated_template(self, reaction_set):
        """`E_LJ(state) = sum_all_pairs - sum_per_molecule_pairs`.

        An isolated template is one molecule, so every pair is intramolecular
        and the two halves must cancel exactly.  Bit-exactness is a fair demand
        rather than a lucky one: both sides evaluate the same `pair_potential`
        on the same pair list.  The tolerance below is for summation order
        alone, and measures ~1e-13 eV against terms of ~1e3.
        """
        errors = {}
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            atoms = isolate(frame, False)
            topology = Topology.from_atoms(atoms)
            topology.set_terms(with_exclusions(reaction_set.get_terms(topology)))

            term_dict = topology.term_dict
            if "exclusion" not in term_dict:
                # A single atom has no pairs at all; nothing to cancel.
                assert len(atoms) == 1, f"{stem.name} has no exclusion terms"
                continue

            total, forces, _ = LennardJones()(
                atoms.positions, atoms.pbc, atoms.cell, term_dict
            )
            cancel, cancel_forces, _ = QForce()(
                atoms.positions,
                atoms.pbc,
                atoms.cell,
                {"exclusion": term_dict["exclusion"]},
            )
            residual = abs(total + cancel)
            if residual > 1e-9 or np.abs(forces + cancel_forces).max() > 1e-9:
                errors[stem.name] = (
                    f"{frame.get_chemical_formula()}: global {total:+.6f} eV, "
                    f"exclusions {cancel:+.6f} eV, residual {residual:.3e} eV"
                )
        assert not errors, (
            "the Lennard-Jones sum does not cancel on an isolated template:\n  "
            + "\n  ".join(f"{k}  {v}" for k, v in sorted(errors.items()))
        )


class TestTemplateGeometries:
    """A template must be at rest at the geometry its reference energy belongs to.

    `TestTemplateEnergies` above pins the *energy* at each template's stored QM
    geometry, and it passes to 1e-13 -- `fit_dissociation_energies` solves a
    depth scale until it does.  For a while nothing pinned the *gradient*
    there, and that turned out to be the whole difference between reproducing a
    number and reproducing a molecule.

    `ZBL` is a real repulsion at bonding distances -- 2.0 eV at the H2 bond
    length, 5.4 at O-H, 11.6 at O-O, with slopes to match -- so the Morse has to
    lean into it for the sum to be flat, and nothing made it.  What that cost,
    measured by relaxing each template under the force field of the time
    (Angstrom, against the same relaxation with `ZBL` stubbed to zero):

        template  bond    QM      +ZBL      err     -ZBL      err
        H2        H-H   0.7445  0.8495  +0.1050  0.7772  +0.0326
        O2        O-O   1.1961  1.2780  +0.0818  1.2095  +0.0134
        HO        O-H   0.9752  1.0164  +0.0412  0.9418  -0.0334
        H2O       O-H   0.9615  1.0117  +0.0503  0.9377  -0.0238
        HO2       O-O   1.3070  2.1314  +0.8245  1.3050  -0.0020
        HO2       O-H   0.9859  1.1217  +0.1358  0.9377  -0.0482
        H2O2      O-O   1.4383  1.4919  +0.0536  1.4237  -0.0147
        H2O2      O-H   0.9652  1.2083  +0.2432  0.9203  -0.0449

    `fit.dissociation.fit_bond_lengths` is the missing condition and these are
    the tests that hold it: one equation per bond type, the total force along it
    vanishing at the reference geometry, solved rather than fitted.  Every bond
    now relaxes to within 0.009 A of its reference and H2 to within 0.0001.

    **What this does not fix, and cannot.**  The minimum is in the right place;
    the curvature there is not.  `ZBL`'s own second derivative at a bond length
    is comparable to the bond's -- 68.5 eV/A**2 at O-H, on its own worth
    4431 cm^-1 against an experimental 3756 -- and the Morse contribution at a
    minimum is positive by definition, so the total can never come *below* the
    repulsion's own.  Every stretching frequency therefore has a floor above
    experiment, and no choice of `r0`, `k`, `D` or `c` reaches under it.  See
    `TestTemplateFrequencies`.
    """

    # Largest force, in eV/A, that any atom may feel at its own reference
    # geometry.  Loose on purpose -- this is not a convergence tolerance but a
    # statement that the template is at a stationary point at all.  The stored
    # geometries are QM minima of a different method and were never going to be
    # exactly stationary here, so the bar is set above the error the force field
    # already had rather than at zero.  Worst force per template, in eV/A:
    #
    #     mol_01 H2    +ZBL   8.88   -ZBL  0.98
    #     mol_02 O2    +ZBL  40.79   -ZBL  2.18
    #     mol_03 HO    +ZBL  13.06   -ZBL  3.69
    #     mol_04 H2O   +ZBL  17.94   -ZBL  3.55
    #     mol_05 HO2   +ZBL  27.01   -ZBL  2.88
    #     mol_06 H2O2  +ZBL  16.53   -ZBL  2.67
    #
    # 4.0 clears the whole `-ZBL` column and is beaten by every entry of the
    # other one, by between 2x and 10x.  The gap is the thing: this is not a
    # tolerance that could be argued either way.
    FORCE_TOLERANCE = 4.0

    def test_the_templates_are_at_rest_at_their_reference_geometry(self, reaction_set):
        errors = {}
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            if len(frame) < 2:
                continue  # a lone atom is stationary by construction
            atoms = isolate(frame, False)
            atoms.calc = DynamicTopology(atoms, reaction_set)
            worst = float(np.abs(atoms.get_forces()).max())
            if worst > self.FORCE_TOLERANCE:
                errors[stem.name] = (
                    f"{frame.get_chemical_formula()}: {worst:.2f} eV/A on some "
                    f"atom at the geometry its reference energy belongs to"
                )
        assert not errors, (
            "templates are not at rest at their own geometries:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in sorted(errors.items()))
        )

    def test_a_relaxed_h2_is_still_perceived_as_bonded(self, reaction_set):
        """The two halves of the code must agree that H2 is a molecule.

        `Topology.from_atoms` cuts H-H at `1.3 * (0.31 + 0.31) = 0.806 A`.
        Before `fit_bond_lengths` the force field's own minimum was at 0.8495,
        *outside* that radius, so a relaxed H2 re-perceived as two free atoms
        with a 6.84 eV step at the crossing.  MD carries its topology and so
        never hit it, but reading a file, `scripts/topologize.py` and
        `run_one.py --restart` all re-perceive.

        The margin is 0.06 A and the two numbers come from different places --
        one from covalent radii, one from a fit -- so this is worth asserting
        rather than assuming.
        """
        from ase.optimize import BFGS

        frame = io.read((_entries("molecules")[0]).with_suffix(".xyz"))
        assert frame.get_chemical_formula() == "H2", "expected mol_01 to be H2"

        atoms = isolate(frame, False)
        atoms.calc = DynamicTopology(atoms, reaction_set)
        BFGS(atoms, logfile=None).run(fmax=1e-3, steps=200)

        relaxed = atoms.get_distance(0, 1)
        perceived = Topology.from_atoms(isolate(atoms, True))
        assert perceived.graph.number_of_edges() == 1, (
            f"a relaxed H2 sits at {relaxed:.4f} A and is not perceived as "
            "bonded at all"
        )


class TestTemplateFrequencies:
    """Every stretching frequency has a floor, and the floor is above experiment.

    A harmonic frequency is the total curvature at the minimum, and the total
    is `Morse + ZBL`.  At a minimum the Morse contribution is positive, so

        nu_total  >=  nu(ZBL alone at that bond length)

    and `ZBL`'s curvature at bonding distances is not a correction.  Measured
    directly on `zbl.pair_potential`:

        pair   r (A)    ZBL k (eV/A**2)   nu from ZBL alone   experimental
        H-H   0.7445             33.9                4276           4401
        O-O   1.1961            139.7                2179           1580
        O-H   0.9615             68.5                4431           3756
        O-O   1.4383             54.9                1366            877

    Three of those four floors are already above the frequency they are
    supposed to leave room for.  No `r0`, `k`, `D` or `c` reaches under a floor,
    so this is a statement about the force field's *form*, not its parameters:
    an untapered screened-nuclear repulsion cannot coexist with correct
    vibrational frequencies.

    **The fit is now sitting on the floor, and that is the point of saying so.**
    `fit_force_constants` prices the total curvature since
    `DEFAULT_MAX_WAVENUMBER` was added, and what it bought was the whole gap
    between the fit's choices and this limit.  Stretch stiffness per bond type
    at its reference geometry, in eV/A**2, against the nonbonded part of the
    same number:

        template  bond   total   nonbonded   nu_total   nu_floor
        H2        H-H     34.2        33.9       4298       4276
        O2        O-O    569.3       139.7       4399       2179
        HO        O-H     62.1        61.3       4219       4194
        H2O       O-H     65.9        65.7       4349       4342
        HO2       O-O    109.5        92.1       1930       1770
        HO2       O-H     63.0        60.1       4252       4152
        H2O2      O-O    140.4        43.6       2185       1217
        H2O2      O-H     65.4        65.2       4332       4324

    Every X-H stretch is within 0.8 eV/A**2 of its floor -- the Morse
    contributes essentially nothing to the curvature any more.  The one bond
    that is not is O2's O-O, which sits at the wavenumber cap rather than at its
    floor because it is where the fit still has to spend for its margins.

    Before the cap existed these read 7881, 5190, 11697, 5487, 1886, 5135, 2505
    and 5168, and the fastest mode set a timestep of 0.19 fs.  They now set
    0.51 fs.

    **So this floor is no longer an observation, it is the binding constraint.**
    Nothing further is available from the bonded parameters: what is left is a
    property of the repulsion's functional form, and the only way under it is to
    change that form.

    The test below is the floor, not the current numbers: it asserts the thing
    that is structural, so it keeps holding while the parameters move, and it
    fails only if the repulsion itself changes -- which is exactly the change
    that would fix this.
    """

    # Bond lengths, in Angstrom, at which the repulsion's own curvature is
    # measured: the reference geometries of the templates that carry them.
    BONDS = [
        ("H", "H", 1.0, 1.0, 1.008, 1.008, 0.7445, 4401.0),
        ("O", "O", 8.0, 8.0, 15.999, 15.999, 1.1961, 1580.0),
        ("O", "H", 8.0, 1.0, 15.999, 1.008, 0.9615, 3756.0),
    ]

    def test_the_repulsion_alone_is_stiffer_than_the_bond_it_sits_on(self):
        """Documents the floor by asserting it, on the two bonds where it bites.

        H-H is the near miss -- 4276 against 4401, so the repulsion accounts
        for 97% of H2's real stiffness on its own and there is almost nothing
        left for the bond.  It is asserted the other way round for that reason:
        if H-H ever came out *under* its floor by a wide margin the repulsion
        would have changed, and this file should say so.
        """
        from DynamicTopology.forcefield.zbl import pair_potential

        step = 1e-4
        floors = {}
        for name_a, name_b, z_a, z_b, mass_a, mass_b, length, reference in self.BONDS:

            def energy(r: float, _a=z_a, _b=z_b) -> float:
                return float(pair_potential(np.array([r]), _a, _b)[0][0])

            curvature = (
                energy(length + step) - 2 * energy(length) + energy(length - step)
            ) / step**2
            reduced = mass_a * mass_b / (mass_a + mass_b) * units._amu
            wavenumber = float(
                np.sqrt(curvature * units._e / 1e-20 / reduced)
                / (2 * np.pi * units._c * 1e2)
            )
            floors[f"{name_a}-{name_b}"] = (wavenumber, reference)

        above = {k: v for k, v in floors.items() if v[0] > v[1]}
        assert set(above) == {"O-O", "O-H"}, (
            "the repulsion's own stiffness at a bond length no longer sits "
            f"where it did: {floors}"
        )
        assert 0.9 < floors["H-H"][0] / floors["H-H"][1] < 1.0, (
            f"H-H floor {floors['H-H'][0]:.0f} cm^-1 is no longer just under "
            f"its experimental {floors['H-H'][1]:.0f}"
        )

    # Wavenumber the fastest stretch on the surface must stay under, and the
    # only reason this file has a number in it at all: the production sweep
    # runs at 0.5 fs, velocity Verlet wants ~15 steps per vibrational period,
    # and `33356 / (15 * 0.5)` is 4448.  A refit that pushes a mode past this
    # has silently taken the timestep away, and 100 ps at 0.05 fs instead of
    # 0.5 is the difference between an eleven-hour job and a ten-day one.
    #
    # 4600 rather than 4448 because the assertion is against a mode drifting,
    # not against the last digit: the fit lands at 4399 with the cap at 4400,
    # and 4600 is 12.9 steps per period -- still integrable, and far enough
    # above the fitted value that this fails on a real regression rather than
    # on Powell stopping somewhere slightly different.
    TIMESTEP_WAVENUMBER = 4600.0

    def test_the_fastest_mode_still_admits_the_production_timestep(self, reaction_set):
        """The deliverable, asserted rather than left in a README.

        Everything else in this class is about *why* the frequencies are where
        they are.  This is the consequence, and it is the one thing a refit can
        break without breaking anything else: margins, geometries and reference
        energies all stay perfect while the timestep quietly falls by a factor
        of three, because nothing downstream of the fit reads a frequency.
        That is exactly what happened before `DEFAULT_MAX_WAVENUMBER` existed.
        """
        from ase.data import atomic_masses, atomic_numbers

        from DynamicTopology.fit.dissociation import bond_curvatures, total_wavenumber

        worst = (0.0, "")
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            terms = read_jsonl(stem.with_suffix(".jsonl"))
            if not any(term["type"] == "bond" for term in terms):
                continue
            seen: set[tuple[float, float]] = set()
            pairs = []
            for term in terms:
                if term["type"] != "bond":
                    continue
                key = (term["kwargs"]["r0"], term["kwargs"]["k"])
                if key in seen:
                    continue
                seen.add(key)
                i, j = term["atoms"].values()
                pairs.append((frame[i].symbol, frame[j].symbol))
            for (a, b), curvature in zip(pairs, bond_curvatures(frame, terms)):
                mass = [atomic_masses[atomic_numbers[e]] for e in (a, b)]
                value = total_wavenumber(curvature, *mass)
                if value > worst[0]:
                    worst = (value, f"{stem.name} {a}-{b}")

        period = 33356.4 / worst[0]
        assert worst[0] < self.TIMESTEP_WAVENUMBER, (
            f"the fastest stretch is {worst[0]:.0f} cm^-1 ({worst[1]}), a "
            f"{period:.2f} fs period -- {period / 15.0:.3f} fs at 15 steps per "
            "period, against the 0.5 fs the production sweep is configured for. "
            "Refit with a tighter --max-wavenumber, or change the timestep in "
            "production/stoichiometry-3000K/sweep.toml and say why."
        )

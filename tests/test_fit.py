"""Tests for the parameter fits in `DynamicTopology.fit`.

The two fits here are nested and each has one thing it must never give up:

  `fit_dissociation_energies` solves a depth scale so a template's bonded energy
  at its own geometry equals its reference atomization energy.  That equality is
  the *constraint*, and it has to survive the outer fit moving the force
  constants out from under it -- which is exactly the case the D-only fit was
  never exercised at.

  `fit_force_constants` solves force constants so every reaction's reference
  barrier lies below both diabats at its transition state, since otherwise
  `fit_amplitude` has no real root.  It buys that with vibrational frequency, so
  the test that matters most is the vacuity check: with the frequency frozen it
  must report no progress at all.  A fit that improves without freedom is
  reporting something other than what it did.

**Open: this file is slow, and only in company.**  Every test here passes, and
the file on its own finishes.  So does the rest of the suite -- 158 tests in 33
seconds.  Run together they take tens of minutes on an idle machine, which is an
order of magnitude more than the sum of the parts.

Three things it is *not*, each measured rather than reasoned about:

  * Not the search budget.  A `mode="shape"` fit converges on `xtol`/`ftol` at
    1219 evaluations and about fifty seconds -- under its own `maxfev`, so there
    is nothing to cap.  See the comment on the `minimize` call in
    `fit.dissociation.fit_force_constants` for the sweep.
  * Not the cost of an objective evaluation.  A full refit of every template is
    26 ms and a full `reaction_margins` pass is 13 ms, neither depending on the
    shape parameter.
  * Not machine contention.  Confirmed with nothing else running.

That leaves an interaction between this file and the others, and the obvious
suspect is the one piece of shared mutable state: `fit_force_constants` mutates
the `ReactionSet` it is handed -- deliberately, so `scripts/fit.py --dry-run`
can fit couplings against refitted diabats -- and other modules hold their own
`ReactionSet` fixtures.  Unverified.  Look there first.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import io

from DynamicTopology.basis import EVBBasis
from DynamicTopology.core import ReactionSet
from DynamicTopology.fit.coupling import (
    CouplingFitError,
    fit_amplitude,
    fit_coupling,
)
from DynamicTopology.fit.dissociation import (
    DEFAULT_MAX_DECAY,
    DEFAULT_MIN_DECAY,
    bond_curvatures,
    bond_types,
    bonded_energy,
    fit_dissociation_energies,
    fit_force_constants,
    frequency,
    reaction_margins,
    scale_force_constants,
    set_shape_decays,
    set_shape_parameters,
    shape_bound,
    stretch_curvatures,
    total_wavenumber,
    wavenumber,
)
from DynamicTopology.forcefield.coupling import EVBCoupling
from DynamicTopology.forcefield.qforce import SHAPE_DECAY, QForce
from DynamicTopology.io.json import read_jsonl


RSET_PATH = Path("datasets/HCombustion/HCombustion.json").resolve()


def _entries(kind: str) -> list[Path]:
    manifest = json.loads(RSET_PATH.read_text())
    return [RSET_PATH.parent / entry["path"] for entry in manifest[kind]]


@pytest.fixture(scope="module")
def templates() -> list[tuple[str, object, list]]:
    return [
        (
            stem.name,
            io.read(stem.with_suffix(".xyz")),
            read_jsonl(stem.with_suffix(".jsonl")),
        )
        for stem in _entries("molecules")
    ]


@pytest.fixture(scope="module")
def reactions() -> list[tuple[str, list]]:
    return [
        (stem.name, io.read(stem.with_suffix(".xyz"), index=":"))
        for stem in _entries("reactions")
    ]


class TestDissociationFit:
    """The atomization-energy equality, which everything else builds on."""

    def test_reproduces_the_reference_energy(self, templates):
        for name, atoms, terms in templates:
            fitted = fit_dissociation_energies(atoms, terms)
            assert bonded_energy(atoms, fitted) == pytest.approx(
                atoms.get_potential_energy(), abs=1e-9
            ), f"{name} does not carry its own atomization energy after the fit"

    @pytest.mark.parametrize("scale", [0.5, 2.0, 6.0])
    def test_survives_a_force_constant_change(self, templates, scale):
        """The equality is a constraint, not a coincidence of the shipped `k`.

        `fit_force_constants` re-solves the depth underneath every force
        constant it tries, so this has to hold far from `scale = 1` or the outer
        fit is trading away the atomization energy without saying so.
        """
        for name, atoms, terms in templates:
            stiffened = scale_force_constants(terms, [scale] * len(bond_types(terms)))
            fitted = fit_dissociation_energies(atoms, stiffened)
            assert bonded_energy(atoms, fitted) == pytest.approx(
                atoms.get_potential_energy(), abs=1e-9
            ), f"{name} lost its atomization energy at k-scale {scale}"

    def test_scaling_leaves_the_force_constants_alone(self, templates):
        """`scale_force_constants` is the outer variable and touches only `k`."""
        for _, _, terms in templates:
            scaled = scale_force_constants(terms, [3.0] * len(bond_types(terms)))
            for before, after in zip(terms, scaled):
                assert before["type"] == after["type"]
                if before["type"] != "bond":
                    assert before["kwargs"] == after["kwargs"]
                    continue
                assert after["kwargs"]["k"] == pytest.approx(
                    3.0 * before["kwargs"]["k"]
                )
                assert after["kwargs"]["D"] == before["kwargs"]["D"]
                assert after["kwargs"]["r0"] == before["kwargs"]["r0"]

    def test_the_fit_writes_its_own_zero_shift(self, templates):
        """The fitted depths must come with an explicit `reference` of zero.

        `ReactionSet.load` synthesizes `E0 = E_atomization + sum(D)` for any
        template that does not state a shift.  Once the depths have been solved
        against the *full* bonded energy that formula no longer evaluates to
        zero -- what is left over is the angle and cross-term contribution at
        the reference geometry -- so a template that stays silent gets that
        leftover added a second time.  It is only a few hundredths of an eV, and
        it is enough to move a marginal channel across the feasibility line.
        """
        for name, atoms, terms in templates:
            if not any(t["type"] == "bond" for t in terms):
                continue  # a free atom has no depth to solve and no shift to state
            fitted = fit_dissociation_energies(atoms, terms)
            shifts = [t for t in fitted if t["type"] == "reference"]
            assert len(shifts) == 1, f"{name} states no reference shift"
            assert shifts[0]["kwargs"]["E0"] == 0.0

            synthesized = ReactionSet._reference_term(atoms, fitted)
            assert synthesized is not None
            assert abs(synthesized["kwargs"]["E0"]) > 1e-6, (
                f"{name}'s synthesized shift is zero anyway, so this test is not "
                "guarding anything"
            )


class TestForceConstantFit:
    """The `k` route: raise the diabats by stiffening the bonds.

    Superseded as the default by the Morse shape parameter -- which reaches more
    channels for no frequency at all -- but kept, and kept tested, because it is
    the thing the shape term has to be measured against.  Every test here names
    `mode="k"` explicitly rather than relying on the default, so that changing
    the default cannot quietly turn these into tests of something else.
    """

    def test_frozen_force_constants_make_no_progress(self, templates, reactions):
        """Vacuity check.  No freedom must mean no improvement, and say so.

        `max_scale = 1.0` collapses the box to a point.  If the margins move
        anyway, the reported "before" and "after" are not measuring the same
        thing and every other number this fit prints is suspect.
        """
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(
            reaction_set, templates, reactions, mode="k", max_scale=1.0
        )

        assert fit.scales == [1.0] * len(fit.variables)
        for name, after in fit.margins.items():
            assert after == pytest.approx(fit.margins_before[name], abs=1e-12)

    def test_refitting_recovers_from_softened_bonds(self, templates, reactions):
        """Degrade the force constants, then require the fit to win them back.

        Deliberately not asserted against the shipped parameters: once the
        dataset is regenerated with fitted force constants there is nothing left
        for this to improve, and a test that reads "the dataset is bad" stops
        testing the fit the moment the dataset stops being bad.  Softening every
        bond deepens every Morse curve, which is precisely the failure mode, so
        this is the same problem posed from a known-worse starting point.
        """
        softened = [
            (name, atoms, scale_force_constants(terms, [0.4] * len(bond_types(terms))))
            for name, atoms, terms in templates
        ]
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(reaction_set, softened, reactions, mode="k")

        before = sum(value > 0.0 for value in fit.margins_before.values())
        after = sum(value > 0.0 for value in fit.margins.values())
        assert after > before, (
            f"{before} channels were fittable before the refit and {after} after; "
            "the fit is not buying what it costs in frequency"
        )

        drift = max(max(s, 1.0 / s) for s in fit.scales) ** 0.5
        assert drift > 1.2, (
            "the refit gained channels without moving any force constant, so "
            "whatever produced the gain, it was not the variable being reported"
        )

        # And it kept the constraint it was solved under.
        for (name, atoms, _), terms in zip(templates, fit.terms.values()):
            assert bonded_energy(atoms, terms) == pytest.approx(
                atoms.get_potential_energy(), abs=1e-9
            ), f"{name} lost its atomization energy to the force-constant fit"

    def test_refitting_never_loses_ground(self, templates, reactions):
        """On whatever the dataset currently is, the fit may not make it worse.

        The starting point is inside the search box, so a fit that returns fewer
        feasible channels than it started with has converged to something it
        should have rejected.
        """
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(reaction_set, templates, reactions, mode="k")
        before = sum(value > 0.0 for value in fit.margins_before.values())
        after = sum(value > 0.0 for value in fit.margins.values())
        assert after >= before

    def test_capping_the_scale_bounds_the_drift(self, templates, reactions):
        """`max_scale` is the hard half of the trade and must actually bind."""
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(
            reaction_set,
            templates,
            reactions,
            mode="k",
            max_scale=2.0,
            frequency_weight=0.0,
        )
        assert max(fit.scales) <= 2.0 + 1e-9
        assert min(fit.scales) >= 0.5 - 1e-9

    def test_the_fit_leaves_the_reaction_set_refitted(self, templates, reactions):
        """`fit_force_constants` mutates its `ReactionSet` so callers can reuse it.

        `scripts/fit.py --force-constants --dry-run` depends on this: it fits
        couplings against the refitted diabats without writing anything, which is
        what makes a parameter sweep possible at all.

        `mode="k"` rather than the default, and for a runtime reason worth
        writing down.  What this asserts -- that the mutation happened -- is the
        same in every mode, but `mode="both"` searches twice as many variables,
        and once the templates on disk already satisfy every margin the hinge is
        flat and Powell spends its whole 200-iteration budget trading the
        regularizer against the geometry penalty for nothing.  Measured: 22
        seconds in `k`, over an hour in `both`, with the same assertion passing.
        The offline `scripts/fit.py` can afford that; a test cannot.
        """
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(reaction_set, templates, reactions, mode="k")
        assert reaction_margins(reaction_set, reactions) == pytest.approx(fit.margins)

    def test_frequency_is_a_wavenumber(self):
        """Sanity on the unit conversion the whole report is quoted in."""
        # q-force's O-H in water, whose stretch is near 3700 cm^-1.
        assert frequency(449613.167, 15.999, 1.008) == pytest.approx(3656, abs=10)
        # H2, near 3750 cm^-1 for this force constant.
        assert frequency(251200.743, 1.008, 1.008) == pytest.approx(3748, abs=10)


class TestDecoupledChannels:
    """What a channel gets when its barrier still cannot be inverted."""

    def test_zero_amplitude_is_labelled_decoupled(self, reactions):
        _, frames = reactions[0]
        terms = fit_coupling(frames, amplitude=0.0)

        assert terms[0]["kwargs"]["A"] == 0.0
        assert terms[0]["provenance"] == "decoupled"
        assert np.isfinite(terms[0]["kwargs"]["a"]), (
            "a zero amplitude has no width condition, but the stored width still "
            "has to be a number someone could fill an amplitude in against"
        )

    def test_a_nonzero_stand_in_is_still_a_placeholder(self, reactions):
        _, frames = reactions[0]
        terms = fit_coupling(frames, amplitude=-10.0)
        assert terms[0]["provenance"] == "placeholder"

    def test_an_uninvertible_barrier_has_no_amplitude(self):
        """The condition the decoupled path exists for."""
        with pytest.raises(CouplingFitError):
            # Barrier above the lower diabat: the EVB ground state is below both
            # by construction, so no real coupling reaches it.
            fit_amplitude(-2.0, -1.0, -0.5)

    def test_a_decoupled_channel_never_enters_a_basis(self):
        """A = 0 gives no stabilization, so the gate can never open on it.

        This is what makes decoupling the honest fallback rather than a small
        placeholder: it is not a weak coupling that might still admit a state at
        some geometry, it is no coupling at any geometry.
        """
        basis = EVBBasis(ReactionSet(RSET_PATH), QForce(), EVBCoupling())
        # `_switch` takes the two diabatic energies and the coupling.  A gap this
        # wide with a real coupling still admits; with zero it cannot.
        assert basis._switch(-10.0, -9.0, 0.5) > 0.0
        assert basis._switch(-10.0, -9.0, 0.0) == 0.0
        # Nor at degeneracy, where the stabilization is |V| itself.
        assert basis._switch(-10.0, -10.0, 0.0) == 0.0


class TestMorseShape:
    """The Hulburt-Hirschfelder term `c`, and the two things it must not break.

    `D`, `r0` and the curvature *at `r0`* come through untouched.  That is an
    exact algebraic claim about an `O(dr**3)` correction, so it is testable
    exactly rather than to a tolerance someone picked, and the tests below
    evaluate it at `r0` for that reason.

    **What this class does not say, and used to imply.**  It said `c` "is free"
    and that the frequency comes through untouched, full stop.  That reading
    held only while `r0` was where the bond sat.  `fit_bond_lengths` now
    displaces `r0` inside the reference bond length so the Morse can lean
    against the repulsion, and away from `r0` this term is the largest single
    contribution to the stiffness of most of HCombustion's bonds.  See
    `TestStretchCurvature` below, which measures it, and
    `fit.dissociation._bonded_curvature`.
    """

    D, R0, K = 436.0, 0.07772, 251200.0

    def _curve(self, r, c, b=SHAPE_DECAY):
        positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        vectors = positions[None, :, :] - positions[:, None, :]
        return QForce(bond_form="morse")._bond_morse(
            vectors,
            np.array([[0, 1]]),
            np.array([self.D]),
            np.array([self.R0]),
            np.array([self.K]),
            np.array([c]),
            np.array([b]),
        )[0]

    @pytest.mark.parametrize("c", [0.0, 0.5, 1.3, -0.8])
    @pytest.mark.parametrize("b", [2.0, SHAPE_DECAY, 8.0])
    def test_the_shape_term_moves_neither_the_well_nor_the_limit(self, c, b):
        assert self._curve(self.R0, c, b) == pytest.approx(-self.D, abs=1e-9)
        assert self._curve(2.0, c, b) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("c", [0.0, 0.5, 1.3, -0.8])
    def test_the_shape_term_moves_no_frequency(self, c):
        """The curvature at the minimum is `k`, whatever `c` is.

        Checked numerically rather than symbolically because it is the
        *implementation* that has to have this property, not the algebra.
        """
        h = 1e-6
        curvature = (
            self._curve(self.R0 + h, c)
            - 2.0 * self._curve(self.R0, c)
            + self._curve(self.R0 - h, c)
        ) / h**2
        assert curvature == pytest.approx(self.K, rel=1e-4)

    def test_the_compressed_branch_is_untouched(self):
        """`c` is clamped off below `r0`, where it would run to minus infinity.

        `s**3 exp(-2s)` continued to negative `s` grows faster than the Morse
        repulsion, so an unclamped correction turns the wall over and an atom
        pushed hard enough falls through.
        """
        for r in (0.03, 0.05, 0.07, self.R0):
            assert self._curve(r, 1.3) == pytest.approx(self._curve(r, 0.0), abs=1e-12)

    @pytest.mark.parametrize(
        "b", [DEFAULT_MIN_DECAY, 2.0, 2.5, 3.0, 4.0, 6.0, DEFAULT_MAX_DECAY]
    )
    def test_the_bound_is_where_dissociation_stops_being_downhill(self, b):
        """`shape_bound(b)` must bind exactly, in both directions, at every `b`.

        Below it the curve rises monotonically to the dissociation limit; above
        it a barrier appears on a channel that has none, and a bound state
        beyond the barrier that would trap fragments that should separate.  A
        bound that is merely *safe* would pass the first half of this and make
        the second half unreachable, so both are asserted.

        Swept over the whole fitted range rather than checked at one decay,
        because `b` is now a fitted parameter and the bound moves with it by a
        factor of five hundred -- 0.180 at b = 1.5, 92.49 at b = 8.  A bound
        that were correct only at the old fixed b = 4 would let the fit walk
        onto a non-monotone curve anywhere else it went.
        """
        radii = self.R0 + np.linspace(1e-4, 1.2, 20000)

        def monotonic(c):
            energies = np.array([self._curve(r, c, b) for r in radii])
            return bool(np.all(np.diff(energies) > -1e-12))

        bound = shape_bound(b)
        assert monotonic(bound), (
            f"c = {bound} at b = {b} already puts a barrier on a dissociation "
            "curve, so the bound is not protecting what it claims to"
        )
        assert not monotonic(1.05 * bound), (
            f"c = {1.05 * bound} at b = {b} still dissociates downhill, so "
            f"{bound} is leaving usable freedom on the table and this test is "
            "not measuring where the limit actually is"
        )

    def test_the_decay_bounds_are_where_the_parameter_stops_meaning_anything(self):
        """Both ends of the `b` box are the edge of usefulness, not a taste.

        At the bottom the monotonicity bound collapses -- it is exactly zero at
        b = 1, so no positive `c` is admissible at all -- and at the top the
        correction has retreated inside the bond, delivering less at its own
        peak than it does in the middle of the range.
        """
        assert shape_bound(1.0) < 1e-3, (
            "the bound has not collapsed by b = 1, so the low end of the box is "
            "not where the parameter stops existing"
        )
        assert shape_bound(DEFAULT_MIN_DECAY) > 0.0

        def delivered(b):
            """Largest correction available at this decay, as a fraction of D."""
            return shape_bound(b) * (3.0 / b) ** 3 * np.exp(-3.0)

        assert delivered(DEFAULT_MIN_DECAY) < 0.10, (
            "the low end of the box still delivers a usable correction, so it "
            "is cutting off freedom the fit could have used"
        )
        assert delivered(DEFAULT_MAX_DECAY) < delivered(4.0), (
            "the correction is still growing at the top of the box, so the "
            "bound is not where the parameter stops paying"
        )

    def test_setting_the_decay_leaves_every_other_parameter_alone(self, templates):
        for _, _, terms in templates:
            decayed = set_shape_decays(terms, [2.5] * len(bond_types(terms)))
            for before, after in zip(terms, decayed):
                assert before["type"] == after["type"]
                if before["type"] != "bond":
                    assert before["kwargs"] == after["kwargs"]
                    continue
                assert after["kwargs"]["b"] == pytest.approx(2.5)
                for key in ("D", "r0", "k"):
                    assert after["kwargs"][key] == before["kwargs"][key]

    def test_a_missing_decay_reads_as_the_documented_default(self):
        """Term files predating `b` must be unchanged by its introduction."""
        for r in (self.R0 + 0.02, self.R0 + 0.05, self.R0 + 0.12):
            assert self._curve(r, 1.3) == pytest.approx(
                self._curve(r, 1.3, SHAPE_DECAY), abs=1e-12
            )

    def test_setting_the_shape_leaves_every_other_parameter_alone(self, templates):
        for _, _, terms in templates:
            shaped = set_shape_parameters(terms, [0.7] * len(bond_types(terms)))
            for before, after in zip(terms, shaped):
                assert before["type"] == after["type"]
                if before["type"] != "bond":
                    assert before["kwargs"] == after["kwargs"]
                    continue
                assert after["kwargs"]["c"] == pytest.approx(0.7)
                for key in ("D", "r0", "k"):
                    assert after["kwargs"][key] == before["kwargs"][key]

    def test_a_frozen_shape_makes_no_progress(self, templates, reactions):
        """Vacuity check, the shape-mode twin of the force-constant one."""
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(
            reaction_set, templates, reactions, mode="shape", max_shape=0.0
        )
        assert fit.scales == [0.0] * len(fit.variables)
        for name, after in fit.margins.items():
            assert after == pytest.approx(fit.margins_before[name], abs=1e-12)

    def test_the_shape_alone_buys_channels_at_no_frequency_cost(
        self, templates, reactions
    ):
        """The claim the whole third parameter rests on.

        Starting from plain Morse -- which is what `c = 0` is, and what every
        template ships as before this fit runs -- the shape term has to gain
        feasible channels while leaving every force constant exactly alone.
        """
        plain = [
            (name, atoms, set_shape_parameters(terms, [0.0] * len(bond_types(terms))))
            for name, atoms, terms in templates
        ]
        reaction_set = ReactionSet(RSET_PATH)
        fit = fit_force_constants(reaction_set, plain, reactions, mode="shape")

        before = sum(value > 0.0 for value in fit.margins_before.values())
        after = sum(value > 0.0 for value in fit.margins.values())
        assert after > before, (
            f"{before} channels were feasible with plain Morse and {after} with "
            "the shape term; it is not buying anything"
        )
        for (_, _, original), terms in zip(plain, fit.terms.values()):
            for a, b in zip(original, terms):
                if a["type"] == "bond":
                    assert a["kwargs"]["k"] == b["kwargs"]["k"], (
                        "the shape fit moved a force constant, which is the one "
                        "thing it exists to avoid"
                    )


class TestStretchCurvature:
    """The stiffness the integrator sees, and the cheap stand-in for it.

    The timestep follows from the fastest mode and nothing else, so the fit has
    to be able to price a wavenumber -- which means computing one inside an
    objective that runs to five figures of evaluations.  `bond_curvatures` is
    the honest measurement and costs two assembled force calls per bond;
    `stretch_curvatures` is the analytic-plus-cached stand-in the objective
    actually calls.  These hold them against each other.
    """

    # Largest disagreement, in eV/A**2, permitted on a bond whose displaced atom
    # carries no other bond.  Those rows agree to about 0.04 -- the residual is
    # the finite-difference step in `bond_curvatures`, not a modelling gap -- so
    # 0.5 is an order of magnitude of headroom and still an order of magnitude
    # under the one row that genuinely differs.
    SIMPLE_TOLERANCE = 0.5

    def test_the_cheap_curvature_matches_the_measured_one(self, templates):
        """Everywhere the stand-in claims to be exact, it is.

        Excludes the one bond type it does not model: `bond_curvatures`
        displaces atom `j` along the `ij` axis and so picks up every term that
        touches `j`, while `stretch_curvatures` takes the bond's own Morse and
        the nonbonded terms and stops.  H2O2's O-O is the only bond in
        HCombustion whose displaced atom carries both another bond and a
        dihedral, and it is checked separately below rather than waved through
        by a tolerance wide enough to hide it.
        """
        errors = {}
        for name, atoms, terms in templates:
            measured = bond_curvatures(atoms, terms)
            cheap = stretch_curvatures(atoms, terms)
            assert len(measured) == len(cheap)
            for index, (a, b) in enumerate(zip(measured, cheap)):
                if (name, index) == ("mol_06", 0):  # H2O2 O-O; see below
                    continue
                if abs(a - b) > self.SIMPLE_TOLERANCE:
                    errors[f"{name}[{index}]"] = (
                        f"bond_curvatures {a:.4f}, stretch_curvatures {b:.4f}, "
                        f"difference {a - b:+.4f} eV/A**2"
                    )
        assert not errors, (
            "the objective's curvature no longer matches the measured one:\n  "
            + "\n  ".join(f"{k}: {v}" for k, v in sorted(errors.items()))
        )

    def test_the_one_bond_the_stand_in_does_not_model_is_the_expected_one(
        self, templates
    ):
        """Anti-vacuity, and it guards a real limit rather than a number.

        If some refactor made `stretch_curvatures` pick up the neighbouring
        bond and dihedral terms as well, the test above would keep passing while
        this one failed -- and that would be worth knowing, because it would
        mean the cheap path had quietly become the expensive one.  Asserted as a
        band rather than a value: the gap is ~22 eV/A**2 on the shipped
        parameters, of which ~16 is the O-H Morse on the moved oxygen and ~6 the
        angle and dihedral terms, and it moves when the fit moves.
        """
        name, atoms, terms = next(t for t in templates if t[0] == "mol_06")
        gap = bond_curvatures(atoms, terms)[0] - stretch_curvatures(atoms, terms)[0]
        assert 1.0 < gap < 100.0, (
            f"H2O2's O-O stand-in is off by {gap:+.3f} eV/A**2. It is supposed "
            "to be off, by roughly 22 -- the terms on the displaced oxygen that "
            "`stretch_curvatures` does not model. A gap near zero means it now "
            "models them and is no longer the cheap path; a large one means "
            "something else changed."
        )

    # How far the stand-in may be from the measured curvature on the one row
    # where they differ, in cm^-1.  The cap is applied to the stand-in, so this
    # is the amount by which the fit can be wrong about where the cap binds.
    CAP_ERROR = 200.0

    def test_the_cap_is_applied_to_a_number_close_enough_to_the_truth(self, templates):
        """H2O2's O-O is now near the cap, so the error there has to be bounded.

        **This assertion was re-derived when `forcefield/lj.py` came back.**  It
        used to require H2O2's O-O to stay under 3500 cm^-1, on the argument that
        the cap binds only on X-H stretches -- the rows the stand-in reproduces
        to 1e-3 -- while the one inexact row was a slow heavy-atom mode nowhere
        near any cap.  The 12-6 refit ended that: the 12-6 is 0.30 eV at H2O2's
        1.45 A O-O and steeply varying, the fit answered by stiffening the bond,
        and the mode went **3228 -> 4285 cm^-1**.  It is now the second-fastest
        in the dataset and the cap does bind on it.

        So the old argument is gone and this asserts what actually has to be
        true instead: not that the inexact row is far from the cap, but that the
        inexact row is not very inexact.  Measured on the shipped parameters,

            bond_curvatures     540.200 eV/A**2   4285.2 cm^-1
            stretch_curvatures  518.653 eV/A**2   4198.9 cm^-1
                                                    86.3 cm^-1, 2.0%

        so `--max-wavenumber 4200` is being enforced against a number 86 cm^-1
        low, and the timestep that follows is 0.529 fs where the truth is
        0.515 fs.  `production/*/sweep.toml` runs at 0.5 fs, which covers both,
        and that margin is the reason this is recorded rather than fixed by
        putting `bond_curvatures` in the objective -- it costs a hundred times
        more per evaluation.  If this ever fails, that trade has stopped paying.
        """
        for name, atoms, terms in templates:
            if name != "mol_06":
                continue
            cheap = total_wavenumber(
                stretch_curvatures(atoms, terms)[0], 15.999, 15.999
            )
            measured = total_wavenumber(
                bond_curvatures(atoms, terms)[0], 15.999, 15.999
            )
            assert abs(measured - cheap) < self.CAP_ERROR, (
                f"H2O2's O-O reads {cheap:.1f} cm^-1 through the objective's "
                f"stand-in and {measured:.1f} cm^-1 measured, a gap of "
                f"{measured - cheap:+.1f}. The cap binds on this mode now, so "
                "the fit is enforcing it against the wrong number by that much. "
                "Either put `bond_curvatures` in the objective or re-derive the "
                "approximation."
            )

    def test_the_two_wavenumber_entry_points_agree(self):
        """`wavenumber` takes the reduction the objective precomputed."""
        for mass_a, mass_b, curvature in [
            (1.008, 1.008, 33.89),
            (15.999, 1.008, 68.48),
            (15.999, 15.999, 139.72),
        ]:
            reduced = mass_a * mass_b / (mass_a + mass_b)
            assert wavenumber(curvature, reduced) == pytest.approx(
                total_wavenumber(curvature, mass_a, mass_b), rel=1e-12
            )

    def test_a_bond_that_is_not_at_a_minimum_has_no_wavenumber(self):
        """Negative curvature must not reach `sqrt`.

        The objective evaluates this at every point Powell tries, including ones
        where a bond type's total curvature has gone negative.  A NaN there
        propagates into the objective and Powell follows it somewhere arbitrary,
        which is a failure mode with no symptom other than a bad answer.
        """
        assert wavenumber(-10.0, 1.0) == 0.0
        assert total_wavenumber(0.0, 1.008, 1.008) == 0.0

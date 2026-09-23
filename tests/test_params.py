"""Do the global force field parameters reach the force field, and only from one place?

`forcefield/params.py` moved eleven constants out of four modules and into the
dataset manifest.  Two things can go wrong with that, and neither is visible to
any other test in this suite:

  * a **stale default**.  A call site that still reads the old constant, or that
    binds the active set into a default argument at import time, evaluates at
    the defaults however the manifest is written.  Every dataset in the
    repository pins the defaults, so such a site agrees with the manifest by
    coincidence and the disagreement only appears on someone else's dataset.
    The tests below therefore override each parameter to a value the defaults do
    *not* contain and demand that the number the force field reports moves.

  * a **silent collision**.  Two datasets fitted at different parameters cannot
    share one process; whichever loaded second would score the first's templates
    on a surface they were not fitted to, and the symptom -- a template no
    longer reproducing its own reference energy -- looks exactly like a bad fit.
    `activate` refuses, and that refusal is checked here.

The overrides are deliberately large.  A 0.1 A nudge to a taper radius is worth
millivolts and would pass a sloppy threshold; 1.5 -> 2.5 A is worth electron
volts on a bond, so a site that ignored it cannot hide in the tolerance.
"""

from pathlib import Path

import numpy as np
import pytest

from DynamicTopology.core import ReactionSet
from DynamicTopology.forcefield import lj as lj_module
from DynamicTopology.forcefield import zbl as zbl_module
from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.exclusions import exclusion_terms
from DynamicTopology.forcefield.params import (
    DEFAULTS,
    ForceFieldParams,
    activate,
    active,
    active_source,
    reset,
    use,
)
from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.forcefield.zbl import ZBL


HCOMBUSTION = Path("datasets/HCombustion/HCombustion.json").resolve()
WATER = Path("datasets/Water/Water.json").resolve()

PBC = np.array([False, False, False])
CELL = np.eye(3) * 20.0


@pytest.fixture(autouse=True)
def _isolate_active():
    """A clean slate in, and whatever the session had back out again.

    Both halves are load-bearing.  **Out**, because `activate` refuses to
    contradict what is in force: a test that left an override behind would not
    merely perturb the next one, it would make every subsequent `ReactionSet`
    load in the session raise.  **In**, because the reverse is also true -- any
    earlier test module that loaded a dataset has left that dataset's parameters
    active and attributed, and the tests below activate their own; run after
    `tests/test_dissociation.py` they would collide with HCombustion rather than with
    each other, and pass or fail depending on collection order.
    """
    before = active()
    before_source = active_source()
    reset()
    yield
    reset()
    if before != DEFAULTS or before_source is not None:
        activate(before, source=before_source)


class TestTheDefaults:
    """The numbers the datasets in this repository were fitted at.

    Not a tautology: `.jsonl` files already on disk were solved against these
    exact values, so a default that drifts invalidates them.  Written out
    literally rather than derived, so that moving one is a decision with a
    failing test attached rather than an edit nobody notices.
    """

    def test_values(self):
        assert DEFAULTS.bond_asymptote == 1.0
        assert DEFAULTS.taper_radius == 1.5
        assert DEFAULTS.taper_width == 0.12
        assert DEFAULTS.switch_radius == 2.2
        assert DEFAULTS.core_fraction == 0.4
        assert DEFAULTS.exclusion_depth == 3
        assert DEFAULTS.exclude_coulomb is True
        assert DEFAULTS.gamma == 2.0
        assert DEFAULTS.accuracy == 1e-8
        assert DEFAULTS.ccoul == 14.4
        assert DEFAULTS.zbl_ccoul == 14.399645
        assert DEFAULTS.electrostatics == "acks2"

    def test_the_switch_width_follows_the_taper_width(self):
        """One sharpness for both switches, so one number sets both.

        A dataset that sharpens ZBL's taper and forgets the 12-6's switch would
        otherwise get two switches of different sharpness without asking for
        them.  (It was `taper_width / 10` while the 12-6 was evaluated in nm --
        the same width in the other unit.)
        """
        assert DEFAULTS.switch_width == pytest.approx(0.12)
        assert ForceFieldParams(taper_width=0.3).switch_width == pytest.approx(0.3)
        # Stated explicitly, the derivation is out of the way.
        explicit = ForceFieldParams(taper_width=0.3, switch_width=0.005)
        assert explicit.switch_width == pytest.approx(0.005)

    def test_every_field_round_trips_through_a_manifest(self):
        """`to_dict` states what a manifest would need to pin this object exactly.

        Including `switch_width`, which is `None` on the way in and a float on
        the way out -- a production run recording its surface must record the
        resolved value, not the sentinel.
        """
        params = ForceFieldParams(taper_radius=1.7, taper_width=0.2)
        assert ForceFieldParams.from_dict(params.to_dict()) == params
        assert params.to_dict()["switch_width"] == pytest.approx(0.2)


class TestTheManifest:
    def test_a_dataset_carries_its_own_parameters(self):
        rset = ReactionSet(HCOMBUSTION)
        assert rset.params.taper_radius == 1.5
        assert rset.params.exclusion_depth == 3
        assert active() == rset.params
        assert active_source() == str(HCOMBUSTION)

    def test_both_datasets_pin_the_values_their_jsonl_was_fitted_at(self):
        """The two manifests must agree, or the suite cannot load both.

        Not a style rule: the fit (fast-forces' `refine`) solved every `.jsonl` in both
        datasets against `E_bonded + E_nonbonded`, and every constant below sits
        inside that sum.  Two datasets disagreeing about one of them is two
        different force fields.
        """
        assert ReactionSet(HCOMBUSTION).params == ReactionSet(WATER).params

    def test_an_omitted_key_takes_the_default(self, tmp_path):
        manifest = tmp_path / "Empty.json"
        manifest.write_text('{"molecules": [], "reactions": []}')
        assert ReactionSet(manifest).params == DEFAULTS

    def test_a_partial_block_leaves_the_rest_alone(self):
        params = ForceFieldParams.from_dict({"taper_radius": 1.9})
        assert params.taper_radius == 1.9
        assert params.gamma == DEFAULTS.gamma
        assert params.exclusion_depth == DEFAULTS.exclusion_depth

    def test_an_unknown_key_is_an_error(self):
        """A typo must not leave the default quietly in force.

        This is the same failure the module exists to close, arriving by a
        different door: a manifest that claims a radius the force field never
        saw, and a dataset whose terms were fitted at neither.
        """
        with pytest.raises(ValueError, match="unknown `global_params` key"):
            ForceFieldParams.from_dict({"taper_radius": 1.9, "tapper_width": 0.2})

    def test_a_non_object_block_is_an_error(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            ForceFieldParams.from_dict([1.5])

    def test_an_unknown_electrostatics_is_an_error(self):
        """A misspelt term name must not fall back to ACKS2."""
        with pytest.raises(ValueError, match="electrostatics must be one of"):
            ForceFieldParams.from_dict({"electrostatics": "point_charge"})

    @pytest.mark.parametrize("field", ["taper_width", "gamma", "accuracy"])
    def test_a_nonpositive_width_is_an_error(self, field):
        """Each of these divides something; zero is a nan, not a soft dataset."""
        with pytest.raises(ValueError, match=field):
            ForceFieldParams.from_dict({field: 0.0})


class TestActivation:
    def test_reloading_one_manifest_is_free(self):
        """The suite builds a `ReactionSet` per test; that must not be a conflict."""
        first = ReactionSet(HCOMBUSTION)
        second = ReactionSet(HCOMBUSTION)
        assert first.params == second.params

    def test_two_disagreeing_datasets_refuse_to_share_a_process(self):
        activate(ForceFieldParams(taper_radius=1.9), source="first.json")
        with pytest.raises(ValueError, match="already active"):
            activate(ForceFieldParams(taper_radius=2.1), source="second.json")

    def test_the_refusal_names_the_fields_that_disagree(self):
        """A collision has to be diagnosable, or it is read as a bad fit.

        The message is what someone sees when a template stops reproducing its
        own reference energy, so it has to say which constant moved rather than
        only that something did.
        """
        activate(ForceFieldParams(taper_radius=1.9, gamma=3.0), source="first.json")
        with pytest.raises(ValueError) as error:
            activate(
                ForceFieldParams(taper_radius=2.1, gamma=3.0), source="second.json"
            )
        assert "taper_radius" in str(error.value)
        assert "1.9" in str(error.value) and "2.1" in str(error.value)
        assert "gamma" not in str(error.value)

    def test_use_never_refuses_and_always_restores(self):
        """`use` is the explicit form: the caller has said which surface it means."""
        activate(ForceFieldParams(taper_radius=1.9), source="first.json")
        with use(taper_radius=2.5) as params:
            assert params.taper_radius == 2.5
            assert active().taper_radius == 2.5
        assert active().taper_radius == 1.9
        assert active_source() == "first.json"

    def test_an_override_composes_with_the_dataset(self):
        """`use(**overrides)` derives from what is active, not from the defaults.

        Otherwise nudging one constant of a dataset silently reverts the other
        ten, which is a different surface than the caller asked for.
        """
        activate(ForceFieldParams(taper_radius=1.9, gamma=3.0), source="first.json")
        with use(taper_radius=2.5):
            assert active().gamma == 3.0

    def test_overriding_the_taper_width_rederives_the_switch_width(self):
        """The coupled default must survive `use`, not just construction.

        `switch_width` is a resolved float on any live object, so a naive
        `replace` would carry the *previous* taper width's derivation forward and
        leave the two switches at different sharpnesses -- exactly what the
        default exists to prevent, arriving through the override path.
        """
        with use(taper_width=0.5) as params:
            assert params.switch_width == pytest.approx(0.5)
        # Stated together, the caller gets what it asked for.
        with use(taper_width=0.5, switch_width=0.001) as params:
            assert params.switch_width == pytest.approx(0.001)

    def test_use_restores_after_an_exception(self):
        activate(ForceFieldParams(taper_radius=1.9), source="first.json")
        with pytest.raises(RuntimeError):
            with use(taper_radius=2.5):
                raise RuntimeError("boom")
        assert active().taper_radius == 1.9


class TestTheParametersReachTheForceField:
    """Override each one and demand the reported number moves.

    A call site that still reads a module constant, or that bound the active set
    into a default argument at import time, passes every other test in the suite
    and fails here.
    """

    def test_the_taper_radius_reaches_zbl(self):
        """Reaching ZBL further out is worth electron volts, not millivolts.

        The measurement in `params.ForceFieldParams.taper_radius` is that every
        0.1 A of extra reach costs ~0.2 eV on a hydrogen bond, so a pair at
        1.8 A -- outside the default taper, inside a 2.5 A one -- is the clean
        probe.
        """
        pos = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])
        numbers = np.array([8, 1])
        near = ZBL()(pos, numbers, PBC, CELL)[0]
        with use(taper_radius=2.5):
            far = ZBL()(pos, numbers, PBC, CELL)[0]
        assert far > near + 0.5

    def test_the_taper_width_reaches_zbl(self):
        """The width alone, at fixed radius: right at the radius `f` is 1/2 either
        way, so the probe has to sit off it."""
        r = np.array([1.7])
        z = np.array([8.0])
        sharp = zbl_module.taper(r)[0][0]
        with use(taper_width=0.5):
            soft = zbl_module.taper(r)[0][0]
        assert soft > sharp
        assert (
            zbl_module.pair_potential(r, z, z)[0][0]
            < (
                zbl_module.pair_potential(r, z, z, ForceFieldParams(taper_width=0.5))[
                    0
                ][0]
            )
        )

    def test_the_switch_radius_reaches_the_12_6(self):
        """Moving the switch out turns the 12-6 off where it was on.

        2.4 A is outside the default switch and inside a 4 A one, and it is one
        of the contacts the term exists to supply -- 0.220 eV of wall by the
        table in `params.ForceFieldParams.switch_radius`.
        """
        r = np.array([2.4])
        sigma = np.array([3.0])
        eps = np.array([0.5])
        on = lj_module.pair_potential(r, sigma, eps)[0][0]
        with use(switch_radius=4.0):
            off = lj_module.pair_potential(r, sigma, eps)[0][0]
        assert abs(on) > 1e-3
        assert abs(off) < abs(on) / 100.0

    def test_the_core_fraction_reaches_the_12_6(self):
        """Where `r**-12` hands over to its own tangent.

        Inside the default core the potential is linear; raising the fraction
        moves the hand-over out, so a separation that was on the `r**-12` branch
        is now on the tangent and reads lower.
        """
        r = np.array([1.5])
        sigma = np.array([3.0])
        eps = np.array([0.5])
        with use(switch_radius=0.0, core_fraction=0.4):
            default = lj_module.pair_potential(r, sigma, eps)[0][0]
        with use(switch_radius=0.0, core_fraction=0.9):
            wide = lj_module.pair_potential(r, sigma, eps)[0][0]
        assert wide != pytest.approx(default)

    def test_the_bond_asymptote_reaches_the_morse(self):
        """The dissociated limit of a stretched bond is the asymptote itself.

        Asserts the number, not just that it moved: a site reading the asymptote
        in the wrong unit would land 96.5 times off.
        """
        term_dict = {
            "bond": {
                "atoms": np.array([[0, 1]]),
                "kwargs": {
                    "D": np.array([4.52]),
                    "r0": np.array([0.7772]),
                    "k": np.array([26.035]),
                },
            }
        }
        far = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        for asymptote in (1.0, 3.0):
            with use(bond_asymptote=asymptote) as params:
                energy = QForce()(far, PBC, CELL, term_dict)[0]
            assert energy == pytest.approx(params.bond_asymptote, abs=1e-6)

    def test_a_bonds_own_asymptote_overrides_the_global_one(self):
        """A bond carrying `h` dissociates to `h`, whatever `bond_asymptote` is."""
        term_dict = {
            "bond": {
                "atoms": np.array([[0, 1]]),
                "kwargs": {
                    "D": np.array([4.52]),
                    "r0": np.array([0.7772]),
                    "k": np.array([26.035]),
                    "h": np.array([2.5 * DEFAULTS.bond_asymptote]),
                },
            }
        }
        far = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]])
        for asymptote in (1.0, 3.0):
            with use(bond_asymptote=asymptote):
                energy = QForce()(far, PBC, CELL, term_dict)[0]
            assert energy == pytest.approx(2.5 * DEFAULTS.bond_asymptote, abs=1e-6)

    def test_the_smearing_width_reaches_acks2(self):
        """`gamma` sets the contact value of the charge kernel, `2 gamma/sqrt(pi)`.

        So it moves the charges themselves, not only the energy they are scored
        with -- which is why it is inside `E_nonbonded` and invalidates a fit.
        """
        term_dict = {
            "atom": {
                "atoms": np.array([[0], [1]]),
                "kwargs": {
                    "mu": np.array([8.12, 1.88]),
                    "eta": np.array([3.74, 7.28]),
                    "soft_amp": np.array([3.88, 2.10]),
                    "soft_decay": np.array([0.44, 0.27]),
                },
            }
        }
        pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        narrow = ACKS2()(pos, PBC, CELL, term_dict)[0]
        with use(gamma=0.5):
            wide = ACKS2()(pos, PBC, CELL, term_dict)[0]
        assert narrow != pytest.approx(wide)

    def test_the_electrostatics_choice_reaches_the_calculator_and_the_fitter(self):
        """`System`, `EVBSystem` and the fitter all resolve the term per call.

        A template carrying both parameter sets scores differently under each,
        through `forcefield.evaluate` -- the single-topology sum fast-forces fits
        every `.jsonl` against, so a site that ignored the switch would fit a
        dataset to the wrong electrostatics.
        """
        from ase import Atoms

        from DynamicTopology.forcefield.electrostatics import Electrostatics
        from DynamicTopology.forcefield.evaluate import evaluate_term_dict
        from DynamicTopology.forcefield.pointcharge import PointCharge

        term_dict = {
            "atom": {
                "atoms": np.array([[0], [1]]),
                "kwargs": {
                    "mu": np.array([8.12, 1.88]),
                    "eta": np.array([3.74, 7.28]),
                    "soft_amp": np.array([3.88, 2.10]),
                    "soft_decay": np.array([0.44, 0.27]),
                },
            },
            "charge": {
                "atoms": np.array([[0], [1]]),
                "kwargs": {"q": np.array([-1.0, 1.0])},
            },
            "lennardjones": {
                "atoms": np.array([[0], [1]]),
                "kwargs": {"sigma": np.zeros(2), "eps": np.zeros(2)},
            },
        }
        atoms = Atoms("OH", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=CELL)
        electrostatics = Electrostatics()
        assert isinstance(electrostatics.get(), ACKS2)
        equilibrated = evaluate_term_dict(atoms, term_dict).energy
        with use(electrostatics="pointcharge"):
            assert isinstance(electrostatics.get(), PointCharge)
            fixed = evaluate_term_dict(atoms, term_dict).energy
        # ZBL and the 12-6 are common to both, so the whole difference is the
        # electrostatics: an ion pair 2 A apart, -14.4 * erf(4) / 2 = -7.2 eV,
        # against ACKS2's fractional charges.
        pos, pbc, cell = atoms.positions, atoms.pbc, atoms.cell
        assert fixed - equilibrated == pytest.approx(
            PointCharge()(pos, pbc, cell, term_dict)[0]
            - ACKS2()(pos, pbc, cell, term_dict)[0]
        )
        assert PointCharge()(pos, pbc, cell, term_dict)[0] == pytest.approx(
            -7.2, abs=1e-6
        )
        assert abs(fixed - equilibrated) > 1.0

    def test_the_coulomb_constant_reaches_acks2(self):
        assert ACKS2().CCOUL == pytest.approx(14.4)
        with use(ccoul=28.8):
            assert ACKS2().CCOUL == pytest.approx(28.8)

    def test_the_exclusion_depth_reaches_the_derived_terms(self):
        """Depth 0 derives nothing; depth 3 derives the 1-2, 1-3 and 1-4 pairs.

        The exclusions are derived at load rather than shipped, so this is the
        one parameter that changes the *term list* a template carries and not
        only the value a term evaluates to.
        """
        terms = [
            {"type": "atom", "atoms": {"p1": i}, "kwargs": {}} for i in range(4)
        ] + [
            {"type": "bond", "atoms": {"p1": i, "p2": i + 1}, "kwargs": {}}
            for i in range(3)
        ]
        numbers = np.array([1, 6, 6, 1])
        counts = {}
        for depth in (0, 1, 3):
            with use(exclusion_depth=depth):
                counts[depth] = len(exclusion_terms(terms, numbers))
        assert counts[0] == 0
        assert counts[1] < counts[3]

    def test_switching_the_coulomb_exclusion_off_reaches_the_derived_terms(self):
        terms = [
            {"type": "atom", "atoms": {"p1": 0}, "kwargs": {}},
            {"type": "atom", "atoms": {"p1": 1}, "kwargs": {}},
            {"type": "bond", "atoms": {"p1": 0, "p2": 1}, "kwargs": {}},
        ]
        numbers = np.array([1, 8])
        on = [t["type"] for t in exclusion_terms(terms, numbers)]
        with use(exclude_coulomb=False):
            off = [t["type"] for t in exclusion_terms(terms, numbers)]
        assert "coulombexclusion" in on
        assert "coulombexclusion" not in off
        assert "zblexclusion" in off


class TestTheTemplateFollowsItsManifest:
    """The whole point, end to end: a template scored on the surface it was fitted to.

    `ReactionSet.load` activates the manifest's parameters *before* it derives a
    single exclusion, because the derivation and the reference shift both depend
    on them.  If it did so afterwards -- or not at all -- the template would be
    built on one surface and evaluated on another, and the symptom would be a
    reference energy that no longer reproduces.
    """

    def test_a_manifest_depth_of_zero_derives_no_exclusions(self, tmp_path):
        """Written as a real manifest rather than a `use` block, so the ordering
        inside `load` is what is under test and not just the plumbing."""
        source = Path("datasets/Water").resolve()
        dataset = tmp_path / "molecules"
        dataset.mkdir()
        for name in ("h2o.xyz", "h2o.jsonl"):
            (dataset / name).write_bytes((source / "molecules" / name).read_bytes())
        manifest = tmp_path / "Bare.json"
        manifest.write_text(
            '{"molecules": [{"id": 1, "path": "molecules/h2o"}], "reactions": [],'
            ' "global_params": {"exclusion_depth": 0}}'
        )

        rset = ReactionSet(manifest)
        assert rset.params.exclusion_depth == 0
        molecule = next(iter(rset.data.molecules.values()))
        assert not any(term["type"].endswith("exclusion") for term in molecule.terms), (
            "a depth of 0 still derived exclusions; the manifest was read too late"
        )

    def test_the_default_depth_does_derive_them(self, tmp_path):
        """Guard the guard: the assertion above must be one the default fails."""
        source = Path("datasets/Water").resolve()
        dataset = tmp_path / "molecules"
        dataset.mkdir()
        for name in ("h2o.xyz", "h2o.jsonl"):
            (dataset / name).write_bytes((source / "molecules" / name).read_bytes())
        manifest = tmp_path / "Full.json"
        manifest.write_text(
            '{"molecules": [{"id": 1, "path": "molecules/h2o"}], "reactions": []}'
        )

        molecule = next(iter(ReactionSet(manifest).data.molecules.values()))
        assert any(term["type"].endswith("exclusion") for term in molecule.terms)

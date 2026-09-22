"""Barrierless bond fission and recombination.

Six of HCombustion's nineteen channels change exactly one bond in one
direction -- `rxn_05` (H2), `rxn_06` (O2), `rxn_07` (HO), `rxn_08` (H2O),
`rxn_09` (HO2) and `rxn_15` (H2O2).  None of
them has a saddle, and that used to mean none of them could be coupled: the
`fit.coupling.fit_amplitude` route inverts a reference barrier, and for a
barrierless channel its discriminant `(Hbar - E)**2 - dH**2` vanishes
identically, so a perfect diabat gives `A = 0` and every imperfection gives no
real root at all.  `EVBBasis` drops a channel below `eps` before building its
product, so H2, OH and H2O could not come apart at any temperature.

Two things together fix that, and the division of labour between them is what
these tests are about:

  `qforce.BOND_ASYMPTOTE`     puts the dissociated limit of a bonded diabat
                              *above* the free-fragment one, so the two diabats
                              genuinely cross -- at 1.80 A for H2, 1.95 for OH,
                              1.93 for H2O, where each bond's fitted asymptote
                              `h` (not the global floor) puts them.
  `fit.coupling.fit_twobody`  fits a Gaussian in the breaking bond's *length*,
                              centred on that crossing, so there is a real
                              off-diagonal for the ordinary gate to admit.

**There is no special admission route any more, and its absence is the point.**
`basis.EVBBasis._barrierless` used to offer these channels past a bare distance
prescreen with no off-diagonal at all, because a zero coupling can never clear
`stab > eps`.  A coupling centred on the crossing clears it *by construction*:
at a crossing the diabats are degenerate, so `stab = |V|`, and a Gaussian
centred there is at its maximum.  `test_the_crossing_is_inside_the_window` is
that property, and `test_the_break_is_smooth` is what the real off-diagonal buys
over the zero one -- the force jumped by 1.45 eV/A at H2's crossing when the two
diabats met with nothing between them, and now jumps by 0.0014.

The scans here are deliberately run through `System` with the topology carried
forward, as an MD run carries it, rather than through the ASE calculator, which
re-perceives bonds from geometry on every call and would decide the question
these tests are asking.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System


RSET_PATH = "datasets/HCombustion/HCombustion.json"

# Where each fission's diabats cross, from `fit.coupling.find_crossing` -- the
# `r0` written into the channel's own term file.  A refit is allowed to move
# these; they are quoted so a scan can be aimed, not as a bar.
#
# 2026-09-22, the per-bond asymptote replacing the shape term: 2.210 -> 1.800
# (H2), 2.078 -> 1.950 (OH).  A higher `h` lifts the bonded diabat faster, so
# it meets the fragments' sooner.
CROSSINGS = {"H2": 1.800, "OH": 1.950}


@pytest.fixture(scope="module")
def reaction_set():
    return ReactionSet(RSET_PATH)


def seeded(atoms: Atoms, bonds: list[tuple[int, int]]) -> Topology:
    """`atoms` with exactly `bonds`, whatever perception would have said."""
    topology = Topology.from_atoms(atoms.copy())
    topology.graph.remove_edges_from(list(topology.graph.edges()))
    topology.graph.add_edges_from(bonds)
    return topology


def walk(reaction_set, atoms, topology, place, values, cutoff=4.0):
    """Carry one system along `values`, `place(atoms, v)` setting each geometry."""
    system = System(atoms, topology, reaction_set, bimol_cutoff=cutoff)
    out = []
    for value in values:
        place(atoms, value)
        system.update(atoms=atoms)
        results = system.calculate()
        system.update(topology=results["topology"])
        block = results["blocks"][0]
        out.append(
            {
                "x": float(value),
                "energy": float(results["energy"]),
                "force": float(results["forces"][-1, 2]),
                "bonds": results["topology"].graph.number_of_edges(),
                "nstates": block["nstates"],
                "placeholders": block["placeholder_channels"],
            }
        )
    return out


def stretch(reaction_set, symbols, radii, bonded, cutoff=4.0):
    """A diatomic pulled along z, seeded bonded or not."""
    atoms = Atoms(symbols, positions=[[0.0, 0.0, 0.0], [0.0, 0.0, radii[0]]])

    def place(a, r):
        a.positions[1, 2] = r

    return walk(
        reaction_set,
        atoms,
        seeded(atoms, [(0, 1)] if bonded else []),
        place,
        radii,
        cutoff,
    )


def first_flip(steps):
    """Index of the first step at which the bond count drops."""
    return next(
        i for i in range(1, len(steps)) if steps[i]["bonds"] < steps[i - 1]["bonds"]
    )


class TestFission:
    """The property the whole route exists for."""

    @pytest.mark.parametrize("symbols", ["H2", "OH"])
    def test_a_stretched_bond_breaks_at_the_crossing(self, reaction_set, symbols):
        """And breaks *where the diabats cross*, not somewhere asymptotic.

        This used to assert nothing: with the dissociated limit of a bonded
        diabat pinned to zero, a Morse well approached its asymptote from below
        and never reached it, so the bonded state stayed infinitesimally the
        lower one at every separation and neither the `eps` gate nor `argmax`
        could ever let go.

        The break lands slightly *past* the crossing -- 1.83 A against 1.80 for
        H2 -- and that is `System._pivot`, not a mis-centred coupling: it swaps
        on a squared eigenvector weight past a threshold, so it waits until the
        dissociated state clearly dominates rather than flipping at degeneracy.

        The tolerance is wide because the crossing is a force-field property that
        a refit is allowed to move; what is asserted is that the break tracks it
        rather than the enumeration bounds on either side.
        """
        start = 0.75 if symbols == "H2" else 0.95
        steps = stretch(reaction_set, symbols, np.arange(start, 6.01, 0.02), True)
        assert steps[0]["bonds"] == 1
        assert steps[-1]["bonds"] == 0
        broke = next(s for s in steps if s["bonds"] == 0)
        assert broke["x"] == pytest.approx(CROSSINGS[symbols], abs=0.25)

    def test_the_energy_is_continuous_through_the_break(self, reaction_set):
        """The diabats are *equal* where they cross, so the swap has no step.

        Compared against a one-sided extrapolation rather than against the
        previous energy, so the bond's own steepness -- 1.45 eV/A at the H2
        crossing -- is not mistaken for a discontinuity.
        """
        centre = CROSSINGS["H2"]
        steps = stretch(
            reaction_set, "H2", np.arange(centre - 0.2, centre + 0.6, 0.002), True
        )
        before, after = steps[first_flip(steps) - 1], steps[first_flip(steps)]
        extrapolated = before["energy"] - before["force"] * (after["x"] - before["x"])
        assert after["energy"] == pytest.approx(extrapolated, abs=5e-3)

    @pytest.mark.parametrize("symbols", ["H2", "OH"])
    def test_the_break_is_smooth(self, reaction_set, symbols):
        """What the fitted off-diagonal buys, asserted so a regression shows up.

        Two diabats crossing with *nothing between them* give a continuous energy
        with a discontinuous slope: the force jumps by the difference of the two
        gradients, about 1.45 eV/A for H2, since the dissociated state is nearly
        flat where the Morse is still pulling. That was the cost of `A = 0`, and
        this test used to bound it at `0.5 < kink < 3.0` rather than remove it.

        A Gaussian centred on the crossing removes it. `fit_twobody` picks the
        largest amplitude that puts no spurious minimum there, which makes the
        adiabatic curve exactly flat in its second derivative at the crossing, so
        what is left is the smooth part of the avoided crossing: measured 0.0014
        eV/A for H2 and 0.0006 for OH across the flip, a thousandfold reduction.

        Bounded at 0.05 rather than at the measured value so that a refit may
        move it, but not by the three orders of magnitude that would mean the
        amplitude had gone back to zero.
        """
        centre = CROSSINGS[symbols]
        steps = stretch(
            reaction_set, symbols, np.arange(centre - 0.4, centre + 0.4, 0.002), True
        )
        flip = first_flip(steps)
        kink = abs(steps[flip]["force"] - steps[flip - 1]["force"])
        assert kink < 0.05, f"force jumped by {kink:.4f} eV/A at the crossing"

        # And nowhere else on the handover either, which the flip alone would not
        # catch: a coupling too narrow to bridge the crossing would leave the
        # jump a few steps to one side of where the pivot happens to swap.
        forces = np.array([s["force"] for s in steps])
        assert np.abs(np.diff(forces)).max() < 0.05

    def test_water_loses_a_hydrogen(self, reaction_set):
        """`rxn_08`, which no earlier approach could make react at all.

        Its two diabats disagreed by 0.385 eV at 10 A -- the O-H Morse is deeper
        in the H2O template than in the OH one, because
        `fit.dissociation.fit_dissociation_energies` constrains each template's
        bond *sum* and never an individual asymptote -- so a degeneracy criterion
        was never going to fire, at any separation.  Raising the asymptote by more
        than that disagreement subsumes it: a cross-template offset can move the
        crossing but can no longer remove it.

        A polyatomic case is not redundant with the diatomics above.  The gap
        that `find_crossing` walks is a difference of two whole diabats, and only
        for a diatomic is it certain to be the bond term alone; here the reactant
        also carries an angle and two cross terms that the product does not.
        """
        theta = 1.824
        atoms = Atoms(
            "OHH",
            positions=[
                [0.0, 0.0, 0.0],
                [0.9615, 0.0, 0.0],
                [0.9615 * np.cos(theta), 0.9615 * np.sin(theta), 0.0],
            ],
        )

        def place(a, r):
            a.positions[2] = [r * np.cos(theta), r * np.sin(theta), 0.0]

        steps = walk(
            reaction_set,
            atoms,
            seeded(atoms, [(0, 1), (0, 2)]),
            place,
            np.arange(0.9615, 6.0, 0.02),
        )
        assert steps[0]["bonds"] == 2
        assert steps[-1]["bonds"] == 1
        broke = next(s for s in steps if s["bonds"] == 1)
        # 2.42 before the per-bond asymptote; 2.00 after, against a 1.93 crossing.
        assert broke["x"] == pytest.approx(2.00, abs=0.4)


class TestAdmission:
    """That the ordinary gate is enough, which is why no special route is left."""

    @pytest.mark.parametrize("symbols", ["H2", "OH"])
    def test_the_crossing_is_inside_the_window(self, reaction_set, symbols):
        """The structural guarantee that replaced `_barrierless`.

        At a crossing the two diabats are degenerate, so the stabilization
        `hypot(dH, V) - |dH|` is exactly `|V|`; a Gaussian centred on the
        crossing is at its maximum `|A|` there; and `fit_twobody` refuses any
        channel whose `|A|` does not exceed `eps`.  So the channel is admitted at
        the crossing by construction, for every fission, without a distance
        prescreen and without a tolerance.

        The old route needed `BARRIERLESS_BREAK_RADIUS = 2.0` to decide which
        channels were worth evaluating, and that constant had to be threaded
        between the most stretched real bond and the earliest crossing by hand.
        It also could not cover `rxn_06`, whose crossing is at 1.72 A, *inside*
        the radius.

        Asserted with a margin either side, so a coupling that only just reached
        the crossing would not pass.
        """
        centre = CROSSINGS[symbols]
        steps = stretch(
            reaction_set, symbols, np.arange(centre - 0.3, centre + 0.3, 0.01), True
        )
        assert all(s["nstates"] == 2 for s in steps), (
            "the channel is not admitted across its own crossing: "
            f"{[(s['x'], s['nstates']) for s in steps if s['nstates'] != 2][:5]}"
        )

    def test_an_equilibrium_molecule_offers_nothing(self, reaction_set):
        """What keeps a condensed phase free of these channels.

        `fit_twobody` quenches the coupling to `eps` at the reactant's own
        minimum, which is `fit_width`'s condition in the bond length, so a
        molecule sitting at equilibrium gets no stabilization from its own
        dissociation channel and the basis stays single-state.  The minimum is
        *found* rather than read off the stored frame for this reason: `rxn_08`'s
        reactant carries one O-H at 0.600 A against a 0.95 A minimum, and
        quenching 0.35 A inside the wall would leave the coupling four times
        `eps` where the molecule actually sits.

        1.3 A used to be on this list and is not any more: with the crossing at
        1.80 A rather than 2.21, H2's channel is admitted from 1.16 A, so a bond
        stretched 0.56 A past equilibrium -- about 2 eV up the well -- is inside
        the window.  1.1 A is still outside it.
        """
        steps = stretch(reaction_set, "H2", [0.7445, 0.9, 1.1], True)
        assert all(s["nstates"] == 1 for s in steps)
        assert all(s["bonds"] == 1 for s in steps)

    def test_every_fission_channel_carries_a_fitted_coupling(self):
        """No channel is left decoupled, and none is a placeholder.

        The six channels that change one bond in one direction all used to read
        `A = 0.0, provenance: decoupled`.  They now carry a `twobody` term fitted
        from their own diabatic crossing, which is what makes them visible to
        `EVBBasis` at all -- and `provenance` is what `Block.placeholder_channels`
        reports on, so a channel that regressed to a stand-in amplitude would show
        up in a trajectory rather than only here.

        `rxn_03` used to be listed here and is not a fission.  Its reactant frame
        was missing the H-H bond, so it parsed as `H + H + OH -> H + H2O` -- one
        bond, no saddle -- and `_fission` routed it accordingly.  With the bond
        stated it is `H2 + OH -> H2O + H`: two bonds, a real barrier (TS 0.112 eV
        above the reactant), and a `threebody` transfer coupling.
        """
        fissions = [
            "rxn_05",
            "rxn_06",
            "rxn_07",
            "rxn_08",
            "rxn_09",
            "rxn_15",
        ]
        root = Path(RSET_PATH).parent / "reactions"
        for name in fissions:
            terms = [
                json.loads(line)
                for line in (root / f"{name}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert len(terms) == 1, name
            term = terms[0]
            assert term["type"] == "twobody", f"{name} is not a bond-length coupling"
            assert term["provenance"] == "fitted", name
            assert abs(term["kwargs"]["A"]) > 0.0, f"{name} is still decoupled"
            assert term["kwargs"]["a"] > 0.0, name
            assert term["kwargs"]["r0"] > 0.0, name

    def test_no_placeholder_channels_on_a_dissociation_path(self, reaction_set):
        """The run-time half of the assertion above."""
        steps = stretch(reaction_set, "H2", np.arange(0.75, 4.01, 0.05), True)
        assert all(s["placeholders"] == [] for s in steps)


class TestRecombination:
    def test_separated_atoms_form_a_bond(self, reaction_set):
        """The reverse channel, which is the same channel.

        `ReactionSet.add_reaction` stores both directions and `Reaction.reverse`
        keeps the same term list and the same template indices, so the one fitted
        Gaussian serves fission and recombination without a second fit.  It does
        need the crossing to fall *inside* `bimol_cutoff`, because that is the
        only region where the reverse channel is enumerated at all; H2's is at
        1.80 A against a 4.0 A cutoff.
        """
        steps = stretch(reaction_set, "H2", np.arange(4.0, 0.99, -0.02), False)
        assert steps[0]["bonds"] == 0
        assert steps[-1]["bonds"] == 1

    def test_the_path_has_no_hysteresis(self, reaction_set):
        """Fission and recombination are the same surface, to the last bit.

        This is the invariance `basis`'s closure is built for, and the reason the
        old distance prescreen had to test one interval from both sides: if
        breaking a bond and re-forming it disagree about whether the pair is
        admissible, the energy depends on which state the trajectory happened to
        be carrying.  A Gaussian in the bond length is symmetric in the
        separation by construction, so there is nothing left to get wrong -- the
        two directions agree to 2.6e-13 eV, which is arithmetic noise on a 4.7 eV
        well.
        """
        grid = np.arange(0.75, 4.01, 0.01)
        forward = stretch(reaction_set, "H2", grid, True)
        reverse = stretch(reaction_set, "H2", grid[::-1], False)
        by_radius = {round(s["x"], 4): s["energy"] for s in forward}
        differences = [
            abs(by_radius[round(s["x"], 4)] - s["energy"])
            for s in reverse
            if round(s["x"], 4) in by_radius
        ]
        assert len(differences) == len(grid)
        assert max(differences) < 1e-9

    def test_leaving_the_basis_costs_nothing(self, reaction_set):
        """The state must be worthless *before* it stops being enumerated.

        `get_network` adds a bimolecular channel only inside `bimol_cutoff`, so
        the state vanishes from the basis there whatever the energies say.  That
        is a real discontinuity in the basis, and `fit_twobody`'s outer quench is
        what makes it harmless: the width is set by whichever end of the path is
        nearer the crossing, the cutoff included, so the coupling is already down
        to `eps` when the state goes.

        This is not free, and it was not free by accident before: `A = 0` carried
        no stabilization to lose.  Fitted without the outer condition, `rxn_08`'s
        coupling was still worth 0.2 eV at 4.0 A and the energy stepped by
        exactly that when a dissociating OH...H drifted past the cutoff.  With it,
        H2 leaves the basis at 2.47 A -- through the `eps` gate, well inside the
        cutoff, which is the point -- for 1.9e-6 eV.

        The distance is a force-field property and a refit moves it, so the scan
        starts well inside it; what is asserted is that the exit is decided by
        the `eps` gate rather than by the cutoff, and that it is free.
        """
        start = CROSSINGS["H2"] + 0.1
        steps = stretch(reaction_set, "H2", np.arange(start, 4.20, 0.002), False)
        left = next(
            i
            for i in range(1, len(steps))
            if steps[i]["nstates"] < steps[i - 1]["nstates"]
        )
        assert steps[left]["x"] < 4.0, (
            "the state survived to the enumeration cutoff, so the cutoff and not "
            "the coupling is deciding when it leaves the basis"
        )
        assert abs(steps[left]["energy"] - steps[left - 1]["energy"]) < 1e-5

"""Does anything stop two molecules from passing through each other?

For most of this project's life, no.  Measured on a 200-atom H2/O2 box at
2000 K (`examples/nvt-n100-d250.xyz`), atoms in *different* molecules reached
**0.044 A** of each other, with 227 intermolecular pairs inside 1.5 A by frame
135 -- and 225 of those 227 involved a hydrogen, which is exactly the atom
q-force assigns a zero Lennard-Jones radius.

The visible symptom was an electrostatic energy running away monotonically,
which was blamed on ACKS2.  ACKS2 was innocent: `erf(2r)/r` and `exp(-r/tau)`
diverge at contact because nothing is supposed to get there.  The force field
simply had no Pauli repulsion, and the charge equilibration was faithfully
reporting the geometry it was handed.

So this file asserts the thing that was missing rather than the symptom that
followed.  It drives two water molecules into each other head-on and requires
them to bounce.

**On why this is a collision and not a box.**  The 200-atom box is where the
number above was measured, but a single force call on it takes seconds and a
trajectory long enough to collapse takes minutes -- too slow to run on every
commit.  Six atoms driven together deliberately reach contact in 50 fs and test
the same wall.  The box remains the place the effect was *found*; this is the
place it is *guarded*.
"""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms, units
from ase.md.verlet import VelocityVerlet

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System

RSET_PATH = "datasets/HCombustion/HCombustion.json"

CELL = 30.0  # cubic box edge, Angstrom; vacuum, nothing wraps
SEPARATION = 5.0  # initial centre-to-centre distance, Angstrom
TIMESTEP = 0.25  # fs
STEPS = 400

# Closest approach that counts as "did not interpenetrate".  Two water
# molecules in contact sit around 1.4-1.5 A at the closest H...O; anything under
# this is inside the electron density of both and is where ACKS2's contact funnel
# takes over.
#
# **Lowered from 1.15 when `ZBL` acquired its taper** (`forcefield/zbl.py`), which
# is the one place in this file the taper is visible.  Measured, against the same
# run with the repulsion stubbed out entirely:
#
#     speed   KE_in     with wall   no wall
#     0.25    1.13 eV     1.287 A     0.818 A     (was 1.81 A untapered)
#     0.50    4.50 eV     1.105 A     0.807 A     (was 1.24 A untapered)
#
# **Re-measured when `forcefield/lj.py` was switched back on**, which moved it
# the other way and is left at 1.05 with the extra margin rather than tightened:
#
#     wall in play        speed 0.25   speed 0.50
#     ZBL + 12-6            1.423 A      1.129 A     <- what ships
#     ZBL alone             1.267 A      1.131 A
#     12-6 alone            0.808 A      0.817 A
#     neither               0.834 A      0.812 A
#
# The third row is the one worth reading: **the 12-6 contributes nothing to this
# collision at all**, and cannot, because `lj.switch` has taken it to zero by
# 1.5 A and these molecules are at 1.1 A when they turn around.  What it buys is
# in the first column -- it slows the approach out at 2-3 A, so less kinetic
# energy arrives -- which is why the slow case gains 0.16 A and the fast one is
# unchanged to within noise.  The inner wall is `ZBL`'s alone and this file is
# still a test of `ZBL`; `test_the_wall_rises_monotonically` is where the 12-6
# is visible.
#
# The taper removes the outer wall between roughly 1.6 and 3 A and leaves the
# inner one alone -- it retains 99% of ZBL at the O-H bond length and 90% at
# 1.24 A -- so what changed is the *approach*, not the barrier: the molecules no
# longer pay on the way in and arrive with more kinetic energy, then turn around
# against a wall that is still there.  A 4.5 eV head-on is 17 kT at 3000 K, so
# the faster of these two is well outside anything a production run samples.
#
# The margin to `COLLAPSED` is thinner than it was, and that is a real cost of
# the taper rather than a bookkeeping change.  It is widest where it matters: the
# anti-vacuity check below runs `SPEEDS[0]`, where the wall still buys 1.287 A
# against 0.818 A.
#
# **Re-measured again when the angle potential acquired its missing factor of
# 1/2** (`qforce.compute_angle`) and both datasets were refit against it.  This
# is the first entry here where nothing about the *wall* changed -- `ZBL` and
# `lj` are untouched -- and the molecules still arrive closer:
#
#     wall in play        speed 0.25   speed 0.50
#     ZBL + 12-6            1.385 A      1.040 A     <- what ships
#     ZBL alone             1.206 A      1.036 A
#     12-6 alone            0.841 A      0.821 A
#     neither               0.813 A      0.819 A
#
# What moved is the *projectile*, not the barrier.  q-force emits its angle
# parameters for `0.5*k*(cos(theta)-cos(theta0))^2` -- the energy expression is
# in the `<CustomAngleForce>` of every `tests/data/*_qforce.xml` -- so before the
# fix this force field was running every angle at twice its intended stiffness.
# A water molecule that is no longer artificially rigid deforms on impact
# instead of bouncing as a unit, and the closest H...O gets 0.09 A nearer at the
# fast speed.  The `neither` row is unchanged to within noise, as it must be:
# with both terms stubbed there is no wall for a stiffer or softer molecule to
# hit.
#
# **Lowered to 1.03, and the margin to `COLLAPSED` is now genuinely thin** --
# 0.03 A at `SPEEDS[1]`, against 0.22 A of wall over the stubbed run at the same
# speed.  The wall is still doing its job and the bar is still above what a
# stubbed run reaches, but this constant no longer has room to absorb another
# change of this size; the next thing that moves it should widen the gap rather
# than step the bar down again.  `SPEEDS[1]` is a 4.5 eV head-on, 17 kT at
# 3000 K and well outside what a production run samples -- `SPEEDS[0]`, which is
# inside it, still clears the bar by 0.36 A.
#
# **`qforce.BOND_ASYMPTOTE` is that next change, and this bar is deliberately NOT
# moved for it.**  Measured at `SPEEDS[1]`, one variable at a time:
#
#     HEAD dataset, asymptote 0      1.0401 A     <- the real baseline
#     refit dataset, asymptote 1.0   1.0283 A     <- now; fails by 0.0017 A
#
# Two things to read off that.  The asymptote costs 0.012 A, half what the angle
# refit above cost.  And **the 0.03 A this comment claims was already stale**:
# the margin at HEAD was 0.0101 A, so the bar was one small change away from
# failing before this one arrived, and it is the accumulation rather than the
# asymptote that spends it.
#
# Left failing on purpose.  The instruction above is to widen the gap, and the
# gap is the repulsion's: `ZBL` provides 0.221 A of wall over a stubbed run here
# and no bonded parameter adds to it -- the same dead end
# `test_reference_energies.py::TestTemplateFrequencies` reaches from the other
# side ("what is left is a property of the repulsion's functional form").
# Changing that form means refitting every `.jsonl` in both datasets, a larger
# job than the one this bar blocks, and stepping the bar to 1.02 for a green
# suite is exactly what the paragraph above forbids.
#
# `SPEEDS[0]` -- the case inside what a production run samples -- is at 1.3747 A
# against 1.3854 at HEAD, clearing by 0.345 A.
CONTACT = 1.03

# What the same run does with the repulsion removed: 0.818 A at `SPEEDS[0]` and
# 0.807 A at `SPEEDS[1]`.  The anti-vacuity check below requires it to go under
# this, so a collapse test that passes without the term it exists to test cannot
# go unnoticed.  Unmoved by the taper, as it must be -- with `ZBL` stubbed there
# is nothing left for a taper to modify.
COLLAPSED = 1.0

# Bond length, in Angstrom, past which an O-H has genuinely come apart rather
# than merely stretched.  Deliberately far above both the 1.261 A bond
# perception radius and the 1.69 A this collision actually reaches: the question
# is dissociation, not excitation, and an O-H at 3 A is unbound on any reading.
DISSOCIATED = 3.0

# Head-on closing speeds, in ASE velocity units, giving 1.1 and 4.5 eV of
# collision energy -- 4 and 17 times kT at the 3000 K these runs are for.  The
# faster is four times the collision energy of the slower, so a wall that is
# merely soft shows up as the two cases disagreeing rather than as both passing.
#
# **These were 0.5 and 1.0, and the bar came down deliberately.**  The old
# repulsion was a 12-6 with q-force's sigma_H = 1.96 A, which is 39 eV at 0.96 A
# and 504 eV at the H2 bond length -- a wall no collision could ever get through,
# so any threshold calibrated against it says nothing about whether the
# repulsion is *right*.  `ZBL` is 1.2 eV at 0.96 A, and it is weakest exactly
# where the closest contact in this collision is: H...H, where Z = 1.  An 18 eV
# head-on -- 70 kT, which nothing in a 3000 K NVT run will ever see -- does push
# it to 0.88 A.  Asserting otherwise would be asserting ZBL's accuracy at a
# separation where it has none, so that case is kept below as
# `test_a_violent_collision_stays_bounded`, which asserts the things that
# actually matter there.
SPEEDS = [0.25, 0.5]

# A collision far outside the thermal range: 18 eV, against kT = 0.26 eV.
VIOLENT_SPEED = 1.0


@pytest.fixture(scope="module")
def reaction_set():
    return ReactionSet(RSET_PATH)


def head_on(reaction_set, speed):
    """Two H2O molecules `SEPARATION` apart, moving into each other along x."""
    atoms = Atoms(cell=np.eye(3) * CELL, pbc=False)
    velocities = []
    for sign, x in ((+1.0, CELL / 3), (-1.0, CELL / 3 + SEPARATION)):
        mol = next(reaction_set.get_molecules(formulas=["H2O"])).atoms.copy()
        mol.positions -= mol.positions.mean(0)
        mol.positions += np.array([x, CELL / 2, CELL / 2])
        atoms += mol
        velocities.extend([[sign * speed, 0.0, 0.0]] * len(mol))
    atoms.set_velocities(np.array(velocities))
    return atoms


def min_intermolecular_distance(atoms):
    """Closest approach between atoms that the perceived topology calls unbonded.

    Perceived rather than carried, deliberately: the question is whether two
    things that are not part of the same molecule can occupy the same space, and
    perception is what decides that at every step of a reactive trajectory.
    """
    graph = Topology.from_atoms(atoms).graph
    component = {}
    for label, nodes in enumerate(nx.connected_components(graph)):
        for node in nodes:
            component[node] = label

    positions = atoms.positions
    closest = np.inf
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            if component[i] != component[j]:
                closest = min(
                    closest, float(np.linalg.norm(positions[i] - positions[j]))
                )
    return closest


def collide(reaction_set, speed):
    """Run the collision.  Returns (closest approach, potential energies)."""
    atoms = head_on(reaction_set, speed)
    atoms.calc = DynamicTopology(atoms, reaction_set)
    dynamics = VelocityVerlet(atoms, timestep=TIMESTEP * units.fs)

    closest, energies = np.inf, []
    for _ in range(STEPS):
        closest = min(closest, min_intermolecular_distance(atoms))
        energies.append(atoms.get_potential_energy())
        dynamics.run(1)
    return closest, np.array(energies)


@pytest.mark.parametrize("speed", SPEEDS)
def test_molecules_do_not_interpenetrate(reaction_set, speed):
    closest, energies = collide(reaction_set, speed)

    # The molecules have to actually meet, or "they never got close" would pass
    # this for the wrong reason.
    assert closest < 2.0, (
        f"the molecules never came within 2.0 A (closest {closest:.3f} A); "
        "the collision is too slow to test anything"
    )
    assert closest > CONTACT, (
        f"atoms in different molecules approached to {closest:.3f} A at closing "
        f"speed {speed}; below {CONTACT} A there is nothing holding them apart "
        "and ACKS2 is being handed a geometry it cannot describe"
    )
    assert np.isfinite(energies).all(), "the potential energy diverged"


def test_a_violent_collision_stays_bounded(reaction_set):
    """Far above anything thermal, the requirement is boundedness, not a wall.

    18 eV head-on is 70 kT at 3000 K.  Nothing in a production run arrives with
    that, and `ZBL` does not claim to stop it -- H...H at 0.88 A is 1.45 eV, and
    for Z = 1 that is about all the screened-nuclear form has to give.

    What must still hold is everything the original failure was: the molecules
    stay molecules, the charge equilibration stays bounded rather than running
    away monotonically, and no energy is infinite.  The pre-repulsion surface
    failed all three -- `E_nonbonded` fell to -46.6 eV and kept going while the
    box was still made of intact diatomics.

    **"Stay molecules" is measured as a bond length, not as a bond count, and
    the difference is not pedantry.**  This asserted `min(bonds) == bonds[0]`
    off `Topology.from_atoms` until the force-constant refit softened the bonds
    towards their experimental frequencies, at which point it started failing on
    an intact molecule.  What it was actually watching: the O-H distance here
    oscillates between 0.92 and 1.69 A -- a hot but bound vibration -- while
    `Topology.from_atoms` cuts O-H at `1.3 * (0.66 + 0.31) = 1.261 A`, so the
    perceived bond count flips 4, 3, 2 and back to 4 as that oscillation crosses
    the threshold.  At 13 eV it dropped to 2 and *recovered*, which is what gave
    it away.  The stiffer the force field, the less a bond stretches, and the
    longer that metric survives by luck.  See the bond-perception cliff recorded
    in `tests/geometry.py`.
    """
    atoms = head_on(reaction_set, VIOLENT_SPEED)
    atoms.calc = DynamicTopology(atoms, reaction_set)
    dynamics = VelocityVerlet(atoms, timestep=TIMESTEP * units.fs)

    # The four O-H bonds, by index: `head_on` builds O H H O H H.
    pairs = [(0, 1), (0, 2), (3, 4), (3, 5)]

    stretches, electrostatics, energies = [], [], []
    for _ in range(STEPS):
        energies.append(atoms.get_potential_energy())
        electrostatics.append(atoms.calc.diagnostics["energy_nonbonded"])
        stretches.append(max(atoms.get_distance(i, j) for i, j in pairs))
        dynamics.run(1)

    assert np.isfinite(energies).all(), "the potential energy diverged"
    assert max(stretches) < DISSOCIATED, (
        f"the molecules came apart: an O-H reached {max(stretches):.2f} A. "
        "Every template is under-bound by its own repulsion energy, which means "
        "the Morse depths have not absorbed it"
    )
    assert min(electrostatics) > -10.0, (
        f"ACKS2 reached {min(electrostatics):.3f} eV; it is being handed a "
        "geometry it cannot describe and nothing is opposing the contact funnel"
    )


def test_the_collision_collapses_without_the_repulsion(reaction_set, monkeypatch):
    """Anti-vacuity: the test above must fail on the pre-LJ surface.

    The removal is a *pair*, and it has to be.  This docstring used to say `ZBL`
    had no second half to forget -- true until `forcefield/exclusions.py` gave it
    one.  `zblexclusion` cancels the whole-system sum between near neighbours, so
    stubbing `ZBL.__call__` alone leaves every molecule carrying the
    cancellation of a term that is no longer there: a spurious intramolecular
    *attraction*, which held the colliding pair at 3.47 A and so made this
    anti-vacuity check pass for precisely the reason it exists to rule out.
    That is the same trap its own predecessor fell into with the 12-6, one term
    later.

    `QForce` dispatches by looking up `compute_<type>` and silently skips a type
    with no method, so deleting the method is the whole removal of the second
    half -- and there is no zero-shaped return to get wrong.
    """
    from DynamicTopology.forcefield.zbl import ZBL
    from DynamicTopology.forcefield.qforce import QForce

    monkeypatch.setattr(
        ZBL,
        "__call__",
        lambda self, pos, numbers, pbc, cell, displacements=None: (
            0.0,
            np.zeros_like(pos),
            np.zeros((3, 3)),
        ),
    )
    monkeypatch.delattr(QForce, "compute_zblexclusion", raising=False)

    closest, _ = collide(reaction_set, SPEEDS[0])
    assert closest < COLLAPSED, (
        f"without the ZBL term the molecules still only reached "
        f"{closest:.3f} A, so something else is holding them apart and this "
        "file is not measuring what it claims to"
    )


# ---------------------------------------------------------------------------
# The wall, and what the diabatic states do and do not see of it
# ---------------------------------------------------------------------------

# Head-on H2 + O2, closest H...O gap in Angstrom.  4.0 A is outside contact and
# is the baseline; 0.6 A is the separation both failing 3000 K probes actually
# reached, in a box that was still ninety-nine intact diatomics.
FAR = 4.0
FUSED = 0.6

# The H2 and O2 bond lengths, Angstrom.  The molecules are held rigid at them
# so that the only thing varying along the scan is the gap between them.
BOND = 0.744
O2_BOND = 1.208

# What the wall has to be worth between those two separations.  Measured:
# +21.8 eV, against kT = 0.26 eV at 3000 K.  ACKS2 alone runs the other way,
# reaching -1.52 eV at 0.6 A with no repulsive branch at all.
WALL = 15.0


def head_on_diatomics(gap):
    """H2 and O2 along z, `gap` Angstrom between the nearest H and O.

    Deliberately a pair for which a bimolecular reaction *is* applicable --
    `O2 + H2 -> HO2 + H` is, at anything under `bimol_cutoff` -- because
    applicability is the thing that used to take the wall down.
    """
    z = [0.0, BOND, BOND + gap, BOND + gap + O2_BOND]
    atoms = Atoms(
        "HHOO", positions=[[0.0, 0.0, zz] for zz in z], cell=[40.0] * 3, pbc=True
    )
    state = Topology.from_atoms(atoms)
    state.graph.remove_edges_from(list(state.graph.edges()))
    state.graph.add_edge(0, 1)
    state.graph.add_edge(2, 3)
    state._hash = None
    state._molecules = None
    state.set_atoms(atoms)
    return atoms, state


class TestTheReactiveWall:
    """Two intact molecules must pay to interpenetrate, whatever is applicable.

    This is the case `test_molecules_do_not_interpenetrate` cannot see: it
    collides two H2O, and no bimolecular reaction in the dataset applies to
    H2O + H2O, so a rule keyed on applicability was never engaged by it.

    The defect this guards shipped.  The repulsion used to be a Lennard-Jones
    with per-state exclusions, and the exclusion set was built for a while from
    every bond any *applicable* reaction would form -- applicability being a
    distance test at `bimol_cutoff` and nothing more.  `O2 + H2 -> HO2 + H` is
    applicable at 4 A, so the wall came off every H...O pair in the box
    including on states where nothing had reacted, leaving ACKS2's contact
    attraction unopposed: a -0.96 eV well at 0.70 A against kT = 0.26 eV at
    3000 K.  Both probe trajectories filled it, reaching 0.60 A and -47 eV of
    `E_nonbonded`.  With that rule in place this class reads -4944 eV.
    """

    def _energy(self, reaction_set, gap):
        atoms, state = head_on_diatomics(gap)
        return System(atoms, state, reaction_set).calculate()["energy"]

    def test_an_unreacted_pair_keeps_its_whole_wall(self, reaction_set):
        atoms, state = head_on_diatomics(FUSED)
        applicable = sum(1 for _ in reaction_set.get_network(state, 4.0).reactions())
        assert applicable > 0, (
            "no reaction is applicable to this geometry, so it cannot show "
            "whether applicability alone takes a wall down"
        )

        wall = self._energy(reaction_set, FUSED) - self._energy(reaction_set, FAR)
        assert wall > WALL, (
            f"two intact molecules driven from {FAR} A to {FUSED} A pay only "
            f"{wall:+.3f} eV, with {applicable} reactions merely applicable to "
            "them. The repulsion is keying on applicability again, and the box "
            "will fuse"
        )

    # Where the dispersion well is allowed to be, and how deep.  Outside
    # `WELL_INSIDE` the approach may be downhill, because since
    # `forcefield/lj.py` came back there is an attractive `-r**-6` tail and a
    # potential that rose monotonically from 4 A would mean it was missing.
    # Inside, nothing may be.
    WELL_INSIDE = 2.5  # A -- the minimum must lie outside this
    WELL_DEPTH = 0.05  # eV -- and be no deeper than this

    def test_the_wall_rises_monotonically(self, reaction_set):
        """No plateau inside the dispersion well, so a restoring force throughout.

        A wall that is bounded but flat satisfies the test above and still lets
        atoms drift through, because the force vanishes inside the plateau.  A
        19 eV plateau on rxn_16's reaction path is exactly what the last design
        produced, by removing a fraction of a wall rather than all or none of it.

        **This used to start at `FAR` and now starts at the well minimum.**  The
        old form asserted the approach was uphill from 4.0 A all the way in,
        which was true only while every nonbonded term was repulsive.  It is not
        any more: the 12-6's `-r**-6` tail makes 4.0 -> 3.0 A downhill by
        3.0 meV, and that is the term doing its job rather than the wall failing.
        Asserting the well's *shape* instead is strictly stronger than the old
        assertion -- it pins the depth and the position as well as the
        monotonicity, so a dispersion tail that had grown into a trap deep enough
        to hold two molecules together at contact would fail here, where under
        the old form it would merely have failed to be uphill.
        """
        gaps = [FAR, 3.5, 3.0, 2.5, 2.0, 1.5, 1.2, 0.9, FUSED]
        energies = [self._energy(reaction_set, gap) for gap in gaps]

        floor = int(np.argmin(energies))
        assert gaps[floor] >= self.WELL_INSIDE, (
            f"the potential still falls at {gaps[floor]} A, inside "
            f"{self.WELL_INSIDE} A; that is not a dispersion well, it is the "
            "wall giving way"
        )
        well = energies[0] - energies[floor]
        assert well < self.WELL_DEPTH, (
            f"the well at {gaps[floor]} A is {well:.4f} eV deep against a "
            f"ceiling of {self.WELL_DEPTH} eV. Two H2 + O2 that bound this "
            "hard would never come apart"
        )

        for near, far, e_near, e_far in zip(
            gaps[floor + 1 :], gaps[floor:], energies[floor + 1 :], energies[floor:]
        ):
            assert e_near > e_far, (
                f"closing from {far} A to {near} A costs {e_near - e_far:+.4f} eV; "
                "the wall is flat or falling there and nothing pushes back"
            )

    def test_the_wall_is_the_zbl_term(self, reaction_set, monkeypatch):
        """Anti-vacuity: with ZBL stubbed almost none of the wall is left.

        **This used to assert the approach went downhill**, because ACKS2's
        global softness moved charge between the two molecules at contact -- a
        funnel that only ZBL opposed.  Fragment ACKS2 moves no charge between
        molecules, and H2 and O2 carry none of their own, so without ZBL the
        approach is flat to within the 12-6's soft core (+0.018 eV at 0.6 A).
        What is still worth pinning is that the wall `test_an_unreacted_pair_
        keeps_its_whole_wall` measures is ZBL's and nobody else's.
        """
        from DynamicTopology.forcefield.zbl import ZBL

        monkeypatch.setattr(
            ZBL,
            "__call__",
            lambda self, pos, numbers, pbc, cell, displacements=None: (
                0.0,
                np.zeros_like(pos),
                np.zeros((3, 3)),
            ),
        )
        wall = self._energy(reaction_set, FUSED) - self._energy(reaction_set, FAR)
        assert wall < 0.01 * WALL, (
            f"with ZBL stubbed out the approach still costs {wall:+.3f} eV, so "
            "something else is holding the molecules apart and this class is "
            "not measuring what it claims to"
        )


class TestTheNonbondedTermsCancelFromEveryGap:
    """The structural claim the whole design rests on, pinned as a number.

    The whole-system repulsion sums -- `ZBL` and the 12-6 -- are functions of the
    geometry and the elements alone.  Neither consults the topology, so each is
    the *same number* on every diabatic state of a block: they add a common
    constant to every EVB diagonal, `np.linalg.eigh` shifts the eigenvalue by
    exactly that and leaves the eigenvectors untouched, and they cancel exactly
    out of every energy difference in the model.

    That is why the previous four attempts at this term are impossible to repeat
    rather than merely fixed.  Each of them -- the union rule, the lost-exclusion
    rule, the coupling gate -- was a case of the repulsion differing between
    states, and each failed in its own way: the box fused at 0.60 A, a 19 eV
    plateau appeared across rxn_16's reaction path, and transition states with
    close non-bonded contacts came back with EVB amplitudes of -72 to -108 eV.

    The electrostatics used to be part of this claim and no longer are: the
    charges follow the bonding, so they sit on the diagonal and differ between
    states by design (`tests/test_acks2_fragment.py`).

    The 12-6 keeps its part only while the templates agree on every atom's
    sigma and epsilon, which the shipped per-element values do and which this
    class holds them to.  Templates that disagree put the switched difference on
    the diagonal instead (`tests/test_lj_states.py`).
    """

    def test_a_broken_bond_moves_only_the_exclusion(self, reaction_set):
        """Same geometry, different topology: identical repulsion sums.

        The H2 diabat that has broken its bond is the state that used to be
        charged 727 eV of Lennard-Jones wall for a pair sitting at the bond
        length -- the artefact that forced a correction to exist at all.
        """
        atoms, intact = head_on_diatomics(2.0)

        broken = Topology(intact.graph.copy())
        broken.graph.remove_edge(0, 1)
        broken._hash = None
        broken._molecules = None
        broken.set_atoms(atoms)

        a, b = (
            System(atoms, topology, reaction_set).calculate()
            for topology in (intact, broken)
        )
        assert a["energy_zbl"] == b["energy_zbl"], (
            "the repulsion differs between two topologies over the same "
            "geometry; it has acquired a state dependence and every failure "
            "mode this design removes is back"
        )
        assert a["energy_lj"] == b["energy_lj"], (
            "the 12-6 whole-system sum differs between two topologies; it has "
            "acquired exclusions of its own"
        )

    def test_the_wall_is_not_vacuously_equal(self, reaction_set):
        """Anti-vacuity: the term has to be large here, not merely equal."""
        atoms, state = head_on_diatomics(FUSED)
        assert System(atoms, state, reaction_set).calculate()["energy_zbl"] > 20.0

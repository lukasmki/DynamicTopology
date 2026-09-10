"""What a written trajectory has to carry, and what a seed has to control.

Both properties here were absent and neither was noticed, because both fail
*silently* into plausible-looking output: a trajectory whose bonding never
changes reads exactly like a run in which nothing reacted, and an unseeded run
reads exactly like a reproducible one until someone tries to reproduce it.

The first is the more expensive of the two.  The reactive topology is the whole
point of this code and it lives on `calc.system.topology`; `atoms.info` holds
whatever the input file said and ASE never updates it.  Every trajectory written
before `nvt.record_topology` existed therefore stamped t = 0 bonding onto every
frame -- measured on the 116-frame `examples/nvt-n100-d250.xyz`, all 116 were
byte-identical -- which made the species panel of `scripts/plot.py` flat by
construction and silently reset chemistry across `--restart`.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms, io, units
from ase.md import Langevin
from ase.md.velocitydistribution import Stationary, thermalize_momenta

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet
from DynamicTopology.core.topology import Topology

from geometry import SWITCHING_PATH_START, SWITCHING_REACTION, reaction_path

RSET_PATH = Path("datasets/HCombustion/HCombustion.json").resolve()
CELL = 24.0


def _load_nvt():
    """Import `scripts/nvt.py`, which is not an installed module."""
    path = Path("scripts/nvt.py").resolve()
    spec = importlib.util.spec_from_file_location("_nvt_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_nvt_script"] = module
    spec.loader.exec_module(module)
    return module


nvt = _load_nvt()


@pytest.fixture(scope="module")
def reaction_set() -> ReactionSet:
    return ReactionSet(RSET_PATH)


class TestRecordedTopology:
    """A frame must describe the bonding the calculator was carrying."""

    def test_a_trajectory_records_the_topology_changing(self, reaction_set, tmp_path):
        """The assertion that would have caught it.

        Drives a trajectory known to switch topology, writes every frame the way
        `nvt.py` does, and requires the *stored* connectivity to differ across
        frames.  Anti-vacuity: if the trajectory never switched, the test says so
        rather than passing on a run that had nothing to record.
        """
        atoms = reaction_path(SWITCHING_REACTION, SWITCHING_PATH_START, CELL)
        thermalize_momenta(atoms, 1000.0, rng=np.random.default_rng(0))
        Stationary(atoms)
        atoms.calc = DynamicTopology(atoms, reaction_set)

        target = tmp_path / "traj.xyz"
        switches = 0
        dyn = Langevin(
            atoms,
            timestep=0.1 * units.fs,
            temperature_K=1000.0,
            friction=0.01 / units.fs,
            fixcm=False,
            rng=np.random.default_rng(0),
        )
        for _ in range(300):
            atoms.get_potential_energy()
            switches += bool(atoms.calc.topology_changed)
            nvt.record_topology(atoms)
            io.write(target, atoms, format="extxyz", append=True)
            dyn.run(1)

        # One, not two.  The assertion below needs the trajectory to *reach* a
        # second topology, which one switch does; two was asking it to recross,
        # which is strictly more than this test uses.  That extra margin stopped
        # being available when `ZBL` acquired its taper and the couplings were
        # refit -- no channel on that surface recrossed on any seed, at any
        # length of run or temperature tried.  `tests/geometry.py`
        # :SWITCHING_REACTION carries the survey.  Unlike
        # `test_energy_conservation.py:TestTopologyChangeContinuity`, this test
        # runs a single seed and cannot pool its way back over a floor of two.
        #
        # **Recrossing came back when `forcefield/lj.py` did**, and this floor
        # deliberately did not follow it up.  Measured under exactly the protocol
        # above -- seed 0, 0.1 fs, 1000 K -- the run switches **twice**, so a
        # floor of two would pass on a margin of exactly zero and would fail on
        # the next refit that moved anything.  One is what the assertion needs
        # and one is what it asks for; the recrossing is recorded in
        # `geometry.py` and guarded there, where three seeds are pooled.
        assert switches >= 1, (
            f"only {switches} topology switches; this test asserts nothing unless "
            "the trajectory actually changes bonding"
        )

        frames = io.read(target, index=":")
        stored = [
            frozenset(frozenset(bond[:2]) for bond in f.info["connectivity"])
            for f in frames
        ]
        assert len(set(stored)) > 1, (
            "every frame stored the same bonding, so the trajectory records the "
            "topology it started with rather than the one it reached"
        )

    def test_a_written_frame_round_trips_to_the_same_topology(
        self, reaction_set, tmp_path
    ):
        """Recording is only worth anything if reading it back agrees.

        This is what `--restart` depends on: re-reading the last frame must
        resume the chemistry the run reached, not re-perceive it from geometry.
        """
        atoms = reaction_path(SWITCHING_REACTION, SWITCHING_PATH_START, CELL)
        atoms.calc = DynamicTopology(atoms, reaction_set)
        atoms.get_potential_energy()
        nvt.record_topology(atoms)

        target = tmp_path / "one.xyz"
        io.write(target, atoms, format="extxyz")
        reloaded = io.read(target)

        before = Topology.from_atoms(atoms).hash()
        after = Topology.from_atoms(reloaded).hash()
        assert before == after

    def test_recording_is_a_no_op_without_a_calculator(self):
        """`nvt.py` also drives the fixed-state EVB calculator, which has none."""
        atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=np.eye(3) * 10)
        nvt.record_topology(atoms)  # must not raise
        assert "connectivity" not in atoms.info


class TestSpeciesCounting:
    """The histogram the sweep is analysed from."""

    def test_species_counts_match_the_packed_composition(self, reaction_set):
        """A box packed as 50 H2 + 50 O2 must be counted as 50 H2 + 50 O2.

        Guards the cutoff bug from the other side: molify's default `scale=1.2`
        puts H-H at 0.7440 A against an 0.7445 A bond, and reported `H: 100`
        here before `BOND_SCALE`.
        """
        atoms = io.read("tests/data/mix-n100-d30.xyz")
        atoms.calc = DynamicTopology(atoms, reaction_set)
        atoms.get_potential_energy()
        assert nvt.species_counts(atoms) == {"H2": 50, "O2": 50}

    def test_species_counts_survive_losing_the_stored_connectivity(self):
        """A frame with no connectivity must still count the same species."""
        atoms = io.read("tests/data/mix-n100-d30.xyz")
        bare = atoms.copy()
        bare.info.pop("connectivity", None)
        assert (
            nvt.species_counts(bare)
            == nvt.species_counts(atoms)
            == {
                "H2": 50,
                "O2": 50,
            }
        )


class TestSeeding:
    """A production run must be re-derivable from its config."""

    @staticmethod
    def _momenta(seed: int | None) -> np.ndarray:
        """Initial momenta drawn exactly as `nvt.py` draws them."""
        atoms = io.read("tests/data/mix-n100-d30.xyz")
        thermal, _ = (
            np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)
        )
        thermalize_momenta(atoms, 3000.0, rng=thermal)
        return atoms.get_momenta()

    def test_the_same_seed_gives_the_same_start(self):
        np.testing.assert_array_equal(self._momenta(7), self._momenta(7))

    def test_a_different_seed_gives_a_different_start(self):
        """Pinned in both directions, so it cannot pass by ignoring the seed."""
        assert not np.array_equal(self._momenta(7), self._momenta(8))

    def test_both_random_streams_are_seeded(self):
        """Seeding only the initial draw is not enough.

        Langevin adds a random force every step, so two runs from identical
        momenta still diverge.  `nvt.py` spawns both streams from one seed; this
        checks the two are distinct, which is what makes spawning the right tool
        rather than passing one generator to both.
        """
        thermal, langevin = (
            np.random.default_rng(s) for s in np.random.SeedSequence(7).spawn(2)
        )
        assert not np.array_equal(thermal.random(16), langevin.random(16))

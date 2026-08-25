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
from ase import Atoms, io

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet

RSET_PATH = Path("datasets/HCombustion/HCombustion.json").resolve()

# Isolated-molecule box.  Large enough that nothing sees its own image, and
# `pbc` is off, so this is a vacuum calculation to match the reference.
CELL = 30.0

# Templates whose atoms are not all the same element, and which therefore carry
# a nonzero ACKS2 energy.  Named rather than derived so that a template quietly
# losing its charges cannot make this file pass by making the question vacuous.
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


class TestPerceivedTopology:
    """What the calculator sees when it is handed a plain geometry.

    Every template is a molecule the database knows, so perceiving its bonds
    from its own equilibrium geometry has exactly one right answer.  Getting a
    different one means a trajectory frame -- which carries no connectivity --
    is scored as a different chemical species than the one that is there.
    """

    def test_every_template_perceives_as_itself(self, reaction_set):
        from DynamicTopology.core import Topology

        wrong = {}
        for stem in _entries("molecules"):
            frame = io.read(stem.with_suffix(".xyz"))
            stored = Topology.from_atoms(isolate(frame, False))
            perceived = Topology.from_atoms(isolate(frame, True))
            if stored.hash() != perceived.hash():
                wrong[stem.name] = (
                    f"{frame.get_chemical_formula()}: stored "
                    f"{sorted(map(sorted, stored.graph.edges()))} vs perceived "
                    f"{sorted(map(sorted, perceived.graph.edges()))}"
                )
        assert not wrong, "templates do not perceive as themselves:\n  " + (
            "\n  ".join(f"{k}  {v}" for k, v in sorted(wrong.items()))
        )

    @pytest.mark.parametrize(
        "path", ["tests/data/mix-n100-d30.xyz", "tests/data/mix-n100-d250.xyz"]
    )
    def test_a_packed_box_perceives_the_molecules_it_was_packed_from(self, path):
        """`molify.pack` wrote the connectivity; perception must agree with it.

        A packed box is the strongest available statement of ground truth about
        bonding: the molecules were placed as molecules, so the stored
        connectivity is what is there by construction, not an opinion about a
        geometry.
        """
        from DynamicTopology.core import Topology

        frame = io.read(path)
        stored = len(frame.info["connectivity"])
        # Its own cell and periodicity, not `isolate`'s vacuum box: the point of
        # a packed box is that molecules straddle the boundary.
        loose = frame.copy()
        loose.calc = None
        loose.info.pop("connectivity", None)
        perceived = Topology.from_atoms(loose)
        assert len(perceived.graph.edges()) == stored, (
            f"{path} was packed with {stored} bonds but perceives "
            f"{len(perceived.graph.edges())}"
        )

"""`forcefield.evaluate` is the energy `System` gives a lone molecule.

The single-topology sum is assembled in one place so that a fitter scores a
template through exactly the terms the calculator will run it with.  This holds
it to that, template by template, for every dataset -- open and periodic, so
the charge kernel's lattice sum and the virial are both in it.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.forcefield.evaluate import evaluate
from DynamicTopology.forcefield.params import ForceFieldParams, use
from DynamicTopology.system import System

MANIFESTS = [
    "datasets/HCombustion/HCombustion.json",
    "datasets/Water/Water.json",
    "datasets/Water-fixed-pc/Water.json",
]


def _params(manifest: str) -> ForceFieldParams:
    return ForceFieldParams.from_dict(
        json.loads(Path(manifest).read_text())["global_params"]
    )


def _templates(manifest: str):
    with use(_params(manifest)):
        reaction_set = ReactionSet(manifest)
    return reaction_set, list(reaction_set.data.molecules.values())


@pytest.mark.parametrize("manifest", MANIFESTS, ids=lambda m: Path(m).parent.name)
@pytest.mark.parametrize("periodic", [False, True], ids=["open", "periodic"])
def test_a_lone_template_scores_as_system_scores_it(manifest, periodic):
    reaction_set, templates = _templates(manifest)
    with use(_params(manifest)):
        for template in templates:
            atoms = template.atoms.copy()
            atoms.calc = None
            atoms.set_cell([14.0, 14.5, 15.0])
            atoms.center()
            atoms.pbc = periodic
            # One state: the lone molecule's own diabat, whatever unimolecular
            # channel the network would otherwise offer it.
            system = System(
                atoms,
                Topology.from_terms(template.terms, atoms),
                reaction_set,
                evb={"max_states": 1},
            )
            expected = system.calculate()
            got = evaluate(atoms, template.terms)
            name = atoms.get_chemical_formula()
            assert got.energy == pytest.approx(expected["energy"], abs=1e-10), name
            np.testing.assert_allclose(
                got.forces, expected["forces"], atol=1e-10, err_msg=name
            )
            np.testing.assert_allclose(
                got.virial, expected["virial"], atol=1e-10, err_msg=name
            )
            assert got.bonded + got.electrostatics + got.zbl + got.lj == (
                pytest.approx(got.energy, abs=1e-12)
            )


def test_exclusions_are_derived_when_the_terms_omit_them():
    """A raw `.jsonl`'s terms score like the loaded template's."""
    manifest = MANIFESTS[1]
    reaction_set, templates = _templates(manifest)
    with use(_params(manifest)):
        for template in templates:
            raw = [t for t in template.terms if not t["type"].endswith("exclusion")]
            assert evaluate(template.atoms, raw).energy == pytest.approx(
                evaluate(template.atoms, template.terms).energy, abs=1e-12
            )

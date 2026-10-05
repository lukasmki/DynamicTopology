"""The optional spherical cutoff on the 12-6 (`params.lj_cutoff`).

Off by default, and every dataset was fitted without it; these hold the cut
form to its own gradients, and its tail correction to the all-pairs sum it
estimates.
"""

import dataclasses

import numpy as np
import pytest
from ase import io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.forcefield.lj import LennardJones, cutoff_switch
from DynamicTopology.forcefield.params import active, use

BOX = "production/density-300K/inputs/w64.xyz"  # 64 waters, 12.43 A cube


@pytest.fixture(scope="module")
def water():
    atoms = io.read(BOX)
    reaction_set = ReactionSet("datasets/Water/Water.json")
    topology = Topology.from_atoms(atoms)
    topology.set_terms(reaction_set.get_terms(topology))
    return atoms, topology.term_dict


def _cut(rc):
    return use(dataclasses.replace(active(), lj_cutoff=rc))


def test_the_switch_is_smooth_and_bounded():
    r = np.linspace(4.0, 7.0, 301)
    s, ds = cutoff_switch(r, 5.0, 6.0)
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert np.all(s[r <= 5.0] == 1.0) and np.all(s[r >= 6.0] == 0.0)
    h = 1e-6
    numeric = (cutoff_switch(r + h, 5.0, 6.0)[0] - cutoff_switch(r - h, 5.0, 6.0)[0]) / (
        2 * h
    )
    inside = (np.abs(r - 5.0) > 2 * h) & (np.abs(r - 6.0) > 2 * h)
    np.testing.assert_allclose(ds[inside], numeric[inside], atol=1e-6)


def test_forces_and_virial_are_the_gradient(water):
    atoms, term_dict = water
    pos, pbc, cell = atoms.positions, atoms.pbc, atoms.cell.array
    lj = LennardJones()
    with _cut(6.0):
        energy, forces, virial = lj(pos, pbc, cell, term_dict)
        h = 1e-5
        for a in range(0, len(pos), 37):
            for k in range(3):
                p = pos.copy()
                p[a, k] += h
                up = lj(p, pbc, cell, term_dict)[0]
                p[a, k] -= 2 * h
                down = lj(p, pbc, cell, term_dict)[0]
                assert -(up - down) / (2 * h) == pytest.approx(forces[a, k], abs=1e-7)
        e = 1e-6
        strained = [
            lj(pos * (1 + s), pbc, cell * (1 + s), term_dict)[0] for s in (e, -e)
        ]
        assert (strained[0] - strained[1]) / (2 * e) == pytest.approx(
            np.trace(virial), rel=1e-7
        )


def test_the_tail_correction_estimates_what_the_cutoff_drops(water):
    """Two cutoffs agree with each other far better than with no correction.

    What a cutoff drops from the all-pairs sum is ~0.25 eV over this box; the
    tail correction puts back an estimate of it, and the estimates from a 5 A
    and a 6 A cutoff differ by a few hundredths of that.  The remaining offset
    from the all-pairs sum is not an error of the cut form: the minimum image
    stops at the cell, the tail correction does not.
    """
    atoms, term_dict = water
    args = (atoms.positions, atoms.pbc, atoms.cell.array, term_dict)
    lj = LennardJones()
    with _cut(5.0):
        five = lj(*args)[0]
    with _cut(6.0):
        six = lj(*args)[0]
    assert abs(five - six) < 0.05


def test_a_cutoff_past_half_the_cell_is_refused(water):
    atoms, term_dict = water
    with _cut(6.3), pytest.raises(ValueError, match="perpendicular width"):
        LennardJones()(atoms.positions, atoms.pbc, atoms.cell.array, term_dict)


def test_no_cutoff_is_the_default():
    assert active().lj_cutoff is None

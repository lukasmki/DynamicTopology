"""The neighbour list every pair term reads (`forcefield/neighbors.py`).

`Geometry.pairs` must select exactly the pairs `rij < rc` selects from the
dense minimum image -- whichever search it takes, k-d tree or cell list -- and
measure each one with the same numbers; and a large periodic box under a
cutoff and the iterative charge solve must never build the `N x N` arrays at
all, which is what the neighbour list is for.
"""

import dataclasses

import numpy as np
import pytest
from ase import io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.forcefield import pointcharge
from DynamicTopology.forcefield.neighbors import (
    Geometry,
    NeighborList,
    list_radius,
    pair_gradients,
)
from DynamicTopology.forcefield.params import active, use
from DynamicTopology.system import System

BOX = "production/density-300K/inputs/w64.xyz"  # 64 waters, 12.43 A cube

CELLS = {
    "cube": np.eye(3) * 12.43,
    "orthorhombic": np.diag([20.0, 13.0, 30.0]),
    "triclinic": np.array([[15.0, 0.0, 0.0], [4.0, 14.0, 0.0], [-3.0, 5.0, 16.0]]),
    "skewed": np.array([[25.0, 0.0, 0.0], [9.0, 22.0, 0.0], [-6.0, 7.0, 24.0]]),
}


def _random(cell, n, seed=0):
    rng = np.random.default_rng(seed)
    # Some atoms well outside the cell, as an unwrapped trajectory has them.
    return rng.random((n, 3)) @ cell + rng.normal(scale=3.0, size=(n, 3))


def _as_set(i, j):
    return set(zip(np.asarray(i).tolist(), np.asarray(j).tolist()))


@pytest.mark.parametrize("name", list(CELLS))
@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_the_pairs_are_the_dense_selection(name, fraction):
    cell = CELLS[name]
    pos = _random(cell, 400)
    widths = 1.0 / np.linalg.norm(np.linalg.inv(cell), axis=0)
    rc = fraction * widths.min()
    geometry = Geometry(pos, [True] * 3, cell)
    assert geometry._cell_list_valid(rc)
    pairs = geometry.pairs(rc)
    assert not geometry.has_dense, "the search fell back to the dense array"
    assert np.all(pairs.i < pairs.j)

    vecs, rij = pointcharge.geometry(pos, [True] * 3, cell)
    i, j = np.nonzero(np.triu(rij < rc, 1))
    assert _as_set(pairs.i, pairs.j) == _as_set(i, j)
    order = np.lexsort((pairs.j, pairs.i))
    np.testing.assert_allclose(pairs.v[order], vecs[i, j], rtol=0, atol=1e-12)
    np.testing.assert_allclose(pairs.r[order], rij[i, j], rtol=0, atol=1e-12)


def test_the_cell_list_agrees_with_the_tree():
    """An orthorhombic cell takes the tree; the cell list must find the same pairs."""
    cell = CELLS["orthorhombic"]
    pos = _random(cell, 600, seed=3)
    geometry = Geometry(pos, [True] * 3, cell)
    tree = geometry._tree(6.0)
    grid = geometry._cell_list(6.0)
    assert _as_set(tree.i, tree.j) == _as_set(grid.i, grid.j)


def test_a_smaller_cutoff_is_cut_from_the_list():
    cell = CELLS["triclinic"]
    pos = _random(cell, 300, seed=1)
    geometry = Geometry(pos, [True] * 3, cell)
    wide = geometry.pairs(6.0)
    narrow = geometry.pairs(4.0)
    assert set(geometry._lists) == {6.0, 4.0}
    fresh = Geometry(pos, [True] * 3, cell).pairs(4.0)
    assert _as_set(narrow.i, narrow.j) == _as_set(fresh.i, fresh.j)
    assert len(narrow) < len(wide)


@pytest.mark.parametrize("pbc", [[False] * 3, [True, True, False]])
def test_open_boundaries_cut_from_the_dense_array(pbc):
    cell = CELLS["cube"]
    pos = _random(cell, 50, seed=2)
    geometry = Geometry(pos, pbc, cell)
    pairs = geometry.pairs(5.0)
    vecs, rij = pointcharge.geometry(pos, pbc, cell)
    i, j = np.nonzero(np.triu(rij < 5.0, 1))
    np.testing.assert_array_equal(pairs.i, i)
    np.testing.assert_array_equal(pairs.j, j)
    np.testing.assert_array_equal(pairs.r, rij[i, j])


def test_every_pair_without_a_cutoff():
    cell = CELLS["cube"]
    pos = _random(cell, 30, seed=4)
    pairs = Geometry(pos, [True] * 3, cell).pairs(None)
    assert len(pairs) == 30 * 29 // 2


@pytest.mark.parametrize("name", ["cube", "skewed"])
def test_between_is_the_dense_minimum_image(name):
    cell = CELLS[name]
    pos = _random(cell, 40, seed=5)
    vecs, rij = pointcharge.geometry(pos, [True] * 3, cell)
    i, j = np.triu_indices(40, 1)
    v, r = Geometry(pos, [True] * 3, cell).between(i, j)
    np.testing.assert_allclose(v, vecs[i, j], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r, rij[i, j], rtol=0, atol=1e-12)


def test_pair_gradients_are_the_gradient():
    """A pair sum's forces and virial against finite differences."""
    cell = CELLS["triclinic"]
    pos = _random(cell, 60, seed=6)

    def energy(p, c):
        pairs = Geometry(p, [True] * 3, c).pairs(5.0)
        return float(np.sum((5.0 - pairs.r) ** 3)), pairs

    _, pairs = energy(pos, cell)
    du = -3.0 * (5.0 - pairs.r) ** 2
    forces, virial = pair_gradients(pairs.i, pairs.j, du, pairs.v, pairs.r, len(pos))
    h = 1e-6
    for a in (0, 17, 41):
        for k in range(3):
            p = pos.copy()
            p[a, k] += h
            up = energy(p, cell)[0]
            p[a, k] -= 2 * h
            down = energy(p, cell)[0]
            assert -(up - down) / (2 * h) == pytest.approx(forces[a, k], abs=1e-6)
    strain = np.zeros((3, 3))
    strain[0, 1] = strain[1, 0] = h
    up = energy(pos @ (np.eye(3) + strain), cell @ (np.eye(3) + strain))[0]
    down = energy(pos @ (np.eye(3) - strain), cell @ (np.eye(3) - strain))[0]
    assert (up - down) / (2 * h) == pytest.approx(2 * virial[0, 1], rel=1e-6)


def test_a_large_box_never_builds_the_dense_geometry(monkeypatch):
    """Under `lj_cutoff` and the iterative solve every pair term reads the list.

    The 64-water box replicated 2x2x2: half its width is 12.4 A, past every
    cutoff (ZBL's 6.9, the 12-6's 6, the real-space 9), so nothing may need
    the `N x N` minimum image.
    """
    base = io.read(BOX)
    atoms = base.repeat((2, 2, 2))
    n0 = len(base)
    atoms.info["connectivity"] = [
        [i + k * n0, j + k * n0, o]
        for k in range(8)
        for i, j, o in base.info["connectivity"]
    ]
    reaction_set = ReactionSet("datasets/Water/Water.json")
    ff = dataclasses.replace(active(), lj_cutoff=6.0, charge_solver="iterative")

    def refuse(*args, **kwargs):
        raise AssertionError("the dense N x N geometry was built")

    monkeypatch.setattr(pointcharge, "geometry", refuse)
    with use(ff):
        system = System(atoms, Topology.from_atoms(atoms), reaction_set)
        results = system.calculate()
    assert np.isfinite(results["energy"])


# -- the shared list, kept across calls ----------------------------------------


def _params(**overrides):
    return use(dataclasses.replace(active(), **overrides))


@pytest.mark.parametrize("name", ["orthorhombic", "skewed"])
def test_a_kept_list_finds_every_pair(name):
    """A random walk, the cell breathing as under NPT: every call's pairs are a
    fresh search's, and the list is searched again only now and then."""
    cell0 = CELLS[name] * 1.4
    rng = np.random.default_rng(8)
    frac = rng.random((900, 3))
    neighbors = NeighborList()
    with _params(neighbor_radius=6.0, neighbor_skin=1.5):
        for step in range(40):
            frac += rng.normal(scale=0.002, size=frac.shape)
            cell = cell0 * (1.0 + 0.01 * np.sin(step / 3.0))
            pos = frac @ cell
            got = Geometry(pos, [True] * 3, cell, neighbors=neighbors).pairs(6.0)
            fresh = Geometry(pos, [True] * 3, cell).pairs(6.0)
            assert _as_set(got.i, got.j) == _as_set(fresh.i, fresh.j)
    assert neighbors.reuses > 0 and neighbors.builds > 1


def test_a_wrapped_atom_has_not_moved():
    cell = CELLS["orthorhombic"] * 1.4
    pos = _random(cell, 500, seed=9)
    neighbors = NeighborList()
    with _params(neighbor_radius=6.0, neighbor_skin=1.0):
        first = Geometry(pos, [True] * 3, cell, neighbors=neighbors).pairs(6.0)
        wrapped = pos - np.floor(pos @ np.linalg.inv(cell)) @ cell
        again = Geometry(wrapped, [True] * 3, cell, neighbors=neighbors).pairs(6.0)
    assert neighbors.builds == 1 and neighbors.reuses == 1
    assert _as_set(first.i, first.j) == _as_set(again.i, again.j)


def test_a_radius_short_of_a_cutoff_is_refused():
    cell = CELLS["orthorhombic"] * 1.4
    geometry = Geometry(_random(cell, 100), [True] * 3, cell, neighbors=NeighborList())
    with _params(neighbor_radius=5.0), pytest.raises(ValueError, match="neighbor_radius"):
        geometry.pairs(6.0)


def test_the_default_radius_is_the_widest_cutoff():
    cell = np.eye(3) * 40.0
    with _params(lj_cutoff=10.0, charge_solver="iterative"):
        assert list_radius(active(), cell) == 10.0
    with _params(lj_cutoff=None, charge_solver="direct"):
        ff = active()
        assert list_radius(ff, cell) == pytest.approx(ff.taper_radius + 45 * ff.taper_width)
    with _params(neighbor_radius=12.0):
        assert list_radius(active(), cell) == 12.0


@pytest.mark.parametrize("bad", [{"neighbor_skin": -0.1}, {"neighbor_radius": 0.0}])
def test_the_parameters_are_checked(bad):
    with pytest.raises(ValueError):
        dataclasses.replace(active(), **bad)


def test_a_skin_changes_nothing_but_the_searches():
    """Sequential calls on a 1536-atom box: with a skin, the same energies and
    forces as searching every call, to rounding, from fewer searches."""
    base = io.read(BOX)
    n0 = len(base)
    reaction_set = ReactionSet("datasets/Water/Water.json")

    def run(skin):
        atoms = base.repeat((2, 2, 2))
        atoms.info["connectivity"] = [
            [i + k * n0, j + k * n0, o]
            for k in range(8)
            for i, j, o in base.info["connectivity"]
        ]
        rng = np.random.default_rng(0)
        out = []
        with _params(lj_cutoff=6.0, charge_solver="iterative", neighbor_skin=skin):
            system = System(atoms, Topology.from_atoms(atoms), reaction_set)
            for _ in range(4):
                results = system.calculate()
                out.append((results["energy"], results["forces"].copy()))
                system.update(topology=results["topology"])
                atoms.positions += rng.normal(scale=0.01, size=atoms.positions.shape)
        return out, system.neighbors

    plain, searched = run(0.0)
    skinned, kept = run(1.0)
    assert searched.builds == 4
    assert kept.builds == 1 and kept.reuses == 3
    for (e0, f0), (e1, f1) in zip(plain, skinned):
        assert e1 == pytest.approx(e0, abs=1e-9)
        np.testing.assert_allclose(f1, f0, rtol=0, atol=1e-10)

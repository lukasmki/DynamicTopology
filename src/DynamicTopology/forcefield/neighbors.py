"""Minimum-image pair geometry, as neighbour lists rather than `N x N` arrays.

Every pair term used to share one `pointcharge.geometry`: the minimum-image
displacement of every pair of atoms, `(N, N, 3)`.  That is `O(N^2)` time and
memory before a single pair energy is evaluated, and at 5184 atoms it is 0.9
GB -- although every term that can be cut off reads only the pairs inside its
cutoff, and the rest read a few pairs inside each molecule.

`Geometry` is the same minimum image, built on demand for what is asked of it:

    geometry.pairs(rc)     -> `Pairs`: every `i < j` with `r_ij < rc`
    geometry.between(i, j) -> `(v, r)` for any given pairs, e.g. within a molecule
    geometry.dense()       -> `(vecs, rij)`, the `N x N` arrays, for a term that
                              really is all-pairs (the dense Ewald kernel, open
                              boundaries, the 12-6 without `lj_cutoff`)

**`pairs` is a neighbour search** whenever the cell is periodic in all three
directions and `rc` is at most half its shortest perpendicular width -- which
is exactly when the image of a pair inside `rc` is unique, so the search and
the minimum image agree on it.  An orthorhombic cell goes to a periodic k-d
tree (`scipy.spatial.cKDTree`); any other to a cell list, the atoms binned on
their fractional coordinates into bins at least `rc` wide across every face
and each bin paired with its neighbours.  Either is `O(N log N)` or better,
and at 5184 atoms and 9 A they take 0.1 and 0.6 s.  Otherwise --
open or partly periodic boundaries, a cutoff past half the cell, or no cutoff
at all -- it is cut from `dense()`, as every term used to do.

Whichever way the pairs are found, each one's displacement is computed by the
same arithmetic `pointcharge.geometry` uses, `v - round(v . cell^-1) . cell`,
so a pair's `v` and `r` do not depend on how it was found, and the pair set is
the one `rij < rc` selects from the dense array.  Only the *order* of the pairs
differs between the two routes, which moves a sum by rounding.

One `Geometry` is built per force call (`System.calculate`) and shared by every
term.  Given a `NeighborList` -- `System` keeps one for its lifetime -- the
terms read one list, searched at `global_params.neighbor_radius` (by default
the widest cutoff any term reads from it) plus `neighbor_skin`, and kept
across force calls while no atom can have crossed the skin; each term's own
cutoff is filtered from it.  Without one, as for a lone evaluation, the list
is searched at the first cutoff asked for and the smaller ones filtered from
it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from DynamicTopology.forcefield.params import ForceFieldParams, active

# A cutoff may equal half the cell's width to within this, relative; the
# validity checks of `lj._check_cutoff` and `ewald.EwaldOperatorSetup` compute
# the same width by a different route.
_WIDTH_SLACK: float = 1e-12


@dataclass(frozen=True)
class Pairs:
    """Pairs `(i[p], j[p])`, `i < j`, with `v = r_i - r_j` (minimum image) and `|v|`.

    `cutoff` is the radius they were selected inside, `None` for every pair.
    """

    i: np.ndarray
    j: np.ndarray
    v: np.ndarray
    r: np.ndarray
    cutoff: float | None

    def __len__(self) -> int:
        return len(self.i)

    def within(self, cutoff: float) -> "Pairs":
        """The pairs closer than `cutoff`, which must not exceed `self.cutoff`."""
        if self.cutoff is not None and cutoff > self.cutoff:
            raise ValueError(f"cutoff {cutoff} exceeds the list's {self.cutoff}")
        keep = self.r < cutoff
        return Pairs(self.i[keep], self.j[keep], self.v[keep], self.r[keep], cutoff)


def perpendicular_widths(cell) -> np.ndarray:
    """The distance between each pair of opposite faces of `cell`."""
    return 1.0 / np.linalg.norm(np.linalg.inv(np.asarray(cell, dtype=float)), axis=0)


def list_radius(ff: ForceFieldParams, cell) -> float:
    """The radius the shared list must cover: `neighbor_radius`, or every cutoff.

    Without `neighbor_radius`, the widest cutoff read from the list under `ff`:
    ZBL's reach, `lj_cutoff`, and the iterative solve's real-space cutoff.
    """
    if ff.neighbor_radius is not None:
        return ff.neighbor_radius
    from DynamicTopology.forcefield.zbl import TAPER_TAIL

    radii = [ff.taper_radius + TAPER_TAIL * ff.taper_width]
    if ff.lj_cutoff is not None:
        radii.append(ff.lj_cutoff)
    if ff.charge_solver == "iterative":
        half = 0.5 * perpendicular_widths(cell).min()
        radii.append(min(ff.real_space_cutoff, half))
    return max(radii)


class NeighborList:
    """The candidate pairs of a periodic box, kept across force calls.

    Searched at `radius + skin`; reused while every pair now inside `radius`
    must have been inside `radius + skin` at the search.  With `d` the largest
    displacement of any atom since then and `e` the cell's strain, a pair's
    separation has changed by at most `2 d + e |v|`, so the list still holds
    every pair inside `radius` while

        2 d + e (radius + skin) <= skin.

    Displacements are taken in fractional coordinates less whole lattice
    vectors, so an atom wrapped back into the cell has not moved; the strain
    is the spectral norm of `H0^-1 H - I`, so an NPT step that rescales the
    cell does not by itself force a search.  The skin is shortened where
    `radius + skin` would pass half the cell's width, beyond which the
    minimum image is no longer the only image inside it.

    `builds` and `reuses` count the searches and the calls that skipped one.
    """

    def __init__(self):
        self._state = None
        self.builds = 0
        self.reuses = 0

    def candidates(self, geometry, radius: float, skin: float):
        """`(i, j)`, `i < j`: a superset of the pairs inside `radius`, or `None`.

        `None` where the list cannot be used: a cell not periodic in all three
        directions, or `radius` past half its width.
        """
        if not geometry.periodic or geometry.natoms < 2:
            return None
        half = 0.5 * perpendicular_widths(geometry.cell).min() * (1.0 + _WIDTH_SLACK)
        if radius > half:
            return None
        reach = min(radius + skin, half)
        skin = reach - radius
        frac = geometry.pos @ geometry._inverse
        state = self._state
        if (
            state is not None
            and state["natoms"] == geometry.natoms
            and state["radius"] == radius
            and state["skin"] == skin
            and self._fresh(state, frac, geometry.cell, reach, skin)
        ):
            self.reuses += 1
            return state["i"], state["j"]
        found = geometry._search(reach)
        self._state = {
            "natoms": geometry.natoms,
            "radius": radius,
            "skin": skin,
            "frac": frac,
            "cell": geometry.cell.copy(),
            "i": found.i,
            "j": found.j,
        }
        self.builds += 1
        return found.i, found.j

    @staticmethod
    def _fresh(state, frac, cell, reach, skin) -> bool:
        moved = frac - state["frac"]
        moved -= np.round(moved)
        d = float(np.sqrt(np.max(np.sum((moved @ cell) ** 2, axis=1))))
        strain = np.linalg.solve(state["cell"], cell) - np.eye(3)
        e = float(np.linalg.norm(strain, 2))
        return 2.0 * d + e * reach <= skin


class Geometry:
    """The minimum-image geometry of one configuration.  See the module docstring.

    `neighbors` is the caller's `NeighborList`, kept across force calls; with
    one, every cutoff is read off the shared list (`_shared`).
    """

    def __init__(self, pos, pbc, cell, dense=None, neighbors: NeighborList | None = None):
        self.pos = np.asarray(pos, dtype=float)
        self.pbc = np.asarray(pbc, dtype=bool)
        self.cell = np.asarray(cell, dtype=float)
        self.natoms = len(self.pos)
        self.periodic = bool(np.all(self.pbc))
        self._inverse = (
            np.linalg.inv(self.cell) if np.any(self.pbc) else None
        )
        self._dense = dense
        self.neighbors = neighbors
        # The pair lists built so far, by cutoff; the widest serves the rest.
        self._lists: dict[float | None, Pairs] = {}
        # The shared list's candidates, measured at this geometry.
        self._measured = None

    # -- the minimum image ----------------------------------------------------

    def _wrap(self, v: np.ndarray) -> np.ndarray:
        """`v` moved to its minimum image, as `pointcharge.geometry` moves it."""
        if self._inverse is None:
            return v
        F = v @ self._inverse
        return v - (self.pbc * np.floor(F + 0.5)) @ self.cell

    def between(self, i, j) -> tuple[np.ndarray, np.ndarray]:
        """`(v, r)` of the pairs `(i[p], j[p])`, any shape, with `v = r_i - r_j`."""
        i, j = np.asarray(i), np.asarray(j)
        if self._dense is not None:
            vecs, rij = self._dense
            return vecs[i, j], rij[i, j]
        v = self._wrap(self.pos[i] - self.pos[j])
        return v, np.sqrt(np.sum(v * v, -1))

    def dense(self) -> tuple[np.ndarray, np.ndarray]:
        """`(vecs, rij)` over every pair, `(N, N, 3)` and `(N, N)`; built once."""
        if self._dense is None:
            from DynamicTopology.forcefield.pointcharge import geometry

            self._dense = geometry(self.pos, self.pbc, self.cell)
        return self._dense

    @property
    def has_dense(self) -> bool:
        return self._dense is not None

    # -- pair lists -----------------------------------------------------------

    def pairs(self, cutoff: float | None = None) -> Pairs:
        """Every pair `i < j` closer than `cutoff`, or every pair if `None`."""
        cached = self._lists.get(cutoff)
        if cached is not None:
            return cached
        if cutoff is not None and self.neighbors is not None:
            out = self._shared(cutoff)
            if out is not None:
                self._lists[cutoff] = out
                return out
        if cutoff is not None:
            wider = [c for c in self._lists if c is not None and c >= cutoff]
            if wider:
                out = self._lists[min(wider)].within(cutoff)
                self._lists[cutoff] = out
                return out
        if cutoff is None:
            vecs, rij = self.dense()
            i, j = np.triu_indices(self.natoms, 1)
            out = Pairs(i, j, vecs[i, j], rij[i, j], None)
        elif self._cell_list_valid(cutoff):
            out = self._search(cutoff)
        else:
            vecs, rij = self.dense()
            i, j = np.nonzero(np.triu(rij < cutoff, 1))
            out = Pairs(i, j, vecs[i, j], rij[i, j], cutoff)
        self._lists[cutoff] = out
        return out

    def _shared(self, cutoff: float) -> Pairs | None:
        """The pairs inside `cutoff`, off the shared `NeighborList`, or `None`."""
        if not self.periodic:
            return None
        ff = active()
        radius = list_radius(ff, self.cell)
        if cutoff > radius * (1.0 + _WIDTH_SLACK):
            raise ValueError(
                f"a pair term reads the neighbour list to {cutoff:g} A, past its "
                f"radius of {radius:g} A; raise `neighbor_radius` to at least "
                "the largest cutoff in force (ZBL's reach, `lj_cutoff`, and the "
                "iterative solve's `real_space_cutoff`), or leave it unset"
            )
        if self._measured is None:
            candidates = self.neighbors.candidates(self, radius, ff.neighbor_skin)
            if candidates is None:
                return None
            i, j = candidates
            self._measured = (i, j, *self.between(i, j))
        i, j, v, r = self._measured
        keep = r < cutoff
        return Pairs(i[keep], j[keep], v[keep], r[keep], cutoff)

    def _search(self, cutoff: float) -> Pairs:
        """The pairs inside `cutoff` by a neighbour search; `_cell_list_valid` must hold."""
        if np.count_nonzero(self.cell - np.diag(np.diag(self.cell))) == 0:
            return self._tree(cutoff)
        return self._cell_list(cutoff)

    def _cell_list_valid(self, cutoff: float) -> bool:
        if not self.periodic or self.natoms < 2:
            return False
        half = 0.5 * perpendicular_widths(self.cell).min()
        return cutoff <= half * (1.0 + _WIDTH_SLACK)

    def _tree(self, cutoff: float) -> Pairs:
        """The pairs inside `cutoff` for an orthorhombic cell, by a periodic k-d tree.

        `scipy.spatial.cKDTree` with `boxsize` is the minimum image of a
        rectangular box, in compiled code -- several times faster than
        `_cell_list`, which handles any cell.  The tree only proposes the
        candidates, over a slightly wider radius; each is then measured as
        `between` measures it, so the cut is made on the same numbers.
        """
        from scipy.spatial import cKDTree

        lengths = np.diag(self.cell)
        frac = self.pos / lengths
        frac -= np.floor(frac)
        # `x - floor(x)` can round up to 1.0, which the tree refuses.
        wrapped = np.minimum(frac * lengths, np.nextafter(lengths, 0.0))
        tree = cKDTree(wrapped, boxsize=lengths)
        found = tree.query_pairs(cutoff * (1.0 + 1e-9) + 1e-9, output_type="ndarray")
        i, j = found[:, 0], found[:, 1]
        # `query_pairs` returns `i < j` already; restated, as `Pairs` promises it.
        i, j = np.minimum(i, j), np.maximum(i, j)
        v, r = self.between(i, j)
        keep = r < cutoff
        return Pairs(i[keep], j[keep], v[keep], r[keep], cutoff)

    def _cell_list(self, cutoff: float) -> Pairs:
        """The pairs inside `cutoff` by binning, for a fully periodic cell.

        A pair whose minimum image lies inside `cutoff` is apart by less than
        `cutoff / width_d` in fractional coordinate `d`, and every bin is at
        least that wide, so the two atoms sit in the same bin or in adjacent
        ones (across the periodic boundary included).  Each atom is paired
        with every atom of the bins around its own, each unordered pair is
        kept once as `i < j`, and the candidates are then measured exactly.
        """
        n = self.natoms
        frac = self.pos @ self._inverse
        frac -= np.floor(frac)
        nbins = np.maximum(
            np.floor(perpendicular_widths(self.cell) / cutoff).astype(int), 1
        )
        b = np.minimum((frac * nbins).astype(int), nbins - 1)
        flat = (b[:, 0] * nbins[1] + b[:, 1]) * nbins[2] + b[:, 2]
        order = np.argsort(flat, kind="stable")
        counts = np.bincount(flat, minlength=int(np.prod(nbins)))
        starts = np.cumsum(counts) - counts
        # The distinct neighbouring bins along each axis: with fewer than three
        # bins, -1 and +1 are the same bin, or the bin itself, and a pair of
        # bins must be visited once.
        shifts = [np.unique(np.array([-1, 0, 1]) % m) for m in nbins]
        atoms = np.arange(n)
        I, J = [], []
        for s0 in shifts[0]:
            for s1 in shifts[1]:
                for s2 in shifts[2]:
                    nb = (b + (s0, s1, s2)) % nbins
                    other = (nb[:, 0] * nbins[1] + nb[:, 1]) * nbins[2] + nb[:, 2]
                    count = counts[other]
                    total = int(count.sum())
                    if total == 0:
                        continue
                    i = np.repeat(atoms, count)
                    # A ragged arange: 0..count[a]-1 for each atom `a` in turn.
                    offset = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
                    j = order[np.repeat(starts[other], count) + offset]
                    keep = i < j
                    I.append(i[keep])
                    J.append(j[keep])
        i = np.concatenate(I) if I else np.zeros(0, dtype=int)
        j = np.concatenate(J) if J else np.zeros(0, dtype=int)
        v, r = self.between(i, j)
        keep = r < cutoff
        return Pairs(i[keep], j[keep], v[keep], r[keep], cutoff)

    # -- subsets --------------------------------------------------------------

    def subset(self, atoms: np.ndarray) -> "Geometry":
        """The geometry of `atoms` alone, in that order; `self` if it is every atom.

        The subset's numbers are the ones a `Geometry` of `pos[atoms]` gives,
        and a dense array already built is cut rather than recomputed.
        """
        atoms = np.asarray(atoms)
        if len(atoms) == self.natoms and np.array_equal(atoms, np.arange(self.natoms)):
            return self
        dense = None
        if self._dense is not None:
            vecs, rij = self._dense
            sub = np.ix_(atoms, atoms)
            dense = (vecs[sub], rij[sub])
        return Geometry(self.pos[atoms], self.pbc, self.cell, dense=dense)


def as_geometry(displacements, pos, pbc, cell) -> Geometry:
    """`displacements` as a `Geometry`: one already, a `(vecs, rij)` pair, or `None`."""
    if isinstance(displacements, Geometry):
        return displacements
    return Geometry(pos, pbc, cell, dense=displacements)


def pair_gradients(
    i: np.ndarray,
    j: np.ndarray,
    dE_dr: np.ndarray,
    v: np.ndarray,
    r: np.ndarray,
    natoms: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forces and virial of a sum over the pairs `(i[p], j[p])`, one term each.

    `dE_dr[p]` is the radial derivative of pair `p`'s term, `v[p] = r_i - r_j`
    and `r[p] = |v[p]|`.  So `dE/dr_i = dE_dr * v / r` and minus that on `j`;
    the virial is `sum_p dE_dr v_a v_b / r`, the same per-pair gradient
    against strain.
    """
    forces = np.zeros((natoms, 3))
    if len(i) == 0:
        return forces, np.zeros((3, 3))
    coeff = dE_dr / r
    grad = coeff[:, None] * v
    for a in range(3):
        forces[:, a] = np.bincount(j, grad[:, a], natoms) - np.bincount(
            i, grad[:, a], natoms
        )
    return forces, grad.T @ v

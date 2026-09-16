"""Fit EVB off-diagonal couplings from a reaction's stationary points.

**There are three forms, and which one a channel gets is decided by its own
connectivity change.**  Routing is `scripts/fit.py`: `_fission` first, then
`_transfer`, then the RMSD fallback.

    fission      V(r)  = A * exp(-a * (r - r_cross)**2)     compute_twobody
    transfer     V(g)  = A * exp(-a * g),  g in A**2        compute_threebody
    anything     V(x)  = A * exp(-a * RMSD(x, x_TS)**2)     compute_rmsd

**Two independent things distinguish them, and it is worth keeping them apart:
where the amplitude comes from, and what the width is measured in.**

    form        amplitude from            width measured in
    twobody     the diabatic crossing     the breaking bond's length
    threebody   the reference barrier     the transferring atom's triangle
    rmsd        the reference barrier     the RMSD to the whole geometry

Only `fit_twobody` changes the amplitude, and only because it has to: a fission
has no saddle, so there is no reference barrier to invert.  A transfer *does*
have one, and a reference barrier is ab-initio data, so `fit_threebody` keeps
`fit_amplitude` untouched and changes only the metric the width lives in.  An
RMSD is a tolerance on all `3N` coordinates at once, so any spectator switches
the coupling off: with `h2o-autoionization`'s transfer atoms held exactly at its
transition state, displacing only the three spectators by 0.1 A -- less than
thermal motion at 300 K -- takes the RMSD coupling from 4.14 eV to 8.4e-3, while
the triangle form holds at 4.14.

The fission form exists because the RMSD one cannot describe that kind of channel
even in principle, not because it is more convenient.  For a barrierless fission
the bound diabat *is* the ground state up to the crossing, so a perfect reactant
diabat puts the reference barrier exactly on it and `fit_amplitude`'s
discriminant `(Hbar - E)**2 - dH**2` vanishes identically: the best attainable
amplitude is zero, every imperfection makes it imaginary, and no refit of the
force field moves that.  A channel with `A = 0` is dropped by `basis.EVBBasis`
at every geometry, which is why H2, OH and H2O could not come apart at all.
`fit_twobody` asks a question that has an answer instead -- where do the two
diabats cross, and what coupling hands one over to the other smoothly -- and
needs neither a transition-state geometry nor a reference energy.

A crossing-centred fit is *not* the right answer for a transfer, and
`h2o-autoionization` is why: its two diabats never cross along the proton
coordinate at all, the products being 9.8 eV uphill in the gas phase, so there is
no degeneracy to centre anything on.  It has a real saddle and a real barrier
instead, which is exactly the data `fit_amplitude` wants.

The RMSD form is fully specified by an amplitude `A` and a width `a`.  Both are
determined by the three frames every `rxn_*.xyz` already carries -- reactant,
transition state, product -- and nothing else:

  A   At the transition state RMSD is zero, so V(x_TS) = A exactly.  That makes
      the 2x2 secular equation invertible at that one geometry: given the two
      diabatic energies there and the reference barrier height, `A` follows in
      closed form.  A single TS energy per reaction is enough.

  a   One point cannot set a width, but the endpoints can.  The coupling is a
      correction to the diabatic picture and must vanish where that picture is
      already correct, i.e. at the reactant and product minima.  Requiring
      |V| <= eps at whichever endpoint is *closer* to the TS pins `a`.

The uniform placeholders (A = -10.0 eV, a = 10.0 A^-2) shipped with the
HCombustion dataset satisfy neither condition: at a = 10 the coupling is still
0.96 to 7.70 eV at the endpoints of the nineteen channels, so it never switches
off, and every molecule collects spurious stabilization from dissociation
channels that should be inactive at its own equilibrium geometry.
"""

import logging
from collections.abc import Callable, Sequence

import numpy as np
from ase import Atoms
from superpose3d import Superpose3D

from DynamicTopology.core.types import Term

logger: logging.Logger = logging.getLogger(__name__)

# Coupling considered switched off at this magnitude (eV).
DEFAULT_EPS: float = 1e-3

# Stand-in amplitude used to give a decoupled channel a finite width.  A zero
# amplitude has no width -- the condition `|V| <= eps` at the endpoints is
# satisfied everywhere -- but writing a nonsense number would make the term
# unreadable if someone later fills the amplitude in by hand.  1 eV is a
# plausible coupling, so the stored width stays meaningful.
NOMINAL_AMPLITUDE: float = 1.0


class CouplingFitError(ValueError):
    """Raised when the reference data cannot define a real coupling."""


def _rmsd(reference: np.ndarray, positions: np.ndarray) -> float:
    """Optimally superposed RMSD, matching EVBCoupling.compute_rmsd."""
    rmsd, _, _, _ = Superpose3D(reference, positions)
    return float(rmsd)


def fit_width(frames: list[Atoms], amplitude: float, eps: float = DEFAULT_EPS) -> float:
    """Width that quenches the coupling to `eps` at both endpoint geometries.

    Uses the endpoint nearer the transition state, so the condition holds at
    both.  Geometry only -- no reference energies required.
    """
    reference = frames[len(frames) // 2].positions
    separations = [
        _rmsd(reference, frames[0].positions),
        _rmsd(reference, frames[-1].positions),
    ]
    nearest = min(separations)
    if nearest <= 0.0:
        raise CouplingFitError(
            "an endpoint coincides with the transition state, so no width can "
            "switch the coupling off there"
        )
    # A decoupled channel is off at every geometry, so every width satisfies the
    # condition and `log(0)` is the arithmetic saying so.  Report the width a
    # plausible amplitude would have needed instead.
    magnitude = abs(amplitude) if amplitude != 0.0 else NOMINAL_AMPLITUDE
    return float(np.log(magnitude / eps) / nearest**2)


def fit_amplitude(
    diabatic_reactant: float, diabatic_product: float, barrier: float
) -> float:
    """Invert the 2x2 secular equation at the transition state.

    With E = Hbar - sqrt(dH**2 + V**2) and V(x_TS) = A,

        A = -sqrt( (Hbar - E)**2 - dH**2 )

    All three energies must share one zero (they do: the dataset stores
    atomization energies referenced to free atoms).
    """
    mean = 0.5 * (diabatic_reactant + diabatic_product)
    half_gap = 0.5 * (diabatic_reactant - diabatic_product)

    # The adiabatic ground state of a two-level block lies below both diabats,
    # strictly so for any nonzero coupling.  A barrier above either of them
    # still yields a real discriminant, so test the ordering directly rather
    # than relying on the square root to catch it.
    if barrier > min(diabatic_reactant, diabatic_product):
        raise CouplingFitError(
            f"reference barrier {barrier:+.4f} eV lies above a diabatic energy at "
            f"the transition state ({diabatic_reactant:+.4f} and "
            f"{diabatic_product:+.4f} eV). The EVB ground state is below every "
            "diabat by construction, so no coupling reproduces this; the "
            "diabatic energies are wrong, not the barrier."
        )

    discriminant = (mean - barrier) ** 2 - half_gap**2
    if discriminant < 0.0:
        raise CouplingFitError(
            f"reference barrier {barrier:+.4f} eV does not lie below both diabatic "
            f"energies at the transition state ({diabatic_reactant:+.4f} and "
            f"{diabatic_product:+.4f} eV), so no real coupling reproduces it. "
            "The adiabatic ground state is by construction below every diabat; a "
            "violation means the diabatic energies are wrong, not the barrier."
        )
    # Negative by convention.  Only A**2 enters the ground-state eigenvalue of a
    # two-level block, so the sign is a phase choice.
    return -float(np.sqrt(discriminant))


# Largest separation (A) searched for the diabatic crossing of a fission
# channel.  Every crossing in HCombustion is inside 4 A, which is also
# `ReactionSet.get_network`'s bimolecular cutoff, so a channel whose diabats do
# not cross by here would not have its reverse enumerated either.
DEFAULT_MAX_SEPARATION: float = 8.0

# Separation (A) past which `ReactionSet.get_network` stops enumerating a
# bimolecular channel, and so past which a coupling has no state to couple to.
# `System`'s own default, repeated here because `fit_twobody` has to quench
# against the cutoff the calculation will run with; a caller using another one
# must pass it.
DEFAULT_BIMOL_CUTOFF: float = 4.0

# Step (A) of the coarse scan that brackets the crossing.  Only has to be finer
# than the distance over which the gap changes sign once; the bisection that
# follows does the accuracy.
DEFAULT_SCAN_STEP: float = 0.1


def _separate(
    positions: np.ndarray, moving: Sequence[int], axis: np.ndarray, distance: float
) -> np.ndarray:
    """`positions` with `moving` rigidly translated `distance` along `axis`."""
    shifted = np.array(positions, dtype=float)
    shifted[list(moving)] += axis * distance
    return shifted


def find_crossing(
    positions: np.ndarray,
    pair: tuple[int, int],
    moving: Sequence[int],
    diabats: Callable[[np.ndarray], tuple[float, float]],
    max_separation: float = DEFAULT_MAX_SEPARATION,
    step: float = DEFAULT_SCAN_STEP,
) -> tuple[float, float, float]:
    """Where a fission's two diabats cross, and where its reactant sits.

    Returns `(r_min, r_cross, slope)` in A and eV/A: the bond length at which the
    *bonded* diabat is lowest along the path, the length at which
    `H_reactant - H_product` changes sign, and `d(gap)/dr` there.

    `r_min` is found rather than read off `positions` because a stored reactant
    frame is not necessarily relaxed in this coordinate, and the quench condition
    `fit_twobody` puts on it is a statement about where the molecule *sits*:
    `rxn_08`'s reactant carries one O-H at 0.600 A against a 0.95 A minimum, and
    quenching 0.35 A inside the wall leaves the coupling four times `eps` at the
    equilibrium geometry -- enough to cost `basis`'s screen on every O-H in a
    condensed phase, though not enough to move an energy.

    The path is a **rigid separation**: the fragment `moving` is translated along
    the breaking bond's axis and nothing relaxes.  That is the right path for
    this purpose and not a shortcut.  The gap is a difference of two diabats that
    share every nonbonded term and, for a fission, differ in exactly one bond
    term, so it is a function of that bond's length and of nothing else --
    relaxing the fragments would move both diabats by the same amount and leave
    the crossing where it is.  Translating along the bond axis also makes the
    pair separation exactly `r_eq + distance`, so the scan variable is the bond
    length itself.

    Raises:
        CouplingFitError: if the diabats do not cross inside `max_separation`,
            which means the bonded state never becomes the higher one -- see
            `qforce.BOND_ASYMPTOTE`, which is what puts the crossing on the path.
    """
    i, j = pair
    separation = positions[j] - positions[i]
    r_stored = float(np.linalg.norm(separation))
    if r_stored <= 0.0:
        raise CouplingFitError("the breaking bond's two atoms coincide")
    axis = separation / r_stored

    def both(r: float) -> tuple[float, float]:
        return diabats(_separate(positions, moving, axis, r - r_stored))

    def gap(r: float) -> float:
        bonded, separated = both(r)
        return bonded - separated

    at_stored = gap(r_stored)
    if at_stored >= 0.0:
        raise CouplingFitError(
            f"the bonded diabat is already the higher one at the stored bond "
            f"length ({r_stored:.3f} A, gap {at_stored:+.4f} eV), so this "
            "channel has no bound reactant to dissociate from"
        )

    low, high = r_stored, None
    r = r_stored
    while r < max_separation:
        r = min(r + step, max_separation)
        if gap(r) >= 0.0:
            high = r
            break
        low = r
    if high is None:
        raise CouplingFitError(
            f"the two diabats do not cross within {max_separation:.1f} A "
            f"(gap {gap(max_separation):+.4f} eV there), so the bonded state is "
            "the lower one at every separation and no coupling can hand over to "
            "the dissociated one. Raise `qforce.BOND_ASYMPTOTE`."
        )

    for _ in range(60):
        middle = 0.5 * (low + high)
        if gap(middle) < 0.0:
            low = middle
        else:
            high = middle
    crossing = 0.5 * (low + high)

    # Central difference over a tenth of the scan step.  The gap is a smooth
    # function of one variable here, so this is accurate to `h**2` and there is
    # nothing to be gained from an analytic derivative that would have to know
    # which bond term the gap came from.
    h = 0.1 * step
    slope = (gap(crossing + h) - gap(crossing - h)) / (2.0 * h)
    if slope <= 0.0:
        raise CouplingFitError(
            f"the diabatic gap is not increasing at the crossing "
            f"({slope:+.4f} eV/A), so the two states cross back"
        )

    # The bonded diabat's own minimum, on the coarse grid first and then by
    # bisecting its derivative.  Bracketed below by a compression the repulsion
    # makes unreachable and above by the crossing, so the minimum is interior and
    # the bisection cannot walk out of the well.
    inner = max(0.3, 0.5 * r_stored)
    grid = np.arange(inner, crossing, step)
    if len(grid) < 3:
        return r_stored, crossing, slope
    bonded = np.array([both(r)[0] for r in grid])
    k = int(np.argmin(bonded))
    if k == 0 or k == len(grid) - 1:
        # The minimum is not bracketed on this path -- a reactant frame already
        # outside the well, or a channel with no bound side at all.  The stored
        # length is then the honest quench point and the caller's own conditions
        # still hold at it.
        return r_stored, crossing, slope
    low, high = grid[k - 1], grid[k + 1]
    for _ in range(40):
        middle = 0.5 * (low + high)
        if both(middle + h)[0] < both(middle - h)[0]:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high), crossing, slope


def fit_twobody(
    positions: np.ndarray,
    pair: tuple[int, int],
    moving: Sequence[int],
    diabats: Callable[[np.ndarray], tuple[float, float]],
    eps: float = DEFAULT_EPS,
    bimol_cutoff: float = DEFAULT_BIMOL_CUTOFF,
    max_separation: float = DEFAULT_MAX_SEPARATION,
    step: float = DEFAULT_SCAN_STEP,
) -> list[Term]:
    """Fit the bond-length coupling of one fission or recombination channel.

    A fission has no saddle, so `fit_amplitude` has nothing to invert: its
    discriminant vanishes identically for a barrierless channel, which is why
    these channels were only ever fittable as `A = 0`.  This route asks a
    different question -- not "what coupling reproduces a reference barrier"
    but "what coupling hands one diabat over to the other smoothly" -- and needs
    no transition-state geometry and no reference energy, only the two diabats
    the force field already defines.

    Three numbers, and each is pinned by its own condition:

      r0  The crossing, from `find_crossing`.  At a crossing the diabats are
          degenerate, so the block's stabilization is exactly `|V|` and the
          adiabatic energy is exactly `H - |V|`: it is the one geometry where the
          coupling alone decides the answer, and the only defensible place to
          centre a Gaussian on a monotone path.  It is also what makes
          `basis.EVBBasis`'s ordinary `stab > eps` gate sufficient for these
          channels -- the coupling is at its maximum where the topology decision
          is taken.

      A   The largest amplitude that introduces **no spurious minimum** at the
          crossing.  With the Gaussian centred there and the gap linearized as
          `g ~ s*u` (`u = r - r0`, `s` the slope), the lower root of the 2x2 is
          `E = g/2 - hypot(g/2, V)`, whose derivatives at `u = 0` are

              E'  = s/2                      (V' = 0 at the centre)
              E'' = 2*a*|A| - s**2/(4*|A|)

          so the curve is rising through the crossing whatever the amplitude, and
          turns from concave to convex -- acquiring a local minimum, i.e. a bound
          radical pair and a barrier to dissociation that should not exist -- at
          exactly `8*a*A**2 = s**2`.  Sitting on that boundary is the smoothest
          handover available: any smaller amplitude leaves more of the `A = 0`
          kink, any larger one buys a spurious well.

      a   The width that quenches the coupling to `eps` at whichever end of the
          path is **nearer** the crossing, `|A|*exp(-a*L**2) = eps` with
          `L = min(r0 - r_min, bimol_cutoff - r0)`.  This is `fit_width`'s
          condition -- and its choice of the nearer endpoint -- in the bond
          length rather than in an RMSD, and the two ends are near for different
          reasons:

            inwards   at its own minimum the reactant's diabatic description is
                      already correct, so a coupling there is spurious
                      stabilization an intact molecule collects from its own
                      dissociation channel.

            outwards   past `bimol_cutoff` the state is *not in the basis*:
                      `ReactionSet.get_network` enumerates the recombination only
                      inside it, and once the pivot has flipped that is the only
                      direction the channel is reached from.  A coupling still
                      worth something there is stabilization the surface loses
                      discontinuously when the pair drifts apart -- measured at
                      **0.203 eV** for `rxn_08` before this condition was
                      imposed, whose crossing at 3.33 A is only 0.67 A inside the
                      cutoff.  `A = 0` made this free by construction and that is
                      why nothing needed it before.

          The dissociated end of the *coupling* needs no condition -- a Gaussian
          in the bond length vanishes as the fragments separate, where the RMSD
          form had to be told to -- but the dissociated end of the *basis* does.

    The last two are one system in two unknowns, and it has exactly one solution.
    Eliminating `a` and writing `u = 2*c/A**2` with `c = (L*s)**2/8` turns them
    into `u + log(u) = log(2*c/eps**2)`, whose left side is strictly increasing
    from `-inf` to `+inf` -- so a root exists, is unique, and `a = u/(2*L**2)`
    follows. (It is `u = W(2*c/eps**2)` in the Lambert W function; bisection
    avoids the `scipy.special` import for a one-dimensional monotone solve.)

    Args:
        positions: the reactant geometry, (n, 3) in A, in template index order.
        pair: template indices of the bond that breaks.
        moving: template indices of the fragment translated away from the other;
            either side will do, the gap depends only on the separation.
        diabats: returns `(H_reactant, H_product)` in eV at a given geometry.
        eps: coupling magnitude (eV) considered switched off at either end.
        bimol_cutoff: separation (A) past which `ReactionSet.get_network` stops
            enumerating the recombination, so past which the coupling must be
            off.  Must match the cutoff the calculation will run with.

    Returns:
        A one-term list ready for `io.json.write_jsonl`.
    """
    minimum, crossing, slope = find_crossing(
        positions, pair, moving, diabats, max_separation, step
    )
    if not minimum < crossing < bimol_cutoff:
        raise CouplingFitError(
            f"the crossing ({crossing:.3f} A) is not between the reactant's "
            f"minimum ({minimum:.3f} A) and the bimolecular cutoff "
            f"({bimol_cutoff:.3f} A), so there is no interval to quench over. A "
            "crossing outside the cutoff is a channel whose recombination is "
            "never enumerated; see `qforce.BOND_ASYMPTOTE`, which sets where it "
            "falls."
        )
    length = min(crossing - minimum, bimol_cutoff - crossing)

    target = np.log(2.0 * (0.125 * (length * slope) ** 2) / eps**2)
    low, high = np.finfo(float).tiny, 1.0
    while high + np.log(high) < target:
        high *= 2.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if middle + np.log(middle) < target:
            low = middle
        else:
            high = middle
    u = 0.5 * (low + high)

    width = u / (2.0 * length**2)
    amplitude = -slope / np.sqrt(8.0 * width)
    if abs(amplitude) <= eps:
        raise CouplingFitError(
            f"the crossing ({crossing:.3f} A) leaves only {length:.3f} A to "
            f"quench over, so the widest admissible Gaussian has amplitude "
            f"{amplitude:+.2e} eV -- at or below eps ({eps:.1e}), which "
            "`basis.EVBBasis` drops. This channel's crossing is too close to an "
            "end of its own path to be coupled through."
        )
    i, j = pair
    return [
        {
            "type": "twobody",
            "atoms": {"p1": int(i), "p2": int(j)},
            "kwargs": {
                "A": float(amplitude),
                "a": float(width),
                "r0": float(crossing),
            },
            "provenance": "fitted",
            # Which end of the path set the width, so a squeezed amplitude is
            # readable from the term file rather than having to be re-derived.
            # "cutoff" says the crossing is nearer `bimol_cutoff` than the
            # reactant's own minimum, which caps the amplitude below what the
            # surface would otherwise support: the channel wants a wider cutoff,
            # not a refit.  Read by `scripts/fit.py`'s report; the force fields
            # transpose only type/atoms/kwargs, so it rides along untouched.
            "limited_by": (
                "cutoff" if bimol_cutoff - crossing < crossing - minimum else "reactant"
            ),
        }
    ]


def _triangle(
    positions: np.ndarray, triple: tuple[int, int, int]
) -> tuple[float, float, float]:
    """`(ra, rb, t)` of the transfer triangle: the two bond lengths and the angle.

    `triple` is `(donor, proton, acceptor)` and `t` is the angle at the proton,
    in radians, matching `QForce`'s own `theta0` convention.
    """
    donor, proton, acceptor = triple
    va = positions[proton] - positions[donor]
    vb = positions[proton] - positions[acceptor]
    ra = float(np.linalg.norm(va))
    rb = float(np.linalg.norm(vb))
    if ra <= 0.0 or rb <= 0.0:
        raise CouplingFitError("the transferring atom coincides with a heavy atom")
    cosine = float(np.dot(va, vb) / (ra * rb))
    return ra, rb, float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _deviation(
    positions: np.ndarray,
    triple: tuple[int, int, int],
    centre: tuple[float, float, float],
) -> float:
    """`g`, the squared deviation `compute_threebody` measures, in A**2."""
    ra0, rb0, t0 = centre
    d0 = np.sqrt(ra0**2 + rb0**2 - 2.0 * ra0 * rb0 * np.cos(t0))
    ra, rb, t = _triangle(positions, triple)
    d = np.sqrt(ra**2 + rb**2 - 2.0 * ra * rb * np.cos(t))
    return (ra - ra0) ** 2 + (rb - rb0) ** 2 + (d - d0) ** 2


def fit_threebody(
    frames: list[Atoms],
    triple: tuple[int, int, int],
    diabatic_energies: tuple[float, float] | None = None,
    amplitude: float | None = None,
    eps: float = DEFAULT_EPS,
) -> list[Term]:
    """Fit the transfer coupling of one atom-transfer channel.

    Same two questions as `fit_coupling`, asked in the transferring atom's own
    internal coordinates instead of in the RMSD to a whole geometry:

      A   unchanged, and deliberately so.  A transfer *has* a saddle and a
          reference barrier, and that barrier is ab-initio data worth fitting to
          -- which is the difference between this and `fit_twobody`, where there
          is no barrier to invert.  `compute_threebody` is centred on the
          transition state's own triangle, so `g = 0` and `V = A` exactly there,
          which is the one property `fit_amplitude` needs to stay valid.

      a   the width that quenches to `eps` at whichever endpoint is nearer the
          transition state, measured in `g` rather than in RMSD.

    **Only the width changes, and it is the width that was wrong.**  An RMSD is a
    tolerance on all `3N` coordinates at once, so it is dominated by whichever
    spectator happens to have moved: `h2o-autoionization` came out at
    `a = 642 1/A**2`, alive only within 0.11 A RMSD of its stored transition
    state, and `datasets/Water/README.md` records a 64-water box never admitting
    a single diabatic state as a result.  In `g` the same channel is a far wider
    function of the coordinate that actually matters, because the spectators are
    no longer in the metric at all.

    Args:
        frames: the frames of a `rxn_*.xyz`, reactant first and product last; the
            middle one is the transition state and sets the centre.
        triple: template indices `(donor, proton, acceptor)` -- the heavy atom the
            transferring atom leaves, the atom itself, and the one it arrives at.
        diabatic_energies: `(H_reactant, H_product)` in eV at the transition-state
            geometry, as `fit_coupling` takes them.
        amplitude: use this instead of fitting one, as in `fit_coupling`.
        eps: coupling magnitude (eV) considered switched off at the endpoints.

    Returns:
        A one-term list ready for `io.json.write_jsonl`.
    """
    if len(frames) < 3:
        raise CouplingFitError(
            f"need at least reactant, transition state and product, got {len(frames)}"
        )
    transition = frames[len(frames) // 2]
    centre = _triangle(transition.positions, triple)

    if amplitude is None:
        provenance = "fitted"
    elif amplitude == 0.0:
        provenance = "decoupled"
    else:
        provenance = "placeholder"

    if amplitude is None:
        if diabatic_energies is None:
            raise CouplingFitError(
                "fitting an amplitude needs the diabatic energies at the "
                "transition-state geometry; pass `amplitude` to skip the fit"
            )
        if transition.calc is None:
            raise CouplingFitError(
                "the transition-state frame carries no reference energy"
            )
        amplitude = fit_amplitude(*diabatic_energies, transition.get_potential_energy())

    # The nearer endpoint, so the condition holds at both -- `fit_width`'s choice,
    # in `g`.  A transfer's two endpoints are generally *not* equidistant here
    # even when they are in RMSD, because `g` sees only the proton and its two
    # heavy atoms.
    deviations = [
        _deviation(frames[0].positions, triple, centre),
        _deviation(frames[-1].positions, triple, centre),
    ]
    nearest = min(deviations)
    if nearest <= 0.0:
        raise CouplingFitError(
            "an endpoint has the same transfer triangle as the transition state, "
            "so no width can switch the coupling off there"
        )
    magnitude = abs(amplitude) if amplitude != 0.0 else NOMINAL_AMPLITUDE
    width = float(np.log(magnitude / eps) / nearest)

    donor, proton, acceptor = triple
    return [
        {
            "type": "threebody",
            "atoms": {"p1": int(donor), "p2": int(proton), "p3": int(acceptor)},
            "kwargs": {
                "A": float(amplitude),
                "a": float(width),
                "ra0": float(centre[0]),
                "rb0": float(centre[1]),
                "t0": float(centre[2]),
            },
            "provenance": provenance,
        }
    ]


def fit_coupling(
    atoms: list[Atoms],
    diabatic_energies: tuple[float, float] | None = None,
    amplitude: float | None = None,
    eps: float = DEFAULT_EPS,
) -> list[Term]:
    """Fit the RMSD coupling for one reaction.

    Args:
        atoms: the frames of a `rxn_*.xyz`, in order: reactant, transition
            state(s), product.  The middle frame is the coupling reference.
        diabatic_energies: (H_reactant, H_product) evaluated with the force
            field *at the transition-state geometry*, in eV on the dataset's
            free-atom zero.  Required to fit the amplitude; combined with the
            reference energy carried by the TS frame.
        amplitude: use this amplitude instead of fitting one.  Needed while a
            dataset has geometries but no reference energies.  Pass `0.0` to
            decouple the channel outright, which is what an unfittable barrier
            gets: no stabilization, so the state never enters an EVB basis.
        eps: coupling magnitude (eV) considered switched off at the endpoints.

    Returns:
        The reaction's term list, ready for `io.json.write_jsonl`.
    """
    if len(atoms) < 3:
        raise CouplingFitError(
            f"need at least reactant, transition state and product, got {len(atoms)} frames"
        )

    transition = atoms[len(atoms) // 2]

    # An amplitude handed in by the caller is a stand-in, not a fit: it is what
    # `scripts/fit.py` falls back to when a channel's reference barrier cannot
    # be inverted.  Recording which is which lets the EVB report the channels a
    # basis was built on unfitted couplings, instead of inferring it from a
    # magic amplitude value.
    #
    # Zero is the honest stand-in and the default one: it contributes no
    # stabilization, so `EVBBasis` never admits the state at all, where a large
    # placeholder would drive the Hamiltonian with a number nobody fitted.
    if amplitude is None:
        provenance = "fitted"
    elif amplitude == 0.0:
        provenance = "decoupled"
    else:
        provenance = "placeholder"

    if amplitude is None:
        if diabatic_energies is None:
            raise CouplingFitError(
                "fitting an amplitude needs the diabatic energies at the "
                "transition-state geometry; pass `amplitude` to skip the fit"
            )
        if transition.calc is None:
            raise CouplingFitError(
                "the transition-state frame carries no reference energy. Run "
                "`scripts/compute.py` over the reaction files, as was done for "
                "the molecule templates, then refit."
            )
        amplitude = fit_amplitude(*diabatic_energies, transition.get_potential_energy())

    width = fit_width(atoms, amplitude, eps)

    # `atoms` here names the reacting-fragment indices the coupling spans.  The
    # RMSD is taken over all of them, so this is positional bookkeeping for the
    # term format rather than a subset selection.
    indices = {f"p{i + 1}": i for i in range(len(transition))}
    return [
        {
            "type": "rmsd",
            "atoms": indices,
            "kwargs": {"A": float(amplitude), "a": float(width)},
            # Read by `DynamicTopology.basis`; the force fields transpose only
            # type/atoms/kwargs, so this rides along untouched.
            "provenance": provenance,
        }
    ]

"""The force field's global parameters: the constants a dataset is fitted at.

Every field here used to be a module constant (`zbl.TAPER_RADIUS`,
`qforce.BOND_ASYMPTOTE`, `ewald.GAMMA`, ...).  Each sits inside
`E_bonded + E_nonbonded`, which fast-forces' `refine` solves every template
against, so a module constant made the fitted surface a property of whichever
source tree happened to be installed.  A manifest now states these under
`global_params`, `ReactionSet.load` activates them before it reads a template,
and everything downstream reads them back through `active()` at call time --
never bound into a default argument, since fast-forces' `refine` imports
before any manifest is loaded and a bound default would pin the wrong one.

Every field is in
ASE units -- Angstrom and eV -- like everything else, `.jsonl` files included.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, fields, replace
from typing import Any, Iterator


# The values `ForceFieldParams.electrostatics` accepts.
ELECTROSTATICS: frozenset[str] = frozenset({"acks2", "pointcharge"})


@dataclass(frozen=True)
class ForceFieldParams:
    """The constants a dataset is fitted at.  Frozen: see `replace` to derive one.

    Each field's comment gives its unit and what it's for.
    """

    # --- the bonded form ---------------------------------------------------

    # How far above the free-fragment limit a broken bond's diabat sits, in eV.
    # Splits the Morse asymptote from its depth so a bonded diabat crosses its
    # own fragments' diabat instead of converging to it -- without this,
    # fast-forces' `coupling.fit_amplitude`'s discriminant vanishes at dissociation and
    # there is nothing for an EVB off-diagonal to interpolate between.
    # Changing it invalidates every `.jsonl` in the dataset that states it.
    # It is also the default and the lower bound for each bond's own fitted
    # asymptote `h`; see `qforce.QForce._bond_morse`.
    bond_asymptote: float = 1.0

    # --- the screened-nuclear repulsion (`zbl`) ----------------------------

    # Where the screened-nuclear form stops being evaluated, in Angstrom.
    # Bounded from both sides: too small and two molecules can drift through
    # each other; too large and it eats into the hydrogen bond.  Changing it
    # invalidates every `.jsonl` in the dataset that states it.
    taper_radius: float = 1.5

    # How sharply `taper_radius`'s switch turns off, in Angstrom -- the
    # smallest width that keeps the term smooth enough to integrate.  Also
    # sets `switch_width` unless that is given explicitly; see there.
    taper_width: float = 0.12

    # --- the switched 12-6 (`lj`) ------------------------------------------

    # Where the 12-6 switches *on*, in Angstrom.  Deliberately **not** `taper_radius`
    # -- a Fermi switch and `r**-12` disagree by orders of magnitude at a bond
    # length, so the two forms are not complementary and leave a gap between
    # them that `ACKS2` alone carries.  See `forcefield/lj.py` for the
    # measurement.
    switch_radius: float = 2.2

    # How sharply the 12-6 switches on, in Angstrom.  `None` takes
    # `taper_width`: the two switches share one sharpness, which is a single
    # decision rather than two.  (It used to be `taper_width / 10`, which was the
    # same width in the nm the 12-6 was then evaluated in.)  Stating it
    # explicitly decouples the two, which nothing has yet needed.
    switch_width: float | None = None

    # The 12-6's soft-core constant `c`, in `4 eps [s**-2 - s**-1]` with
    # `s = c + (r/sigma)**6`.  Dimensionless.  `c = 0` is the bare 12-6; any
    # `c > 0` makes it finite at contact, `4 eps (1/c**2 - 1/c)`.  See
    # `lj.pair_potential` for what the bound is for; the well depth is `eps`
    # whatever `c` is.
    soft_core: float = 0.01

    # A spherical cutoff for the 12-6, in Angstrom, or `None` for none.  With
    # `None` -- the default, and what every dataset was fitted with -- the
    # 12-6 is summed over every minimum-image pair in the cell, so its reach
    # is the cell's own shape: half a cell along an axis, more toward a corner.
    # A cutoff switches each pair smoothly to zero over the last
    # `lj_cutoff_width` before it and, under full periodicity, adds the
    # uniform-density tail correction for what lies beyond, so the energy no
    # longer depends on the cell's shape and costs `O(N)` pairs rather than
    # `O(N^2)`.  It must not exceed half the cell's shortest perpendicular
    # width.  Not a rounding change: it moves the energy by the dispersion of
    # the pairs the minimum image happened to include past the cutoff, less the
    # tail correction's estimate of them.  See `lj.LennardJones`.
    lj_cutoff: float | None = None
    lj_cutoff_width: float = 1.0

    # --- the intramolecular exclusions (`exclusions`) ----------------------

    # How far along the bond graph the nonbonded interactions are excluded, in
    # bonds.  3 excludes 1-2, 1-3 and 1-4 pairs, so the terms act between atoms
    # four or more bonds apart and between atoms in different molecules --
    # GROMACS's `nrexcl = 3`, and q-force's own convention.
    exclusion_depth: int = 3

    # Whether to emit `coulombexclusion` terms; see `forcefield/exclusions.py`.
    # Only `PointCharge` reads them: fragment ACKS2 keeps a molecule's own
    # electrostatics off its bonded terms by subtracting the molecule's isolated
    # minimum instead, so this flag does not change an ACKS2 energy.
    exclude_coulomb: bool = True

    # --- the electrostatics -------------------------------------------------

    # Which electrostatic term the dataset was fitted with: `"acks2"`, charge
    # equilibration solved per diabatic state from each template's reference
    # charges `q0` (`forcefield/acks2.py`), or `"pointcharge"`, fixed charges
    # carried by each template's `charge` terms (`forcefield/pointcharge.py`).
    # Both localize the +1 on a hydronium and move it with the proton; ACKS2
    # also polarizes each molecule, and point charges are its zero-softness
    # limit.  Changing it invalidates every `.jsonl` in the dataset, since the
    # fit solves against whichever term this names.
    electrostatics: str = "acks2"

    # --- the charge kernel (`ewald`, `acks2`, `pointcharge`) -----------------

    # Charge-smearing width of the ACKS2 kernel `erf(gamma r) / r`, in
    # 1/Angstrom.  Shared by both kernel forms, so the open-boundary and
    # periodic sums smear identically.
    gamma: float = 2.0

    # Target relative error of the truncated lattice sum.  Dimensionless.  Sets
    # `kappa` and the reciprocal cutoff together, so both halves converge to
    # the same level.  Do not loosen this casually: the cost is a cube root in
    # reciprocal vector count, but the *stress* error pays for it linearly.
    # Periodic only, so it invalidates no
    # template (every dataset template carries `pbc="F F F"`).
    accuracy: float = 1e-8

    # How ACKS2 solves for the charges.  `"direct"` -- the default, and what
    # every dataset was fitted with -- forms the periodic kernel as a dense
    # matrix (`ewald.Ewald`) and LU-factors the system: exact, `O(N^2)` memory
    # and `O(N^3)` time.  `"iterative"` keeps the kernel as an operator -- a
    # real-space cutoff and particle-mesh Ewald (`ewald.EwaldOperatorSetup`)
    # -- and solves with preconditioned conjugate gradients to
    # `solver_tolerance`; it needs full periodicity, and any call it cannot
    # take (a molecule kept explicit in the solve) falls back to the direct
    # path.  Not a rounding change: PME and the residual each move the forces
    # by up to ~1e-6 of their size.  See `acks2.ACKS2`.
    charge_solver: str = "direct"

    # The iterative solver's real-space cutoff, in Angstrom (shortened to half
    # the cell where that is less), and its relative residual.
    real_space_cutoff: float = 9.0
    solver_tolerance: float = 1e-10

    # Coulomb constant in eV*Angstrom, as `ACKS2` carries it.
    ccoul: float = 14.4

    # The same physical constant as `ccoul`, to the digits ASE uses, as
    # `zbl.pair_potential` carries it.  **Two fields for one constant,
    # deliberately.**  The two values differ in the sixth digit and always have;
    # collapsing them onto one field would change one of the two terms for every
    # dataset already fitted, which is precisely the silent invalidation this
    # module exists to prevent.  A dataset may set them equal.
    zbl_ccoul: float = 14.399645

    def __post_init__(self) -> None:
        if self.switch_width is None:
            # Resolved once, here, so that every reader sees a float and no call
            # site has to know about the derivation.  `object.__setattr__`
            # because the dataclass is frozen; this is the documented way.
            object.__setattr__(self, "switch_width", self.taper_width)
        if self.exclusion_depth < 0:
            raise ValueError(
                f"exclusion_depth must be >= 0, got {self.exclusion_depth}"
            )
        object.__setattr__(self, "exclusion_depth", int(self.exclusion_depth))
        object.__setattr__(self, "exclude_coulomb", bool(self.exclude_coulomb))
        if self.charge_solver not in ("direct", "iterative"):
            raise ValueError(
                "charge_solver must be 'direct' or 'iterative', "
                f"got {self.charge_solver!r}"
            )
        if self.electrostatics not in ELECTROSTATICS:
            raise ValueError(
                f"electrostatics must be one of {sorted(ELECTROSTATICS)}, "
                f"got {self.electrostatics!r}"
            )
        # Every other field is a float; coerce by exclusion rather than an
        # inclusion list, so a new float field needs no update here.
        for f in fields(self):
            if f.name in (
                "exclusion_depth",
                "exclude_coulomb",
                "electrostatics",
                "charge_solver",
            ):
                continue
            if f.name == "lj_cutoff" and self.lj_cutoff is None:
                continue
            object.__setattr__(self, f.name, float(getattr(self, f.name)))
        for name in ("taper_width", "switch_width", "gamma", "accuracy"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")
        if self.lj_cutoff is not None and not (
            0.0 < self.lj_cutoff_width < self.lj_cutoff
        ):
            raise ValueError(
                f"lj_cutoff_width must lie in (0, lj_cutoff), got "
                f"{self.lj_cutoff_width} with lj_cutoff {self.lj_cutoff}"
            )
        if self.soft_core < 0.0:
            # A negative `c` puts a pole at `r = (-c)**(1/6) sigma`.
            raise ValueError(f"soft_core must be >= 0, got {self.soft_core}")

    @classmethod
    def from_dict(cls, mapping: Any, source: str | None = None) -> "ForceFieldParams":
        """Build from a manifest's `global_params`, defaults for what it omits.

        **An unknown key is an error, not a no-op.**  A typo would otherwise
        leave the default in force and produce a dataset whose manifest claims a
        radius the force field never saw -- the exact failure mode this module
        was written to close, arriving by a different door.
        """
        if mapping is None:
            return cls()
        if not isinstance(mapping, dict):
            raise ValueError(
                f"`global_params` must be a JSON object, got {type(mapping).__name__}"
                + (f" in {source}" if source else "")
            )
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(mapping) - known)
        if unknown:
            raise ValueError(
                f"unknown `global_params` key(s) {unknown}"
                + (f" in {source}" if source else "")
                + f"; known parameters are {sorted(known)}"
            )
        return cls(**mapping)

    def to_dict(self) -> dict[str, Any]:
        """Every field, resolved -- `switch_width` included, never `None`.

        What a manifest would have to state to pin this object exactly, which is
        how a production run records the surface it ran on.
        """
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def differences(self, other: "ForceFieldParams") -> dict[str, tuple[Any, Any]]:
        """`{field: (mine, theirs)}` for the fields the two disagree about."""
        return {
            field.name: (getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        }


DEFAULTS = ForceFieldParams()

# The parameters every force field reads, and the manifest that put them there.
# Process-global, because the alternative is threading one object through
# `zbl.taper`, `lj.switch`, `exclusions.exclusion_terms`, twenty functions in
# fast-forces' `refine` and the module-level force field singletons the fitter
# builds at import time -- and any call site that forgot to pass it would
# silently evaluate at the defaults, which is the same silent invalidation as
# before with more places to hide.  One source of truth, read at call time.
_active: ForceFieldParams = DEFAULTS
_source: str | None = None


def active() -> ForceFieldParams:
    """The parameters in force.  Call this at use time, never at import time."""
    return _active


def active_source() -> str | None:
    """Which manifest activated the current parameters, if any."""
    return _source


def resolve(params: ForceFieldParams | None) -> ForceFieldParams:
    """`params` if given, else the active set -- the one line every call site needs.

    Every module function that takes an optional `params` resolves it this way
    rather than in its own signature; see the module docstring for why.
    """
    return active() if params is None else params


def activate(params: ForceFieldParams, source: str | None = None) -> None:
    """Install `params` for the process, refusing to contradict what is in force.

    **A conflict is an error.**  Two datasets fitted at different radii cannot
    be evaluated by one force field at once: whichever loaded second would score
    the first's templates on a surface they were not fitted to, and the symptom
    -- a template no longer reproducing its own reference energy -- would be
    attributed to the fit rather than to the collision.  Loading both in one
    process is legitimate only when the caller is explicit about which surface
    it wants, which is what `use` is for.

    Re-activating the same values is not a conflict, so loading one manifest
    repeatedly (as the tests do) is free.
    """
    global _active, _source
    if params == _active:
        # Keep the first attribution: it is the one that established the values.
        if _source is None:
            _source = source
        return
    if _active != DEFAULTS or _source is not None:
        differing = ", ".join(
            f"{name}: {mine!r} vs {theirs!r}"
            for name, (mine, theirs) in _active.differences(params).items()
        )
        raise ValueError(
            "force field parameters already active"
            + (f" from {_source}" if _source else "")
            + f" disagree with {source or 'the requested set'} ({differing}). "
            "Two datasets fitted at different global parameters cannot share one "
            "force field; load them in separate processes, or wrap the second in "
            "`forcefield.params.use(...)` to say which surface you mean."
        )
    _active = params
    _source = source


@contextlib.contextmanager
def use(
    params: ForceFieldParams | None = None, **overrides: Any
) -> Iterator[ForceFieldParams]:
    """Run a block with `params` in force, restoring whatever was there after.

    `use(taper_radius=1.6)` derives from the *currently active* set rather than
    from the defaults, so an override composes with a dataset's own parameters
    instead of quietly discarding them.

    Overriding `taper_width` alone re-derives `switch_width`, since the two are
    the same physical width and `replace` would otherwise carry the old
    derivation forward.  State both to decouple them.

    Unlike `activate` this never refuses: the caller has said which surface it
    means, which is the whole distinction between the two.  The attribution
    follows the values -- a block that changes nothing keeps the manifest's
    name, and one that changes something has no manifest to name.
    """
    global _active, _source
    base = params if params is not None else _active
    if "taper_width" in overrides and "switch_width" not in overrides:
        # `switch_width` is a resolved float on `base`, whatever it started as,
        # so `replace` would carry the *old* taper width's derivation forward
        # and silently decouple the two switches -- the one thing the default is
        # there to prevent.  Overriding the taper width alone re-derives it.
        overrides["switch_width"] = None
    resolved = replace(base, **overrides) if overrides else base
    previous, previous_source = _active, _source
    _active = resolved
    _source = previous_source if resolved == previous else None
    try:
        yield resolved
    finally:
        _active, _source = previous, previous_source


def reset() -> None:
    """Back to the defaults, with no manifest attributed.  For test teardown."""
    global _active, _source
    _active, _source = DEFAULTS, None

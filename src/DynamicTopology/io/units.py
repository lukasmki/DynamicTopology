"""The one place the on-disk unit convention is written down.

A `.jsonl` stores its parameters the way q-force and OpenMM state them: lengths
in nm, energies in kJ/mol.  Everything in memory -- every `Term`, every
`term_dict`, every force field, `params.ForceFieldParams` -- is in ASE units,
Angstrom and eV, so the conversion happens exactly twice in the life of a
parameter: once when a file is read, once when one is written.  Both go through
`from_disk` / `to_disk` here and nowhere else.

This used to be the other way round.  `QForce` and `LennardJones` worked in nm
and kJ/mol and converted their results at the end of `__call__`, every other
term worked in Angstrom and eV, and the boundary ran through the middle of the
force field: the virial had to be rescaled by the energy factor alone (a length
factor too would have put every pressure out by exactly ten), a ZBL exclusion
had to convert its distances going in and its result coming back, and
`bond_asymptote` needed a kJ/mol twin for the five call sites that added it to
a depth.  Moving the boundary out to the file removed all of those.

`UNIT_POWERS[term][parameter]` is `(length_power, energy_power)`, the dimension
of that parameter.  A bond force constant is an energy over a length squared,
so `(-2, 1)`.  A parameter with powers `(0, 0)` is stored as it is used.

**The exceptions store ASE units on disk as well**: the ACKS2 `atom` block (eV
and Angstrom, as `ACKS2` reads it), `charge` (elementary charges), and every EVB
coupling (`A` in eV, lengths in Angstrom, `a` in 1/Angstrom**2).  Their powers
are zero, which is what "stored as used" means, and writing their true
dimensions here instead would silently rescale every coupling ever fitted.

An unknown term type or parameter is an error, not a pass-through: a parameter
whose unit this table does not know would otherwise be read in the wrong one,
which is precisely the silent factor of ten or 96.5 this module exists to rule
out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ase import units

if TYPE_CHECKING:  # `core` imports this module; a runtime import would cycle
    from DynamicTopology.core.types import Term

# Angstrom -> nm, and eV -> kJ/mol: multiply an ASE-unit value by these to get
# the on-disk one.
LENGTH: float = 0.1
ENERGY: float = 1.0 / (units.kJ / units.mol)

# Significant digits a converted value is written with.  A float carries 15.95,
# and `x / f * f` is not always `x` -- 27 of the 314 converted parameters in the
# shipped datasets come back one ulp off -- so writing the full repr would
# change the last digit of parameters nothing touched every time a file was
# refit.  At 15 digits every value that was stored with 15 or fewer reads back
# and rewrites byte-identical.  Unconverted values (powers `(0, 0)`) are written
# verbatim.
WRITE_DIGITS: int = 15

# `(length_power, energy_power)` per parameter.
UNIT_POWERS: dict[str, dict[str, tuple[int, int]]] = {
    "atom": {
        "mu": (0, 0),
        "eta": (0, 0),
        "soft_amp": (0, 0),
        "soft_decay": (0, 0),
    },
    "charge": {"q": (0, 0)},
    "lennardjones": {"sigma": (1, 0), "eps": (0, 1)},
    # `h` is the per-bond asymptote height, an energy like `D`.
    "bond": {"r0": (1, 0), "k": (-2, 1), "D": (0, 1), "h": (0, 1)},
    # `QForce.compute_angle` is `0.5*k*(cos-cos0)^2`, so `k` is a plain energy.
    "angle": {"theta0": (0, 0), "k": (0, 1)},
    "bondbond": {"r1_0": (1, 0), "r2_0": (1, 0), "k": (-2, 1)},
    "bondangle": {"theta0": (0, 0), "r0": (1, 0), "k": (-1, 1)},
    "angleangle": {"theta1_0": (0, 0), "theta2_0": (0, 0), "k": (0, 1)},
    "dihedralangle": {"k": (0, 1), "theta0": (0, 0), "n": (0, 0), "phi0": (0, 0)},
    "dihedralbond": {"k": (-1, 1), "r0": (1, 0), "n": (0, 0), "phi0": (0, 0)},
    "dihedralangleangle": {
        "k": (0, 1),
        "theta0_1": (0, 0),
        "theta0_2": (0, 0),
        "n": (0, 0),
        "phi0": (0, 0),
    },
    "periodicdihedral": {"k": (0, 1), "n": (0, 0), "phi0": (0, 0)},
    "reference": {"E0": (0, 1)},
    # Derived at load (`forcefield/exclusions.py`) and never shipped, but a
    # dataset that states them explicitly is honoured, so they have to read.
    "exclusion": {"sigma": (1, 0), "eps": (0, 1)},
    "zblexclusion": {"z1": (0, 0), "z2": (0, 0)},
    "coulombexclusion": {},
    "rmsd": {"A": (0, 0), "a": (0, 0)},
    "twobody": {"A": (0, 0), "a": (0, 0), "r0": (0, 0)},
    "threebody": {
        "A": (0, 0),
        "a": (0, 0),
        "ra0": (0, 0),
        "rb0": (0, 0),
        "t0": (0, 0),
    },
}


def factor(term: str, parameter: str) -> float:
    """Multiplier taking one parameter from ASE units to on-disk units."""
    try:
        length_power, energy_power = UNIT_POWERS[term][parameter]
    except KeyError:
        known = sorted(UNIT_POWERS.get(term, {}))
        raise KeyError(
            f"no unit is recorded for `{term}.{parameter}`"
            + (f" (known {term} parameters: {known})" if term in UNIT_POWERS else "")
            + "; add it to `DynamicTopology.io.units.UNIT_POWERS` rather than "
            "reading it in whatever unit it happens to be stored in"
        ) from None
    return (LENGTH**length_power) * (ENERGY**energy_power)


def to_disk(term: str, parameter: str, value: Any) -> Any:
    """One parameter, ASE units -> on-disk (nm, kJ/mol)."""
    f = factor(term, parameter)
    if f == 1.0:
        return value
    return float(f"{value * f:.{WRITE_DIGITS}g}")


def from_disk(term: str, parameter: str, value: Any) -> Any:
    """One parameter, on-disk (nm, kJ/mol) -> ASE units."""
    f = factor(term, parameter)
    if f == 1.0:
        return value
    return value / f


def _convert(term: Term, fn) -> Term:
    kind = term["type"]
    return {
        **term,
        "kwargs": {
            name: fn(kind, name, value) for name, value in term["kwargs"].items()
        },
    }


def term_from_disk(term: Term) -> Term:
    """A copy of one row as read from a `.jsonl`, in ASE units."""
    return _convert(term, from_disk)


def term_to_disk(term: Term) -> Term:
    """A copy of one in-memory term, in the units a `.jsonl` stores."""
    return _convert(term, to_disk)

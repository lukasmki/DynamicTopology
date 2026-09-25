"""The table of what converts how between ASE units and q-force's.

A `.jsonl` stores every parameter in ASE units -- Angstrom and eV -- which is
also what every `Term`, every `term_dict`, every force field and
`params.ForceFieldParams` holds, so reading and writing a term file converts
nothing.  q-force's XML and OpenMM state the same parameters in nm and kJ/mol,
and this module is the one place that boundary is crossed: fast-forces'
`import-qforce` converts on the way in (`term_from_openmm`) and its OpenMM export
on the way out (`factor`, `LENGTH`, `ENERGY`).

The `.jsonl` used to be in nm and kJ/mol too, converted once on read and once on
write.  That made a file's numbers recognisable q-force numbers, at the price of
a conversion on every read, an ulp of churn on every rewrite (`x / f * f` is not
always `x`) and a factor of ten or 96.5 between what a file said and what the
force field used.  Every shipped file was rewritten in eV/Angstrom with the
exact doubles the old reader produced, so no loaded value changed.

`UNIT_POWERS[term][parameter]` is `(length_power, energy_power)`, the dimension
of that parameter.  A bond force constant is an energy over a length squared,
so `(-2, 1)`.  A parameter with powers `(0, 0)` is the same number in both.

**The exceptions have zero powers whatever their dimension**: the ACKS2 `atom`
block, `charge`, and every EVB coupling were never stored in q-force's units,
and neither OpenMM nor q-force states them, so there is nothing to convert.

An unknown term type or parameter is an error, not a pass-through: a parameter
whose unit this table does not know would otherwise be converted by the wrong
factor, which is precisely the silent ten or 96.5 this module exists to rule
out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ase import units

if TYPE_CHECKING:  # `core` imports this module; a runtime import would cycle
    from DynamicTopology.core.types import Term

# Angstrom -> nm, and eV -> kJ/mol: multiply an ASE-unit value by these to get
# q-force's and OpenMM's.
LENGTH: float = 0.1
ENERGY: float = 1.0 / (units.kJ / units.mol)

# `(length_power, energy_power)` per parameter.
UNIT_POWERS: dict[str, dict[str, tuple[int, int]]] = {
    "atom": {
        "mu": (0, 0),
        "eta": (0, 0),
        "soft_amp": (0, 0),
        "soft_decay": (0, 0),
        # The reference charge fragment ACKS2 equilibrates from; optional.
        "q0": (0, 0),
    },
    "charge": {"q": (0, 0)},
    # `A`/`B` are q-force's `4 eps sigma**12` and `4 eps sigma**6`, which only
    # an XML import carries; the datasets state `sigma`/`eps`.
    "lennardjones": {"sigma": (1, 0), "eps": (0, 1), "A": (12, 1), "B": (6, 1)},
    # `h` is the per-bond asymptote height, an energy like `D`.
    "bond": {"r0": (1, 0), "k": (-2, 1), "D": (0, 1), "h": (0, 1)},
    # `QForce.compute_angle` is `0.5*k*(cos-cos0)^2`, so `k` is a plain energy.
    "angle": {"theta0": (0, 0), "k": (0, 1)},
    "bondbond": {"r1_0": (1, 0), "r2_0": (1, 0), "k": (-2, 1)},
    "bondangle": {"theta0": (0, 0), "r0": (1, 0), "k": (-1, 1)},
    "angleangle": {"theta1_0": (0, 0), "theta2_0": (0, 0), "k": (0, 1)},
    # q-force's XML also names a `theta0_1`/`theta0_2` pair here.
    "dihedralangle": {
        "k": (0, 1),
        "theta0": (0, 0),
        "theta0_1": (0, 0),
        "theta0_2": (0, 0),
        "n": (0, 0),
        "phi0": (0, 0),
    },
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
    """Multiplier taking one parameter from ASE units to nm and kJ/mol."""
    try:
        length_power, energy_power = UNIT_POWERS[term][parameter]
    except KeyError:
        known = sorted(UNIT_POWERS.get(term, {}))
        raise KeyError(
            f"no unit is recorded for `{term}.{parameter}`"
            + (f" (known {term} parameters: {known})" if term in UNIT_POWERS else "")
            + "; add it to `DynamicTopology.io.units.UNIT_POWERS` rather than "
            "converting it by whatever factor happens to fit"
        ) from None
    return (LENGTH**length_power) * (ENERGY**energy_power)


def to_openmm(term: str, parameter: str, value: Any) -> Any:
    """One parameter, ASE units -> nm and kJ/mol."""
    f = factor(term, parameter)
    if f == 1.0:
        return value
    return value * f


def from_openmm(term: str, parameter: str, value: Any) -> Any:
    """One parameter, nm and kJ/mol -> ASE units."""
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


def term_from_openmm(term: Term) -> Term:
    """A copy of one term stated in nm and kJ/mol, in ASE units."""
    return _convert(term, from_openmm)


def term_to_openmm(term: Term) -> Term:
    """A copy of one in-memory term, in nm and kJ/mol."""
    return _convert(term, to_openmm)

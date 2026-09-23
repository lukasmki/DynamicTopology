"""The energy of one topology, summed exactly as `System` sums a diabat.

`System.calculate` never evaluates a single topology in isolation -- it builds
an EVB basis, and each diabat's energy is assembled from pieces that live in
different places: `QForce` on the diagonal (the intramolecular exclusions
included, as ordinary terms), the electrostatics prepared before the
diagonalization and screened after it, `ZBL` and the 12-6 once for the whole
system.  Anything that has to score one bonding pattern on its own -- a fitter
matching a template to its reference energy, a test pinning a template, a
report decomposing an energy -- needs that same sum, and assembling it by hand
is how the fitter once scored templates without the 12-6 the calculator applied.
This is the one place it is assembled.

**It is the same number `System` produces for a lone molecule with no reaction
admitted**, and `tests/test_evaluate.py` holds it to that for every template of
every dataset.  `terms` is a template's term list in the units every term is
held in (see `io/units.py`); its exclusions are derived here if it does not
state them, exactly as `ReactionSet.load` derives them, and no reference shift
is synthesized -- the energy is that of the terms given.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from DynamicTopology.core.types import Term
from DynamicTopology.forcefield.electrostatics import Electrostatics
from DynamicTopology.forcefield.exclusions import with_exclusions
from DynamicTopology.forcefield.lj import LennardJones
from DynamicTopology.forcefield.params import active, use
from DynamicTopology.forcefield.qforce import QForce
from DynamicTopology.forcefield.zbl import ZBL

# Stateless apart from the per-cell Ewald caches `Electrostatics` keeps, which
# are keyed on the cell and so safe to share.
_QFORCE = {"morse": QForce("morse"), "harmonic": QForce("harmonic")}
_ELECTROSTATICS = Electrostatics()
_ZBL = ZBL()
_LJ = LennardJones()


@dataclass(frozen=True)
class Evaluation:
    """Energy (eV), forces (eV/A) and virial (eV) of one topology, and its parts.

    `bonded` is everything `QForce` evaluates -- the bonded terms, the reference
    shift and the ZBL and 12-6 exclusions; `electrostatics` is ACKS2 or the
    point charges (see `surface`), with its own exclusion screen applied.  The
    four parts sum to `energy`.

    `charges` are per atom, in global order: solved for by ACKS2, or read off
    the `charge` terms under `pointcharge`.  Zero for an atom that carries no
    electrostatic term.  A term list with no electrostatic block, or no
    `lennardjones` block, scores zero for that part.
    """

    energy: float
    forces: np.ndarray
    virial: np.ndarray
    bonded: float
    electrostatics: float
    zbl: float
    lj: float
    charges: np.ndarray


def term_dict(atoms: Atoms, terms: list[Term]) -> dict:
    """The vectorized `term_dict` for `terms`, exclusions derived if absent."""
    from DynamicTopology.core.topology import Topology

    terms = with_exclusions(terms, atoms.get_atomic_numbers())
    return Topology.from_terms(terms, atoms).term_dict


@contextlib.contextmanager
def surface(td: dict):
    """The active parameters, with the electrostatics `td` can be scored under.

    Which electrostatic term runs is a dataset-wide `global_params` choice, so a
    term list carrying only the other block would otherwise score no
    electrostatics at all, silently -- a fixed-charge template under an ACKS2
    surface loses its charges.  A term list carrying exactly one of the two
    blocks is therefore scored under that one; carrying both or neither, under
    whatever is active.
    """
    present = [kind for block, kind in _BLOCKS.items() if block in td]
    wanted = present[0] if len(present) == 1 else active().electrostatics
    if wanted == active().electrostatics:
        yield active()
    else:
        with use(electrostatics=wanted) as params:
            yield params


# The electrostatic block each `global_params.electrostatics` value reads.
_BLOCKS = {"atom": "acks2", "charge": "pointcharge"}


def evaluate(atoms: Atoms, terms: list[Term], bond_form: str = "morse") -> Evaluation:
    """Energy, forces and virial of `terms` at `atoms`' geometry, cell and pbc."""
    return evaluate_term_dict(atoms, term_dict(atoms, terms), bond_form)


def _charges(td: dict, natoms: int) -> np.ndarray:
    """Per-atom charges of the evaluation that just ran, in global order."""
    charges = np.zeros(natoms)
    if active().electrostatics == "pointcharge":
        block = td.get("charge")
        if block is not None:
            charges[block["atoms"][:, 0]] = block["kwargs"]["q"]
    else:
        block = td.get("atom")
        solved = _ELECTROSTATICS.get().Q
        if block is not None and solved is not None:
            charges[block["atoms"][:, 0]] = solved
    return charges


def _nonbonded_parts(atoms: Atoms, td: dict) -> tuple:
    """Electrostatics, whole-system ZBL and 12-6, each as `(E, F, W)`.

    A term list with no electrostatic or 12-6 block contributes nothing for it,
    rather than raising as `System` does -- a fitter scoring the bonded part
    alone, or a field under construction, has neither yet.  Called under
    `surface(td)`.
    """
    pos, pbc, cell = atoms.positions, atoms.pbc, atoms.cell.array
    nothing = (0.0, np.zeros_like(pos), np.zeros((3, 3)))
    electrostatic = "charge" if active().electrostatics == "pointcharge" else "atom"
    return (
        _ELECTROSTATICS(pos, pbc, cell, td) if electrostatic in td else nothing,
        _ZBL(pos, atoms.numbers, pbc, cell),
        _LJ(pos, pbc, cell, td) if "lennardjones" in td else nothing,
    )


def nonbonded(atoms: Atoms, td: dict) -> tuple[float, np.ndarray, np.ndarray]:
    """`(E, F, W)` of everything `evaluate` adds on top of `QForce`.

    The electrostatics, screened by its own exclusion, the whole-system ZBL and
    the switched 12-6 -- for a caller that memoizes this half across a fit of
    the bonded parameters, which it does not depend on.
    """
    with surface(td):
        (e_q, f_q, w_q), (e_z, f_z, w_z), (e_l, f_l, w_l) = _nonbonded_parts(atoms, td)
    # Added in this order, as `evaluate_term_dict` adds them -- not `sum()`,
    # which compensates float sums and so rounds differently.
    return float(e_q + e_z + e_l), f_q + f_z + f_l, w_q + w_z + w_l


def evaluate_term_dict(atoms: Atoms, td: dict, bond_form: str = "morse") -> Evaluation:
    """`evaluate` from an already-built `term_dict`, exclusions included.

    For a caller that scores one topology at many geometries -- a calculator
    over a trajectory, a fitter over a training set -- so the exclusions are
    derived and the terms vectorized once rather than at every call.
    """
    pos, pbc, cell = atoms.positions, atoms.pbc, atoms.cell.array
    e_b, f_b, w_b = _QFORCE[bond_form](pos, pbc, cell, td)
    with surface(td):
        (e_q, f_q, w_q), (e_z, f_z, w_z), (e_l, f_l, w_l) = _nonbonded_parts(atoms, td)
        charges = _charges(td, len(atoms))
    return Evaluation(
        energy=float(e_b + e_q + e_z + e_l),
        forces=f_b + f_q + f_z + f_l,
        virial=w_b + w_q + w_z + w_l,
        bonded=float(e_b),
        electrostatics=float(e_q),
        zbl=float(e_z),
        lj=float(e_l),
        charges=charges,
    )

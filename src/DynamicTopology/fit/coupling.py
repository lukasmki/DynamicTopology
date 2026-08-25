"""Fit EVB off-diagonal couplings from a reaction's stationary points.

The coupling used by `forcefield.coupling.EVBCoupling` is a Gaussian in the
RMSD to a transition-state reference geometry,

    V(x) = A * exp(-a * RMSD(x, x_TS)**2)

so it is fully specified by an amplitude `A` and a width `a`.  Both are
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

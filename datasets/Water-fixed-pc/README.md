# Water-fixed-pc

The `Water` dataset with **fixed per-template point charges** in place of ACKS2
(`global_params.electrostatics = "pointcharge"`; see `forcefield/pointcharge.py`).
Same species, same geometries, same reference energies at wB97X-V/aug-cc-pVTZ,
same bonded terms and couplings up to the refit below.

Why: ACKS2 solves one set of charges for the whole system under a single
neutrality constraint, so it cannot put the +1 of a proton transfer on the
hydronium rather than on the water next to it, and a hop moves no charge.  Here
the charge is a property of the template, so it moves with the proton, and the
electrostatic energy becomes part of what distinguishes one diabatic state from
another.

| id | template | species | total q | charges (e) |
|----|----------|---------|---------|-------------|
| 1 | `molecules/h`   | H (free atom) |  0 | H 0 |
| 2 | `molecules/o`   | O (free atom) |  0 | O 0 |
| 3 | `molecules/h1o` | **OH-**       | -1 | O -1.1983, H +0.1983 |
| 4 | `molecules/h2o` | H2O           |  0 | O -0.6862, H +0.3431 |
| 5 | `molecules/h3o` | **H3O+**      | +1 | O -0.4329, H +0.4776 |

| id | channel | A (eV) | a (1/A^2) | margin (eV) |
|----|---------|--------|-----------|-------------|
| 1 | `reactions/h3o-h2o-transfer`   | -3.213 | 19.4 | 3.21 |
| 2 | `reactions/h2o-oh-transfer`    | -2.764 | 22.5 | 2.76 |
| 3 | `reactions/h2o-autoionization` | -2.535 | 44.1 | 1.51 |

## The charges

fast-forces' `fastforces.charges` fits them: Merz-Kollman ESP charges at the
dataset's own level of theory, each at its template's geometry, with the total
constrained to the formal charge and the equivalent hydrogens averaged.  Relative
RMS error of the fitted potential: H2O 0.19, H3O+ 0.016, OH- 0.031.  The free
atoms carry zero.

    from fastforces.charges import mk_charges, charge_terms
    q, rms = mk_charges(atoms, charge=+1, basis="aug-cc-pvtz")   # one template
    terms = charge_terms(q)                                       # the `charge` rows of its .jsonl

These are **gas-phase** charges.  The water dipole comes out at 1.94 D, against
1.85 D experimental in the gas and 2.35 D for SPC/E, which is inflated on
purpose to stand in for liquid-phase polarization.  They were chosen because
they have a provenance, and because they sit close to ACKS2's `q_H = +0.304`
(1.71 D), which is what `sigma_O = 3.05 A` was tuned against.  They are also the
obvious lever if the liquid needs more cohesion.

`--write` drops each template's ACKS2 `atom` terms, which nothing reads under
point charges.  `lennardjones` and every bonded term are kept.

## What changed from `Water`, and what did not

Refit once with `fast-forces refit --force-constants`, starting from `Water`'s
fitted state.  The `k` values there are q-force's own, so one pass is the valid
use.  The run finished at 3 of 3 channels, fastest mode 3667 cm^-1, 0 of 3 bond
types over the 4400 cap, so still dt = 0.5 fs.

- **Bonded terms: unchanged to 1e-7.**  Every intramolecular pair in these
  templates is within `exclusion_depth`, so the intramolecular Coulomb is
  excluded under either electrostatic term and the atomization fits see the same
  target.  H3O+'s asymptote `h` settled on the 1.0 eV floor (it was 1.0004).
- **Coupling amplitudes: all three fell**, from -3.91 / -3.89 / -4.11 eV to
  -3.21 / -2.76 / -2.54 eV.  At a transition state each diabat now carries its
  own intermolecular charges: the ion pair of the autoionization product is
  stabilized by its own Coulomb attraction, and a hop's two diabats put the +1
  on different oxygens.  So less of each barrier has to come from `A`.

Water dimer on the Cs geometry `tests/test_water_structure.py` pins, binding
energy in eV relative to 14 A:

    R_OO (A)      2.70     2.80     2.85     2.91     3.00     3.30     4.00
    Water        -0.039   -0.089   -0.098   -0.102   -0.100   -0.075   -0.035
    this set     -0.045   -0.099   -0.109   -0.114   -0.113   -0.088   -0.042

against a CCSD(T) -0.218 eV at 2.91 A.  The liquid has not been re-measured
with these charges: density, pressure and g(r) as reported in `../Water/README.md`
are ACKS2 numbers.

## Checks

- `tests/test_pointcharge.py`: forces and virial through `System` against finite
  differences, including two interacting Zundel complexes.  That case needs the
  block sweep; stopping after one pass leaves 0.058 eV/A of error.
- NVE, 800 x 0.25 fs from the Zundel transition state with three spectator
  waters, open boundaries: max |E - E0| 1.8 meV with no drift, a mixed block on
  every step, 5 topology changes.
- NVE, 400 x 0.25 fs, H3O+ in 26 waters in a 9.3 A periodic box (Ewald, charged
  cell): max |E - E0| 3.0 meV.  26 ms per step, against 27.5 ms for `Water` on
  the same box.

## Caveats

- The EVB admission gate (`basis.py`) still screens candidate states on their
  *bonded* gap.  A state that the solvent's field stabilizes strongly, like an
  autoionization product in the liquid, is screened as though it were in vacuum.
  The energy stays conservative, because the gate is smooth, but the basis can
  be missing states the electrostatics would have favoured.
- Everything `../Water/README.md` says about protonation states, electron
  bookkeeping, geometry sources, the degenerate channels and the coupling width
  applies here unchanged.

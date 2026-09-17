# Equation Reference

The potential energy of **one fixed bonding pattern**.  Every equation below is
a function of atomic positions and a parameter set; nothing here knows about
reaction networks or state mixing.

```
E  =  E_bonded  +  E_ZBL  +  E_12-6  +  E_Coulomb  -  E_excl
      \________/  \______________________________/  \______/
       qforce.py   summed over every pair in the     removes the
       per term    system, no bond graph consulted   near-neighbour
                   (zbl.py, lj.py, acks2.py)         double count
```

## Units

| where | length | energy | notes |
| --- | --- | --- | --- |
| `qforce.py` internals, `lj.pair_potential` | nm | kJ/mol | converted to eV / Å at the end of `QForce.__call__` |
| `zbl.py`, `acks2.py` | Å | eV | ZBL constants and `CCOUL` are stated in these units |
| everything returned to ASE | Å | eV | forces eV/Å, stress eV/Å³ |

Angles are radians.  `SHAPE_DECAY`, `PHI_B` and the like are dimensionless;
`SCREENING_LENGTH` is Å, `CCOUL` is eV·Å, `GAMMA` is 1/Å.

---

## Bonded terms (`qforce.py`)

Each is a `compute_<type>` method dispatched by term type.  `r` is a bond
length, `θ` a bond angle, `φ` a dihedral.  Throughout

```
dr  = r - r0                       θ and φ enter only as
dc  = cos θ - cos θ0               cosines, never as angles
cosφ_n = 1 + cos(n φ - φ0)
```

### bond — Morse with a one-sided shape term (default)

```
a  = sqrt( k / 2 Dw )              Dw = D + BOND_ASYMPTOTE   if dr > 0
                                   Dw = D                    if dr <= 0
s  = a * max(dr, 0)

E  = Dw [ 1 - exp(-a dr) ]^2  -  D  +  Dw * c * s^3 * exp(-b s)
```

Parameters `D, r0, k, c, b`.  The `-D` offset puts the minimum at `-D`; the
dissociated limit is `BOND_ASYMPTOTE = 1.0 eV` above zero, so the well the
exponential climbs is `D + BOND_ASYMPTOTE` deep while the minimum stays at `-D`.
The join at `dr = 0` is C2 (the curvature there is `2 Dw a^2 = k` whatever `Dw`
is), so no fitted frequency sees the branch.  The shape term is
`O(s^3)`, so `D`, `r0` and the curvature at `dr = 0` are untouched by `c`; it is
clamped off on the compressed branch.  `b` defaults to `SHAPE_DECAY` for term
files that predate it, and is subject to `c <= c_max(b)` from
`fit.dissociation.shape_bound`.

### bond — harmonic (`bond_form="harmonic"`, non-reactive use)

```
E = 0.5 * k * dr^2                 D unused
```

### angle — cosine harmonic

```
E = 0.5 * k * (cos θ - cos θ0)^2
```

### Cross terms

| type | energy | parameters | atoms |
| --- | --- | --- | --- |
| `bondbond` | `max( k dr1 dr2 , -10 )` | `r1_0, r2_0, k` | 4 (two bonds) |
| `bondangle` | `max( k dr dc , -20 )` | `theta0, r0, k` | 5 (angle + bond) |
| `angleangle` | `k dc1 dc2` | `theta1_0, theta2_0, k` | 6 (two angles) |

The two lower clips are floors on the energy, in kJ/mol, applied to the raw
product; the gradient is zeroed where the clip is active.

### Dihedrals

`φ = atan2(S, C)` about the central bond, with `S = (axis × u)·v`, `C = u·v`,
`axis = vb/|vb|`, and `u`, `v` the projections of the outer bond vectors
perpendicular to `axis`.

| type | energy | parameters | atoms |
| --- | --- | --- | --- |
| `periodicdihedral` | `k (1 + cos(n φ - φ0))` | `phi0, n, k` | 4 |
| `dihedralbond` | `k dr (1 + cos(n φ - φ0))` | `+ r0` | 6 |
| `dihedralangle` | `k dc (1 + cos(n φ - φ0))` | `+ theta0` | 7 |
| `dihedralangleangle` | `k dc1 dc2 (1 + cos(n φ - φ0))` | `+ theta0_1, theta0_2` | 4 |

### reference

```
E = E0                             constant, zero force, zero virial
```

A per-molecule additive shift that puts different templates on a common
absolute energy scale.  Set at load time as the residual between the reference
atomization energy and the depth the Morse bonds already supply.

---

## Repulsion — tapered ZBL (`zbl.py`)

Summed over every pair, parameterized by atomic number alone.

```
a    = SCREENING_LENGTH / ( Z1^0.23 + Z2^0.23 )
x    = r / a
phi(x) = sum_k C_k exp(-B_k x)          C = (0.18175, 0.50986, 0.28022, 0.02817)
                                        B = (3.19980, 0.94229, 0.40290, 0.20162)
f(r) = 1 / ( 1 + exp( (r - TAPER_RADIUS) / TAPER_WIDTH ) )

u(r) = f(r) * CCOUL * Z1 * Z2 * phi(x) / r
```

`f` is a Fermi switch that turns the term **off** above `TAPER_RADIUS`; its
derivative is `df/dr = -f (1 - f) / TAPER_WIDTH`.  The switch and its derivative
are applied inside `pair_potential`, so callers differentiate a single
consistent function.

```
SCREENING_LENGTH = 0.46850 Å        TAPER_RADIUS = 1.5  Å
CCOUL            = 14.399645 eV·Å   TAPER_WIDTH  = 0.12 Å
```

Retained fraction `f` at the bond lengths the fit leans on: 0.998 (H–H, 0.741 Å),
0.989 (O–H, 0.958 Å), 0.918 (O–O, 1.208 Å).

---

## Dispersion / contact — switched 12-6 (`lj.py`)

Summed over every pair.  Pair parameters are the geometric mean of the per-atom
ones, `σ_ij = sqrt(σ_i σ_j)`, `ε_ij = sqrt(ε_i ε_j)`.

```
g(r)  = 1 - 1 / ( 1 + exp( (r - SWITCH_RADIUS) / SWITCH_WIDTH ) )
r_e   = max( r, CORE_FRACTION * σ )

u_126 = 4 ε [ (σ/r_e)^12 - (σ/r_e)^6 ]          plus, for r < CORE_FRACTION σ,
        + du/dr|_{r_e} * (r - CORE_FRACTION σ)   the C1 tangent continuation

u(r)  = g(r) * u_126(r)
```

`g` is `zbl.taper` reflected — same width, turning the term **on** above
`SWITCH_RADIUS`, with `dg/dr = g (1 - g) / SWITCH_WIDTH`.  The linear core
continuation bounds what a compressed bond can contribute (≈5700 eV rather than
1e21 eV at 0.024 Å), which is what keeps the exclusion subtraction below
numerically exact.

```
SWITCH_RADIUS = 0.22 nm (2.2 Å)     CORE_FRACTION = 0.4
SWITCH_WIDTH  = TAPER_WIDTH / 10    EXCLUSION_DEPTH = 3
```

`SWITCH_RADIUS` is deliberately **not** `zbl.TAPER_RADIUS`: the two are not
complementary and leave a gap from ~1.6 to ~2.0 Å where both are small.

---

## Electrostatics — ACKS2 (`acks2.py`, `ewald.py`)

### Charge kernel

```
K_ij = erf( GAMMA * r_ij ) / r_ij            GAMMA = 2.0 / Å
```

Finite at contact (→ 2·GAMMA/√π ≈ 2.257), so the charges saturate rather than
diverge.  Two implementations behind one interface, selected on `pbc`:

- `MinimumImage` — nearest image only; `K_ii = 0`.  Open boundaries, and the
  fallback for a slab or wire.
- `Ewald` — the full lattice sum, split at `kappa`:

```
K_ij = sum_n' [ erf(GAMMA |r_ij + n|) - erf(kappa |r_ij + n|) ] / |r_ij + n|
     + (4 pi / V) sum_{k != 0} exp( -k^2 / 4 kappa^2 ) / k^2 * cos(k . r_ij)
     - delta_ij * 2 kappa / sqrt(pi)
     + background,        background = -pi / (kappa^2 V)
```

Here `K_ii != 0` — an atom interacts with its own images.  The `k = 0` term is
dropped and its neutralizing background added back explicitly.  `kappa` and the
reciprocal cutoff are both set by `ACCURACY = 1e-8`.

### The solve

Per-atom parameters `mu` (electronegativity), `eta` (hardness), `soft_amp` and
`soft_decay` (the softness kernel).  The stationary point of the ACKS2
functional in the charges `Q` and the Kohn–Sham potentials `u` is the linear
system `A x = b`, `x = [Q, u, λ_tot, λ_KS]`:

```
        | K + 2 diag(eta)     -I        -1   0 |        | -mu |
  A  =  |      -I           -L_X         0  -1 |   b =  |  0  |
        |      -1^T            0         0   0 |        |  0  |
        |       0            -1^T        0   0 |        |  0  |

  X_ij   = soft_amp_i * soft_amp_j * exp( -r_ij / tau_ij ),
           tau_ij = ( soft_decay_i + soft_decay_j ) / 2
  L_X    = diag( sum_j X_ij ) - X                 (the graph Laplacian of X)
```

The last two rows are the constraints `sum_i Q_i = 0` and `sum_i u_i = 0`.  Only
the `K` block and the `X` block depend on geometry — `compute_response_forces`
differentiates exactly those two.  The hardness is *added* to the Coulomb
diagonal, not written over it: under periodic boundaries `K_ii` is a real
self-image interaction and belongs in the equilibration alongside `2 eta`.  The
solve uses the **unmasked** kernel over every pair, so the charges are a
function of positions and elements alone.

Note that `A` carries the bare kernel while the energy below carries `CCOUL`, so
`mu` and `eta` are in the units that implies rather than in eV directly.

### Energy and forces

```
E = (CCOUL / 2) * sum_ij  S_ij * Q_i * Q_j * K_ij            CCOUL = 14.4 eV·Å
```

with `S` the exclusion screen (below).  The charges move when the atoms do, so
the gradient has two pieces:

```
dE/dr  =  (dE/dr)|_Q  -  lam^T (dA/dr) x,        lam = A^-1 (dE/dx)
```

one extra solve rather than one per coordinate (`A` is symmetric, so no
transpose).  The screen belongs to `dE/dx` only — `A` is the unscreened matrix,
so `dA/dr` must be contracted against an unmasked weight.

---

## Exclusions (`exclusions.py`)

The three nonbonded terms above are summed over every pair with no reference to
the bond graph.  Pairs within `EXCLUSION_DEPTH = 3` bonds of each other are then
removed, so that a molecule's geometry is set by its bonded terms:

| term | how it is removed | where |
| --- | --- | --- |
| 12-6 | `exclusion` term, `E = -u_LJ(r, σ_ij, ε_ij)` | `QForce.compute_exclusion` |
| ZBL | `zblexclusion` term, `E = -u_ZBL(r, Z1, Z2)` | `QForce.compute_zblexclusion` |
| Coulomb | screen `S_ij = 0` on the energy functional | `ACKS2.compute` |

The first two are ordinary additive pair corrections and go through the *same*
`pair_potential` as the whole-system sum, switch included, so the cancellation
is exact.  Coulomb cannot be subtracted that way — the charges come from a
solve whose matrix contains the kernel — so it is screened in the energy
functional while the solve stays unmasked.

---

## Gradient conventions

Every term returns `(energy, forces, virial)`.

```
F_i     = -dE/d(pos_i)
W_ab    =  sum_v  v_a * (dE/dv)_b              virial, a 3x3
stress  =  W / V                               what ase.py publishes
```

Every energy here is a function of minimum-image displacement vectors alone, so
a homogeneous strain maps `v -> (I + e) v` and the virial is built from the same
per-pair gradient the forces are scattered from — no new derivative is needed.

**Unit trap:** the virial is an energy.  It converts with `units.kJ / units.mol`
and *no* length factor, unlike the forces.

Adding a new functional form means adding one `compute_<type>` method returning
that triple, plus a case in `tests/test_gradients.py` (finite differences of the
forces) and one in `tests/test_stress.py` (finite differences of the virial
against the cell).

---

## Constants at a glance

| constant | value | module | changing it invalidates |
| --- | --- | --- | --- |
| `BOND_ASYMPTOTE` | 1.0 eV | `qforce` | every `.jsonl` in every dataset |
| `SHAPE_DECAY` | fallback `b` | `qforce` | — |
| `TAPER_RADIUS` | 1.5 Å | `zbl` | every `.jsonl` in every dataset |
| `TAPER_WIDTH` | 0.12 Å | `zbl` | every `.jsonl` in every dataset |
| `SWITCH_RADIUS` | 0.22 nm | `lj` | every `.jsonl` in every dataset |
| `CORE_FRACTION` | 0.4 | `lj` | — |
| `EXCLUSION_DEPTH` | 3 | `lj` | every `.jsonl` in every dataset |
| `GAMMA` | 2.0 /Å | `ewald` | every `.jsonl` in every dataset |
| `ACCURACY` | 1e-8 | `ewald` | — (periodic only; templates are non-periodic) |
| `CCOUL` | 14.4 eV·Å (`acks2`), 14.399645 (`zbl`) | | |

The three radii and the exclusion depth sit inside `E_nonbonded`, which `fit/dissociation.py` solves
against, so any change to them requires re-running `scripts/fit.py
--force-constants` for both datasets.

---

*Note: `atom` terms carry a `soft_scale` parameter in the shipped `.jsonl`
files that no code currently reads.*

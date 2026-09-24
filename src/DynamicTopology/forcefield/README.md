# Equation Reference

Every equation this directory evaluates, and nothing about how they are
assembled.  There are two kinds.  The **diagonal** is the potential energy of
one fixed bonding pattern; the **off-diagonal** is the coupling between two of
them.  Both are functions of atomic positions and a parameter set alone —
which states exist, which channels connect them and how the Hamiltonian is
diagonalized is `basis.py` and `system.py`.

```
H_ss  =  E_bonded  +  E_ZBL  +  E_12-6  +  E_Coulomb  -  E_excl
         \________/  \______________________________/  \______/
          qforce.py   summed over every pair in the     removes the
          per term    system, no bond graph consulted   near-neighbour
                      (zbl.py, lj.py, acks2.py)         double count

H_st  =  V(x)        coupling.py -- one Gaussian per reaction channel,
                     nonzero only near the geometry it is centred on
```

Only two of those five depend on the bonding pattern: `E_bonded` and `E_excl`.
The three whole-system sums are deliberately blind to the bond graph, so they
are the same number on every state and are evaluated once, outside the
Hamiltonian — which is what makes the exclusions the interesting term rather
than a bookkeeping detail.

**The exception is `electrostatics = "pointcharge"`** (`pointcharge.py`), which
replaces ACKS2 with charges fixed per template.  Those differ between the states
of a proton transfer, so `E_Coulomb` joins `E_bonded` as a function of the
bonding pattern and sits on the diagonal — see "Electrostatics — fixed point
charges" below.

## Units

| where | length | energy | notes |
| --- | --- | --- | --- |
| every force field, every in-memory term, `ForceFieldParams` | Å | eV | ASE units throughout; nothing converts during a force call |
| `.jsonl` on disk | Å | eV | stored as held; nothing converts on read or write |
| everything returned to ASE | Å | eV | forces eV/Å, stress eV/Å³ |
| q-force XML, OpenMM export (fast-forces) | nm | kJ/mol | `io/units.py` converts at that boundary |

`io/units.py:UNIT_POWERS` is the one table of what converts how between ASE and
q-force/OpenMM units, and a parameter missing from it is refused rather than
converted by a guessed factor.  The ACKS2 `atom` block, `charge` and every EVB
coupling convert by 1: neither q-force nor OpenMM states them.  A bond `r0`
under 0.5 in a `.jsonl` is refused on read as a file from before the move to
eV/Å (`io.json.LEGACY_R0`).

Angles are radians.  `PHI_B` and the like are dimensionless;
`SCREENING_LENGTH` is Å, `ccoul` is eV·Å, `gamma` is 1/Å.  A coupling's `A` is
eV and its `a` is 1/Å²; `r0`, `ra0`, `rb0` are Å and `t0` radians.  The global
parameters and their units are tabulated at the bottom of this file.

---

## Term types at a glance

`core/types.py` defines a term as `{"type", "atoms", "kwargs"}`; each force field
dispatches on the type to its own `compute_<type>` and silently skips a type it
has no method for.  Every type in play, and where it comes from:

| type | evaluated by | kwargs | comes from |
| --- | --- | --- | --- |
| `bond` | `QForce` | `D, r0, k, h` | template `.jsonl` |
| `angle` | `QForce` | `theta0, k` | template `.jsonl` |
| `bondbond`, `bondangle`, `angleangle` | `QForce` | see cross terms | template `.jsonl` |
| `periodicdihedral`, `dihedralbond`, `dihedralangle`, `dihedralangleangle` | `QForce` | see dihedrals | template `.jsonl` |
| `reference` | `QForce` | `E0` | set by `ReactionSet.load` |
| `atom` | `ACKS2` | `mu, eta, soft_amp, soft_decay` | template `.jsonl` |
| `charge` | `PointCharge` | `q`, in e | template `.jsonl` |
| `lennardjones` | `LennardJones` | `sigma, eps` | template `.jsonl` |
| `exclusion` | `QForce` | `sigma, eps`, already combined | derived at load |
| `zblexclusion` | `QForce` | `z1, z2` | derived at load |
| `coulombexclusion` | `ACKS2` or `PointCharge` | none — the pair is the whole term | derived at load |
| `twobody`, `threebody`, `rmsd` | `EVBCoupling` | see coupling | reaction `.jsonl` |

`atom`, `charge` and `lennardjones` carry no energy of their own; they are how a template
hands its per-atom nonbonded parameters to a whole-system sum.  `ZBL` takes no
terms at all — it reads atomic numbers straight off the `Atoms`, so the only
thing it can be a function of is the geometry and the elements.

The three exclusion types are **not** in any `.jsonl`.  `ReactionSet.load`
derives them per template from the bond graph at the dataset's
`exclusion_depth`, which is why a raw term list scored outside the loader reads
hundreds of eV high — H2 came out at +842 eV against a reference of −4.67.
Anything evaluating a term list against a reference energy goes through
`exclusions.with_exclusions`.

---

## Bonded terms (`qforce.py`)

Each is a `compute_<type>` method dispatched by term type.  `r` is a bond
length, `θ` a bond angle, `φ` a dihedral.  Throughout

```
dr  = r - r0                       θ and φ enter only as
dc  = cos θ - cos θ0               cosines, never as angles
cosφ_n = 1 + cos(n φ - φ0)
```

### bond — Morse with a per-bond asymptote (default)

```
a  = sqrt( k / 2 Dw )              Dw = D + h   if dr > 0
                                   Dw = D       if dr <= 0

E  = Dw [ 1 - exp(-a dr) ]^2  -  D
```

Parameters `D, r0, k, h`, with `h` optional (eV, like `D`) and defaulting to
`bond_asymptote`.  The `-D` offset puts the minimum at `-D`; the dissociated
limit is `h` above zero, so the well the exponential climbs is `D + h` deep while
the minimum stays at `-D`.  The join at `dr = 0` is C2 (the curvature there is
`2 Dw a^2 = k` whatever `Dw` is), so no fitted frequency sees the branch or `h`.
At fixed `k`, raising `h` lifts every stretched geometry monotonically towards
the harmonic `k dr^2 / 2`, and the curve stays a Morse, so it cannot turn over.
fast-forces' `refine.fit_force_constants` fits `h` per bond type, bounded below by
`bond_asymptote`; HCombustion's fitted values run from 1.0 (the floor) to 6.6 eV
(O2).

This replaced a one-sided Hulburt-Hirschfelder shape term
`Dw c s^3 exp(-b s)`, `s = a max(dr, 0)`, which lifted the mid-range with two
parameters per bond under a monotonicity bound `c <= c_max(b)`.  From the same
q-force start both reach 19/19 HCombustion channels at q-force's frequencies;
`h` does it with one parameter and no bound, in 16 s against about 650.  The two
lift different parts of the curve, the shape term mid-range and `h` out towards
dissociation.

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
| `bondbond` | `max( k dr1 dr2 , -10 kJ/mol )` | `r1_0, r2_0, k` | 4 (two bonds) |
| `bondangle` | `max( k dr dc , -20 kJ/mol )` | `theta0, r0, k` | 5 (angle + bond) |
| `angleangle` | `k dc1 dc2` | `theta1_0, theta2_0, k` | 6 (two angles) |

The two lower clips are floors on the energy -- q-force's -10 and -20 kJ/mol,
held in eV as `qforce.CLIP_BONDBOND` / `CLIP_BONDANGLE` -- applied to the raw
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
f(r) = 1 / ( 1 + exp( (r - taper_radius) / taper_width ) )

u(r) = f(r) * zbl_ccoul * Z1 * Z2 * phi(x) / r
```

`f` is a Fermi switch that turns the term **off** above `taper_radius`; its
derivative is `df/dr = -f (1 - f) / taper_width`.  The switch and its derivative
are applied inside `pair_potential`, so callers differentiate a single
consistent function.

```
SCREENING_LENGTH = 0.46850 Å        taper_radius = 1.5  Å
zbl_ccoul        = 14.399645 eV·Å   taper_width  = 0.12 Å
```

Retained fraction `f` at the bond lengths the fit leans on: 0.998 (H–H, 0.741 Å),
0.989 (O–H, 0.958 Å), 0.918 (O–O, 1.208 Å).

---

## Dispersion / contact — switched 12-6 (`lj.py`)

Summed over every pair.  Pair parameters are the geometric mean of the per-atom
ones, `σ_ij = sqrt(σ_i σ_j)`, `ε_ij = sqrt(ε_i ε_j)`.

```
g(r)  = 1 - 1 / ( 1 + exp( (r - switch_radius) / switch_width ) )
s(r)  = soft_core + (r/σ)^6

u_126 = 4 ε [ s^-2 - s^-1 ]                     the soft-core 12-6; bare at soft_core = 0

u(r)  = g(r) * u_126(r)
```

`g` is `zbl.taper` reflected — same width, turning the term **on** above
`switch_radius`, with `dg/dr = g (1 - g) / switch_width`.  The soft core
bounds what a compressed bond can contribute — `4 ε (1/c² - 1/c)` at contact,
292 eV for the O–O pair rather than 1e21 eV at 0.024 Å — which is what keeps the
exclusion subtraction below numerically exact.

```
switch_radius = 2.2 Å               soft_core = 0.01
switch_width  = taper_width         exclusion_depth = 3
```

`switch_radius` is deliberately **not** `taper_radius`: the two are not
complementary and leave a gap from ~1.6 to ~2.0 Å where both are small.

---

## Electrostatics — ACKS2 (`acks2.py`, `ewald.py`)

### Charge kernel

```
K_ij = erf( gamma * r_ij ) / r_ij            gamma = 2.0 / Å
```

Finite at contact (→ 2·gamma/√π ≈ 2.257), so the charges saturate rather than
diverge.  Two implementations behind one interface, selected on `pbc`:

- `MinimumImage` — nearest image only; `K_ii = 0`.  Open boundaries, and the
  fallback for a slab or wire.
- `Ewald` — the full lattice sum, split at `kappa`:

```
K_ij = sum_n' [ erf(gamma |r_ij + n|) - erf(kappa |r_ij + n|) ] / |r_ij + n|
     + (4 pi / V) sum_{k != 0} exp( -k^2 / 4 kappa^2 ) / k^2 * cos(k . r_ij)
     - delta_ij * 2 kappa / sqrt(pi)
     + background,        background = -pi / (kappa^2 V)
```

Here `K_ii != 0` — an atom interacts with its own images.  The `k = 0` term is
dropped and its neutralizing background added back explicitly.  `kappa` and the
reciprocal cutoff are both set by `accuracy = 1e-8`.

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

Note that `A` carries the bare kernel while the energy below carries `ccoul`, so
`mu` and `eta` are in the units that implies rather than in eV directly.

### Energy and forces

```
E = (ccoul / 2) * sum_ij  S_ij * Q_i * Q_j * K_ij            ccoul = 14.4 eV·Å
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

## Electrostatics — fixed point charges (`pointcharge.py`)

Selected by `electrostatics = "pointcharge"` in the manifest, in place of ACKS2.
Every template carries a `charge` term per atom, summing to its formal charge,
so an H3O+ carries +1 and the water it hops to carries 0: the charge moves with
the proton, which ACKS2's single sum-zero constraint cannot express.  Same
kernel `K` as ACKS2, so the same Ewald sum under full periodicity.

Each state `s` of a block `B` gets, on its diagonal,

```
H_ss += E_intra(q_s) + q_s . V_B
E_intra(q_s) = (ccoul/2) q_s . K_BB . q_s  -  ccoul sum_{(i,j) in excl(s)} q_si q_sj g(r_ij)
V_B          = ccoul K_{B, not B} . qbar,      qbar_B' = sum_t w_t q_t   (other blocks)
```

with `g(r) = erf(gamma r)/r` at the minimum image.  The blocks see each other only
through their averaged charges, so `System.calculate` diagonalizes each
multi-state block in turn and repeats until no weight moves (`SCF_TOLERANCE`);
with one multi-state block that is one pass.  The total is

```
E = sum_B lambda_B - E_inter          E_inter = (ccoul/2) sum_{i,j in different blocks} qbar_i qbar_j K_ij
```

and at fixed weights the whole Coulomb term is one contraction,
`W_ij = (ccoul/2) sum_s w_s q_si q_sj` within a block and
`(ccoul/2) qbar_i qbar_j` between blocks, less the excluded pairs.  Stationary in
every eigenvector at convergence, so Hellmann-Feynman forces and virial are
exact.  No solve, no response term.

**The exclusion removes only the direct pair `g(r_ij)`, not `K_ij`.**  Under Ewald
`K_ij` also contains `i`'s interaction with every image of `j`; removing that as
well leaves `(ccoul/2) K_self sum_i q_i^2` per molecule — −0.91 eV per water in a
12.43 Å box, falling only as 1/L, so it acts on the pressure.  With the direct
pair alone the residue is the molecule's dipole–image energy, −1.9 meV.  ACKS2's
screen removes the whole `K_ij`; see the note in `pointcharge.py`.

---

## Exclusions (`exclusions.py`)

The three nonbonded terms above are summed over every pair with no reference to
the bond graph.  Pairs within `exclusion_depth = 3` bonds of each other are then
removed, so that a molecule's geometry is set by its bonded terms:

| term | how it is removed | where |
| --- | --- | --- |
| 12-6 | `exclusion` term, `E = -u_126(r, σ_ij, ε_ij)` | `QForce.compute_exclusion` |
| ZBL | `zblexclusion` term, `E = -u_ZBL(r, Z1, Z2)` | `QForce.compute_zblexclusion` |
| Coulomb | `coulombexclusion` term, in two places (below) | `ACKS2`, `System.calculate` |
| Coulomb, point charges | `coulombexclusion` term, per state inside `E_intra` | `PointCharge` |

The first two are ordinary additive pair corrections, evaluated per diabatic
state alongside the bonded terms, and they go through the *same* `pair_potential`
as the whole-system sum, switch included, so the cancellation is exact.  A pair
gets an `exclusion` only if both atoms carry a nonzero σ and ε: `sigma_H = 0` in
the Water set makes the whole-system 12-6 identically zero on every H–H pair, so
there is nothing to cancel and a term would be an expensive no-op.  ZBL and
Coulomb have no such escape — ZBL has no free parameters and the charge kernel is
element-independent — so every near pair gets one of each.

### Why Coulomb takes two

`ACKS2`'s charges come from a solve whose matrix *contains* the kernel, so the
exclusion cannot simply be subtracted from the energy.  Masking the kernel inside
the solve would be the more obviously consistent thing to do and is still wrong:
it makes the charges a function of the bond graph, and a term evaluated once
outside the Hamiltonian must not be (measured at **0.88 eV** of pivot dependence,
`tests/test_evb_invariants.py`).  So the charges are solved once from the
**unmasked** kernel and the exclusion is applied afterwards, as the same quantity
seen from two directions:

```
per state   H_ss += -ccoul * sum_{(i,j) in excl(s)} Q_i Q_j K_ij
            one lookup per excluded pair into a kernel already built
            (ACKS2.exclusion_energy).  This is the half that lets the
            exclusion decide which bonding pattern is lower.

once        S_ij = 1 - sum_s w_s M_ij^s
            E    = (ccoul / 2) sum_ij S_ij Q_i Q_j K_ij
            M^s the mask of state s, w_s its ground-state weight
            (ACKS2.screen_matrix).  One contraction, one adjoint solve,
            for the whole system.
```

Fractional entries of `S` are the normal case and not an interpolation: every
state's correction is linear in its own mask and the kernel contraction is linear
in its weight, so the Hellmann-Feynman sum `sum_s w_s dE_s/dr` collapses into a
single weight matrix.  A pair bonded in every state of a block comes out at
exactly 0; one bonded in some of them is removed in proportion.  `S` is held
fixed under the derivative, which is what Hellmann-Feynman says to do with
eigenvector components.

That is why `ACKS2` is split in two.  `prepare` solves the charges *before* the
diagonalization, because the per-state diagonal corrections need them; `compute`
applies the screen *after* it, because the screen needs the weights the
diagonalization produced.  One asymmetry has to be got right in
`compute_response_forces`: `dE/dQ` is screened because the energy is, while the
`-lam^T (dA/dr) x` weight is not, because `A` never was.  Either mistake shows up
only as NVE drift.

**What the exclusion costs is the intramolecular half of the polarization
response, and that is a real 0.07 eV of the hydrogen bond.**  The water dimer
well goes from 0.1677 eV at 2.85 Å to 0.1024 eV at 2.91 Å — datasets held fixed,
so this is the mechanism and not a refit.  The molecules polarize each other, the
charges grow (`q_H` +0.30399 → +0.31264 at 2.85 Å), and the intramolecular
Coulomb energy falls along with the intermolecular one; booking that gain inside
the molecule is exactly what the exclusion exists to stop.  `eta` is the lever
that pays it back, since the intermolecular term scales as `q²` — see
`datasets/Water/README.md`.

The charges themselves are untouched by all of this: an isolated water still
comes out at `q_H = +0.30399`, so nothing about a molecule's dipole moves.  The
exclusion is, however, the first thing here to contract the periodic kernel
against a **non-neutral** weight, which is what exposed the missing `k = 0`
background above — an individual `K_ij` was not a well-defined number until that
was carried explicitly.

---

## Coupling — the off-diagonal (`coupling.py`)

Everything above is a diagonal entry.  This module is `H_st`: one Gaussian per
reaction channel, dispatched by term type exactly as `QForce` is, in ASE units
(Å, eV) throughout.

**There are three forms, and a channel's own connectivity change picks which.**
`fast-forces refit` routes on the bond-graph difference between the first and last
frame of the channel's `.xyz` — `_fission` first, then `_transfer`, then the RMSD
fallback:

| type | the channel | V | parameters |
| --- | --- | --- | --- |
| `twobody` | one bond broken or formed, separating two fragments | `A exp(-a (r - r0)²)` | `A, a, r0` |
| `threebody` | one bond broken and one formed, sharing an atom | `A exp(-a g)` | `A, a, ra0, rb0, t0` |
| `rmsd` | anything else | `(A/M) Σ_m exp(-a ρ_m²)` | `A, a` + a TS ensemble |

`twobody`'s `r` is the length of the bond that changes.  `threebody`'s `g` is the
transferring atom's own triangle, on `atoms = (donor, transferring atom,
acceptor)` with the mover central, as `compute_angle` orders a vertex:

```
ra = |H - D|,   rb = |H - A|,   d = |A - D|
d0 = sqrt( ra0² + rb0² - 2 ra0 rb0 cos t0 )        law of cosines
g  = (ra - ra0)² + (rb - rb0)² + (d - d0)²         Å²
```

`t0` is stored as an angle because that is the readable parameter and used as a
length so that one width `a` is dimensionally consistent across all three sides.
The three sides are a **complete** description of the triangle and a
non-redundant one, so this is the transfer's full geometry, not a projection of
it.  `rmsd`'s `ρ_m` is the optimally superposed RMSD to the m-th frame of the
channel's stored transition-state ensemble — Diamond's quaternion form of the
rotational superposition (`kabsch`), with unit weights and no rescaling, batched
over the channels of one template.  `twobody` and `threebody` need no ensemble at
all: they are *centred* on a geometry rather than measured against one, and they
accept the argument only for the uniform `compute_*` signature.

**Two independent things distinguish the forms: where the amplitude comes from,
and what the width is measured in.**

| form | amplitude from | width measured in |
| --- | --- | --- |
| `twobody` | the diabatic crossing | the breaking bond's length |
| `threebody` | the reference barrier | the transferring atom's triangle |
| `rmsd` | the reference barrier | the RMSD to the whole geometry |

Only `twobody` changes the amplitude, and only because it has to.  A transfer's
amplitude is the reference barrier `E*` inverted through the 2x2 secular
equation at the transition state, where `V = A` exactly:

```
A = -sqrt( (Hm - E*)^2 - dH^2 ),     Hm = (H_R + H_P) / 2
                                     dH = (H_R - H_P) / 2
```

A fission has no saddle, so there is no barrier to invert.  The bound diabat
*is* the ground state up to the crossing, so a perfect reactant diabat puts the
reference barrier exactly on it, the discriminant above vanishes identically,
every imperfection makes it imaginary, and the best attainable `A` is zero —
which `basis.EVBBasis` drops at every geometry.  That is why H2, OH and H2O could
not come apart at all, and no refit of the force field moves it.  Centring on
the crossing asks a question that has an answer instead.  It also makes the
ordinary admission gate sufficient: at a crossing the diabats are degenerate, so
the stabilization `hypot(dH, V) - |dH|` is `|A|` exactly, and a Gaussian centred
there is at its maximum exactly there — so a fission channel is admitted at the
one geometry where the topology decision is taken, by construction rather than
by luck.

A transfer *does* have a saddle and a reference barrier, and a barrier is
ab-initio data worth fitting to, so `threebody` leaves `fit_amplitude` untouched
— `g = 0` at the reference triangle, so `V = A` there exactly, which is the one
property the inversion above needs — and changes only the metric the **width**
lives in.  An RMSD is a tolerance on all `3N` coordinates at once, so any
spectator switches the coupling off: with
`h2o-autoionization`'s three transfer atoms held exactly at its transition state,
displacing only the three spectators by 0.1 Å — less than thermal motion at 300 K
— takes the RMSD coupling from 4.14 eV to 8.4e-3, while the triangle form holds
at 4.14.  Under the RMSD form a 64-water box never admitted a single diabatic
state at any `eps` (`datasets/Water/README.md`); neutral water at 300 K never
puts all `3N` coordinates that close at once.

A crossing-centred fit is equally not the right answer for a transfer:
`h2o-autoionization`'s two diabats never cross along the proton coordinate at
all, the products being 9.8 eV uphill in the gas phase, so there is no degeneracy
to centre anything on.

**No shipped channel uses `rmsd` any more.**  All 22 channels across HCombustion
and Water carry a fitted `twobody` (6) or `threebody` (16) term, none on a
placeholder and none decoupled.  The RMSD form is retained as the fallback for a
channel that is neither a fission nor a transfer — two independent bond changes
sharing no atom have no single coordinate to be a function of.

Two conventions differ from `QForce`.  These `compute_*` methods return
`(E, F)`, not a triple; `__call__` forms the virial once, from absolute
positions:

```
W_ab  =  sum_i pos_a (dE/dpos_i)_b  =  -sum_i pos_a F_b
```

which is origin-independent because a coupling's forces sum to zero — for `rmsd`
because the superposition removes the centroid, for the other two because they
are pair and triangle terms.  (An RMSD to a fixed template is not
scale-invariant the way it is rotation- and translation-invariant, which is why
this term carries a stress at all rather than dropping out.)  And `pos` may carry
a leading batch axis: every channel of one template in one force call shares `n`
and the ensemble, so the fixed cost of the superposition is paid once per
template instead of once per channel.  Under PBC the fragment is unwrapped by
cumulative minimum image first, which is affine in the cell, so the expression
above is still the strain derivative.

The admission gate, the switching ramp that brings a channel's coupling in
smoothly, and the caching of channel couplings are `basis.py`, not here.

---

## Gradient conventions

Every diagonal term returns `(energy, forces, virial)`; a coupling returns
`(energy, forces)` and `EVBCoupling.__call__` forms the virial for it.

```
F_i     = -dE/d(pos_i)
W_ab    =  sum_v  v_a * (dE/dv)_b              virial, a 3x3
stress  =  W / V                               what ase.py publishes
```

Every energy on the diagonal is a function of minimum-image displacement vectors
alone, so a homogeneous strain maps `v -> (I + e) v` and the virial is built from
the same per-pair gradient the forces are scattered from — no new derivative is
needed.  A coupling is the one term written over absolute positions rather than
separations, and it gets away with it because its forces sum to zero; see its
section above.

The virial is an energy, in eV.  (It used to be a unit trap: while `QForce`
worked in nm it had to be rescaled by the energy factor alone, not the forces'
energy-over-length one.  Nothing rescales now.)

Adding a new functional form means adding one `compute_<type>` method returning
that triple — or that pair, for a coupling — plus a case in
`tests/test_gradients.py` (finite differences of the forces) and one in
`tests/test_stress.py` (finite differences of the virial against the cell).
Nothing else needs to change: dispatch is by name, and a type with no matching
method is silently skipped.

---

## Global parameters at a glance

These are **not module constants**.  They are fields of
`params.ForceFieldParams`, stated per dataset under `global_params` in the
manifest, and read through `params.active()` at call time.  The defaults below
are what a manifest that omits a key gets.

| parameter | default | unit | read by | changing it invalidates |
| --- | --- | --- | --- | --- |
| `bond_asymptote` | 1.0 | eV | `qforce`, `fit` | every `.jsonl` in the dataset |
| `taper_radius` | 1.5 | Å | `zbl` | every `.jsonl` in the dataset |
| `taper_width` | 0.12 | Å | `zbl`, and `switch_width` by default | every `.jsonl` in the dataset |
| `switch_radius` | 2.2 | Å | `lj` | every `.jsonl` in the dataset |
| `switch_width` | `taper_width` | Å | `lj` | every `.jsonl` in the dataset |
| `soft_core` | 0.01 | — | `lj` | every `.jsonl` in the dataset |
| `exclusion_depth` | 3 | bonds | `exclusions` | every `.jsonl` in the dataset |
| `exclude_coulomb` | `true` | — | `exclusions` | every `.jsonl` in the dataset |
| `gamma` | 2.0 | 1/Å | `ewald` | every `.jsonl` in the dataset |
| `electrostatics` | `"acks2"` | — | `system`, `evb`, `fit` | every `.jsonl` in the dataset |
| `accuracy` | 1e-8 | — | `ewald` | — (periodic only; templates are non-periodic) |
| `ccoul` | 14.4 | eV·Å | `acks2`, `pointcharge` | every `.jsonl` in the dataset |
| `zbl_ccoul` | 14.399645 | eV·Å | `zbl` | every `.jsonl` in the dataset |

`switch_width` defaults to `taper_width`, so setting the taper width sets both
unless `switch_width` is given explicitly.  `ccoul` and `zbl_ccoul` are the same
physical constant to different precision; they are two fields because collapsing
them would change one of the two terms for every dataset already fitted.

Every field is in ASE units, Å and eV, as the force field works in them.
(`switch_radius` was stated in nm while `lj` evaluated in nm; manifests fitted
before the switch to ASE units carry `0.22` and now state `2.2`.)

### Measurements behind the defaults

**`bond_asymptote` (1.0 eV).** Read off the fittable channel count and two
margins, refit at each candidate value:

| asymptote | fittable | rxn_16 | rxn_11 | rxn_15 |
| --- | --- | --- | --- | --- |
| 0.00 | 14 | −0.402 | +0.086 | +0.144 |
| 0.50 | 15 | −0.152 | +0.426 | +0.416 |
| 0.75 | 15 | −0.028 | +0.591 | +0.549 |
| 1.00 | 16 | +0.093 | +0.753 | +0.680 |
| 1.50 | 16 | +0.333 | +1.068 | +0.938 |
| 2.00 | 16 | +0.567 | +1.371 | +1.190 |

It is also the floor and default for each bond's fitted `h`.  1.0 is the first
value at which `rxn_16` — a genuine saddle, the one channel
that was ever a fitting failure rather than a barrierless one — comes out
fittable; past 1.5 nothing further is fittable. It also sets where a bonded
diabat crosses its own fragments':

| asymptote | H2 | HO | H2O |
| --- | --- | --- | --- |
| 0.75 | 2.53 Å | 3.15 Å | 4.05 Å |
| 1.00 | 2.38 Å | 2.93 Å | 3.74 Å |

Both rows are inside `ReactionSet.get_network`'s 4.0 Å bimolecular cutoff, so
the reverse channel is enumerated where the forward one hands over.

**`taper_radius` (1.5 Å), bounded from both sides.** From below, the wall has
to stay ahead of the ACKS2 contact funnel at every separation: at 1.2 Å the
H2 + O2 approach in `tests/test_collapse.py` already reads +0.52 eV against
+3.14 eV untapered, and by 1.0 Å that approach is downhill. From above, every
0.1 Å of extra reach costs roughly another 0.2 eV of hydrogen-bond depth —
measured on the water dimer, the minimum moves −0.148 eV at 1.4 → −0.116 eV at
1.5 → −0.090 eV at 1.6, against a −0.218 eV reference at 2.91 Å. 1.5 puts the
minimum at the right *separation* and takes the depth deficit as a known
residual.

**`taper_width` (0.12 Å)** is the smallest that keeps the switch smooth enough
to integrate: its peak contribution to `du/dr` is 2.4 eV/Å at the O-H bond
length, still far under the 17.8 eV/Å the unmodified term already carries
there.

**`switch_radius` (2.2 Å) is deliberately not `taper_radius`.** The tidy
design, `g = 1 - f` at ZBL's own radius, does not survive contact with
q-force's parameters: a Fermi switch decays by one factor of `e` per width
while 12-6 grows as `r**-12`, and at `rc = 1.5 Å` the switch is down to 1.1e-2
where 12-6 is up at 953 eV — a product of **+10.3 eV per O-H bond**. Bare 12-6,
and what survives at 2.2 Å / 0.12 Å:

| pair | bare | switched |
| --- | --- | --- |
| O–H bond, 0.958 Å | 953 eV | 0.0024 eV |
| H–H bond, 0.741 Å | 892 eV | 0.0002 eV |
| O–O bond, 1.208 Å | 1375 eV | 0.035 eV |
| water's 1-3 H···H, 1.51 Å | 0.138 eV | 0.0004 eV |
| the H-bond O···H, 1.94 Å | 0.146 eV | 0.014 eV |
| O···O contact, 2.6 Å | 0.076 eV | 0.069 eV |
| O···O contact, 2.4 Å | 0.262 eV | 0.202 eV |

The switched column is the soft-cored form (`soft_core = 0.01`) the force field
evaluates; with the bare 12-6 it read 0.031, 0.004 and 0.353 eV at the three
bond lengths and 0.073 and 0.220 eV at the two contacts.  The last two rows are
the wall this term exists to supply, retained at 91% and 77% (97% and 84% of
that by the switch, the rest by the soft core, which softens the wall inside
sigma by 6-8% there). The gap from ~1.6 to ~2.0 Å is deliberate: neither term acts there
(`zbl.taper` is down to 0.076 at 1.8 Å and 12-6 isn't yet up), and `ACKS2`
alone already binds the water dimer at −0.112 eV through it —
`tests/test_collapse.py` checks that an intermolecular approach stays uphill
through the gap.

**`soft_core` (0.01).** `u = 4 ε [s^-2 - s^-1]` with `s = c + (r/σ)^6` is
finite at contact, `4 ε (1/c² - 1/c) = 39600 ε`: 76 eV for H–H, 149 eV for
O–H and 292 eV for O–O, where the linear-tangent core it replaced (`core_fraction
= 0.4`) reached 5900, 11600 and 22800 eV.  The minimum stays at `s = 2`, so the
well depth is exactly ε at any `c` and its position moves in by 0.08%; the zero
crossing moves in by 0.17%, and beyond σ the potential differs from the bare
12-6 by at most 4cε = 0.04 ε, at r = σ.  What it does change is the wall inside σ, which it lowers by
`~2c/(r/σ)^6` in relative terms: 7% for O–O at 2.5 Å, 12% at 2.2 Å.  Unlike the
linear core, which never engaged above 0.4σ, this reaches the intermolecular
contact the term exists for, so it is a real change to the water wall, not
only a numerical guard.

**`accuracy` (1e-8).** Loosening it is cheap in reciprocal vector count
(`(-log accuracy)^3`) but expensive in the *stress*: `kappa` is derived from
the cell, so a strained cell truncates at a slightly different splitting, and
that residual is a spurious contribution to the finite-difference virial with
no analytic counterpart, amplified by `2 log(1/accuracy)` relative to the
truncation error itself. At 1e-8 it lands around 1e-6 eV, comfortably under
`tests/test_stress.py`'s tolerance; at 1e-6 it would be at the edge of it.

### Why they live in the manifest

Everything in the table but `accuracy` sits inside
`E_bonded + E_nonbonded`, which fast-forces' `refine` solves each template's
depth scale against.  As module constants they were a property of the installed
**source tree**: a checkout whose `zbl.TAPER_RADIUS` had moved evaluated every
dataset at the new radius and reported energies that no longer matched the fit,
with nothing to say so.  As manifest fields they are a property of the
**dataset**, so a set fitted at one radius carries that radius with it.

```json
"global_params": {
    "taper_radius": 1.5,
    "exclusion_depth": 3
}
```

`ReactionSet.load` reads the block before it touches a single template — the
exclusion derivation and the reference shift both depend on it — and activates
it for the process.  An unknown key is an error rather than a no-op: a typo
would otherwise leave the default in force and produce a dataset whose manifest
claims a radius the force field never saw.

Both datasets in this repository pin the eight parameters that enter the fitted
surface — `bond_asymptote`, `taper_radius`, `taper_width`, `switch_radius`,
`soft_core`, `exclusion_depth`, `exclude_coulomb`, `gamma` — so a future
change to one of *those* defaults cannot silently invalidate the `.jsonl` files
already on disk.  `switch_width` follows from the pinned `taper_width`, which leaves `ccoul` and `zbl_ccoul` as the
two that are still taken from the defaults and would move a fitted surface if
they changed.  Changing a pinned value still requires re-running `fast-forces refit
--force-constants` for that dataset.

**Two datasets fitted at different parameters cannot share one process.**
`params.activate` refuses the second one, naming the fields that disagree,
because whichever loaded second would score the first's templates on a surface
they were not fitted to. A caller that genuinely wants a different surface says
so with `params.use(...)`, which never refuses and restores on exit.


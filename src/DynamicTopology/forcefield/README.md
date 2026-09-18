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
| `zbl.py`, `acks2.py` | Å | eV | ZBL constants and `ccoul` are stated in these units |
| everything returned to ASE | Å | eV | forces eV/Å, stress eV/Å³ |

Angles are radians.  `shape_decay`, `PHI_B` and the like are dimensionless;
`SCREENING_LENGTH` is Å, `ccoul` is eV·Å, `gamma` is 1/Å.  The global
parameters and their units are tabulated at the bottom of this file.

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
a  = sqrt( k / 2 Dw )              Dw = D + bond_asymptote   if dr > 0
                                   Dw = D                    if dr <= 0
s  = a * max(dr, 0)

E  = Dw [ 1 - exp(-a dr) ]^2  -  D  +  Dw * c * s^3 * exp(-b s)
```

Parameters `D, r0, k, c, b`.  The `-D` offset puts the minimum at `-D`; the
dissociated limit is `bond_asymptote = 1.0 eV` above zero, so the well the
exponential climbs is `D + bond_asymptote` deep while the minimum stays at `-D`.
The join at `dr = 0` is C2 (the curvature there is `2 Dw a^2 = k` whatever `Dw`
is), so no fitted frequency sees the branch.  The shape term is
`O(s^3)`, so `D`, `r0` and the curvature at `dr = 0` are untouched by `c`; it is
clamped off on the compressed branch.  `b` defaults to `shape_decay` for term
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
r_e   = max( r, core_fraction * σ )

u_126 = 4 ε [ (σ/r_e)^12 - (σ/r_e)^6 ]          plus, for r < core_fraction σ,
        + du/dr|_{r_e} * (r - core_fraction σ)   the C1 tangent continuation

u(r)  = g(r) * u_126(r)
```

`g` is `zbl.taper` reflected — same width, turning the term **on** above
`switch_radius`, with `dg/dr = g (1 - g) / switch_width`.  The linear core
continuation bounds what a compressed bond can contribute (≈5700 eV rather than
1e21 eV at 0.024 Å), which is what keeps the exclusion subtraction below
numerically exact.

```
switch_radius = 0.22 nm (2.2 Å)     core_fraction = 0.4
switch_width  = taper_width / 10    exclusion_depth = 3
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

## Exclusions (`exclusions.py`)

The three nonbonded terms above are summed over every pair with no reference to
the bond graph.  Pairs within `exclusion_depth = 3` bonds of each other are then
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

## Global parameters at a glance

These are **not module constants**.  They are fields of
`params.ForceFieldParams`, stated per dataset under `global_params` in the
manifest, and read through `params.active()` at call time.  The defaults below
are what a manifest that omits a key gets.

| parameter | default | unit | read by | changing it invalidates |
| --- | --- | --- | --- | --- |
| `bond_asymptote` | 1.0 | eV | `qforce`, `fit` | every `.jsonl` in the dataset |
| `shape_decay` | 4.0 | — | `qforce`, `fit` | — (fallback `b` only) |
| `taper_radius` | 1.5 | Å | `zbl` | every `.jsonl` in the dataset |
| `taper_width` | 0.12 | Å | `zbl`, and `switch_width` by default | every `.jsonl` in the dataset |
| `switch_radius` | 0.22 | nm | `lj` | every `.jsonl` in the dataset |
| `switch_width` | `taper_width / 10` | nm | `lj` | every `.jsonl` in the dataset |
| `core_fraction` | 0.4 | — | `lj` | every `.jsonl` in the dataset |
| `exclusion_depth` | 3 | bonds | `exclusions` | every `.jsonl` in the dataset |
| `exclude_coulomb` | `true` | — | `exclusions` | every `.jsonl` in the dataset |
| `gamma` | 2.0 | 1/Å | `ewald` | every `.jsonl` in the dataset |
| `accuracy` | 1e-8 | — | `ewald` | — (periodic only; templates are non-periodic) |
| `ccoul` | 14.4 | eV·Å | `acks2` | every `.jsonl` in the dataset |
| `zbl_ccoul` | 14.399645 | eV·Å | `zbl` | every `.jsonl` in the dataset |

`switch_width` is the same physical width as `taper_width` stated in the other
module's unit — 0.12 Å is 0.012 nm — so setting the taper width sets both unless
`switch_width` is given explicitly.  `ccoul` and `zbl_ccoul` are the same
physical constant to different precision; they are two fields because collapsing
them would change one of the two terms for every dataset already fitted.

Units are the code's, not the manifest author's convenience: `taper_radius` is
in Å and `switch_radius` in nm because that is what `zbl` and `lj` respectively
work in.

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

1.0 is the first value at which `rxn_16` — a genuine saddle, the one channel
that was ever a fitting failure rather than a barrierless one — comes out
fittable; past 1.5 nothing further is fittable. It also sets where a bonded
diabat crosses its own fragments':

| asymptote | H2 | HO | H2O |
| --- | --- | --- | --- |
| 0.75 | 2.53 Å | 3.15 Å | 4.05 Å |
| 1.00 | 2.38 Å | 2.93 Å | 3.74 Å |

Both rows are inside `ReactionSet.get_network`'s 4.0 Å bimolecular cutoff, so
the reverse channel is enumerated where the forward one hands over.

**`shape_decay` (4.0, fallback only — `b` is fitted per bond type).**
Refitting the whole pipeline from the q-force baseline at each fixed `b`:

| b | c_max | metathesis | dissociation rms | fastest mode | dt |
| --- | --- | --- | --- | --- | --- |
| 2.0 | 1.31 | 12/13 | 0.700 eV | 4517 cm⁻¹ | 0.492 fs |
| 2.5 | 3.84 | 13/13 | 0.783 | 4402 | 0.505 |
| 3.0 | 7.80 | 13/13 | 1.098 | 4352 | 0.511 |
| 4.0 | 19.33 | 13/13 | 1.710 | 4400 | 0.505 |
| 5.0 | 34.50 | 13/13 | 2.660 | 4402 | 0.505 |
| 6.0 | 52.20 | 12/13 | 3.181 | 4432 | 0.502 |

The timestep is flat across the whole range — the curvature cap absorbs
whatever `b` does — so `b` costs nothing and 4.0 was simply the worst
reachable value for the dissociation curves; hence `fit.dissociation` fits it
instead of fixing it. `c_max(b)` is the monotonicity limit
(`fit.dissociation.shape_bound`), and it is why low `b` isn't free either: at
2.0 it is 1.31 and binds on five of eight bonds.

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

**`switch_radius` (0.22 nm) is deliberately not `taper_radius`.** The tidy
design, `g = 1 - f` at ZBL's own radius, does not survive contact with
q-force's parameters: a Fermi switch decays by one factor of `e` per width
while 12-6 grows as `r**-12`, and at `rc = 1.5 Å` the switch is down to 1.1e-2
where 12-6 is up at 953 eV — a product of **+10.3 eV per O-H bond**. Bare 12-6,
and what survives at 2.2 Å / 0.12 Å:

| pair | bare | switched |
| --- | --- | --- |
| O–H bond, 0.958 Å | 953 eV | 0.031 eV |
| H–H bond, 0.741 Å | 892 eV | 0.005 eV |
| O–O bond, 1.208 Å | 1375 eV | 0.353 eV |
| water's 1-3 H···H, 1.51 Å | 0.138 eV | 0.0004 eV |
| the H-bond O···H, 1.94 Å | 0.146 eV | 0.015 eV |
| O···O contact, 2.6 Å | 0.076 eV | 0.073 eV |
| O···O contact, 2.4 Å | 0.262 eV | 0.220 eV |

The last two rows are the wall this term exists to supply, retained at 97%
and 84%. The gap from ~1.6 to ~2.0 Å is deliberate: neither term acts there
(`zbl.taper` is down to 0.076 at 1.8 Å and 12-6 isn't yet up), and `ACKS2`
alone already binds the water dimer at −0.112 eV through it —
`tests/test_collapse.py` checks that an intermolecular approach stays uphill
through the gap.

**`core_fraction` (0.4).** 0.4·σ is 0.78 Å for an H–H pair and 1.18 Å for
O–O — inside the Morse core of a real bond and far inside any intermolecular
contact — so the linear-tangent continuation never engages on a pair whose
repulsion is doing physical work; the wall there is already 451 eV (H-H) and
1750 eV (O-O).

**`accuracy` (1e-8).** Loosening it is cheap in reciprocal vector count
(`(-log accuracy)^3`) but expensive in the *stress*: `kappa` is derived from
the cell, so a strained cell truncates at a slightly different splitting, and
that residual is a spurious contribution to the finite-difference virial with
no analytic counterpart, amplified by `2 log(1/accuracy)` relative to the
truncation error itself. At 1e-8 it lands around 1e-6 eV, comfortably under
`tests/test_stress.py`'s tolerance; at 1e-6 it would be at the edge of it.

### Why they live in the manifest

Everything in the table but `accuracy` and `shape_decay` sits inside
`E_bonded + E_nonbonded`, which `fit/dissociation.py` solves each template's
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

Both datasets in this repository pin their values explicitly, so a future change
to a default cannot silently invalidate the `.jsonl` files already on disk.
Changing a pinned value still requires re-running `scripts/fit.py
--force-constants` for that dataset.

**Two datasets fitted at different parameters cannot share one process.**
`params.activate` refuses the second one, naming the fields that disagree,
because whichever loaded second would score the first's templates on a surface
they were not fitted to. A caller that genuinely wants a different surface says
so with `params.use(...)`, which never refuses and restores on exit.

---

*Note: `atom` terms carry a `soft_scale` parameter in the shipped `.jsonl`
files that no code currently reads.*

# Density of bulk water at 300 K

One number: the density this force field settles at under 1 bar at 300 K, measured by relaxing a
packed box at fixed volume and then letting the box move. Everything is pinned in `sweep.toml`, and
each run writes its resolved parameters to `output/<run-id>/config.json`, so a result is traceable
to the numbers that produced it.

This is not a sweep. There is no variable being scanned; the three seeds put an error bar on a
single measurement rather than comparing anything. The directory keeps the other sweep's shape —
`sweep.toml`, an integer index per run, `run_one.py` — so that the harness is the same program.

## Design

| | |
|---|---|
| system | 64 H2O, started at 997 kg/m³ (a 12.43 Å cube) |
| stage 1 | **NVT** 5 ps at fixed volume, Langevin, friction 0.01/fs |
| stage 2 | **NPT** 20 ps at 1 bar, Berendsen, τ_p = 1 ps |
| seeds | 3 — 3 runs |
| temperature | 300 K |
| timestep | **0.5 fs** |
| EVB | `eps` 0.05 eV, `max_depth` 2, `max_states` 16, cutoff **3.5 Å** |
| measured | mean density over the last 15 ps, per seed |

**Why NVT first.** `molify.pack` produces a non-overlapping random arrangement, not an equilibrated
liquid: the packed box has 2.17 Å closest O–O contacts against the 2.8 Å of real water, and relaxing
it releases enough energy to spike the temperature to ~1500 K before the thermostat takes it back.
Handing that to a barostat directly would rescale the cell against a pressure that is mostly the
packer's close contacts. Measured: the pressure falls from +267 kbar to a plateau within ~300 fs and
then stops falling, so 5 ps is many relaxation times.

**Why Berendsen.** The measurement is `<V>`, and the Berendsen barostat gives a correct mean volume
with a robust, well-damped approach to it. It does not sample the true isobaric ensemble, so its
volume *fluctuations* are not physical and no compressibility may be read off them. `ase.md.npt.NPT`
(Melchionna/MTK) is the rigorous alternative; it needs an upper-triangular cell and brings its own
Nosé–Hoover thermostat.

**Why the EVB restrictions.** `eps` is the lever. It is the stabilisation a channel must supply to
enter the basis, and the default 1e-3 eV admits very nearly everything; 0.05 eV is 2 kT at 300 K, so
only channels that genuinely mix survive. Raising it stays smooth — `basis.py` ramps a coupling up
from zero across `[eps, eps + switch_width]` rather than switching it on at a threshold — and it
never sets `Block.capped`. `max_states` and `max_depth` are different in kind: they refuse a state
outright, which makes the surface seed-dependent wherever they fire, so they are a ceiling on cost
rather than the mechanism. `ncapped` in the log says whether they ever bit, and it should stay zero.

The 3.5 Å cutoff (against the 4.0 default) still contains water's 2.8 Å O–O first shell and its
1.8 Å hydrogen-bonded O–H, so no hydrogen-bonded pair is dropped. Measured on the 64-water box:
**2.54 → 1.45 s per force call**, a 1.75× speedup, with the total energy unchanged to all printed
digits.

**Why 0.5 fs.** `datasets/Water/README.md` measures the fastest mode on this surface at 4359 cm⁻¹
(the h2o O–H stretch). Velocity Verlet wants ~15 steps per period, so `dt_max = 33356 / (15 × 4359)
= 0.51 fs`. Re-read that number after any refit rather than carrying this one across — it was 4386
before the ZBL taper and the refit that followed it.

## What this needed first

**The calculator had no stress.** `implemented_properties` was `["energy", "forces"]`, and every ASE
barostat calls `atoms.get_stress()`, so an NPT run raised `PropertyNotImplementedError` before
taking a step. The analytic virial was added across all four force-field layers for this directory:

- every `compute_*` in `forcefield/qforce.py` now returns `(energy, forces, virial)`;
- `forcefield/acks2.py` (including the charge response), `forcefield/zbl.py`, and
  `forcefield/coupling.py` likewise;
- `basis.py` carries `virials` / `coupling_virials` per block, including the strain gradient of the
  admission-ramp weight, and `system.py` contracts them with the ground-state eigenvector exactly as
  it does the forces;
- `ase.py` divides by the cell volume and publishes `stress`.

It is cheap because every energy here is a function of minimum-image displacement vectors only, so a
homogeneous strain maps `v → (I + e) v` and `W_ab = Σ v_a (dE/dv)_b` — the same per-pair gradient the
forces are already scattered from. `tests/test_stress.py` differences all of it against the cell with
the atoms scaled affinely, and also asserts the virial is symmetric, which catches a transposed
contraction that a pressure-only check would pass. Verified independently on this box: the analytic
pressure agrees with a numerical `-dE/dV` to seven significant figures.

## Running

```sh
# from the repository root
uv run python production/density-300K/make_inputs.py      # the box, plus a measured cost line
uv run python production/density-300K/run_one.py --list   # the 3 runs
uv run python production/density-300K/run_one.py 0        # one of them

sbatch production/density-300K/submit.slurm               # all 3 as an array
bash production/density-300K/run_all_local.sh 3           # or locally

# smoke test first -- ~8 minutes, exercises both stages and the analysis
uv run python production/density-300K/run_one.py 0 --steps 120
uv run python production/density-300K/density.py production/density-300K/output \
    --equilibration 0
```

An index is the entire interface to a run, which keeps `submit.slurm` the only scheduler-specific
file here. A job that hits the wall continues in place with `--restart`, which resumes whichever
stage was in progress and does not redo a completed NVT stage.

## Cost

Measured by `make_inputs.py` on the development machine: **1.89 s per force call** at 192 atoms, so
50,000 steps is **~26 hours per run** and ~79 core-hours for the three. That is half the 48 h wall,
which is the margin a slower node needs — re-run `make_inputs.py` on the cluster rather than
trusting this number.

The force call is roughly ten times the H2/O2 sweep's at a comparable atom count. The reason is the
hydrogen-bond network: far more pairs sit inside the bimolecular cutoff, so far more reaction
channels are enumerated per step. That is what the `[evb]` block is fighting.

## Reading the result

```sh
uv run python production/density-300K/density.py production/density-300K/output
```

It prints a density per seed with a within-run block error, then the mean and the spread across
seeds — **quote the between-seed spread**, since consecutive frames of one trajectory are not
independent samples of the volume.

Then the checks that say whether to believe it, and each has a specific failure in mind:

| check | what a bad value means |
|---|---|
| `capped` | `max_states`/`max_depth` refused a state; the surface is seed-dependent where that fired |
| unfitted couplings | all three Water channels are `"provenance": "fitted"`, so any name here means the dataset moved |
| mean temperature | should sit near 300 K; a persistent offset is a thermostat or timestep problem |
| smallest cell edge | must stay above twice the 3.5 Å cutoff — minimum image is undefined below that, and a *contracting* box is the way this breaks |
| final species besides H2O | a density is only a density if the thing is still water |

## Files

- `sweep.toml` — every pinned parameter, with the measurement behind each one in its comment.
- `make_inputs.py` — packs the one box and times a force call on it.
- `run_one.py` — index → one two-stage run; `--list`, `--restart`, `--steps`, `--dry-run`.
- `density.py` — the measurement, and the diagnostics that qualify it.
- `submit.slurm` / `run_all_local.sh` — the two ways to run all three.
- `inputs/w64.xyz` — the packed starting box (tracked; it is what every number descends from).

## Status: two changes to the repulsion, and the density crossed 997 from both sides

This directory's history is three force fields, and the number it measures has been badly wrong in
both directions before landing near the right one.

| | equilibrium ρ (kg/m³) | vs. experiment (997) | what was wrong |
|---|---|---|---|
| untapered ZBL | ~250–350 | 3–4× too low | a keV-stopping potential charging +251 kbar at 2–3 Å |
| **+ `zbl.taper`** | ≳1250 | ≥25% too high | the taper removed the wall and nothing replaced it |
| **+ `lj.switch`** | **900–960** | **4–10% too low** | under-attractive: no depth to the hydrogen bond |

The last row is a band rather than a figure because the two independent readings do not quite
agree: the equation-of-state crossing is 898 kg/m³ corrected to 300 K, and a 600 fs NPT run
settles at 953. Both are short and both are noisy; see **The NPT run** below for why neither
deserves three digits.

### 1. The ZBL taper (previous entry, kept for the record)

**This directory's first result was that the force field could not hold liquid water together at
997 kg/m³** — the pressure there was about +160,000 bar and an equation-of-state scan put the zero
crossing near 250–350 kg/m³, a quarter of ambient. That was traced to `forcefield/zbl.py`, which
applied the ZBL screened-nuclear repulsion with no cutoff and no taper. ZBL is fitted for keV
nuclear stopping; evaluated out to 3 Å it was charging **+0.72 eV at a water dimer's hydrogen
bond** — three times the whole bond, the wrong way — and **+251 kbar** across a 64-water box.

`ZBL` now carries a Fermi taper at `TAPER_RADIUS = 1.5 Å`, `TAPER_WIDTH = 0.12 Å`. What that
changed, measured on this directory's own box:

| | before the taper | after |
|---|---|---|
| water dimer at R_OO = 2.91 Å | **+0.598 eV**, unbound, no minimum at any R | **−0.112 eV** |
| ZBL's share of that | +0.719 eV | +0.009 eV |
| intermolecular ZBL, 64 waters | 172 eV | 0.16 eV |
| its contribution to P | +251 kbar | +1 kbar |
| P at 997 kg/m³, relaxed | ~+165,000 bar | **−7,600 bar** |
| equilibrium density | ~250–350 kg/m³ | **≳1250 kg/m³** |

The dimer bound, at nearly the right separation, and the pressure at ambient density fell by a factor
of 20 **and crossed zero**. It overshot: the box no longer wanted to explode, it wanted to collapse.
The cause was named in that entry and is what the next one fixes — the taper removed the
intermolecular wall and there was nothing behind it, because the O–O Lennard-Jones core every water
model carries had been retired with `forcefield/lj.py`, and the model had no dispersion at all.

### 2. The 12-6, switched on where ZBL switches off

`forcefield/lj.py` is back. It was retired four times over, always for the same reason — q-force's
12-6 is 500–1400 eV at a bond length, so a diabatic state that has *broken* a bond pays hundreds of
eV for a pair that is merely close, and every attempt to strip that penalty made the repulsion
differ between diabatic states. `lj.switch` removes the root instead of the symptoms: a Fermi
function at `SWITCH_RADIUS = 0.22 nm` takes the term to 0.03–0.35 eV at a bond length, three to four
orders of magnitude down, at which size **nothing has to be excluded at all** — so the term is
identical on every diabat, adds a common shift to every EVB diagonal, and sits outside the
Hamiltonian exactly where `ZBL` and `ACKS2` already do. `DYNAMICTOPOLOGY.md` §2.2.1 has the
argument; `forcefield/lj.py` has the measurements behind the radius, including why the tidy choice of
making the two switches complementary puts **+20.6 eV on a single water molecule**.

Both datasets were refit against it.

**The water dimer keeps its minimum and gains a wall.**

| R_OO (Å) | 2.40 | 2.60 | 2.75 | **2.91** | 3.10 | 3.30 | 3.60 |
|---|---|---|---|---|---|---|---|
| total (eV) | **+0.776** | +0.069 | −0.078 | **−0.100** | −0.090 | −0.080 | −0.062 |
| ACKS2 | −0.278 | −0.197 | −0.155 | −0.121 | −0.092 | −0.071 | −0.050 |
| ZBL | +0.833 | +0.191 | +0.046 | +0.009 | +0.001 | 0.000 | 0.000 |
| 12-6 | +0.221 | +0.076 | +0.031 | +0.012 | +0.001 | −0.009 | −0.012 |

The minimum stays at 2.91 Å and loses 0.012 eV of its 0.112 — 11%, and the reference is −0.218, so
the binding was already the weak part. What is new is the first column: two waters at 2.40 Å used to
be free to keep closing and now pay 0.78 eV. Across the whole box the 12-6 supplies **7.78 eV of
intermolecular repulsion** where tapered ZBL supplies 2.02, so it is essentially the entire wall
above 2 Å. Its intramolecular part is 3.76 eV over 64 molecules — 0.059 eV each, absorbed by the
refit.

### The new equation of state

Same method as both previous entries — from the relaxed configuration, rescale, 250 Langevin steps
at a 300 K setpoint, average the last 150:

| target ρ (kg/m³) | edge (Å) | P (bar) | sd | T (K) reached | P at 300 K (bar) |
|---|---|---|---|---|---|
| 800 | 13.376 | −2,330 | 8,696 | 472 | −4,235 |
| 850 | 13.108 | −539 | 4,226 | 380 | −1,479 |
| **900** | 12.861 | **+912** | 4,979 | 368 | **+60** |
| 950 | 12.631 | +3,838 | 3,117 | 357 | +3,094 |
| 997 | 12.430 | +6,723 | 4,402 | 358 | +5,927 |
| 1100 | 12.029 | +14,378 | 6,028 | 473 | +12,024 |

The raw crossing is **869 kg/m³**; the last column subtracts the excess ideal-gas term (`NkT/V` at
the temperature reached, less the same at 300 K) and moves it to **898 kg/m³**, which is the number
to quote.

**Read the temperature column before the pressure.** The scan heats as it compresses and the
thermostat does not catch up in 250 steps, so the raw column is not a 300 K pressure. The sd column
says the rest: this pressure is a near-cancellation of large terms and a single frame is worth
several kbar, so the crossing is good to a few tens of kg/m³ and no better.

Against the experimental 997 that is about **10% low**.

Either way the sign of the error has flipped again, and it is now the smallest it has been: the model
is slightly **under**-dense, which is what an under-attractive surface with a sound wall should give.
The dimer binds at −0.100 eV against −0.218, and `ACKS2` supplies −0.121 of that on its own — so what
is missing is depth in the hydrogen bond, not repulsion. That is the next thing to fix if this number
matters, and it is a smaller and better-posed problem than the one before it.

### What the refit cost: nothing in channels, one frequency

**HCombustion stayed at 13 of 19 fittable channels, losing exactly the same six.** Most surviving
margins improved — `rxn_12`'s pre-fit margin went from −0.71 to +0.02 eV — which is the mirror image
of what the taper did, and for the mirror-image reason: the taper *removed* repulsion at
transition-state geometries, the 12-6 *adds* it back at exactly the intermolecular contacts a
transition state is made of.

The 13 is measured against a baseline produced by re-running the *same* pipeline over the *same*
`.jsonl` with the term disabled (`lj.SWITCH_RADIUS = 1e6` makes it identically zero), because
`fit.py` is not idempotent and a before/after quoted across two different input states measures
nothing. The taper entry above records 12 for the previous pass; both are right for their own
comparison.

What it did cost is a frequency, on the one mode the fit does not measure exactly. **H₂O₂'s O–O
stretch went 3228 → 4285 cm⁻¹** — the 12-6 is 0.30 eV at that 1.45 Å bond and steeply varying, so the
fit stiffened it — making it the second-fastest mode in the dataset and the one bond type
`fit.dissociation.stretch_curvatures` is inexact on. The cap is enforced through that stand-in, which
reads 86 cm⁻¹ (2.0%) low. `sweep.toml`'s 0.5 fs covers both readings. Water lost nothing: 3 of 3
channels, 4359 cm⁻¹, still under the 4400 cap.

`tests/geometry.py` moved in the good direction for once: `SWITCHING_PATH_START` went from 0.2 to
0.1 because **recrossing came back**. On the tapered surface no channel recrossed at all; `rxn_12`
at 0.1 now switches 3, 3 and 1 times on three seeds.

### The reactive machinery is dormant here

Every block in this box logs `max_nstates = 1` — **no diabatic state is ever admitted** — at the
production `eps` of 0.05 eV *and* at the 1e-3 default, so the `[evb]` block is not gating anything
out. `h2o-autoionization`'s fitted width is `a = 611 Å⁻²` and the coupling is `exp(-a·rmsd²)`, so it
switches on only within ~0.04 Å RMSD of its stored transition state; neutral water at 300 K never
gets there. That is physically right — autoionization is rare — but it means **this density is a
fixed-topology measurement**, and the reactive machinery it is nominally testing is idle. The hop
channels need an ion present to do anything at all. Unchanged by either refit.

### The NPT run

600 fs of `NPTBerendsen` at 1 bar / 300 K from the NVT-equilibrated 997 kg/m³ configuration
(`output/s1/npt.jsonl`, 1200 steps of 0.5 fs, τ_p = 1 ps, τ_T = 100 fs):

| t (fs) | 0 | 60 | 120 | 180 | 240 | 300 | 360 | 420 | 480 | 540 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ρ (kg/m³) | 997.0 | 966.1 | 960.4 | 957.1 | 955.2 | 953.4 | 950.1 | 951.1 | 952.5 | 953.2 | 956.7 |
| edge (Å) | 12.430 | 12.561 | 12.586 | 12.600 | 12.608 | 12.616 | 12.631 | 12.626 | 12.620 | 12.617 | 12.602 |
| P (bar) | +26,401 | +1,998 | +3,169 | +4,135 | −840 | +2,341 | −1,948 | +444 | −786 | +727 | −3,291 |
| T (K) | 283 | 365 | 366 | 337 | 359 | 299 | 335 | 343 | 330 | 307 | 268 |

**The box expands and then stops, and the pressure changes sign.** Over the last 20 frames
(400–600 fs) ρ = **952.9 ± 1.5 kg/m³** and P = **−865 bar** with a per-frame sd of 2,040, i.e. a mean
consistent with zero at this sample length. The density turns around at 400 fs — 949.5 kg/m³, the
minimum — and drifts back up, which is what a barostat oscillating about a crossing does, not what a
run that is still relaxing does.

This is the comparison that matters. The same stage on the previous surface contracted
**monotonically past 1188 kg/m³ and was still falling** when it was stopped; there was no crossing to
oscillate about, because above 2 Å there was no wall. The direction of the error has reversed and its
size has collapsed.

Three cautions on the number itself, in decreasing order of size:

- **600 fs is shorter than one barostat time constant.** τ_p is 1 ps, so 952.9 is where the box goes
  in the first six tenths of a relaxation time, not an equilibrated ⟨V⟩. The production configuration
  runs 20 ps for this reason. Read it as "the box holds near ambient density", not as three figures.
- **It is a 314 K reading, not a 300 K one.** T over the same window means 314 with τ_T = 100 fs — the
  expansion does work and the thermostat is chasing it. That biases the density low relative to 300 K,
  by roughly the ideal-gas correction the equation-of-state table applies.
- **The instantaneous pressure is worth kbar per frame.** The +26,401 bar at t = 0 is one frame of a
  distribution whose sd over the whole run is 4,100; the run-mean of a near-cancellation of two very
  large terms is the only pressure here that means anything.

It sits above the equation-of-state crossing (898 kg/m³ corrected to 300 K) rather than on it. Given
that the scan locates its crossing to a few tens of kg/m³ and this run is hotter, shorter than τ_p,
and averaged over 200 fs, the two are not in conflict — but the honest statement is that this force
field puts liquid water somewhere in the **900–960 kg/m³** band, a few percent under experiment,
rather than at any single figure.

The reactive diagnostics are clean throughout: `max_nstates` is 1 in every frame, `ncapped` is 0,
`min_switch` never leaves 1.0, and no `placeholder_channels` appear.

**The NPT stage started from an NVT trajectory equilibrated on the previous surface**, which is a
feature rather than an oversight: both entries' NPT runs leave from the same configuration, so the
traces above and the contracting one are directly comparable. It also gives the refit absorption for
free. Evaluating that one configuration on both surfaces, `energy_nonbonded` and `energy_zbl` agree to
seven figures (nothing else moved), the 12-6 adds **+11.543 eV**, and `energy_bonded` drops by
**−3.778 eV** — which is the intramolecular 12-6, 3.76 eV over 64 molecules, absorbed into the refit
Morse depths to within 0.5%. That is the claim in the section above, measured directly rather than
inferred.

### One thing that did not break

The cell **grows** under the barostat, and minimum image only fails when a cell shrinks toward twice
the interaction range. Nothing here goes that way, and the check is not close either direction: the
floor is twice the 3.5 Å cutoff, 7.0 Å, and the box would have to reach ~5600 kg/m³ to reach it
against a run that never leaves 12.4–12.7 Å. `density.py` checks `min(edge)` against that floor
explicitly and says so. It was the previous surface, contracting past 1188 kg/m³, that made this
worth checking at all.


## Limits of the instrument

These bound the result and belong here rather than being rediscovered later.

1. **The hydrogen bond is about half as deep as it should be.** The dimer binds at −0.100 eV against
   a CCSD(T) reference of −0.218, and `ACKS2` supplies −0.121 of that unaided; the 12-6's dispersion
   tail adds nothing at 2.91 Å — it is **+0.012 eV** there, still repulsive, and only turns attractive
   beyond ~3.2 Å where the dimer curve is already flat. This is now the largest systematic error in
   the number and it has a sign: the density comes out **too low**. It replaces the previous item 1 — "no repulsive core at van der Waals
   range, and no dispersion at all" — which `forcefield/lj.py` fixed, and which had the opposite
   sign.
   There is still a gap from ~1.6 to ~2.0 Å where neither repulsion acts and `ACKS2` carries the
   contact alone; see `lj.SWITCH_RADIUS`. `tests/test_collapse.py` checks nothing squeezes through
   it, but it is a real seam and it is where the hydrogen bond lives.
2. **Electrostatics are bare minimum-image, not Ewald.** ACKS2 sums all pairs under the minimum-image
   convention with no cutoff and no lattice sum. In a ~12 Å box that is a real approximation to the
   Coulomb energy of a polar liquid, and the density inherits it.
3. **The Berendsen barostat's mean is the measurement**, not its fluctuations.
4. **The `[evb]` settings are not what makes this number.** They were pinned on the expectation that
   they select which channels enter the basis; measured, no channel enters at *any* `eps`, so this
   is a fixed-topology density and the block is a cost control rather than a physical choice. It
   still lands in every `config.json`, because that stops being true the moment an ion is present.
   Re-checked after the 12-6 refit: still `max_nstates = 1` everywhere.
5. **20 ps against a 1–10 ps volume relaxation time** is a few relaxation times, so a
   several-percent statistical error is expected; the seeds are what turn that into a stated
   uncertainty rather than a false precision.
6. `datasets/Water/README.md` records that `h3o`'s terms are set **by analogy to `h2o`** and are the
   weakest link in the set, and that the two degenerate hop channels are each stored twice
   (energetically inert, ~5% redundant network edges).

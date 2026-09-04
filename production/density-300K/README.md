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

**Why 0.5 fs.** `datasets/Water/README.md` measures the fastest mode on this surface at 4386 cm⁻¹
(the h2o O–H stretch). Velocity Verlet wants ~15 steps per period, so `dt_max = 33356 / (15 × 4386)
= 0.51 fs`. Re-read that number after any refit rather than carrying this one across.

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

## Status: submittable, and expect a number far from experiment

The harness runs end to end and the analysis reads it. **What the smoke runs and an equation-of-state
scan already show is that this force field does not hold liquid water together at 997 kg/m³**, and
anyone reading the eventual result needs that up front rather than as a surprise.

At the ambient starting density the pressure is about **+160,000 bar**. That is not a packing
artifact and it is not a bug in the new virial: it survives 5 ps of relaxation (falling from +267 to
a ~+165 kbar plateau within 300 fs and then stopping), and the analytic pressure agrees with a
numerical `-dE/dV` on the same box to seven significant figures.

A scan across densities, each from the same relaxed configuration and each given 250 steps of
Langevin at a 300 K setpoint:

| target ρ (kg/m³) | edge (Å) | P (bar) | T (K) reached |
|---|---|---|---|
| 997 | 12.43 | +159,550 | 329 |
| 700 | 13.99 | +57,524 | 456 |
| 500 | 15.65 | +19,756 | 629 |
| 350 | 17.62 | +4,530 | 825 |
| 250 | 19.71 | −474 | 1024 |
| 150 | 23.37 | −2,231 | 1345 |

The pressure crosses zero somewhere around **250–350 kg/m³**, roughly a quarter to a third of
ambient. So the NPT stage will expand the box by a factor of three or four in volume, and the density
this sweep measures will be of that order.

**Read the last column before trusting the third.** Each point releases potential energy as it
expands and the thermostat has not caught up within 250 steps, so the low-density rows are hot and
their pressures are not 300 K pressures. The scan locates the crossing to within a factor of
something, not to a figure — the production runs are what measure it. What the scan does establish
firmly is the sign and the order of magnitude at ambient density, and those are enough to know what
the answer will look like.

Two consequences for the runs as configured, both already handled:

- The cell **grows**, and minimum image only fails when a cell *shrinks* toward twice the
  interaction range. At 250 kg/m³ the edge is ~19.7 Å against the 7.0 Å floor. `density.py` checks
  this explicitly and says so.
- 20 ps is long enough for the expansion itself — the smoke run moves 997 → 728 kg/m³ in the first
  60 fs — so the 5 ps discarded as equilibration is many box-relaxation times.

### Where the pressure comes from

Decomposed on the packed box: ZBL alone contributes **+877 kbar** and everything else about
−610 kbar, netting the +267 kbar seen before relaxation. The result is a near-cancellation of two
very large numbers, which is why the instantaneous pressure is so noisy and why it must be averaged.

Most of the ZBL term is intramolecular — it is applied to bonded pairs too, by design, and those
values are absorbed into the fitted Morse depths (`forcefield/zbl.py` documents this). That absorption
is exact for the *energy* at the template geometry and only approximate for the *virial*, because the
Water templates use literature geometries rather than ones optimised against this force field: an
isolated H2O in a 30 Å box already carries +21 bar of internal stress, which scales to ~19 kbar for
64 molecules at ambient volume. That accounts for about a tenth of the excess. The remaining ~145 kbar
is genuine intermolecular repulsion, which is the thing to investigate if this density matters.

None of that is a defect in this directory's work — it is what the instrument reads, now that there
is an instrument.

## Limits of the instrument

These bound the result and belong here rather than being rediscovered later.

1. **Electrostatics are bare minimum-image, not Ewald.** ACKS2 sums all pairs under the minimum-image
   convention with no cutoff and no lattice sum. In a ~12 Å box that is a real approximation to the
   Coulomb energy of a polar liquid, and the density inherits it. This is the largest systematic
   error in the number.
2. **The Berendsen barostat's mean is the measurement**, not its fluctuations.
3. **The density is a property of the pinned `[evb]` settings.** Raising `eps` from 1e-3 to 0.05 eV
   changes the potential energy surface. That is why they land in every `config.json`.
4. **20 ps against a 1–10 ps volume relaxation time** is a few relaxation times, so a
   several-percent statistical error is expected; the seeds are what turn that into a stated
   uncertainty rather than a false precision.
5. `datasets/Water/README.md` records that `h3o`'s terms are set **by analogy to `h2o`** and are the
   weakest link in the set, and that the two degenerate hop channels are each stored twice
   (energetically inert, ~5% redundant network edges).

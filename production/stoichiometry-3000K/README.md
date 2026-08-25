# H2/O2 stoichiometry sweep at 3000 K

One variable: the H2:O2 ratio. Everything else is pinned in `sweep.toml`, and each run writes its
resolved parameters to `output/<run-id>/config.json`, so a result is traceable to the numbers that
produced it.

## Design

| | |
|---|---|
| compositions | H2:O2 = 1:2, 1:1, **2:1 (stoichiometric)**, 4:1, 8:1 |
| seeds | 5 per composition — 25 runs |
| temperature | 3000 K, Langevin, friction 0.01/fs |
| box | **22.44 Å fixed** — constant *number* density, ~200 atoms throughout |
| timestep | 0.25 fs |
| length | 100 ps (400,000 steps), a frame every 25 fs |

**Why the box is fixed rather than the density.** O2 is sixteen times heavier than H2, so holding
mass density at the usual 250 kg/m³ while sweeping composition would run the box from 24.4 Å at
1:2 down to 15.2 Å at 8:1 — a four-fold range in volume. Collision rates would then move with
composition and confound the only variable being swept. Holding the volume fixed lets the mass
density vary instead (320 → 78 kg/m³), which is the honest trade: one of the two has to move, and
concentration is the one that would otherwise masquerade as chemistry.

**Why 0.25 fs.** The refit in `fit.dissociation` took the H–H stretch to 5282 cm⁻¹, a 6.3 fs
period. The script's 0.5 fs default is ~13 steps per period, which is acceptable at 2000 K and not
worth relying on at 3000 K.

## Running

```sh
# from the repository root
uv run python production/stoichiometry-3000K/make_inputs.py    # 5 boxes, all 22.440 A
uv run python production/stoichiometry-3000K/run_one.py --list # the 25 configs
uv run python production/stoichiometry-3000K/run_one.py 0      # one of them

sbatch production/stoichiometry-3000K/submit.slurm             # all 25 as an array
bash production/stoichiometry-3000K/run_all_local.sh 8         # or locally, 8 at a time
```

An index is the entire interface to a run, which is what keeps `submit.slurm` the only
scheduler-specific file here. Indices are ratio-major, so a partially completed array leaves whole
compositions finished rather than one seed of each.

A job that hits the wall clock continues in place — `run_one.py <index> --restart` resumes from its
own last frame, which now carries the bonding the run had reached rather than resetting it.

## Cost

~290 ms per force call at 200 atoms, so 400,000 steps is **~32 h per run** and ~800 core-hours for
the sweep. The 25 runs are independent and single-threaded; parallelism comes from the array.

## Reading the result

```sh
uv run python scripts/analyze.py production/stoichiometry-3000K/output
uv run python scripts/plot.py -i production/stoichiometry-3000K/output/r2_1_s1/traj.xyz
```

`analyze.py` prints, per composition: product counts, peak H2O, time to first product, and two
diagnostics that decide whether the rest is trustworthy — how often the basis was **capped** (where
it fires, the surface is seed-dependent) and which channels entered a basis on an **unfitted**
coupling. Only `rxn_06` (O2 → O + O) should ever appear in the latter; it carries a hand-set
amplitude because it is the one channel of nineteen the fit could not invert. Anything else there
means the dataset moved.

## Files

- `sweep.toml` — the parameters; the single source of truth for both scripts below
- `make_inputs.py` — one packed box per composition, into `inputs/`
- `run_one.py` — index → one run, into `output/<run-id>/`
- `submit.slurm` — SLURM array over the indices
- `run_all_local.sh` — the same, on one machine

## Status: built and verified, NOT yet submitted

A probe at 3000 K found a blocker that is **not** in this directory and not in the sweep design.
`E_nonbonded` runs away monotonically (+0.60 → −66.95 eV in 0.1 ps on the 2:1 box) while the bonded
energy stays flat and no species ever changes. The cause is in `forcefield/acks2.py`: the
bond-softness block is built over *all* atom pairs with a 2.7–4.4 Å decay rather than being
restricted to bonded pairs, and only *total* charge is constrained, so charge flows freely between
molecules that share no bond. After 0.1 ps, 44 of 99 neutral H2/O2 molecules carry net charge, one
of them 1.29 e.

It is pre-existing — re-scoring the old pre-refit trajectory shows the same runaway (−124.65 eV by
0.29 ps) — but it means these 25 runs would measure an electrostatics artifact rather than
combustion. Fix the charge constraint first; everything here is ready to run unchanged once that
lands.

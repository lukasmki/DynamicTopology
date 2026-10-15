# Stoichiometric H2/O2 density sweep at 4000 K

One variable: the mass density. Everything else is pinned in `sweep.toml`, and each run writes its
resolved parameters to `output/<run-id>/config.json`, so a result is traceable to the numbers that
produced it.

## Design

| | |
|---|---|
| composition | H2:O2 = 2:1 (stoichiometric), 66 H2 + 33 O2 = 198 atoms |
| densities | 30, 55, 100, **175**, 250 kg/m³ — boxes 40.4, 33.0, 27.0, 22.4, 19.9 Å |
| seeds | 5 per density (1–5) — 25 runs |
| temperature | **4000 K**, Langevin, friction 0.005/fs |
| timestep | 0.5 fs |
| length | 100 ps (200,000 steps), a frame every 25 fs |

**Why the molecule count is fixed rather than the box.** The box edge carries the density, so every
run has the same 198 atoms and the same per-molecule chemistry to count. The other way, a fixed box
holding 10–80 molecules, would put the dilute end's statistics on a handful of molecules.

**Why roughly geometric spacing.** Bimolecular collision rates scale with density, so a reaction
that is collision-limited moves by the same factor between each pair of points. Even spacing in
ρ would put four of five points in the dense half.

**Why 4000 K.** The top of the 3000–4000 K range, and the temperature
`../stoichiometry-3000K/sweep.toml` actually ran at (the directory name predates the change). At
175 kg/m³ those 2:1 runs formed their first H₂O at 8–45 ps; collision rates fall ~6× from there to
30 kg/m³, so the dilute end is already marginal for ignition inside 100 ps at this temperature and
would see almost nothing at 3000 K.

**175 kg/m³ is the shared state point.** The stoichiometry sweep's 2:1 box is 174.7 kg/m³ (99
molecules in 22.44 Å), and `../temperature-hc` runs its whole ladder at 175. At 4000 K, all three
sweeps sample the same conditions. `../temperature-hc`'s box is byte-identical to `inputs/d175.xyz`
and uses seeds 6–10, which gives ten independent replicates at that point across the two
directories. The stoichiometry sweep's 2:1 box was packed at a 0.012 Å different edge, so it is a
different packing and contributes five more.

## Running

```sh
# from the repository root
uv run python production/density-hc/make_inputs.py    # 5 boxes, one per density
uv run python production/density-hc/run_one.py --list # the 25 configs
uv run python production/density-hc/run_one.py 0      # one of them

sbatch production/density-hc/submit.slurm             # all 25 as an array
bash production/density-hc/run_all_local.sh 8         # or locally, 8 at a time
```

Indices are density-major, so a partially completed array leaves whole densities finished.

`steps` in `sweep.toml` is a run's **total** length. `run_one.py <i> --restart` integrates only the
steps the run has left (counted from its own `log.jsonl`), so resubmitting a job the wall cut short
is safe and resubmitting a finished one does nothing. To extend a run, restart it with a larger
target: `run_one.py <i> --restart --steps 400000` takes it to 200 ps.

**This differs from `../stoichiometry-3000K/run_one.py`**, where `--restart` integrates another full
`steps` on top of what is there. That is how its logs reached 500 ps.

## Cost

Measured on a Perlmutter Milan node (one core per run, as `submit.slurm` requests), from the
packed boxes:

| density | s/step | 100 ps | basis capped |
|---|---|---|---|
| 30 kg/m³ | 0.70 (10 min) | ~39 h | never |
| 175 kg/m³ | 0.75 (10 min) | ~42 h | never |
| 250 kg/m³ | **1.4 and 2.7** (two seeds, 20 min) | **80–150 h** | **4 of 29 frames** in 0.4 ps |

The first ~0.4 ps from a packed box is probably the expensive end: the stoichiometry sweep's 100 ps
production jobs took 17.5–29.5 h, i.e. 0.32–0.53 s/step at 175 kg/m³. 175 kg/m³ and below should
fit the 48 h wall, and any that does not resumes cleanly with `--restart`.

**250 kg/m³ does not fit, and the cost is not the only issue there.** Its network percolates: the
independent EVB blocks fall from 32 to 6–9 within 0.2 ps, and nearly all of a force call goes into
`basis.py:build` screening the extra channels that come with them (0.94 of 1.44 s). Several of those
blocks then hit the 128-state cap. A capped basis makes the surface seed-dependent where it fires,
and 4 of 29 frames is far above the 1.4–2.7% of frames the stoichiometry sweep capped over 100 ps.
The top of the ladder needs a decision before it is submitted (raise `max_states`, tighten the EVB
admission as `../density-300K` does, or stop the ladder lower), and every option changes either the
cost or the surface relative to the other two sweeps.

## Reading the result

```sh
uv run python scripts/analyze.py production/density-hc/output --group-by density
```

Same columns as for the stoichiometry sweep: product counts, peak H₂O, time to first product, how
often the basis was **capped**, and which channels entered on an **unfitted** coupling. A rate
compared across densities should be normalized per molecule (all runs have 99) or per volume, not
read off raw counts against time.

## Files

- `sweep.toml` — the parameters; the single source of truth for both scripts below
- `make_inputs.py` — one packed box per density, into `inputs/`
- `run_one.py` — index → one run, into `output/<run-id>/`
- `submit.slurm` — SLURM array over the indices
- `run_all_local.sh` — the same, on one machine

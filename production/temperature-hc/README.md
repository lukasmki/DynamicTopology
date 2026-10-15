# Stoichiometric H2/O2 temperature sweep at 175 kg/m³

One variable: the thermostat temperature. Everything else is pinned in `sweep.toml`, and each run
writes its resolved parameters to `output/<run-id>/config.json`, so a result is traceable to the
numbers that produced it.

## Design

| | |
|---|---|
| composition | H2:O2 = 2:1 (stoichiometric), 66 H2 + 33 O2 = 198 atoms |
| density | **175 kg/m³**, a 22.43 Å box, one packing shared by every run |
| temperatures | 3000, 3250, 3500, 3750, **4000** K, Langevin, friction 0.005/fs |
| seeds | 5 per temperature (**6–10**) — 25 runs |
| timestep | 0.5 fs |
| length | 100 ps (200,000 steps), a frame every 25 fs |

**Why one box.** Every run starts from the same packing, so runs differ only in thermostat setpoint,
initial velocities and Langevin noise. The temperature comparison then isn't also comparing
different arrangements of the same molecules.

**Why even spacing.** Over a 1.33× range in T, the Arrhenius axis 1/T is linear in T to within a
few percent, so an even ladder in T is already an even one in 1/T.

**Why 175 kg/m³.** It is the state point the other two sweeps share. The stoichiometry sweep's 2:1
box is 174.7 kg/m³ (99 molecules in 22.44 Å) and ran at 4000 K, and `../density-hc` runs 175 kg/m³
at 4000 K. This ladder's top rung therefore lands on both of them.

**Why seeds 6–10.** `inputs/2_1_d175.xyz` is byte-identical to `../density-hc/inputs/d175.xyz`:
same composition, same density, same packing seed. With seeds 1–5, the 4000 K column here would
re-run `density-hc`'s 175 kg/m³ column step for step. Disjoint seeds make it five more independent
replicates of the shared point instead.

**Read the temperature off the log, not the setpoint.** At 0.5 fs the Langevin thermostat runs
cool. The bias was 49 K before the last refit, and `../stoichiometry-3000K/sweep.toml` leaves it as
an open item not re-measured since. It grows with temperature, so it matters most at the 4000 K end.
For an Arrhenius fit, use each run's mean `temperature_K`.

## Running

```sh
# from the repository root
uv run python production/temperature-hc/make_inputs.py    # the one box
uv run python production/temperature-hc/run_one.py --list # the 25 configs
uv run python production/temperature-hc/run_one.py 0      # one of them

sbatch production/temperature-hc/submit.slurm             # all 25 as an array
bash production/temperature-hc/run_all_local.sh 8         # or locally, 8 at a time
```

Indices are temperature-major, so a partially completed array leaves whole temperatures finished.

`steps` in `sweep.toml` is a run's **total** length. `run_one.py <i> --restart` integrates only the
steps the run has left (counted from its own `log.jsonl`), so resubmitting a job the wall cut short
is safe and resubmitting a finished one does nothing. To extend a run, restart it with a larger
target: `run_one.py <i> --restart --steps 400000` takes it to 200 ps. **This differs from
`../stoichiometry-3000K/run_one.py`**, where `--restart` integrates another full `steps`.

## Cost

Measured on a Perlmutter Milan node (one core per run, as `submit.slurm` requests), from the
packed box at 3000 K: **0.80 s/step** over 10 minutes, ~44 h per 100 ps, with the basis never
capped. The box is the same at every temperature, and 4000 K on it cost the same over a 400-step
probe.

The first ~0.4 ps from a packed box is probably the expensive end: the stoichiometry sweep's 100 ps
production jobs on a 174.7 kg/m³ box took 17.5–29.5 h, i.e. 0.32–0.53 s/step. Either way a run
should fit the 48 h wall, and one that does not resumes cleanly with `--restart`. The sweep is about
25 × 25–44 h ≈ 600–1100 core-hours.

## Reading the result

```sh
uv run python scripts/analyze.py production/temperature-hc/output --group-by temperature
```

**The cold end may not ignite inside 100 ps.** At 4000 K and this density the first H₂O appears at
8–45 ps. Nothing has been measured at 3000 K beyond a 2 ps probe that did not react, and an
activated onset can stretch by much more than the 1.33× change in T. If `analyze.py` shows no
products at 3000 K, extend those runs with `--restart --steps 400000` rather than reading "no
reaction" as a result.

## Files

- `sweep.toml` — the parameters; the single source of truth for both scripts below
- `make_inputs.py` — the one packed box, into `inputs/`
- `run_one.py` — index → one run, into `output/<run-id>/`
- `submit.slurm` — SLURM array over the indices
- `run_all_local.sh` — the same, on one machine

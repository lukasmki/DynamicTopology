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
| timestep | **0.5 fs** |
| length | 100 ps (200,000 steps), a frame every 25 fs |

**Why the box is fixed rather than the density.** O2 is sixteen times heavier than H2, so holding
mass density at the usual 250 kg/m³ while sweeping composition would run the box from 24.4 Å at
1:2 down to 15.2 Å at 8:1 — a four-fold range in volume. Collision rates would then move with
composition and confound the only variable being swept. Holding the volume fixed lets the mass
density vary instead (320 → 78 kg/m³), which is the honest trade: one of the two has to move, and
concentration is the one that would otherwise masquerade as chemistry.

**Why 0.5 fs.** The timestep is set by the fastest vibrational mode and by nothing else — velocity
Verlet wants ~15 steps per period. That mode was 11697 cm⁻¹ (a 2.84 fs period) because nothing in the
force-constant fit priced curvature; it is now 4399 cm⁻¹, and 0.5 fs follows. Measured, not reasoned
about: see `sweep.toml` for the NVE ladder, the thermostatted comparison against 0.25 fs, and what to
re-measure after a refit.

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

~202 ms per force call at 200 atoms, so 200,000 steps is **~11 hours per run** and ~280 core-hours for
the sweep. The 25 runs are independent and single-threaded; parallelism comes from the array, and a
run now fits inside the 48 h wall on its own.

This used to be ~10 days per run and ~250 days across the sweep. The entire difference is the
timestep, and the entire reason the timestep could move is that the force field's stretching
frequencies came down by a factor of 2.7 — see below.

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

## Status: submittable

Every blocker previously recorded here is resolved. What follows is the history, because each fix
was found by the previous one and the sequence is the argument.

### 1. There was no repulsion in the force field

A probe at 3000 K found `E_nonbonded` running away monotonically (+0.60 → −66.95 eV in 0.1 ps on the
2:1 box) while the bonded energy stayed flat and no species ever changed.

**The cause was first attributed to `forcefield/acks2.py` — that was wrong.** ACKS2 was being handed
geometries no charge model can describe. There was no Pauli repulsion anywhere in the force field, so
nothing opposed two molecules occupying the same space: on `examples/nvt-n100-d250.xyz`, atoms in
*different* molecules reached **0.044 Å**, 225 of the 227 sub-1.5 Å pairs involving a hydrogen —
exactly the atom q-force assigns a zero Lennard-Jones radius. q-force had been emitting those
parameters all along and `io/xml.py` was discarding them, because a `<Particle>` carries no atom
indices and every per-particle force parsed to an empty `atoms` dict.

**`forcefield/lj.py` supplied that term for four design iterations and never fixed it.** q-force's
12-6 is 100–1000× too strong where reactive chemistry lives — 504 eV at the H2 bond length, 930 eV at
O–H, 1348 eV at O–O — so a dissociated diabat had to have its wall *removed*, and every rule written
to remove it (the union rule, the lost-exclusion rule, the reference topology, the coupling gate) took
away either too much or too little. The box fused at 0.60 Å under one of them.

`forcefield/zbl.py` supplies it now: the ZBL screened-nuclear repulsion over every pair, no
exclusions, no cutoff, and **no topology**. Being identical across every diabatic state, it adds the
same constant to every EVB diagonal, which `np.linalg.eigh` removes from the eigenvectors exactly —
so it cannot produce a plateau, a spurious coupling amplitude, or a discontinuity at `bimol_cutoff`.

### 2. Adding it broke the geometries, because nothing constrained the gradient

ZBL is a real repulsion at bonding distances — 2.0 eV at the H2 bond length, 11.6 at O–O — and the
dissociation fit had no `r0` degree of freedom to absorb it. It matched *energies* at each template's
fixed QM geometry (to 1e-13) while nothing at all constrained the *gradient* there, so every template
relaxed somewhere else: HO2's O–O opened by 0.825 Å, and the strain released as heat was the whole
3000 K temperature excess (a packed box reached 5693 K in NVE with its total energy flat).

`fit.dissociation.fit_bond_lengths` is the missing condition — one equation per bond type, the total
force along it vanishing at the reference geometry, solved rather than fitted. Every bond now relaxes
to within 0.009 Å of its reference.

### 3. Fixing that broke the frequencies, and the frequencies were the timestep

The Morse has to lean into the repulsion for the sum to be flat, which pulls `r0` 0.04–0.22 Å inside
the reference bond length — and away from `r0` the Hulburt–Hirschfelder shape term `c` is not the
free parameter three files claimed it was. Its curvature at that displacement was the largest single
contribution to the stiffness of most bonds: 63 eV/Å² of H2's 115, 509 of O2's 793, 323 of HO's 480.

Nothing priced it. `fit_force_constants` bounded the `k`-scale at 1.41× in wavenumbers and reported
that as the frequency cost, while the surface it produced carried an **11697 cm⁻¹** stretch against
an experimental 3738. That is a 2.84 fs period, and it is the entire reason this sweep was configured
at 0.05 fs.

`fit.dissociation.DEFAULT_MAX_WAVENUMBER` prices the total curvature directly. The fastest mode is
now 4399 cm⁻¹ and the sweep runs at 0.5 fs. **The cap cost nothing:** 14 of 19 channels fittable,
which is exactly what the uncapped fit gets — the fit had simply been spending `c` in the region
where `c` is expensive, and there was an equally good region it had no reason to prefer.

What it did cost is basis richness at one geometry: surveying all nineteen channels either side of
the refit, `rxn_10` went from three states to two, `rxn_05` from two to one, and `rxn_11` from one to
two. The channels are all still fitted; three amplitudes are no longer large enough to admit an extra
diabat at their own transition states. `tests/geometry.py` moved to `rxn_13` and `rxn_14` because of
it, and carries the survey.

### The ZBL taper changed this dataset again, and this one did cost channels

`forcefield/zbl.py` now switches ZBL off with a Fermi function at 1.5 Å, because used unmodified it
was worth +0.72 eV at a water dimer's hydrogen bond and +251 kbar in a water box — a screened
*nuclear* potential evaluated an order of magnitude outside the range it was fitted in. Both datasets
were refit against the tapered form.

**The cost lands here rather than on water.** A transition state is where close intermolecular
contacts are, and therefore where the removed tail was largest, so the taper lowers the diabats
exactly at the geometries the coupling fit is inverted at. The channel count went from **14 of 19 to
12 of 19** — `rxn_06`, `rxn_11` and `rxn_16` are now decoupled — and two of those three had margins
under 0.1 eV before it. It is not a knob setting: `--frequency-weight` at 200× the default buys none
of them back.

The cap also had to tighten. The tapered surface is stiffer for H₂, and at `--max-wavenumber 4400`
the fit landed at 4602 cm⁻¹ — over its own cap. At **4200** it lands at 4325 cm⁻¹ (0.514 fs at 15
steps per period), and that tightening cost no further channels. `sweep.toml`'s 0.5 fs still holds,
with less margin than before.

The two datasets were regenerated with:

```sh
uv run python scripts/fit.py -r datasets/Water/Water.json --force-constants
uv run python scripts/fit.py -r datasets/HCombustion/HCombustion.json \
    --force-constants --max-wavenumber 4200
```

`fit.py` is **not idempotent** — `--max-k-scale` is relative to the `.jsonl` as they stand, so
re-running over already-fitted output compounds the bound and loses channels on its own. The
untapered form re-run over already-fitted files gives 15 of 19, not the 14 recorded above, which is
that effect and not a real difference. Refit once from the previous committed state; do not iterate.

Nothing in `output/` predates this: the sweep has not been run. If it had been, these runs would need
redoing — the force field is a different one.

### Re-enabling the 12-6 changed it again, and this time cost no channels

`forcefield/lj.py` is back in the force field. The taper above left the model with no intermolecular
wall at all above 1.5 Å and no dispersion anywhere, and a 64-water box came out a quarter too dense
as a result; the 12-6 is what supplies both. What makes it usable in a *reactive* model — it was
retired four times over — is `lj.switch`, a Fermi function that takes it from 500–1400 eV at a bond
length to 0.03–0.35 eV, small enough that it needs no exclusions and so is identical on every
diabatic state. See `forcefield/lj.py` and `DYNAMICTOPOLOGY.md` §2.2.1.

Both datasets were refit against it, and **the channel count did not move: 13 of 19, before and
after, losing exactly the same six.** Most surviving margins improved — `rxn_12`'s pre-fit margin
went from −0.71 to +0.02 eV, `rxn_19`'s from −0.71 to −0.34 — which is the opposite of what the ZBL
taper did, and for the mirror-image reason: the taper *removed* repulsion at transition-state
geometries and lowered the diabats there, while the 12-6 *adds* it back at exactly the intermolecular
contacts a transition state is made of.

**Read the 13 carefully — the pre-taper section above says 14, and both are right.** 13 is what the
*unchanged* pipeline produces when re-run over the current `.jsonl`, measured by disabling the new
term (`lj.SWITCH_RADIUS = 1e6` makes the 12-6 identically zero) and running the same two commands.
The 12 recorded in the taper section was the count on the previous pass. `fit.py` is not idempotent,
so a before/after quoted across two different input states is not a measurement of anything; 13 → 13
is.

What it did cost is a frequency, and the cost is on a mode the fit does not measure exactly.
**H₂O₂'s O–O stretch went from 3228 to 4285 cm⁻¹** — the 12-6 is 0.30 eV at that 1.45 Å bond and
steeply varying, so the fit stiffened it — and it is now the second-fastest mode in the dataset and
the one bond type `fit.dissociation.stretch_curvatures` is inexact on. The cap is applied through
that stand-in, which reads 4198.9 cm⁻¹ against 4285.2 measured, so `--max-wavenumber 4200` is being
enforced 86 cm⁻¹ (2.0%) low. The timestep that follows is 0.515 fs measured against 0.529 claimed;
`sweep.toml`'s 0.05 fs here and 0.5 fs in `density-300K` both cover it. `tests/test_fit.py`
:`test_the_cap_is_applied_to_a_number_close_enough_to_the_truth` bounds the gap, and its old form —
"H₂O₂'s O–O must stay below 3500 cm⁻¹" — is what caught this.

The fastest mode overall is 4314 cm⁻¹ (mol_06 O–H), down slightly from 4326. The regeneration
commands are unchanged:

```sh
uv run python scripts/fit.py -r datasets/Water/Water.json --force-constants
uv run python scripts/fit.py -r datasets/HCombustion/HCombustion.json \
    --force-constants --max-wavenumber 4200
```

`tests/geometry.py` moved again, and this time in the good direction: `SWITCHING_PATH_START` went
from 0.2 to 0.1 because **recrossing came back**. On the tapered surface no channel recrossed at all;
`rxn_12` at 0.1 now switches 3, 3 and 1 times on three seeds. Nine of the thirteen fitted channels
still never switch at any start.

### Where the frequencies stop

Every X–H stretch now sits *on* the repulsion's own curvature — H2 at 34.2 eV/Å² against ZBL's 33.9,
water's O–H at 65.9 against 65.7. The Morse contributes essentially nothing. That floor is a property
of the repulsion's functional form, not of any parameter, and it is what puts the O–H stretch at
4349 cm⁻¹ against an experimental 3756.

So the chemistry is still stiff by 15–20% on the X–H stretches and by more on O2, and going under it
means changing the repulsion's form — a softer, longer-ranged core that keeps the divergence at
contact but drops the curvature at bonding distances. That is a real piece of work with a full refit
behind it, and it buys accuracy rather than timestep: 0.5 fs is already reached.

### What the harness does now

Validated end to end: `run_one.py 10 --steps 20` resolves the config, runs the calculator, and writes
`config.json`, `log.jsonl` and `traj.xyz`; all 25 indices enumerate; 5 input boxes are packed.

### Confirmed on this box at 3000 K over 2 ps, at 0.5 fs

| | failing runs | required | measured |
| --- | --- | --- | --- |
| closest intermolecular approach | 0.60 / 0.62 Å | above 1.0 Å | **1.162 Å** |
| `E_nonbonded` | −41.7 / −46.6 eV, monotone | bounded | **+0.02 … +0.60 eV** |
| temperature | 3806 / 3960 / 6800 K | 3000 ± 150 K | **2960 K** mean |
| basis capped | — | never | **0** |
| NVE drift ratio per halving | — | ≥ 3 (dt²) | **4.21, 4.04** |

The box does not react over 2 ps and is not expected to: the admission gate correctly closes at
near-equilibrium geometries, and a barrier crossing is a rare event. That is why the sweep is 100 ps.

### Two things previously recorded here that were wrong

- *"No H2 + O2 initiation channel exists."* False. `Reaction(O2 + H2 -> HO2 + H)` is applicable to
  the sweep box; 8 reactions are found there. The error was reading reactant formulas off the stored
  forward direction, forgetting that `ReactionSet` stores both directions, reversed under the product
  hash.
- *"Reactions never fire, which may be the fundamental blocker."* False. At transition-state
  geometries the basis goes to 3 and 2 states on the two reactions the suite exercises. Packed boxes
  sit single-state because the admission gate correctly closes at near-equilibrium geometries, and a
  40 fs probe cannot sample a barrier crossing. That is why the sweep is 100 ps.

### The compute decision, which is no longer a decision

At 202 ms/step, 100 ps at 0.5 fs is 2e5 steps ≈ **11 hours of serial compute per run**, ~280
core-hours across the 25 runs, against `submit.slurm`'s 48 h wall. A run fits on its own.

This section used to weigh three ways of fitting ~10 days per run into 48 hours — chained restarts,
a longer allocation, or a shorter trajectory. Only the last was scientific, and it is the one that no
longer has to be considered: 100 ps stays at 100 ps. `run_one.py --restart` still works and is still
worth keeping for a job that hits the wall for some other reason, but nothing depends on it now.

### What remains

Nothing blocking. Two things worth knowing before reading results:

1. **The stretching frequencies are still 15–20% high on X–H and more on O2**, held there by the
   repulsion's own curvature rather than by any parameter. See *Where the frequencies stop* above.
   Rates that depend on a vibrational partition function inherit that.
2. **`Topology.from_atoms` still cuts bonds at a hard covalent-radii threshold** — O–H at 1.261 Å —
   and a hot O–H oscillates across it. MD carries its topology and never sees this; reading a frame
   back does, so `topologize.py`, `--restart` and any analysis that re-perceives can disagree with
   the run about what is bonded. `analyze.py` reads the stored connectivity and is unaffected.

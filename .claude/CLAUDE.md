# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Scope: inference only

**DynamicTopology is the inference package**: the force field, the reactive EVB machinery and the
MD scripts that run on a dataset. **All fitting lives in fast-forces** (`../fast-forces`), which
depends on this package and fits *through* its force field — `forcefield.evaluate` is the entry
point it scores templates with, so a template is fitted on exactly the sum it is simulated with.
The dataset tools that used to live here moved there:

| was (here) | now (fast-forces) |
| --- | --- |
| `scripts/fit.py`, `fit/dissociation.py`, `fit/coupling.py` | `fast-forces refit <manifest>`, `fastforces.refine`, `fastforces.coupling` |
| `scripts/compute.py` | `fast-forces label` |
| `scripts/convert.py`, `io/xml.py` | `fast-forces import-qforce`, `fastforces.qforce_xml` |
| `datasets/*/make_water.py`, `fit_charges.py`, `scripts/topologize.py`, `examples/run_calc*.sh` | removed; the ESP charge fit survives as `fastforces.charges` (`mk_charges`, `charge_terms`) |
| `.claude/skills/new-dataset/` | `.claude/skills/new-dataset/` |
| `tests/test_fit.py` | `tests/test_refine.py` |

`datasets/` stays here: it is what the simulations load and what these tests pin. fast-forces writes
into it. Anything that produces or refines a parameter belongs in fast-forces, not here.

## Environment

Managed with `uv` (Python >= 3.13, build backend `uv_build`, package root `src/DynamicTopology`).

```sh
uv sync                       # install deps + dev group (pytest, ipykernel)
uv run pytest                 # full test suite (run from repo root)
uv run pytest tests/test_gradients.py::TestQForceGradients::test_bond   # single test
uv run ruff check && uv run ruff format   # ruff is used but not declared in pyproject
```

Tests hardcode repo-root-relative paths (`datasets/HCombustion/HCombustion.json`,
`tests/data/...`), so pytest must be invoked from the repo root.

Run scripts with `uv run python scripts/<name>.py`:

- `singlepoint.py -i <xyz>` — single-point energies through the ASE calculator
- `nvt.py -i <xyz> [-o out.xyz]` — Langevin MD with reactive topology updates
- `mixture.py -n 100 -d 30 -x 1:1` — pack an H2/O2 box via `molify.pack`
- `npt.py`, `analyze.py`, `plot.py` — Berendsen NPT / aggregate sweep logs / plot MD trajectories

Note: `pyproject.toml` declares a `dt` console script pointing at `DynamicTopology:main`, but
`src/DynamicTopology/__init__.py` is empty — the entry point is not wired up.

A separate, older copy of this project lives at `../DynamicTopology` (flat `_molecules.py` /
`_reactionset.py` layout). It is **not** this repo — verify the working directory is
`.../DynamicTopo` before editing.

## Architecture

Reactive MD: bond topology changes during the simulation. Each force call rebuilds a local
reaction network, forms a multi-state EVB Hamiltonian, and takes the ground state — the
diagonalized state also decides the topology carried into the next step.

### Data flow (one `calculate()`)

`ase.py:DynamicTopology` (an ASE `Calculator`) → `system.py:System.calculate()`:

1. `ReactionSet.get_network(topology, bimol_cutoff)` builds an instantaneous `ReactionNetwork`:
   nodes = molecules (connected components), edges = applicable reactions. Unimolecular reactions
   are self-loops; bimolecular ones are added only when the minimum interatomic distance (with
   PBC) is under the cutoff.
2. For each network edge, `forcefield/coupling.py:EVBCoupling` computes an off-diagonal coupling
   from the reaction's stored TS geometry ensemble (a batched Kabsch alignment for the RMSD form); results
   are stashed back onto the edge dict as `coupling_energy` / `coupling_forces`.
3. `ReactionNetwork.states()` splits into connected subnetworks and enumerates diabatic states
   per subnetwork: state 0 is the current topology, each subsequent state is `Reaction.apply()`
   of one edge.
4. Each state's bonded energy comes from `forcefield/qforce.py:QForce`. The EVB matrix is built
   with state energies on the diagonal and couplings in row/column 0 only (state 0 couples to all
   others; others do not couple to each other). Ground state via `np.linalg.eigh`, energy and
   forces via Hellmann-Feynman (`np.einsum` with the lowest eigenvector).
5. The state with squared eigenvector weight > 0.9 becomes that subnetwork's new topology; the
   subnetwork topologies are merged and returned as `results["topology"]`, which the calculator
   feeds back into `System` — this is how bonds break and form across MD steps.
6. Two nonbonded repulsion terms — short-range repulsion (`forcefield/zbl.py:ZBL`) and the
   switched 12-6 (`forcefield/lj.py:LennardJones`) — are computed once on the whole system,
   topology-independent, and added on top; neither sits on the EVB diagonal. Their
   *intramolecular exclusions* do, because which pairs are 1-2/1-3/1-4 is exactly what a diabatic
   state disagrees about: `forcefield/exclusions.py` derives them at load and `QForce` evaluates
   them as ordinary additive terms.
7. **Electrostatics are per diabatic state and sit on the diagonal**, under either term, chosen
   by `global_params.electrostatics` through `forcefield/electrostatics.py`:
   - `"acks2"` (`forcefield/acks2.py`) is *fragment* ACKS2: each state minimizes its own ACKS2
     functional, with reference charges `q0` from its templates' `atom` terms (so an H3O+ holds
     +1 and a hop moves it) and softness only within a molecule (so no charge crosses between
     molecules). The energy is the minimum **less each molecule's isolated minimum**, so a lone
     template scores exactly zero and no `coulombexclusion` is needed; ACKS2 ignores those terms
     and its kernel is whole. The functional is stationary, so there is no charge-response
     term. The atoms outside every multi-state block are LU-factored once and folded into each
     state's small block solve (a Schur complement), so all states cost about one solve.
   - `"pointcharge"` (`forcefield/pointcharge.py`, dataset `datasets/Water-fixed-pc`) carries a
     `charge` term per atom and subtracts its `coulombexclusion` pairs' direct kernel per state.
     It is fragment ACKS2's zero-softness limit.

   Blocks couple through each other's ground-state-averaged charges (a Hartree product), so
   `System.calculate` sweeps the multi-state blocks until no weight moves (`SCF_TOLERANCE`; one
   sweep when at most one block is multi-state). Both terms sit behind one protocol — `prepare` /
   `bind` / `corrections` / `update` / `evaluate`, plus `__call__` for one topology — which the
   fitter and `evb.py` use too. `prepare` runs *before* `EVBBasis.build`, because the admission
   gate screens on the bonded gap plus the fixed-charge electrostatic one
   (`electrostatics.ElectrostaticGap`, off for a dataset with no nonzero charges).

**Electrostatics is the one nonbonded term that is not short ranged**, so under full periodicity it
is summed over images rather than truncated at the nearest one. `forcefield/ewald.py` holds both
forms of the ACKS2 charge kernel `erf(2r)/r` behind one interface — `matrix()` for the (n, n)
kernel, `contract(W)` for `dS/dr` and `dS/de` of `S = sum_ij W_ij K_ij` — and `acks2.py` never
branches on `pbc`. `MinimumImage` is the open-boundary kernel (and the fallback for a slab or wire,
where a 3D Ewald sum would be the wrong sum); `Ewald` is the lattice sum. `contract` is the reason
the kernel is an object: ACKS2's force (`W = E[q q^T]/2` over its states), `PointCharge`'s and the
admission gate's are the same contraction under different weights `W`, and the reciprocal-space
part of each is not a sum over pair separations at all.

Three things about the periodic kernel that the open-boundary one has no analogue for:

- **`K_ii` is nonzero** — an atom interacts with its own images — so the ACKS2 solve *adds* the
  hardness to the Coulomb diagonal instead of overwriting it. It is 80% of the rock-salt energy in
  `test_ewald.py`, not a rounding term.
- **The `k = 0` term is dropped and its background put back.** A constant added to every entry
  of `K` scales as `(sum_i q_i)^2` in the energy and `sum_i q_i` in each ACKS2 charge row (which
  the molecules' multipliers absorb, so the charges never see it). That made every *neutral*
  contraction right and every individual `K_ij` meaningless — it drifted with `kappa`. A charged
  cell (an H3O+ under per-molecule formal charges), a point-charge exclusion and the gate's
  difference of two states all contract `K` against a non-neutral weight, so `Ewald.background`
  is carried explicitly.
- **Reciprocal vectors strain inversely to positions** (`k -> (I - e) k`), which is where the
  reciprocal virial comes from; the structure factor is invariant, so there is no `v_a v_b` term.

`params.accuracy` sets `kappa` and the reciprocal cutoff together. Loosening it is cheap in the
reciprocal vector count (`(-log accuracy)^3`) and expensive in the *stress*, for the reason its
comment gives. Reciprocal vectors are selected on an integer ellipsoid rather than by `|k|`, so the
set is piecewise constant in the cell and a strain does not move a whole degenerate shell across the
cutoff. Cost is +24 ms on a 200-atom box against a 200-450 ms force call.

**This did not invalidate any `.jsonl`**, unlike the three radii below: every dataset template
carries `pbc="F F F"`, so the fitter (fast-forces' `refine`) sees the unchanged open-boundary kernel. The
bit-identical `energy_bonded` in `test_performance.py`'s reference block is the evidence.

**The repulsion is two terms with a hand-over between them,** and the pair is the thing to
understand before touching either:

| | form | Fermi switch | carries |
| --- | --- | --- | --- |
| `zbl.py:ZBL` | screened nuclear, no free parameters | off above `taper_radius = 1.5` Å, `taper_width = 0.12` Å | bond lengths and closer |
| `lj.py:LennardJones` | 12-6, q-force's σ and ε | on above `switch_radius = 2.2` Å, same width | intermolecular contact and dispersion |

Bare ZBL is fitted for keV nuclear stopping, and reaching it into the 1.5–3 Å range put +0.72 eV on
a water dimer's hydrogen bond and +251 kbar in a water box; the taper keeps >90% of it at every bond
length, so the wall stays where the Morse depths absorb it. Bare 12-6 is the opposite problem — 500
to 1400 eV at a bond length, which is why it had been retired — and its switch plus a soft core
(`u = 4ε[s⁻² − s⁻¹]`, `s = soft_core + (r/σ)⁶`, finite at contact) take it to 0.0002–0.035 eV there. **That is what lets its whole-system sum carry no exclusions**, hence be identical on
every diabatic state, hence be addable outside the Hamiltonian; the four historical failure modes
in `lj.py`'s docstring all descend from exclusions that sum no longer has. (Its small intramolecular
correction is an ordinary per-state `exclusion` term, as the ZBL one is — step 6 above.)

The two radii are deliberately *not* equal, so there is a gap from ~1.6 to ~2.0 Å where both are
small and `ACKS2` carries the hydrogen bond alone. Making them complementary is the obvious-looking
change and it is wrong: at 1.5 Å the switch is still 1.1e-2 where 12-6 is 953 eV, which puts +20.6 eV
on a single water molecule. `params.ForceFieldParams.switch_radius` has the measurements.

**The taper radii, the exclusion depth, the Morse asymptote and the charge smearing width are
`global_params` in the dataset manifest, not module constants.** `forcefield/params.py` holds
`ForceFieldParams` — the fields, the defaults, and the measurement behind each one.
`ReactionSet.load` reads the manifest's `global_params` block before it touches a template and
activates it process-wide; every force field reads it through `params.active()` at call time, never
bound into a default argument, since the fitter imports long before any manifest is read. Two
datasets whose blocks disagree cannot share a process: `params.activate` refuses the second and
names the fields that differ. `params.use(...)` is the explicit override.

**Changing a pinned value still invalidates that dataset's `.jsonl` files** —
fast-forces' `refine` solves against `E_QForce + E_nonbonded` and all three nonbonded terms are
inside it, so `fast-forces refit <manifest> --force-constants` has to be re-run for that dataset.
Both datasets here pin their values explicitly, so a change to a *default* no longer invalidates
anything silently. Note the refit is **not idempotent** with `--force-constants` (a coupling-only
refit is): re-running it over already-fitted output moves the
channel count on its own, so refit once from the previous state rather than iterating, and quote a
regression only against a baseline produced by the *same* pipeline over the *same* input files. The
cheapest way to get one is to disable the new physics (`switch_radius` at 1e6 makes the 12-6
identically zero) and re-run.

`forcefield/evaluate.py:evaluate(atoms, terms)` is the energy, forces and virial of **one**
topology, summed exactly as `System` sums a diabat (bonded + exclusions through `QForce`, the active
electrostatics, ZBL, 12-6), with the parts broken out. It is the entry point for anything that
scores a template on its own — a fitter, a report — and `tests/test_evaluate.py` holds it to
`System` on every template of every dataset to 1e-10.

`evb.py:EVBSystem` (exposed as `ase.py:EVB`) is the simpler alternative: a fixed list of states
with an empirical geometric-mean coupling `H_ij = sqrt((1+h)*H_ii*H_jj)`, no network rebuild and
no topology update.

### Core objects (`core/`)

- `Topology` — an `nx.Graph` of bonds plus an optional `Atoms` and a term list. Identity is the
  Weisfeiler-Lehman graph hash over `atomic_number` (cached in `_hash`); `molecules()` yields
  connected-component subgraphs that **keep global node indices**, which is what makes the
  index remapping throughout the codebase work. `_hash` and `_molecules` caches are invalidated
  only by `set_atoms`.
- `Reaction` — reactant `Topology`, product `Topology`, and a list of TS `Atoms` (the coupling
  ensemble). `get_mapping()` uses `nx.isomorphism.GraphMatcher` to map template indices onto
  live global indices; `apply()` diffs reactant/product edges and returns a new `Topology`.
- `ReactionNetwork` — `nx.MultiGraph` over molecules; `states()` produces the diabatic state list.
- `ReactionSetData` — the database, a dataclass: `molecules: {wl_hash -> Topology}`,
  `reactions: {wl_hash -> list[Reaction]}` (both directions stored, forward under the reactant
  hash and reversed under the product hash), the manifest's `ids`/`formulas` indexes onto
  molecule hashes, the dataset's `params`, and the manifest `source`. `from_manifest()` is the
  whole of the parse, which is what keeps the "activate `global_params` before reading a single
  template" ordering inside one function. Lookups are `channels()`, `by_id()`, `by_formula()`.
- `ReactionSet` — the cached query engine over one `ReactionSetData` (held as `.data`), mapping
  it onto a live system's atom indices. `get_terms()` looks up a molecule's parameter template
  and remaps indices into the live system, caching on the molecule signature; four caches hang
  off it and `load()` resets them all. `set_template_terms()` is how the fitter (fast-forces' `refine.install_templates`) installs a
  refit without writing files — it clears the term cache, which is the half that is easy to
  forget.

### The "term" format

`core/types.py` defines `Term = dict[str, Any]`, in practice
`{"type": str, "atoms": {name: index}, "kwargs": {param: value}}`. `Topology.set_terms()` /
`Reaction.set_terms()` / `ReactionNetwork.set_terms()` each transpose a term list into a
`term_dict` of `{type: {"atoms": (n, k) int array, "kwargs": {param: (n,) array}}}` — the
vectorized layout every force field consumes. (The three `set_terms` implementations are
near-duplicates.)

Force field classes dispatch by convention: `__call__` iterates `term_dict` and looks up
`self.compute_<term_type>`, silently skipping types with no matching method. Adding a new
functional form means adding a `compute_*` method that returns `(energy, forces, virial)` —
nothing else needs to change.

**Units: eV and Å everywhere, on disk included.** Every force field, every `Term`, every
`term_dict`, `ForceFieldParams` and every `.jsonl` is in ASE units, so `io.json.read_jsonl` /
`write_jsonl` move rows through untouched. nm and kJ/mol exist only at the q-force/OpenMM boundary
in fast-forces (`import-qforce` in, the OpenMM export out), and `io/units.py` is the table both
convert through: `UNIT_POWERS` records each parameter's dimension, and a parameter missing from it
raises. (Until 2026-09-23 the `.jsonl` was nm/kJ/mol, converted on every read; every shipped file
was rewritten with the exact doubles the old reader produced, and `read_jsonl` refuses a bond `r0`
under 0.5 as a file that missed the move.) Hand-built test terms go through
`test_gradients.make_term`, which states its literals in q-force units and converts them through
`io.units.from_openmm`.

The `virial` is `dE/d(strain)`, a 3x3 — every energy here is a function of minimum-image
displacement vectors only, so a homogeneous strain maps `v -> (I + e) v` and
`W_ab = sum v_a (dE/dv)_b`. `QForce._virial` builds it from the same per-pair gradient the forces
are scattered from, so no new derivative is needed. The virial is an energy, in eV. `System.calculate`
contracts the per-state virials with the ground-state eigenvector exactly as it does the forces,
and `ase.py` divides by the cell volume to publish `stress`, which is what the ASE barostats need.

### I/O and datasets

`io/json.py` reads/writes the canonical JSONL term format (one term per line), in eV/Å with
no conversion. There is no `.h5` route and no `ReactionSet.save()` — a manifest is the only thing a
set loads from. (q-force XML import is fitting-side: `fast-forces import-qforce`.)

A dataset is a manifest JSON (`datasets/HCombustion/HCombustion.json`) listing molecule and
reaction entries by *extensionless* path; loading pairs each `<path>.xyz` (geometry; reactions
read `index=":"` as reactant/TS.../product) with `<path>.jsonl` (parameters). Each entry also
carries a `smiles` — a molecule SMILES, or a `reactants>>products` reaction SMILES whose two
sides match the first and last frames of the `.xyz`. It is documentation only: identity is the
WL hash of the bond graph, and `from_manifest` never reads the field. The manifest also
carries `global_params`, the force field constants the dataset was fitted at — see the Architecture
note above and `forcefield/params.py`.

### Tests

- `test_evaluate.py` — `forcefield.evaluate` against `System` on every lone template, open and
  periodic.
- `test_paramio.py` — the `.jsonl` I/O: eV/Å on disk and in memory, a read/write of every dataset
  file is byte-identical, a legacy nm file is refused, and the q-force/OpenMM unit table.
- `test_gradients.py` — every analytic force is checked against central finite differences.
  Any new or edited `compute_*` method must get a case here.
- `test_stress.py` — the same for the virial, differenced against the *cell* with the atoms scaled
  affinely. A new or edited `compute_*` needs a case here as well as in `test_gradients.py`; it
  also asserts the virial is symmetric, which catches a transposed contraction that a
  pressure-only (trace) check would pass.
- `test_ewald.py` — the periodic charge kernel against values rather than against itself: the NaCl
  Madelung constant (which `MinimumImage` misses by 17%), independence from the Ewald splitting
  parameter, and the 1/L³ approach to the open-boundary kernel. Finite differences cannot see a
  lattice sum converging to the wrong number, which is what this file is for.
- `test_acks2_fragment.py` — fragment ACKS2 against values: each state's Schur solve equals the
  full whole-system solve, every molecule keeps its formal charge, a lone template scores zero,
  the zero-softness limit is `PointCharge`, a nearby anion flips which proton position is lower,
  and the admission gate equals the point-charge diagonal difference (with its gradient).
- `test_pointcharge.py` — the point-charge path through `System`: the block sweep (forces are
  off by 0.058 eV/Å on two interacting Zundels if it stops after one pass), a charged Ewald cell's
  forces and virial, and the exclusion's periodic self-image. Runs Water-fixed-pc under
  `params.use` so it cannot collide with the other datasets in the process.
- `test_get_network.py` — fingerprint regression over `ReactionSet.get_network` (node/edge counts,
  reaction hashes, atom mappings) on a 250-molecule H2/O2 box.
- `test_optimizations.py` — locks in the caching behavior of `Topology.hash`/`molecules` and
  `ReactionSet.get_terms_topology`, so caches must stay correct-by-key, not just fast.
- `test_params.py` — each global parameter is overridden to a value the defaults do not contain and
  the reported number has to move, which is the only way to catch a call site that still reads a
  stale default; plus the manifest plumbing and the two-dataset collision.
- `test_energy_conservation.py` — NVE drift through the ASE calculators, plus a dt-halving check
  that the residual is O(dt²) integrator error rather than inconsistent forces. H2O2 + HO2 is the
  case that exercises intermolecular ACKS2 (a lone molecule's electrostatics are zero).
- `tests/profile/` holds cProfile drivers, not pytest tests.

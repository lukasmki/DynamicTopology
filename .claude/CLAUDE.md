# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- `convert.py -i <dir|xml> -o <dir>` — OpenMM-style XML force field → `.jsonl` terms
- `topologize.py`, `plot.py` — re-perceive bonds / plot MD trajectories

Note: `pyproject.toml` declares a `dt` console script pointing at `DynamicTopology:main`, but
`src/DynamicTopology/__init__.py` is empty — the entry point is not wired up. `forcefield/acks2.py`
imports `scipy`, which is also not declared in `pyproject.toml` (it arrives transitively).

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
   from the RMSD to the reaction's stored TS geometry ensemble (`superpose3d` alignment); results
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
6. Three nonbonded terms — electrostatics (`forcefield/acks2.py:ACKS2`), short-range repulsion
   (`forcefield/zbl.py:ZBL`) and the switched 12-6 (`forcefield/lj.py:LennardJones`) — are computed
   once on the whole system, topology-independent, and added on top. None sits on the EVB diagonal.

**Electrostatics is the one nonbonded term that is not short ranged**, so under full periodicity it
is summed over images rather than truncated at the nearest one. `forcefield/ewald.py` holds both
forms of the ACKS2 charge kernel `erf(2r)/r` behind one interface — `matrix()` for the (n, n)
kernel, `contract(W)` for `dS/dr` and `dS/de` of `S = sum_ij W_ij K_ij` — and `acks2.py` never
branches on `pbc`. `MinimumImage` is the open-boundary kernel (and the fallback for a slab or wire,
where a 3D Ewald sum would be the wrong sum); `Ewald` is the lattice sum. `contract` is the reason
the kernel is an object: the explicit Coulomb force and the `-lam^T (dA/dr) x` charge response are
the same contraction under different weights `W`, and the reciprocal-space part of each is not a
sum over pair separations at all.

Three things about the periodic kernel that the open-boundary one has no analogue for:

- **`K_ii` is nonzero** — an atom interacts with its own images — so `build_system` *adds* the
  hardness to the Coulomb diagonal instead of overwriting it. It is 80% of the rock-salt energy in
  `test_ewald.py`, not a rounding term.
- **The `k = 0` term is dropped**, which is only legal because `build_system` constrains
  `sum_i q_i = 0`: a constant added to every entry of `K` scales as `(sum_i q_i)^2` in the energy
  and `sum_i q_i` in each ACKS2 row. A charged system would need it back.
- **Reciprocal vectors strain inversely to positions** (`k -> (I - e) k`), which is where the
  reciprocal virial comes from; the structure factor is invariant, so there is no `v_a v_b` term.

`ewald.ACCURACY` sets `kappa` and the reciprocal cutoff together. Loosening it is cheap in the
reciprocal vector count (`(-log accuracy)^3`) and expensive in the *stress*, for the reason its
comment gives. Reciprocal vectors are selected on an integer ellipsoid rather than by `|k|`, so the
set is piecewise constant in the cell and a strain does not move a whole degenerate shell across the
cutoff. Cost is +24 ms on a 200-atom box against a 200-450 ms force call.

**This did not invalidate any `.jsonl`**, unlike the three radii below: every dataset template
carries `pbc="F F F"`, so `fit/dissociation.py` sees the unchanged open-boundary kernel. The
bit-identical `energy_bonded` in `test_performance.py`'s reference block is the evidence.

**The repulsion is two terms with a hand-over between them,** and the pair is the thing to
understand before touching either:

| | form | Fermi switch | carries |
| --- | --- | --- | --- |
| `zbl.py:ZBL` | screened nuclear, no free parameters | off above `TAPER_RADIUS = 1.5` Å, `TAPER_WIDTH = 0.12` Å | bond lengths and closer |
| `lj.py:LennardJones` | 12-6, q-force's σ and ε | on above `SWITCH_RADIUS = 0.22` nm, same width | intermolecular contact and dispersion |

Bare ZBL is fitted for keV nuclear stopping, and reaching it into the 1.5–3 Å range put +0.72 eV on
a water dimer's hydrogen bond and +251 kbar in a water box; the taper keeps >90% of it at every bond
length, so the wall stays where the Morse depths absorb it. Bare 12-6 is the opposite problem — 500
to 1400 eV at a bond length, which is why it had been retired — and its switch takes it to 0.03–0.35
eV there. **That is what lets it carry no exclusions**, hence be identical on every diabatic state,
hence be addable outside the Hamiltonian; the four historical failure modes in `lj.py`'s docstring
all descend from exclusions it no longer has.

The two radii are deliberately *not* equal, so there is a gap from ~1.6 to ~2.0 Å where both are
small and `ACKS2` carries the hydrogen bond alone. Making them complementary is the obvious-looking
change and it is wrong: at 1.5 Å the switch is still 1.1e-2 where 12-6 is 953 eV, which puts +20.6 eV
on a single water molecule. `lj.SWITCH_RADIUS`'s comment has the measurements.

**Changing `TAPER_RADIUS`, `TAPER_WIDTH` or `SWITCH_RADIUS` invalidates every `.jsonl` in every
dataset** — `fit/dissociation.py` solves against `E_QForce + E_nonbonded` and all three nonbonded
terms are inside it, so `scripts/fit.py --force-constants` has to be re-run for both datasets. Note
`fit.py` is **not idempotent**: re-running it over already-fitted output moves the channel count on
its own, so refit once from the previous state rather than iterating, and quote a regression only
against a baseline produced by the *same* pipeline over the *same* input files. The cheapest way to
get one is to disable the new physics (`SWITCH_RADIUS = 1e6` makes the 12-6 identically zero) and
re-run.

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
- `ReactionSet` — the database. Keyed by WL hash: `data["molecules"][hash] -> Topology`,
  `data["reactions"][hash] -> list[Reaction]` (both directions stored, forward under the reactant
  hash and reversed under the product hash). `get_terms()` looks up a molecule's parameter
  template and remaps indices into the live system, caching on `(mol_hash, frozenset(nodes))`.

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

`QForce` works internally in nm and kJ/mol and converts to ASE units (eV, Å) at the end of
`__call__`; the per-term `compute_*` methods return the *unconverted* values.

The `virial` is `dE/d(strain)`, a 3x3 — every energy here is a function of minimum-image
displacement vectors only, so a homogeneous strain maps `v -> (I + e) v` and
`W_ab = sum v_a (dE/dv)_b`. `QForce._virial` builds it from the same per-pair gradient the forces
are scattered from, so no new derivative is needed. **Unit trap:** the virial is an energy, so it
converts with `units.kJ / units.mol` and *no* length factor, unlike the forces. `System.calculate`
contracts the per-state virials with the ground-state eigenvector exactly as it does the forces,
and `ase.py` divides by the cell volume to publish `stress`, which is what the ASE barostats need.

### I/O and datasets

`io/xml.py` parses OpenMM-style `<Forces>` XML (as emitted by q-force) into terms; `io/json.py`
reads/writes the canonical JSONL term format (one term per line). `io/h5.py` is empty and
`ReactionSet.load()` raises `NotImplementedError` for `.h5`; `ReactionSet.save()` is a stub.

A dataset is a manifest JSON (`datasets/HCombustion/HCombustion.json`) listing molecule and
reaction entries by *extensionless* path; loading pairs each `<path>.xyz` (geometry; reactions
read `index=":"` as reactant/TS.../product) with `<path>.jsonl` (parameters).

### Tests

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
- `test_get_network.py` — fingerprint regression over `ReactionSet.get_network` (node/edge counts,
  reaction hashes, atom mappings) on a 250-molecule H2/O2 box.
- `test_optimizations.py` — locks in the caching behavior of `Topology.hash`/`molecules` and
  `ReactionSet.get_terms_topology`, so caches must stay correct-by-key, not just fast.
- `test_energy_conservation.py` — NVE drift through the ASE calculators, plus a dt-halving check
  that the residual is O(dt²) integrator error rather than inconsistent forces. Two xfails pin the
  ACKS2 frozen-charge approximation (see below).
- `tests/profile/` holds cProfile drivers, not pytest tests.

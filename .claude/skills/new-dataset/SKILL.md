---
name: new-dataset
description: Build, validate, fit and run a new DynamicTopology reaction-set dataset (molecule templates + reaction channels + EVB couplings) like datasets/Water or datasets/HCombustion. Use when asked to create, generate, add, extend, check, debug, fit or run a dataset, reaction set, reaction network, molecule template, EVB coupling, or a new chemistry for reactive MD.
---

# Building a new reaction-set dataset

Paths are relative to the repo root. Run everything from there — `scripts/` and the
tests hardcode repo-root-relative paths.

A dataset is a manifest JSON plus, per entry, an `.xyz` (geometry + `energy=`) and a
`.jsonl` (force-field terms). `datasets/Water/` is the worked example built with this
skill; read its [README](../../../datasets/Water/README.md) alongside this.

**The hard part is not the file format — it is that a wrong dataset does not crash.**
Missing an ACKS2 term, or shipping a 2-frame reaction, produces a silently wrong
energy. That is what `dataset_driver.py` exists to catch.

## The driver

`.claude/skills/new-dataset/dataset_driver.py` — four subcommands, used in order.
`check` and `probe` need no quantum chemistry, so iterate against those.

```bash
uv run python .claude/skills/new-dataset/dataset_driver.py check datasets/Water/Water.json
uv run python .claude/skills/new-dataset/dataset_driver.py probe datasets/Water/Water.json
uv run python .claude/skills/new-dataset/dataset_driver.py run   datasets/Water/Water.json --template h2o -n 64
```

| subcommand | what it proves |
|---|---|
| `scaffold <dir>` | manifest + skeleton + placeholder couplings (breaks the load/fit cycle below) |
| `check <manifest>` | every defect that loads fine and is still wrong |
| `probe <manifest>` | reaction table, species closure, **energy bookkeeping** |
| `run <manifest>` | packs a box, builds the network, calls the force field, integrates NVE |

`run` is the only step that proves the dataset works. On Water it prints:

```
loaded Water.json: 5 templates, 6 stored reactions
box: 64 x H2O = 192 atoms, 12.43 A cube at 997.0 kg/m^3
network: 64 nodes, 3888 edges (0.20s at cutoff 4.0 A)
force call: 2.52s   E = -466.750861 eV   |F|max = 3.7454 eV/A   finite = True
NVE 40 x 0.5 fs: drift = ... meV/atom
```

## Workflow

### 1. Scaffold

```bash
uv run python .claude/skills/new-dataset/dataset_driver.py scaffold datasets/MySet \
    --name MySet --description "..." --author "..." \
    -m h o h2o -x my-transfer
```

Writes the manifest and `molecules/`, `reactions/`. Re-run it after the reaction
`.xyz` files exist to emit the placeholder coupling terms (it needs the atom count).

**Why placeholders are mandatory:** `ReactionSet.load` opens the `.jsonl` of *every*
manifest entry, reactions included — but reaction couplings are what `scripts/fit.py`
writes, and `fit.py` builds a `ReactionSet` to do it. Without a seeded placeholder the
dataset cannot be loaded by the tool whose job is to write its terms.

### 2. Geometries

Write a generator script in the dataset directory; `datasets/Water/make_water.py` is
the model. Build frames from internal coordinates, not hand-typed Cartesians.

Molecule `.xyz`: **exactly one frame**. Reaction `.xyz`: **>= 3 frames**, ordered
`[reactant, TS..., product]`, identical atom count and ordering in every frame.
Set `atoms.info["connectivity"] = [[i, j, None], ...]` and
`atoms.set_initial_charges(np.zeros(n))`, then `io.write(path, frames, format="extxyz")`.
Frames 0 and −1 define the reactant and product topologies; the middle frame's
connectivity is ignored and only its positions are used (the coupling reference).

Reactant and product frames must sit in a **minimum region**. For the Water hops the
gas-phase symmetric complex *is* the global minimum, so a reactant placed at contact
has no barrier to switch the coupling off against; those frames are held at 2.70–2.75 Å
O–O instead of 2.4. Check this — `probe`'s `TS-R`/`TS-P` columns show it directly.

### 3. Energies

```bash
uv run python scripts/compute.py -i <f>.xyz -o <f>.xyz -c <charge> -s <spin> \
    -b aug-cc-pvtz --atom-cache atoms.json
```

Writes `energy=` = atomization energy in eV against neutral free atoms. `-c`/`-s`
apply to **every frame in the file**, so one charge per file. Reading happens before
the output is unlinked, so `-i` and `-o` may be the same path. Times measured here:
3 s for H2O/cc-pVTZ, 2.5 min for a 7-atom cation at aug-cc-pVTZ.

**Charged species — the trap that raises nothing.** Compute each species at its own
formal charge against *neutral* free atoms. The atomic references then cancel exactly,
provided the channel conserves electrons across its templates. Verify with `probe`:
Water's `dE templates` for autoionization is +9.79 eV against a literature ~9.8.
A number off by ~13.6 or ~1.8 eV means an ionization potential or electron affinity
is being double-counted.

Use `aug-cc-pVTZ` if any template is an anion; plain cc-pVTZ badly underconverges
electron affinities.

**Templates cannot carry charge.** A template is keyed by a Weisfeiler–Lehman hash over
atomic numbers, so OH⁻ and the OH radical are the *same* template. Pick the protonation
state your chemistry actually needs and say so in the dataset README.

### 4. Fit

```bash
uv run python scripts/fit.py -r datasets/MySet/MySet.json --force-constants -n   # dry run
uv run python scripts/fit.py -r datasets/MySet/MySet.json --force-constants
```

`--force-constants` (implies `--bonds`) is the route the shipped datasets took. It
rescales Morse depths so the bonded energy at the template geometry equals the
atomization energy, writes `reference E0: 0.0`, then fits each coupling.

**Do not skip it.** Without `--bonds` the synthesized reference is a large constant
shift that survives into stretched geometries, and the diabats at the transition state
blow up. Measured on Water: amplitudes of −38/−24/−33 eV before, −4.6/−4.6/−5.2 after.
HCombustion's are −0.7 to −5.1; an amplitude far outside that range means this step
was skipped or the energy zeros do not line up.

Read the two lines it prints at the end: `N of M channels fittable`, and
`fastest mode ... -> N fs at 15 steps/period` — that quotient **is** your MD timestep.

**`fit.py` is not idempotent.** `--max-k-scale` and the shape bound are relative to
the `.jsonl` files *as they stand*, so a second run compounds them: re-running the dry
run over Water's fitted output reports `4367 cm^-1 (h1o O-H)` where the fit that
produced those files reported `4386 (h2o O-H)`. Fit once from the q-force terms. To
refit, restore the unfitted `.jsonl` first.

### 5. Verify

```bash
uv run python .claude/skills/new-dataset/dataset_driver.py check datasets/MySet/MySet.json
uv run python .claude/skills/new-dataset/dataset_driver.py probe datasets/MySet/MySet.json
uv run python .claude/skills/new-dataset/dataset_driver.py run   datasets/MySet/MySet.json --template h2o
uv run pytest -q     # 193 passed in ~4 min; adding a dataset should not move this
```

## Gotchas

Each of these was measured in this repo, not inferred.

- **A missing ACKS2 `atom` term does not raise.** It equilibrates charge over the atoms
  it was given and returns a wrong energy — 1.15 eV on a 7-atom Zundel. Every template
  needs one `atom` term per atom. `check` enforces this.
- **A 2-frame reaction `.xyz` does not raise.** `Reaction.from_atoms` slices
  `atoms[1:-1]` for the TS ensemble, so it becomes empty and the reaction degrades to a
  no-op whose equation reads `H2O + HO -> H2O + HO`. The force call still returns a
  number. `check` enforces >= 3 frames.
- **A trailing blank line in a `.jsonl` raises `JSONDecodeError`** in `load`
  (`readlines()` + `json.loads`). One trailing newline is fine; a blank line is not.
  The shipped files carry neither — `write_jsonl` joins with `"\n"` and adds none.
- **Every term of one type in one file must state the same kwarg keys.**
  `Topology.set_terms` transposes them into dense arrays and raises otherwise.
- **`atoms` dicts are consumed positionally** (`tuple(term["atoms"].values())`).
  The key names `p1`, `p2`, ... are convention; the order is load-bearing.
- **Unknown term types are silently skipped.** Force fields dispatch to
  `compute_<type>`; a typo'd type costs you the term with no error.
- **Manifest paths are extensionless** and relative to the manifest's directory.
- **Every connected component of every reactant and product needs its own template**,
  or `get_terms_topology` raises at runtime. `probe` checks closure up front.
- **Degenerate reactions are stored twice.** `add_reaction` files a reaction under its
  reactant hash and its reverse under its product hash; when those are equal (a
  proton hop) both land in the same list, and `get_network` enumerates every such
  channel twice — 4086 edges against 3886 on a 64-water box. It is **energetically
  inert**: `basis.py` keys admitted states by product topology, so duplicates collapse
  and the total energy is bit-identical with and without them, including at the
  transition-state geometries. The cost is ~5% redundant edges. `probe` flags it.
- **`packmol` inherits the template's `connectivity`**, which indexes the template's
  atoms, not the box's. Drop `box.info["connectivity"]` before building a `Topology`
  or the perceived bonding is garbage. The driver does this.
- **`molify.pack` takes `list[list[Atoms]]`** — `pack([[h2o]], [64], 997.0)`, not
  `pack([h2o], ...)`, which fails with `'Atom' object has no attribute 'get_masses'`.
- **`Topology.BOND_SCALE` is 1.3, `molify`'s default is 1.2.** `scripts/topologize.py`
  calls `ase2networkx(a, False)` positionally and so uses 1.2, which splits H2
  (cutoff 0.744 Å vs an equilibrium 0.7445). Only matters for files without explicit
  connectivity.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: .../reactions/<name>.jsonl` from `ReactionSet` | Seed placeholders: `scaffold` again after the reaction `.xyz` exist. |
| `json.decoder.JSONDecodeError: Expecting value: line N` at load | Trailing blank line in that `.jsonl`. |
| `AssertionError` in `load`, no message | A molecule `.xyz` has more than one frame. |
| `ValueError: N terms but only M state 'x'` | One term of a type is missing a kwarg its siblings have. |
| Long `_unknown_molecule_message` at runtime | A reactant/product fragment has no template. Run `probe`. |
| `fit.py` reports channels `decoupled` / `NO` margin | Diabats sit below the reference barrier. Use `--force-constants`, not `--bonds` alone. |
| Coupling amplitudes of tens of eV | `--bonds`/`--force-constants` was skipped; the constant reference shift is inflating the diabats. |
| `'Atom' object has no attribute 'get_masses'` | `molify.pack` needs `list[list[Atoms]]`. |
| NVE drift >> 0.01 meV/atom in `run` | Timestep too large. Use the `fastest mode` quotient from `fit.py`. |

## Environment

Verified on macOS (darwin 24.6.0) with `uv`, Python 3.13. `uv sync` installs
everything; `pyscf`, `packmol` and `superpose3d` are already in `.venv`. There is
**no geometry optimizer** (`geometric` and `pyberny` are both absent), which is why
geometries come from literature/symmetry and only single points are computed.
`scipy` and `ruff` are used but not declared in `pyproject.toml`.

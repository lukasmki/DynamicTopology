# Water

Bulk water proton transfer: the two degenerate Grotthuss hops and autoionization,
at wB97X-V/aug-cc-pVTZ.

| id | template | species | E_atomization (eV) |
|----|----------|---------|--------------------|
| 1 | `molecules/h`   | H (free atom)  | 0.000000 |
| 2 | `molecules/o`   | O (free atom)  | 0.000000 |
| 3 | `molecules/h1o` | **OH-**        | -6.335442 |
| 4 | `molecules/h2o` | H2O            | -9.966135 |
| 5 | `molecules/h3o` | **H3O+**       | -3.810780 |

| id | channel | atoms | A (eV) | a (1/A^2) |
|----|---------|-------|--------|-----------|
| 1 | `reactions/h3o-h2o-transfer`  | H3O+ + H2O <-> H2O + H3O+ (Zundel) | 7 | -4.580 | 95.6 |
| 2 | `reactions/h2o-oh-transfer`   | OH- + H2O <-> H2O + OH- (H3O2-)    | 5 | -4.578 | 258.5 |
| 3 | `reactions/h2o-autoionization`| H2O + H2O <-> H3O+ + OH-           | 6 | -5.184 | 659.8 |

Reactions 1 and 2 are degenerate and their product frame is the reactant's mirror
image, so they are exactly thermoneutral by construction.

## Protonation states

A template is keyed by the Weisfeiler-Lehman hash over atomic numbers, so the
force field **cannot distinguish OH- from the OH radical, or H3O+ from the
(unbound) H3O radical**.  This is a bulk-water set, so both are built as the
ions.  That is the single most important thing to know before reusing these
templates: `h1o` here is *not* HCombustion's `mol_03`, and this set must not be
merged with a radical-chemistry set that needs the neutral OH.

It is also why there is no O-H homolysis channel.  `h` and `o` are carried only
so an atom that ends up isolated during MD still has a template.

## Electron bookkeeping

Each species is computed at its own formal charge against **neutral** free atoms.
Every channel conserves electrons across the templates it connects
(H3O+ 10 + H2O 10 = H2O 10 + H3O+ 10; OH- 10 + H2O 10 = H2O 10 + OH-;
2 x H2O 20 = H3O+ 10 + OH- 10), so the atomic references cancel exactly and no
ionization or electron-affinity correction is needed anywhere.

The check that this is right: -3.810780 + -6.335442 - 2 x -9.966135 = **+9.79 eV**
for gas-phase autoionization, against a literature value of ~9.8 eV.
aug-cc-pVTZ rather than HCombustion's cc-pVTZ because two of the five templates
are anions and one channel makes an anion; without diffuse functions the
electron affinity that sets the autoionization energy is badly underconverged.

## Provenance

```sh
uv run python datasets/Water/make_water.py           # geometries only
# then, per file, with the charge/spin in the table above:
uv run python scripts/compute.py -i <f>.xyz -o <f>.xyz -c <q> -s <s> \
    -b aug-cc-pvtz --atom-cache atoms.json
uv run python scripts/fit.py -r datasets/Water/Water.json --force-constants
```

Geometry sources, all literature/symmetry rather than optimized here (there is
no geometry optimizer in this environment):

- `h2o` 0.9584 A / 104.45 deg; `h3o` 0.976 A / 111.8 deg C3v; `h1o` 0.964 A.
- Reaction frames are laid out with both oxygens on the x axis and the
  transferring proton on that axis, so a frame is fully specified by the O-O
  distance and where the proton sits along it.  See `make_water.py`.
- The two hops hold their **reactant** frames at 2.75 / 2.70 A rather than at the
  2.4 A where the symmetric Zundel and H3O2- are the gas-phase global minima.
  At contact there is no barrier at all, so a reactant frame placed there is not
  a minimum and `fit.coupling.fit_width` -- which asks where the coupling must
  switch off -- has nothing to switch off against.

Term sources:

- `atom` (ACKS2 mu/eta/softness) and `lennardjones`: HCombustion's O and H
  values verbatim.  Nothing in the repo regenerates these.
- `h`, `o`: byte-identical to HCombustion `mol_08` / `mol_07`.
- `h1o`, `h2o`: HCombustion `mol_03` / `mol_04` q-force terms, re-referenced to
  this dataset's energies by `fit.py --force-constants`.
- `h3o`: **by analogy to `h2o`**, and the weakest link in the set.  Angle and
  cross-term force constants are H2O's; `theta0` carries H2O's +2.73 deg q-force
  offset onto H3O+'s 111.8; the Morse `r0`/`k` are H2O's (`r0` there is a fitted
  parameter ~0.26 A inside the real bond, not a bond length).  Only the depth
  and shape are genuinely fitted, and `--bonds` scales H3O+'s depths to 0.67 of
  H2O's -- an artifact of a cation's atomization energy being small, not of its
  O-H bonds being weak.  Replace these with q-force output when it is available.

## Fit and cost

All 3 channels are fittable; margins 0.71-3.39 eV after the refit.  Fastest mode
4359 cm^-1 (h2o O-H), just under the 4400 cap, i.e. **dt = 0.5 fs**.

Refit twice, for the two changes to the repulsion, both of which sit inside the
`E_nonbonded` the fit solves against: `ZBL` acquiring its taper
(`forcefield/zbl.py`), and `forcefield/lj.py` being switched back on.  Neither
cost this dataset anything -- still 3 of 3 channels, still under the cap, both
times -- unlike HCombustion, which lost two to the first and nothing to the
second.  The margins moved because the diabats moved, not because the fit got
worse.

**What the taper bought here is the hydrogen bond, and the 12-6 kept it.**
Against a CCSD(T) reference of -0.218 eV at 2.91 A, the water dimer reads

    R_OO (A)      2.40     2.60     2.75     2.91     3.10     3.30     3.60
    tapered ZBL  ~   -    -0.007   -0.109   -0.112   -0.091   -0.071   -0.050
    + the 12-6  +0.776  +0.069   -0.078   -0.100   -0.090   -0.080   -0.062

Before the taper it was **+0.598 eV at 2.91 A with no minimum at any
separation** -- ZBL alone contributed +0.719 eV there, three times the whole
hydrogen bond and the wrong way.  It is now +0.009 eV.

The 12-6 costs 0.012 eV of the binding, 11%, and leaves the minimum exactly
where it was.  What it adds is the first column: **a wall**.  Two waters at
2.40 A used to be free to keep closing, which is why the liquid came out a
quarter too dense; they now pay 0.78 eV.  What is still missing is depth --
`ACKS2` supplies -0.121 eV of the reference -0.218 and the dispersion tail adds
little at 2.91 A -- so expect a bound but under-cohesive liquid.

**The reactive machinery is dormant in neutral water at 300 K**, and that is a
property of this dataset rather than of the settings.  A 64-water box logs
`max_nstates = 1` for every block -- no diabatic state is ever admitted -- at the
production `eps` of 0.05 eV *and* at the 1e-3 default, so nothing is being gated
out.  The cause is `h2o-autoionization`'s fitted width, `a = 611 1/A**2`: the
coupling is `exp(-a * rmsd**2)`, so it switches on only within ~0.04 A RMSD of
its stored transition state, and neutral water at 300 K never gets there.  That
is physically right -- autoionization is rare -- but it means a neutral-water
trajectory is effectively fixed-topology, and any density measured from one is a
fixed-topology number.  The hop channels need an ion present to do anything.

Verified on a packed 64-water box (12.43 A cube, 997 kg/m^3): forces finite, NVE
drift 0.002 meV/atom over 40 x 0.5 fs, ~2.5 s per force call.

## Note on the degenerate channels

`ReactionSet.add_reaction` stores a reaction under its reactant hash and its
reverse under its product hash.  For reactions 1 and 2 those hashes are equal,
so each is stored twice and `get_network` enumerates every hop channel twice
(204 + 204 and 22 + 22 on the box above, 4086 edges against 3886).  It is
**energetically inert** -- `basis.py` keys admitted states by product topology,
so the duplicates collapse, and the total energy is bit-identical with and
without them, at the transition-state geometries included.  The cost is ~5%
redundant network edges.  HCombustion never exposed this because none of its 19
channels is degenerate.

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
| 1 | `reactions/h3o-h2o-transfer`  | H3O+ + H2O <-> H2O + H3O+ (Zundel) | 7 | -3.882 | 19.9 |
| 2 | `reactions/h2o-oh-transfer`   | OH- + H2O <-> H2O + OH- (H3O2-)    | 5 | -3.987 | 23.5 |
| 3 | `reactions/h2o-autoionization`| H2O + H2O <-> H3O+ + OH-           | 6 | -4.140 | 46.8 |

All three are **atom transfers**, so all three carry a `threebody` coupling: a
Gaussian in the transferring proton's own triangle,

    g = (ra - ra0)**2 + (rb - rb0)**2 + (d - d0)**2,   V = A exp(-a g)

with `ra`/`rb` the proton's distances to donor and acceptor and `d` the
donor-acceptor separation.  The amplitudes are **unchanged** from the RMSD form
that preceded it -- they come from inverting each channel's reference barrier and
that arithmetic is untouched -- and the widths are not comparable between the two,
`a` having changed units of meaning from RMSD-squared to this `g`.

**None of the three is a bond fission**, so none uses `fit.coupling.fit_twobody`:
each breaks one bond and forms another, with a real saddle and a real reference
barrier between two real diabats.  `h2o-autoionization` could not use a
crossing-centred fit even in principle -- its two diabats never cross along the
proton coordinate, the products being 9.8 eV uphill in the gas phase.

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

This is also why `fit.coupling.fit_twobody` is not used anywhere in this set even
though it exists for exactly that kind of channel.  A homolysis here would have to
be written `H2O -> OH- + H`, which does not conserve electrons (10 -> 10 + 1) and
would break the bookkeeping the next section depends on.  Adding one needs the
neutral `h1o`, i.e. a different dataset.

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
  a minimum and the width fit -- which asks where the coupling must switch off --
  has nothing to switch off against.  This still applies on `fit_threebody`: it
  quenches at whichever endpoint is nearer the transition state *in its own
  metric*, so an endpoint that is not a minimum is just as useless there.

Term sources:

- `atom` (ACKS2 mu/eta/softness) and `lennardjones`: **HCombustion's O and H
  values, retuned for the liquid.**  `mu`, `soft_amp` and `soft_decay` are
  HCombustion's verbatim; three things moved:

      eta_O      3.7445 -> 2.8084     both scaled by 0.75
      eta_H      7.2846 -> 5.4635
      sigma_H    0.196  -> 0.0        eps_H zeroed with it
      sigma_O    0.296  -> 0.305 nm

  Nothing in the repo regenerates these.  `eta` was set against the dimer well
  and `sigma_O` against the pressure of a 997 kg/m3 box; the reasoning for both
  is under "The hydrogen bond" below.  HCombustion is deliberately *not*
  changed: the two datasets carry their own copies of these lines, so this
  retune is local to water and `tests/test_performance.py`'s references still
  hold.

  **`sigma_O` needs no refit and the other three do.**  `eta` and `sigma_H` are
  inside the `E_nonbonded` that `fit/dissociation.py` solves against, so they
  invalidate every Morse `D` and `r0` here.  `sigma_O` does not: with `sigma_H`
  zero and one oxygen per template, the intramolecular 12-6 is identically zero
  on every molecule in this dataset, so `sigma_O` is invisible to the fit target
  and can be moved on its own.  That is a property of water, not a general one --
  it would not hold for a template with two oxygens.
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

All 3 channels are fittable; margins 0.78-3.24 eV after the refit.  Fastest mode
4289 cm^-1 (h2o O-H), just under the 4400 cap, i.e. **dt = 0.5 fs**.

Refit twice, for the two changes to the repulsion, both of which sit inside the
`E_nonbonded` the fit solves against: `ZBL` acquiring its taper
(`forcefield/zbl.py`), and `forcefield/lj.py` being switched back on.  Neither
cost this dataset anything -- still 3 of 3 channels, still under the cap, both
times -- unlike HCombustion, which lost two to the first and nothing to the
second.  The margins moved because the diabats moved, not because the fit got
worse.

**What the taper bought here is the hydrogen bond, and the 12-6 kept it.**
Before the taper the dimer was **+0.598 eV at 2.91 A with no minimum at any
separation** -- ZBL alone contributed +0.719 eV there, three times the whole
hydrogen bond and the wrong way.  It is now +0.009 eV, and the 12-6 supplies the
wall that stops two waters at 2.40 A from closing, which is why the liquid no
longer comes out a quarter too dense.

## The hydrogen bond

**The bond the taper made room for was still only a third of the real one, and
the cause was the charges.**  Through the full calculator, on the Cs dimer
`tests/test_water_structure.py` builds (donor O-H along the O-O axis, acceptor
bisector tilted 110 deg, both monomers rigid at the experimental geometry, so
every bonded term cancels):

    R_OO (A)     2.60     2.70     2.80     2.85     2.91     3.10     3.60
    before      +0.110   -0.014   -0.060   -0.069   -0.072   -0.066   -0.047
    after       -0.017   -0.131   -0.165   -0.168   -0.164   -0.137   -0.074

against a CCSD(T) reference of -0.218 eV at 2.91 A.  The well is 2.3x deeper and
its minimum has moved in from 2.91 to 2.85 A.  (These are not the numbers in the
table this section used to carry: that one was measured on a different dimer
orientation and reported the nonbonded terms separately.  This one is the total
energy the calculator returns, on the geometry the test file pins, so the two
are not comparable term by term.)

What was wrong was not the repulsion but `ACKS2`.  The shipped parameters put

    q_H = +0.224,  mu = 1.258 D     against a gas-phase experimental 1.855 D

and the electrostatic part of a hydrogen bond goes as `mu^2`, so that is not a
32% error, it is a factor of 2.2 in the binding.  For scale, every fixed-charge
water model sits *above* the gas-phase value, not below it: SPC/E `q_H = +0.4238`
(2.35 D), TIP3P `+0.417`, TIP4P/2005 `+0.5564`.  `eta` is the lever that moves
this -- `soft_amp`, `soft_decay` and `ewald.GAMMA` are not, and scaling any of
them by 2 moves the dipole by under 6% -- so `eta` is scaled by 0.75, giving
`q_H = +0.3040` and 1.711 D.

Two smaller contributions were found at the same time and only one was acted on:

- **`sigma_H = 1.96 A` sits inside the hydrogen bond.**  `lj.py`'s own docstring
  says every water model zeroes it; the Fermi switch flattens the O...H term
  rather than removing it, leaving a +0.009 to +0.015 eV repulsive plateau right
  across 1.5-2.4 A.  Now zeroed, along with `eps_H`.
- **`sigma_O = 2.96 A` is too small for the retuned charges**, and this only
  became visible once they were fixed.  See "What `sigma_O` is for" below.
- **The HOH angle opens to 114.4 deg in MD** against an experimental 104.5.  The
  bonded terms alone minimise at 107.5 (q-force's `theta0` is already 107.2);
  `ACKS2` pushes it to 110.5 because opening the angle separates the two like
  charged H, and thermal anharmonicity does the rest.  The dipole goes as
  `cos(theta/2)`, so this costs a further 12% of it.  **Not addressed** -- it is
  a bonded-parameter question and was left for a pass that moves one variable.

### What `sigma_O` is for

Fixing the charges alone **overshot the cohesion**, and nothing in the dimer
curve shows it -- it is only visible in the liquid.  Pressure of the 64-water box
at 997 kg/m3, 300 K Langevin, measured over the second half of a 5 ps run:

    eta 1.00, sigma_O 2.96      +1320 +/- 121 bar      wants to expand
    eta 0.75, sigma_O 2.96      -1736 +/- 110 bar      wants to contract
    eta 0.75, sigma_O 3.05        -76 +/- 155 bar   <- ships
    eta 0.75, sigma_O 3.10      +1021 +/- 268 bar
    eta 0.75, sigma_O 3.25      +4544 +/- 337 bar
    eta 0.75, sigma_O 3.40     +15330 +/- 263 bar

The first row is a check on the method rather than a result: +1320 bar against a
~22 kbar bulk modulus predicts 942 kg/m3, and `production/density-300K` had
independently measured 900-960 for that force field.  The second row is the
overshoot -- -1736 bar predicts ~1080 kg/m3, so the retune traded 6% too light
for 8% too heavy.

`sigma_O` is the right knob for it because it is nearly orthogonal to the
hydrogen bond: it acts on sixfold O-O contact in the liquid and on a single pair
in the dimer, so moving it from 2.96 to 3.05 A spends 1660 bar and only 0.014 eV
of the dimer well (-0.182 -> -0.168).  For scale, SPC/E uses 3.166 A and
TIP4P/2005 3.1589; 2.96 came from q-force and was never intended to carry a
liquid.

    O-O g(r)            before    after     experiment
    first peak          3.07 A    2.82 A    2.80 A
    peak height         2.27      2.98      ~2.9
    first minimum       4.67 A    4.37 A    ~3.4 A
    O-O within 3.5 A    6.04      6.04      4.3-4.5
    O...H 1.2-2.5 A     1.57      2.01      ~1.8 donated
    pressure         +1320 bar   -76 bar    ~1 bar

The first peak and its height are now right and the pressure is zero at ambient
density.  **The coordination number and the first minimum are not, and they are
the same defect**: 6.04 neighbours inside 3.5 A with a first minimum that never
drops below 0.79 is a close-packed liquid with the right nearest-neighbour
distance, not a tetrahedral one.  Tetrahedrality is a directional, cooperative
property, and the next section is why this model cannot have it.  The 114 deg
HOH angle -- see above, unaddressed -- works the same way.

### The ceiling: there is no cooperativity, at any parameter setting

Charge equilibration moves charge *between atoms*, so the induced dipole is
confined to the span of the interatomic vectors.  Water is planar, so:

                     in-plane   out-of-plane   dipole axis   isotropic
    this model         1.373        0.000          0.482        0.618
    experiment         1.53         1.42           1.47         1.45

The out-of-plane polarizability is **exactly zero and no parameter can lift it**.
The consequence is visible in the liquid: a water there carries a 1.117 D dipole
against 1.111 D for the same geometries isolated -- a +0.6% condensed-phase
enhancement where real water shows +40 to +60%, unchanged at +0.6-0.8% under
every variation tried including `soft_amp x5`.  Hydrogen-bonded chains and rings
do not reinforce one another here; there are only independent pairs.

**That is why the dimer is deliberately left over-bound relative to its own
reference** -- -0.182 eV where the pair reference is -0.218 eV is the *intended*
answer once the sign of the compensation is accounted for, and it is the same
trade every non-polarizable model makes when it carries a 2.35 D dipole.
Lifting the ceiling needs a degree of freedom that is not a charge: atomic
dipoles, or a Drude particle.  `tests/test_water_structure.py::TestPolarizability`
pins the zero so it is not rediscovered.

**The reactive machinery is dormant in neutral water at 300 K**, and that is a
property of this dataset rather than of the settings.  A 64-water box logs
`max_nstates = 1` for every block -- no diabatic state is ever admitted -- at the
production `eps` of 0.05 eV *and* at the 1e-3 default, so nothing is being gated
out.

**The cause is geometric, and it is not the coupling width.**  This paragraph
used to blame the width: `a = 611 1/A**2` on the RMSD form meant the coupling was
live only within ~0.11 A RMSD of the stored transition state.  Widening it
thirteenfold by moving to `compute_threebody` changed nothing here -- the box's
energy is bit-identical to 4 decimal places and `max_nstates` is still 1 -- so the
width was never the binding constraint.  Measured over all 3888 transfer channels
the box enumerates, the closest any proton comes to a transfer geometry is

    g = 0.869 A**2:   ra = 0.958 A,  rb = 2.188 A,  angle(D-H-A) = 98 deg

against a reference `ra0 = 1.210`, `rb0 = 1.290`, `t0 = 180 deg`.  The proton is
still firmly on its own oxygen, the acceptor is 0.9 A too far, and the transfer is
nowhere near linear.  Admitting that geometry at `eps = 1e-3` would need
`a < 9.6 1/A**2`, which is a coupling live at ordinary hydrogen-bond geometries
and would be wrong.  So the dormancy is right -- autoionization is rare -- but it
means a neutral-water trajectory is effectively fixed-topology, and any density
measured from one is a fixed-topology number.  The hop channels need an ion
present to do anything.

**What the transfer form does buy is immunity to the spectators.**  With the three
transfer atoms held exactly at `h2o-autoionization`'s transition state and only
the other three displaced, the two forms behave completely differently:

    spectator displacement    0.05 A    0.10 A    0.20 A    0.40 A
    RMSD width, |V| (eV)       2.05    8.4e-03   1.7e-05   7.4e-14
    triangle width, |V| (eV)   4.14      4.14      4.14      4.14

A 0.1 A spectator jiggle is less than thermal motion at 300 K, so the RMSD form
was switching the coupling off for reasons that have nothing to do with the
reaction. That is the defect this fixed; the dormancy above is a separate fact.

Verified on a packed 64-water box (12.43 A cube, 997 kg/m^3): forces finite, NVE
drift 0.002 meV/atom over 40 x 0.5 fs, ~2.5 s per force call.

## Note on the coupling width

`a` is not comparable across the two forms and the numbers above are not a
before/after.  On the RMSD form it multiplied a squared RMSD over all `3N`
coordinates; on `threebody` it multiplies `g`, a sum of three squared side-length
deviations of one triangle.  The fitted values happen to fall by 5-14x, but the
metrics differ, and a wider-looking `a` on a narrower metric can be a tighter
coupling.  What *is* comparable is the amplitude, and it is unchanged to every
digit: `fit.coupling.fit_threebody` reuses `fit_amplitude` verbatim, because the
transition state is still where `V = A`.

To reproduce the RMSD-width fit for comparison:

```sh
uv run python scripts/fit.py -r datasets/Water/Water.json --rmsd-width
```

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

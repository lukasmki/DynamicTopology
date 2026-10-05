# Dynamic Topology Molecular Dynamics

## Running on a CPU node

A force call at the sizes the production sweeps run -- 192-atom water, 200-atom
H2/O2 -- is single-core work: EVB bookkeeping in Python plus a few
hundred-square matrices.  A Perlmutter CPU node is used by running one
trajectory per core, which is what `production/*/submit.slurm` does with a Slurm
array on the `shared` QOS.  Threading one trajectory does not help at this size
and, left to its defaults, actively hurts.

**One BLAS thread per run.**  OpenBLAS starts a thread per CPU it can see, and
numpy and scipy each bring their own copy, so a Python process on a full node
starts 2 x 63 threads for matrices too small to split.  Several runs sharing a
node -- `run_all_local.sh`, an interactive allocation -- then fight over the
cores.  `scripts/nvt.py` and `scripts/npt.py` default `OMP_NUM_THREADS` and
`OPENBLAS_NUM_THREADS` to 1 unless they are already set, and the launchers in
`production/` export them.  Measured on Perlmutter (AMD EPYC 7763), 192-atom
water, seconds per force call:

    OPENBLAS_NUM_THREADS   1      2      4      8      16     32     unset
    one run                0.110  0.101  0.095  0.103  0.103  0.126  0.225
    four runs at once      0.111                                      4.2

(a 64-CPU slice, i.e. 32 cores; "four runs at once" is `run_all_local.sh 4`).
Unpinned, four runs sharing the slice are **38x slower** each; pinned, they
run as fast as one alone.

A Slurm array task with `--cpus-per-task=1` was already safe -- OpenBLAS sizes
its pool from the CPU mask Slurm hands it -- so the sweeps as submitted were
not affected; anything sharing a larger allocation was.

**Reactive frames cost far more than equilibrium ones, and that is bookkeeping.**
On a hot 3000 K H2/O2 frame, blocks reach the 128-state cap and the basis
closure (`basis.EVBBasis._close`) was over 90% of the force call.  It has been
made faster without changing any number it produces -- energies, forces,
virials, block weights and the carried topology are bit-identical over
sequential calls on every dataset, checked against the previous code:

    box (one core, mean of 2-3 calls)    before     after     (s / call)
    H2/O2, 3000 K, hottest frame          5.05       2.40
    H2/O2, 3000 K, earlier frame          4.18       1.88
    H2/O2 mix-n100-d250                   0.210      0.179
    H2/O2 mix-n100-d30                    0.217      0.202
    water, 64 molecules                   0.246      0.232

  - The product of a candidate channel is no longer built (a copy of the whole
    block graph) before checking whether it is a state already admitted; its
    state key is the parent's edge set with the reaction's bond changes
    applied.  Nine in ten candidates on a hot box were copies thrown away.
  - The bonded energy of a reacting fragment is memoized per force call on its
    node order and edge set, which fix the float sum exactly; the same pair is
    screened again from every state of a block that leaves it alone.
  - Molecule signatures and those memo keys are read straight off the
    underlying graph dicts rather than through networkx's filtered subgraph
    views.  The signature is sorted, so its visit order cannot show; the
    memo key's node order follows `FilterAtlas.__iter__`'s own rule, and
    `basis._CHECK_KEYS` asserts it against networkx.
  - Each admitted state's block-local term arrays are its own vectorized
    arrays renumbered through a lookup table, rather than its whole term list
    remapped in Python and vectorized again.
  - The Ewald and minimum-image kernels keep the radial derivative `_screened`
    computes alongside the kernel, instead of a second `erf`/`exp` pass over
    every pair for the forces.

`tests/test_performance.py` guards the two fast paths that read networkx's
internals against networkx's own answers.

**Large boxes: the charge solve, then the pair kernels.**  Replicating the
64-water box (one thread, fixed cell, seconds per call):

    atoms                    192     1536     5184
    before                   0.11    5.8      77     (8.2 GB)
    now                      0.081   2.19     22.0   (6.4 GB)
    now, OPENBLAS_NUM_THREADS=8        1.72     12.8   (with `lj_cutoff` 10 A)

What changed, each checked against the previous code over the same sequential
calls on every dataset (`$SCRATCH/hpc-opt-work/tol_check.py`):

  - **The Kohn-Sham potentials are eliminated from ACKS2's linear system**,
    one molecule at a time (`acks2._Piece.system`): `2n + 2m` unknowns become
    `n + m`, an eighth of the LU (2.9 s to 0.38 s at 1536 atoms).  A molecule
    whose softness is nearly disconnected -- a bond the topology still holds
    stretched past 2-4 A, which a hot frame has a handful of -- keeps its
    potentials as unknowns: eliminating them would add entries of order
    `1/X ~ e^40` to the hardness and cancel away the precision of the
    molecule's other modes (it cost 2.5e-7 eV/A on the 3000 K frame before
    they were kept).
  - **The reciprocal half of the Ewald kernel is one `dsyrk`**, half the
    flops of the two products it replaces, and **its force contraction uses a
    thin factor of the weight matrix** -- the mean charges plus one column per
    state of each block -- rather than the dense `N x N` weight (0.9 s to
    0.01 s at 1536 atoms).
  - **One minimum-image geometry per force call**, shared by ACKS2, ZBL and
    the 12-6, which each used to build their own `N x N x 3` copy (since
    replaced by a neighbour list; see *Neighbour lists* below); and the
    pair contractions' virials are one matrix product instead of a
    three-operand `einsum`.
  - **ZBL sums only the pairs within `taper_radius + 45 taper_width`**
    (6.9 A), past which the taper is below e^-45.

The same changes take the production sizes down too: 0.131 to 0.100 s per
call for the 64-water box, 0.18 to 0.13 s for H2/O2 `mix-n100-d250`, 2.5 to
2.1 s for the hottest 3000 K frame (eight runs sharing a node, as above).

**Numbers are reproduced to rounding, not to the bit.**  Energies agree with
the previous code to 4e-12 eV, forces to 2e-12 eV/A, block weights to 3e-13,
and the carried topology exactly.  The two `test_energy_matches_reference`
cases hold the total to `REFERENCE_TOL = 1e-8` eV rather than to the bit --
which also fixes their old failure on Perlmutter, where the pinned numbers
were 1e-12 off.  `tests/test_acks2_fragment.py` checks the reduced system
against the full one, with and without a molecule kept explicit.

**BLAS threads now help a large box, and change nothing that matters**: every
thread count agrees with one thread to rounding.  Give a large box a few cores
(`OPENBLAS_NUM_THREADS=8`); at production size they are still better spent on
more trajectories.

**An optional cutoff for the 12-6 (`global_params.lj_cutoff`).**  By default
the 12-6 is summed over every minimum-image pair, so its reach is the cell:
half a cell along an axis, more toward a corner.  That makes the energy depend
on the box size -- replicating the 64-water box 2x2x2 lowers it by 3.6 meV per
molecule, nearly all of it dispersion the small box never sees.  With
`"lj_cutoff": 6.0` (at most half the cell's shortest perpendicular width),
each pair is switched smoothly to zero over the last `lj_cutoff_width` (1 A)
and a uniform-density tail correction adds what lies beyond; the 192- and
1536-atom boxes then agree to 0.2 meV per molecule.  It is a change to the
model, not to the rounding -- the 64-water box moves by -4.3 meV per molecule
-- so it is off unless a dataset sets it, and the datasets were fitted
without it.  NVE over 1 ps (0.5 fs steps, 300 K) conserves energy equally
well either way: drift -0.07 meV/ps/atom without it, -0.08 with it.
`tests/test_lj_cutoff.py` holds its forces and virial to finite differences.

**An iterative charge solve for large boxes
(`global_params.charge_solver = "iterative"`).**  The direct solve forms the
periodic kernel as a dense matrix and LU-factors ACKS2's system, `O(N^2)`
memory and `O(N^3)` time.  The iterative one never forms the kernel:

  - **The kernel is an operator** (`ewald.EwaldOperator`): the real-space
    half cut off at `real_space_cutoff` (9 A, or half the cell) as a sparse
    matrix, and the reciprocal half by smooth particle-mesh Ewald
    (`forcefield/pme.py`, order-8 B-splines, the mesh sized from `accuracy`).
    Applied to charges it agrees with the dense matrix to ~1e-8 of its size,
    and so do its force and virial contractions.
  - **The environment's system is solved by conjugate gradients**
    (`acks2._EnvironmentSolver`), projected onto charges that keep every
    molecule's sum and preconditioned molecule by molecule, to a relative
    residual of `solver_tolerance` (1e-10), each call starting from the
    previous call's charges: four iterations on an MD step.  A multi-state
    block's coupling to the environment is one solve per block atom; its own
    states stay dense and small.
  - **The admission gate** reads the few kernel columns a fragment needs
    instead of the matrix.
  - **Any call it cannot take falls back to the direct solve**: a molecule
    whose softness is nearly disconnected (`acks2.SOFTNESS_FLOOR`) keeps its
    Kohn-Sham potentials as unknowns, which the projected solve does not
    handle; so does a cell that is not fully periodic.  (`PointCharge`,
    which solves nothing, takes the same operator; see below.)

Water boxes, one thread, seconds per call (direct -> iterative):

    atoms            192      1536     5184
    direct           0.081    2.24     22.1
    iterative        0.062    1.35     10.8
    max |dF|, eV/A   4e-7     2e-7     2e-7     (forces up to ~4 eV/A)
    |dE| / E         5e-9     2e-9     2e-9

NVE over 1 ps (64 waters, 0.5 fs, 300 K) conserves energy as well as the
direct solve: drift -0.06 meV/ps/atom against -0.07.  It is opt-in because
it is not the same arithmetic -- PME and the residual move the forces by
~1e-7 eV/A -- and every dataset was fitted with the direct solve.
`tests/test_iterative_solver.py` holds PME to the direct reciprocal sum, the
operator to the matrix, the solve (with and without a multi-state block) to
the direct solve, its forces to its own energy, and the fallback to the
direct numbers exactly.

**Neighbour lists for the pair terms (`forcefield/neighbors.py`).**  Every
pair term used to read one shared minimum-image geometry, the displacement of
every pair of atoms -- `N x N x 3`, 0.9 GB at 5184 atoms -- even where it cut
off at a few Angstrom.  Now each force call builds one `Geometry`, and each
term asks it for what it reads:

  - **pairs inside a cutoff**, from a neighbour search: a periodic k-d tree
    (`scipy.spatial.cKDTree`) for an orthorhombic cell, a cell list for any
    other, built once at the widest cutoff asked for and cut down for the
    rest.  ZBL (6.9 A), the 12-6 under `lj_cutoff`, and the real-space half
    of the iterative solve's Ewald operator read these;
  - **given pairs** -- within a molecule, an exclusion, a fragment the gate
    screens -- measured one by one, for ACKS2's softness and isolated
    references and the point-charge exclusions;
  - **the dense arrays**, only for a term that is all-pairs by construction:
    the direct solve's dense Ewald kernel, open boundaries, and the 12-6
    without `lj_cutoff`.

**One list, kept between calls: `neighbor_radius` and `neighbor_skin`.**
`System` keeps a `NeighborList` for its lifetime.  Each call, every term's
cutoff is cut from one list searched at `global_params.neighbor_radius` -- by
default the widest cutoff in force (ZBL's 6.9 A, `lj_cutoff`, the iterative
solve's real-space cutoff); set explicitly, a term asking for more raises --
plus `neighbor_skin`.  The list is searched again only once an atom could have
crossed the skin: twice the largest displacement since the search plus the
cell's strain times the list's reach exceeding the skin, so wrapping an atom
back into the cell does not count and an NPT step need not either.  The skin
is shortened where the reach would pass half the cell.  Neither parameter
changes which pairs a term sums -- energies are identical to the last digit
printed over 8 MD-sized steps -- so neither is a model change; the default
skin is 0, which searches every call.  Cutoff 6 A and the iterative solve,
seconds per call spent finding pairs (of 5.5 and 22.7 s per call):

    neighbor_skin       0       1 A     2 A
    5184 atoms          0.23    0.15    0.19
    12288 atoms         0.56    0.38    0.47     (one search in 9 calls)

Small, because the k-d tree an orthorhombic box takes is already cheap, and a
kept list still measures every listed pair each call, more of them the wider
the skin.  A triclinic box, whose cell-list search is six times slower,
stands to gain more.

Every pair's displacement is the same arithmetic the dense geometry did, and
the search returns exactly the pairs `rij < rc` selected
(`tests/test_neighbors.py`, triclinic cells included), so this changes the
order of sums and nothing else.  Against the previous code over the same
sequential calls, on every dataset under each of default, `lj_cutoff`,
`iterative` and both: energies agree to 1e-12 eV, forces to 3e-14 eV/A,
block weights and topologies exactly.  (The one exception is the first call
of a box replicated exactly, where pairs sit at precisely half the cell: the
dense contraction gave both atoms of such a pair the same displacement, a net
force of 2e-3 eV/A on the box, and the pair list gives them opposite ones.)

The 12-6 is now a sum over a pair list in both of its paths -- one topology,
and the per-state one, where each varying block keeps its states' energies
for just the pairs that touch it -- and its tail correction, a sum over every
ordered pair of atoms that depends only on their parameters, is summed over
parameter *types* instead of spread over an `N x N` matrix.  ACKS2's
intramolecular pairs and per-molecule systems no longer pass through `n x n`
arrays either.

**`PointCharge` takes the operator under `charge_solver = "iterative"`.**  It
used to fall back to the dense kernel.  It solves nothing, and every product
it takes with `K` is with a charge vector or a multi-state block's few
columns, so it now uses `ewald.EwaldOperator` as ACKS2 does and contracts its
weight as a factor.  The same energy to the lattice sums' accuracy (4e-6 eV
on 619, forces to 5e-7 eV/A); NVE drift -0.09 meV/ps/atom against -0.08 to
-0.09 dense.  A single-state block's diagonal correction, which `System` only
reports, is reported as zero under the operator, as ACKS2 reports it: it
would cost a PME pass per atom.

Water boxes, one thread, `lj_cutoff` 6 A and the iterative solve, seconds per
call (and peak memory):

    atoms                    192      1536      5184           12288
    ACKS2, before            0.059    1.17      8.3  (5.0 GB)  38.3 (25 GB)
    ACKS2, now               0.058    0.90      5.5  (3.4 GB)  22.2 (16 GB)
    point charges, before    0.092    2.49      22.2 (4.9 GB)
    point charges, now       0.053    0.89      5.4  (3.3 GB)  22.1 (16 GB)

The nonbonded terms in those calls -- ACKS2's setup, solve and forces, the
12-6, ZBL and the neighbour search -- now cost 0.53 s at 5184 atoms and 1.3 s
at 12288, linear in the atoms.  The default settings (direct solve, no
`lj_cutoff`) are all-pairs by construction and run as before (22 s at 5184).

**What remains `O(N^2)` is the EVB basis**, 23 of the 24 s per call at 12288
atoms: the molecule separations the reaction network screens on
(`ReactionSet._molecule_separations`, 8.5 s, and the block-wide
`squared_separations` `basis.py` builds), and the per-molecule bookkeeping of
the closure.  The first is a pair search within `bimol_cutoff`, which
`Geometry.pairs` already provides.

**Hot reactive frames are bookkeeping, and the closure now shares it between
states.**  On a 3000 K H2/O2 frame a block reaches the 128-state cap, and the
states of a block differ from one another by a molecule or two -- yet the
closure used to work out each state from scratch.  Now (seconds per call, the
eight-case comparison above):

    H2/O2 3000 K, hottest frame     2.10 -> 0.98
    H2/O2 3000 K, earlier frame     1.72 -> 0.83
    H2/O2 mix-n100-d250             0.134 -> 0.131

  - **Each state's reactions** come from `ReactionSet.state_channels`, which
    lists exactly what `get_network(...).reactions()` would -- the same
    objects, in the same order, which matters because at the state cap that
    order decides which states are admitted -- without building the
    MultiGraph and its subgraph views, and with the molecule separations cut
    from one distance matrix per block instead of the minimum-image
    arithmetic per state.
  - **Each channel is worked out once per force call** (`_channel_info`): its
    bond changes, its coupling and whether it clears the `eps` gate belong to
    the channel, whichever state reaches it -- 55k channel visits for a few
    hundred channels on the hot frame.  So is its weight wherever the
    electrostatic gap is identically zero (HCombustion): the fragment is whole
    molecules matched by signature, bonded the same from every parent.
  - **The fragment's memo key** is read off the parent graph's dicts
    (`_fragment_key`) rather than through a subgraph view per channel.
  - **A state's term arrays** are its molecules' cached arrays concatenated
    (`ReactionSet.assign_terms`), and **its bonded energy is its molecules'**,
    from the molecule cache, rather than a block-wide evaluation per state.

Everything but the last is bit-identical to the code before it; the last sums
the same per-molecule terms in a different order (1e-12).
`tests/test_performance.py` runs the hot frame with `_CHECK_CHANNELS` and
`_CHECK_KEYS` on, which assert each fast path against what it replaced, and
compares `assign_terms` with `set_terms` array by array.

ACKS2's per-state setup shares its per-molecule work too: each molecule's
isolated minimum and Kohn-Sham block are cached across a block's states, keyed
by its atoms and their parameters (`_Piece.memo`), which sums the isolated
minima in a different order (1e-12).  It is worth about 5% (0.98 -> 0.93 s on
the hottest frame) -- the eigen- and linear solves it saves were smaller than
the per-state assembly around them.  What remains is spread thin: each
state's LU, the per-state term assembly, and the product copies.

Usage
=====

All quantities in memory and at the API boundary are in ASE units: positions in
Å, energies in eV, forces in eV/Å and stress in eV/Å³. The ``.jsonl``
parameter files on disk use the same units (see :doc:`concepts`).

Loading a reaction set
----------------------

A simulation starts from a *reaction set*: a dataset manifest listing parameter
templates for every molecule and a transition-state ensemble for every reaction
channel (see :doc:`datasets`).

.. code-block:: python

   from DynamicTopology.core import ReactionSet

   reaction_set = ReactionSet("datasets/HCombustion/HCombustion.json")
   print(reaction_set.params)           # the dataset's global force field parameters
   print(reaction_set.data.n_reactions) # stored reactions, both directions

Loading a manifest also *activates* its ``global_params`` process-wide (see
:ref:`global-params`). Two datasets whose global parameters disagree therefore
cannot be loaded into the same process; the second load raises and names the
fields that differ.

Single-point energies
---------------------

:class:`DynamicTopology.ase.DynamicTopology` is an ASE calculator. It perceives
the initial bond topology from the ``Atoms`` object when it is constructed:
from an explicit ``atoms.info["connectivity"]`` list if one is present
(extended-XYZ files written by the MD scripts carry one), otherwise from
covalent-radius distance cutoffs.

.. code-block:: python

   from ase import io
   from DynamicTopology.ase import DynamicTopology

   atoms = io.read("examples/mix-n100-d250.xyz")
   atoms.calc = DynamicTopology(atoms, reaction_set)

   energy = atoms.get_potential_energy()  # eV
   forces = atoms.get_forces()            # eV/Å, shape (natoms, 3)
   stress = atoms.get_stress()            # eV/Å³, periodic cells only

Constructor options:

``bimol_cutoff`` (Å, default ``4.0``)
   A bimolecular reaction channel is considered only when the minimum interatomic
   distance between the two molecules (minimum image under PBC) is below this.

``evb`` (dict, optional)
   Passed through to :class:`~DynamicTopology.basis.EVBBasis`: ``eps`` (the
   stabilization in eV a channel must supply to enter the basis, default
   ``1e-3``), ``switch_width`` (width of the admission ramp, default ``eps``),
   ``max_states`` (hard cap on basis size, default ``128``) and ``max_depth``.
   These change the potential energy surface, so record them with any result.

Beyond the ASE properties, the calculator exposes per-call diagnostics:

.. code-block:: python

   calc = atoms.calc
   calc.diagnostics["energy_bonded"]     # EVB ground-state bonded energy
   calc.diagnostics["energy_nonbonded"]  # electrostatics
   calc.diagnostics["energy_zbl"]        # short-range screened-nuclear repulsion
   calc.diagnostics["energy_lj"]         # switched 12-6
   calc.diagnostics["blocks"]            # per-EVB-block states and weights
   calc.topology_changed                 # did the last call change any bond?
   calc.system.topology                  # the topology carried to the next step

Molecular dynamics
------------------

Because the calculator updates its own topology after every force call, any ASE
integrator runs reactive dynamics without further setup:

.. code-block:: python

   from ase import units
   from ase.constraints import FixCom
   from ase.md import Langevin
   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

   atoms.set_constraint(FixCom())
   MaxwellBoltzmannDistribution(atoms, temperature_K=2000)

   dyn = Langevin(
       atoms,
       timestep=0.5 * units.fs,
       temperature_K=2000,
       friction=0.005 / units.fs,
       fixcm=False,
   )
   dyn.run(1000)

.. important::

   ``atoms.info["connectivity"]`` is **not** updated by ASE as the simulation
   runs; the evolving bonding lives on ``atoms.calc.system.topology``. To write
   frames that record the current bonding, copy the topology's edges into
   ``atoms.info["connectivity"]`` before writing, as ``record_topology`` in
   ``scripts/nvt.py`` does.

The timestep has to resolve the fastest vibrational mode of the fitted surface
(roughly 15 steps per period). The fast-forces refit report prints that mode and
the implied timestep; re-check it after every refit.

Periodic cells and stress
-------------------------

For a periodic ``Atoms`` object, electrostatics are summed over images with an
Ewald sum and the calculator reports ``stress`` (the virial divided by the cell
volume), which is what ASE barostats require. A non-periodic system reports no
stress. See ``scripts/npt.py`` for a Berendsen NPT run.

Evaluating one topology
-----------------------

:func:`DynamicTopology.forcefield.evaluate.evaluate` returns the energy, forces
and virial of a single fixed topology, summed exactly as the reactive calculator
sums one diabatic state, with the parts broken out. It is the entry point for
scoring a template on its own:

.. code-block:: python

   from DynamicTopology.forcefield.evaluate import evaluate

   water = reaction_set.data.by_formula("H2O")
   result = evaluate(water.atoms, water.terms)
   result.energy          # total, eV
   result.bonded          # QForce terms, reference shift and exclusions
   result.electrostatics  # ACKS2 or point charges
   result.zbl, result.lj  # the two repulsion terms
   result.charges         # per-atom charges, e

Overriding global parameters
----------------------------

:func:`DynamicTopology.forcefield.params.use` temporarily overrides the active
global parameters, deriving from the currently active set:

.. code-block:: python

   from DynamicTopology.forcefield import params

   with params.use(switch_radius=1e6):   # switch the 12-6 off entirely
       print(evaluate(water.atoms, water.terms).lj)   # 0.0

Values changed this way are not the values the dataset was fitted at, so the
resulting surface is no longer the fitted one.

Fixed-state EVB
---------------

:class:`DynamicTopology.ase.EVB` is a simpler, non-reactive alternative: it
enumerates the diabatic states reachable from the initial topology once, at
construction, and couples them with an empirical geometric-mean coupling. There
is no network rebuild and no topology update during the run.

.. code-block:: python

   from DynamicTopology.ase import EVB

   atoms.calc = EVB(atoms, reaction_set)

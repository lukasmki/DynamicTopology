Concepts
========

This page summarizes how a force call is assembled. The equations each term
evaluates are documented in ``src/DynamicTopology/forcefield/README.md``.

One force call
--------------

:meth:`DynamicTopology.ase.DynamicTopology.calculate` delegates to
:meth:`DynamicTopology.system.System.calculate`, which:

1. **Builds the diabatic basis.** :class:`~DynamicTopology.basis.EVBBasis`
   starts from the current topology and repeatedly asks the
   :class:`~DynamicTopology.core.ReactionSet` which reactions apply
   (:meth:`~DynamicTopology.core.ReactionSet.get_network`). A molecule's
   unimolecular channels always apply; a bimolecular channel applies only when
   the two molecules are within ``bimol_cutoff``. Applying a reaction to a
   state gives a new state. The closure is split into independent *blocks*,
   groups of molecules that share reactive states.
2. **Scores each state.** A state's diagonal energy is its bonded energy from
   :class:`~DynamicTopology.forcefield.qforce.QForce`, using the parameter
   template of each molecule in it. The off-diagonal couplings come from
   :class:`~DynamicTopology.forcefield.coupling.EVBCoupling` and are centred on
   each reaction's stored transition-state geometries.
3. **Diagonalizes.** Each multi-state block's EVB Hamiltonian is diagonalized
   with :func:`numpy.linalg.eigh`. Energy, forces and virial are the
   Hellmann-Feynman average over the ground-state eigenvector.
4. **Carries the topology forward.** The state with the largest ground-state
   weight (with a small hysteresis against the incumbent) becomes that block's
   topology. The merged topology is returned and fed back into the ``System``
   for the next step.
5. **Adds the nonbonded terms.** Electrostatics, the short-range screened-nuclear
   repulsion (:class:`~DynamicTopology.forcefield.zbl.ZBL`) and the switched
   12-6 (:class:`~DynamicTopology.forcefield.lj.LennardJones`) are evaluated
   once for the whole system. Only their intramolecular exclusions depend on the
   bonding pattern, and those are applied per state.

Electrostatics
--------------

Two electrostatic models are available, selected by the dataset's
``global_params.electrostatics``:

``"acks2"`` (default)
   Charge equilibration (:class:`~DynamicTopology.forcefield.acks2.ACKS2`),
   solved once per force call and identical on every diabatic state.

``"pointcharge"``
   Fixed charges carried by each template's ``charge`` terms
   (:class:`~DynamicTopology.forcefield.pointcharge.PointCharge`). A
   hydronium's +1 moves with its proton, so the Coulomb energy differs between
   states and sits on the EVB diagonal. Blocks interact through each other's
   ground-state-averaged charges and are iterated to self-consistency.

Under full periodicity both are summed over images with an Ewald sum
(:mod:`DynamicTopology.forcefield.ewald`). Otherwise the minimum-image kernel
is used.

Core objects
------------

:class:`~DynamicTopology.core.Topology`
   A :class:`networkx.Graph` of bonds, with an optional ``Atoms`` and term list.
   Its identity is the Weisfeiler-Lehman hash of the graph over atomic numbers.
   :meth:`~DynamicTopology.core.Topology.molecules` yields connected components
   that keep their *global* atom indices.

:class:`~DynamicTopology.core.Reaction`
   A reactant and a product ``Topology`` plus a list of transition-state
   geometries (the coupling ensemble).
   :meth:`~DynamicTopology.core.Reaction.apply` maps the template onto live atom
   indices and returns the product topology.

:class:`~DynamicTopology.core.ReactionNetwork`
   A multigraph whose nodes are molecules and whose edges are applicable
   reactions.

:class:`~DynamicTopology.core.ReactionSet`
   The query engine over a loaded dataset
   (:class:`~DynamicTopology.core.reactionset.ReactionSetData`). It maps
   templates onto a live system's atom indices and caches the results.

The term format
---------------

A *term* is a plain dict:

.. code-block:: python

   {"type": "bond", "atoms": {"a1": 0, "a2": 1}, "kwargs": {"D": ..., "r0": ..., "k": ..., "h": ...}}

``Topology.set_terms`` transposes a list of terms into a vectorized
``term_dict`` of ``{type: {"atoms": (n, k) int array, "kwargs": {name: (n,) array}}}``,
which is what the force fields consume. A force field dispatches each term type
to a method named ``compute_<type>`` and ignores types it has no method for, so
adding a new functional form means adding one ``compute_*`` method that returns
``(energy, forces, virial)``.

Units
-----

.. list-table::
   :header-rows: 1

   * - Where
     - Length
     - Energy
   * - In memory: every force field, term and ``ForceFieldParams``
     - Å
     - eV
   * - ``.jsonl`` files on disk
     - Å
     - eV
   * - Results returned to ASE
     - Å
     - eV (forces eV/Å, stress eV/Å³)
   * - q-force XML and OpenMM (fast-forces import and export only)
     - nm
     - kJ/mol

A ``.jsonl`` stores every parameter exactly as it is held in memory, so reading
and writing one converts nothing. nm and kJ/mol appear only where fast-forces
imports q-force XML or exports to OpenMM, and both conversions go through the
table in :mod:`DynamicTopology.io.units`. The ACKS2 ``atom`` block, ``charge``
and the EVB couplings are the same numbers in either unit system. The virial is
``dE/d(strain)``, a 3×3 array in eV.

Files written before 2026-09-23 stored the bonded and 12-6 parameters in nm and
kJ/mol. :func:`~DynamicTopology.io.json.read_jsonl` refuses a bond ``r0`` below
0.5, which only such a file can have. Convert one row at a time with
:func:`~DynamicTopology.io.units.term_from_openmm`.

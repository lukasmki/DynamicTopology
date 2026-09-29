Concepts
========

This page summarizes how a force call is assembled. The equations each term
evaluates are in :doc:`reference`.

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
   state gives a new state. A channel is admitted when mixing with it lowers
   the energy by more than ``eps``, judged on the gap between the two states'
   bonded energies plus the change in their fixed-charge electrostatics
   (:class:`~DynamicTopology.forcefield.electrostatics.ElectrostaticGap`). The
   closure is split into independent *blocks*, groups of molecules that share
   reactive states.
2. **Scores each state.** A state's diagonal energy is its bonded energy from
   :class:`~DynamicTopology.forcefield.qforce.QForce`, using the parameter
   template of each molecule in it, plus its electrostatic energy (see
   below): every state carries its own charges. The off-diagonal couplings come from
   :class:`~DynamicTopology.forcefield.coupling.EVBCoupling` and are centred on
   each reaction's stored transition-state geometries.
3. **Diagonalizes.** Each multi-state block's EVB Hamiltonian is diagonalized
   with :func:`numpy.linalg.eigh`. Energy, forces and virial are the
   Hellmann-Feynman average over the ground-state eigenvector. Blocks see each
   other's charges, so when two or more blocks are multi-state they are
   diagonalized in turn until no weight changes.
4. **Carries the topology forward.** The state with the largest ground-state
   weight (with a small hysteresis against the incumbent) becomes that block's
   topology. The merged topology is returned and fed back into the ``System``
   for the next step.
5. **Adds the nonbonded terms.** The electrostatic energy, forces and virial
   are evaluated once at the ground-state weights. The short-range
   screened-nuclear repulsion (:class:`~DynamicTopology.forcefield.zbl.ZBL`)
   and the switched 12-6 (:class:`~DynamicTopology.forcefield.lj.LennardJones`)
   do not depend on the bonding, so they are evaluated once for the whole system
   and added on top. Only their intramolecular exclusions depend on the bonding
   pattern, and those are applied per state.

Electrostatics
--------------

The electrostatics are computed per diabatic state: each state carries its own
charges, so a proton transfer moves its +1, and the surrounding charges decide
which proton position is lower. Two models are available, selected by the
dataset's ``global_params.electrostatics``:

``"acks2"`` (default)
   Fragment charge equilibration (:class:`~DynamicTopology.forcefield.acks2.ACKS2`).
   Each state solves its own ACKS2 problem:

   * **Reference charges.** Each template's ``atom`` terms may state a
     reference charge ``q0`` per atom (zero if absent), which carries the
     molecule's formal charge.
   * **Charge stays within a molecule.** Charge redistributes only inside each
     molecule, so every molecule keeps exactly its formal charge. Charge moves
     between molecules only when the bonding changes.
   * **Energy relative to isolated molecules.** A state's energy is its ACKS2
     minimum less each molecule's minimum in isolation. A lone template
     therefore contributes zero electrostatic energy, and its gas-phase
     electrostatics belong to its bonded terms.

   The atoms outside every reacting block are solved once. Each state then
   needs only a small solve over its own block, and the rest of the system
   still responds to that state exactly.

``"pointcharge"``
   Fixed charges carried by each template's ``charge`` terms
   (:class:`~DynamicTopology.forcefield.pointcharge.PointCharge`). This is the
   ACKS2 model with zero polarizability, and needs no solve.

In both, a state's electrostatic energy sits on its EVB diagonal, and blocks
interact through each other's ground-state-averaged charges. Under full
periodicity the kernel is summed over images with an Ewald sum
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

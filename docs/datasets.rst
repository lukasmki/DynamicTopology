Datasets
========

A dataset is a directory with a JSON manifest, per-molecule parameter templates
and per-reaction transition-state ensembles. Three ship with the repository:

``datasets/HCombustion``
   Hydrogen combustion: eight species (H2, O2, OH, H2O, HO2, H2O2, O, H) and
   the reaction channels between them, computed at ωB97X-V/cc-pVTZ.

``datasets/Water``
   Proton transfer in bulk water (the Zundel and hydroxide hops, plus
   autoionization), computed at ωB97X-V/aug-cc-pVTZ, with ACKS2
   electrostatics.

``datasets/Water-fixed-pc``
   The same chemistry with fixed per-template point charges
   (``electrostatics = "pointcharge"``), so that the charge of hydronium and
   hydroxide moves with the proton.

Datasets are produced and refit with fast-forces
(``fast-forces refit <manifest>``), which writes into these directories.

Layout
------

.. code-block:: text

   datasets/HCombustion
   ├── HCombustion.json
   ├── molecules
   │   ├── mol_01.jsonl
   │   ├── mol_01.xyz
   │   └── ...
   └── reactions
       ├── rxn_01.jsonl
       ├── rxn_01.xyz
       └── ...

The manifest
------------

.. code-block:: json

   {
       "name": "HCombustion",
       "description": "...",
       "version": "1.0.0",
       "global_params": {
           "bond_asymptote": 1.0,
           "taper_radius": 1.5,
           "taper_width": 0.12,
           "switch_radius": 2.2,
           "core_fraction": 0.4,
           "exclusion_depth": 3,
           "exclude_coulomb": true,
           "gamma": 2.0
       },
       "molecules": [
           {"id": 1, "smiles": "[H][H]", "path": "molecules/mol_01"}
       ],
       "reactions": [
           {"id": 5, "smiles": "[H][H]>>[H].[H]", "path": "reactions/rxn_05"}
       ]
   }

Each entry's ``path`` has no extension. Loading pairs ``<path>.xyz`` with
``<path>.jsonl``:

* **Molecules**: the ``.xyz`` is one geometry. Its energy, if present, is the
  atomization energy used to put every template on a common reference scale.
  The ``.jsonl`` holds the parameter terms, one per line.
* **Reactions**: the ``.xyz`` holds several frames, read as reactant,
  transition-state frames, then product. The ``.jsonl`` holds the coupling
  terms.

The ``smiles`` field is documentation only. A molecule's identity is the
Weisfeiler-Lehman hash of its bond graph, so nothing reads the SMILES string.
Every reaction is stored in both directions.

Intramolecular exclusion terms (``exclusion``, ``zblexclusion``,
``coulombexclusion``) are not stored in the ``.jsonl``. They are derived from
the bond graph at load time.

.. _global-params:

Global parameters
-----------------

``global_params`` holds the force field constants the dataset was fitted at:
taper and switch radii, exclusion depth, the Morse asymptote, the charge-kernel
width and the electrostatics model. Fields left out take the defaults in
:class:`~DynamicTopology.forcefield.params.ForceFieldParams`.

:meth:`ReactionSet.load <DynamicTopology.core.ReactionSet.load>` activates
these before it reads a single template, and every force field reads them
through :func:`~DynamicTopology.forcefield.params.active` at call time.
Changing a value in a manifest invalidates that dataset's fitted ``.jsonl``
files. Refit them with fast-forces afterwards.

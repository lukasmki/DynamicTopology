DynamicTopology
===============

**DynamicTopology** is a reactive molecular dynamics force field in which the
bond topology is allowed to change during a simulation. Every force call builds
a local reaction network around the current bonding, assembles a multi-state
empirical valence bond (EVB) Hamiltonian over the diabatic states that network
admits, and takes its ground state. The dominant diabatic state then decides the
topology carried into the next step, which is how bonds break and form.

The package is exposed to the rest of the Python ecosystem as an
:class:`ase.calculators.calculator.Calculator`, so any ASE dynamics, optimizer
or barostat can drive it.

.. note::

   DynamicTopology is the **inference** package: the force field, the reactive
   EVB machinery and the MD scripts that run on a dataset. Fitting parameters
   (labelling reference data, refitting templates and couplings, importing
   q-force output) lives in the companion package **fast-forces**, which fits
   through this package's :func:`~DynamicTopology.forcefield.evaluate.evaluate`
   so that a template is fitted on exactly the energy it is simulated with.

.. code-block:: python

   from ase import io
   from DynamicTopology.core import ReactionSet
   from DynamicTopology.ase import DynamicTopology

   reaction_set = ReactionSet("datasets/HCombustion/HCombustion.json")
   atoms = io.read("examples/mix-n100-d250.xyz")
   atoms.calc = DynamicTopology(atoms, reaction_set)
   print(atoms.get_potential_energy())  # eV

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   usage
   concepts
   datasets
   scripts

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index


Indices
-------

* :ref:`genindex`
* :ref:`modindex`

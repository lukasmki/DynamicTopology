Force field
===========

Every force field takes positions in Å and parameters in eV and Å, and returns
``(energy, forces, virial)``. The equations are in
:doc:`/reference`.

Evaluating one topology
-----------------------

.. automodule:: DynamicTopology.forcefield.evaluate

Global parameters
-----------------

.. automodule:: DynamicTopology.forcefield.params

Bonded terms
------------

.. automodule:: DynamicTopology.forcefield.qforce

EVB couplings
-------------

.. automodule:: DynamicTopology.forcefield.coupling

Pair geometry
-------------

.. automodule:: DynamicTopology.forcefield.neighbors

Electrostatics
--------------

.. automodule:: DynamicTopology.forcefield.electrostatics

.. automodule:: DynamicTopology.forcefield.acks2

.. automodule:: DynamicTopology.forcefield.pointcharge

.. automodule:: DynamicTopology.forcefield.ewald

.. automodule:: DynamicTopology.forcefield.pme

Repulsion and dispersion
------------------------

.. automodule:: DynamicTopology.forcefield.zbl

.. automodule:: DynamicTopology.forcefield.lj

Intramolecular exclusions
-------------------------

.. automodule:: DynamicTopology.forcefield.exclusions

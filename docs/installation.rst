Installation
============

DynamicTopology requires Python 3.13 or newer and is managed with
`uv <https://docs.astral.sh/uv/>`_. From a clone of the repository:

.. code-block:: sh

   uv sync            # install the package, its dependencies and the dev group

The runtime dependencies are ASE, networkx, NumPy, SciPy and
`molify <https://pypi.org/project/molify/>`_ (bond perception and box packing).
The ``dev`` dependency group adds pytest, Jupyter kernel support, matplotlib and
Sphinx.

Running the tests
-----------------

The tests refer to datasets and fixtures by repository-relative paths, so run
them from the repository root:

.. code-block:: sh

   uv run pytest
   uv run pytest tests/test_gradients.py::TestQForceGradients::test_bond

Building this documentation
---------------------------

.. code-block:: sh

   uv run make -C docs html     # output in docs/_build/html

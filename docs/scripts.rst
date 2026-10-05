Scripts
=======

The ``scripts/`` directory holds command-line drivers built on the calculator.
Run them from the repository root with ``uv run python scripts/<name>.py``. Each
accepts ``-r/--rnet`` to choose the dataset manifest and ``--help`` for the full
option list.

``mixture.py``: pack a box
--------------------------

Packs a periodic box of molecules taken from the reaction set's templates, using
``molify.pack``.

.. code-block:: sh

   uv run python scripts/mixture.py -n 100 -d 250 -x 1:1 -o mix.xyz          # H2:O2 by mass density (kg/m³)
   uv run python scripts/mixture.py -n 100 -b 20 -x 1:2 -o mix.xyz           # fixed cubic edge (Å)
   uv run python scripts/mixture.py -r datasets/Water/Water.json --formulas H2O -x 1 -n 64 -d 1000

Use ``--box`` rather than ``--density`` for a composition sweep, so that the
volume is held fixed while the ratio changes.

``singlepoint.py``: energies of existing frames
-----------------------------------------------

.. code-block:: sh

   uv run python scripts/singlepoint.py -i examples/mix-n100-d250.xyz

Evaluates every frame of the input and prints the calculator results. It plots
the energy when there is more than one frame.

``nvt.py``: reactive Langevin MD
--------------------------------

.. code-block:: sh

   uv run python scripts/nvt.py -i mix.xyz -o nvt.xyz -l log.jsonl -n 2000 -T 2000 --seed 1

Writes two outputs:

* ``--output``: an extended-XYZ trajectory in which every frame carries the
  connectivity the calculator was actually using.
* ``--log``: one JSON object per logged frame with energies, temperature,
  species counts and the EVB basis diagnostics.

Key options are ``--timestep`` (fs, default 0.5), ``--friction`` (1/fs),
``--interval`` (steps between frames), ``--restart`` (continue from the last
frame of ``--output``), ``--minimize`` and the basis controls
``--bimol-cutoff``, ``--eps``, ``--switch-width``, ``--max-states`` and
``--max-depth``.

Both MD scripts set ``OMP_NUM_THREADS`` and ``OPENBLAS_NUM_THREADS`` to 1
unless they are already set. A box of a few hundred atoms is single-core work,
and left to their defaults numpy's and scipy's OpenBLAS each start a thread per
visible CPU: four runs sharing 32 cores were 38× slower each. For a box of
thousands of atoms, set them yourself (``OPENBLAS_NUM_THREADS=8``); see the
README's *Running on a CPU node*.

``npt.py``: reactive Berendsen NPT
----------------------------------

.. code-block:: sh

   uv run python scripts/npt.py -i examples/water-64.xyz -o npt.xyz -l log.jsonl

Has the same output layout as ``nvt.py``, and adds the cell, volume and density
to each log record. It defaults to ``datasets/Water/Water.json``. ``--taut`` and
``--taup`` set the thermostat and barostat time constants.

``analyze.py`` and ``plot.py``
------------------------------

.. code-block:: sh

   uv run python scripts/analyze.py <sweep-dir>          # aggregate <sweep-dir>/*/log.jsonl
   uv run python scripts/plot.py -i nvt.xyz              # energies, temperature, species

``analyze.py`` groups a sweep's runs by a key in each run's ``config.json``
(``--group-by``, default ``ratio``). It reports product formation next to the
basis diagnostics.

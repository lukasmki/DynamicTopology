"""Profile the pivot-invariant basis closure on a full box.

The closure evaluates a diabatic energy for every *candidate* state, including
the ones the gate rejects, where the old star generation evaluated one per
admitted state and enumerated reactions once for the whole system.  That makes
`EVBBasis.build` the part of a force call most able to make MD unusably slow, so
it gets its own driver rather than being read off the MD profile.

Two costs dominate and both have already been addressed once; this driver is how
a regression in either shows up:

  - `QForce.__call__` forms a full N x N x 3 displacement array per call, so
    diabats are evaluated over their own block's atoms rather than the whole
    system (`EVBBasis._energy`).
  - `ReactionSet.get_network` is called once for the whole system, and the
    per-block seed reaction lists are primed from that same network rather than
    re-derived per block (`EVBBasis.build`).

Run:  uv run python tests/profile/basis.py -i tests/data/mix-n100-d30.xyz
"""

import cProfile
import pstats
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase import Atoms, io

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System


def summarize(results: dict) -> None:
    blocks = results["blocks"]
    sizes = np.array([block["basis_size"] for block in blocks])
    print(f"blocks              {len(blocks)}")
    print(f"states total        {sizes.sum()}")
    print(f"largest block       {sizes.max()}")
    print(f"size histogram      {np.bincount(sizes).tolist()}")
    print(f"max depth reached   {max(block['depth'] for block in blocks)}")
    print(f"blocks capped       {sum(block['capped'] for block in blocks)}")

    placeholders = sorted({c for block in blocks for c in block["placeholder_channels"]})
    print(f"placeholder channels {len(placeholders)}")
    for channel in placeholders:
        print(f"    {channel}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", type=Path, default="tests/data/mix-n100-d30.xyz")
    parser.add_argument("-r", "--rnet", type=Path, default="datasets/HCombustion/HCombustion.json")
    parser.add_argument("-n", "--repeats", type=int, default=5)
    parser.add_argument("--eps", type=float, default=None, help="Override the admission threshold.")
    args = parser.parse_args()

    atoms = io.read(args.input, index=0)
    assert isinstance(atoms, Atoms)
    reaction_set = ReactionSet(args.rnet)
    system = System(atoms, Topology.from_atoms(atoms), reaction_set)
    if args.eps is not None:
        system.basis.eps = args.eps

    print(f"{len(atoms)} atoms, eps = {system.basis.eps:g}, "
          f"max_states = {system.basis.max_states}")
    summarize(system.calculate())  # also warms the ReactionSet caches

    start = time.perf_counter()
    for _ in range(args.repeats):
        system.calculate()
    elapsed = (time.perf_counter() - start) / args.repeats
    print(f"\nforce call          {elapsed:.3f} s")

    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(args.repeats):
        system.calculate()
    profiler.disable()

    print()
    pstats.Stats(profiler).strip_dirs().sort_stats("cumtime").print_stats(20)


if __name__ == "__main__":
    main()

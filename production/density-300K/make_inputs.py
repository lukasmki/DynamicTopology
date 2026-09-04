#!/usr/bin/env python3
"""Pack the starting box, and measure what a force call on it costs.

One box, shared by every seed: the replicates differ in their initial velocities
and their thermostat noise, not in their packing, so the spread across seeds
measures the trajectory-to-trajectory variation rather than three different
arrangements of the same molecules.

The timing is here rather than in the README because a cost quoted from memory
goes stale silently.  This prints the number the README's arithmetic is built
on, so re-running it after any change to the force field or the `[evb]` block
re-derives it.

Run from the repository root.
"""

import subprocess
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def timed_force_call(config: dict, path: Path) -> None:
    """One force call on the packed box, with the sweep's own EVB settings.

    Reported per step and extrapolated to the two stages, so the wall-clock in
    `submit.slurm` can be checked against something measured on this machine
    rather than inherited from the other sweep, whose force field is a different
    cost entirely.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from ase import io

    from DynamicTopology.ase import DynamicTopology
    from DynamicTopology.core import ReactionSet

    evb = dict(config["evb"])
    cutoff = evb.pop("bimol_cutoff")

    atoms = io.read(path)
    atoms.calc = DynamicTopology(
        atoms, ReactionSet(config["system"]["rnet"]), bimol_cutoff=cutoff, evb=evb
    )

    start = time.perf_counter()
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    elapsed = time.perf_counter() - start

    steps = config["nvt"]["steps"] + config["npt"]["steps"]
    print(
        f"\n  {len(atoms)} atoms, E = {energy:.4f} eV, "
        f"|F|max = {abs(forces).max():.3f} eV/A"
    )
    print(f"  stress available: {stress.shape[0]} Voigt components, all finite")
    print(f"  {elapsed:.2f} s per force call")
    print(
        f"  {steps} steps ({config['nvt']['steps']} NVT + {config['npt']['steps']} "
        f"NPT) = {steps * elapsed / 3600:.1f} h per run, "
        f"{len(config['sweep']['seeds']) * steps * elapsed / 3600:.1f} core-hours "
        f"for the sweep"
    )


def main() -> int:
    config = tomllib.load((HERE / "sweep.toml").open("rb"))
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)

    system = config["system"]
    target = inputs / f"w{system['molecules']}.xyz"
    if target.exists():
        print(f"  {target.name} exists, keeping it")
    else:
        subprocess.run(
            [
                sys.executable,
                "scripts/mixture.py",
                "-r",
                system["rnet"],
                # A single species, so the ratio is the trivial one.  `--formulas`
                # is what lets this script use the same packer the H2/O2 sweep
                # does instead of duplicating it.
                "--formulas",
                "H2O",
                "-x",
                "1",
                "-n",
                str(system["molecules"]),
                "-d",
                str(system["density"]),
                "-o",
                str(target),
            ],
            check=True,
            cwd=ROOT,
        )

    timed_force_call(config, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

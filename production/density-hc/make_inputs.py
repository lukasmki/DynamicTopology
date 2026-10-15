#!/usr/bin/env python3
"""Pack one starting box per density in `sweep.toml`.

One box per density, shared by that density's seeds: the replicates differ in
their initial velocities, not their packing, so the density comparison is not
also comparing five different arrangements of the same molecules.

Run from the repository root.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def box_name(density: float) -> str:
    return f"d{density:g}.xyz"


def main() -> int:
    config = tomllib.load((HERE / "sweep.toml").open("rb"))
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)

    for density in config["sweep"]["densities"]:
        target = inputs / box_name(density)
        if target.exists():
            print(f"  {target.name} exists, keeping it")
            continue
        subprocess.run(
            [
                sys.executable,
                "scripts/mixture.py",
                "-r",
                config["system"]["rnet"],
                "-x",
                config["system"]["ratio"],
                "-n",
                str(config["system"]["molecules"]),
                "-d",
                str(density),
                "-o",
                str(target),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

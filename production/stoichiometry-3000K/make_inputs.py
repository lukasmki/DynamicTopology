#!/usr/bin/env python3
"""Pack one starting box per composition in `sweep.toml`.

One box per ratio, shared by that ratio's seeds: the replicates differ in their
initial velocities, not their packing, so the composition comparison is not also
comparing five different arrangements of the same molecules.

Run from the repository root.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    config = tomllib.load((HERE / "sweep.toml").open("rb"))
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)

    for ratio in config["sweep"]["ratios"]:
        target = inputs / f"{ratio.replace(':', '_')}.xyz"
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
                ratio,
                "-n",
                str(config["system"]["molecules"]),
                "-b",
                str(config["system"]["box"]),
                "-o",
                str(target),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

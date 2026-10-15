#!/usr/bin/env python3
"""Pack the one starting box in `sweep.toml`.

One box shared by every temperature and every seed: the runs differ in their
thermostat setpoint, initial velocities and Langevin noise, not their packing,
so the temperature comparison is not also comparing different arrangements of
the same molecules.

Run from the repository root.
"""

import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent


def box_name(config: dict) -> str:
    system = config["system"]
    return f"{system['ratio'].replace(':', '_')}_d{system['density']:g}.xyz"


def main() -> int:
    config = tomllib.load((HERE / "sweep.toml").open("rb"))
    inputs = HERE / "inputs"
    inputs.mkdir(exist_ok=True)

    target = inputs / box_name(config)
    if target.exists():
        print(f"  {target.name} exists, keeping it")
        return 0
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
            str(config["system"]["density"]),
            "-o",
            str(target),
        ],
        check=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

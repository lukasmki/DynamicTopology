#!/usr/bin/env python3
"""Run one configuration of the sweep, addressed by index.

An index is the whole interface, which is what makes this portable: a SLURM
array supplies `$SLURM_ARRAY_TASK_ID`, and locally a `seq` loop does the same
job.  Nothing here is scheduler-specific.

    uv run python production/density-300K/run_one.py 0
    uv run python production/density-300K/run_one.py --list

A run is **two stages**, and the split is the measurement's design rather than a
convenience.  `scripts/nvt.py` relaxes the packing at fixed volume, then
`scripts/npt.py` picks up its last frame -- positions, velocities' worth of
thermal history, and the bonding the run had reached, since extxyz carries the
connectivity -- and lets the box move.  Handing a freshly packed box straight to
a barostat would rescale the cell against a pressure that is mostly the packer's
close contacts.

Run from the repository root.
"""

import json
import subprocess
import sys
import tomllib
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def configurations(config: dict) -> list[dict]:
    """Every run in the sweep, in a fixed order -- one per seed."""
    system = config["system"]
    source = HERE / "inputs" / f"w{system['molecules']}.xyz"
    runs = []
    for seed in config["sweep"]["seeds"]:
        runs.append(
            {
                "run_id": f"s{seed}",
                "seed": seed,
                "input": str(source),
                "rnet": system["rnet"],
                "molecules": system["molecules"],
                "start_density": system["density"],
                **config["md"],
                "nvt": config["nvt"],
                "npt": config["npt"],
                # Copied in full rather than referenced: these change the
                # potential energy surface, so a density is only interpretable
                # next to the settings that produced it.
                "evb": config["evb"],
            }
        )
    return runs


def evb_flags(evb: dict) -> list[str]:
    """The `[evb]` block as command-line flags for the MD scripts."""
    return [
        "--bimol-cutoff",
        str(evb["bimol_cutoff"]),
        "--eps",
        str(evb["eps"]),
        "--switch-width",
        str(evb["switch_width"]),
        "--max-states",
        str(evb["max_states"]),
        "--max-depth",
        str(evb["max_depth"]),
    ]


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "index", type=int, nargs="?", help="0-based index into the sweep"
    )
    parser.add_argument("--list", action="store_true", help="print the sweep and exit")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="continue an interrupted run from its own trajectory. Applied to "
        "whichever stage was in progress: a complete NVT stage is not redone.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override BOTH stages' step counts, for a smoke test. The values "
        "actually used are what land in config.json, so a short run is not "
        "mistakable for a full one.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = tomllib.load((HERE / "sweep.toml").open("rb"))
    runs = configurations(config)

    if args.list:
        for i, run in enumerate(runs):
            print(
                f"{i:3}  {run['run_id']:<6} seed {run['seed']}  "
                f"{run['nvt']['steps']} NVT + {run['npt']['steps']} NPT steps"
            )
        print(f"\n{len(runs)} runs; array indices 0-{len(runs) - 1}")
        return 0

    if args.index is None:
        parser.error("an index is required (or --list)")
    if not 0 <= args.index < len(runs):
        parser.error(f"index {args.index} outside 0-{len(runs) - 1}")

    run = runs[args.index]
    if args.steps is not None:
        run = {
            **run,
            "nvt": {**run["nvt"], "steps": args.steps},
            "npt": {**run["npt"], "steps": args.steps},
        }
    source = Path(run["input"])
    if not source.exists():
        parser.error(f"{source} missing; run make_inputs.py first")

    outdir = HERE / "output" / run["run_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(run, indent=2) + "\n")

    nvt_xyz = outdir / "nvt.xyz"
    npt_xyz = outdir / "npt.xyz"
    flags = evb_flags(run["evb"])

    stages: list[tuple[str, list[str]]] = []

    # Stage 1 is skipped when it already finished, so a --restart that died in
    # the NPT stage does not throw away 5 ps of equilibration.
    nvt_done = nvt_xyz.exists() and npt_xyz.exists()
    if not nvt_done:
        stages.append(
            (
                "nvt",
                [
                    sys.executable,
                    "scripts/nvt.py",
                    "-r",
                    run["rnet"],
                    "-i",
                    run["input"],
                    "-o",
                    str(nvt_xyz),
                    "-l",
                    str(outdir / "nvt.jsonl"),
                    "-n",
                    str(run["nvt"]["steps"]),
                    "--seed",
                    str(run["seed"]),
                    "-T",
                    str(run["temperature"]),
                    "--timestep",
                    str(run["timestep"]),
                    "--friction",
                    str(run["nvt"]["friction"]),
                    "--interval",
                    str(run["interval"]),
                ]
                + flags
                + (["--restart"] if args.restart and nvt_xyz.exists() else []),
            )
        )

    stages.append(
        (
            "npt",
            [
                sys.executable,
                "scripts/npt.py",
                "-r",
                run["rnet"],
                # Seeded from the NVT stage's last frame, which carries the
                # relaxed geometry and the bonding it had reached.
                "-i",
                str(nvt_xyz),
                "-o",
                str(npt_xyz),
                "-l",
                str(outdir / "npt.jsonl"),
                "-n",
                str(run["npt"]["steps"]),
                "--seed",
                str(run["seed"]),
                "-T",
                str(run["temperature"]),
                "-P",
                str(run["npt"]["pressure"]),
                "--timestep",
                str(run["timestep"]),
                "--taut",
                str(run["npt"]["taut"]),
                "--taup",
                str(run["npt"]["taup"]),
                "--compressibility",
                str(run["npt"]["compressibility"]),
                "--interval",
                str(run["interval"]),
            ]
            + flags
            + (["--restart"] if args.restart and npt_xyz.exists() else []),
        )
    )

    for name, command in stages:
        print(f"[{args.index}] {run['run_id']} {name}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

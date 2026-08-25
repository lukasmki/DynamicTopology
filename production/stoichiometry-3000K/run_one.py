#!/usr/bin/env python3
"""Run one configuration of the sweep, addressed by index.

An index is the whole interface, which is what makes this portable: a SLURM
array supplies `$SLURM_ARRAY_TASK_ID`, and locally a `seq` loop does the same
job.  Nothing here is scheduler-specific.

    uv run python production/stoichiometry-3000K/run_one.py 0
    uv run python production/stoichiometry-3000K/run_one.py --list

Run from the repository root.
"""

import json
import subprocess
import sys
import tomllib
from argparse import ArgumentParser
from pathlib import Path

HERE = Path(__file__).resolve().parent


def configurations(config: dict) -> list[dict]:
    """Every run in the sweep, in a fixed order.

    Ratio-major so that consecutive indices are replicates of one composition:
    a partially completed array then has complete compositions rather than one
    seed of each, which is the difference between a usable partial result and
    an unusable one.
    """
    runs = []
    for ratio in config["sweep"]["ratios"]:
        for seed in config["sweep"]["seeds"]:
            runs.append(
                {
                    "run_id": f"r{ratio.replace(':', '_')}_s{seed}",
                    "ratio": ratio,
                    "seed": seed,
                    "input": str(HERE / "inputs" / f"{ratio.replace(':', '_')}.xyz"),
                    **config["md"],
                    "rnet": config["system"]["rnet"],
                    "box": config["system"]["box"],
                    "molecules": config["system"]["molecules"],
                }
            )
    return runs


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "index", type=int, nargs="?", help="0-based index into the sweep"
    )
    parser.add_argument("--list", action="store_true", help="print the sweep and exit")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="continue an interrupted run from its own trajectory",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="override the sweep's step count, for a smoke test or a probe. The "
        "value actually used is what lands in config.json, so a short run is not "
        "mistakable for a full one.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = tomllib.load((HERE / "sweep.toml").open("rb"))
    runs = configurations(config)

    if args.list:
        for i, run in enumerate(runs):
            print(
                f"{i:3}  {run['run_id']:<12} ratio {run['ratio']:<4} seed {run['seed']}"
            )
        print(f"\n{len(runs)} runs; array indices 0-{len(runs) - 1}")
        return 0

    if args.index is None:
        parser.error("an index is required (or --list)")
    if not 0 <= args.index < len(runs):
        parser.error(f"index {args.index} outside 0-{len(runs) - 1}")

    run = runs[args.index]
    if args.steps is not None:
        run = {**run, "steps": args.steps}
    source = Path(run["input"])
    if not source.exists():
        parser.error(f"{source} missing; run make_inputs.py first")

    outdir = HERE / "output" / run["run_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(run, indent=2) + "\n")

    command = [
        sys.executable,
        "scripts/nvt.py",
        "-r",
        run["rnet"],
        "-i",
        run["input"],
        "-o",
        str(outdir / "traj.xyz"),
        "-l",
        str(outdir / "log.jsonl"),
        "-n",
        str(run["steps"]),
        "--seed",
        str(run["seed"]),
        "-T",
        str(run["temperature"]),
        "--timestep",
        str(run["timestep"]),
        "--friction",
        str(run["friction"]),
        "--interval",
        str(run["interval"]),
    ]
    if args.restart:
        command.append("--restart")

    print(f"[{args.index}] {run['run_id']}: {' '.join(command)}", flush=True)
    if args.dry_run:
        return 0
    return subprocess.run(command).returncode


if __name__ == "__main__":
    raise SystemExit(main())

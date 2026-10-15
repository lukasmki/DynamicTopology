#!/usr/bin/env python3
"""Run one configuration of the sweep, addressed by index.

An index is the whole interface, which is what makes this portable: a SLURM
array supplies `$SLURM_ARRAY_TASK_ID`, and locally a `seq` loop does the same
job.  Nothing here is scheduler-specific.

    uv run python production/temperature-hc/run_one.py 0
    uv run python production/temperature-hc/run_one.py --list

Run from the repository root.
"""

import json
import subprocess
import sys
import tomllib
from argparse import ArgumentParser
from pathlib import Path

from make_inputs import box_name

HERE = Path(__file__).resolve().parent


def configurations(config: dict) -> list[dict]:
    """Every run in the sweep, in a fixed order.

    Temperature-major so that consecutive indices are replicates of one
    temperature: a partially completed array then has complete temperatures
    rather than one seed of each, which is the difference between a usable
    partial result and an unusable one.
    """
    source = str(HERE / "inputs" / box_name(config))
    runs = []
    for temperature in config["sweep"]["temperatures"]:
        for seed in config["sweep"]["seeds"]:
            runs.append(
                {
                    "run_id": f"t{temperature:g}_s{seed}",
                    "temperature": temperature,
                    "seed": seed,
                    "input": source,
                    **config["md"],
                    "rnet": config["system"]["rnet"],
                    "ratio": config["system"]["ratio"],
                    "density": config["system"]["density"],
                    "molecules": config["system"]["molecules"],
                }
            )
    return runs


def completed_steps(log: Path) -> int:
    """Steps a run has already integrated, summed over its segments.

    `scripts/nvt.py` counts from zero on every `--restart` and appends a new
    header to the log, so each segment's progress is the step of the last frame
    after its header.  Steps after a segment's last frame were never written
    out and are integrated again, which costs at most `interval` steps.
    """
    done = last = 0
    for line in log.open():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a job killed mid-write leaves a partial last line
        if record.get("record") == "header":
            done, last = done + last, 0
        elif record.get("record") == "frame":
            last = record["step"]
    return done + last


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "index", type=int, nargs="?", help="0-based index into the sweep"
    )
    parser.add_argument("--list", action="store_true", help="print the sweep and exit")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="continue an interrupted run from its own trajectory, for the steps "
        "it has left. `steps` is the run's total length, so restarting a finished "
        "run does nothing; extend one by restarting with a larger --steps.",
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
                f"{i:3}  {run['run_id']:<10} T {run['temperature']:>6g} K  "
                f"seed {run['seed']}"
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

    steps = run["steps"]
    if args.restart:
        steps -= completed_steps(outdir / "log.jsonl")
        if steps <= 0:
            print(f"[{args.index}] {run['run_id']}: complete, nothing to restart")
            return 0

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
        str(steps),
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
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Aggregate a sweep's JSONL logs into the composition dependence.

Reads every `<dir>/*/log.jsonl` written by `scripts/nvt.py`, groups runs by
whatever varied (composition, here), and reports both the chemistry and the
diagnostics that say whether to believe it.

The diagnostics are not decoration.  A basis that was capped is seed-dependent
where it fired, and a channel entering on an unfitted coupling is being driven
by a number nobody fitted -- either can make a species curve look like
chemistry when it is an artifact.  They are printed alongside the result rather
than in a separate place nobody looks.

    uv run python scripts/analyze.py production/stoichiometry-3000K/output
"""

import json
import statistics
from argparse import ArgumentParser
from pathlib import Path

# Species that only exist because something reacted.  H2 and O2 are the
# reactants and are present from the start, so counting them says nothing about
# whether the run did anything.
PRODUCTS = ("H2O", "HO", "H", "O", "HO2", "H2O2", "O3", "H3O")


def read_run(path: Path) -> dict | None:
    """One run's header and frames, or None if the log has no frames yet."""
    header, frames = None, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A job killed mid-write leaves a partial last line.  Everything
                # before it is still good, and discarding the whole run over one
                # truncated record would throw away most of a 32-hour job.
                continue
            if record.get("record") == "header":
                header = record
            elif record.get("record") == "frame":
                frames.append(record)
    if not frames:
        return None
    return {"header": header or {}, "frames": frames, "path": path}


def first_product(run: dict) -> float | None:
    """Time (fs) of the first frame containing any product species."""
    for frame in run["frames"]:
        if any(frame["species"].get(s, 0) for s in PRODUCTS):
            return frame["time_fs"]
    return None


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--group-by",
        default="ratio",
        help="key in each run's config.json to group by (default: ratio)",
    )
    args = parser.parse_args()

    runs: dict[str, list[dict]] = {}
    for log in sorted(args.directory.glob("*/log.jsonl")):
        run = read_run(log)
        if run is None:
            print(f"  (no frames yet: {log.parent.name})")
            continue
        config_path = log.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        run["config"] = config
        runs.setdefault(str(config.get(args.group_by, "?")), []).append(run)

    if not runs:
        print(f"no runs with frames under {args.directory}")
        return 1

    print(
        f"{args.group_by:<8}{'runs':>5}{'ps':>8}{'products':>10}"
        f"{'H2O max':>9}{'t_first (ps)':>14}{'capped':>8}{'unfitted':>9}"
    )
    for key in sorted(runs, key=lambda k: (len(k), k)):
        group = runs[key]
        simulated = [r["frames"][-1]["time_fs"] / 1000.0 for r in group]
        # Peak count of any product, per run, then averaged: "did anything
        # happen, and how much" without assuming which product matters.
        peaks = [
            max(
                (sum(f["species"].get(s, 0) for s in PRODUCTS) for f in r["frames"]),
                default=0,
            )
            for r in group
        ]
        water = [
            max((f["species"].get("H2O", 0) for f in r["frames"]), default=0)
            for r in group
        ]
        onsets = [t for t in (first_product(r) for r in group) if t is not None]
        capped = sum(sum(f.get("ncapped", 0) > 0 for f in r["frames"]) for r in group)
        unfitted = sorted(
            {
                c
                for r in group
                for f in r["frames"]
                for c in f.get("placeholder_channels", [])
            }
        )
        onset = f"{statistics.mean(onsets) / 1000:.2f}" if onsets else "-"
        print(
            f"{key:<8}{len(group):>5}{statistics.mean(simulated):>8.1f}"
            f"{statistics.mean(peaks):>10.1f}{statistics.mean(water):>9.1f}"
            f"{onset:>14}{capped:>8}{len(unfitted):>9}"
        )

    total_frames = sum(len(r["frames"]) for g in runs.values() for r in g)
    reacted = sum(1 for g in runs.values() for r in g if first_product(r) is not None)
    total = sum(len(g) for g in runs.values())
    print(
        f"\n{total} runs, {total_frames} frames; {reacted} showed any product species"
    )

    if reacted == 0:
        print(
            "\nNothing reacted in any run. The sweep measures nothing as it stands: "
            "either the runs are too short, or the temperature is too low for this "
            "surface. Check a single trajectory before spending more compute."
        )
    every_unfitted = sorted(
        {
            c
            for g in runs.values()
            for r in g
            for f in r["frames"]
            for c in f.get("placeholder_channels", [])
        }
    )
    if every_unfitted:
        print(
            f"\nChannels that entered a basis on an unfitted coupling: "
            f"{', '.join(every_unfitted)}. Expected for rxn_06 (O2 -> O + O), which "
            "carries a hand-set amplitude; anything else means the dataset moved."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Average each run's NPT volume into a density, with an honest uncertainty.

Sweep-local rather than a change to `scripts/analyze.py`, whose `PRODUCTS` tuple
and "time to first product" framing describe a combustion run and say nothing
about this one.  What it shares is the principle: the diagnostics that decide
whether to believe a number are printed next to the number, not somewhere else.

    uv run python production/density-300K/density.py production/density-300K/output

Two uncertainties are reported and they mean different things:

  * the **within-run** spread is the standard deviation of block means, and
    understates the truth -- consecutive frames of a 20 ps trajectory are not
    independent samples of the volume.
  * the **between-seed** spread is the one to quote.  Three trajectories that
    share a starting configuration and differ only in their thermal noise are
    as close to independent estimates as this budget buys.
"""

import json
import statistics
from argparse import ArgumentParser
from pathlib import Path

# Number of blocks the production window is cut into for the within-run error.
# Blocks must be long against the volume autocorrelation time (1-10 ps for
# water) or the error bar is measuring the correlation, not the uncertainty --
# at 20 ps of production that allows only a handful, hence five.
NBLOCKS = 5


def read_log(path: Path) -> tuple[dict, list[dict]]:
    """One run's header and frames.  A truncated last line is skipped.

    A job killed mid-write leaves a partial record; discarding the whole run
    over it would throw away a 26-hour job.
    """
    header, frames = {}, []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("record") == "header":
                header = record
            elif record.get("record") == "frame":
                frames.append(record)
    return header, frames


def block_means(values: list[float], nblocks: int = NBLOCKS) -> list[float]:
    """Means of `nblocks` consecutive, equal-length blocks."""
    size = len(values) // nblocks
    if size == 0:
        return [statistics.mean(values)] if values else []
    return [statistics.mean(values[i * size : (i + 1) * size]) for i in range(nblocks)]


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--equilibration",
        type=float,
        default=None,
        help="fs to discard before averaging. Defaults to the value in each "
        "run's config.json, which is what the sweep pinned.",
    )
    args = parser.parse_args()

    rows = []
    for log in sorted(args.directory.glob("*/npt.jsonl")):
        header, frames = read_log(log)
        if not frames:
            print(f"  (no frames yet: {log.parent.name})")
            continue

        config_path = log.parent / "config.json"
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
        drop = args.equilibration
        if drop is None:
            drop = config.get("npt", {}).get("equilibration", 0.0)

        production = [f for f in frames if f["time_fs"] >= drop]
        if not production:
            print(
                f"  ({log.parent.name}: no frames past {drop:.0f} fs of "
                f"equilibration; run is {frames[-1]['time_fs']:.0f} fs long)"
            )
            continue

        densities = [f["density_kg_m3"] for f in production]
        blocks = block_means(densities)
        rows.append(
            {
                "run": log.parent.name,
                "ps": production[-1]["time_fs"] / 1000.0,
                "density": statistics.mean(densities),
                "block_sd": statistics.stdev(blocks) if len(blocks) > 1 else 0.0,
                "start": frames[0]["density_kg_m3"],
                "edge": production[-1]["cell"][0],
                "pressure": statistics.mean(f["pressure_bar"] for f in production),
                "temperature": statistics.mean(f["temperature_K"] for f in production),
                "capped": sum(f.get("ncapped", 0) > 0 for f in production),
                "cutoff": config.get("evb", {}).get("bimol_cutoff", 4.0),
                "channels": sorted(
                    {c for f in production for c in f.get("placeholder_channels", [])}
                ),
                "species": production[-1]["species"],
            }
        )

    if not rows:
        print(f"no runs with frames under {args.directory}")
        return 1

    print(
        f"{'run':<6}{'ps':>7}{'rho (kg/m3)':>13}{'+-block':>9}{'from':>8}"
        f"{'edge (A)':>10}{'P (bar)':>11}{'T (K)':>8}{'capped':>8}"
    )
    for r in rows:
        print(
            f"{r['run']:<6}{r['ps']:>7.1f}{r['density']:>13.1f}{r['block_sd']:>9.1f}"
            f"{r['start']:>8.0f}{r['edge']:>10.3f}{r['pressure']:>11.1f}"
            f"{r['temperature']:>8.1f}{r['capped']:>8}"
        )

    densities = [r["density"] for r in rows]
    mean = statistics.mean(densities)
    print(f"\ndensity = {mean:.1f} kg/m^3", end="")
    if len(densities) > 1:
        print(f" +- {statistics.stdev(densities):.1f} (sd over {len(densities)} seeds)")
    else:
        print("  (one seed; no uncertainty available)")

    # --- the diagnostics that say whether to believe it --------------------
    print("\nchecks:")

    capped = sum(r["capped"] for r in rows)
    print(
        f"  basis capped in {capped} frame(s)"
        + ("" if capped == 0 else "  <- the surface is seed-dependent where this fired")
    )

    channels = sorted({c for r in rows for c in r["channels"]})
    print(
        f"  unfitted couplings: {', '.join(channels) if channels else 'none'}"
        + (
            ""
            if not channels
            else "  <- all three Water channels are fitted; this means the dataset moved"
        )
    )

    temps = [r["temperature"] for r in rows]
    print(f"  mean temperature {statistics.mean(temps):.1f} K against 300 K target")

    # Minimum image is only defined while the box stays well above twice the
    # interaction range, so a contracting cell is the failure to watch for.
    cutoff = max(r["cutoff"] for r in rows)
    edge = min(r["edge"] for r in rows)
    print(
        f"  smallest cell edge {edge:.2f} A against a {cutoff:.1f} A cutoff "
        f"(needs edge > {2 * cutoff:.1f} A)"
        + ("" if edge > 2 * cutoff else "  <- MINIMUM IMAGE BROKEN, result invalid")
    )

    # A density is only a density if the thing is still water.
    for r in rows:
        others = {k: v for k, v in r["species"].items() if k != "H2O"}
        if others:
            print(f"  {r['run']} final species besides H2O: {others}")

    pressures = [r["pressure"] for r in rows]
    print(
        f"  mean virial pressure {statistics.mean(pressures):.1f} bar against the "
        "1 bar setpoint"
    )
    print(
        "\n  The Berendsen barostat gives a correct mean volume but not the true\n"
        "  isobaric ensemble, so no compressibility may be read off the volume\n"
        "  fluctuations.  Electrostatics are bare minimum-image with no Ewald\n"
        "  sum, which is the largest systematic error in the number above."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

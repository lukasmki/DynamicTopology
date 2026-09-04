#!/usr/bin/env python3
"""Berendsen NPT with a reactive topology, for measuring a density.

The NVT counterpart is `scripts/nvt.py`, and this shares its output contract
deliberately: the same two files with the same jobs, the same per-frame
connectivity stamping, the same JSONL diagnostics.  `record_topology` and
`species_counts` are imported from it rather than copied, so a fix to how the
evolving bonding is recorded lands in both.

What is different is the cell, and everything here follows from that:

  * the box moves, so every frame's `Lattice=` is part of the result rather than
    a constant of the run.  `volume`, `density_kg_m3` and the cell go into each
    log record, which is what `production/density-300K/density.py` averages.
  * this needs `atoms.get_stress()`, which the calculator gained along with
    `production/density-300K`.  See `tests/test_stress.py`.
  * no `FixCom`.  A fixed centre of mass fights the barostat's rescaling of
    every position, and the pairing is not meaningful.

**Why Berendsen.**  The measurement is `<V>`, and the Berendsen barostat gives a
correct mean volume with a robust, well-damped approach to it.  It does not
sample the true isobaric ensemble, so its volume *fluctuations* are not
physical: this script reports a density, and nothing that depends on the width
of the volume distribution -- a compressibility, most obviously -- may be read
off its output.  `ase.md.npt.NPT` (Melchionna/MTK) is the rigorous alternative
and would need an upper-triangular cell and its own Nose-Hoover thermostat.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase import Atoms, io, units
from ase.md.md import MolecularDynamics
from ase.md.nptberendsen import NPTBerendsen
from ase.md.velocitydistribution import thermalize_momenta

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet

from nvt import (
    DEFAULT_INTERVAL,
    DEFAULT_TIMESTEP,
    record_topology,
    species_counts,
)

DEFAULT_TEMPERATURE = 300.0  # K -- this script exists for liquids, not flames
DEFAULT_PRESSURE = 1.0  # bar
DEFAULT_TAUT = 100.0  # fs, thermostat coupling time
DEFAULT_TAUP = 1000.0  # fs, barostat coupling time

# Isothermal compressibility of liquid water at 300 K, 1/bar.  The barostat uses
# it only to convert a pressure error into a volume rescaling, so it sets how
# fast the box responds, not where it settles: a wrong value changes the
# approach, not the answer.  Water's is the right order for any dense liquid.
DEFAULT_COMPRESSIBILITY = 4.57e-5


def density_kg_m3(atoms: Atoms) -> float:
    """Mass density of the current cell, in kg/m^3."""
    amu_to_kg = 1.66053906660e-27
    volume_m3 = atoms.get_volume() * 1e-30
    return float(atoms.get_masses().sum()) * amu_to_kg / volume_m3


def pressure_bar(atoms: Atoms) -> float:
    """Instantaneous pressure, `-tr(sigma)/3`, in bar.

    `include_ideal_gas=True` because that is what `NPTBerendsen` itself uses to
    decide how to rescale the cell, and a log that reported the configurational
    part alone would disagree with the quantity the barostat is driving to the
    setpoint -- by `NkT/V`, which is hundreds of bar for this system.  The
    calculator supplies only the virial half; ASE adds the kinetic half here.

    Single-frame values are enormously noisy -- the virial of a dense liquid is
    a near-cancellation of two large numbers -- so this is a diagnostic to
    average, never to read one frame at a time.
    """
    stress = atoms.get_stress(voigt=False, include_ideal_gas=True)
    return float(-np.trace(stress) / 3.0 / units.bar)


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/Water/Water.json"
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="continue from the last frame of --output, appending to it. The "
        "cell comes back with it -- extxyz carries `Lattice=` -- so a resumed "
        "run continues from the volume the barostat had reached, not from the "
        "input's.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=False, default=Path("npt.xyz")
    )
    parser.add_argument(
        "-l",
        "--log",
        type=Path,
        default=None,
        help="JSONL diagnostics, one object per trajectory frame. This is what "
        "the density is measured from; the .xyz is for looking at.",
    )
    parser.add_argument("-n", "--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("-T", "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "-P", "--pressure", type=float, default=DEFAULT_PRESSURE, help="bar"
    )
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP, help="fs")
    parser.add_argument("--taut", type=float, default=DEFAULT_TAUT, help="fs")
    parser.add_argument("--taup", type=float, default=DEFAULT_TAUP, help="fs")
    parser.add_argument(
        "--compressibility", type=float, default=DEFAULT_COMPRESSIBILITY, help="1/bar"
    )
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--bimol-cutoff",
        type=float,
        default=4.0,
        help="minimum interatomic distance under which a bimolecular reaction "
        "channel is considered, in Angstrom.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="stabilization (eV) a channel must supply to enter the EVB basis. "
        "The default 1e-3 admits nearly everything; raising it is the smooth way "
        "to restrict the basis, since the admission ramp still switches the "
        "coupling on continuously and no state is refused outright.",
    )
    parser.add_argument("--switch-width", type=float, default=None, help="eV")
    parser.add_argument(
        "--max-states",
        type=int,
        default=None,
        help="hard cap on basis size. Unlike --eps this refuses states outright, "
        "which makes the surface seed-dependent where it fires -- the frames "
        "where it did are flagged `ncapped` in the log.",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    args = parser.parse_args()

    if args.output.exists() and not args.restart:
        parser.error(
            f"{args.output} already exists. Appending to it would concatenate "
            "two runs into one file with no marker between them; pass --restart "
            "to continue it deliberately, or choose another path."
        )
    if args.restart and not args.output.exists():
        parser.error(f"--restart given but {args.output} does not exist")

    source = args.output if args.restart else args.input
    atoms: Atoms | list[Atoms] = io.read(source, index=-1)
    assert isinstance(atoms, Atoms)
    if not atoms.pbc.any() or atoms.get_volume() <= 0.0:
        parser.error(
            f"{source} has no periodic cell; a barostat has nothing to act on "
            "and the stress would be undefined"
        )

    thermal_rng, langevin_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2)
    )
    if not args.restart:
        thermalize_momenta(atoms, args.temperature, rng=thermal_rng)

    evb = {
        key: value
        for key, value in (
            ("eps", args.eps),
            ("switch_width", args.switch_width),
            ("max_states", args.max_states),
            ("max_depth", args.max_depth),
        )
        if value is not None
    }
    reaction_set = ReactionSet(args.rnet)
    atoms.calc = DynamicTopology(
        atoms, reaction_set, bimol_cutoff=args.bimol_cutoff, evb=evb or None
    )

    log_file = None
    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.log.open("a" if args.restart else "w")
        log_file.write(
            json.dumps(
                {
                    "record": "header",
                    "input": str(args.input),
                    "rnet": args.rnet,
                    "seed": args.seed,
                    "temperature_K": args.temperature,
                    "pressure_bar": args.pressure,
                    "timestep_fs": args.timestep,
                    "taut_fs": args.taut,
                    "taup_fs": args.taup,
                    "compressibility_per_bar": args.compressibility,
                    "interval": args.interval,
                    "steps": args.steps,
                    "natoms": len(atoms),
                    "cell": atoms.cell.lengths().tolist(),
                    "density_kg_m3": density_kg_m3(atoms),
                    "bimol_cutoff": args.bimol_cutoff,
                    "evb": evb,
                    "restart": args.restart,
                }
            )
            + "\n"
        )
        log_file.flush()

    def status(atoms: Atoms, dyn: MolecularDynamics):
        PE = float(atoms.get_potential_energy())
        KE = float(atoms.get_kinetic_energy())
        print(
            f"{dyn.nsteps:5} E={KE + PE:14.6f} T={atoms.get_temperature():8.2f} K "
            f"P={pressure_bar(atoms):+12.1f} bar rho={density_kg_m3(atoms):8.2f} "
            f"kg/m^3 V={atoms.get_volume():9.2f} A^3"
        )
        diagnostics = getattr(atoms.calc, "diagnostics", None)
        if diagnostics is None:
            return
        capped = [i for i, b in enumerate(diagnostics["blocks"]) if b["capped"]]
        if capped:
            print(
                f"     WARN basis truncated in {len(capped)} block(s) "
                f"{capped[:5]}; the surface is seed-dependent where this fires"
            )

    def write_frame(atoms: Atoms, dyn: MolecularDynamics):
        record_topology(atoms)
        io.write(args.output, atoms, format="extxyz", append=True)

        if log_file is None:
            return
        diagnostics = getattr(atoms.calc, "diagnostics", None)
        blocks = diagnostics["blocks"] if diagnostics else []
        record = {
            "record": "frame",
            "step": int(dyn.nsteps),
            "time_fs": float(dyn.nsteps) * args.timestep,
            "energy": float(atoms.get_potential_energy()),
            "kinetic": float(atoms.get_kinetic_energy()),
            "temperature_K": float(atoms.get_temperature()),
            # The measurement itself.
            "volume": float(atoms.get_volume()),
            "density_kg_m3": density_kg_m3(atoms),
            "pressure_bar": pressure_bar(atoms),
            "cell": atoms.cell.lengths().tolist(),
            "species": species_counts(atoms),
            "nbonds": len(atoms.info.get("connectivity", [])),
        }
        if diagnostics is not None:
            record.update(
                {
                    "energy_bonded": float(diagnostics["energy_bonded"]),
                    "energy_nonbonded": float(diagnostics["energy_nonbonded"]),
                    "energy_zbl": float(diagnostics["energy_zbl"]),
                    "topology_changed": bool(diagnostics["topology_changed"]),
                    "nblocks": len(blocks),
                    "max_nstates": max((b["nstates"] for b in blocks), default=0),
                    "nreactive_blocks": sum(1 for b in blocks if b["nstates"] > 1),
                    "ncapped": sum(1 for b in blocks if b["capped"]),
                    "min_switch": min((b["min_switch"] for b in blocks), default=1.0),
                    "placeholder_channels": sorted(
                        {c for b in blocks for c in b["placeholder_channels"]}
                    ),
                }
            )
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()  # a killed cluster job must still leave usable output

    dyn = NPTBerendsen(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        pressure_au=args.pressure * units.bar,
        taut=args.taut * units.fs,
        taup=args.taup * units.fs,
        compressibility_au=args.compressibility / units.bar,
        fixcm=False,
    )
    dyn.attach(status, 1, atoms, dyn)
    dyn.attach(write_frame, args.interval, atoms, dyn)
    try:
        dyn.run(steps=args.steps)
    finally:
        if log_file is not None:
            log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

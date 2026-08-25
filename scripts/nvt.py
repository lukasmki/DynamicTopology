#!/usr/bin/env python3
"""Langevin MD with a reactive topology, and a record of what the topology did.

The trajectory this writes is the primary output, so the thing that has to be in
it is the *evolving* bonding -- which bonds broke and formed, step by step.  It
did not used to be.  `atoms.info["connectivity"]` is set once when the input file
is read and is not touched again by ASE, while the reactive topology lives on
`calc.system.topology`; writing `atoms` therefore stamped every frame with the
bonding at t = 0.  Measured on the 116-frame `examples/nvt-n100-d250.xyz`, all
116 frames carried byte-identical connectivity, so `scripts/plot.py`'s species
panel was flat by construction and `--restart` silently reset the chemistry.
`record_topology` below is the fix, and it is why this script is worth trusting
for production.

Two outputs, with different jobs:

  `--output`  the .xyz trajectory, for looking at.  Every frame carries the
              connectivity the calculator was actually carrying.
  `--log`     one JSON object per logged frame, for analysing across runs:
              energies, temperature, species counts, and the basis diagnostics
              that say whether to believe the rest.
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from ase import Atoms, io, units
from ase.constraints import FixCom
from ase.md import Langevin
from ase.md.md import MolecularDynamics
from ase.md.velocitydistribution import thermalize_momenta

from DynamicTopology.ase import DynamicTopology
from DynamicTopology.core import ReactionSet
from DynamicTopology.core.topology import Topology

# Defaults preserved from before this script grew options, so existing
# invocations keep behaving the same way.
DEFAULT_TEMPERATURE = 2000.0  # K
DEFAULT_FRICTION = 0.01  # 1/fs
DEFAULT_INTERVAL = 5  # steps between trajectory frames

# 0.5 fs was comfortable against q-force's force constants.  The refit
# (`fit.dissociation`) took the H-H stretch to 5282 cm^-1 -- a 6.3 fs period, so
# 0.5 fs is ~13 steps per period.  Fine at 2000 K; halve it at 3000 K, and halve
# it again if a run heats up without a reaction to explain it.
#
# Most of the depth the diabats needed comes from the Morse shape parameter `c`,
# which is O(dr**3) at the minimum and so costs no frequency at all.  Without it
# the same channel count needed 3.3x in force constant, or H2 at 12402 cm^-1,
# and a timestep to match.
DEFAULT_TIMESTEP = 0.5  # fs


def record_topology(atoms: Atoms) -> None:
    """Stamp the calculator's current bonding onto `atoms.info`.

    `Topology.graph` is keyed by global atom index, so its edges are exactly the
    `[i, j, bond_order]` triples extended-XYZ stores and `molify.ase2networkx`
    reads back in preference to perceiving from geometry.  Writing them makes a
    frame self-describing: reloading one restores the chemistry it was in, which
    is what `--restart` needs and what any analysis of the run depends on.
    """
    system = getattr(getattr(atoms, "calc", None), "system", None)
    topology = getattr(system, "topology", None)
    if topology is None:  # the EVB calculator holds fixed states
        return
    atoms.info["connectivity"] = [
        [int(i), int(j), None] for i, j in topology.graph.edges()
    ]


def species_counts(atoms: Atoms) -> dict[str, int]:
    """Formula histogram of the connected components of the current topology."""
    system = getattr(getattr(atoms, "calc", None), "system", None)
    topology = getattr(system, "topology", None)
    if topology is None:
        topology = Topology.from_atoms(atoms)
    counts: dict[str, int] = {}
    for molecule in topology.molecules():
        formula = atoms[sorted(molecule.graph.nodes())].get_chemical_formula()
        counts[formula] = counts.get(formula, 0) + 1
    return counts


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "-r", "--rnet", required=False, default="datasets/HCombustion/HCombustion.json"
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument(
        "--restart",
        action="store_true",
        help="continue from the last frame of --output, appending to it. Without "
        "this, an --output that already exists is an error rather than something "
        "to concatenate onto.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=False, default=Path("nvt.xyz")
    )
    parser.add_argument(
        "-l",
        "--log",
        type=Path,
        default=None,
        help="JSONL diagnostics, one object per trajectory frame. This is what a "
        "sweep is analysed from; the .xyz is for looking at.",
    )
    parser.add_argument("-n", "--steps", type=int, default=2000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for both random streams -- the initial Maxwell-Boltzmann draw "
        "and Langevin's per-step random force. Without it the run is not "
        "reproducible, which for a production sweep means its results cannot be "
        "checked. Recorded in the log header.",
    )
    parser.add_argument("-T", "--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--timestep", type=float, default=DEFAULT_TIMESTEP, help="fs")
    parser.add_argument("--friction", type=float, default=DEFAULT_FRICTION, help="1/fs")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL)
    args = parser.parse_args()

    if args.output.exists() and not args.restart:
        parser.error(
            f"{args.output} already exists. Appending to it would concatenate two "
            "runs into one file with no marker between them; pass --restart to "
            "continue it deliberately, or choose another path."
        )
    if args.restart and not args.output.exists():
        parser.error(f"--restart given but {args.output} does not exist")

    # On a restart the last frame carries the bonding the run had reached, and
    # `Topology.from_atoms` prefers it over re-perceiving -- so continuing now
    # resumes the chemistry instead of resetting it to the input's.
    source = args.output if args.restart else args.input
    atoms: Atoms | list[Atoms] = io.read(source, index=-1)
    assert isinstance(atoms, Atoms)

    # Two independent sources of randomness, and seeding only the first is not
    # enough: Langevin draws a random force every step, so trajectories from
    # identical initial momenta still diverge.  Derived from one seed via
    # SeedSequence so a run is reproducible from a single integer while the two
    # streams stay independent.
    thermal_rng, langevin_rng = (
        np.random.default_rng(s) for s in np.random.SeedSequence(args.seed).spawn(2)
    )

    if not args.restart:
        thermalize_momenta(atoms, args.temperature, rng=thermal_rng)

    reaction_set = ReactionSet(args.rnet)
    atoms.calc = DynamicTopology(atoms, reaction_set)

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
                    "timestep_fs": args.timestep,
                    "friction_per_fs": args.friction,
                    "interval": args.interval,
                    "steps": args.steps,
                    "natoms": len(atoms),
                    "cell": atoms.cell.lengths().tolist(),
                    "restart": args.restart,
                }
            )
            + "\n"
        )
        log_file.flush()

    # Previous total energy, so the jump across a topology change can be
    # reported.  A pivot-invariant basis means changing the carried topology is
    # bookkeeping and must not move the energy, so a large jump here is the
    # symptom to watch for.
    previous: list[float] = []

    def status(atoms: Atoms, dyn: MolecularDynamics):
        PE = float(atoms.get_potential_energy())
        KE = float(atoms.get_kinetic_energy())
        TE: float = KE + PE
        T = float(atoms.get_temperature())
        print(f"{dyn.nsteps:4} {TE=:15.6f}, {PE=:15.6f}, {KE=:15.6f}, {T=:15.6f}")

        diagnostics = getattr(atoms.calc, "diagnostics", None)
        if diagnostics is None:  # the EVB calculator holds fixed states
            return

        blocks = diagnostics["blocks"]
        capped = [i for i, b in enumerate(blocks) if b["capped"]]
        if capped:
            print(
                f"     WARN basis truncated in {len(capped)} block(s) "
                f"{capped[:5]}; the surface is seed-dependent where this fires"
            )

        if diagnostics["topology_changed"]:
            jump = f"{TE - previous[-1]:+.6f} eV" if previous else "n/a"
            reactive = [
                (i, b["nstates"], float(b["weights"].max()), b["gap"], b["min_switch"])
                for i, b in enumerate(blocks)
                if b["nstates"] > 1
            ]
            print(f"     TOPOLOGY CHANGED  dE_total = {jump}")
            for i, nstates, weight, gap, switch in reactive[:5]:
                # `switch` below 1 means a channel is partway through the
                # admission ramp, i.e. a state is currently joining or leaving
                # the basis. That is the intended behaviour, not a warning.
                print(
                    f"       block {i}: {nstates} states, "
                    f"max c^2 = {weight:.3f}, gap = {gap:.4f} eV, "
                    f"min switch = {switch:.3f}"
                )
            placeholders = sorted(
                {c for b in blocks for c in b["placeholder_channels"]}
            )
            if placeholders:
                print(f"       on unfitted couplings: {', '.join(placeholders)}")

        previous.append(TE)

    def write_frame(atoms: Atoms, dyn: MolecularDynamics):
        """Trajectory frame plus its log record, both carrying the topology."""
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
            "species": species_counts(atoms),
            "nbonds": len(atoms.info.get("connectivity", [])),
        }
        if diagnostics is not None:
            record.update(
                {
                    "energy_bonded": float(diagnostics["energy_bonded"]),
                    "energy_nonbonded": float(diagnostics["energy_nonbonded"]),
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

    dyn = Langevin(
        atoms,
        timestep=args.timestep * units.fs,
        temperature_K=args.temperature,
        friction=args.friction / units.fs,
        fixcm=False,
        rng=langevin_rng,
    )
    atoms.set_constraint(FixCom())
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

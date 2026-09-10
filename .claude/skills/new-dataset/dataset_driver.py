#!/usr/bin/env python3
"""Scaffold, check and *run* a DynamicTopology reaction-set dataset.

Four subcommands, meant to be used in this order:

    scaffold  write the manifest + directory skeleton + placeholder couplings
    check     static validation -- every trap that makes `ReactionSet.load`
              throw, or makes it load something quietly wrong
    probe     the reaction table and the energy bookkeeping, which is where
              a charged dataset goes wrong without any error being raised
    run       load it, pack a box, build the network, call the force field,
              integrate NVE.  This is the only step that proves it works.

`check` and `probe` need no ab-initio anything, so they are fast and are what
you iterate against.  Run from the repo root (paths in manifests are relative
to the manifest, but the force field's own tests assume repo root).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

OK, BAD, WARN = "ok  ", "FAIL", "warn"


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def __call__(self, status: str, msg: str) -> None:
        if status == BAD:
            self.failures += 1
        elif status == WARN:
            self.warnings += 1
        print(f"  [{status}] {msg}")

    def check(self, cond: bool, msg: str, warn_only: bool = False) -> bool:
        self(OK if cond else (WARN if warn_only else BAD), msg)
        return cond


# ---------------------------------------------------------------- scaffold


def scaffold(args: argparse.Namespace) -> int:
    """Write the skeleton.  Order matters and is the reason this exists.

    `ReactionSet.load` opens the `.jsonl` of *every* entry in the manifest,
    reactions included, before anything else happens -- but a reaction's
    coupling terms are what `scripts/fit.py` produces, and `fit.py` builds a
    `ReactionSet` to do it.  A dataset with real geometries and no reaction
    `.jsonl` therefore cannot be loaded by the tool whose job is to write them.
    Seeding placeholders breaks the cycle; `fit.py` overwrites them.
    """
    root = Path(args.directory).resolve()
    name = args.name or root.name
    (root / "molecules").mkdir(parents=True, exist_ok=True)
    (root / "reactions").mkdir(parents=True, exist_ok=True)

    manifest = {
        "name": name,
        "description": args.description or f"{name} reaction set",
        "version": "1.0.0",
        "author": args.author or "",
        "molecules": [
            {"id": i, "path": f"molecules/{m}"} for i, m in enumerate(args.molecules, 1)
        ],
        "reactions": [
            {"id": i, "path": f"reactions/{r}"} for i, r in enumerate(args.reactions, 1)
        ],
    }
    path = root / f"{name}.json"
    path.write_text(json.dumps(manifest, indent=4) + "\n")
    print(f"wrote {path}")

    for entry in args.reactions:
        stem = root / "reactions" / entry
        jsonl = stem.with_suffix(".jsonl")
        xyz = stem.with_suffix(".xyz")
        if jsonl.exists():
            print(f"kept  {jsonl} (exists)")
            continue
        natoms = 0
        if xyz.exists() and xyz.stat().st_size:
            from ase import io

            natoms = len(io.read(xyz, index=":")[0])
        if not natoms:
            print(f"SKIP  {jsonl}: write {xyz.name} first, then re-run scaffold")
            continue
        term = {
            "type": "rmsd",
            "atoms": {f"p{i + 1}": i for i in range(natoms)},
            "kwargs": {"A": -1.0, "a": 10.0},
            "provenance": "placeholder",
        }
        # No trailing newline: `ReactionSet.load` uses `readlines()` +
        # `json.loads`, and a trailing blank line raises JSONDecodeError.
        jsonl.write_text(json.dumps(term))
        print(f"wrote {jsonl} ({natoms} atoms, placeholder coupling)")
    return 0


# ------------------------------------------------------------------- check


def _read_jsonl(path: Path, r: Report) -> list[dict] | None:
    raw = path.read_text()
    # Measured: one trailing newline loads fine; a trailing *blank* line makes
    # `ReactionSet.load`'s readlines() + json.loads raise JSONDecodeError.
    # The shipped datasets carry neither.
    if raw.endswith("\n\n"):
        r(BAD, f"{path.name}: ends with a blank line -> JSONDecodeError in ReactionSet.load")
    try:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError as exc:
        r(BAD, f"{path.name}: {exc}")
        return None


def _check_terms(terms: list[dict], natoms: int, path: Path, r: Report) -> None:
    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for t in terms:
        missing = {"type", "atoms", "kwargs"} - set(t)
        if missing:
            r(BAD, f"{path.name}: term missing {sorted(missing)}: {t}")
            continue
        by_type[t["type"]].append(t)

    for ttype, group in sorted(by_type.items()):
        # `Topology.set_terms` transposes a term list into dense arrays and
        # raises if one term of a type states a kwarg its siblings do not.
        keysets = {frozenset(t["kwargs"]) for t in group}
        r.check(
            len(keysets) == 1,
            f"{path.name}: every '{ttype}' term states the same kwargs"
            + ("" if len(keysets) == 1 else f" -- got {[sorted(k) for k in keysets]}"),
        )
        widths = {len(t["atoms"]) for t in group}
        r.check(
            len(widths) == 1,
            f"{path.name}: every '{ttype}' term has the same atom count"
            + ("" if len(widths) == 1 else f" -- got {sorted(widths)}"),
        )
        for t in group:
            bad = [i for i in t["atoms"].values() if not 0 <= int(i) < natoms]
            if bad:
                r(BAD, f"{path.name}: '{ttype}' indexes atoms {bad} outside 0..{natoms - 1}")

    # ACKS2 runs once over the whole system and raises KeyError for *everything*
    # if any molecule in the box lacks per-atom terms.
    covered = {int(i) for t in by_type.get("atom", []) for i in t["atoms"].values()}
    # Measured: a template missing one `atom` term does NOT raise. ACKS2 just
    # equilibrates charge over the atoms it was given, and the energy comes out
    # silently wrong -- 1.15 eV on a 7-atom Zundel.
    r.check(
        covered == set(range(natoms)),
        f"{path.name}: an 'atom' (ACKS2) term for each of {natoms} atoms"
        + ("" if covered == set(range(natoms)) else f" -- missing {sorted(set(range(natoms)) - covered)}"),
    )
    return None


def check(args: argparse.Namespace) -> int:
    from ase import Atoms, io

    r = Report()
    mpath = Path(args.manifest).resolve()
    manifest = json.loads(mpath.read_text())
    root = mpath.parent
    print(f"\n== manifest {mpath}")
    for key in ("molecules", "reactions"):
        r.check(key in manifest, f"manifest has '{key}'")
    ids = [e["id"] for e in manifest.get("molecules", [])]
    r.check(len(ids) == len(set(ids)), "molecule ids unique")
    ids = [e["id"] for e in manifest.get("reactions", [])]
    r.check(len(ids) == len(set(ids)), "reaction ids unique")

    print("\n== molecules")
    for entry in manifest.get("molecules", []):
        stem = root / entry["path"]
        xyz, jsonl = stem.with_suffix(".xyz"), stem.with_suffix(".jsonl")
        if not r.check(xyz.exists() and xyz.stat().st_size > 0, f"{xyz.name} exists and is non-empty"):
            continue
        if not r.check(jsonl.exists(), f"{jsonl.name} exists"):
            continue
        frames = io.read(xyz, index=":")
        # `load` asserts isinstance(atoms, Atoms): a multi-frame molecule file
        # fails there with a bare AssertionError.
        r.check(len(frames) == 1, f"{xyz.name}: exactly 1 frame (got {len(frames)})")
        atoms: Atoms = frames[0]
        try:
            atoms.get_potential_energy()
            has_e = True
        except Exception:
            has_e = False
        r.check(
            has_e,
            f"{xyz.name}: carries energy= (atomization, eV). Without it no "
            "reference shift is synthesized and states are compared on "
            "different zeros.",
            warn_only=True,
        )
        terms = _read_jsonl(jsonl, r)
        if terms is not None:
            _check_terms(terms, len(atoms), jsonl, r)

    print("\n== reactions")
    for entry in manifest.get("reactions", []):
        stem = root / entry["path"]
        xyz, jsonl = stem.with_suffix(".xyz"), stem.with_suffix(".jsonl")
        if not r.check(xyz.exists() and xyz.stat().st_size > 0, f"{xyz.name} exists and is non-empty"):
            continue
        r.check(
            jsonl.exists(),
            f"{jsonl.name} exists (load opens it before fit.py can write it -- "
            "use `scaffold` to seed a placeholder)",
        )
        frames = io.read(xyz, index=":")
        # Measured: 2 frames does NOT raise. `Reaction.from_atoms` slices
        # atoms[1:-1] for the TS ensemble, so it becomes empty, the reaction
        # degrades to a no-op (reactant == product in its own equation) and the
        # force call returns a silently wrong energy.
        r.check(
            len(frames) >= 3,
            f"{xyz.name}: >=3 frames [reactant, TS..., product] (got {len(frames)}); "
            "fewer gives an empty TS ensemble and a silently wrong energy",
        )
        if len(frames) < 3:
            continue
        sym = [tuple(f.get_chemical_symbols()) for f in frames]
        r.check(len(set(sym)) == 1, f"{xyz.name}: identical atom count and ordering in every frame")
        for label, idx in (("reactant", 0), ("product", -1)):
            conn = frames[idx].info.get("connectivity")
            r.check(
                conn is not None,
                f"{xyz.name}: {label} frame carries connectivity= "
                "(this is what defines the topology; without it bonds are "
                "perceived from distance at BOND_SCALE=1.3)",
            )
        r0 = frames[0].info.get("connectivity")
        rp = frames[-1].info.get("connectivity")
        if r0 is not None and rp is not None:
            same = {frozenset(b[:2]) for b in r0} == {frozenset(b[:2]) for b in rp}
            r.check(not same, f"{xyz.name}: reactant and product bonding differ")
        ts = frames[len(frames) // 2]
        try:
            ts.get_potential_energy()
            has_e = True
        except Exception:
            has_e = False
        r.check(
            has_e,
            f"{xyz.name}: TS frame carries energy= (fit.py needs it to invert "
            "the amplitude; without it the channel is skipped)",
            warn_only=True,
        )
        if jsonl.exists():
            terms = _read_jsonl(jsonl, r)
            if terms:
                rmsd = [t for t in terms if t.get("type") == "rmsd"]
                r.check(len(rmsd) == 1, f"{jsonl.name}: exactly one 'rmsd' term")
                if rmsd:
                    r.check(
                        len(rmsd[0]["atoms"]) == len(frames[0]),
                        f"{jsonl.name}: rmsd term spans all {len(frames[0])} atoms",
                    )

    print(f"\n{r.failures} failure(s), {r.warnings} warning(s)")
    return 1 if r.failures else 0


# ------------------------------------------------------------------- probe


def probe(args: argparse.Namespace) -> int:
    """Reaction table, species closure and energy bookkeeping.

    The bookkeeping block is the one that catches a charged dataset: nothing
    raises if a template's `energy=` is on the wrong electron count, the
    reaction energies are just silently wrong by an ionization potential.
    """
    from ase import io

    from DynamicTopology.core import ReactionSet
    from DynamicTopology.core.topology import Topology

    mpath = Path(args.manifest).resolve()
    manifest = json.loads(mpath.read_text())
    rs = ReactionSet(mpath)

    print("\n== templates")
    energies: dict[str, float] = {}
    for entry in manifest["molecules"]:
        stem = mpath.parent / entry["path"]
        atoms = io.read(stem.with_suffix(".xyz"))
        try:
            e = atoms.get_potential_energy()
        except Exception:
            e = float("nan")
        formula = atoms.get_chemical_formula()
        energies[formula] = e
        print(f"  {stem.name:<14} {formula:<8} E_atomization = {e:>12.6f} eV")

    print("\n== reaction table (database key -> stored templates)")
    for h, lst in rs.data["reactions"].items():
        flag = "  <-- degenerate: stored twice, see SKILL.md" if len(lst) > 1 and lst[0].equation() == lst[1].reverse().equation() else ""
        print(f"  {h[:10]}  n={len(lst)}  {[x.equation() for x in lst]}{flag}")

    print("\n== energy bookkeeping (templates vs frames)")
    print(f"  {'reaction':<26}{'dE templates':>14}{'dE frames':>12}{'TS-R':>9}{'TS-P':>9}")
    for entry in manifest["reactions"]:
        stem = mpath.parent / entry["path"]
        frames = io.read(stem.with_suffix(".xyz"), index=":")
        ts = frames[len(frames) // 2]

        def formulas(frame):
            t = Topology.from_atoms(frame)
            return [
                frame[sorted(m.graph.nodes)].get_chemical_formula()
                for m in t.molecules()
            ]

        try:
            d_tpl = sum(energies.get(f, float("nan")) for f in formulas(frames[-1])) - sum(
                energies.get(f, float("nan")) for f in formulas(frames[0])
            )
        except Exception:
            d_tpl = float("nan")
        try:
            eR, eP, eT = (f.get_potential_energy() for f in (frames[0], frames[-1], ts))
            d_frm, dR, dP = eP - eR, eT - eR, eT - eP
        except Exception:
            d_frm = dR = dP = float("nan")
        print(f"  {stem.name:<26}{d_tpl:>14.4f}{d_frm:>12.4f}{dR:>9.4f}{dP:>9.4f}")
    print(
        "\n  dE templates is what the force field will reproduce for separated\n"
        "  fragments; dE frames includes the complexes' binding.  A large\n"
        "  disagreement in *sign* means the energy zeros do not line up --\n"
        "  most often an ion computed against the wrong electron count."
    )

    print("\n== species closure")
    missing = set()
    for entry in manifest["reactions"]:
        stem = mpath.parent / entry["path"]
        frames = io.read(stem.with_suffix(".xyz"), index=":")
        for frame in (frames[0], frames[-1]):
            for mol in Topology.from_atoms(frame).molecules():
                if rs.hash_molecule(mol) not in rs.data["molecules"]:
                    missing.add(frame[sorted(mol.graph.nodes)].get_chemical_formula())
    if missing:
        print(f"  [FAIL] no template for {sorted(missing)} -- add them to the manifest")
        return 1
    print("  [ok  ] every reactant/product fragment has a template")
    return 0


# --------------------------------------------------------------------- run


def run(args: argparse.Namespace) -> int:
    """Load it, pack a box, build the network, call the force field, integrate.

    Nothing before this proves the dataset works.
    """
    from ase import io, units
    from ase.md.velocitydistribution import thermalize_momenta
    from ase.md.verlet import VelocityVerlet
    from molify import pack

    from DynamicTopology.ase import DynamicTopology
    from DynamicTopology.core import ReactionSet
    from DynamicTopology.core.topology import Topology

    mpath = Path(args.manifest).resolve()
    manifest = json.loads(mpath.read_text())
    rs = ReactionSet(mpath)
    print(f"loaded {mpath.name}: {len(rs.data['molecules'])} templates, "
          f"{sum(len(v) for v in rs.data['reactions'].values())} stored reactions")

    stems = {(mpath.parent / e["path"]).name: mpath.parent / e["path"] for e in manifest["molecules"]}
    if args.template not in stems:
        print(f"--template must be one of {sorted(stems)}")
        return 1
    seed = io.read(stems[args.template].with_suffix(".xyz"))

    box = pack([[seed]], [args.count], args.density, seed=args.seed)
    # The packed box inherits the template's connectivity, which indexes the
    # template's atoms, not the box's.  Drop it so the topology is perceived.
    box.info.pop("connectivity", None)
    edge = box.cell.lengths()[0]
    print(f"box: {args.count} x {seed.get_chemical_formula()} = {len(box)} atoms, "
          f"{edge:.2f} A cube at {args.density} kg/m^3")

    topo = Topology.from_atoms(box)
    species = collections.Counter(
        box[sorted(m.graph.nodes)].get_chemical_formula() for m in topo.molecules()
    )
    print(f"perceived species: {dict(species)}")

    t0 = time.time()
    net = rs.get_network(topo, bimol_cutoff=args.cutoff)
    t_net = time.time() - t0
    channels = collections.Counter(d["reaction"].equation() for _, _, d in net.graph.edges(data=True))
    print(f"network: {net.graph.number_of_nodes()} nodes, {net.graph.number_of_edges()} edges "
          f"({t_net:.2f}s at cutoff {args.cutoff} A)")
    for eq, n in sorted(channels.items(), key=lambda kv: -kv[1]):
        print(f"    {n:>6}  {eq}")

    box.calc = DynamicTopology(box, rs)
    t0 = time.time()
    energy = box.get_potential_energy()
    forces = box.get_forces()
    t_force = time.time() - t0
    finite = bool(np.isfinite(forces).all()) and np.isfinite(energy)
    print(f"\nforce call: {t_force:.2f}s   E = {energy:.6f} eV   "
          f"|F|max = {np.abs(forces).max():.4f} eV/A   finite = {finite}")
    diag = {k: v for k, v in box.calc.diagnostics.items() if k != "blocks"}
    print(f"    {diag}")
    if not finite:
        print("[FAIL] non-finite energy or forces")
        return 1

    # Seeded so the reported drift is reproducible run to run.
    np.random.seed(args.seed)
    thermalize_momenta(box, temperature_K=args.temperature)
    e0 = box.get_total_energy()
    t0 = time.time()
    VelocityVerlet(box, timestep=args.timestep * units.fs).run(args.steps)
    drift = box.get_total_energy() - e0
    print(f"\nNVE {args.steps} x {args.timestep} fs ({time.time() - t0:.1f}s): "
          f"drift = {drift * 1000:.2f} meV ({drift / len(box) * 1000:+.4f} meV/atom)")
    print("    A drift far above ~0.01 meV/atom means the timestep is wrong for")
    print("    this surface -- read the 'fastest mode' line from scripts/fit.py.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scaffold", help="write manifest + skeleton + placeholder couplings")
    s.add_argument("directory")
    s.add_argument("--name")
    s.add_argument("--description")
    s.add_argument("--author")
    s.add_argument("-m", "--molecules", nargs="*", default=[])
    s.add_argument("-x", "--reactions", nargs="*", default=[])
    s.set_defaults(func=scaffold)

    s = sub.add_parser("check", help="static validation, no force field")
    s.add_argument("manifest")
    s.set_defaults(func=check)

    s = sub.add_parser("probe", help="reaction table, closure, energy bookkeeping")
    s.add_argument("manifest")
    s.set_defaults(func=probe)

    s = sub.add_parser("run", help="pack a box and actually integrate it")
    s.add_argument("manifest")
    s.add_argument("--template", required=True, help="molecule stem to pack, e.g. h2o")
    s.add_argument("-n", "--count", type=int, default=64)
    s.add_argument("-d", "--density", type=float, default=997.0)
    s.add_argument("--cutoff", type=float, default=4.0)
    s.add_argument("--steps", type=int, default=40)
    s.add_argument("--timestep", type=float, default=0.5)
    s.add_argument("--temperature", type=float, default=300.0)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=run)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

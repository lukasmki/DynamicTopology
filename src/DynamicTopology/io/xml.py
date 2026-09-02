from pathlib import Path
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from DynamicTopology.core.types import Term


def read_xml(path: str | Path) -> list[Term]:
    if isinstance(path, str):
        path = Path(path).resolve()
    with open(path) as fp:
        return read_xmls(fp.read())


def read_xmls(data: str) -> list[Term]:
    terms: list[Term] = []

    root = ET.fromstring(data)
    forces = root.find("Forces")
    if forces is None:
        return []

    for force in forces.findall("Force"):
        # get term name
        name = force.get("name")
        if name is None:
            raise ValueError("Term with no name")

        atom_params = force.find("PerParticleParameters")
        bond_params = force.find("PerBondParameters")
        angle_params = force.find("PerAngleParameters")
        torsion_params = force.find("PerTorsionParameters")
        # only one of these will not be None
        try:
            params = next(
                p
                for p in (
                    atom_params,
                    bond_params,
                    angle_params,
                    torsion_params,
                )
                if p is not None
            )
        except StopIteration:
            params = None

        parameter_map: dict[str, str] = {}
        # has_named_params = params is not None
        if params is not None:
            params: Element[str]
            # map parameter indices to named parameters
            for i, param in enumerate(params.findall("Parameter")):
                param_name = param.get("name")
                if param_name is None:
                    # skip if not named
                    continue
                parameter_map[f"param{i + 1}"] = param_name
        else:
            pass

        # get term lines
        atoms = force.find("Particles")
        bonds = force.find("Bonds")
        angles = force.find("Angles")
        torsions = force.find("Torsions")
        dofs: Element[str] = next(
            d
            for d in (
                atoms,
                bonds,
                angles,
                torsions,
            )
            if d is not None
        )

        # only one of these will not be none
        if dofs is None:
            # skip if no particles are present
            continue

        # loop over children
        for position, dof in enumerate(dofs):
            attrib: dict[str, str] = dof.attrib
            keys: list[str] = list(attrib.keys())
            args: dict[str, float] = {
                k: float(attrib.pop(k)) for k in keys if k.startswith("param")
            }

            if params is not None:
                for k in parameter_map.keys():
                    args[parameter_map[k]] = args.pop(k)
                # A <Bond>/<Angle>/<Torsion> names its atoms explicitly ("p1",
                # "p2", ...), but a <Particle> does not: its index is its
                # position in the list, and every attribute it carries is a
                # parameter.  Reading the indices off `attrib` therefore left
                # every per-particle force with an empty `atoms` dict, which is
                # how q-force's Lennard-Jones parameters were parsed for years
                # without ever being attachable to an atom.
                if dofs.tag == "Particles":
                    idx: dict[str, int] = {"p0": position}
                else:
                    idx = {k: int(v) for k, v in attrib.items()}
                term: Term = {"type": name.lower(), "atoms": idx, "kwargs": args}
                terms.append(term)
    return terms

from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System


class DynamicTopology(Calculator):
    implemented_properties: list[str] = ["energy", "forces"]

    def __init__(
        self,
        atoms: Atoms,
        reaction_set: ReactionSet,
        **kwargs,
    ):
        super().__init__(**kwargs)
        topology = Topology.from_atoms(atoms)
        self.system = System(atoms, topology, reaction_set)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy", "forces"],
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        self.system.update(atoms=atoms)
        results: dict[str, Any] = self.system.calculate()
        self.system.update(topology=results["topology"])

        self.results: dict[str, np.ndarray] = {
            "energy": results["energy"].detach().cpu().numpy(),
            "forces": results["forces"].detach().cpu().numpy(),
        }

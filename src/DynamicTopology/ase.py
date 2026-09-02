from DynamicTopology.evb import EVBSystem
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

        previous = self.system.topology
        previous_edges = (
            frozenset(frozenset(edge) for edge in previous.graph.edges())
            if self.system.topology is not None
            else None
        )
        current_edges = (
            frozenset(frozenset(edge) for edge in results["topology"].graph.edges())
            if results["topology"] is not None
            else None
        )
        self.topology_changed: bool = previous_edges != current_edges
        self.system.update(topology=results["topology"])

        # Not in implemented_properties: these are diagnostics, not ASE
        # properties, and are read off the calculator directly.
        self.diagnostics: dict[str, Any] = {
            "energy_bonded": results["energy_bonded"],
            "energy_nonbonded": results["energy_nonbonded"],
            "energy_zbl": results["energy_zbl"],
            "blocks": results["blocks"],
            "topology_changed": self.topology_changed,
        }

        self.results: dict[str, np.ndarray] = {
            "energy": results["energy"],
            "forces": results["forces"],
        }


class EVB(Calculator):
    implemented_properties: list[str] = ["energy", "forces"]

    def __init__(
        self,
        atoms: Atoms,
        reaction_set: ReactionSet,
        **kwargs,
    ):
        super().__init__(**kwargs)
        topology = Topology.from_atoms(atoms)

        network = reaction_set.get_network(topology)
        states = [state for subnet in network.states() for rxn_data, state in subnet]

        for state in states:
            terms = reaction_set.get_terms(state)
            state.set_terms(terms)

        self.system = EVBSystem(atoms, states=states, reaction_set=reaction_set)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] = ["energy", "forces"],
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        self.system.update(atoms=atoms)
        results: dict[str, Any] = self.system.calculate()

        self.results: dict[str, np.ndarray] = {
            "energy": results["energy"],
            "forces": results["forces"],
        }

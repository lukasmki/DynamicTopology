from DynamicTopology.evb import EVBSystem
from typing import Any

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from DynamicTopology.core import ReactionSet, Topology
from DynamicTopology.system import System


class DynamicTopology(Calculator):
    # `stress` is what every ASE barostat asks for, so an NPT run needs it
    # present here or it raises before taking a step.  See `tests/test_stress.py`
    # for how the virial behind it is built and checked.
    implemented_properties: list[str] = ["energy", "forces", "stress"]

    def __init__(
        self,
        atoms: Atoms,
        reaction_set: ReactionSet,
        bimol_cutoff: float = 4.0,
        evb: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        topology = Topology.from_atoms(atoms)
        # `bimol_cutoff` and the `EVBBasis` knobs in `evb` were previously
        # unreachable through the calculator -- `**kwargs` goes to ASE's
        # `Calculator`, not to `System` -- so a run could not tighten the basis
        # without reaching into `calc.system` after the fact.  Defaults are
        # unchanged.
        self.system = System(
            atoms, topology, reaction_set, bimol_cutoff=bimol_cutoff, evb=evb
        )

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
            "energy_lj": results["energy_lj"],
            "blocks": results["blocks"],
            "topology_changed": self.topology_changed,
        }

        self.results: dict[str, np.ndarray] = {
            "energy": results["energy"],
            "forces": results["forces"],
        }

        # Stress is the virial per unit volume.  A non-periodic system has no
        # volume to divide by and no stress to report -- offering one would be a
        # divide-by-zero at best and a meaningless number at worst -- so the key
        # is simply absent there, which is what ASE expects of an unavailable
        # property.
        volume = float(self.system.atoms.get_volume())
        if volume > 0.0:
            self.results["stress"] = full_3x3_to_voigt_6_stress(
                results["virial"] / volume
            )


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

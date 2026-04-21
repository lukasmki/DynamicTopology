from DynamicTopology.forcefield.coupling_np import EVBCoupling
import numpy as np
from DynamicTopology.core import ReactionSet, Topology
import cProfile
import pstats
from ase import Atoms, io


reaction_set = ReactionSet("datasets/HCombustion/HCombustion.json")
atoms = io.read("tests/data/mix-n100-d250.xyz", index=0)
assert isinstance(atoms, Atoms)

topo = Topology.from_atoms(atoms)
terms = reaction_set.get_terms_topology(topo)
topo.set_terms(terms)

coupling = EVBCoupling()
network = reaction_set.get_network(topo, 5.0)
reactions = network.reactions()

profiler = cProfile.Profile()
profiler.enable()

for rxn in reactions:
    imol, jmol, rxn_data = rxn
    reaction = rxn_data["reaction"]
    mapping: dict = rxn_data["mapping"]

    reactant = atoms.positions[list(mapping.values())]
    ensemble = np.stack([a.positions for a in reaction.atoms])
    energy = coupling(reactant, atoms.pbc, atoms.cell, ensemble, reaction.term_dict)

profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumtime").strip_dirs().print_stats()

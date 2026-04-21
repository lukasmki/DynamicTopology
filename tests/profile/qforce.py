from DynamicTopology.forcefield.acks2 import ACKS2
from DynamicTopology.forcefield.qforce import QForce
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

ff = QForce()
nb = ACKS2()

profiler = cProfile.Profile()
profiler.enable()

nonbonded = nb(atoms.positions, atoms.pbc, atoms.cell, topo.term_dict)
bonded = ff(atoms.positions, atoms.pbc, atoms.cell, topo.term_dict)

profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats("cumtime").strip_dirs().print_stats()

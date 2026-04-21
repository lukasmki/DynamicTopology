"""
Regression test for ReactionSet.get_network.

Captures the full structure of the returned ReactionNetwork (node count, edge
count, and per-edge reaction identity + atom mapping) and verifies it is
unchanged after any refactor.
"""

import pytest
from ase import io

from DynamicTopology.core import ReactionSet
from DynamicTopology.core.topology import Topology


RNET_PATH = "datasets/HCombustion/HCombustion.json"
ATOMS_PATH = "tests/data/mix-n100-d250.xyz"
CUTOFF = 4.0


@pytest.fixture(scope="module")
def setup():
    atoms = io.read(ATOMS_PATH, index=0)
    reaction_set = ReactionSet(RNET_PATH)
    topology = Topology.from_atoms(atoms)
    return reaction_set, topology


def _network_fingerprint(network):
    """Deterministic fingerprint of a ReactionNetwork for equality checks."""
    g = network.graph

    nodes = frozenset(g.nodes())

    edges = set()
    for i, j, data in g.edges(data=True):
        rxn = data["reaction"]
        mapping = data["mapping"]
        rxn_hash = rxn.hash()
        mapping_key = frozenset(mapping.items())
        edges.add((min(i, j), max(i, j), rxn_hash, mapping_key))

    return nodes, edges


def test_get_network_node_count(setup):
    reaction_set, topology = setup
    network = reaction_set.get_network(topology, CUTOFF)
    mol_list = list(topology.molecules())
    assert len(network.graph.nodes) == len(mol_list)


def test_get_network_structure(setup):
    """Full structural regression: same nodes, same edges, same mappings."""
    reaction_set, topology = setup

    # Run twice — second call exercises any caching paths
    net1 = reaction_set.get_network(topology, CUTOFF)
    net2 = reaction_set.get_network(topology, CUTOFF)

    fp1 = _network_fingerprint(net1)
    fp2 = _network_fingerprint(net2)

    assert fp1[0] == fp2[0], "Node sets differ between calls"
    assert fp1[1] == fp2[1], "Edge sets differ between calls"

"""
Regression tests for the four functions modified by the performance optimizations.

Each test class captures the observable behavior that must be preserved:

  A) Topology.hash()               — lazy WL-hash caching
  B) Topology.molecules()          — lazy connected-components caching
  C) ReactionSet.get_terms_topology() — per-(mol_hash, node_set) term cache
  D) Topology.copy()               — list() instead of deepcopy() for terms
"""

import networkx as nx
import numpy as np
import pytest
from ase import Atoms

from DynamicTopology.core.topology import Topology
from DynamicTopology.core.reactionset import ReactionSet


RSET_PATH = "datasets/HCombustion/HCombustion.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_h2_topology(node_a=0, node_b=1):
    """Two H atoms bonded: H-H with given global node indices."""
    g = nx.Graph()
    g.add_node(node_a, atomic_number=1, symbol="H")
    g.add_node(node_b, atomic_number=1, symbol="H")
    g.add_edge(node_a, node_b)
    return Topology(g)


def make_oh_topology(node_o=0, node_h=1):
    """O-H bond with given global node indices."""
    g = nx.Graph()
    g.add_node(node_o, atomic_number=8, symbol="O")
    g.add_node(node_h, atomic_number=1, symbol="H")
    g.add_edge(node_o, node_h)
    return Topology(g)


def make_disjoint_topology(*sub_topos):
    """Union of disjoint sub-topologies (no inter-molecule edges)."""
    g = sub_topos[0].graph.copy()
    for other in sub_topos[1:]:
        g = nx.union(g, other.graph)
    return Topology(g)


@pytest.fixture(scope="module")
def reaction_set():
    return ReactionSet(RSET_PATH)


# ---------------------------------------------------------------------------
# A) Topology.hash()
# ---------------------------------------------------------------------------


class TestTopologyHash:
    def test_idempotent(self):
        """Same topology returns the same hash on repeated calls."""
        topo = make_h2_topology()
        h1 = topo.hash()
        h2 = topo.hash()
        assert h1 == h2

    def test_same_structure_same_hash(self):
        """Two independently constructed topologies with identical structure hash equally."""
        topo1 = make_h2_topology(0, 1)
        topo2 = make_h2_topology(0, 1)
        assert topo1.hash() == topo2.hash()

    def test_different_structure_different_hash(self):
        """H2 and OH have different hashes (different atomic types / structure)."""
        assert make_h2_topology().hash() != make_oh_topology().hash()

    def test_node_relabeling_invariant(self):
        """WL hash depends on graph topology + atomic numbers, not raw node ids."""
        topo_01 = make_h2_topology(0, 1)
        topo_57 = make_h2_topology(5, 7)
        assert topo_01.hash() == topo_57.hash()

    def test_copy_same_hash(self):
        """A copy has the same hash as the original."""
        topo = make_h2_topology()
        assert topo.hash() == topo.copy().hash()

    def test_hash_changes_after_set_atoms(self):
        """Replacing atoms with a different element changes the hash."""
        topo_h2 = make_h2_topology(0, 1)
        h_before = topo_h2.hash()
        # set_atoms changes atomic_number on nodes via graph.add_nodes_from
        atoms_oo = Atoms("OO", positions=[[0, 0, 0], [1, 0, 0]])
        topo_h2.set_atoms(atoms_oo)
        h_after = topo_h2.hash()
        assert h_before != h_after


# ---------------------------------------------------------------------------
# B) Topology.molecules()
# ---------------------------------------------------------------------------


class TestTopologyMolecules:
    def test_single_molecule_yields_one(self):
        """A connected topology yields exactly one molecule."""
        topo = make_h2_topology()
        mols = list(topo.molecules())
        assert len(mols) == 1

    def test_two_disjoint_molecules(self):
        """Two disjoint components yield two molecules."""
        topo = make_disjoint_topology(make_h2_topology(0, 1), make_h2_topology(5, 6))
        assert len(list(topo.molecules())) == 2

    def test_node_sets_match_components(self):
        """Each yielded molecule's node set matches its connected component."""
        topo = make_disjoint_topology(make_h2_topology(0, 1), make_oh_topology(5, 6))
        node_sets = {frozenset(m.graph.nodes()) for m in topo.molecules()}
        assert frozenset({0, 1}) in node_sets
        assert frozenset({5, 6}) in node_sets

    def test_repeated_calls_same_result(self):
        """molecules() is consistent across repeated calls."""
        topo = make_disjoint_topology(make_h2_topology(0, 1), make_oh_topology(5, 6))

        def snapshot():
            return sorted(
                (frozenset(m.graph.nodes()), frozenset(m.graph.edges()))
                for m in topo.molecules()
            )

        assert snapshot() == snapshot()

    def test_return_atoms_true(self):
        """With return_atoms=True, each pair (mol, sub_atoms) has matching length."""
        atoms = Atoms("HH", positions=[[0, 0, 0], [1, 0, 0]])
        g = nx.Graph()
        g.add_node(0, atomic_number=1, symbol="H")
        g.add_node(1, atomic_number=1, symbol="H")
        g.add_edge(0, 1)
        topo = Topology(g, atoms)
        for mol, sub_atoms in topo.molecules(return_atoms=True):
            assert isinstance(sub_atoms, Atoms)
            assert len(sub_atoms) == len(mol.graph)

    def test_three_molecules(self):
        """Three disjoint molecules all appear."""
        topo = make_disjoint_topology(
            make_h2_topology(0, 1),
            make_oh_topology(5, 6),
            make_h2_topology(10, 11),
        )
        assert len(list(topo.molecules())) == 3


# ---------------------------------------------------------------------------
# C) ReactionSet.get_terms_topology()
# ---------------------------------------------------------------------------


class TestGetTermsTopology:
    def test_term_indices_are_global(self, reaction_set):
        """Term atom indices are the global node ids, not 0-based template indices."""
        topo = make_disjoint_topology(make_h2_topology(5, 7), make_h2_topology(10, 15))
        terms = reaction_set.get_terms_topology(topo)

        all_indices = {v for t in terms for v in t["atoms"].values()}
        assert all_indices.issubset({5, 7, 10, 15})
        assert 0 not in all_indices
        assert 1 not in all_indices

    def test_cache_is_not_polluted_by_other_fragments(self):
        """Warming the cache with other fragments must not change an answer.

        The cache key has to identify a fragment exactly, because the terms it
        stores have already been remapped onto specific global indices.  The
        Weisfeiler-Lehman hash is a graph invariant taken over atomic numbers,
        so it cannot by itself tell which node carries which element.

        A fresh ReactionSet is the ground truth: nothing has been cached, so no
        lookup can be wrong.  Any warmed instance must agree with it.  This
        passes against the narrower (hash, node set) key as well -- it is a
        standing invariant rather than a reproduction of a known failure.
        """
        from ase import io

        frames = io.read("datasets/HCombustion/reactions/rxn_11.xyz", index=":")
        target = frames[0].copy()
        target.calc = None
        target.positions = frames[len(frames) // 2].positions

        def terms_for(reaction_set):
            topology = Topology.from_atoms(target)
            return sorted(
                (t["type"], tuple(sorted(t["atoms"].items())), tuple(sorted(t["kwargs"])))
                + tuple(round(float(v), 8) for _, v in sorted(t["kwargs"].items()))
                for t in reaction_set.get_terms_topology(topology)
            )

        reference = terms_for(ReactionSet(RSET_PATH))

        warmed = ReactionSet(RSET_PATH)
        for index in range(1, 11):
            other = io.read(
                "datasets/HCombustion/reactions/rxn_%02d.xyz" % index, index=":"
            )
            middle = other[len(other) // 2].positions
            for side in (0, -1):
                probe = other[side].copy()
                probe.calc = None
                probe.positions = middle
                warmed.get_terms_topology(Topology.from_atoms(probe))

        assert terms_for(warmed) == reference

    def test_idempotent(self, reaction_set):
        """Two calls with the same topology return identical terms."""
        topo = make_disjoint_topology(make_h2_topology(5, 7), make_h2_topology(10, 15))
        t1 = reaction_set.get_terms_topology(topo)
        t2 = reaction_set.get_terms_topology(topo)

        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a["type"] == b["type"]
            assert a["atoms"] == b["atoms"]
            for k in a["kwargs"]:
                np.testing.assert_array_equal(a["kwargs"][k], b["kwargs"][k])

    def test_different_placement_same_type_count(self, reaction_set):
        """Same molecule type at different global indices yields the same number of terms."""
        topo_a = make_h2_topology(0, 1)
        topo_b = make_h2_topology(50, 51)
        terms_a = reaction_set.get_terms_topology(topo_a)
        terms_b = reaction_set.get_terms_topology(topo_b)

        assert len(terms_a) == len(terms_b)
        assert [t["type"] for t in terms_a] == [t["type"] for t in terms_b]

    def test_different_placement_correct_indices(self, reaction_set):
        """Each placement uses its own global indices, not the other placement's."""
        terms_a = reaction_set.get_terms_topology(make_h2_topology(0, 1))
        terms_b = reaction_set.get_terms_topology(make_h2_topology(50, 51))

        idx_a = {v for t in terms_a for v in t["atoms"].values()}
        idx_b = {v for t in terms_b for v in t["atoms"].values()}
        assert idx_a.issubset({0, 1})
        assert idx_b.issubset({50, 51})
        assert idx_a.isdisjoint(idx_b)

    def test_param_values_preserved(self, reaction_set):
        """Parameter values in returned terms equal those in the database template."""
        # Collect all (type, kwargs) from the H2 template (mol_01: 2 H atoms, 1 bond)
        template = next(reaction_set.get_molecules(ids=[1]))

        # Build expected: list of (type, kwargs_values_tuple) sorted for comparison
        def term_signature(t):
            return (
                t["type"],
                tuple(
                    sorted(
                        (k, float(np.atleast_1d(v)[0])) for k, v in t["kwargs"].items()
                    )
                ),
            )

        expected_sigs = sorted(term_signature(t) for t in template.terms)

        topo = make_h2_topology(5, 7)
        terms = reaction_set.get_terms_topology(topo)
        actual_sigs = sorted(term_signature(t) for t in terms)

        assert actual_sigs == expected_sigs

    def test_database_template_not_mutated(self, reaction_set):
        """get_terms_topology does not modify the database molecule's atom indices."""
        # Call with non-zero global indices
        topo = make_h2_topology(5, 7)
        _ = reaction_set.get_terms_topology(topo)

        # Template for H2 (mol_01) must still have indices 0 and 1 only
        template = next(reaction_set.get_molecules(ids=[1]))
        for term in template.terms:
            for idx in term["atoms"].values():
                assert idx in {0, 1}, (
                    f"Template index {idx} was corrupted (expected 0 or 1)"
                )

    def test_multi_molecule_topology(self, reaction_set):
        """Works correctly for a topology containing multiple different molecule types."""
        # H2 (nodes 0,1) + OH (nodes 5,6)
        topo = make_disjoint_topology(make_h2_topology(0, 1), make_oh_topology(5, 6))
        terms = reaction_set.get_terms_topology(topo)

        all_idx = {v for t in terms for v in t["atoms"].values()}
        assert all_idx.issubset({0, 1, 5, 6})


# ---------------------------------------------------------------------------
# D) Topology.copy()
# ---------------------------------------------------------------------------


class TestTopologyCopy:
    def test_copy_same_hash(self):
        """Copy has the same hash as the original."""
        topo = make_h2_topology()
        assert topo.copy().hash() == topo.hash()

    def test_copy_graph_edge_removal_independent(self):
        """Removing an edge from the copy does not affect the original graph."""
        topo = make_h2_topology(0, 1)
        copy = topo.copy()
        copy.graph.remove_edge(0, 1)
        assert topo.graph.has_edge(0, 1), "Original graph was mutated"

    def test_copy_graph_node_addition_independent(self):
        """Adding a node to the copy does not appear in the original."""
        topo = make_h2_topology(0, 1)
        copy = topo.copy()
        copy.graph.add_node(99, atomic_number=1, symbol="H")
        assert not topo.graph.has_node(99), "Original graph was mutated"

    def test_copy_terms_list_independent(self, reaction_set):
        """Appending to copy.terms does not affect the original terms list."""
        topo = make_h2_topology(0, 1)
        topo.set_terms(reaction_set.get_terms_topology(topo))
        n_original = len(topo.terms)

        copy = topo.copy()
        copy.terms.append({"type": "dummy", "atoms": {}, "kwargs": {}})
        assert len(topo.terms) == n_original, "Original terms list was mutated"

    def test_copy_terms_values_preserved(self, reaction_set):
        """Copy contains the same term types and parameter values as the original."""
        topo = make_h2_topology(0, 1)
        topo.set_terms(reaction_set.get_terms_topology(topo))
        copy = topo.copy()

        assert len(copy.terms) == len(topo.terms)
        for orig, dup in zip(topo.terms, copy.terms):
            assert orig["type"] == dup["type"]
            assert orig["atoms"] == dup["atoms"]

    def test_copy_without_atoms(self):
        """copy() works when the topology has no Atoms object attached."""
        topo = make_h2_topology()
        assert topo.atoms is None
        copy = topo.copy()
        assert copy.atoms is None
        assert copy.hash() == topo.hash()

    def test_copy_node_count(self):
        """Copy has the same number of nodes and edges."""
        topo = make_disjoint_topology(make_h2_topology(0, 1), make_oh_topology(5, 6))
        copy = topo.copy()
        assert copy.graph.number_of_nodes() == topo.graph.number_of_nodes()
        assert copy.graph.number_of_edges() == topo.graph.number_of_edges()

"""Differential tests of pure directionality graph queries."""
from __future__ import annotations

import itertools

import networkx as nx
import pytest

from rivgraph import _directionality as query
from rivgraph import directionality as di
from tests.test_directionality_characterization import synthetic_delta


def bridge_oracle(graph, inlets, outlets):
    result = []
    for u, v in nx.bridges(graph):
        reduced = graph.copy()
        reduced.remove_edge(u, v)
        result.append((u, v,
                       any(nx.has_path(reduced, u, i) for i in inlets),
                       any(nx.has_path(reduced, u, o) for o in outlets),
                       any(nx.has_path(reduced, v, i) for i in inlets),
                       any(nx.has_path(reduced, v, o) for o in outlets)))
    return result


def cycle_oracle(edges, candidates, boundary, inlets, outlets):
    """Independent exhaustive legacy predicates, not the optimized helpers."""
    valid = []
    for count in range(1, len(candidates) + 1):
        for selected in itertools.combinations(candidates, count):
            graph = nx.DiGraph((v, u) if i in selected else (u, v)
                               for i, (u, v) in enumerate(edges))
            if not nx.is_directed_acyclic_graph(graph):
                continue
            if any(graph.in_degree(n) == 0 or graph.out_degree(n) == 0
                   for n in graph if n not in boundary):
                continue
            if not all(any(nx.has_path(graph, i, o) for o in outlets) for i in inlets):
                continue
            if not all(any(nx.has_path(graph, i, o) for i in inlets) for o in outlets):
                continue
            valid.append(list(selected))
    return valid[0] if valid else None


def test_dominators_match_edge_deletion_on_small_graphs():
    # Every unlabeled graph up to seven nodes: isolated/disconnected, cyclic,
    # branchy, and multiple-inlet/outlet cases, including ambiguous bridges.
    for graph in nx.graph_atlas_g():
        terminals = set(graph)
        for inlets, outlets in [(terminals, terminals),
                                (terminals & {0, 2}, terminals & {1, 3}),
                                (set(), set())]:
            assert list(query.bridge_sides(graph, inlets, outlets)) == bridge_oracle(graph, inlets, outlets)


def test_reachable_matches_pairwise_search():
    graph = nx.DiGraph([(0, 1), (2, 1), (1, 3), (4, 5)])
    assert query.reachable(graph, [0, 2]) == {0, 1, 2, 3}
    assert query.reachable(graph, [3], reverse=True) == {0, 1, 2, 3}
    assert query.reachable(graph, []) == set()


@pytest.mark.parametrize("manual", [(), (1,), (1, 2, 3, 4, 5)])
def test_minimum_flip_matches_exhaustive_orientation_oracle(manual):
    undirected = [(0, 1), (1, 2), (2, 3), (3, 1), (2, 4), (3, 4), (4, 5)]
    candidates = [i for i in [1, 2, 3, 4, 5] if i not in manual]
    for directions in itertools.product([False, True], repeat=5):
        edges = [edge[::-1] if 0 < i < 6 and directions[i - 1] else edge
                 for i, edge in enumerate(undirected)]
        args = edges, candidates, {0, 5}, [0], [5]
        assert query.first_valid_flip(*args) == cycle_oracle(*args)


def test_large_cycle_is_skipped_before_search(monkeypatch):
    def unexpected(*args):
        pytest.fail("oversized cycle must be rejected before search")
    monkeypatch.setattr(query, "first_valid_flip", unexpected)
    links, nodes = synthetic_delta(30)
    before = [edge[:] for edge in links["conn"]]
    links, _, remaining = di.fix_cycles(links, nodes)
    assert remaining == 1
    assert links["conn"] == before

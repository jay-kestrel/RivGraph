"""Light-weight tests for directionality helpers."""
from __future__ import annotations

import networkx as nx

from tests._helpers import require_raster_runtime


def _directionality_module():
    require_raster_runtime()
    from rivgraph import directionality as di

    return di


def test_directionlity_trackers():
    di = _directionality_module()
    links = {'id': [0, 1]}
    nodes = {'id': [0, 1]}
    di.add_directionality_trackers(links, nodes, 'delta')
    assert 'certain' in links.keys()
    assert 'certain_order' in links.keys()
    assert 'certain_alg' in links.keys()
    assert 'guess' in links.keys()
    assert 'guess_alg' in links.keys()
    assert 'maxang' not in links.keys()
    di.add_directionality_trackers(links, nodes, 'river')
    assert 'maxang' in links.keys()


def test_algmap_values():
    di = _directionality_module()
    expected = {
        'sourcesinkfix': -2,
        'manual_set': -1,
        'inletoutlet': 0,
        'continuity': 1,
        'parallels': 2,
        'artificials': 2.1,
        'main_chans': 4,
        'bridges': 5,
        'known_fdr': 6,
        'known_fdr_rs': 6.1,
        'syn_dem': 10,
        'syn_dem_med': 10.1,
        'sym_dem_leftover': 10.2,
        'sp_links': 11,
        'sp_nodes': 12,
        'longest_steepest': 13,
        'three_agree': 15,
        'syn_dem_and_sp': 16,
        'cl_dist_guess': 20,
        'cl_ang_guess': 21,
        'cl_dist_set': 22,
        'cl_ang_set': 23,
        'cl_ang_rs': 23.1,
        'cl_dist_and_ang': 24,
        'short_no_bktrck': 25,
        'wid_pctdiff': 26,
    }
    for key, value in expected.items():
        assert di.algmap(key) == value


def test_merge_list_of_lists():
    di = _directionality_module()
    inlist = [[1, 2, 3], [1, 2, 4], [2, 4, 6]]
    merged = di.merge_list_of_lists(inlist)
    assert merged == [[1, 2, 3, 4, 6]]


def test_flip_links_in_G():
    di = _directionality_module()
    G = nx.DiGraph()
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G = di.flip_links_in_G(G, 'all')
    [x, y] = G.edges.data()
    assert x == (2, 1, {})
    assert y == (3, 2, {})


def test_dir_shortest_paths_nodes_repeated_invocation():
    """Guessing the same link twice must not raise.

    The guess lookup called .index(alg) on links['guess_alg'] (the outer
    list of per-link lists) instead of the per-link list, so a second
    invocation of the algorithm on a link that already carries a guess
    raised 'ValueError: 12 is not in list'.
    """
    di = _directionality_module()

    # 4-node line network: nodes 0-1-2-3, outlet at node 3.
    links = {
        'id': [0, 1, 2],
        'conn': [[0, 1], [1, 2], [2, 3]],
        'len': [1.0, 1.0, 1.0],
        'idx': [[[0, 0], [0, 1]], [[0, 1], [0, 2]], [[0, 2], [0, 3]]],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'idx': [[0, 0], [0, 1], [0, 2], [0, 3]],
        'conn': [[0], [0, 1], [1, 2], [2]],
        'inlets': [],
        'outlets': [3],
    }
    links, nodes = di.add_directionality_trackers(links, nodes, 'delta')

    links, nodes = di.dir_shortest_paths_nodes(links, nodes)
    # Second pass over the same network: raised ValueError before the fix.
    links, nodes = di.dir_shortest_paths_nodes(links, nodes)

    assert links['guess_alg'][0] == [12]
    assert links['guess'][0] == [0]


def test_fix_sources_and_sinks_applies_fix_at_link_index_zero():
    """A repair chosen at link index 0 must be applied.

    ``if linkidx:`` tests the truthiness of a list index, so the valid
    repair was silently skipped whenever the shortest fixable link sat at
    index 0.
    """
    di = _directionality_module()

    # 3-node line: node 1 is a source (both links flow away from it) and
    # link 0 is the shortest fixable link, so link index 0 gets chosen.
    links = {
        'id': [0, 1],
        'conn': [[1, 0], [1, 2]],
        'len': [1.0, 2.0],
        'idx': [[[0, 1], [0, 0]], [[0, 1], [0, 2]]],
    }
    nodes = {
        'id': [0, 1, 2],
        'idx': [[0, 0], [0, 1], [0, 2]],
        'conn': [[0], [0, 1], [1]],
        'inlets': [],
        'outlets': [0, 2],
    }
    links, nodes = di.add_directionality_trackers(links, nodes, 'delta')
    links['certain'][:] = 1  # both links have set directions, so node 1 is checked

    assert di.check_continuity(links, nodes) == [1]

    links, nodes = di.fix_sources_and_sinks(links, nodes)

    assert di.check_continuity(links, nodes) == []
    assert links['conn'][0] == [0, 1]

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from rivgraph.layers import (
    MAX_PATHWAY_K,
    EdgeType,
    LayerType,
    add_landscape_layer,
    add_patch_layer,
    adjacency_degree,
    boundary_degree,
    boundary_weight,
    build_multiplex,
    contact_counts_per_link,
    contact_degree,
    contact_degree_distribution,
    cross_layer_components,
    distance_to_channel,
    isolated_lakes,
    landscape_nodes,
    patch_degree,
    patch_importance,
    pathway_counts,
    validate_multiplex,
)
from rivgraph.ln_utils import mark_network_ids_finalized


def _finalized_line_network():
    """Channel path 0 -> 1 -> 2 with finalized IDs (the D1 fixture network)."""
    links = {
        'id': [10, 11],
        'conn': [[0, 1], [1, 2]],
    }
    nodes = {
        'id': [0, 1, 2],
        'conn': [[10], [10, 11], [11]],
        'inlets': [0],
        'outlets': [2],
    }
    return mark_network_ids_finalized(links, nodes)


def _contacted_and_isolated_lakes():
    """Lake layer: label 100 contacts channel node 2, label 101 never connects."""
    layer_nodes = {
        100: {'centroid_xy': [5.0, 5.0], 'area_m2': 900.0, 'perimeter_m': 120.0},
        101: {'centroid_xy': [9.0, 9.0], 'area_m2': 400.0, 'perimeter_m': 80.0},
    }
    layer_edges = [
        (100, 2, {'weight': 3.0, 'buffer_px': 1, 'snapped_xy': [5.0, 1.0],
                  'snapped_linkid': 11}),
    ]
    return layer_nodes, layer_edges


def _multiplex_with_lakes():
    """Lakes on the line network: landscape ids 3 (contacted) and 4 (isolated)."""
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes, layer_edges = _contacted_and_isolated_lakes()
    add_landscape_layer(G, LayerType.LAKE, layer_nodes, layer_edges)
    return G


def _three_patch_multiplex():
    """Two full-height channel lines with three wetlands (the D3 fixture network).

    Wetlands: labels 1, 2, 3 -> landscape ids 4, 5, 6. Each neighboring pair
    shares one full-height (20 px) channel bank, so adjacency fires on
    (4, 5) and (5, 6); every patch touches the channel with 20 px of bank
    per facing side (patch 5 faces both lines).
    """
    Imask = np.zeros((20, 40), dtype=bool)
    Imask[:, 10] = True
    Imask[:, 20] = True
    links = {
        'id': [30, 31],
        'conn': [[0, 1], [2, 3]],
        'idx': [[r * 40 + 10 for r in range(20)], [r * 40 + 20 for r in range(20)]],
        'wid_adj': [1.0, 1.0],
        'len_adj': [20.0, 20.0],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'conn': [[30], [30], [31], [31]],
        'idx': [10, 19 * 40 + 10, 20, 19 * 40 + 20],
        'inlets': [],
        'outlets': [],
    }
    links, nodes = mark_network_ids_finalized(links, nodes)
    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask, closing_iterations=0,
                    min_shared_boundary_px=1)
    return G


def _wetland_chain_multiplex():
    """Line network plus a contacted wetland (id 3) and its landlocked neighbor (id 4).

    W1 (label 7) touches channel node 2; W2 (label 8) reaches the channel only
    through W1 over a patch-adjacency pair. Both wetlands carry area 100.
    """
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes = {
        7: {'centroid_xy': [1.0, 1.0], 'area_m2': 100.0, 'perimeter_m': 4.0},
        8: {'centroid_xy': [2.0, 2.0], 'area_m2': 100.0, 'perimeter_m': 4.0},
    }
    layer_edges = [
        (7, 2, {'weight': 2.0, 'buffer_px': 1, 'snapped_xy': [2.0, 1.0]}),
    ]
    add_landscape_layer(G, LayerType.WETLAND, layer_nodes, layer_edges)
    adjacency = {'edge_type': EdgeType.PATCH_ADJACENCY, 'interlayer': False, 'weight': 5.0}
    G.add_edge(3, 4, **dict(adjacency))
    G.add_edge(4, 3, **dict(adjacency))
    validate_multiplex(G)
    return G


def _bridge_chain_multiplex():
    """Three equal-area wetlands in an adjacency chain: 3 - 4 - 5 (no channel contacts).

    Patch 4 is the bridge: removing it disconnects 3 from 5 in the patch
    network. The D4 acceptance row for dI% is built on this graph.
    """
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes = {
        7: {'centroid_xy': [1.0, 1.0], 'area_m2': 100.0, 'perimeter_m': 4.0},
        8: {'centroid_xy': [2.0, 2.0], 'area_m2': 100.0, 'perimeter_m': 4.0},
        9: {'centroid_xy': [3.0, 3.0], 'area_m2': 100.0, 'perimeter_m': 4.0},
    }
    add_landscape_layer(G, LayerType.WETLAND, layer_nodes, [])
    adjacency = {'edge_type': EdgeType.PATCH_ADJACENCY, 'interlayer': False, 'weight': 10.0}
    for u, v in ((3, 4), (4, 5)):
        G.add_edge(u, v, **dict(adjacency))
        G.add_edge(v, u, **dict(adjacency))
    validate_multiplex(G)
    return G


# ---------------------------------------------------------------------------
# Lake metrics: degree-zero lakes, contact degrees, distributions, per-link
# ---------------------------------------------------------------------------

def test_isolated_lakes_counts_degree_zero_lakes():
    G = _multiplex_with_lakes()

    assert isolated_lakes(G) == (4,)
    assert len(isolated_lakes(G)) == 1  # the degree-zero (never-connected) count


def test_isolated_lakes_empty_when_every_lake_contacts():
    G = _three_patch_multiplex()  # wetlands only, no lakes

    assert isolated_lakes(G) == ()


def test_contact_degree_counts_distinct_contact_groups():
    G = _multiplex_with_lakes()

    assert contact_degree(G) == {3: 1, 4: 0}
    assert contact_degree(G, LayerType.LAKE) == {3: 1, 4: 0}
    assert contact_degree(G, 'wetland') == {}  # plain-str layer filter


def test_contact_degree_distribution_summarizes_degree_values():
    G = _multiplex_with_lakes()

    # One lake with a single contact, one never-connected: degrees 1 and 0.
    assert contact_degree_distribution(G) == {0: 1, 1: 1}
    assert contact_degree_distribution(G, 'lake') == {0: 1, 1: 1}


def test_contact_degree_distribution_on_patch_multiplex():
    G = _three_patch_multiplex()

    # Wetlands 4 and 6 each have zero contacts (only boundary edges), wetland
    # 5 likewise: no contact-type edges exist in a patch-only multiplex.
    assert contact_degree_distribution(G) == {0: 3}


def test_contact_counts_per_link_reads_snapped_link_ids():
    G = _multiplex_with_lakes()

    # The lake contact snapped to link 11.
    assert contact_counts_per_link(G) == {11: 1}


def test_contact_counts_per_link_aggregates_boundary_channel_linkids():
    G = _three_patch_multiplex()

    # Boundary edges carry channel_linkid: patches 4 and 5 face link 30,
    # patches 5 and 6 face link 31 -- two contact groups per link.
    assert contact_counts_per_link(G) == {30: 2, 31: 2}


def test_contact_counts_per_link_keeps_contacts_without_link_ids_visible():
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes = {100: {'centroid_xy': [5.0, 5.0], 'area_m2': 900.0, 'perimeter_m': 120.0}}
    layer_edges = [
        (100, 1, {'weight': 3.0, 'buffer_px': 1, 'snapped_xy': [5.0, 1.0]}),
    ]
    add_landscape_layer(G, LayerType.LAKE, layer_nodes, layer_edges)

    # No snapped_linkid/channel_linkid attr: counted under None, not dropped.
    assert contact_counts_per_link(G) == {None: 1}


# ---------------------------------------------------------------------------
# Patch metrics: adjacency, boundary, and combined per-patch degree
# ---------------------------------------------------------------------------

def test_adjacency_degree_counts_patch_neighbors():
    G = _three_patch_multiplex()

    assert adjacency_degree(G) == {4: 1, 5: 2, 6: 1}


def test_boundary_degree_counts_contact_groups_per_patch():
    G = _three_patch_multiplex()

    # Patch 5 faces both channel lines: one contiguous contact group per side.
    assert boundary_degree(G) == {4: 1, 5: 2, 6: 1}


def test_boundary_weight_sums_shared_bank_length():
    G = _three_patch_multiplex()

    # Each facing side is 20 bank pixels at pixlen=1; patch 5 faces two sides.
    assert boundary_weight(G) == {4: 20.0, 5: 40.0, 6: 20.0}


def test_patch_degree_combines_boundary_and_adjacency():
    G = _three_patch_multiplex()

    assert patch_degree(G) == {4: 2, 5: 4, 6: 2}


# ---------------------------------------------------------------------------
# Distance to channel (graph space)
# ---------------------------------------------------------------------------

def test_distance_to_channel_one_hop_from_contacted_lake():
    G = _multiplex_with_lakes()

    assert distance_to_channel(G, 3) == 1.0


def test_distance_to_channel_unreachable_for_isolated_lake():
    G = _multiplex_with_lakes()

    assert distance_to_channel(G, 4) == float('inf')


def test_distance_to_channel_counts_hops_through_adjacent_patch():
    G = _wetland_chain_multiplex()

    assert distance_to_channel(G, 3) == 1.0  # direct contact
    assert distance_to_channel(G, 4) == 2.0  # W2 -> W1 -> channel node 2


def test_distance_to_channel_sums_edge_weights_when_asked():
    G = _wetland_chain_multiplex()

    # W2 -> W1 (adjacency weight 5.0) -> node 2 (contact weight 2.0).
    assert distance_to_channel(G, 4, weight='weight') == pytest.approx(7.0)


def test_distance_to_channel_rejects_unknown_source():
    G = _multiplex_with_lakes()

    with pytest.raises(ValueError, match='not in the graph'):
        distance_to_channel(G, 99)


# ---------------------------------------------------------------------------
# Pathway counts (spec D4 acceptance: hand-computed Ak values)
# ---------------------------------------------------------------------------

def test_pathway_counts_match_hand_computed_walks():
    G = _wetland_chain_multiplex()
    # Channel 0 -> 1 -> 2; contact pair 3 <-> 2; adjacency pair 4 <-> 3.
    # Directed walks from source 0:
    #   length 1: {1: 1}
    #   length 2: {2: 1}
    #   length 3: {3: 1}
    #   length 4: {2: 1, 4: 1}   (0-1-2-3-2 and 0-1-2-3-4)
    # Cumulative landscape-target counts: k=3 -> {3: 1}; k=4 -> {3: 1, 4: 1}.
    assert pathway_counts(G, 0, k=3) == {3: 1}
    assert pathway_counts(G, 0, k=4) == {3: 1, 4: 1}


def test_pathway_counts_omit_unreachable_and_zero_count_targets():
    G = _multiplex_with_lakes()
    # From inlet 0: lake 3 (contact at node 2) is first reached by a length-3
    # walk; lake 4 is unreachable, so it never appears.
    assert pathway_counts(G, 0, k=2) == {}
    assert pathway_counts(G, 0, k=3) == {3: 1}


def test_pathway_counts_equal_adjacency_power_sum():
    G = _wetland_chain_multiplex()
    nodelist = sorted(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodelist, weight=None)

    source_row = np.zeros(len(nodelist))
    source_row[nodelist.index(0)] = 1.0
    cumulative = np.zeros(len(nodelist))
    for _ in range(4):
        source_row = source_row @ A
        cumulative += source_row
    expected = {n: int(cumulative[nodelist.index(n)]) for n in (3, 4)
                if cumulative[nodelist.index(n)] > 0}

    assert pathway_counts(G, 0, k=4) == expected


def test_pathway_counts_count_parallel_edges_as_distinct_pathways():
    links, nodes = _finalized_line_network()
    # Two parallel links from node 0 to node 1 (distinct link ids).
    links['id'] = [10, 11, 12]
    links['conn'] = [[0, 1], [0, 1], [1, 2]]
    nodes['conn'] = [[10, 11], [10, 11, 12], [12]]
    links, nodes = mark_network_ids_finalized(links, nodes)
    G = build_multiplex(links, nodes)
    add_landscape_layer(G, LayerType.LAKE,
                        {100: {'centroid_xy': [1.0, 1.0], 'area_m2': 9.0, 'perimeter_m': 12.0}},
                        [(100, 2, {'weight': 1.0, 'buffer_px': 1, 'snapped_xy': [2.0, 1.0]})])

    # Two distinct channel edges 0 -> 1, so two walks reach lake 3 in 3 steps.
    assert pathway_counts(G, 0, k=3) == {3: 2}


def test_pathway_counts_reject_out_of_range_k():
    G = _wetland_chain_multiplex()

    with pytest.raises(ValueError, match='k must be >= 1'):
        pathway_counts(G, 0, k=0)
    with pytest.raises(ValueError, match=f'k must be <= {MAX_PATHWAY_K}'):
        pathway_counts(G, 0, k=MAX_PATHWAY_K + 1)


# ---------------------------------------------------------------------------
# Patch importance dI% (spec D4 acceptance: bridge patch ranks highest)
# ---------------------------------------------------------------------------

def test_patch_importance_ranks_bridge_patch_highest():
    G = _bridge_chain_multiplex()

    importance = patch_importance(G)

    # Equal areas a=100: IIC_full = 3a^2 + 4a^2/2 + 2a^2/3 = 170000/3.
    # Removing the bridge leaves two isolated patches: IIC = 2a^2 = 20000,
    # dI% = 1100/17 ~ 64.71; removing an end patch leaves one adjacency,
    # dI% = 800/17 ~ 47.06.
    assert importance[4] == pytest.approx(1100 / 17)
    assert importance[3] == pytest.approx(800 / 17)
    assert importance[5] == pytest.approx(800 / 17)
    assert max(importance, key=importance.get) == 4


def test_patch_importance_reflects_intrapatch_habitat_availability():
    G = _multiplex_with_lakes()  # lakes carry no patch_adjacency edges

    # With no adjacency structure, IIC reduces to the intrapatch area terms
    # (p*ii = 1 in the habitat-availability convention), so dI% equals each
    # patch's share of the squared habitat: 900^2 / (900^2 + 400^2).
    importance = patch_importance(G)

    assert importance[3] == pytest.approx(8100 / 9700 * 100)
    assert importance[4] == pytest.approx(1600 / 9700 * 100)


def test_patch_importance_rejects_unsupported_index():
    G = _bridge_chain_multiplex()

    with pytest.raises(ValueError, match='iic'):
        patch_importance(G, index='pc')


# ---------------------------------------------------------------------------
# Cross-layer connectivity
# ---------------------------------------------------------------------------

def test_cross_layer_components_span_channel_and_lake_layers():
    G = _multiplex_with_lakes()

    components = cross_layer_components(G)

    assert len(components) == 1
    (component,) = components
    layers = {G.nodes[n]['layer'] for n in component}
    assert layers == {LayerType.CHANNEL, LayerType.LAKE}
    assert component == frozenset({0, 1, 2, 3})  # isolated lake 4 excluded


def test_cross_layer_components_join_patches_and_channel():
    G = _three_patch_multiplex()

    components = cross_layer_components(G)

    assert len(components) == 1
    (component,) = components
    assert {G.nodes[n]['layer'] for n in component} == {LayerType.CHANNEL, LayerType.WETLAND}
    assert set(component) == set(G.nodes)


def test_cross_layer_components_empty_for_channel_only_graph():
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)

    assert cross_layer_components(G) == ()


# ---------------------------------------------------------------------------
# Read-only contract and package surface
# ---------------------------------------------------------------------------

def test_metrics_are_read_only_over_the_graph():
    G = _three_patch_multiplex()
    add_landscape_layer(G, LayerType.LAKE,
                        {100: {'centroid_xy': [5.0, 5.0], 'area_m2': 900.0, 'perimeter_m': 120.0}},
                        [(100, 2, {'weight': 3.0, 'buffer_px': 1, 'snapped_xy': [5.0, 1.0],
                                   'snapped_linkid': 11})])
    before = nx.node_link_data(G)

    landscape_nodes(G)
    isolated_lakes(G)
    contact_degree(G)
    contact_degree_distribution(G)
    contact_counts_per_link(G)
    adjacency_degree(G)
    boundary_degree(G)
    boundary_weight(G)
    patch_degree(G)
    distance_to_channel(G, 4)
    pathway_counts(G, 0, k=4)
    patch_importance(G)
    cross_layer_components(G)

    assert nx.node_link_data(G) == before


def test_landscape_nodes_filters_by_layer():
    G = _multiplex_with_lakes()

    assert landscape_nodes(G) == [3, 4]
    assert landscape_nodes(G, 'lake') == [3, 4]
    assert landscape_nodes(G, LayerType.WETLAND) == []


def test_metrics_exports_are_additive_over_d1_d2_d3_d5():
    import rivgraph.layers as layers_pkg

    original_19 = {
        'EdgeType', 'LayerType', 'NodeType', 'PatchExtraction',
        'add_lake_layer', 'add_landscape_layer', 'add_patch_layer',
        'build_multiplex', 'channel_view', 'close_landmass', 'extract_lakes',
        'extract_patches', 'label_patches', 'multiplex_from_node_link_data',
        'multiplex_to_node_link_data', 'read_multiplex_json',
        'shared_boundary_lengths', 'validate_multiplex', 'write_multiplex_json',
    }

    assert original_19 <= set(layers_pkg.__all__)  # every pre-existing export survives
    assert len(layers_pkg.__all__) == len(set(layers_pkg.__all__))  # no duplicates
    for name in layers_pkg.__all__:  # every listed export resolves
        assert hasattr(layers_pkg, name), f'missing export: {name}'

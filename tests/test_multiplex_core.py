from __future__ import annotations

import json

import networkx as nx
import pytest

from rivgraph.layers import (
    EdgeType,
    LayerType,
    NodeType,
    add_landscape_layer,
    build_multiplex,
    channel_view,
    validate_multiplex,
)
from rivgraph.ln_utils import mark_network_ids_finalized, mark_network_ids_provisional


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


def _lake_layer():
    """Two lakes: label 100 contacts channel node 1, label 101 never connects."""
    layer_nodes = {
        100: {
            'centroid_xy': [5.0, 5.0],
            'area_m2': 900.0,
            'perimeter_m': 120.0,
            'polygon': [[0, 0], [0, 30], [30, 30], [30, 0], [0, 0]],
        },
        101: {
            'centroid_xy': [9.0, 9.0],
            'area_m2': 400.0,
            'perimeter_m': 80.0,
            'mask_date': '2026-10-30',
        },
    }
    layer_edges = [
        (100, 1, {'weight': 3.0, 'buffer_px': 1, 'snapped_xy': [5.0, 1.0]}),
    ]
    return layer_nodes, layer_edges


def _multiplex_with_lakes():
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes, layer_edges = _lake_layer()
    add_landscape_layer(G, LayerType.LAKE, layer_nodes, layer_edges)
    return G


def test_build_multiplex_returns_validated_multidigraph():
    links, nodes = _finalized_line_network()

    G = build_multiplex(links, nodes)

    assert isinstance(G, nx.MultiDiGraph)
    assert G.graph['rivgraph_multiplex'] is True
    assert set(G.nodes) == {0, 1, 2}
    assert G.number_of_edges() == 2
    # Channel nodes carry layer and node_type (junction/inlet/outlet).
    assert G.nodes[0]['layer'] == LayerType.CHANNEL == 'channel'
    assert G.nodes[0]['node_type'] == NodeType.INLET
    assert G.nodes[1]['node_type'] == NodeType.JUNCTION
    assert G.nodes[2]['node_type'] == NodeType.OUTLET
    # Channel edges follow the graphiphy contract (weight, linkid) plus the
    # multiplex schema (edge_type, interlayer).
    assert dict(G[0][1][0]) == {'edge_type': EdgeType.CHANNEL, 'interlayer': False, 'weight': 1.0, 'linkid': 10}
    assert G[1][2][0]['linkid'] == 11
    validate_multiplex(G)  # does not raise


def test_build_multiplex_rejects_unfinalized_ids():
    links, nodes = _finalized_line_network()
    mark_network_ids_provisional(links, nodes)

    with pytest.raises(RuntimeError, match='finalized'):
        build_multiplex(links, nodes)


def test_channel_layer_matches_graphiphy_structure():
    from rivgraph.deltas.delta_metrics import graphiphy

    links, nodes = _finalized_line_network()

    G = build_multiplex(links, nodes)
    Gg = graphiphy(links, nodes)

    assert set(G.nodes) == set(Gg.nodes)
    gm_edges = {(u, v, d['weight'], d['linkid']) for u, v, d in G.edges(data=True)}
    gg_edges = {(u, v, d['weight'], d['linkid']) for u, v, d in Gg.edges(data=True)}
    assert gm_edges == gg_edges


def test_channel_view_is_channel_only():
    G = _multiplex_with_lakes()

    view = channel_view(G)

    assert set(view.nodes) == {0, 1, 2}
    assert all(view.nodes[n]['layer'] == LayerType.CHANNEL for n in view)
    assert all(view.nodes[n]['node_type'] != NodeType.LANDSCAPE for n in view)


def test_channel_view_carries_all_channel_edges():
    G = _multiplex_with_lakes()
    view = channel_view(G)

    channel_edges = [
        (u, v, k, d)
        for u, v, k, d in G.edges(keys=True, data=True)
        if G.nodes[u]['layer'] == LayerType.CHANNEL and G.nodes[v]['layer'] == LayerType.CHANNEL
    ]
    assert len(channel_edges) > 0
    assert view.number_of_edges() == len(channel_edges)
    for u, v, k, d in channel_edges:
        assert view.has_edge(u, v)
        assert view[u][v][k] == d


def test_contact_edges_exist_in_multiplex_and_are_excluded_from_view():
    G = _multiplex_with_lakes()

    # Lake label 100 got the first landscape id: max(channel ids) + 1 == 3.
    lake_id = 3
    assert G.nodes[lake_id]['label_id'] == 100
    # Contact stored as a symmetric bidirectional pair.
    assert G.has_edge(lake_id, 1) and G.has_edge(1, lake_id)
    forward = G[lake_id][1][0]
    reverse = G[1][lake_id][0]
    assert forward == reverse
    assert forward['edge_type'] == EdgeType.CONTACT == 'contact'
    assert forward['interlayer'] is True

    # The channel-only projection excludes every contact edge.
    view = channel_view(G)
    assert lake_id not in view
    assert all(t != EdgeType.CONTACT for _, _, t in view.edges(data='edge_type'))


def test_contact_attribute_passthrough():
    G = _multiplex_with_lakes()
    lake_id, never_connected = 3, 4

    contact = G[lake_id][1][0]
    assert contact['weight'] == 3.0
    assert contact['buffer_px'] == 1
    assert contact['snapped_xy'] == [5.0, 1.0]
    assert G[1][lake_id][0] == contact  # identical attrs on the reverse pair

    attached = G.nodes[lake_id]
    assert attached['layer'] == 'lake'  # plain-string comparison (post-roundtrip form)
    assert attached['node_type'] == NodeType.LANDSCAPE
    assert attached['centroid_xy'] == [5.0, 5.0]
    assert attached['area_m2'] == 900.0
    assert attached['perimeter_m'] == 120.0
    assert attached['polygon'] == [[0, 0], [0, 30], [30, 30], [30, 0], [0, 0]]

    # Degree-0 lake kept as a node; optional attrs default to None.
    isolated = G.nodes[never_connected]
    assert isolated['label_id'] == 101
    assert isolated['mask_date'] == '2026-10-30'
    assert G.degree(never_connected) == 0
    assert G.nodes[never_connected]['polygon'] is None


def test_landscape_ids_assigned_deterministically():
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes, layer_edges = _lake_layer()

    add_landscape_layer(G, 'lake', layer_nodes, layer_edges)  # plain str accepted too
    add_landscape_layer(G, LayerType.WETLAND, {7: {'centroid_xy': [1.0, 2.0], 'area_m2': 10.0, 'perimeter_m': 12.0}}, [])

    # Ascending label-raster order, starting past the maximum existing id so
    # repeated calls never collide across layers.
    label_ids = {d['label_id']: n for n, d in G.nodes(data=True) if 'label_id' in d}
    assert label_ids == {100: 3, 101: 4, 7: 5}
    assert all(G.nodes[n]['layer'] == LayerType.LAKE for n in (3, 4))
    assert G.nodes[5]['layer'] == LayerType.WETLAND


def test_validate_rejects_missing_node_attrs():
    G = _multiplex_with_lakes()
    del G.nodes[0]['layer']
    with pytest.raises(ValueError, match='missing required attrs'):
        validate_multiplex(G)

    G2 = _multiplex_with_lakes()
    del G2.nodes[3]['node_type']
    with pytest.raises(ValueError, match='missing required attrs'):
        validate_multiplex(G2)


def test_validate_rejects_asymmetric_contact_pair():
    G = _multiplex_with_lakes()
    G.remove_edge(1, 3)  # drop one direction of the bidirectional pair

    with pytest.raises(ValueError, match='symmetric reverse pair'):
        validate_multiplex(G)


def test_validate_rejects_landscape_id_collision_with_channel_ids():
    G = build_multiplex(*_finalized_line_network())
    # Simulate a bad attach: a landscape id colliding with a channel id
    # silently merges the node in NetworkX, so validation must catch it.
    G.add_node(1, layer='lake', node_type='landscape', label_id=99,
               centroid_xy=[0.0, 0.0], area_m2=1.0, perimeter_m=1.0)

    with pytest.raises(ValueError, match='disjoint'):
        validate_multiplex(G)


def test_validate_rejects_invalid_edge_type():
    G = _multiplex_with_lakes()
    G[0][1][0]['edge_type'] = 'bogus'

    with pytest.raises(ValueError, match='invalid edge_type'):
        validate_multiplex(G)


def test_add_landscape_layer_rejects_invalid_layer_name():
    G = build_multiplex(*_finalized_line_network())
    layer_nodes, layer_edges = _lake_layer()

    with pytest.raises(ValueError, match='landscape type'):
        add_landscape_layer(G, LayerType.CHANNEL, layer_nodes, layer_edges)
    with pytest.raises(ValueError):
        add_landscape_layer(G, 'bogus', layer_nodes, layer_edges)


def test_add_landscape_layer_rejects_bad_contact_targets():
    G = build_multiplex(*_finalized_line_network())
    layer_nodes, _ = _lake_layer()
    contact_attrs = {'weight': 1.0, 'buffer_px': 1, 'snapped_xy': [0.0, 0.0]}

    with pytest.raises(ValueError, match='not in the graph'):
        add_landscape_layer(G, LayerType.LAKE, layer_nodes, [(100, 99, dict(contact_attrs))])
    with pytest.raises(ValueError, match='references label'):
        add_landscape_layer(G, LayerType.LAKE, layer_nodes, [(999, 1, dict(contact_attrs))])


def test_add_landscape_layer_rejects_contact_to_landscape_node():
    links, nodes = _finalized_line_network()
    G = build_multiplex(links, nodes)
    layer_nodes, layer_edges = _lake_layer()
    add_landscape_layer(G, LayerType.LAKE, layer_nodes, layer_edges)

    # Node 3 is the attached lake: contacts must attach to channel nodes only.
    with pytest.raises(ValueError, match='channel node'):
        add_landscape_layer(G, LayerType.WETLAND,
                            {7: {'centroid_xy': [0.0, 0.0], 'area_m2': 1.0, 'perimeter_m': 1.0}},
                            [(7, 3, {'weight': 1.0, 'buffer_px': 1, 'snapped_xy': [0.0, 0.0]})])


def test_add_landscape_layer_rejects_missing_contact_attrs():
    G = build_multiplex(*_finalized_line_network())
    layer_nodes, _ = _lake_layer()

    with pytest.raises(ValueError, match='missing required attrs'):
        add_landscape_layer(G, LayerType.LAKE, layer_nodes,
                            [(100, 1, {'weight': 1.0, 'buffer_px': 1})])


def test_add_landscape_layer_rejects_wrong_edge_type():
    G = build_multiplex(*_finalized_line_network())
    layer_nodes, _ = _lake_layer()

    with pytest.raises(ValueError, match="edge_type 'contact'"):
        add_landscape_layer(G, LayerType.LAKE, layer_nodes,
                            [(100, 1, {'weight': 1.0, 'buffer_px': 1, 'snapped_xy': [0.0, 0.0],
                                       'edge_type': 'channel'})])


def test_weight_attribute_from_links():
    links, nodes = _finalized_line_network()
    links['len'] = [5.0, 7.0]
    links, nodes = mark_network_ids_finalized(links, nodes)

    G = build_multiplex(links, nodes, weight='len')

    assert G[0][1][0]['weight'] == 5.0
    assert G[1][2][0]['weight'] == 7.0


def test_invalid_weight_key_and_nonpositive_weights_raise():
    links, nodes = _finalized_line_network()
    with pytest.raises(RuntimeError, match='weight key not in links'):
        build_multiplex(links, nodes, weight='nope')

    links['len'] = [0.0, 7.0]
    links, nodes = mark_network_ids_finalized(links, nodes)
    with pytest.raises(ValueError, match='weights must be positive'):
        build_multiplex(links, nodes, weight='len')


def test_multiplex_json_roundtrip_preserves_edges_and_attrs():
    G = _multiplex_with_lakes()

    data = nx.node_link_data(G)
    G2 = nx.node_link_graph(json.loads(json.dumps(data)))

    assert isinstance(G2, nx.MultiDiGraph)
    assert G2.graph['rivgraph_multiplex'] is True
    assert set(G2.nodes) == set(G.nodes)
    assert G2.number_of_edges() == G.number_of_edges()
    for u, v, k, d in G.edges(keys=True, data=True):
        assert G2.has_edge(u, v)
        assert G2[u][v][k] == d
    for n, d in G.nodes(data=True):
        assert G2.nodes[n] == d
    # The projection still works on the reconstructed graph.
    view = channel_view(G2)
    assert set(view.nodes) == {0, 1, 2}

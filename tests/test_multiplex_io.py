"""Round-trip and CLI tests for multiplex node-link serialization (D5).

The fixture multiplex carries parallel channel edges (two links between the
same node pair), parallel cross-layer contact edges (two distinct contact
groups onto one channel node), and full node/edge attributes. Every
export->import test asserts the node set, the ``(u, v, key)`` edge multiset,
and all attributes reproduce exactly, per the D5 acceptance criteria.
"""
from __future__ import annotations

import json

import networkx as nx
import pytest

import rivgraph.cli as cli
from rivgraph.layers import (
    EdgeType,
    LayerType,
    NodeType,
    add_landscape_layer,
    build_multiplex,
    multiplex_from_node_link_data,
    multiplex_to_node_link_data,
    read_multiplex_json,
    validate_multiplex,
    write_multiplex_json,
)
from rivgraph.ln_utils import mark_network_ids_finalized


def _finalized_channel_network():
    """Channel 0 -> 1 -> 2 plus a parallel second link 0 -> 1 (finalized)."""
    links = {
        'id': [10, 11, 12],
        'conn': [[0, 1], [0, 1], [1, 2]],
    }
    nodes = {
        'id': [0, 1, 2],
        'conn': [[10, 11], [10, 11, 12], [12]],
        'inlets': [0],
        'outlets': [2],
    }
    return mark_network_ids_finalized(links, nodes)


def _multiplex_fixture():
    """Small multiplex with parallel channel edges, parallel cross-layer
    contact edges, and full node/edge attributes.

    Coordinate attributes use tuples on purpose (exercising import
    restoration); ``polygon`` stays JSON-native (see layers/io.py docstring).
    """
    links, nodes = _finalized_channel_network()
    G = build_multiplex(links, nodes)

    layer_nodes = {
        100: {
            'centroid_xy': (5.0, 5.0),
            'area_m2': 900.0,
            'perimeter_m': 120.0,
            'mask_date': '2026-10-30',
            'polygon': [[0, 0], [0, 30], [30, 30], [30, 0]],
        },
        101: {
            'centroid_xy': (9.0, 9.0),
            'area_m2': 400.0,
            'perimeter_m': 80.0,
        },
    }
    layer_edges = [
        # Two distinct contiguous contacts onto the same channel node:
        # stored as two parallel bidirectional pairs (keys 0 and 1).
        (100, 1, {'weight': 3.0, 'buffer_px': 1, 'snapped_xy': (5.0, 1.0),
                  'mask_date': '2026-10-30'}),
        (100, 1, {'weight': 4.0, 'buffer_px': 2, 'snapped_xy': (5.0, 2.0)}),
        (101, 2, {'weight': 1.0, 'buffer_px': 1, 'snapped_xy': (9.0, 2.0)}),
    ]
    add_landscape_layer(G, LayerType.LAKE, layer_nodes, layer_edges)
    return G


def _assert_same_multiplex(G, H):
    """Assert H reproduces G exactly: node set, edge multiset, attributes."""
    assert isinstance(H, nx.MultiDiGraph)
    assert H.is_directed() == G.is_directed()
    assert set(H.nodes) == set(G.nodes)
    assert H.graph == G.graph
    for n in G:
        assert H.nodes[n] == G.nodes[n], f'node {n} attrs differ'
    edges_G = sorted((u, v, k) for u, v, k in G.edges(keys=True))
    edges_H = sorted((u, v, k) for u, v, k in H.edges(keys=True))
    assert edges_H == edges_G
    for u, v, k in edges_G:
        assert H[u][v][k] == G[u][v][k], f'edge ({u}, {v}, {k}) attrs differ'


def test_roundtrip_file_preserves_nodes_edges_and_attrs(tmp_path):
    G = _multiplex_fixture()
    path = tmp_path / 'multiplex.json'

    write_multiplex_json(G, path)
    H = read_multiplex_json(path)

    _assert_same_multiplex(G, H)
    validate_multiplex(H)  # does not raise

    # Schema attribute types are restored, not left in their JSON forms.
    lake = next(n for n in H if H.nodes[n]['layer'] == LayerType.LAKE)
    assert isinstance(H.nodes[lake]['layer'], LayerType)
    assert isinstance(H.nodes[lake]['node_type'], NodeType)
    assert isinstance(H.nodes[lake]['centroid_xy'], tuple)
    channel_edge = next(iter(dict(H[0][1]).values()))
    assert isinstance(channel_edge['edge_type'], EdgeType)


def test_roundtrip_in_memory():
    G = _multiplex_fixture()

    data = multiplex_to_node_link_data(G)
    H = multiplex_from_node_link_data(data)

    _assert_same_multiplex(G, H)


def test_parallel_channel_and_contact_edges_survive():
    G = _multiplex_fixture()

    H = multiplex_from_node_link_data(multiplex_to_node_link_data(G))

    # Parallel channel edges (links 10 and 11 between nodes 0 and 1) keep
    # their distinct multigraph keys and per-key attributes.
    assert sorted(H[0][1]) == [0, 1]
    assert H[0][1][0]['linkid'] == 10
    assert H[0][1][1]['linkid'] == 11
    # Parallel cross-layer contacts (lake 3 onto channel node 1) likewise.
    lake = next(n for n in H if H.nodes[n].get('label_id') == 100)
    assert sorted(H[lake][1]) == [0, 1]
    weights = sorted(H[lake][1][k]['weight'] for k in H[lake][1])
    assert weights == [3.0, 4.0]
    # The bidirectional-pair convention survives: both directions exist.
    assert H.has_edge(1, lake) and H.has_edge(lake, 1)


def test_export_rejects_invalid_multiplex():
    G = _multiplex_fixture()
    first_lake = next(n for n in G if G.nodes[n]['layer'] == LayerType.LAKE)
    G.nodes[first_lake].pop('area_m2')  # break the landscape-node contract

    with pytest.raises(ValueError, match='area_m2'):
        multiplex_to_node_link_data(G)


def test_import_rejects_asymmetric_bidirectional_pair():
    data = multiplex_to_node_link_data(_multiplex_fixture())
    # Corrupt: drop the reverse contact edges so the pair is asymmetric.
    data['edges'] = [
        e for e in data['edges'] if not (e['source'] == 1 and e['target'] == 3)
    ]

    with pytest.raises(ValueError, match='symmetric'):
        multiplex_from_node_link_data(data)


def test_import_rejects_non_multiplex_documents():
    plain = nx.MultiDiGraph()
    plain.add_node(1, layer='channel', node_type='junction')

    with pytest.raises(ValueError, match='multiplex marker'):
        multiplex_from_node_link_data(nx.node_link_data(plain))
    with pytest.raises(ValueError, match='multigraph'):
        data = nx.node_link_data(nx.DiGraph())
        multiplex_from_node_link_data(data)


def test_import_rejects_malformed_json(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{not json')

    with pytest.raises(ValueError, match='not valid JSON'):
        read_multiplex_json(path)


def test_cli_export_multiplex_writes_file(tmp_path):
    G = _multiplex_fixture()
    in_path = tmp_path / 'in.json'
    out_path = tmp_path / 'out.json'
    write_multiplex_json(G, in_path)

    assert cli.main(['export-multiplex', str(in_path), '--out', str(out_path)]) == 0

    H = read_multiplex_json(out_path)
    _assert_same_multiplex(G, H)


def test_cli_export_multiplex_prints_json_to_stdout(tmp_path, capsys):
    G = _multiplex_fixture()
    in_path = tmp_path / 'in.json'
    write_multiplex_json(G, in_path)

    assert cli.main(['export-multiplex', str(in_path)]) == 0

    data = json.loads(capsys.readouterr().out)
    H = multiplex_from_node_link_data(data)
    _assert_same_multiplex(G, H)


def test_cli_export_multiplex_rejects_invalid_input(tmp_path, capsys):
    path = tmp_path / 'bad.json'
    path.write_text('{not json')

    assert cli.main(['export-multiplex', str(path)]) == 1
    assert 'Error' in capsys.readouterr().err


def test_cli_export_multiplex_rejects_non_multiplex(tmp_path, capsys):
    path = tmp_path / 'plain.json'
    plain = nx.DiGraph()
    path.write_text(json.dumps(nx.node_link_data(plain)))

    assert cli.main(['export-multiplex', str(path)]) == 1
    assert 'multiplex' in capsys.readouterr().err


def test_cli_export_multiplex_missing_file(tmp_path, capsys):
    assert cli.main(['export-multiplex', str(tmp_path / 'nope.json')]) == 1
    assert 'no such file' in capsys.readouterr().err

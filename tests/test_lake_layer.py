from __future__ import annotations

import numpy as np
import pytest

from rivgraph.layers import (
    EdgeType,
    LayerType,
    NodeType,
    add_lake_layer,
    build_multiplex,
    validate_multiplex,
)
from rivgraph.layers.lakes import extract_lakes
from rivgraph.ln_utils import mark_network_ids_finalized, mark_network_ids_provisional


def _delta_masks():
    """Two-link L-shaped channel with four lakes, in pixel space (x=col, y=row).

    - Lake A (rows 6-8, cols 4-6) touches link 10 -> one 3-pixel contact group.
    - Lake B (rows 6-8, cols 8-11) sits beside BOTH links; its boundary
      contacts form one contiguous group -> still one contact, snapped to the
      nearer link 10.
    - Lake D (rows 8-10, cols 14-16) lies 2 px off link 11 -> contacts only
      appear at buffer_px >= 2.
    - Lake C (rows 10-12, cols 2-4) is far from every link -> never connects.
    """
    channel = np.zeros((15, 20), dtype=bool)
    channel[5, 2:13] = True  # link 10: row 5, cols 2..12
    channel[5:13, 12] = True  # link 11: col 12, rows 5..12

    lakes = np.zeros((15, 20), dtype=bool)
    lakes[6:9, 4:7] = True  # A
    lakes[6:9, 8:12] = True  # B
    lakes[8:11, 14:17] = True  # D
    lakes[10:13, 2:5] = True  # C

    return channel | lakes, channel


def _finalized_network():
    """The D1 fixture channel (ids 0/1/2, links 10/11) with pixel-space geometry."""
    links = {
        'id': [10, 11],
        'conn': [[0, 1], [1, 2]],
        'geometry': [[(2, 5), (12, 5)], [(12, 5), (12, 12)]],
    }
    nodes = {
        'id': [0, 1, 2],
        'conn': [[10], [10, 11], [11]],
        'inlets': [0],
        'outlets': [2],
        'geometry': [(2, 5), (12, 5), (12, 12)],
    }
    return mark_network_ids_finalized(links, nodes)


def _lake_multiplex(buffer_px=1, mask_date=None):
    """Multiplex graph for the fixture delta with the lake layer attached."""
    water, channel = _delta_masks()
    links, nodes = _finalized_network()
    G = build_multiplex(links, nodes)
    add_lake_layer(G, water, channel, links, nodes,
                   buffer_px=buffer_px, mask_date=mask_date)
    return G


def _lake_node_by_label(G, label_id):
    matches = [n for n, d in G.nodes(data=True) if d.get('label_id') == label_id]
    assert len(matches) == 1
    return matches[0]


def _contact_pairs(G):
    """Contact edges counted once per pair, from the landscape-node side."""
    return [(u, v, d) for u, v, d in G.edges(data=True)
            if d['edge_type'] == EdgeType.CONTACT
            and G.nodes[u]['layer'] == LayerType.LAKE]


def test_lake_components_become_landscape_nodes():
    G = _lake_multiplex()
    validate_multiplex(G)  # does not raise

    lake_nodes = {d['label_id']: n for n, d in G.nodes(data=True)
                  if d['layer'] == LayerType.LAKE}
    # Components-minus-channel yields exactly the four fixture lakes.
    assert set(lake_nodes) == {1, 2, 3, 4}
    # Deterministic landscape ids: max(channel ids) + 1 in label-raster order.
    assert [lake_nodes[k] for k in sorted(lake_nodes)] == [3, 4, 5, 6]
    assert set(G.nodes) == {0, 1, 2, 3, 4, 5, 6}

    # Lake A: 3x3 pixels, hand-computed attributes (pixel_size_m = 1).
    a = G.nodes[lake_nodes[1]]
    assert a['node_type'] == NodeType.LANDSCAPE
    assert a['centroid_xy'] == [5.0, 7.0]
    assert a['area_m2'] == 9.0
    assert a['perimeter_m'] == 8.0  # skimage perimeter convention (island machinery)


def test_lake_touching_one_link_gets_one_contact_pair():
    G = _lake_multiplex()
    lake_a = _lake_node_by_label(G, 1)

    # One contact -> one symmetric bidirectional pair, nothing else.
    assert G.degree(lake_a) == 2
    assert G.has_edge(lake_a, 0) and G.has_edge(0, lake_a)
    assert not any(G.has_edge(lake_a, n) for n in (1, 2))

    forward = G[lake_a][0][0]
    reverse = G[0][lake_a][0]
    assert forward == reverse
    assert forward['edge_type'] == EdgeType.CONTACT == 'contact'
    assert forward['interlayer'] is True
    assert forward['weight'] == 3.0  # one contiguous 3-pixel group, not 3 edges
    assert forward['buffer_px'] == 1
    assert forward['snapped_xy'] == [5.0, 5.0]  # projected onto link 10 (row 5)
    assert forward['snapped_linkid'] == 10


def test_lake_spanning_two_links_groups_into_one_contact_set():
    G = _lake_multiplex()
    lake_b = _lake_node_by_label(G, 2)

    # Lake B's boundary runs beside links 10 AND 11, but the contact pixels
    # are one contiguous group -> exactly ONE pair, snapped to the nearer link.
    assert len(_contact_pairs(G)) == 2  # lake A + lake B
    assert G.degree(lake_b) == 2
    assert G.has_edge(lake_b, 1) and G.has_edge(1, lake_b)
    assert not any(G.has_edge(lake_b, n) for n in (0, 2))

    forward = G[lake_b][1][0]
    assert forward['snapped_linkid'] == 10
    assert forward['snapped_xy'] == [10.0, 5.0]
    assert forward['weight'] == 6.0


def test_isolated_lake_kept_as_degree_zero_node():
    G = _lake_multiplex()
    lake_c = _lake_node_by_label(G, 4)

    # Degree-0 (never-connected) lakes are load-bearing, not noise.
    assert G.degree(lake_c) == 0
    assert G.nodes[lake_c]['area_m2'] == 9.0
    assert G.nodes[lake_c]['centroid_xy'] == [3.0, 11.0]
    assert not any(lake_c in (u, v) for u, v, d in G.edges(data=True)
                   if d['edge_type'] == EdgeType.CONTACT)


def test_buffer_px_controls_contact_detection():
    # At the default buffer, lake D (2 px off link 11) has no contact.
    G1 = _lake_multiplex(buffer_px=1)
    lake_d_1 = _lake_node_by_label(G1, 3)
    assert G1.degree(lake_d_1) == 0
    assert len(_contact_pairs(G1)) == 2

    # Widening the buffer to the lake's separation distance connects it.
    G2 = _lake_multiplex(buffer_px=2)
    lake_d_2 = _lake_node_by_label(G2, 3)
    assert G2.degree(lake_d_2) == 2
    assert G2.has_edge(lake_d_2, 2) and G2.has_edge(2, lake_d_2)
    forward = G2[lake_d_2][2][0]
    assert forward['buffer_px'] == 2
    assert forward['snapped_linkid'] == 11
    assert forward['snapped_xy'] == [12.0, 9.0]
    assert forward['weight'] == 3.0
    assert len(_contact_pairs(G2)) == 3

    # Lake C remains isolated even at the wider buffer.
    assert G2.degree(_lake_node_by_label(G2, 4)) == 0


def test_mask_date_stamps_nodes_and_contacts():
    G = _lake_multiplex(mask_date='2026-10-30')

    lake_a = _lake_node_by_label(G, 1)
    assert G.nodes[lake_a]['mask_date'] == '2026-10-30'
    assert G[lake_a][0][0]['mask_date'] == '2026-10-30'
    assert G[0][lake_a][0]['mask_date'] == '2026-10-30'


def test_extract_lakes_requires_finalized_ids():
    water, channel = _delta_masks()
    links, nodes = _finalized_network()
    mark_network_ids_provisional(links, nodes)

    with pytest.raises(RuntimeError, match='finalized'):
        extract_lakes(water, channel, links, nodes)


def test_extract_lakes_validates_inputs():
    water, channel = _delta_masks()
    links, nodes = _finalized_network()

    with pytest.raises(ValueError, match='shape'):
        extract_lakes(water, channel[:, :10], links, nodes)
    with pytest.raises(ValueError, match='buffer_px'):
        extract_lakes(water, channel, links, nodes, buffer_px=-1)

    no_geometry = {k: v for k, v in links.items() if k != 'geometry'}
    with pytest.raises(RuntimeError, match='geometry'):
        extract_lakes(water, channel, no_geometry, nodes)

from __future__ import annotations

import numpy as np
import pytest

from rivgraph.layers import (
    EdgeType,
    LayerType,
    NodeType,
    PatchExtraction,
    add_patch_layer,
    build_multiplex,
    channel_view,
    close_landmass,
    extract_patches,
    label_patches,
    shared_boundary_lengths,
    validate_multiplex,
)
from rivgraph.ln_utils import mark_network_ids_finalized, mark_network_ids_provisional


def _finalized(links, nodes):
    return mark_network_ids_finalized(links, nodes)


def _three_patch_network():
    """Two full-height 1-px channel lines at cols 10 and 20 of a 20x40 rect.

    The landmass (complement of the channels) splits into three border-touching
    wetland patches: label 1 (cols 0-9), label 2 (cols 11-19), label 3
    (cols 21-39). Each pair of neighboring patches shares one full-height
    channel line, so their shared channel bank is 20 pixels.
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
    return _finalized(links, nodes), Imask


def _ring_network():
    """A square 3-px-wide channel ring with four corner nodes on a 40x40 raster.

    The ring interior is a bounded landmass patch; the exterior touches the
    image border (mainland). Link pixel indices align with their conn nodes:
    link 10 = top (nw-ne), 11 = right (ne-se), 12 = bottom (se-sw), 13 = left
    (sw-nw).
    """
    Imask = np.zeros((40, 40), dtype=bool)
    Imask[10:30, 10:30] = True
    Imask[13:27, 13:27] = False
    links = {
        'id': [10, 11, 12, 13],
        'conn': [[0, 1], [1, 3], [3, 2], [2, 0]],
        'idx': [
            [r * 40 + c for r in range(10, 13) for c in range(10, 30)],
            [r * 40 + c for r in range(10, 30) for c in range(27, 30)],
            [r * 40 + c for r in range(27, 30) for c in range(10, 30)],
            [r * 40 + c for r in range(10, 30) for c in range(10, 13)],
        ],
        'wid_adj': [3.0, 3.0, 3.0, 3.0],
        'len_adj': [20.0, 20.0, 20.0, 20.0],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'conn': [[10, 13], [10, 11], [12, 13], [11, 12]],
        'idx': [10 * 40 + 10, 10 * 40 + 29, 29 * 40 + 10, 29 * 40 + 29],
        'inlets': [0],
        'outlets': [3],
    }
    return _finalized(links, nodes), Imask


def _bar_network():
    """A 4-px-thick channel bar with a single 1-px land island inside it.

    The island (pixel (9, 20)) is far from both bar-end nodes, so the island
    machinery's loop detection returns its NaN fallback and the wiring must
    fall back to the mask-intersection rule.
    """
    Imask = np.zeros((12, 60), dtype=bool)
    Imask[8:12, :] = True
    Imask[9, 20] = False
    links = {
        'id': [7],
        'conn': [[0, 1]],
        'idx': [[r * 60 + c for r in range(8, 12) for c in range(60) if (r, c) != (9, 20)]],
        'wid_adj': [4.0],
        'len_adj': [239.0],
    }
    nodes = {
        'id': [0, 1],
        'conn': [[7], [7]],
        'idx': [9 * 60, 9 * 60 + 59],
        'inlets': [0],
        'outlets': [1],
    }
    return _finalized(links, nodes), Imask


def _protrusion_network():
    """A 2-px water slit splitting the landmass, plus a channel protrusion.

    The slit (cols 15-16, a waterway outside the extracted channel network)
    splits the landmass in two; closing bridges it. The channel is a 1-px
    protrusion from the left border at row 10 (cols 0-5), which never splits
    the landmass, so it survives subtraction without re-splitting the closed
    patch.
    """
    landmass = np.ones((20, 40), dtype=bool)
    landmass[:, 15:17] = False
    Imask = np.zeros((20, 40), dtype=bool)
    Imask[10, 0:6] = True
    links = {
        'id': [40],
        'conn': [[0, 1]],
        'idx': [[10 * 40 + c for c in range(6)]],
        'wid_adj': [1.0],
        'len_adj': [6.0],
    }
    nodes = {
        'id': [0, 1],
        'conn': [[40], [40]],
        'idx': [10 * 40, 10 * 40 + 5],
        'inlets': [],
        'outlets': [],
    }
    return _finalized(links, nodes), Imask, landmass


def _lakeside_network():
    """A landmass rect sitting above the left half of a 1-px channel line.

    The line (row 20, cols 5-35) is one link; the rect (rows 10-19, cols 0-19)
    touches the image border and the line's left stretch. Its single contact
    group centroid is (x, y) = (12.0, 19.0), unambiguously nearest the left
    endpoint node.
    """
    landmass = np.zeros((30, 40), dtype=bool)
    landmass[10:20, 0:20] = True
    Imask = np.zeros((30, 40), dtype=bool)
    Imask[20, 5:36] = True
    links = {
        'id': [50],
        'conn': [[0, 1]],
        'idx': [[20 * 40 + c for c in range(5, 36)]],
        'wid_adj': [1.0],
        'len_adj': [31.0],
    }
    nodes = {
        'id': [0, 1],
        'conn': [[50], [50]],
        'idx': [20 * 40 + 5, 20 * 40 + 35],
        'inlets': [],
        'outlets': [],
    }
    return _finalized(links, nodes), Imask, landmass


def _lake_island_network():
    """A lake hole in the landmass with a smaller land island inside the lake.

    The lake-enclosed island (label 3) touches no channel pixel, so it must
    get no boundary edges even though the surrounding patches do.
    """
    landmass = np.ones((20, 40), dtype=bool)
    landmass[5:15, 15:30] = False
    landmass[8:12, 19:26] = True
    Imask = np.zeros((20, 40), dtype=bool)
    Imask[:, 5] = True
    links = {
        'id': [60],
        'conn': [[0, 1]],
        'idx': [[r * 40 + 5 for r in range(20)]],
        'wid_adj': [1.0],
        'len_adj': [20.0],
    }
    nodes = {
        'id': [0, 1],
        'conn': [[60], [60]],
        'idx': [5, 19 * 40 + 5],
        'inlets': [],
        'outlets': [],
    }
    return _finalized(links, nodes), Imask, landmass


def _node_for_label(G, label_id):
    matches = [n for n, d in G.nodes(data=True) if d.get('label_id') == label_id]
    assert len(matches) == 1, f'label {label_id} matched {len(matches)} nodes'
    return matches[0]


def _patch_nodes_by_layer(G, layer):
    return [n for n, d in G.nodes(data=True) if d['layer'] == layer]


def test_extract_patches_finds_wetland_components_and_shared_banks():
    (links, nodes), Imask = _three_patch_network()

    ex = extract_patches(links, nodes, Imask, closing_iterations=0, min_shared_boundary_px=5)

    assert isinstance(ex, PatchExtraction)
    assert set(ex.layer_nodes) == {1, 2, 3}
    assert ex.island_labels == frozenset()
    # All three patches belong to the border-attached mainland component.
    assert all(spec['area_m2'] == px for spec, px in zip(
        [ex.layer_nodes[1], ex.layer_nodes[2], ex.layer_nodes[3]], [200.0, 180.0, 380.0]))
    # Shared channel banks: patches 1-2 across link 30, 2-3 across link 31.
    assert {(a, b): e['weight'] for a, b, e in ex.adjacency_edges} == {(1, 2): 20.0, (2, 3): 20.0}
    assert all(e['edge_type'] == EdgeType.PATCH_ADJACENCY and e['interlayer'] is False
               for _, _, e in ex.adjacency_edges)


def test_adjacency_edge_requires_shared_boundary_above_threshold():
    (links, nodes), Imask = _three_patch_network()

    above = extract_patches(links, nodes, Imask, closing_iterations=0, min_shared_boundary_px=5)
    at_threshold = extract_patches(links, nodes, Imask, closing_iterations=0, min_shared_boundary_px=20)
    below = extract_patches(links, nodes, Imask, closing_iterations=0, min_shared_boundary_px=21)

    # The shared bank is exactly 20 px: inclusive threshold, then excluded.
    assert len(above.adjacency_edges) == 2
    assert len(at_threshold.adjacency_edges) == 2
    assert below.adjacency_edges == ()

    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask, closing_iterations=0, min_shared_boundary_px=21)
    assert not [e for e in G.edges(data=True) if e[2]['edge_type'] == EdgeType.PATCH_ADJACENCY]


def test_adjacency_edges_attach_as_bidirectional_pairs():
    (links, nodes), Imask = _three_patch_network()
    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask, closing_iterations=0, min_shared_boundary_px=5)

    adjacent = [(u, v, d) for u, v, d in G.edges(data=True)
                if d['edge_type'] == EdgeType.PATCH_ADJACENCY]
    assert len(adjacent) == 4  # two unordered pairs, two directions each
    for u, v, d in adjacent:
        reversed_attrs = [dict(rd) for rd in G[v][u].values() if dict(rd) == dict(d)]
        assert reversed_attrs, f'edge ({u}, {v}) lacks a symmetric reverse pair'
        assert d['interlayer'] is False
        assert d['weight'] == 20.0


def test_every_patch_touching_the_channel_gets_boundary_edge_pairs():
    (links, nodes), Imask = _three_patch_network()

    ex = extract_patches(links, nodes, Imask, closing_iterations=0)
    # Patch 1 touches link 30, patch 2 touches both links, patch 3 touches 31.
    assert {label: sorted(e[1]['channel_linkid'] for e in edges)
            for label, edges in ex.boundary_edges.items()} == {1: [30], 2: [30, 31], 3: [31]}
    assert all(e[1]['edge_type'] == EdgeType.BOUNDARY and e[1]['interlayer'] is True
               for edges in ex.boundary_edges.values() for e in edges)

    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask, closing_iterations=0)
    for label in (1, 2, 3):
        node = _node_for_label(G, label)
        boundary_edges = [(v, d) for _, v, d in G.edges(node, data=True)
                          if d['edge_type'] == EdgeType.BOUNDARY]
        assert boundary_edges, f'patch label {label} has no boundary edges'
        for v, d in boundary_edges:
            reverse_matches = [dict(rd) for rd in G[v][node].values() if dict(rd) == dict(d)]
            assert reverse_matches, 'boundary edge lacks its bidirectional pair'
    validate_multiplex(G)


def test_boundary_edge_snaps_to_nearest_channel_node():
    (links, nodes), Imask, landmass = _lakeside_network()

    ex = extract_patches(links, nodes, Imask, landmass)

    assert list(ex.layer_nodes) == [1]
    assert len(ex.boundary_edges[1]) == 1
    channel_node, attrs = ex.boundary_edges[1][0]
    # Centroid (12.0, 19.0): 7 px from node 0, 23 px from node 1.
    assert channel_node == 0
    assert attrs['channel_linkid'] == 50
    assert attrs['weight'] == 15.0  # 15 bank pixels facing the line
    assert attrs['snapped_xy'] == [12.0, 20.0]


def test_patch_not_touching_the_channel_has_no_boundary_edges():
    (links, nodes), Imask, landmass = _lake_island_network()

    ex = extract_patches(links, nodes, Imask, landmass)

    assert set(ex.layer_nodes) == {1, 2, 3}
    assert ex.island_labels == frozenset()
    assert ex.boundary_edges[1] and ex.boundary_edges[2]
    assert ex.boundary_edges[3] == []
    # Only the two channel-banked patches are adjacent; the lake separates
    # the island from its surrounding patch.
    assert [(a, b) for a, b, _ in ex.adjacency_edges] == [(1, 2)]


def test_closing_merges_fragments_split_by_water_intrusion():
    (links, nodes), Imask, landmass = _protrusion_network()

    unclosed = extract_patches(links, nodes, Imask, landmass, closing_iterations=0)
    closed = extract_patches(links, nodes, Imask, landmass, closing_iterations=1)

    # The 2-px water slit splits the landmass in two; closing bridges it. The
    # channel protrusion never splits the landmass, so subtracting it after
    # closing leaves the merged patch whole.
    assert set(unclosed.layer_nodes) == {1, 2}
    assert set(closed.layer_nodes) == {1}
    assert closed.layer_nodes[1]['area_m2'] == 800 - 6  # full rect minus the channel bite


def test_close_landmass_fills_narrow_protrusions():
    (_links, _nodes), _Imask, landmass = _protrusion_network()

    assert close_landmass(landmass, iterations=0).sum() == landmass.sum()  # no-op
    assert close_landmass(landmass, iterations=1).all()  # 2-px slit filled
    with pytest.raises(ValueError, match='iterations'):
        close_landmass(landmass, iterations=-1)


def test_island_patches_reuse_the_surrounding_link_machinery():
    (links, nodes), Imask = _ring_network()

    ex = extract_patches(links, nodes, Imask)

    # The ring interior is bounded, hence an island; the exterior is mainland.
    assert ex.island_labels == frozenset({2})
    assert sorted(ex.island_link_ids[2]) == [10, 11, 12, 13]
    # The interior island's spec geometry: 14x14 pixels centered mid-ring.
    assert ex.layer_nodes[2]['area_m2'] == 196.0

    # The machinery's surrounding links map directly to boundary edge pairs,
    # one per link, each at that link's nearest endpoint node.
    island_edges = ex.boundary_edges[2]
    assert sorted(e[1]['channel_linkid'] for e in island_edges) == [10, 11, 12, 13]
    assert {e[0] for e in island_edges} == {0, 1, 2, 3}
    assert all(e[1]['weight'] == 14.0 for e in island_edges)  # 14 bank px per side

    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask)
    island_node = _node_for_label(G, 2)
    assert G.nodes[island_node]['layer'] == LayerType.ISLAND
    assert G.nodes[island_node]['node_type'] == NodeType.LANDSCAPE
    assert sorted(G.nodes[island_node]['sur_link_ids']) == [10, 11, 12, 13]
    mainland_nodes = _patch_nodes_by_layer(G, LayerType.WETLAND)
    assert len(mainland_nodes) == 1
    validate_multiplex(G)


def test_island_without_detected_loop_falls_back_to_mask_rule():
    (links, nodes), Imask = _bar_network()

    ex = extract_patches(links, nodes, Imask)

    # The 1-px island is bounded, so it stays island-typed, but loop detection
    # returned the NaN fallback (surrounding links empty) and its single
    # contact group is wired by the mask-intersection rule.
    assert ex.island_labels == frozenset({2})
    assert ex.island_link_ids[2] == []
    assert len(ex.boundary_edges[2]) == 1
    channel_node, attrs = ex.boundary_edges[2][0]
    assert channel_node == 0  # island at col 20: 20 px from node 0, 39 from node 1
    assert attrs['channel_linkid'] == 7
    assert attrs['weight'] == 4.0  # all four 4-neighbors are channel pixels

    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask)
    island_node = _node_for_label(G, 2)
    assert G.nodes[island_node]['layer'] == LayerType.ISLAND
    assert G.nodes[island_node]['sur_link_ids'] == []


def test_detect_islands_false_keeps_every_patch_wetland():
    (links, nodes), Imask = _ring_network()

    ex = extract_patches(links, nodes, Imask, detect_islands=False)

    assert ex.island_labels == frozenset()
    assert ex.island_link_ids == {}
    assert all(len(edges) >= 1 for edges in ex.boundary_edges.values())


def test_patch_layers_are_additive_and_channel_view_is_unchanged():
    (links, nodes), Imask = _three_patch_network()
    G = build_multiplex(links, nodes)
    channel_ids = set(G.nodes)

    add_patch_layer(G, links, nodes, Imask, closing_iterations=0)

    assert channel_ids < set(G.nodes)
    # Landscape node ids are assigned from max(channel ids) + 1, keeping them
    # disjoint from and above every channel id; raster labels stay unique.
    landscape_nodes = set(G.nodes) - channel_ids
    assert all(n > max(channel_ids) for n in landscape_nodes)
    labels = [G.nodes[n]['label_id'] for n in landscape_nodes]
    assert len(labels) == len(set(labels))
    view = channel_view(G)
    assert set(view.nodes) == channel_ids
    assert view.number_of_edges() == sum(1 for u, v, d in G.edges(data=True)
                                         if u in channel_ids and v in channel_ids)


def test_mask_date_flows_to_patch_nodes():
    (links, nodes), Imask = _three_patch_network()

    ex = extract_patches(links, nodes, Imask, closing_iterations=0, mask_date='2026-10-30')
    assert all(spec['mask_date'] == '2026-10-30' for spec in ex.layer_nodes.values())

    G = build_multiplex(links, nodes)
    add_patch_layer(G, links, nodes, Imask, closing_iterations=0, mask_date='2026-10-30')
    assert all(G.nodes[n]['mask_date'] == '2026-10-30'
               for n in set(G.nodes) if G.nodes[n]['layer'] != LayerType.CHANNEL)


def test_pixel_size_scales_weights_and_areas():
    (links, nodes), Imask = _three_patch_network()

    ex = extract_patches(links, nodes, Imask, closing_iterations=0,
                         pixlen=30.0, pixarea=900.0, min_shared_boundary_px=5)

    assert ex.layer_nodes[1]['area_m2'] == 200 * 900.0
    assert ex.adjacency_edges[0][2]['weight'] == 20 * 30.0
    assert ex.boundary_edges[1][0][1]['weight'] == 20 * 30.0


def test_extract_patches_rejects_provisional_ids():
    (links, nodes), Imask = _three_patch_network()
    mark_network_ids_provisional(links, nodes)

    with pytest.raises(RuntimeError, match='finalized'):
        extract_patches(links, nodes, Imask)


def test_empty_landmass_attaches_nothing():
    (links, nodes), Imask = _ring_network()
    G = build_multiplex(links, nodes)
    before = (set(G.nodes), G.number_of_edges())

    ex = extract_patches(links, nodes, Imask, landmass=np.zeros(Imask.shape, dtype=bool))

    assert ex.layer_nodes == {}
    add_patch_layer(G, links, nodes, Imask, landmass=np.zeros(Imask.shape, dtype=bool))
    assert (set(G.nodes), G.number_of_edges()) == before
    validate_multiplex(G)


def test_shared_boundary_lengths_counts_only_facing_channel_pixels():
    (_links, _nodes), Imask, landmass = _lake_island_network()
    Ipatch, _ = label_patches(landmass, Imask)

    shared = shared_boundary_lengths(Ipatch, Imask)

    # Patches 1 and 2 face each other across the full-height line (20 px);
    # the lake-enclosed island shares no channel pixel with anyone.
    assert shared == {(1, 2): 20}

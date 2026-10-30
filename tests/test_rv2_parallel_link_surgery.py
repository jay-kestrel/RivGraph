"""Regression tests for the RivGraph RV-2 parallel-link surgery fixes.

One test per acceptance criterion (Remediation Program Spec RV-2; findings F1
and F8 of the RivGraph adversarial review, art_0AbV9nXo), converted from the
review's runtime probes and pinned to deterministic fixtures:

F1  add_artificial_nodes crashes with ValueError on >=3 parallel links sharing
    two nodes: find_parallel_links emits overlapping pair/triplet entries whose
    link ids are deleted while processing earlier entries. Its connectivity
    fix-up also reverses a local copy of links['conn'] without writing back,
    so a flipped link's split pieces silently invert orientation.
F8  one-pixel links produce a NaN sinuosity (0/0 in link_widths_and_lengths),
    and the SWORD aggregation boundary cannot handle the degenerate
    single-pixel segment (empty width slice).

The conn[0]-is-upstream convention is load-bearing across directionality
(set_link, set_parallel_links) and the exporters (links_to_gpd, geofiles,
SWORD builder), so the split pieces must preserve the original link's
connectivity orientation exactly.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from pyproj import CRS
from scipy.ndimage import distance_transform_edt

from rivgraph import directionality as dir_mod
from rivgraph import io_utils as iou
from rivgraph import ln_utils as lnu

IMSHAPE = (20, 20)
GT = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
WKT = 'EPSG:32615'


# ------------------------------------------------------------------ fixtures

def _path(rows_cols):
    return np.array([int(np.ravel_multi_index((r, c), IMSHAPE))
                     for r, c in rows_cols])


# Three distinct 8-connected paths between the same two junction pixels
# (10, 2) and (10, 17). STRAIGHT is the shortest; UPPER/LOWER are longer.
STRAIGHT = [(10, c) for c in range(2, 18)]
UPPER = [(10, 2)] + [(9, c) for c in range(3, 17)] + [(10, 17)]
LOWER = [(10, 2)] + [(11, c) for c in range(3, 17)] + [(10, 17)]


def _parallel_network(conns, paths, certain=1):
    """Build a synthetic network of parallel links between nodes 0 and 1."""
    links = {
        'id': [1, 2, 3][:len(conns)],
        'conn': [list(c) for c in conns],
        'idx': [_path(p) for p in paths],
        'certain': [certain] * len(conns),
        'certain_alg': [0] * len(conns),
        'certain_order': list(range(1, len(conns) + 1)),
    }
    nodes = {
        'id': [0, 1],
        'idx': [int(np.ravel_multi_index((10, 2), IMSHAPE)),
                int(np.ravel_multi_index((10, 17), IMSHAPE))],
        'conn': [[1, 2, 3][:len(conns)], [1, 2, 3][:len(conns)]],
        'inlets': [0],
        'outlets': [1],
    }
    return links, nodes


@pytest.fixture
def strip_idt():
    """Distance transform of a 3x16 channel strip containing all paths."""
    mask = np.zeros(IMSHAPE, dtype=np.uint8)
    mask[9:12, 2:18] = 1
    return distance_transform_edt(mask)


def _assert_referential_integrity(links, nodes):
    """Every link endpoint must reference an existing node and vice versa."""
    node_ids = set(nodes['id'])
    link_ids = set(links['id'])
    dangling_nodes = [
        lid for lid, conn in zip(links['id'], links['conn'])
        if any(n not in node_ids for n in conn)
    ]
    dangling_links = [
        nid for nid, conn in zip(nodes['id'], nodes['conn'])
        if any(l not in link_ids for l in conn)
    ]
    assert not dangling_nodes, f'links reference missing nodes: {dangling_nodes}'
    assert not dangling_links, f'nodes reference missing links: {dangling_links}'


# ------------------------------------------------------------------ F1: crash

def test_add_artificial_nodes_survives_three_parallel_links(strip_idt):
    """F1: >=3 parallel links must not crash add_artificial_nodes.

    The review's probe: three parallel links between two nodes make
    find_parallel_links emit overlapping pair/triplet entries; breaking a
    link deletes it, and the later entry still referencing the deleted id
    raised ValueError: <id> is not in list.
    """
    links, nodes = _parallel_network([[0, 1]] * 3, [STRAIGHT, UPPER, LOWER])

    out_links, out_nodes = lnu.add_artificial_nodes(links, nodes, IMSHAPE, GT)

    _assert_referential_integrity(out_links, out_nodes)

    # Exactly one unsplit (direct) link remains between the two junctions,
    # and it is the shortest of the parallel set.
    direct = [conn for conn in out_links['conn']
              if conn[0] == 0 and conn[1] == 1]
    assert len(direct) == 1

    # Two artificial nodes were inserted (one per broken parallel link).
    assert len(out_nodes['arts']) == 2

    # The stored 'parallels' attribute is refreshed: no parallel links remain
    # after the split, and directionality.set_parallel_links (which consumes
    # links['parallels']) must not encounter deleted link ids.
    out_links, _ = lnu.find_parallel_links(out_links, out_nodes)
    assert out_links['parallels'] == []
    dir_mod.set_parallel_links(out_links, out_nodes, knownlink=out_links['id'][0])


# ------------------------------------------------------- F1: conn[0] invariant

def test_split_preserves_flipped_link_orientation(strip_idt):
    """F1: splitting a flipped link must preserve its connectivity orientation.

    The original link has conn [1, 0]: node 1 is its upstream end. Its two
    split pieces must keep node 1 at conn[0] and node 0 at conn[1]. The old
    fix-up reversed a local copy, so the pieces silently carried the pixel
    walk order instead (flow reversed in every conn[0] consumer).
    """
    links, nodes = _parallel_network([[0, 1], [1, 0]], [STRAIGHT, UPPER])

    out_links, out_nodes = lnu.add_artificial_nodes(links, nodes, IMSHAPE, GT)

    # The unsplit (shortest) link keeps its orientation untouched.
    assert out_links['conn'][out_links['id'].index(1)] == [0, 1]

    # The flipped link is broken into two pieces joined by artificial node 2.
    assert out_nodes['id'][-1] == 2
    piece_conns = [conn for lid, conn in zip(out_links['id'], out_links['conn'])
                   if lid != 1]
    assert sorted(sorted(c) for c in piece_conns) == [[0, 2], [1, 2]]

    piece_at_us = next(c for c in piece_conns if 1 in c)
    piece_at_ds = next(c for c in piece_conns if 0 in c)
    assert piece_at_us[0] == 1, \
        f'original conn[0]=1 must remain upstream: got {piece_at_us}'
    assert piece_at_ds[1] == 0, \
        f'original conn[1]=0 must remain downstream: got {piece_at_ds}'


def test_directionality_confirms_split_orientation(strip_idt):
    """F1: directionality must agree with the split pieces' orientation.

    set_link(link, usnode) moves usnode to conn[0]; if the split preserved the
    original orientation, setting each piece with its semantic upstream node
    must not flip it. Before the fix this both flipped the piece and crashed
    set_parallel_links on the stale links['parallels'] entries.
    """
    links, nodes = _parallel_network([[0, 1], [1, 0]], [STRAIGHT, UPPER])
    orig_us, orig_ds = links['conn'][links['id'].index(2)]
    orig_ids = set(links['id'])

    out_links, out_nodes = lnu.add_artificial_nodes(links, nodes, IMSHAPE, GT)
    before = [list(c) for c in out_links['conn']]
    art_id = out_nodes['id'][-1]

    for lid in list(out_links['id']):
        lidx = out_links['id'].index(lid)
        conn = out_links['conn'][lidx]
        # Semantic upstream per piece: the original link's upstream node for
        # its us piece, the artificial node for its ds piece; unsplit links
        # keep their own orientation.
        if lid in orig_ids:
            usnode = conn[0]
        elif orig_us in conn:
            usnode = orig_us
        else:
            usnode = art_id
        out_links, out_nodes = dir_mod.set_link(
            out_links, out_nodes, lidx, usnode)

    assert [list(c) for c in out_links['conn']] == before


def test_exporters_preserve_conn0_upstream(strip_idt):
    """F1: both exporter paths must report the preserved orientation.

    links_to_gpd exports 'us node'/'ds node' from conn[0]/conn[1]; the SWORD
    builder derives rg_us_nd/rg_ds_nd the same way. After the split, exported
    direction must match the pieces' connectivity exactly.
    """
    links, nodes = _parallel_network([[0, 1], [1, 0]], [STRAIGHT, UPPER])

    out_links, out_nodes = lnu.add_artificial_nodes(links, nodes, IMSHAPE, GT)

    # Widths must be recomputed after the split (they were dropped on purpose).
    out_links = lnu.link_widths_and_lengths(out_links, strip_idt, pixlen=1)

    # Finalize IDs before exporting (the documented post-topology workflow).
    out_links, out_nodes = lnu.finalize_network_ids(out_links, out_nodes)

    links_gpd = lnu.links_to_gpd(out_links, IMSHAPE, GT, WKT)
    assert list(links_gpd['us node']) == [c[0] for c in out_links['conn']]
    assert list(links_gpd['ds node']) == [c[1] for c in out_links['conn']]

    crs = CRS.from_epsg(32615)
    sword_nodes, sword_reaches = iou.build_sword_geodataframes(
        out_links, out_nodes, IMSHAPE, GT, crs, 'meter')
    assert list(sword_reaches['rg_us_nd']) == [c[0] for c in out_links['conn']]
    assert list(sword_reaches['rg_ds_nd']) == [c[1] for c in out_links['conn']]
    assert all(sword_reaches['fdir_set'])


# ------------------------------------------------------------------ F8: widths

def test_one_pixel_link_defined_aggregates(strip_idt):
    """F8: a one-pixel link must get a defined sinuosity and wid_med.

    The review's probe: sinuosity was length/straight-line = 0/0 = NaN with a
    RuntimeWarning. A degenerate link has no direction to deviate from, so its
    sinuosity is 1.0; wid_med is the median over its (single) pixel width.
    """
    pix = int(np.ravel_multi_index((10, 8), IMSHAPE))
    links = {'id': [0], 'idx': [np.array([pix])], 'conn': [[0, 1]]}

    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any RuntimeWarning fails the test
        out = lnu.link_widths_and_lengths(links, strip_idt, pixlen=1)

    assert out['sinuosity'][0] == pytest.approx(1.0)
    assert out['wid_med'][0] == pytest.approx(float(strip_idt[10, 8] * 2))
    assert np.isfinite(out['wid'][0])
    assert np.isfinite(out['wid_adj'][0])
    assert out['len'][0] == pytest.approx(1.0)
    assert out['len_adj'][0] == pytest.approx(1.0)


def test_mixed_links_shapes_and_dtypes_preserved(strip_idt):
    """F8: aggregates stay finite, aligned, and float-typed for mixed links.

    The spec's NaN-filtering criterion: filtering the degenerate NaNs out of
    the per-link aggregates must preserve output shapes (one value per link
    per attribute) and dtypes (float), with no NaN left behind.
    """
    strip_path = _path(STRAIGHT)
    lone_pix = int(np.ravel_multi_index((10, 8), IMSHAPE))
    links = {
        'id': [0, 1],
        'conn': [[0, 1], [2, 3]],
        'idx': [strip_path, np.array([lone_pix])],
    }

    out = lnu.link_widths_and_lengths(links, strip_idt, pixlen=1)

    attribute_keys = ('wid_pix', 'wid', 'wid_adj', 'wid_med', 'len',
                      'len_adj', 'sinuosity')
    for key in attribute_keys:
        assert len(out[key]) == len(out['id']), key

    for key in ('wid', 'wid_adj', 'wid_med', 'len_adj', 'sinuosity'):
        values = out[key]
        assert all(np.isfinite(v) for v in values), key
        assert all(isinstance(v, (float, np.floating)) for v in values), key

    # The one-pixel link is perfectly sinuous with a defined median width.
    assert out['sinuosity'][1] == pytest.approx(1.0)
    assert out['wid_med'][1] == pytest.approx(float(strip_idt[10, 8] * 2))
    # The straight multi-pixel link keeps sinuosity exactly 1.
    assert out['sinuosity'][0] == pytest.approx(1.0)


def test_sword_export_handles_one_pixel_links(strip_idt):
    """F8: the SWORD aggregation boundary must survive a one-pixel link.

    The degenerate single-pixel segment produced an empty width slice:
    np.mean/np.var returned NaN and np.max raised ValueError. Aggregates must
    stay finite with one output row per segment (shapes preserved).
    """
    pix = int(np.ravel_multi_index((10, 8), IMSHAPE))
    links = {
        'id': [0],
        'conn': [[0, 1]],
        'idx': [np.array([pix])],
        'certain': [1],
    }
    nodes = {
        'id': [0, 1],
        'idx': [pix, pix + 1],
        'conn': [[0], [0]],
        'inlets': [0],
        'outlets': [1],
    }
    links = lnu.link_widths_and_lengths(links, strip_idt, pixlen=1)

    # Finalize IDs before exporting (the documented post-topology workflow).
    links, nodes = lnu.finalize_network_ids(links, nodes)

    crs = CRS.from_epsg(32615)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # no NaN RuntimeWarnings may escape
        sword_nodes, sword_reaches = iou.build_sword_geodataframes(
            links, nodes, IMSHAPE, GT, crs, 'meter')

    # One segment row for the one-pixel link, with finite aggregates.
    assert len(sword_nodes) == 1
    assert len(sword_reaches) == 1
    assert np.isfinite(sword_nodes['width'].iloc[0])
    assert np.isfinite(sword_nodes['max_width'].iloc[0])
    assert np.isfinite(sword_nodes['width_var'].iloc[0])
    assert sword_nodes['sinuosity'].iloc[0] == pytest.approx(1.0)

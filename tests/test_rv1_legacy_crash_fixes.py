"""Regression tests for the RivGraph RV-1 legacy crash + integrity fixes.

One test per adversarial-review finding (art_0AbV9nXo), converted from the
review's runtime probes and pinned to deterministic fixtures:

F3  remove_super_apex_from_network iterates the live connectivity list while
    delete_link mutates it, leaving a dangling link->node reference.
F9  dist_from_apex indexes nodes['id'] with the inlets LIST for single-inlet
    deltas (ValueError: [0] is not in list).
F4  inflection_pts_oversmooth crashes on intersection geometry without len()
    (Point/MultiPoint in shapely 2.1), and its window ladder can emit a
    window equal to the polyorder (savgol ValueError).
F5  cl_migration_transect_matching.get_along_cl_distance breaks after the
    first split segment regardless of the origin match.
F6  clip_by_shoreline leaves min_idx/max_idx unbound (or stale from a previous
    link) for non-point intersection geometry and constructs-but-does-not-
    raise its RuntimeError.
F7  skel_to_graph silently drops isolated pixels (or crashes with KeyError
    'pop from an empty set' when one is the only component).
"""
from __future__ import annotations

import numpy as np
import pytest
from loguru import logger
from shapely.geometry import LineString, Point

from rivgraph import mask_to_graph as mtg
from rivgraph.deltas import delta_metrics as dm
from rivgraph.deltas import delta_utils as du
from rivgraph.ln_utils import links_to_gpd
from rivgraph.rivers import centerline_utils as cu


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def y_network():
    """Two-inlet network joined at a shared junction (probe fixture for F3)."""
    imshape = (10, 10)
    links = {
        "id": [0, 1],
        "conn": [[0, 2], [1, 2]],
        "idx": [[2, 55], [7, 55]],
        "wid_adj": [10.0, 10.0],
        "wid": [10.0, 10.0],
        "wid_med": [10.0, 10.0],
        "sinuosity": [1.0, 1.0],
        "len": [53.0, 48.0],
        "len_adj": [53.0, 48.0],
    }
    nodes = {
        "id": [0, 1, 2],
        "idx": [2, 7, 55],
        "conn": [[0], [1], [0, 1]],
        "inlets": [0, 1],
    }
    return links, nodes, imshape


def _assert_referential_integrity(links, nodes):
    """Every link endpoint must reference an existing node and vice versa."""
    node_ids = set(nodes["id"])
    link_ids = set(links["id"])
    dangling_nodes = [
        lid for lid, conn in zip(links["id"], links["conn"])
        if any(n not in node_ids for n in conn)
    ]
    dangling_links = [
        nid for nid, conn in zip(nodes["id"], nodes["conn"])
        if any(l not in link_ids for l in conn)
    ]
    assert not dangling_nodes, f"links reference missing nodes: {dangling_nodes}"
    assert not dangling_links, f"nodes reference missing links: {dangling_links}"


# ------------------------------------------------------------------ F3

def test_remove_super_apex_from_network_preserves_integrity(y_network):
    """F3: removing the super apex must not leave dangling references.

    The probe showed a surviving link whose conn pointed at the deleted
    super-apex node because delete_link mutated the list being iterated.
    """
    import networkx as nx

    links, nodes, imshape = y_network
    links, nodes = dm.add_super_apex_to_network(links, nodes, imshape)
    super_apex = nodes["super_apex"]

    links, nodes = dm.remove_super_apex_from_network(links, nodes)

    assert super_apex not in nodes["id"]
    _assert_referential_integrity(links, nodes)

    # The surviving network must build as a graph without KeyErrors.
    G = nx.MultiGraph()
    G.add_nodes_from(nodes["id"])
    for lk, lc in zip(links["id"], links["conn"]):
        G.add_edge(lc[0], lc[1], key=lk)
    assert G.number_of_edges() == len(links["id"])


# ------------------------------------------------------------------ F9

def test_dist_from_apex_single_inlet():
    """F9: a single-inlet delta must not index nodes['id'] with a list."""
    nodes = {"id": [0, 1], "idx": [0, 22], "inlets": [0], "conn": [[0], [0]]}
    imshape = (10, 10)

    dists = dm.dist_from_apex(nodes, imshape)

    assert len(dists) == 2
    assert dists[0] == pytest.approx(0.0)  # apex node sits at distance zero
    assert dists[1] == pytest.approx(1.0)  # farthest node normalizes to 1


def test_dist_from_apex_multi_inlet():
    """The multi-inlet averaging branch keeps working after the F9 fix."""
    nodes = {"id": [0, 1, 2], "idx": [2, 7, 55], "inlets": [0, 1],
             "conn": [[0], [1], [0, 1]]}
    imshape = (10, 10)

    dists = dm.dist_from_apex(nodes, imshape)

    assert len(dists) == 3
    arr = np.asarray(dists)
    assert np.all(arr >= 0) and np.all(arr <= 1)


# ------------------------------------------------------------------ F4

def test_inflection_pts_oversmooth_short_line_returns_indices():
    """F4: a short line skips the window search and must survive the final
    intersection extraction (loop is skipped because post - prev <= 10).

    At HEAD 06f54b4 the unpatched function cannot reach the final block for
    any input (float smoothing window for short lines, polyorder-equal window
    or unsized MultiPoint for long lines); the fix repairs those paths and
    this fixture pins the single-Point/MultiPoint normalization contract.
    """
    n = 45
    xs = np.arange(n, dtype=float)
    ys = 20.0 * np.exp(-(xs / 6.0) ** 2) + np.sin(2 * np.pi * xs / 90.0)

    idx, smoothline = cu.inflection_pts_oversmooth(xs, ys, 2)

    assert smoothline.shape == (2, n)
    idx = np.asarray(idx)
    assert len(idx) == 5  # pinned: stable MultiPoint final intersection
    assert np.all(np.diff(idx) > 0)  # sorted, unique
    assert np.all((idx >= 0) & (idx < n))


def test_inflection_pts_oversmooth_long_line_no_crash():
    """F4: the window search and final extraction must survive MultiPoint
    intersections (shapely 2.1 Multi* types have no len())."""
    n = 240
    xs = np.arange(n, dtype=float)
    ys = np.sin(2 * np.pi * xs / n)

    idx, smoothline = cu.inflection_pts_oversmooth(xs, ys, 2)

    assert smoothline.shape == (2, n)
    idx = np.asarray(idx)
    assert np.all(np.diff(idx) > 0)
    assert np.all((idx >= 0) & (idx < n))


def test_inflection_pts_oversmooth_window_ladder_no_polyorder_window():
    """F4: the window ladder must never emit a window <= polyorder.

    For lengths whose raw linspace contains the polyorder itself (e.g. 200),
    the unpatched ladder reaches savgol(window=polyorder) and raises."""
    n = 200
    xs = np.arange(n, dtype=float)
    ys = np.sin(2 * np.pi * xs / n)

    idx, smoothline = cu.inflection_pts_oversmooth(xs, ys, 2)

    assert smoothline.shape == (2, n)
    idx = np.asarray(idx)
    assert np.all(np.diff(idx) > 0)
    assert np.all((idx >= 0) & (idx < n))


def test_intersection_point_coords_normalizes_geometry():
    """F4 unit: the normalizer handles every intersection type deterministically.

    The review's probe: two crossing LineStrings intersect in a bare Point,
    and len(Point) raises TypeError. The normalizer must accept it."""
    # bare Point (the review's probe)
    pt = LineString([(0, 0), (2, 2)]).intersection(LineString([(0, 2), (2, 0)]))
    assert pt.geom_type == "Point"
    coords = cu._intersection_point_coords(pt)
    assert coords.shape == (1, 2)
    assert coords[0][0] == pytest.approx(1.0)
    assert coords[0][1] == pytest.approx(1.0)

    # MultiPoint
    from shapely.geometry import MultiPoint
    mp = MultiPoint([(0, 0), (5, 5)])
    coords = cu._intersection_point_coords(mp)
    assert coords.shape == (2, 2)

    # GeometryCollection containing a Point and a LineString
    from shapely.geometry import GeometryCollection
    gc = GeometryCollection([Point(1, 1), LineString([(0, 0), (1, 0)])])
    coords = cu._intersection_point_coords(gc)
    assert coords.shape == (1, 2)

    # empty and non-point geometries yield no points
    assert cu._intersection_point_coords(Point(0, 0).buffer(0).intersection(Point(1, 1))).shape == (0, 2)
    assert cu._intersection_point_coords(LineString([(0, 0), (1, 1)])).shape == (0, 2)


# ------------------------------------------------------------------ F5

def _migration_fixture(tmp_path):
    """Straight parallel centerlines with one user matcher (probe fixture)."""
    import geopandas as gpd

    x1 = np.array([0.0, 5.0, 10.0])
    y1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([0.0, 5.0, 10.0])
    y2 = np.array([1.0, 1.0, 1.0])
    mfile = tmp_path / "matchers.gpkg"
    gpd.GeoDataFrame(
        geometry=[LineString([(5, -1), (5, 2)])], crs="EPSG:4326"
    ).to_file(mfile, driver="GPKG")
    return str(mfile), x1, y1, x2, y2


def test_cl_migration_transect_matching_natural_order(tmp_path):
    """F5: the public migration API must complete on a straight fixture.

    Pre-fix this raised TypeError: 'GeometryCollection' object is not
    iterable (shapely 2.1 split results are not directly iterable)."""
    mfile, x1, y1, x2, y2 = _migration_fixture(tmp_path)

    mr_matched, pts1, pts2 = cu.cl_migration_transect_matching(
        mfile, x1, y1, x2, y2, 1.0, 1.0
    )

    # second centerline is the first translated by exactly 1 unit
    assert len(pts1) == len(pts2)
    rates = np.asarray(mr_matched)
    assert np.all(np.isfinite(rates))
    assert rates == pytest.approx(np.full_like(rates, 1.0))


def test_cl_migration_transect_matching_reversed_split_order(tmp_path, monkeypatch):
    """F5: the break must sit under its origin-match condition so both split
    segment orders give the same distances.

    The monkeypatch reverses the emission order GEOS does not naturally emit
    for this fixture; pre-fix the unconditional break examined only the first
    segment and returned an unbound dist."""
    from shapely.geometry import GeometryCollection
    from shapely.ops import split as shp_split

    real_split = shp_split

    def reversed_split(linestring, splitter):
        result = real_split(linestring, splitter)
        if isinstance(result, GeometryCollection):
            return GeometryCollection(list(result.geoms)[::-1])
        return result

    mfile, x1, y1, x2, y2 = _migration_fixture(tmp_path)

    mr_natural, pts1_natural, pts2_natural = cu.cl_migration_transect_matching(
        mfile, x1, y1, x2, y2, 1.0, 1.0
    )

    import rivgraph.rivers.centerline_utils as clu
    monkeypatch.setattr(clu, "split", reversed_split)
    mr_reversed, pts1_rev, pts2_rev = cu.cl_migration_transect_matching(
        mfile, x1, y1, x2, y2, 1.0, 1.0
    )
    monkeypatch.undo()

    np.testing.assert_allclose(mr_reversed, mr_natural)
    assert len(pts1_rev) == len(pts1_natural)


# ------------------------------------------------------------------ F6

WKT = "GEOGCS[\"WGS 84\",DATUM[\"WGS_84\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]]"
IMSHAPE = (12, 12)
GT = (0, 1, 0, 0, 0, 1)
SHORE = LineString([(-1, 5.5), (13, 5.5)])


def _make_links(chains_rc):
    chains = [[r * 12 + c for (r, c) in ch] for ch in chains_rc]
    n = len(chains)
    return {
        "id": list(range(n)),
        "conn": [[0, 1], [1, 2]][:n],
        "idx": chains,
        "wid_adj": [10.0] * n,
        "wid": [10.0] * n,
        "wid_med": [10.0] * n,
        "sinuosity": [1.0] * n,
        "len": [10.0] * n,
        "len_adj": [10.0] * n,
    }


def _make_nodes(pixels_rc, conns, inlets):
    return {
        "id": list(range(len(pixels_rc))),
        "idx": [r * 12 + c for (r, c) in pixels_rc],
        "conn": conns,
        "inlets": list(inlets),
    }


@pytest.fixture
def shoreline_file(tmp_path):
    import geopandas as gpd

    path = tmp_path / "shoreline.gpkg"
    # the 'id' column is required for the spatial join's id_left suffix
    gpd.GeoDataFrame({"id": [0]}, geometry=[SHORE], crs="EPSG:4326").to_file(
        path, driver="GPKG"
    )
    return str(path)


def test_clip_by_shoreline_raises_on_non_point_intersection(shoreline_file):
    """F6: a link whose shoreline intersection carries no cut point must
    raise, not fall through with unbound cut indices.

    Pre-fix this raised UnboundLocalError on min_idx."""
    links = _make_links([[(3, 2), (5, 4), (5, 6), (7, 7)]])  # middle segment runs along the shore
    nodes = _make_nodes([(3, 2), (7, 7)], [[0], [0]], [0])

    with pytest.raises(RuntimeError, match="[Uu]nexpected"):
        du.clip_by_shoreline(links, nodes, shoreline_file, IMSHAPE, GT, WKT)


def test_clip_by_shoreline_handles_geometry_collection(shoreline_file):
    """F6: GeometryCollection intersections must cut at their own point
    parts; stale indices from a previous link must never be reused.

    Pre-fix, linkB's GeometryCollection matched no branch, so linkA's cut
    indices were silently reused and linkB was clipped at the wrong pixels
    (its remnant vanished), with the inlet node misreported as an outlet."""
    # linkA crosses the shore twice (MultiPoint); linkB touches it with a
    # mixed Point/LineString GeometryCollection (geometry probed at HEAD)
    links = _make_links([
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (6, 6), (4, 8)],
        [(4, 8), (6, 9), (5, 9), (5, 10), (7, 11)],
    ])
    links["conn"] = [[0, 1], [1, 2]]
    nodes = _make_nodes([(0, 0), (4, 8), (7, 11)], [[0], [0, 1], [1]], [0])

    lg = links_to_gpd(links, IMSHAPE, GT, WKT)
    inter_a = lg["geometry"][0].intersection(SHORE)
    inter_b = lg["geometry"][1].intersection(SHORE)
    assert inter_a.geom_type == "MultiPoint"  # two cut points on linkA
    assert inter_b.geom_type == "GeometryCollection"  # mixed geometry on linkB

    out_links, out_nodes = du.clip_by_shoreline(
        links, nodes, shoreline_file, IMSHAPE, GT, WKT
    )

    _assert_referential_integrity(out_links, out_nodes)

    # pinned: the landward remnant of linkA survives the largest-network
    # prune with exactly its upstream pixels (cut before the crossings)
    assert out_links["idx"] == [[0, 13, 26]]


# ------------------------------------------------------------------ F7

SKEL_LINE_PLUS_PIXEL = np.zeros((7, 7), dtype=int)
SKEL_LINE_PLUS_PIXEL[3, 1:4] = 1  # a 3-pixel line component
SKEL_LINE_PLUS_PIXEL[0, 6] = 1  # an isolated pixel

SKEL_PIXEL_ONLY = np.zeros((3, 3), dtype=int)
SKEL_PIXEL_ONLY[1, 1] = 1  # nothing but an isolated pixel

_NATIVE_PATHS = pytest.mark.parametrize("native_env", ["1", "0"], ids=["native", "python"])


@_NATIVE_PATHS
def test_skel_to_graph_isolated_pixel_dropped_with_warning(native_env, monkeypatch):
    """F7: an isolated pixel beside a real component is dropped, loudly."""
    records = []
    handler = logger.add(records.append, level="WARNING")
    try:
        monkeypatch.setenv("PYDELTARCM_NATIVE", native_env)
        links, nodes = mtg.skel_to_graph(SKEL_LINE_PLUS_PIXEL.copy())
    finally:
        logger.remove(handler)

    # the isolated pixel is absent from the resolved network
    isolated_idx = 0 * 7 + 6
    all_link_idx = {i for idxs in links["idx"] for i in idxs}
    all_node_idx = set(nodes["idx"])
    assert isolated_idx not in all_link_idx
    assert isolated_idx not in all_node_idx

    # ...and the drop is announced
    assert any("isolated" in str(r).lower() for r in records), records

    # the surviving line component is intact
    _assert_referential_integrity(links, nodes)
    assert len(links["id"]) == 1


@_NATIVE_PATHS
def test_skel_to_graph_isolated_pixel_only_raises(native_env, monkeypatch):
    """F7: a skeleton containing only an isolated pixel raises a descriptive
    error instead of KeyError: 'pop from an empty set'."""
    monkeypatch.setenv("PYDELTARCM_NATIVE", native_env)

    with pytest.raises(ValueError, match="isolated"):
        mtg.skel_to_graph(SKEL_PIXEL_ONLY.copy())

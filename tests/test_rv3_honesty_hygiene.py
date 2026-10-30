"""RV-3 regression tests: directionality honesty and metrics API hygiene.

Converts the review probes of the RivGraph adversarial review (RV-3 scope,
findings F2, F10-F14) into deterministic regression tests, and pins the
conn[0]-is-upstream invariant through directionality.set_continuity (the
RV-2 research handoff).

Reviews: Adversarial Review -- RivGraph (art_0AbV9nXo); RV-2 research
(art_BF5FplFU) handoff note on set_continuity's parallel-handling block.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString

from rivgraph import directionality as dy
from rivgraph import ln_utils as lnu
from rivgraph.classes import centerline, rivnetwork
from rivgraph.deltas import delta_metrics as dm
from rivgraph.rivers import centerline_utils as rcu
from rivgraph.rivers import river_directionality as rdi
from rivgraph.rivers import river_utils as ru

REPO_ROOT = Path(__file__).resolve().parents[1]
GT = (0, 1, 0, 11, 0, -1)
IMSHAPE = (20, 20)


def _path(rows_cols):
    return np.array([int(np.ravel_multi_index((r, c), IMSHAPE))
                     for r, c in rows_cols])


# ---------------------------------------------------------------------- F2

def test_unresolved_links_stay_uncertain(loguru_records):
    """F2: links no algorithm evaluated keep certain=0.

    The old tail of set_directionality warned that remaining links "were
    randomly set" while actually setting nothing and then promoting every
    link to certain=1 -- making arbitrary orientations indistinguishable
    from verified ones downstream (continuity checks, exporters).
    """
    links = {'id': [0, 1, 2], 'certain': np.array([1.0, 0.0, 0.0])}
    records = loguru_records()

    out_links = rdi.mark_unresolved_links(links)

    # Never-evaluated links are not promoted; certain stays 1-D.
    assert list(out_links['certain']) == [1.0, 0.0, 0.0]
    assert out_links['certain'].shape == (3,)

    warnings = [r for r in records if r['level'].name == 'WARNING']
    assert warnings, 'expected a warning about unresolved links'
    assert any('2' in r['message'] for r in warnings), warnings
    assert any('uncertain' in r['message'].lower() for r in warnings), warnings
    assert not any('randomly set' in r['message'] for r in warnings), warnings


def test_set_directionality_leaves_nothing_silently_promoted(
        monkeypatch, tmp_path, loguru_records):
    """F2 wiring: set_directionality runs the unresolved-links honesty step.

    Spy-wraps the seam while delegating to it, then runs the meander river
    pipeline end-to-end and checks the postconditions on the real network:
    certain stays 1-D with only {0, 1} values.
    """
    original = rdi.mark_unresolved_links
    calls = []

    def spy(links):
        calls.append(list(links['id']))
        return original(links)

    monkeypatch.setattr(rdi, 'mark_unresolved_links', spy)

    from rivgraph.classes import river

    d = river(
        'Meandering',
        str(REPO_ROOT / 'examples' / 'data' / 'Meandering_River' / 'meander_mask.tif'),
        results_folder=str(tmp_path),
        exit_sides='NS',
    )
    d.skeletonize()
    d.compute_network()
    d.prune_network()
    d.assign_flow_directions()

    assert calls, 'set_directionality never invoked the unresolved-links step'
    certain = np.asarray(d.links['certain'])
    assert certain.shape == (len(d.links['id']),)
    assert set(np.unique(certain)) <= {0.0, 1.0}


# ---------------------------------------------------- RV-2 handoff (pin)

def test_set_continuity_preserves_conn0_upstream_invariant():
    """Handoff pin: set_continuity routes unknown-link resolution through
    set_link, which writes conn and idx back -- the conn[0]-is-upstream
    invariant holds for a flipped-storage link resolved at a 2-link node.

    Link 1 is stored with its pixel path oriented from its downstream node
    (idx[0] at node 2's pixel); once continuity resolves it as flowing out
    of node 1, both its conn and idx must be persisted accordingly.
    """
    links = {
        'id': [0, 1],
        'conn': [[0, 1], [1, 2]],
        'idx': [_path([(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
                       (10, 8), (10, 9)]),
                _path([(10, 16), (10, 15), (10, 14), (10, 13), (10, 12),
                       (10, 11), (10, 10), (10, 9)])],
        'wid_pix': [np.full(8, 2.0), np.full(8, 2.0)],
    }
    nodes = {
        'id': [0, 1, 2],
        'idx': [int(np.ravel_multi_index((10, 2), IMSHAPE)),
                int(np.ravel_multi_index((10, 9), IMSHAPE)),
                int(np.ravel_multi_index((10, 16), IMSHAPE))],
        'conn': [[0], [0, 1], [1]],
        'inlets': [0],
        'outlets': [2],
    }
    links, nodes = dy.add_directionality_trackers(links, nodes, 'delta')
    links['certain'][0] = 1  # link 0 flows into node 1

    links, nodes = dy.set_continuity(links, nodes, checknodes=[1])

    # Link 1 must be persisted flowing out of node 1 (upstream == node 1).
    assert links['conn'][1] == [1, 2]
    assert links['conn'][1][0] == 1
    assert links['idx'][1][0] == nodes['idx'][1]
    assert links['certain'][1] == 1

    # Global invariant: no link references a missing node, and every
    # two-ended link keeps distinct endpoints in upstream-first order.
    node_ids = set(nodes['id'])
    for conn in links['conn']:
        assert all(n in node_ids for n in conn)
        assert conn[0] != conn[1]


# --------------------------------------------------------------------- F10

def _diamond_network():
    links = {
        'id': [10, 11, 12, 13],
        'conn': [[0, 1], [0, 2], [1, 3], [2, 3]],
        'wid_adj': [1.0, 1.0, 1.0, 1.0],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'conn': [[10, 11], [10, 12], [11, 13], [12, 13]],
        'inlets': [0],
        'outlets': [3],
    }
    return links, nodes


def _ner_deltavars():
    links, nodes = _diamond_network()
    _, deltavars = dm.compute_delta_metrics(
        links, nodes, metrics=['n_alt_paths'], warn_experimental=False,
        return_intermediates=True)
    return deltavars


def test_delta_ner_seed_is_reproducible():
    """F10: the same seed produces the same randomizations, every run."""
    deltavars = _ner_deltavars()

    first = dm.delta_nER(deltavars, N=20, seed=42)
    second = dm.delta_nER(deltavars, N=20, seed=42)

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]


def test_delta_ner_accepts_generator():
    """F10: an explicit Generator is honored (two identical draws)."""
    deltavars = _ner_deltavars()

    first = dm.delta_nER(deltavars, N=20, seed=np.random.default_rng(7))
    second = dm.delta_nER(deltavars, N=20, seed=np.random.default_rng(7))

    assert first[2] == second[2]


def test_delta_ner_seed_none_remains_supported():
    """F10: seed=None is the documented unseeded escape hatch and runs."""
    deltavars = _ner_deltavars()
    nER_Delta, pExc, randA = dm.delta_nER(deltavars, N=5, seed=None)

    assert len(randA) == 5
    assert 0.0 <= pExc <= 1.0


def test_compute_delta_metrics_seed_reproducible():
    """F10: compute_delta_metrics threads the seed through to nER."""
    links, nodes = _diamond_network()
    names = ['nonlin_entropy_rate', 'nER_prob_exceedence', 'nER_randomized']

    first = dm.compute_delta_metrics(
        links, nodes, metrics=names, warn_experimental=False, seed=42)
    second = dm.compute_delta_metrics(
        links, nodes, metrics=names, warn_experimental=False, seed=42)

    assert first['nonlin_entropy_rate'] == second['nonlin_entropy_rate']
    assert first['nER_prob_exceedence'] == second['nER_prob_exceedence']
    assert first['nER_randomized'] == second['nER_randomized']


# --------------------------------------------------------------------- F11

def _bifurcation_network():
    """Y-junction at node 1: link 0 enters, links 1 and 2 leave.

    Link 2 bends within the first few pixels so that pixel weighting
    changes its direction vector -- distinguishing 'lin'/'linear' results
    from the unweighted default.
    """
    links = {
        'id': [0, 1, 2],
        'conn': [[0, 1], [1, 2], [1, 3]],
        'idx': [_path([(10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7),
                       (10, 8), (10, 9)]),
                _path([(10, 9), (10, 10), (10, 11), (10, 12), (10, 13),
                       (10, 14), (10, 15), (10, 16)]),
                _path([(10, 9), (11, 9), (12, 9), (13, 9), (14, 9), (15, 10),
                       (16, 11), (16, 12)])],
        'wid_adj': [5.0, 5.0, 5.0],
        'wid_pix': [np.full(8, 2.0), np.full(8, 2.0), np.full(8, 2.0)],
        'certain': [1, 1, 1],
    }
    nodes = {
        'id': [0, 1, 2, 3],
        'idx': [int(np.ravel_multi_index((10, 2), IMSHAPE)),
                int(np.ravel_multi_index((10, 9), IMSHAPE)),
                int(np.ravel_multi_index((10, 16), IMSHAPE)),
                int(np.ravel_multi_index((16, 12), IMSHAPE))],
        'conn': [[0], [0, 1, 2], [1], [2]],
        'inlets': [0],
        'outlets': [2, 3],
    }
    return links, nodes


def test_junction_angles_accepts_documented_lin_weight():
    """F11: the documented weight='lin' behaves as the implemented 'linear'."""
    links, nodes = _bifurcation_network()

    nodes_lin = lnu.junction_angles(links, nodes, IMSHAPE, pixlen=1,
                                    weight='lin')
    nodes_linear = lnu.junction_angles(links, nodes, IMSHAPE, pixlen=1,
                                       weight='linear')
    nodes_none = lnu.junction_angles(links, nodes, IMSHAPE, pixlen=1,
                                     weight=None)

    assert nodes_lin['int_ang'][1] == nodes_linear['int_ang'][1]
    # The bifurcation bends, so weighting must matter at all -- otherwise
    # this test could not distinguish 'lin' from the unweighted default.
    assert nodes_lin['int_ang'][1] != nodes_none['int_ang'][1]


# --------------------------------------------------------------------- F12

def _island_net():
    net = object.__new__(rivnetwork)
    imask = np.ones((11, 11), dtype=bool)
    imask[4:7, 4:7] = 0
    net.Imask = imask
    net.pixlen = 1.0
    net.pixarea = 1.0
    net.crs = None
    net.gt = GT
    return net


def test_get_islands_does_not_mutate_shared_default():
    """F12: the props default list is never mutated in place.

    The old signature bound a shared default list and removed
    'surrounding_links' from it on the first call, silently changing the
    documented behavior of every later call (and of caller-provided lists).
    """
    net = _island_net()

    net.get_islands()
    default_now = rivnetwork.get_islands.__defaults__[0]
    assert default_now is None or 'surrounding_links' in default_now

    net.get_islands()  # a second call still sees the full documented default

    my_props = ['area', 'surrounding_links']
    net.get_islands(props=my_props)
    assert my_props == ['area', 'surrounding_links']


# --------------------------------------------------------------------- F13

def test_compute_eBI_raises_when_all_intersecting_widths_zero(tmp_path):
    """F13: a transect crossing only zero-width links raises as advertised.

    The old code filtered widths to positives, divided by an all-NaN sum,
    and -- because NaN == 0 is False -- silently returned eBI == 1 instead
    of the documented RuntimeError.
    """
    links_gdf = gpd.GeoDataFrame(
        {'wid_adj': [0.0]},
        geometry=[LineString([(0, 0), (10, 0)])],
    )
    mesh_gdf = gpd.GeoDataFrame(
        geometry=[LineString([(5, -1), (5, 1)])],
    )
    links_f = tmp_path / 'links.shp'
    mesh_f = tmp_path / 'mesh.shp'
    links_gdf.to_file(links_f)
    mesh_gdf.to_file(mesh_f)

    with pytest.raises(RuntimeError, match='width 0'):
        ru.compute_eBI(str(mesh_f), str(links_f), method='avg')


# --------------------------------------------------------------------- F14

def test_mig_rate_zs_is_retired_with_clear_error():
    """F14: the unusable ZS path is retired with an explanatory error.

    The old method appended a hardcoded personal Windows path and imported
    an unbundled module -- a ModuleNotFoundError naming C:\\Users\\... on
    any machine, instead of a actionable message.
    """
    cl = object.__new__(centerline)

    with pytest.raises(RuntimeError, match='mig_rate_zs'):
        cl.mig_rate_zs(1, 2, 3)


def test_infsC_computes_inflection_points_from_curvature():
    """F14: infsC feeds the computed curvature array to inflection_points.

    The old method passed the bound method C itself (the hasattr guard was
    always true since C is a method), which crashed on subscripting inside
    inflection_points. Here C is exercised on real centerline coordinates.
    """

    class _CurvatureHost:
        C = centerline.C  # bound via descriptor protocol; C(x, y) is pure

    host = _CurvatureHost()
    xs = np.linspace(0.0, 10.0, 200)
    ys = np.sin(xs)

    centerline.infsC(host, x=xs, y=ys)

    assert isinstance(host.infs_C, np.ndarray)
    # The sine centerline has zero-crossings of curvature: inflection
    # indices exist and are increasing.
    assert host.infs_C.size > 0
    assert np.all(np.diff(host.infs_C) > 0)


# --------------------------------------------------- curvature cross-check

def test_curvars_and_inflection_points_agree_on_sine_signal():
    """Guard for the infsC fix: curvars' output is directly consumable by
    inflection_points (the exact call chain infsC now uses)."""
    xs = np.linspace(0.0, 10.0, 200)
    ys = np.sin(xs)

    Cs, _, _ = rcu.curvars(xs, ys, unwrap=True)
    infs = rcu.inflection_points(Cs)

    assert isinstance(infs, np.ndarray)
    assert infs.size > 0

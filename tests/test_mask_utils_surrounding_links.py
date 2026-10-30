"""Regression tests for surrounding_link_properties (finding I-M6).

Islands whose surroundings contain no cycle (e.g. a straight reach with no
junctions) must receive NaN surrounding-link metrics instead of crashing on
the internal assert after the warning.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from rivgraph import mask_utils as mu


def _straight_link_network(shape):
    """One straight horizontal reach: two nodes joined by a single link."""
    imask = np.zeros(shape, dtype=bool)
    imask[15, 3:27] = True
    link_idx = np.ravel_multi_index((np.full(24, 15), np.arange(3, 27)), shape)
    links = {"id": [100], "idx": [link_idx], "conn": [[1, 2]]}
    nodes = {
        "id": [1, 2],
        "idx": [int(link_idx[0]), int(link_idx[-1])],
        "conn": [[100], [100]],
    }
    return imask, links, nodes


def test_no_surrounding_cycle_island_gets_nan_metrics_without_raising():
    shape = (30, 30)
    imask, links, nodes = _straight_link_network(shape)

    iislands = np.zeros(shape, dtype=int)
    iislands[9:12, 9:12] = 1
    islands = gpd.GeoDataFrame({"id": [1], "geometry": [box(9, 9, 12, 12)]})

    out = mu.surrounding_link_properties(
        links, nodes, imask, islands, iislands, 1.0, 1.0
    )

    assert len(out) == 1
    assert np.isnan(out.sur_area.values[0])
    assert np.isnan(out.sur_avg_wid.values[0])
    assert np.isnan(out.sur_max_wid.values[0])
    assert np.isnan(out.sur_min_wid.values[0])
    assert out.sur_link_ids.values[0] == ""

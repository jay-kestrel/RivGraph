"""Small tests for walk.py helpers."""
from __future__ import annotations

import numpy as np

from tests._helpers import TESTS_ROOT, require_raster_runtime


def _walk_module():
    require_raster_runtime()
    from rivgraph import walk

    return walk


def test_idcs_no_turnaround_cases():
    walk = _walk_module()
    Iskel = np.zeros((5, 5))
    expected = {
        (0, 6): [7, 11, 12],
        (0, 5): [9, 10, 11],
        (0, 4): [3, 8, 9],
        (3, 4): [0, 5, 10],
        (4, 0): [-5, -4, 1],
        (5, 0): [-6, -5, -4],
        (6, 0): [-1, -6, -5],
    }
    for idcs, out in expected.items():
        poss_walk_idcs = walk.idcs_no_turnaround(list(idcs), Iskel)
        assert np.all(poss_walk_idcs == out)


def test_idcs_no_turnaround_westward_branch():
    """The idxdif == 1 (walking left) branch must offer down-left, not duplicate up-left.

    The branch listed the up-left offset (-ncols-1) twice and never offered
    the down-left offset (ncols-1), so links walking leftward could not step
    down-left and terminated early or hit the 'Ambiguous which step' error.
    """
    walk = _walk_module()

    # Findings-doc repro: [54, 53] walks left on a 10-column grid; the three
    # allowed steps are up-left (-11), left (-1), down-left (+9) from idx 53.
    assert np.all(walk.idcs_no_turnaround([54, 53], np.zeros((10, 10))) == [42, 52, 62])

    # Same missing branch on the 5-column grid used by
    # test_idcs_no_turnaround_cases: from idx 4, offsets -6, -1, +4.
    assert np.all(walk.idcs_no_turnaround([5, 4], np.zeros((5, 5))) == [-2, 3, 8])


def test_isbp_parsimonious_single_branchpoint_returns_flat_list():
    """Single-branchpoint results are flat index lists, not nested lists.

    The len(maxnax) == 1 and len(maxfour) == 1 paths wrapped the already-flat
    result in an extra list ([[idx]]), violating the flat-list contract of
    every other return path in the function.
    """
    walk = _walk_module()
    from rivgraph import im_utils as iu

    # A plus junction at (4, 4): the only conn>2 pixel in the image, so the
    # single-branchpoint path (len(maxnax) == 1) is exercised.
    I = np.zeros((9, 9))
    I[4, :] = 1
    I[2:7, 4] = 1
    Ic = iu.im_connectivity(I)
    Ina = walk.naxes_connectivity(I)
    Inf = iu.nfour_connectivity(I)

    bps = walk.isbp_parsimonious(Ic, np.ravel(Ic), np.ravel(Ina), np.ravel(Inf))

    assert bps == [40]  # was [[40]] before the fix


def test_is_bp_window_growth_pairs_bounds_with_dimensions():
    """The window-growth loop must test each dimension against its own bound.

    The loop compared column coordinates against the row bound and row
    coordinates against the column bound while growing the opposite
    dimension. On this seeded random-walk skeleton (square 7x7 seed window,
    junction pixel well inside the borders) the defective pairing settles on
    a truncated window and misclassifies the junction as a branchpoint.
    """
    walk = _walk_module()

    Iskel = np.load(TESTS_ROOT / "data" / "is_bp_window_growth_skeleton.npy")
    idx = np.ravel_multi_index((20, 19), Iskel.shape)

    assert walk.is_bp(idx, Iskel) == 0  # was 1 before the fix

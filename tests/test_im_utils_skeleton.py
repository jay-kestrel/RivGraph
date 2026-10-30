"""Regression tests for im_utils skeleton utilities.

Each test reproduces a confirmed defect from the RivGraph bug hunt
(Bug-fix spec, findings doc art_h9VmNBNs):

- I-H1: skel_pixel_curvature walked down-RIGHT for direction 5, which is
  down-LEFT per the neighbor_idcs layout.
- I-H2: skel_branchpoints end-pixel removal indexed the row twice
  (Ibps[row, row] instead of Ibps[row, col]).
- I-M4: downsample_binary_image cast scaled coordinates to uint16, which
  wraps silently for coordinates >= 65536.
- L2-io: neighbors_flat's numpy-scalar guard re-converted idx instead of
  ncols.
"""
from __future__ import annotations

import numpy as np

from rivgraph import im_utils as iu


def _mirrored_L(direction_five_arm: bool) -> np.ndarray:
    """L skeleton with a right arm (dir 4) plus a diagonal arm.

    direction_five_arm=False -> arm extends up-left (dir 0)
    direction_five_arm=True  -> arm extends down-left (dir 5)
    The two Ls are mirror images through the horizontal axis, so the
    corner curvature must be identical for both.
    """
    L = np.zeros((11, 11))
    L[5, 5] = 1
    for i in range(1, 5):
        L[5, 5 + i] = 1
        if direction_five_arm:
            L[5 + i, 5 - i] = 1
        else:
            L[5 - i, 5 - i] = 1
    return L


def test_skel_pixel_curvature_symmetric_for_dir0_and_dir5_arms():
    """I-H1: dir-0 and dir-5 mirrored Ls must give identical corner angles."""
    curv_dir0 = iu.skel_pixel_curvature(_mirrored_L(False), nwalk=4)
    curv_dir5 = iu.skel_pixel_curvature(_mirrored_L(True), nwalk=4)

    # 45-degree corners; the dir-5 arm previously returned 135.0
    assert np.isclose(curv_dir0[5, 5], 45.0)
    assert np.isclose(curv_dir5[5, 5], 45.0)


def test_skel_branchpoints_clears_end_pixel_at_row_col_not_row_row():
    """I-H2: end-of-run branchpoint removal must clear (row, col).

    The fixture is a 12x12 skeleton whose branchpoint layout includes a
    run whose end pixel (5, 7) has fewer than two external neighbors, so
    the end-pixel removal must clear it. The unfixed code wrote
    Ibps[5, 5] instead, over-retaining the branchpoint at (5, 7).
    """
    Iskel = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    Ibps = iu.skel_branchpoints(Iskel)

    # (5, 7) is an end pixel with < 2 external neighbors: must be cleared
    assert Ibps[5, 7] == 0
    # The buggy code wrote Ibps[5, 5] instead: (5, 5) must stay untouched
    assert Ibps[5, 5] == 0


def test_downsample_binary_image_no_wraparound_above_uint16_range():
    """I-M4: scaled coordinates >= 65536 must not wrap through uint16.

    A source pixel at row 131072 scales to 65536 at ds_factor 0.5, which
    exceeds the uint16 maximum and previously wrapped to row 0.
    """
    I = np.zeros((140000, 2))
    I[131072, 0] = 1
    I[139999, 1] = 1

    out = iu.downsample_binary_image(I, (70000, 2), thresh=0.05)

    assert out[65536, 0] == 1
    assert out[69999, 1] == 1
    # neither source pixel may land on row 0 through wraparound
    assert out[0, 0] == 0
    assert out[0, 1] == 0


def test_neighbors_flat_accepts_numpy_scalar_ncols():
    """L2-io: numpy-scalar ncols must be unwrapped, not idx.

    The previous guard re-converted idx (a no-op or, for a python-int
    idx, an AttributeError); a numpy-scalar ncols propagated instead.
    """
    imflat = np.arange(15)  # flattened 3x5 image
    idx = 7  # interior pixel: row 1, col 2

    result = iu.neighbors_flat(idx, imflat, np.int64(5), filt='all')

    expected = iu.neighbors_flat(idx, imflat, 5, filt='all')
    assert np.array_equal(np.asarray(result), np.asarray(expected))

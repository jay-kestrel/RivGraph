"""Ordered, exact COBOL/reference parity: no topology canonicalization.

The COBOL route keeps every set-order-sensitive decision in Python; the
kernel only marches. These tests hold the route to the same bar as the Rust
walker: exact ordered equality with the reference across the full matrix.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from scipy.ndimage import gaussian_filter

from rivgraph import _cobol, _native, mask_to_graph
from rivgraph.ordered_set import OrderedSet

ROOT = Path(__file__).resolve().parents[1]


def assert_graph_equal(actual, expected):
    for got, want in zip(actual, expected):
        assert got.keys() == want.keys()
        for key in want:
            # OrderedSet equality with another OrderedSet is order-sensitive.
            assert type(got[key]) is type(want[key]), key
            assert got[key] == want[key], key


def _so_built():
    return _cobol.so_path().exists()


# Parity through the kernel needs the built shared object. Skipped quietly in
# environments without gnucobol3 unless COBOL routing is explicitly requested
# (then the FileNotFoundError is the test failure).
requires_so = pytest.mark.skipif(
    not _so_built() and os.environ.get("RIVGRAPH_COBOL") != "1",
    reason="COBOL backend not built; run `make -C cobol` (requires gnucobol3)",
)


@pytest.fixture(scope="module", params=["synthetic", "Colville", "Brahmaputra"])
def skeleton(request):
    if request.param == "synthetic":
        field = gaussian_filter(np.random.RandomState(42).rand(600, 600), 7)
        mask = field > np.percentile(field, 85)
    else:
        directory = {
            "Colville": "Colville_Delta",
            "Brahmaputra": "Brahmaputra_Braided_River",
        }[request.param]
        with rasterio.open(
            ROOT / "examples/data" / directory / f"{request.param}_mask.tif"
        ) as src:
            mask = src.read(1)
    return mask_to_graph.skeletonize_mask(mask)


@requires_so
def test_exact_real_mask_parity(skeleton, monkeypatch):
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    before = skeleton.copy()
    reference = mask_to_graph.skel_to_graph(skeleton)
    monkeypatch.delenv("PYDELTARCM_NATIVE")
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    actual = mask_to_graph.skel_to_graph(skeleton)
    monkeypatch.delenv("RIVGRAPH_COBOL")
    assert_graph_equal(actual, reference)
    assert_graph_equal(mask_to_graph.skel_to_graph(skeleton), actual)
    np.testing.assert_array_equal(skeleton, before)
    assert isinstance(actual[0]["id"], OrderedSet)


@requires_so
@pytest.mark.parametrize(
    "kind", ["line", "diagonal", "cross", "loop", "disconnected", "edge"]
)
def test_small_topologies(kind, monkeypatch):
    mask = np.zeros((31, 31), dtype=np.uint8)
    if kind in {"line", "cross", "disconnected"}:
        mask[15, 3:28] = 1
    if kind == "cross":
        mask[3:28, 15] = 1
    elif kind == "diagonal":
        np.fill_diagonal(mask, 1)
    elif kind == "loop":
        mask[5, 5:25] = mask[24, 5:25] = 1
        mask[5:25, 5] = mask[5:25, 24] = 1
    elif kind == "disconnected":
        mask[3, 3:28] = 1
    elif kind == "edge":
        mask[:, 0] = 1
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@requires_so
@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64, np.float64])
def test_strided_and_readonly_arrays(dtype, monkeypatch):
    mask = np.zeros((40, 40), dtype=dtype)
    mask[20, 2:38] = 1
    mask = mask.T[::-1]
    mask.flags.writeable = False
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@requires_so
@pytest.mark.parametrize("seed", range(30))
def test_seeded_branched_masks(seed, monkeypatch):
    field = gaussian_filter(np.random.RandomState(seed).rand(48, 64), 1.5)
    mask = mask_to_graph.skeletonize_mask(field > np.percentile(field, 65))
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@requires_so
def test_cobol_takes_precedence_over_reference_flag(monkeypatch):
    # Port contract C3: RIVGRAPH_COBOL=1 wins even when the shared
    # PYDELTARCM_NATIVE var requests the Python reference.
    sentinel = ({"id": OrderedSet([0]), "idx": [[1]], "conn": [[-1]],
                 "n_networks": 1}, {"id": OrderedSet([0]), "idx": [1],
                                    "conn": [[0]]})
    monkeypatch.setattr(mask_to_graph, "_skel_to_graph_cobol", lambda m: sentinel)
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    assert mask_to_graph.skel_to_graph(np.zeros((20, 20), dtype=np.uint8)) is sentinel


@requires_so
def test_cobol_does_not_call_native(monkeypatch):
    # Symmetric guard: COBOL routing must not reach the Rust extension.
    def forbidden(*args):
        raise AssertionError("native backend called by COBOL route")

    monkeypatch.setattr(_native, "skel_to_graph", forbidden)
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    mask = np.eye(10, dtype=np.uint8)
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


def test_missing_backend_raises(monkeypatch):
    # AC5: no silent fallback - requesting COBOL without a built backend
    # must raise, not quietly run the Python or Rust route.
    monkeypatch.setattr(_cobol, "_march", None)
    monkeypatch.setattr(_cobol, "so_path", lambda: Path("/nonexistent/lib.so"))
    monkeypatch.setenv("RIVGRAPH_COBOL", "1")
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[10, 3:17] = 1
    with pytest.raises(FileNotFoundError, match="make -C cobol"):
        mask_to_graph.skel_to_graph(mask)


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((0, 0), dtype=np.uint8),
        np.zeros((9, 9), dtype=np.uint8),
        np.ones((1, 1), dtype=np.uint8),
    ],
)
def test_reference_errors(mask, monkeypatch):
    # Error parity: the COBOL route raises the same exception type as the
    # reference on degenerate skeletons (no backend required for these).
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    monkeypatch.delenv("RIVGRAPH_COBOL", raising=False)
    with pytest.raises(Exception) as reference:
        mask_to_graph._skel_to_graph_python(mask)
    with pytest.raises(type(reference.value)):
        mask_to_graph._skel_to_graph_cobol(mask)


@pytest.mark.parametrize(
    "mask", [np.zeros(3), np.full((3, 3), 2), np.full((3, 3), np.nan)]
)
def test_invalid_input(mask):
    with pytest.raises(ValueError, match="two-dimensional binary"):
        mask_to_graph._skel_to_graph_cobol(mask)

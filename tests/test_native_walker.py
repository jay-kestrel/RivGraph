"""Ordered, exact native/reference parity: no topology canonicalization."""

import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from scipy.ndimage import gaussian_filter

from rivgraph import _native, mask_to_graph
from rivgraph.ordered_set import OrderedSet

ROOT = Path(__file__).resolve().parents[1]


def assert_graph_equal(actual, expected):
    for got, want in zip(actual, expected):
        assert got.keys() == want.keys()
        for key in want:
            # OrderedSet equality with another OrderedSet is order-sensitive.
            assert type(got[key]) is type(want[key]), key
            assert got[key] == want[key], key


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


def test_exact_real_mask_parity(skeleton, monkeypatch):
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    before = skeleton.copy()
    reference = mask_to_graph.skel_to_graph(skeleton)
    monkeypatch.delenv("PYDELTARCM_NATIVE")
    actual = mask_to_graph.skel_to_graph(skeleton)
    assert_graph_equal(actual, reference)
    assert_graph_equal(mask_to_graph.skel_to_graph(skeleton), actual)
    np.testing.assert_array_equal(skeleton, before)
    assert isinstance(actual[0]["id"], OrderedSet)


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
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.int64, np.float64])
def test_strided_and_readonly_arrays(dtype, monkeypatch):
    mask = np.zeros((40, 40), dtype=dtype)
    mask[20, 2:38] = 1
    mask = mask.T[::-1]
    mask.flags.writeable = False
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@pytest.mark.parametrize("seed", range(30))
def test_seeded_branched_masks(seed, monkeypatch):
    field = gaussian_filter(np.random.RandomState(seed).rand(48, 64), 1.5)
    mask = mask_to_graph.skeletonize_mask(field > np.percentile(field, 65))
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@pytest.mark.skipif(
    os.environ.get("RIVGRAPH_COBOL") == "1",
    reason="guard protects the native route; the COBOL route intentionally "
    "keeps branchpoint checks (walk.is_bp) in Python",
)
def test_native_does_not_call_python_walker(monkeypatch):
    from rivgraph import walk

    def forbidden(*args):
        raise AssertionError("Python walker called by native backend")

    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    monkeypatch.setattr(mask_to_graph, "_skel_to_graph_python", forbidden)
    monkeypatch.setattr(mask_to_graph, "_skel_to_graph_cobol", forbidden)
    monkeypatch.setattr(walk, "is_bp", forbidden)
    mask = np.zeros((31, 31), dtype=np.uint8)
    mask[15, 3:28] = mask[3:28, 15] = 1
    assert len(mask_to_graph.skel_to_graph(mask)[0]["id"]) == 4


def test_fallback_does_not_call_native(monkeypatch):
    def forbidden(*args):
        raise AssertionError("native backend called in fallback mode")

    monkeypatch.setattr(_native, "skel_to_graph", forbidden)
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    mask = np.eye(10, dtype=np.uint8)
    assert_graph_equal(
        mask_to_graph.skel_to_graph(mask), mask_to_graph._skel_to_graph_python(mask)
    )


@pytest.mark.parametrize(
    "mask",
    [
        np.zeros((0, 0), dtype=np.uint8),
        np.zeros((9, 9), dtype=np.uint8),
        np.ones((1, 1), dtype=np.uint8),
    ],
)
def test_reference_errors(mask, monkeypatch):
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    with pytest.raises(Exception) as reference:
        mask_to_graph._skel_to_graph_python(mask)
    with pytest.raises(type(reference.value)):
        mask_to_graph.skel_to_graph(mask)


@pytest.mark.parametrize(
    "mask", [np.zeros(3), np.full((3, 3), 2), np.full((3, 3), np.nan)]
)
def test_invalid_input(mask, monkeypatch):
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    with pytest.raises(ValueError, match="two-dimensional binary"):
        mask_to_graph.skel_to_graph(mask)

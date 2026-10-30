"""R3 parity and semantics tests: native width ray marching vs Python.

The native path (``rivgraph._native.link_widths_and_lengths``) must reproduce
the Python reference in ``ln_utils.link_widths_and_lengths``:

- per-pixel ``wid_pix`` widths bit-exactly (the ray DDA performs identical
  float operations in identical order),
- per-link width/length statistics to ``rtol <= 1e-9`` (numpy's pairwise
  summation vs left-to-right summation of means is the only sanctioned
  difference),
- the ``np.isclose(step, 0.0)`` epsilon semantics of the zero-step gate
  (|step| <= 1e-8 with numpy's defaults), pinned against the reference.
"""
import copy
import os
from pathlib import Path

import numpy as np
import pytest
import rasterio
from scipy.ndimage import distance_transform_edt, gaussian_filter

from rivgraph import _native, ln_utils, mask_to_graph

ROOT = Path(__file__).resolve().parents[1]

STAT_KEYS = ("len", "wid", "wid_adj", "wid_med", "len_adj", "sinuosity")


def _synthetic_mask():
    """The seed-42 braided synthetic mask used throughout R2/R3."""
    field = gaussian_filter(np.random.RandomState(42).rand(600, 600), 7)
    return field > np.percentile(field, 85)


@pytest.fixture(scope="module", params=["synthetic", "Colville", "Brahmaputra"])
def width_inputs(request):
    """Links plus distance transform for each parity mask.

    Links are built with the native walker (R2: proven exactly equal to the
    Python walker) so the width stage is the only backend under test here.
    """
    if request.param == "synthetic":
        mask = _synthetic_mask()
    else:
        directory = {
            "Colville": "Colville_Delta",
            "Brahmaputra": "Brahmaputra_Braided_River",
        }[request.param]
        with rasterio.open(
            ROOT / "examples/data" / directory / f"{request.param}_mask.tif"
        ) as src:
            mask = src.read(1)

    prior = os.environ.pop("PYDELTARCM_NATIVE", None)
    try:
        links, _ = mask_to_graph.skel_to_graph(mask_to_graph.skeletonize_mask(mask))
    finally:
        if prior is not None:
            os.environ["PYDELTARCM_NATIVE"] = prior
    return links, distance_transform_edt(mask > 0)


def run_both_backends(links, Idt, monkeypatch):
    """Run the width phase on both backends over identical inputs."""
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    reference = ln_utils.link_widths_and_lengths(copy.deepcopy(links), Idt)
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    actual = ln_utils.link_widths_and_lengths(copy.deepcopy(links), Idt)
    return reference, actual


def assert_parity(reference, actual):
    """wid_pix must be bit-equal; statistics within the spec's 1e-9 rtol."""
    assert len(actual["wid_pix"]) == len(reference["wid_pix"])
    for got, want in zip(actual["wid_pix"], reference["wid_pix"]):
        np.testing.assert_array_equal(got, want)

    for key in STAT_KEYS:
        got = np.asarray(actual[key], dtype=float)
        want = np.asarray(reference[key], dtype=float)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=0.0, equal_nan=True)


def test_widths_match_python_reference(width_inputs, monkeypatch):
    links, Idt = width_inputs
    reference, actual = run_both_backends(links, Idt, monkeypatch)
    assert_parity(reference, actual)


def test_native_output_containers(width_inputs, monkeypatch):
    """The native wrapper appends the same list-of-arrays / list-of-scalars."""
    links, Idt = width_inputs
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    out = ln_utils.link_widths_and_lengths(copy.deepcopy(links), Idt)
    assert isinstance(out["wid_pix"], list)
    assert all(isinstance(w, np.ndarray) for w in out["wid_pix"])
    for key in STAT_KEYS:
        assert isinstance(out[key], list)
        assert len(out[key]) == len(out["wid_pix"])


# --- crafted edge cases over a hand-made strip distance transform ----------


def _strip_idt():
    mask = np.zeros((20, 40), dtype=bool)
    mask[10, 5:25] = True  # horizontal strip of channel
    return distance_transform_edt(mask)


def _crafted_links(idt, rows, cols):
    """Links dict with one link per (rows, cols) pixel pair of lists."""
    flat = [
        np.asarray(r, dtype=np.int64) * idt.shape[1] + np.asarray(c, dtype=np.int64)
        for r, c in zip(rows, cols)
    ]
    return {"idx": flat}


CRAFTED_CASES = {
    "single_pixel": ([[10]], [[12]]),
    "two_pixel": ([[10, 10]], [[12, 13]]),
    "diagonal": ([[5, 6, 7, 8]], [[10, 11, 12, 13]]),
    "edge_of_mask": ([[10] * 5], [[35, 36, 37, 38, 39]]),
    "repeated_pixel": ([[10, 10]], [[12, 12]]),
    "multi_link": (
        [[10, 10, 10], [10, 10]],
        [[5, 6, 7], [20, 21]],
    ),
}


@pytest.mark.parametrize("kind", sorted(CRAFTED_CASES))
@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in scalar divide:RuntimeWarning"
)
def test_crafted_edge_cases(kind, monkeypatch):
    # Upstream reference behavior: a 1-pixel link has zero straight-line
    # length, so its sinuosity is a 0/0 divide -> NaN (ln_utils.py's stats
    # section). That warning became an error under master's global
    # filterwarnings=error; ignoring it here only stops the escalation —
    # parity still asserts NaN == NaN via equal_nan=True.
    rows, cols = CRAFTED_CASES[kind]
    idt = _strip_idt()
    links = _crafted_links(idt, rows, cols)
    reference, actual = run_both_backends(links, idt, monkeypatch)
    assert_parity(reference, actual)


def test_empty_links(monkeypatch):
    links = {"idx": []}
    reference, actual = run_both_backends(links, _strip_idt(), monkeypatch)
    assert reference["wid_pix"] == []
    assert actual["wid_pix"] == []
    assert_parity(reference, actual)


def test_noncontiguous_idt_is_made_contiguous(monkeypatch):
    """A strided Idt view is ascontiguousarray'd before crossing the boundary."""
    idt = _strip_idt()
    padded = np.zeros((idt.shape[0] + 3, idt.shape[1] * 2))
    padded[3:, ::2] = idt
    view = padded[3:, ::2]
    assert not view.flags["C_CONTIGUOUS"]

    links = _crafted_links(view, [[10, 10]], [[12, 13]])
    reference, actual = run_both_backends(links, view, monkeypatch)
    np.testing.assert_array_equal(actual["wid_pix"][0], reference["wid_pix"][0])
    for key in STAT_KEYS:
        np.testing.assert_allclose(
            np.asarray(actual[key], dtype=float),
            np.asarray(reference[key], dtype=float),
            rtol=1e-9,
            atol=0.0,
            equal_nan=True,
        )


# --- ray-level exactness and epsilon semantics ------------------------------


@pytest.mark.parametrize(
    "step_r,step_c",
    [
        (0.0, 1.0),
        (1.0, 0.0),
        (-1.0, 0.0),
        (0.0, -1.0),
        (1.0, 1.0),
        (-1.0, 1.0),
        (0.5, -1.5),
        (-0.5, -0.5),
        (1e-9, 1.0),
        (1e-8, 1.0),
        (2e-8, 1.0),
        (-1e-8, 0.5),
        (1e-7, -2.0),
    ],
)
def test_scan_mask_ray_exact(step_r, step_c, monkeypatch):
    """Native ray distances are bit-equal to the Python reference."""
    idt = _strip_idt()
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    py = ln_utils._scan_mask(idt > 0, idt.shape, 10, 15, step_r, step_c)
    native = _native.scan_mask(np.ascontiguousarray(idt), 10.0, 15.0, step_r, step_c)
    assert py == native


@pytest.mark.parametrize(
    "step_r",
    [0.0, -0.0, 5e-324, 1e-300, 1e-9, 1e-8, 1.0000001e-8, 2e-8, 1e-7, 0.5,
     -1e-9, -1e-8, -2e-8, -0.5],
)
def test_scan_mask_epsilon_matches_isclose(step_r, monkeypatch):
    """The isclose(0.0) zero-gate boundary is pinned through the threshold.

    np.isclose(step, 0.0) with defaults reduces to |step| <= 1e-8; the native
    gate must agree with the reference on both sides of that boundary, where
    the ray flips between axis-aligned marching and drifting diagonally.
    """
    idt = _strip_idt()
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    py = ln_utils._scan_mask(idt > 0, idt.shape, 10.0, 15.0, step_r, 1.0)
    native = _native.scan_mask(np.ascontiguousarray(idt), 10.0, 15.0, step_r, 1.0)
    assert py == native


def test_epsilon_boundary_is_live():
    """The parity sweeps cannot pass vacuously: a drifting ray differs."""
    # A 5-pixel-tall block: an axis-aligned ray exits through the right edge
    # while a drifting one leaves through the top edge first.
    mask = np.zeros((20, 40), dtype=bool)
    mask[8:13, 5:35] = True
    idt = distance_transform_edt(mask)
    straight = ln_utils._scan_mask(idt > 0, idt.shape, 10.0, 15.0, 0.0, 1.0)
    drifting = ln_utils._scan_mask(idt > 0, idt.shape, 10.0, 15.0, 0.2, 1.0)
    assert straight != drifting


def test_scan_mask_rejects_zero_steps_on_both_axes(monkeypatch):
    """The reference while-loop never advances on (0, 0) steps; the native
    entry point raises instead of hanging the interpreter uninterruptibly.
    Unreachable from link_widths_and_lengths (the normal has norm >= 1)."""
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    with pytest.raises(ValueError, match="zero steps"):
        _native.scan_mask(np.ascontiguousarray(_strip_idt()), 10.0, 10.0, 0.0, 0.0)


# --- backend wiring ----------------------------------------------------------


def test_native_widths_does_not_call_python_reference(width_inputs, monkeypatch):
    links, Idt = width_inputs
    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)

    def boom(*args, **kwargs):
        raise AssertionError("native path must not call the Python reference")

    monkeypatch.setattr(ln_utils, "_scan_mask", boom)
    out = ln_utils.link_widths_and_lengths(copy.deepcopy(links), Idt)
    assert len(out["wid_pix"]) == len(links["idx"])


def test_fallback_widths_does_not_call_native(width_inputs, monkeypatch):
    links, Idt = width_inputs
    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")

    def boom(*args, **kwargs):
        raise AssertionError("fallback path must not call the native phase")

    monkeypatch.setattr(_native, "link_widths_and_lengths", boom)
    out = ln_utils.link_widths_and_lengths(copy.deepcopy(links), Idt)
    assert len(out["wid_pix"]) == len(links["idx"])

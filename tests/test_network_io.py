"""Targeted tests for light-weight IO helpers."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np

from rivgraph.ordered_set import OrderedSet
from tests._helpers import require_io_utils, require_rivgraph_classes


import pytest


@pytest.mark.parametrize(
    ("kind", "expected_entries"),
    [
        ("binary", {0: (0, 0, 0, 0), 1: (255, 255, 255, 100)}),
        ("mask", {0: (0, 0, 0, 0), 1: (0, 128, 0, 100)}),
        ("tile", {0: (0, 0, 0, 0), 1: (0, 0, 255, 100)}),
        ("GSW", {0: (0, 0, 0, 0), 1: (0, 0, 0, 0), 2: (176, 224, 230, 100)}),
    ],
)
def test_colortable_entries(kind, expected_entries):
    io_utils = require_io_utils()
    color_table = io_utils.colortable(kind)

    for idx, rgba in expected_entries.items():
        assert np.all(color_table.GetColorEntry(idx) == rgba)


def test_create_manual_dir_csv(tmp_path):
    io_utils = require_io_utils()
    csvpath = tmp_path / "csvtest.csv"
    io_utils.create_manual_dir_csv(str(csvpath))
    assert csvpath.is_file()


def test_prep_paths(tmp_path):
    io_utils = require_io_utils()
    resultsfolder = tmp_path
    name = "new"
    basetiff = tmp_path / "dummy_mask.tif"
    paths = io_utils.prepare_paths(resultsfolder, name, str(basetiff))
    assert isinstance(paths, dict)
    assert paths["basepath"] == os.path.normpath(resultsfolder)
    assert paths["maskpath"] == str(basetiff)
    assert paths["Iskel"] == os.path.join(tmp_path, "new_skel.tif")
    assert paths["Idist"] == os.path.join(tmp_path, "new_dist.tif")
    assert paths["network_json"] == os.path.join(tmp_path, "new_network.json")
    assert paths["fixlinks_csv"] == os.path.join(tmp_path, "new_fixlinks.csv")
    assert paths["linkdirs"] == os.path.join(tmp_path, "new_link_directions.tif")
    assert paths["metrics"] == os.path.join(tmp_path, "new_metrics.pkl")
    assert paths["shoreline"] == os.path.join(tmp_path, "new_shoreline.shp")
    assert paths["inlet_nodes"] == os.path.join(tmp_path, "new_inlet_nodes.shp")



# --- Safe network serialization (JSON, not pickle) ---


def _canary_write_sentinel(sentinel_path):
    """Reduction target for the canary pickle; runs only if unpickled."""
    with open(sentinel_path, "w", encoding="utf-8") as f:
        f.write("pickle payload executed")


class _CanaryPayload:
    """Pickle payload whose __reduce__ writes a sentinel file when unpickled."""

    def __init__(self, sentinel_path):
        self.sentinel_path = str(sentinel_path)

    def __reduce__(self):
        return _canary_write_sentinel, (self.sentinel_path,)


def _sample_links_nodes():
    """Representative links/nodes dicts covering the value types networks carry."""
    links = {
        "id": OrderedSet([0, 1, 2]),
        "n_links": 3,
        "us_node": np.array([0, 1, 2], dtype=np.int64),
        "ds_node": np.array([1, 2, 0], dtype=np.int64),
        "lengths": np.array([1.5, 2.25, 0.0], dtype=np.float64),
        "conn": [[0], [1, 2], []],
        "isloop": np.array([False, True, False], dtype=bool),
        "cplan": "braided",
        "wid_adj": None,
        "attrs": {"a": np.array([1.0, 2.0], dtype=np.float32), "b": [1, "x", None]},
        "two_d": np.array([[1, 2], [3, 4]], dtype=np.int16),
        "empty": np.array([], dtype=np.int64),
    }
    nodes = {
        "id": OrderedSet([0, 1, 2]),
        "n_nodes": 3,
        "idx": [OrderedSet([5, 6]), OrderedSet([9]), OrderedSet([12])],
        "conn": [[0], [0, 1], [2]],
        "inlets": [0],
        "outlets": [],
        "coords": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
    }
    return links, nodes


def _assert_network_equal(got, want, msg=""):
    """Structural equality for network dicts containing ndarrays and OrderedSets."""
    if isinstance(want, np.ndarray):
        assert isinstance(got, np.ndarray), msg
        assert got.dtype == want.dtype, msg
        np.testing.assert_array_equal(got, want, err_msg=msg)
    elif isinstance(want, OrderedSet):
        assert isinstance(got, OrderedSet), msg
        assert list(got) == list(want), msg
    elif isinstance(want, dict):
        assert isinstance(got, dict), msg
        assert set(got.keys()) == set(want.keys()), msg
        for key in want:
            _assert_network_equal(got[key], want[key], msg=f"{msg}.{key}")
    elif isinstance(want, (list, tuple)):
        assert isinstance(got, list), msg
        assert len(got) == len(want), msg
        for i, (g, w) in enumerate(zip(got, want)):
            _assert_network_equal(g, w, msg=f"{msg}[{i}]")
    else:
        assert got == want, msg




def test_roundtrip_links_and_nodes(tmp_path):
    """save/load round-trips links and nodes with type fidelity."""
    io_utils = require_io_utils()
    links, nodes = _sample_links_nodes()
    netpath = tmp_path / "test_network.json"

    io_utils.save_links_and_nodes(links, nodes, str(netpath))

    got_links, got_nodes = io_utils.load_links_and_nodes(str(netpath))
    _assert_network_equal(got_links, links)
    _assert_network_equal(got_nodes, nodes)


def _colville_skeleton():
    """Real skeleton built from the committed Colville example mask."""
    import rasterio

    m2g = pytest.importorskip("rivgraph.mask_to_graph")
    root = Path(__file__).resolve().parents[1]
    with rasterio.open(root / "examples" / "data" / "Colville_Delta" / "Colville_mask.tif") as src:
        mask = src.read(1)
    return m2g.skeletonize_mask(mask)


def test_roundtrip_network_from_real_mask(tmp_path, monkeypatch):
    """Round-trip a real network built from the committed Colville mask (fallback backend)."""
    io_utils = require_io_utils()
    m2g = pytest.importorskip("rivgraph.mask_to_graph")
    skeleton = _colville_skeleton()

    monkeypatch.setenv("PYDELTARCM_NATIVE", "0")
    links, nodes = m2g.skel_to_graph(skeleton)

    netpath = tmp_path / "skel_network.json"
    io_utils.save_links_and_nodes(links, nodes, str(netpath))
    got_links, got_nodes = io_utils.load_links_and_nodes(str(netpath))

    _assert_network_equal(got_links, links)
    _assert_network_equal(got_nodes, nodes)


def test_roundtrip_network_from_real_mask_native(tmp_path, monkeypatch):
    """Same round-trip against the native walker when the built extension is available."""
    try:
        from rivgraph import _native  # noqa: F401
    except ImportError:
        pytest.skip("native walker extension not built in this environment")

    io_utils = require_io_utils()
    m2g = pytest.importorskip("rivgraph.mask_to_graph")
    skeleton = _colville_skeleton()

    monkeypatch.delenv("PYDELTARCM_NATIVE", raising=False)
    links, nodes = m2g.skel_to_graph(skeleton)

    netpath = tmp_path / "skel_network.json"
    io_utils.save_links_and_nodes(links, nodes, str(netpath))
    got_links, got_nodes = io_utils.load_links_and_nodes(str(netpath))

    _assert_network_equal(got_links, links)
    _assert_network_equal(got_nodes, nodes)


def test_load_rejects_pickle_canary_without_executing(tmp_path):
    """A crafted pickle must be refused without running its __reduce__ (sentinel absent)."""
    io_utils = require_io_utils()
    sentinel = tmp_path / "sentinel.txt"
    canary_path = tmp_path / "canary_network.pkl"
    with open(canary_path, "wb") as f:
        pickle.dump([{"id": _CanaryPayload(sentinel)}, {}], f)

    with pytest.raises(ValueError, match="pickle"):
        io_utils.load_links_and_nodes(str(canary_path))

    assert not sentinel.exists()


def test_load_rejects_pickle_renamed_to_json(tmp_path):
    """A pickle renamed to .json is refused by content sniffing, not executed."""
    io_utils = require_io_utils()
    sentinel = tmp_path / "sentinel.txt"
    canary_path = tmp_path / "canary_network.json"
    with open(canary_path, "wb") as f:
        pickle.dump([{"id": _CanaryPayload(sentinel)}, {}], f)

    with pytest.raises(ValueError, match="pickle"):
        io_utils.load_links_and_nodes(str(canary_path))

    assert not sentinel.exists()


def test_save_refuses_pkl_suffix(tmp_path):
    """The writer refuses .pkl paths so the code-executing format cannot reappear."""
    io_utils = require_io_utils()
    links, nodes = _sample_links_nodes()

    with pytest.raises(ValueError, match="pickle"):
        io_utils.save_links_and_nodes(links, nodes, str(tmp_path / "network.pkl"))

    assert not (tmp_path / "network.pkl").exists()


def test_load_does_not_shadow_ordered_set_module(tmp_path):
    """Loading must not pollute sys.modules with the bare 'ordered_set' key."""
    io_utils = require_io_utils()
    sys.modules.pop("ordered_set", None)
    links, nodes = _sample_links_nodes()
    netpath = tmp_path / "shadow_network.json"
    io_utils.save_links_and_nodes(links, nodes, str(netpath))

    got_links, _ = io_utils.load_links_and_nodes(str(netpath))

    assert "ordered_set" not in sys.modules
    assert isinstance(got_links["id"], OrderedSet)


def test_save_and_load_network_class_api(tmp_path):
    """save_network/load_network default to paths['network_json'] and round-trip."""
    _, _, rivnetwork = require_rivgraph_classes()
    links, nodes = _sample_links_nodes()

    net = rivnetwork.__new__(rivnetwork)
    net.paths = {"network_json": str(tmp_path / "api_network.json")}
    net.links = links
    net.nodes = nodes
    net.save_network()
    assert (tmp_path / "api_network.json").is_file()

    loader = rivnetwork.__new__(rivnetwork)
    loader.paths = {"network_json": str(tmp_path / "api_network.json")}
    loader.load_network()

    _assert_network_equal(loader.links, links)
    _assert_network_equal(loader.nodes, nodes)

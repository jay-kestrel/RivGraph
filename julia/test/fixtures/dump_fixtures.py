"""Dump Julia-port parity fixtures for the RivGraph walker, widths, and delta kernels.

Everything here runs the PYTHON REFERENCE path (PYDELTARCM_NATIVE=0); the Rust
native extension is not used. Outputs land in fixtures_out/:

  walker/<name>/Iskel.npy          packed-bits skeleton (see manifest shape)
  walker/<name>/expected.json      ordered links/nodes dictionaries
  widths/synthetic/...             mask, Idt (gzipped), links_in, links_out
  widths/colville/...              full bench-chain inputs and outputs
  delta/<name>/...                 adjacency A, expected F/SubN/P, nER
  manifest.json                    versions, hashes, params, error cases

Consumed by julia/test/runtests.jl. Regenerate only with the pinned Python
3.12 environment (set-object iteration order is version-dependent).
"""
import gzip
import hashlib
import json
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
import skimage
import networkx
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "julia" / "test" / "fixtures" / "generated"

os.environ["PYDELTARCM_NATIVE"] = "0"

import rasterio  # noqa: E402
from scipy import ndimage  # noqa: E402

from rivgraph import mask_to_graph  # noqa: E402
from rivgraph import ln_utils  # noqa: E402
from rivgraph.ordered_set import OrderedSet  # noqa: E402
from rivgraph.deltas._delta_metrics_core import solve_adjacency_steady_state  # noqa: E402
from rivgraph.deltas.delta_metrics import nl_entropy_rate  # noqa: E402

MANIFEST = {"generator": "dump_fixtures.py", "inputs": {}, "versions": {
    "python": platform.python_version(),
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "scikit-image": skimage.__version__,
    "networkx": networkx.__version__,
    "rivgraph_path": str(REPO),
    "rivgraph_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
}}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_array(arr, path):
    np.save(path, arr)
    return {"file": path.name, "dtype": str(arr.dtype), "shape": list(arr.shape),
            "sha256": sha256(path), "bytes": path.stat().st_size}


def save_packed_bits(boolimg, path):
    """Pack a 2-D binary image into bits, row-major, with shape in the name."""
    packed = np.packbits(boolimg.astype(bool).ravel())
    np.save(path, packed)
    return {"file": path.name, "packing": "packbits_rowmajor",
            "shape": list(boolimg.shape), "dtype": "bool",
            "sha256": sha256(path), "bytes": path.stat().st_size}


def save_gzip_array(arr, path):
    with open(path, "wb") as f:
        np.save(f, arr, allow_pickle=False)
    gz = Path(str(path) + ".gz")
    # mtime=0 keeps the gzip bytes deterministic, so a regeneration is
    # byte-reproducible and manifest SHA-256s are stable across runs.
    with open(path, "rb") as fin, gzip.GzipFile(gz, "wb", mtime=0) as fout:
        shutil.copyfileobj(fin, fout)
    os.remove(path)
    return {"file": gz.name, "dtype": str(arr.dtype), "shape": list(arr.shape),
            "sha256": sha256(gz), "bytes": gz.stat().st_size, "gzip": True}


def jdump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)
    return {"file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}


def scalar_json(v):
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return v


def ser_graph(d):
    out = {}
    types = {}
    for k, v in d.items():
        types[k] = type(v).__name__
        if isinstance(v, OrderedSet):
            out[k] = [scalar_json(e) for e in v]
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
            types[k] += f":nd{v.ndim}"
        elif isinstance(v, (list, tuple)):
            if len(v) > 0 and isinstance(v[0], (list, tuple, np.ndarray)):
                out[k] = [[scalar_json(e) for e in sub] for sub in v]
            else:
                out[k] = [scalar_json(e) for e in v]
        else:
            out[k] = scalar_json(v)
    return {"values": out, "types": types}


def dump_walker_case(name, iskel, group="walker"):
    d = OUT / group / name
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": name}
    meta["Iskel"] = save_packed_bits(iskel, d / "Iskel.npy")
    links, nodes = mask_to_graph.skel_to_graph(iskel)
    meta["links"] = jdump(ser_graph(links), d / "links.json")
    meta["nodes"] = jdump(ser_graph(nodes), d / "nodes.json")
    meta["counts"] = {"links": len(links["id"]), "nodes": len(nodes["idx"]),
                      "skeleton_pixels": int(np.count_nonzero(iskel))}
    MANIFEST["inputs"][f"{group}/{name}"] = meta
    print(f"  {group}/{name}: {len(links['id'])} links, {len(nodes['idx'])} nodes")
    return links, nodes


def walker_fixtures():
    print("walker fixtures:")

    # 1. Synthetic 600x600 (same construction as the Rust parity test).
    field = gaussian_filter(np.random.RandomState(42).rand(600, 600), 7)
    mask = field > np.percentile(field, 85)
    iskel = mask_to_graph.skeletonize_mask(mask)
    dump_walker_case("synthetic", iskel)

    # 2-3. Real deltas from committed masks.
    for label, folder in [("Colville", "Colville_Delta"),
                          ("Brahmaputra", "Brahmaputra_Braided_River")]:
        with rasterio.open(REPO / "examples/data" / folder / f"{label}_mask.tif") as src:
            m = src.read(1)
        iskel = mask_to_graph.skeletonize_mask(m)
        dump_walker_case(label.lower(), iskel)

    # 4. Small topologies (test_small_topologies masks, used directly).
    for kind in ["line", "diagonal", "cross", "loop", "disconnected", "edge"]:
        m = np.zeros((31, 31), dtype=np.uint8)
        if kind in {"line", "cross", "disconnected"}:
            m[15, 3:28] = 1
        if kind == "cross":
            m[3:28, 15] = 1
        elif kind == "diagonal":
            np.fill_diagonal(m, 1)
        elif kind == "loop":
            m[5, 5:25] = m[24, 5:25] = 1
            m[5:25, 5] = m[5:25, 24] = 1
        elif kind == "disconnected":
            m[3, 3:28] = 1
        elif kind == "edge":
            m[:, 0] = 1
        dump_walker_case(f"small_{kind}", m)

    # 5. Strided/readonly view case: dump the effective values.
    m = np.zeros((40, 40), dtype=np.float64)
    m[20, 2:38] = 1
    dump_walker_case("strided40", np.ascontiguousarray(m.T[::-1]))

    # 6. eye(10) fallback case.
    dump_walker_case("eye10", np.eye(10, dtype=np.uint8))

    # 7. Seeded branched masks, seeds 0..29 (48x64).
    for seed in range(30):
        f = gaussian_filter(np.random.RandomState(seed).rand(48, 64), 1.5)
        m = mask_to_graph.skeletonize_mask(f > np.percentile(f, 65))
        dump_walker_case(f"branched{seed:02d}", m)

    # 8. Isolated-pixel warning case.
    m = np.zeros((31, 31), dtype=np.uint8)
    m[15, 3:28] = 1
    m[3:28, 15] = 1
    m[0, 0] = 1
    dump_walker_case("isolated_pixel", m)

    # Error-contract cases (no arrays; recorded as expected exception types).
    MANIFEST["error_cases"] = {
        "skel_to_graph_invalid": [
            {"name": "1d_input", "input": "np.zeros(3)", "error": "ValueError",
             "message_contains": "two-dimensional binary"},
            {"name": "nonbinary_values", "input": "np.full((3,3), 2)",
             "error": "ValueError", "message_contains": "two-dimensional binary"},
            {"name": "nan_values", "input": "np.full((3,3), np.nan)",
             "error": "ValueError", "message_contains": "two-dimensional binary"},
        ],
        "reference_path": [
            {"name": "empty00", "input": "np.zeros((0,0),uint8)",
             "error": "RuntimeError",
             "message": "Skeleton contains no walkable pixels; cannot resolve graph."},
            {"name": "small99", "input": "np.zeros((9,9),uint8)",
             "error": "RuntimeError",
             "message": "Skeleton contains no walkable pixels; cannot resolve graph."},
            {"name": "ones11", "input": "np.ones((1,1),uint8)", "error": "KeyError",
             "message": "'pop from an empty set'"},
        ],
        "isolated_pixel_warning": {
            "logger": "loguru (rivgraph.mask_to_graph)",
            "message": "Removing 1 isolated skeleton pixels (no neighbors); they cannot form links.",
        },
    }


def widths_fixtures():
    print("widths fixtures:")

    # Synthetic: mask -> Idt; links straight from the walker.
    field = gaussian_filter(np.random.RandomState(42).rand(600, 600), 7)
    mask = field > np.percentile(field, 85)
    iskel = mask_to_graph.skeletonize_mask(mask)
    idt = ndimage.distance_transform_edt(mask)
    links_in, _ = mask_to_graph.skel_to_graph(iskel)
    import copy as _copy
    links_out = ln_utils.link_widths_and_lengths(
        _copy.deepcopy(links_in), idt, pixlen=1.0)

    d = OUT / "widths" / "synthetic"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": "synthetic", "pixlen": 1.0}
    meta["Idt"] = save_gzip_array(idt.astype(np.float64), d / "Idt.npy")
    meta["links_in"] = jdump(ser_graph(links_in), d / "links_in.json")
    meta["links_out"] = jdump(ser_graph(links_out), d / "links_out.json")
    MANIFEST["inputs"]["widths/synthetic"] = meta
    print("  widths/synthetic done")

    # Colville: full bench chain (delta class), as in bench/conftest.py.
    from rivgraph.classes import delta

    colville_dir = REPO / "examples/data/Colville_Delta"
    with tempfile.TemporaryDirectory() as td:
        dv = delta("Colville", str(colville_dir / "Colville_mask.tif"),
                   results_folder=td, verbose=False)
        dv.skeletonize()
        dv.compute_network()
        dv.prune_network(
            path_shoreline=str(colville_dir / "Colville_shoreline.shp"),
            path_inletnodes=str(colville_dir / "Colville_inlet_nodes.shp"),
        )
        dv.compute_distance_transform()
        links_in = _copy.deepcopy(dv.links)
        idt = dv.Idist
        pixlen = float(dv.pixlen)
        links_out = ln_utils.link_widths_and_lengths(
            _copy.deepcopy(links_in), idt, pixlen=pixlen)

    d = OUT / "widths" / "colville"
    d.mkdir(parents=True, exist_ok=True)
    meta = {"name": "colville", "pixlen": pixlen}
    meta["Idt"] = save_gzip_array(np.asarray(idt, dtype=np.float64), d / "Idt.npy")
    meta["links_in"] = jdump(ser_graph(links_in), d / "links_in.json")
    meta["links_out"] = jdump(ser_graph(links_out), d / "links_out.json")
    MANIFEST["inputs"]["widths/colville"] = meta
    print(f"  widths/colville done (pixlen={pixlen})")


def random_dag(seed, n, p_edge, chain_like=False, sink_fraction=0.0):
    """A[v, u] = edge u->v (row=downstream, col=upstream); node 0 is the apex.

    With sink_fraction > 0, a deterministic pseudo-random subset of interior
    nodes loses its chain backbone edge, creating additional outlets so the
    entropy-rate port is exercised on multi-outlet deltas.
    """
    rs = np.random.RandomState(seed)
    A = np.zeros((n, n), dtype=np.float64)
    for u in range(n - 1):
        vs = range(u + 1, n)
        for v in vs:
            if chain_like:
                hit = (v == u + 1)
            else:
                # Chain backbone guarantees a single apex (zero in-degree);
                # extra forward edges add merging downstream pathways.
                hit = (v == u + 1) or (rs.rand() < p_edge)
            if hit:
                A[v, u] = rs.uniform(0.5, 2.0)
    if sink_fraction > 0:
        for u in range(1, n - 1):
            if rs.rand() < sink_fraction:
                A[u + 1:, u] = 0.0
        # Repair the single-apex invariant: zeroing u's outgoing edges can
        # strip a downstream node's only in-edge, turning it into a second
        # apex. Restore a backbone in-edge from v-1 for any such node.
        for v in range(1, n):
            if A[v, :].sum() == 0.0:
                A[v, v - 1] = 1.0
    return A


def delta_fixtures():
    print("delta fixtures:")
    cases = {
        "chain": random_dag(0, 12, 0.0, chain_like=True),
        "fork": None,          # binary tree below
        "dag_small": random_dag(1, 24, 0.06),
        "dag_medium": random_dag(2, 60, 0.045),
        "dag_large": random_dag(3, 120, 0.035),
        "dag_bushy": random_dag(4, 80, 0.08),
        "dag_multioutlet_a": random_dag(5, 48, 0.05, sink_fraction=0.10),
        "dag_multioutlet_b": random_dag(6, 72, 0.04, sink_fraction=0.15),
    }
    # Binary-tree DAG: each node feeds two children (last level trimmed).
    n = 31
    A = np.zeros((n, n))
    rs = np.random.RandomState(7)
    for u in range(15):
        for k in (1, 2):
            v = 2 * u + k
            if v < n:
                A[v, u] = rs.uniform(0.5, 2.0)
    cases["fork"] = A

    for name, A in cases.items():
        res = solve_adjacency_steady_state(A, atol=10.0 ** -10)
        F = np.squeeze(res.node_flux)
        SubN = res.subnetwork_membership
        P = res.transition_matrix
        nER = float(nl_entropy_rate(A))

        d = OUT / "delta" / name
        d.mkdir(parents=True, exist_ok=True)
        meta = {"name": name, "epsilon": 10.0 ** -10,
                "n": int(A.shape[0]), "apex": int(res.apex),
                "outlets": [int(o) for o in res.outlets.tolist()],
                "nER": nER}
        meta["A"] = save_gzip_array(A, d / "A.npy")
        meta["expected"] = jdump({
            "F": F.tolist(), "SubN": SubN.tolist(), "P": P.tolist(),
            "node_flux_sha_bits": [struct.pack("<d", x).hex() for x in F.tolist()],
        }, d / "expected.json")
        MANIFEST["inputs"][f"delta/{name}"] = meta
        print(f"  delta/{name}: n={A.shape[0]} outlets={len(res.outlets)} nER={nER:.6f}")


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    walker_fixtures()
    widths_fixtures()
    delta_fixtures()
    with open(OUT / "manifest.json", "w") as f:
        json.dump(MANIFEST, f, indent=1)
    total = sum(meta.get("bytes", 0) for g in [MANIFEST["inputs"]] for meta in g.values()
                for meta in (meta if isinstance(meta, dict) else []) if isinstance(meta, dict))
    print("total fixture bytes (rough):", total)
    print("manifest written")


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "walker"):
        walker_fixtures()
    if which in ("all", "widths"):
        widths_fixtures()
    if which in ("all", "delta"):
        delta_fixtures()
    if which == "all":
        with open(OUT / "manifest.json", "w") as f:
            json.dump(MANIFEST, f, indent=1)
        print("manifest written")

"""Fixtures and metadata reporting for the RivGraph benchmark suite.

Every benchmark pins its workload to the committed Colville inputs
(``examples/data/Colville_Delta``: mask plus shoreline and inlet-node
shapefiles). The fixture chain mirrors the documented delta workflow
(``delta.skeletonize() -> compute_network() -> prune_network() ->
compute_link_width_and_length()``), each step run once per session so that the
timed regions measure only the benchmarked stage.

Masks are read via rasterio, as in the package itself (``io_utils`` /
``rasters``).

Per-round state pinning (verified empirically before writing this suite):

- ``skel_to_graph`` pads a copy of its input internally and never mutates it,
  so rounds can share one skeleton array.
- ``link_widths_and_lengths`` receives a fresh deep copy of the pruned links
  in each round's untimed setup, so rounds never observe each other's output
  keys.
- ``assign_flow_directions`` resets and recomputes all directionality state
  from the link/node geometry; each round's untimed setup restores the
  pre-directionality snapshot, so rounds start from identical state.
"""
from __future__ import annotations

import copy
import hashlib
import os
import platform
from pathlib import Path

import numpy as np
import pytest

try:
    # Check the hook-registering submodule, not the package shell: a leftover
    # uninstall directory can still import as an empty namespace package.
    import pytest_benchmark.plugin  # noqa: F401

    _HAVE_PYTEST_BENCHMARK = True
except ImportError:
    # The main test suite installs no pytest-benchmark; defining plugin hooks
    # unconditionally would crash collection there with an unknown-hook error.
    _HAVE_PYTEST_BENCHMARK = False

REPO_ROOT = Path(__file__).resolve().parents[1]
COLVILLE_DIR = REPO_ROOT / "examples" / "data" / "Colville_Delta"

# Populated by the session fixtures below, emitted in the terminal summary and
# embedded in the ``--benchmark-json`` artifact.
_ENV_INFO = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "cpus": os.cpu_count(),
    "packages": {},
    "inputs": {},
    "workload": {},
}


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 digest of a file, streamed in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_versions() -> dict:
    """Versions of the packages whose performance this suite measures."""
    import importlib.metadata as md

    versions = {}
    for name in [
        "rivgraph",
        "numpy",
        "scipy",
        "networkx",
        "scikit-image",
        "rasterio",
        "pytest",
        "pytest-benchmark",
    ]:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


@pytest.fixture(scope="session")
def colville_inputs() -> dict:
    """Paths and checksums of the committed Colville benchmark inputs."""
    inputs = {
        "mask": COLVILLE_DIR / "Colville_mask.tif",
        "shoreline": COLVILLE_DIR / "Colville_shoreline.shp",
        "inlet_nodes": COLVILLE_DIR / "Colville_inlet_nodes.shp",
    }
    for name, path in inputs.items():
        if not path.is_file():
            pytest.fail(f"Benchmark input missing: {path}")
    _ENV_INFO["packages"] = _package_versions()
    for name, path in inputs.items():
        _ENV_INFO["inputs"][name] = {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
    return inputs


@pytest.fixture(scope="session")
def colville_delta(tmp_path_factory, colville_inputs):
    """Colville delta prepared through skeletonize, network, and prune.

    Widths are intentionally *not* computed here: the widths benchmark needs
    the pre-width links state, and the directionality stage performs the
    ``compute_link_width_and_length`` call itself (its body is a single
    ``link_widths_and_lengths`` invocation).
    """
    from rivgraph.classes import delta

    results_dir = tmp_path_factory.mktemp("bench_colville")
    d = delta(
        "Colville",
        str(colville_inputs["mask"]),
        results_folder=str(results_dir),
        verbose=False,
    )
    d.skeletonize()
    d.compute_network()
    d.prune_network(
        path_shoreline=str(colville_inputs["shoreline"]),
        path_inletnodes=str(colville_inputs["inlet_nodes"]),
    )
    d.compute_distance_transform()

    _ENV_INFO["workload"] = {
        "skeleton_pixels": int(np.count_nonzero(d.Iskel)),
        "post_prune_links": len(d.links["id"]),
        "post_prune_nodes": len(d.nodes["id"]),
        "inlets": len(d.nodes["inlets"]),
        "outlets": len(d.nodes["outlets"]),
    }
    return d


@pytest.fixture(scope="session")
def width_stage_inputs(colville_delta):
    """Inputs for the widths stage: (links snapshot, distance transform, pixlen)."""
    d = colville_delta
    return copy.deepcopy(d.links), d.Idist, d.pixlen


@pytest.fixture(scope="session")
def directionality_stage(colville_delta, width_stage_inputs):
    """Colville delta with widths attached, plus pre-directionality snapshots.

    Reproduces ``compute_link_width_and_length``'s exact call, then deep-copies
    the resulting links/nodes so every directionality round can restart from
    the identical state.
    """
    from rivgraph import ln_utils

    d = colville_delta
    links_base, idt, pixlen = width_stage_inputs
    d.links = ln_utils.link_widths_and_lengths(
        copy.deepcopy(links_base), idt, pixlen=pixlen
    )
    links_snapshot = copy.deepcopy(d.links)
    nodes_snapshot = copy.deepcopy(d.nodes)
    return d, links_snapshot, nodes_snapshot


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print versions, input checksums, and workload descriptors after the run."""
    if not _ENV_INFO["packages"]:
        return
    terminalreporter.section("RivGraph benchmark environment", sep="=", blue=True)
    terminalreporter.write_line(f"python    : {_ENV_INFO['python']}")
    terminalreporter.write_line(f"platform  : {_ENV_INFO['platform']} ({_ENV_INFO['machine']}, {os.cpu_count()} cpus)")
    terminalreporter.write_line("packages  :")
    for name, version in _ENV_INFO["packages"].items():
        terminalreporter.write_line(f"  {name:16s} {version}")
    terminalreporter.write_line("inputs (sha256) :")
    for name, meta in _ENV_INFO["inputs"].items():
        terminalreporter.write_line(f"  {name:12s} {meta['sha256']}  {meta['path']} ({meta['bytes']} bytes)")
    terminalreporter.write_line("workload  :")
    for key, value in _ENV_INFO.get("workload", {}).items():
        terminalreporter.write_line(f"  {key:20s} {value}")


if _HAVE_PYTEST_BENCHMARK:

    def pytest_benchmark_update_json(config, benchmarks, output_json):
        """Embed environment, checksum, and workload metadata in the JSON artifact."""
        if _ENV_INFO["packages"]:
            output_json["rivgraph_bench_env"] = _ENV_INFO

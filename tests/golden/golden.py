"""Golden-oracle core for the RivGraph PF-1 fixture and the CHK-6 gate.

The spec (shared port-fidelity and oracle spec, PF-1 RivGraph row) pins one
golden fixture per repo: a frozen mask/geometry input set plus expected graph
outputs. RivGraph's graph-extraction core is RNG-free, so the fixture is fully
deterministic given the pinned commit, inputs, and environment.

This module holds the pure pieces both the mint script and the pytest gate
use:

- :func:`run_reference_pipeline` -- runs the Python reference pipeline
  (``PYDELTARCM_NATIVE=0``) on the frozen inputs and returns the graph dicts.
- :func:`extract_fixture` / :func:`serialize_fixture` -- turn the graph into
  byte-stable JSON (floats as shortest-round-trip decimal strings, like the
  spec's RNG-vector format, so no JSON float parsing sits between reference
  and candidate).
- :func:`compare_graphs` -- the CHK-6 comparator: node/link id sets,
  connectivity, and topology exact (Tier A); lengths, widths, and other float
  attributes within 1e-9 relative with a 1e-12 absolute floor (Tier B).

Fixture contract details live in ``tests/golden/README.md``.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
MASK_PATH = REPO_ROOT / "tests/regression/data/river_brahma_clipped/inputs/mask.tif"
FIXLINKS_PATH = REPO_ROOT / "tests/regression/data/river_brahma_clipped/inputs/fixlinks.csv"

# The pipeline parameters below are frozen in the mint commit (spec PF-1:
# mint parameters are chosen once and never silently changed). exit_sides
# matches the river_brahma_clipped regression case; the fixlinks CSV is
# header-only but is wired through to mirror the vetted regression workflow.
EXIT_SIDES = "ns"
PIPELINE_STEPS = (
    "compute_network",
    "prune_network",
    "compute_link_width_and_length",
    "assign_flow_directions",
)

# Tier A: exact. Tier B: PF-3 defaults (rtol 1e-9, atol 1e-12).
TIER_B_FIELDS = frozenset(
    {"len", "len_adj", "sinuosity", "wid", "wid_adj", "wid_med",
     "wid_pctdiff", "clangs", "cldists", "maxang"}
)
RTOL = 1e-9
ATOL = 1e-12

# Package versions pinned in the fixture header; a change here shifts the
# fixture digest and forces an explicit re-mint (spec PF-1 discipline).
PINNED_PACKAGES = (
    "numpy", "scipy", "networkx", "scikit-image", "shapely", "geopandas",
    "rasterio", "pyproj", "pandas", "pyogrio", "affine",
)


def run_reference_pipeline(
    mask_path: Path = MASK_PATH,
    fixlinks_path: Path = FIXLINKS_PATH,
    results_folder: Path | None = None,
) -> tuple[dict, dict]:
    """Run the Python reference pipeline on the frozen inputs.

    Forces ``PYDELTARCM_NATIVE=0`` and clears ``RIVGRAPH_COBOL`` for the
    duration of the run so the run is the pure-Python reference regardless
    of the ambient environment -- the fixture binds to the reference
    implementation, not to a backend. The prior environment is restored
    afterwards so callers cannot leak backend selection into other tests.
    Returns the ``(links, nodes)`` graph dicts after the full pipeline.
    """
    import contextlib
    import os

    overridden = {
        name: os.environ.get(name)
        for name in ("PYDELTARCM_NATIVE", "RIVGRAPH_COBOL")
    }
    try:
        os.environ["PYDELTARCM_NATIVE"] = "0"
        os.environ.pop("RIVGRAPH_COBOL", None)

        from rivgraph.classes import river

        r = river(
            "brahma_golden",
            str(mask_path),
            results_folder=str(results_folder) if results_folder else None,
            exit_sides=EXIT_SIDES,
            verbose=False,
        )
        r.paths["fixlinks_csv"] = str(fixlinks_path)
        for step in PIPELINE_STEPS:
            getattr(r, step)()
    finally:
        for name, prior in overridden.items():
            if prior is None:
                with contextlib.suppress(KeyError):
                    del os.environ[name]
            else:
                os.environ[name] = prior
    return r.links, r.nodes


def _float_strings(values: Any) -> list[str]:
    """Shortest-round-trip decimal strings; 'nan'/'inf' round-trip via float()."""
    return [repr(float(v)) for v in values]


def extract_fixture(links: dict, nodes: dict) -> dict[str, Any]:
    """Extract the CHK-6 graph fixture from pipeline output.

    Tier A fields (ids, connectivity, topology, directionality decisions) are
    stored natively. Tier B measurement floats are stored as decimal strings.
    ``links['idx']`` (per-link pixel chains) and ``links['wid_pix']`` (per-pixel
    widths) are excluded: CHK-6 pins graph structure and link/node attributes,
    not pixel geometry, and excluding them keeps the fixture committable.
    """
    return {
        "links": {
            "id": list(links["id"]),
            "conn": [list(pair) for pair in links["conn"]],
            "link_conn": [list(neighbors) for neighbors in links["link_conn"]],
            "guess": [list(pair) for pair in links["guess"]],
            "guess_alg": [list(pair) for pair in links["guess_alg"]],
            "certain": _float_strings(links["certain"]),
            "certain_alg": _float_strings(links["certain_alg"]),
            "certain_order": _float_strings(links["certain_order"]),
            "parallels": [list(pair) for pair in links["parallels"]],
            "n_networks": int(links["n_networks"]),
            "len": _float_strings(links["len"]),
            "len_adj": _float_strings(links["len_adj"]),
            "sinuosity": _float_strings(links["sinuosity"]),
            "wid": _float_strings(links["wid"]),
            "wid_adj": _float_strings(links["wid_adj"]),
            "wid_med": _float_strings(links["wid_med"]),
            "wid_pctdiff": _float_strings(links["wid_pctdiff"]),
            "clangs": _float_strings(links["clangs"]),
            "cldists": _float_strings(links["cldists"]),
            "maxang": _float_strings(links["maxang"]),
            "_ids_finalization_method": links["_ids_finalization_method"],
            "_ids_finalized": bool(links["_ids_finalized"]),
        },
        "nodes": {
            "id": list(nodes["id"]),
            "conn": [list(neighbors) for neighbors in nodes["conn"]],
            "idx": [int(i) for i in nodes["idx"]],
            "inlets": list(nodes["inlets"]),
            "outlets": list(nodes["outlets"]),
            "_ids_finalization_method": nodes["_ids_finalization_method"],
            "_ids_finalized": bool(nodes["_ids_finalized"]),
        },
    }


def capture_environment() -> dict[str, str]:
    """Versions of the pinned stack plus the interpreter, for the fixture header."""
    env = {"python": platform.python_version()}
    for package in PINNED_PACKAGES:
        try:
            env[package] = version(package)
        except PackageNotFoundError:
            env[package] = "not-installed"
    return env


def build_fixture(graph: dict[str, Any], reference_commit: str, env: dict[str, str]) -> dict[str, Any]:
    """Assemble the full fixture document: provenance header plus graph."""
    links, nodes = graph["links"], graph["nodes"]
    fixture = extract_fixture(links, nodes)
    return {
        "format": "rivgraph-golden-fixture/1",
        "spec_ref": "PF-1 RivGraph row; gate CHK-6; tolerances PF-3",
        "reference_commit": reference_commit,
        "dataset": {
            "name": "river_brahma_clipped",
            "kind": "river",
            "mask_path": "tests/regression/data/river_brahma_clipped/inputs/mask.tif",
            "mask_sha256": _sha256_file(MASK_PATH),
            "fixlinks_path": "tests/regression/data/river_brahma_clipped/inputs/fixlinks.csv",
            "fixlinks_sha256": _sha256_file(FIXLINKS_PATH),
            "exit_sides": EXIT_SIDES,
        },
        "backend": {
            "PYDELTARCM_NATIVE": "0",
            "RIVGRAPH_COBOL": None,
            "note": "pure-Python reference (skel_to_graph and link widths), forced by run_reference_pipeline",
        },
        "pipeline_steps": list(PIPELINE_STEPS),
        "environment": env,
        "counts": {
            "n_links": len(links["id"]),
            "n_nodes": len(nodes["id"]),
            "n_inlets": len(nodes["inlets"]),
            "n_outlets": len(nodes["outlets"]),
            "n_parallels": len(links["parallels"]),
        },
        "excluded_fields": {
            "links.idx": "per-link pixel chains; CHK-6 pins structure/attributes, not pixel geometry (fixture size)",
            "links.wid_pix": "per-pixel width diagnostics; aggregates pinned via wid/wid_adj/wid_med (fixture size)",
        },
        "graph": fixture,
    }


def serialize_fixture(fixture: dict[str, Any]) -> bytes:
    """Byte-stable serialization: sorted keys, fixed indent, trailing newline."""
    text = json.dumps(fixture, indent=1, sort_keys=True, ensure_ascii=True)
    return (text + "\n").encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_floats(strings: list[str]) -> list[float]:
    return [float(s) for s in strings]


def _first_tier_a_mismatch(
    field: str, expected: list[Any], actual: list[Any],
) -> str | None:
    if len(expected) != len(actual):
        return (
            f"{field}: length mismatch (expected {len(expected)}, "
            f"got {len(actual)})"
        )
    for i, (exp, act) in enumerate(zip(expected, actual)):
        if exp != act:
            return f"{field}[{i}]: expected {exp!r}, got {act!r}"
    return None


def _first_tier_b_mismatch(
    field: str, expected: list[str], actual: list[float],
) -> str | None:
    """First element outside |a-b| <= max(ATOL, RTOL * max(|a|, |b|)).

    NaN == NaN is an exact match; NaN vs number is a mismatch (the sign of a
    divergence, not a tolerance question).
    """
    if len(expected) != len(actual):
        return (
            f"{field}: length mismatch (expected {len(expected)}, "
            f"got {len(actual)})"
        )
    for i, (exp_s, act) in enumerate(zip(expected, actual)):
        exp = float(exp_s)
        if math.isnan(exp) and math.isnan(act):
            continue
        if math.isnan(exp) != math.isnan(act):
            return f"{field}[{i}]: NaN mismatch (expected {exp!r}, got {act!r})"
        diff = abs(exp - act)
        tolerance = max(ATOL, RTOL * max(abs(exp), abs(act)))
        if diff > tolerance:
            return (
                f"{field}[{i}]: |{exp!r} - {act!r}| = {diff:.3e} "
                f"exceeds PF-3 bar {tolerance:.3e}"
            )
    return None


def compare_graphs(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    """CHK-6 comparator: returns violations; empty list means the gate passes.

    Tier A (exact): id sets and order, connectivity (``conn``), topology
    (``link_conn`` / node ``conn``), inlets/outlets, directionality decisions
    (``guess*``, ``certain*``), parallels, and finalization metadata.
    Tier B (PF-3): lengths, widths, sinuosity, junction angles, max angle.
    """
    violations: list[str] = []
    exp_links, act_links = expected["links"], actual["links"]
    exp_nodes, act_nodes = expected["nodes"], actual["nodes"]

    for field, exp in exp_links.items():
        act = act_links.get(field)
        if act is None:
            violations.append(f"links.{field}: missing from rebuilt graph")
            continue
        if field in TIER_B_FIELDS:
            mismatch = _first_tier_b_mismatch(f"links.{field}", exp, _parse_floats(act))
        elif isinstance(exp, list):
            mismatch = _first_tier_a_mismatch(f"links.{field}", exp, act)
        else:
            mismatch = None if exp == act else f"links.{field}: expected {exp!r}, got {act!r}"
        if mismatch:
            violations.append(mismatch)

    for field, exp in exp_nodes.items():
        act = act_nodes.get(field)
        if act is None:
            violations.append(f"nodes.{field}: missing from rebuilt graph")
            continue
        if isinstance(exp, list):
            mismatch = _first_tier_a_mismatch(f"nodes.{field}", exp, act)
        else:
            mismatch = None if exp == act else f"nodes.{field}: expected {exp!r}, got {act!r}"
        if mismatch:
            violations.append(mismatch)

    return violations

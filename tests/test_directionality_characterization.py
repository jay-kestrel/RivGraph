"""Frozen directionality outputs captured before R4 from origin/master.

The Colville case uses the same preparation as the R1 benchmark. Synthetic
cases pin bridge guesses, minimum-flip tie breaking, manual constraints and
all orientation-dependent link fields, not merely DAG existence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.conftest import (
    colville_inputs as colville_inputs,
    colville_delta as colville_delta,
    width_stage_inputs as width_stage_inputs,
    directionality_stage as directionality_stage,
)
from rivgraph import directionality as di

GOLDEN = Path(__file__).parent / "data" / "directionality.json"


def normalize(value):
    if isinstance(value, np.ndarray):
        return normalize(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    return value


def direction_state(links, nodes):
    keys = ("id", "conn", "idx", "certain", "certain_alg", "certain_order",
            "guess", "guess_alg")
    return normalize({"links": {k: list(links[k]) for k in keys},
                      "nodes": {k: list(nodes[k]) for k in ("id", "conn", "inlets", "outlets")}})


def synthetic_delta(size=6, manual=()):
    edges = [[0, 1]] + [[i, i + 1] for i in range(1, size)]
    edges += [[size, 1], [size, size + 1]]
    ids = list(range(len(edges)))
    nodes = {"id": list(range(size + 2)), "inlets": [0], "outlets": [size + 1]}
    nodes["conn"] = [[i for i, edge in enumerate(edges) if n in edge] for n in nodes["id"]]
    links = {"id": ids, "conn": edges, "idx": [edge[:] for edge in edges],
             "len": [1.0] * len(edges)}
    links, nodes = di.add_directionality_trackers(links, nodes, "delta")
    links["certain"][:] = 1
    for i in manual:
        links["certain_alg"][i] = di.algmap("manual_set")
    return links, nodes


def synthetic_state(manual=()):
    links, nodes = synthetic_delta(manual=manual)
    links, nodes = di.dir_bridges(links, nodes)
    links, nodes, remaining = di.fix_cycles(links, nodes)
    return {"state": direction_state(links, nodes), "remaining": remaining}


@pytest.mark.parametrize("manual", [(), (6,), (1, 2, 3, 4, 5, 6)])
def test_synthetic_delta_characterization(manual):
    expected = json.loads(GOLDEN.read_text())
    assert synthetic_state(manual) == expected["synthetic"][str(list(manual))]


def test_colville_directionality_characterization(directionality_stage):
    d, _, _ = directionality_stage
    d.assign_flow_directions()
    expected = json.loads(GOLDEN.read_text())
    assert direction_state(d.links, d.nodes) == expected["colville"]

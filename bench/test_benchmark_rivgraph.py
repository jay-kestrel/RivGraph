"""Benchmarks for RivGraph's three measured hot stages (spec deliverable R1).

Stages and workloads, all pinned to the committed Colville inputs
(``examples/data/Colville_Delta``):

1. ``skel_to_graph`` — the skeleton->network walker (``walk.py`` plus its
   helpers) on the full Colville skeleton (~24k pixels).
2. ``link_widths_and_lengths`` — the ``scan_mask`` ray-marcher over every
   pruned link, with the pipeline's own ``pixlen``.
3. ``assign_flow_directions`` — end-to-end directionality on the pruned
   Colville network (Dijkstra shortest paths, cycle fixing, manual fixes
   from the committed csv path when present).

Repeats are honest because each round starts from identical state: stages 1
and 2 are pure functions over fixed arrays (``skel_to_graph`` pads a copy of
its input internally), and stage 3's untimed setup restores the
pre-directionality links/nodes snapshots. See ``bench/conftest.py`` for the
fixture chain and the workload descriptors.
"""
from __future__ import annotations

import copy

import pytest

from rivgraph import ln_utils, mask_to_graph

pytest.importorskip("pytest_benchmark.plugin")

# >= 5 timed rounds with median and min reported, per the harness contract.
BENCH_ROUNDS = 5
BENCH_WARMUP_ROUNDS = 1


def test_benchmark_skel_to_graph(benchmark, colville_delta):
    """Skeleton->network walker on the full Colville skeleton."""
    benchmark.pedantic(
        mask_to_graph.skel_to_graph,
        args=(colville_delta.Iskel,),
        rounds=BENCH_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )


def test_benchmark_link_widths_and_lengths(benchmark, width_stage_inputs):
    """Width/length ray marching over every pruned Colville link."""
    links_base, idt, pixlen = width_stage_inputs

    def make_args():
        # Fresh links dict per round, built in the untimed setup so rounds
        # never observe each other's appended width/length keys.
        return (copy.deepcopy(links_base), idt), {"pixlen": pixlen}

    benchmark.pedantic(
        ln_utils.link_widths_and_lengths,
        setup=make_args,
        rounds=BENCH_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )


def test_benchmark_assign_flow_directions(benchmark, directionality_stage):
    """End-to-end flow-direction assignment on the pruned Colville network."""
    d, links_snapshot, nodes_snapshot = directionality_stage

    def make_args():
        # Untimed reset: every round re-derives directions from the
        # pre-directionality state instead of run N-1's output.
        d.links = copy.deepcopy(links_snapshot)
        d.nodes = copy.deepcopy(nodes_snapshot)
        return (), {}

    benchmark.pedantic(
        d.assign_flow_directions,
        setup=make_args,
        rounds=BENCH_ROUNDS,
        warmup_rounds=BENCH_WARMUP_ROUNDS,
    )

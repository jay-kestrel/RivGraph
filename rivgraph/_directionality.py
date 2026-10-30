"""Pure graph queries for directionality; no link state is mutated here.

Modified derivative of RivGraph directionality.py.
Copyright 2019 Triad National Security, LLC. All rights reserved.
Produced under U.S. Government contract 89233218CNA000001 for LANL.
See ../LICENSE.txt for the complete LANL notice and BSD-3 terms.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence

import networkx as nx


def bridge_sides(
    graph: nx.Graph, inlets: set[int], outlets: set[int],
) -> Iterator[tuple[int, int, bool, bool, bool, bool]]:
    """Yield bridges in graph edge order and terminal presence on either side.

    Split each undirected edge with a private vertex. That vertex dominates
    an endpoint exactly when deleting the edge disconnects the component.
    Subtree terminal counts then answer every side query without deleting
    edges or repeating reachability searches. A super-root attaches to one
    node per component, so disconnected inputs retain component-local totals.
    """
    root = object()
    directed = nx.DiGraph()
    directed.add_node(root)
    components = {}
    totals = {}
    for component in nx.connected_components(graph):
        anchor = next(iter(component))
        directed.add_edge(root, anchor)
        totals[anchor] = (len(component & set(inlets)), len(component & set(outlets)))
        components.update(dict.fromkeys(component, anchor))
    split_edges = []
    for u, v in graph.edges:
        split = object()
        directed.add_edges_from(((u, split), (split, u), (v, split), (split, v)))
        split_edges.append((u, v, split))
    parents = nx.immediate_dominators(directed, root)
    children = {node: [] for node in parents}
    for node, parent in parents.items():
        if node is not root:
            children[parent].append(node)
    counts = {node: [int(node in inlets), int(node in outlets)] for node in parents}
    order = [root]
    for node in order:
        order.extend(children[node])
    for node in reversed(order[1:]):
        parent = parents[node]
        counts[parent][0] += counts[node][0]
        counts[parent][1] += counts[node][1]
    for u, v, split in split_edges:
        if parents.get(u) is split:
            child = u
        elif parents.get(v) is split:
            child = v
        else:
            continue
        up, down = counts[child]
        total_up, total_down = totals[components[child]]
        child_side = (up > 0, down > 0)
        other_side = (total_up > up, total_down > down)
        yield (u, v, *(child_side + other_side if child == u else other_side + child_side))


def reachable(graph: nx.DiGraph, starts: Sequence[int], reverse: bool = False) -> set[int]:
    """Multi-source reachability, including the sources themselves."""
    seen = set(starts)
    stack = list(seen)
    neighbors = graph.predecessors if reverse else graph.successors
    while stack:
        for node in neighbors(stack.pop()):
            if node not in seen:
                seen.add(node)
                stack.append(node)
    return seen


def valid_orientation(
    graph: nx.DiGraph, boundary: set[int], inlets: Sequence[int], outlets: Sequence[int],
) -> bool:
    """The four legacy acceptance predicates, with manual flips excluded earlier."""
    if any(graph.in_degree(n) == 0 or graph.out_degree(n) == 0
           for n in graph if n not in boundary):
        return False
    if not nx.is_directed_acyclic_graph(graph):
        return False
    return (set(outlets) <= reachable(graph, inlets)
            and set(inlets) <= reachable(graph, outlets, reverse=True))


def first_valid_flip(
    edges: Sequence[Sequence[int]], candidates: Sequence[int], boundary: set[int],
    inlets: Sequence[int], outlets: Sequence[int],
) -> list[int] | None:
    """First feasible nonempty combination, cardinality then input order.

    The caller enforces the legacy ten-flippable-edge budget *before* calling.
    Stop at the first success: the old exhaustive minimum-cardinality choice
    uses this exact ordering. No exponential list or per-candidate graph copy.
    """
    # Key edges by link position so flips remain reversible for parallel links.
    graph = nx.MultiDiGraph()
    graph.add_edges_from((u, v, i) for i, (u, v) in enumerate(edges))
    for count in range(1, len(candidates) + 1):
        for selected in itertools.combinations(candidates, count):
            for index in selected:
                u, v = edges[index]
                graph.remove_edge(u, v, index)
                graph.add_edge(v, u, index)
            valid = valid_orientation(graph, boundary, inlets, outlets)
            for index in reversed(selected):
                u, v = edges[index]
                graph.remove_edge(v, u, index)
                graph.add_edge(u, v, index)
            if valid:
                return list(selected)
    return None

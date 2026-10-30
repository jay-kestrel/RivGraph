# -*- coding: utf-8 -*-
"""
Multiplex node-link I/O (layers/io.py)
======================================

Node-link JSON serialization for multiplex graphs (spec deliverable D5):
export a multiplex ``MultiDiGraph`` to a node-link JSON document and import
it back with round-trip fidelity for parallel edges and attributes.

Node-link is the default multiplex format per the feature spec: unlike
GraphML/GEXF it round-trips multigraph parallel keys and arbitrary
attributes, and unlike pickle it is a safe, inspectable text format.

Two type-fidelity gaps between the graph and its JSON form are closed by a
schema-aware adapter on import:

- JSON has no tuple type, so coordinate attributes (``centroid_xy`` on
  landscape nodes, ``snapped_xy`` on contact edges) come back as lists; the
  adapter restores the tuple form the schema uses.
- The StrEnum attributes (``layer``, ``node_type``, ``edge_type``) serialize
  as plain strings; the adapter restores the enum members. Values outside
  the vocabulary are left untouched so :func:`validate_multiplex` reports
  them with the contract's own error message.

Round-trip contract: node ids, the ``(u, v, key)`` edge multiset, and every
node/edge/graph attribute compare exactly equal after export and import,
for graphs whose attributes are JSON-serializable. Free-form attributes
holding tuples other than the two schema coordinates (e.g. a hand-built
``polygon``) serialize as nested lists -- keep such attributes JSON-native
(lists) when exact type recovery matters.
"""
from __future__ import annotations

import json

import networkx as nx

from rivgraph.layers.core import (
    EdgeType,
    LayerType,
    MULTIPLEX_FLAG,
    NodeType,
    validate_multiplex,
)

__all__ = [
    'multiplex_from_node_link_data',
    'multiplex_to_node_link_data',
    'read_multiplex_json',
    'write_multiplex_json',
]

#: Node attributes holding an (x, y) coordinate pair: export writes lists,
#: import restores tuples.
XY_PAIR_NODE_ATTRS = frozenset({'centroid_xy'})

#: Edge attributes holding an (x, y) coordinate pair.
XY_PAIR_EDGE_ATTRS = frozenset({'snapped_xy'})

#: Enum-valued attributes and the vocabulary each round-trips to.
_ENUM_NODE_ATTRS = {'layer': LayerType, 'node_type': NodeType}
_ENUM_EDGE_ATTRS = {'edge_type': EdgeType}


def _restore_attrs(attrs, xy_pair_attrs, enum_vocabs):
    """Return a copy of ``attrs`` with schema coordinates restored to tuples
    and enum vocabulary values restored to StrEnum members."""
    restored = dict(attrs)
    for attr in xy_pair_attrs:
        value = restored.get(attr)
        if isinstance(value, list) and len(value) == 2:
            restored[attr] = tuple(value)
    for attr, vocab in enum_vocabs.items():
        value = restored.get(attr)
        if isinstance(value, str):
            try:
                restored[attr] = vocab(value)
            except ValueError:
                # Not in the vocabulary: leave the raw value for
                # validate_multiplex to report as a contract violation.
                pass
    return restored


def multiplex_to_node_link_data(multiplex):
    """Serialize a validated multiplex graph to a JSON-ready node-link dict.

    Parameters
    ----------
    multiplex : networkx.MultiDiGraph
        A multiplex graph built by :func:`rivgraph.layers.build_multiplex`
        (with landscape layers attached), or one that passes
        :func:`rivgraph.layers.validate_multiplex`.

    Returns
    -------
    dict
        Node-link document (``nodes``/``edges`` lists plus graph attributes).

    Raises
    ------
    ValueError
        If ``multiplex`` violates the multiplex attribute contract: an
        invalid graph must not serialize into a file consumers trust.
    """
    validate_multiplex(multiplex)
    return nx.node_link_data(multiplex)


def multiplex_from_node_link_data(data):
    """Rebuild a validated multiplex graph from a node-link dict, restoring
    schema attribute types (enums and coordinate tuples).

    Parameters
    ----------
    data : dict
        Node-link document as produced by :func:`multiplex_to_node_link_data`.

    Returns
    -------
    networkx.MultiDiGraph
        The multiplex graph, validated after restoration.

    Raises
    ------
    ValueError
        If ``data`` is not a node-link document, does not describe a directed
        multigraph, lacks the multiplex marker, or breaks the multiplex
        attribute contract.
    """
    try:
        G = nx.node_link_graph(data)
    except Exception as exc:
        raise ValueError(
            f'Data is not a node-link document networkx can read: '
            f'{type(exc).__name__}: {exc}') from exc
    if not (G.is_directed() and G.is_multigraph()):
        raise ValueError(
            'Node-link document does not describe a directed multigraph; '
            'the multiplex format is a MultiDiGraph.')
    if not G.graph.get(MULTIPLEX_FLAG, False):
        raise ValueError(
            f'Node-link document lacks the multiplex marker '
            f'(G.graph[{MULTIPLEX_FLAG!r}]); not a RivGraph multiplex.')

    for _, attrs in G.nodes(data=True):
        attrs.update(_restore_attrs(attrs, XY_PAIR_NODE_ATTRS, _ENUM_NODE_ATTRS))
    for _, _, attrs in G.edges(data=True):
        attrs.update(_restore_attrs(attrs, XY_PAIR_EDGE_ATTRS, _ENUM_EDGE_ATTRS))

    validate_multiplex(G)
    return G


def write_multiplex_json(multiplex, path_out):
    """Write a multiplex graph to ``path_out`` as node-link JSON.

    The graph is validated before writing: an invalid multiplex must not
    serialize.
    """
    data = multiplex_to_node_link_data(multiplex)
    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def read_multiplex_json(path_in):
    """Read a node-link JSON file into a validated multiplex graph.

    Raises
    ------
    ValueError
        If the file is not JSON, not a node-link document, or breaks the
        multiplex contract (see :func:`multiplex_from_node_link_data`).
    """
    with open(path_in, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{path_in} is not valid JSON: {exc}') from exc
    return multiplex_from_node_link_data(data)

# -*- coding: utf-8 -*-
"""
Multiplex core (layers/core.py)
===============================

Core construction of the landscape-layer multiplex graph: a flat NetworkX
``MultiDiGraph`` in which the channel network is one layer and landscape
features (lakes, wetlands, islands) attach as additional nodes and edges.

Design follows the approved feature spec:

- The channel layer is a sibling of ``graphiphy``
  (``rivgraph.deltas.delta_metrics``): same weight/linkid edge attributes, but
  built on ``MultiDiGraph`` so a cross-layer contact edge can legitimately
  coexist with a channel edge at the same node pair.
- Layers attach only after the channel IDs are finalized
  (``rivgraph.ln_utils.finalize_network_ids``); builders refuse to run on
  provisional IDs because landscape contacts must reference finalized IDs.
- DAG-guarded consumers never read the full graph; they read ``channel_view``,
  the channel-node-only projection whose structure matches ``graphiphy``.

Module layout note: lake extraction (D2), patch extraction (D3), metrics (D4),
and node-link I/O (D5) live in sibling modules of this package.
"""
import enum

import networkx as nx
import numpy as np

from rivgraph.ln_utils import network_ids_are_finalized


class LayerType(enum.StrEnum):
    """Vocabulary of the node ``layer`` attribute."""
    CHANNEL = 'channel'
    LAKE = 'lake'
    WETLAND = 'wetland'
    ISLAND = 'island'


class NodeType(enum.StrEnum):
    """Vocabulary of the node ``node_type`` attribute."""
    JUNCTION = 'junction'
    INLET = 'inlet'
    OUTLET = 'outlet'
    LANDSCAPE = 'landscape'


class EdgeType(enum.StrEnum):
    """Vocabulary of the edge ``edge_type`` attribute."""
    CHANNEL = 'channel'
    CONTACT = 'contact'
    PATCH_ADJACENCY = 'patch_adjacency'
    BOUNDARY = 'boundary'


# ---------------------------------------------------------------------------
# Attribute schema (the contract enforced by validate_multiplex)
# ---------------------------------------------------------------------------

#: Attributes required on every node.
NODE_REQUIRED_ATTRS = frozenset({'layer', 'node_type'})

#: Attributes required on every landscape node (layer is not CHANNEL). The
#: node id (graph node key) and ``label_id`` (label-raster id) identify the
#: feature; geometry and size come from the mask extraction.
LANDSCAPE_NODE_REQUIRED_ATTRS = frozenset({'area_m2', 'perimeter_m', 'centroid_xy', 'label_id'})

#: Attributes required on every edge.
EDGE_REQUIRED_ATTRS = frozenset({'edge_type', 'interlayer'})

#: Additional attributes required per edge type. Channel edges follow the
#: graphiphy contract; contacts carry the buffer and snapped coordinate that
#: produced them.
EDGE_TYPE_REQUIRED_ATTRS = {
    EdgeType.CHANNEL: frozenset({'weight', 'linkid'}),
    EdgeType.CONTACT: frozenset({'weight', 'buffer_px', 'snapped_xy'}),
    EdgeType.PATCH_ADJACENCY: frozenset({'weight'}),
    EdgeType.BOUNDARY: frozenset({'weight'}),
}

#: Edge types stored as symmetric bidirectional pairs. NetworkX cannot mix
#: directed and undirected edges in one graph (and GraphML/GEXF reject mixed
#: graphs), so undirected landscape adjacencies are directed pairs with
#: identical attributes; consumers filter by ``edge_type``.
BIDIRECTIONAL_EDGE_TYPES = frozenset({EdgeType.CONTACT, EdgeType.PATCH_ADJACENCY, EdgeType.BOUNDARY})

#: Edge types that cross between a landscape node and a channel node.
INTERLAYER_EDGE_TYPES = frozenset({EdgeType.CONTACT, EdgeType.BOUNDARY})

#: Marker set on the graph dict by build_multiplex.
MULTIPLEX_FLAG = 'rivgraph_multiplex'


def _channel_multidigraph(links, nodes, weight=None):
    """Build the channel layer of the multiplex graph.

    Mirrors ``graphiphy`` (rivgraph.deltas.delta_metrics): nodes from
    ``nodes['id']``, directed edges from ``links['conn']`` carrying ``weight``
    and ``linkid``. Parallel links are kept as parallel MultiDiGraph edges (no
    synthetic-node splitting is needed -- that machinery exists because
    ``DiGraph`` collapses them).

    Every channel node also gets ``node_type``: 'inlet'/'outlet' when listed
    in the corresponding nodes keys, 'junction' otherwise.
    """
    if weight is not None and weight not in links:
        raise RuntimeError('Provided weight key not in links dictionary.')

    if weight is None:
        weights = np.ones((len(links['conn']), 1))
    else:
        weights = np.array(links[weight], dtype=float)

    if np.sum(weights <= 0) > 0:
        raise ValueError('All weights must be positive.')

    inlets = set(nodes.get('inlets', []))
    outlets = set(nodes.get('outlets', []))

    G = nx.MultiDiGraph()
    for nid in nodes['id']:
        if nid in inlets:
            node_type = NodeType.INLET
        elif nid in outlets:
            node_type = NodeType.OUTLET
        else:
            node_type = NodeType.JUNCTION
        G.add_node(nid, layer=LayerType.CHANNEL, node_type=node_type)

    for conn, wt, lid in zip(links['conn'], weights, links['id']):
        G.add_edge(conn[0], conn[1],
                   edge_type=EdgeType.CHANNEL,
                   interlayer=False,
                   weight=float(np.asarray(wt).squeeze()),
                   linkid=lid)

    return G


def build_multiplex(channel_links, channel_nodes, weight=None):
    """Build the multiplex graph from finalized channel links/nodes dicts.

    Parameters
    ----------
    channel_links : dict
        RivGraph links dictionary (dict of aligned lists) with finalized IDs.
    channel_nodes : dict
        RivGraph nodes dictionary (dict of aligned lists) with finalized IDs.
    weight : str, optional
        Link attribute to weight channel edges, as in ``graphiphy``. Default
        None gives every channel edge weight 1.

    Returns
    -------
    G : networkx.MultiDiGraph
        The validated channel-only multiplex graph, ready for
        :func:`add_landscape_layer` calls.

    Raises
    ------
    RuntimeError
        If the channel IDs are not finalized, or the weight key is missing.
    ValueError
        If the built graph violates the attribute contract.
    """
    if not network_ids_are_finalized(channel_links, channel_nodes):
        raise RuntimeError(
            'Channel network IDs must be finalized (finalize_network_ids) before '
            'building the multiplex: landscape contacts may only reference '
            'finalized IDs.')

    G = _channel_multidigraph(channel_links, channel_nodes, weight=weight)
    G.graph[MULTIPLEX_FLAG] = True
    validate_multiplex(G)
    return G


def add_landscape_layer(G, layer_name, layer_nodes, layer_edges):
    """Append one landscape layer to a multiplex graph and re-validate.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Multiplex graph from :func:`build_multiplex`, possibly with landscape
        layers already attached.
    layer_name : str
        Landscape layer type: one of the LayerType members other than
        'channel' ('lake', 'wetland', 'island').
    layer_nodes : mapping
        Maps label-raster id to a feature spec dict. Required keys:
        'centroid_xy', 'area_m2', 'perimeter_m'. Optional: 'polygon',
        'mask_date'. Landscape node ids are assigned from ``max(G.nodes) + 1``
        (max(channel ids) + 1 on a fresh channel graph) in ascending
        label-raster sort order, keeping landscape ids disjoint from channel
        ids deterministically.
    layer_edges : sequence
        ``(label_id, channel_node_id, attrs)`` contact tuples. ``attrs`` must
        carry the contact schema ('weight', 'buffer_px', 'snapped_xy',
        optional 'mask_date', 'activation_level'); ``edge_type='contact'`` and
        ``interlayer=True`` are set here. Each contact is stored as a
        symmetric bidirectional edge pair.

    Returns
    -------
    G : networkx.MultiDiGraph
        The same graph, validated after the layer is attached.

    Raises
    ------
    ValueError
        On an invalid layer name, missing feature attributes, unknown labels,
        non-channel contact endpoints, or missing contact attributes.
    """
    layer = LayerType(layer_name)
    if layer == LayerType.CHANNEL:
        raise ValueError('layer_name must be a landscape type (lake, wetland, island), not channel.')

    # Deterministic id assignment: ascending label-raster order, starting past
    # every existing id so repeated calls never collide across layers.
    next_id = (max(G.nodes) + 1) if len(G) else 0
    label_to_id = {}
    for label in sorted(layer_nodes):
        spec = layer_nodes[label]
        missing = {'centroid_xy', 'area_m2', 'perimeter_m'} - set(spec)
        if missing:
            raise ValueError(f'Landscape node for label {label} is missing required attrs: {sorted(missing)}')
        node_id = next_id
        next_id += 1
        label_to_id[label] = node_id
        G.add_node(node_id,
                   layer=layer,
                   node_type=NodeType.LANDSCAPE,
                   label_id=label,
                   centroid_xy=spec['centroid_xy'],
                   area_m2=spec['area_m2'],
                   perimeter_m=spec['perimeter_m'],
                   polygon=spec.get('polygon'),
                   mask_date=spec.get('mask_date'))

    for label, channel_node, attrs in layer_edges:
        if label not in label_to_id:
            raise ValueError(f'Contact edge references label {label}, which is not in layer_nodes.')
        if channel_node not in G:
            raise ValueError(f'Contact edge references node {channel_node}, which is not in the graph.')
        if G.nodes[channel_node]['layer'] != LayerType.CHANNEL:
            raise ValueError(f'Contact edge must attach to a channel node; {channel_node} is not one.')
        if 'edge_type' in attrs and attrs['edge_type'] != EdgeType.CONTACT:
            raise ValueError("Contact edge attrs must use edge_type 'contact'.")
        missing = {'weight', 'buffer_px', 'snapped_xy'} - set(attrs)
        if missing:
            raise ValueError(f'Contact edge for label {label} is missing required attrs: {sorted(missing)}')

        edge_attrs = dict(attrs)
        edge_attrs['edge_type'] = EdgeType.CONTACT
        edge_attrs['interlayer'] = True
        landscape_id = label_to_id[label]
        G.add_edge(landscape_id, channel_node, **edge_attrs)
        G.add_edge(channel_node, landscape_id, **edge_attrs)

    validate_multiplex(G)
    return G


def channel_view(multiplex):
    """Channel-node-only projection of a multiplex graph.

    The only projection DAG-guarded metrics may consume: landscape nodes are
    hidden and every landscape-touching edge (contacts) is excluded with them,
    while all channel-channel edges -- including parallel edges and their
    attributes -- are preserved. The returned view is live: it reflects later
    changes to the underlying graph.

    Parameters
    ----------
    multiplex : networkx.MultiDiGraph
        A multiplex graph built by :func:`build_multiplex`.

    Returns
    -------
    networkx.MultiDiGraph
        A subgraph view containing only nodes with ``layer == 'channel'``.
    """
    return nx.subgraph_view(
        multiplex,
        filter_node=lambda n: multiplex.nodes[n]['layer'] == LayerType.CHANNEL)


def validate_multiplex(G):
    """Validate the multiplex attribute contract, raising ValueError if broken.

    Checks that:

    - every node has a valid ``layer`` and ``node_type``;
    - landscape nodes carry the required feature attributes;
    - every edge has a valid ``edge_type``, the required attributes for its
      type, and an ``interlayer`` flag consistent with its type;
    - undirected adjacencies (contact, patch_adjacency, boundary) are stored
      as symmetric bidirectional pairs with identical attributes;
    - landscape node ids are disjoint from channel ids (every landscape id
      exceeds the maximum channel id, the deterministic assignment rule).

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The graph to validate.

    Raises
    ------
    ValueError
        On the first contract violation found.
    """
    layer_values = frozenset(LayerType)
    node_type_values = frozenset(NodeType)
    edge_type_values = frozenset(EdgeType)

    channel_ids = []
    landscape_ids = []
    for n, data in G.nodes(data=True):
        missing = NODE_REQUIRED_ATTRS - set(data)
        if missing:
            raise ValueError(f'Node {n} is missing required attrs: {sorted(missing)}')
        if data['layer'] not in layer_values:
            raise ValueError(f"Node {n} has invalid layer: {data['layer']!r}")
        if data['node_type'] not in node_type_values:
            raise ValueError(f"Node {n} has invalid node_type: {data['node_type']!r}")
        if data['layer'] == LayerType.CHANNEL:
            channel_ids.append(n)
        else:
            if data['node_type'] != NodeType.LANDSCAPE:
                raise ValueError(f"Node {n} on a landscape layer must have node_type 'landscape', "
                                 f"got {data['node_type']!r}")
            missing = LANDSCAPE_NODE_REQUIRED_ATTRS - set(data)
            if missing:
                raise ValueError(f'Landscape node {n} is missing required attrs: {sorted(missing)}')
            landscape_ids.append(n)

    if landscape_ids and channel_ids and min(landscape_ids) <= max(channel_ids):
        raise ValueError('Landscape node ids must be disjoint from channel ids: '
                         'every landscape id must exceed the maximum channel id.')

    for u, v, data in G.edges(data=True):
        missing = EDGE_REQUIRED_ATTRS - set(data)
        if missing:
            raise ValueError(f'Edge ({u}, {v}) is missing required attrs: {sorted(missing)}')
        edge_type = data['edge_type']
        if edge_type not in edge_type_values:
            raise ValueError(f'Edge ({u}, {v}) has invalid edge_type: {edge_type!r}')
        expected_interlayer = edge_type in INTERLAYER_EDGE_TYPES
        if bool(data['interlayer']) is not expected_interlayer:
            raise ValueError(f'Edge ({u}, {v}) of type {edge_type} must have interlayer={expected_interlayer}.')
        for attr in EDGE_TYPE_REQUIRED_ATTRS[edge_type]:
            if attr not in data:
                raise ValueError(f'Edge ({u}, {v}) of type {edge_type} is missing required attr {attr!r}.')
        if edge_type in BIDIRECTIONAL_EDGE_TYPES:
            if not G.has_edge(v, u) or not any(data == reverse_data for reverse_data in G[v][u].values()):
                raise ValueError(f'Edge ({u}, {v}) of type {edge_type} lacks a symmetric reverse pair '
                                 '(undirected adjacencies are stored as bidirectional pairs).')

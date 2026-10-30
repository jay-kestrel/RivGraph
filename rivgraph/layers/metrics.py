"""
Layer metrics (layers/metrics.py)
=================================

Spec D4: structural and graph-theoretic metrics over the landscape-layer
multiplex graph, computed with plain functions on the flat ``MultiDiGraph``.

Every function here is a pure read over the graph it is given -- none of them
adds or mutates nodes, edges, or attributes (tests pin this with a full
node-link snapshot comparison). Per the spec's locked decision 6, v1 metrics
are structural only: hydraulic quantities (exposure time, flux, leakage) need
discharge or water-level data and are follow-on work.

Conventions inherited from the core contract:

- Landscape nodes carry ``layer`` other than ``'channel'``; per-layer metrics
  filter on it (``layer=None`` means all landscape nodes).
- Undirected landscape adjacencies (contact, patch_adjacency, boundary) are
  stored as symmetric bidirectional pairs, so one physical adjacency is
  exactly one outgoing edge from a landscape node; the per-node counters
  below exploit that to report distinct adjacencies rather than directed
  edge counts (which would double every landscape edge).
- ``pathway_counts`` uses dense NumPy adjacency powers, consistent with
  ``adjacency_matrix`` (``rivgraph.classes``), with ``k`` capped for memory;
  the scaling limit is a documented spec risk, and a sparse implementation
  can later take its place behind the same interface.

Distance-to-channel scope note: the spec's metric table defines distance to
channel in mask space (min/mean distance from feature pixels to the channel
mask), which needs the rasters this graph-only module does not receive. The
v1 graph-theoretic analogue implemented here is the directed shortest-path
distance from a landscape node to the nearest channel node over contact and
channel edges; mask-space distances remain with the extraction modules.
"""
import networkx as nx
import numpy as np

from rivgraph.layers.core import EdgeType, LayerType

#: Upper bound on the ``k`` accepted by :func:`pathway_counts`. Dense
#: adjacency powers grow with delta size (a documented spec risk); the cap
#: keeps a call's work bounded while leaving headroom above the default of 3.
MAX_PATHWAY_K = 10


def landscape_nodes(G, layer=None):
    """Sorted landscape node ids of ``G``, optionally restricted to one layer.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        A validated multiplex graph (see ``validate_multiplex``).
    layer : str, optional
        Restrict to one landscape layer ('lake', 'wetland', 'island').
        Default None returns every node on a non-channel layer.

    Returns
    -------
    list
        Node ids in ascending order.
    """
    if layer is None:
        return sorted(n for n, d in G.nodes(data=True) if d['layer'] != LayerType.CHANNEL)
    return sorted(n for n, d in G.nodes(data=True) if d['layer'] == LayerType(layer))


def _distinct_outgoing(G, node, edge_type):
    """Distinct adjacencies of ``edge_type`` at ``node`` (one per bidirectional pair)."""
    return sum(1 for _, _, d in G.edges(node, data=True) if d['edge_type'] == edge_type)


def isolated_lakes(G):
    """Degree-zero lake node ids: lakes with no contact edges.

    Never-connected lakes are scientifically load-bearing (the Mackenzie
    record's 2,899 invariant-connectivity lakes), so the lake layer keeps
    them as nodes; this metric is how callers count them.

    Returns
    -------
    tuple
        Sorted lake node ids with zero distinct contact adjacencies.
    """
    return tuple(n for n in landscape_nodes(G, LayerType.LAKE)
                 if _distinct_outgoing(G, n, EdgeType.CONTACT) == 0)


def contact_degree(G, layer=None):
    """Distinct contact-edge count per landscape node, by layer.

    Each contiguous contact group contributes exactly one adjacency (the
    bidirectional pair convention), so the count is of groups, not pixels
    or directed edges.

    Returns
    -------
    dict
        Maps landscape node id to its number of distinct contact edges.
    """
    return {n: _distinct_outgoing(G, n, EdgeType.CONTACT) for n in landscape_nodes(G, layer)}


def contact_degree_distribution(G, layer=None):
    """Distribution of contact degrees across landscape nodes.

    Returns
    -------
    dict
        Maps a contact-degree value to the number of landscape nodes
        (optionally of one ``layer``) having that degree.
    """
    dist = {}
    for degree in contact_degree(G, layer).values():
        dist[degree] = dist.get(degree, 0) + 1
    return dict(sorted(dist.items()))


def contact_counts_per_link(G):
    """Contact-group count per channel link, across all landscape layers.

    Lake contacts carry the snapped link as ``snapped_linkid``; patch
    boundary edges carry ``channel_linkid``. Iterating from landscape nodes
    counts each undirected pair once (the reverse direction starts at the
    channel node). Contacts lacking both attrs are counted under the None
    key so nothing silently disappears.

    Returns
    -------
    dict
        Maps channel link id (or None) to its number of distinct contact
        groups; ascending link-id order, None last.
    """
    counts = {}
    for node in landscape_nodes(G):
        for _, _, d in G.edges(node, data=True):
            if d['edge_type'] not in (EdgeType.CONTACT, EdgeType.BOUNDARY):
                continue
            link = d.get('snapped_linkid', d.get('channel_linkid'))
            counts[link] = counts.get(link, 0) + 1
    ordered = {link: counts[link] for link in sorted(k for k in counts if k is not None)}
    if None in counts:
        ordered[None] = counts[None]
    return ordered


def adjacency_degree(G, layer=None):
    """Distinct patch-adjacency count per landscape node (patch-to-patch edges).

    Returns
    -------
    dict
        Maps landscape node id to its number of distinct ``patch_adjacency``
        adjacencies.
    """
    return {n: _distinct_outgoing(G, n, EdgeType.PATCH_ADJACENCY) for n in landscape_nodes(G, layer)}


def boundary_degree(G, layer=None):
    """Distinct channel-boundary contact count per landscape node.

    Returns
    -------
    dict
        Maps landscape node id to its number of distinct ``boundary`` edges
        (one per contiguous patch-channel contact group).
    """
    return {n: _distinct_outgoing(G, n, EdgeType.BOUNDARY) for n in landscape_nodes(G, layer)}


def boundary_weight(G, layer=None):
    """Total channel-boundary weight per landscape node.

    Returns
    -------
    dict
        Maps landscape node id to the summed ``weight`` of its boundary
        edges -- the shared-boundary length (metres) facing the channel
        network, in the extraction's pixel units when ``pixlen=1``.
    """
    return {n: float(sum(d['weight'] for _, _, d in G.edges(n, data=True)
                         if d['edge_type'] == EdgeType.BOUNDARY))
            for n in landscape_nodes(G, layer)}


def patch_degree(G, layer=None):
    """Total per-patch degree: channel contacts plus adjacent patches.

    The spec's per-patch degree metric ("channel contacts and adjacent
    patches"): the sum of :func:`boundary_degree` and :func:`adjacency_degree`.

    Returns
    -------
    dict
        Maps landscape node id to its boundary plus adjacency degree.
    """
    return {n: _distinct_outgoing(G, n, EdgeType.BOUNDARY)
            + _distinct_outgoing(G, n, EdgeType.PATCH_ADJACENCY)
            for n in landscape_nodes(G, layer)}


def distance_to_channel(G, source, weight=None):
    """Directed shortest-path distance from ``source`` to the channel layer.

    Traverses directed edges (contact pairs are symmetric by construction;
    channel edges flow downstream) and returns the minimum length path from
    ``source`` to any channel-layer node -- the graph analogue of the spec's
    distance-to-channel metric.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        A validated multiplex graph.
    source : node
        Landscape node to measure from (a channel source returns 0).
    weight : str, optional
        Edge attribute to sum (e.g. ``'weight'``). Default None counts hops.

    Returns
    -------
    float
        The shortest distance; ``inf`` when no directed path reaches the
        channel layer (e.g. a degree-0 lake).

    Raises
    ------
    ValueError
        If ``source`` is not in the graph.
    """
    if source not in G:
        raise ValueError(f'Source node {source} is not in the graph.')
    if weight is None:
        lengths = nx.single_source_shortest_path_length(G, source)
    else:
        lengths = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    reachable = [lengths[n] for n, d in G.nodes(data=True)
                 if d['layer'] == LayerType.CHANNEL and n in lengths]
    return min(reachable) if reachable else float('inf')


def pathway_counts(G, source, k=3):
    """Distinct channel-to-landscape pathways of at most ``k`` steps.

    Counts directed walks (distinct edge sequences; nodes may repeat) from
    ``source`` to every landscape node reachable within ``k`` steps: the
    cumulative sum of the dense adjacency-matrix powers ``A**i`` for
    ``i = 1..k`` applied to the source, as the spec's pathway-count metric
    prescribes (``Ak . 1`` on the dense adjacency, parallel edges counting
    as distinct pathways).

    Parameters
    ----------
    G : networkx.MultiDiGraph
        A validated multiplex graph.
    source : node
        Origin node, typically a channel node (the metric is
        channel-to-landscape); any node is accepted.
    k : int, optional
        Maximum pathway length in steps (default 3). Capped at
        :data:`MAX_PATHWAY_K` because adjacency powers are dense.

    Returns
    -------
    dict
        Maps landscape node id to its cumulative walk count over 1..k steps;
        targets with zero pathways are absent, ascending node-id order.

    Raises
    ------
    ValueError
        If ``source`` is not in the graph, or ``k`` is outside 1..MAX_PATHWAY_K.
    """
    if source not in G:
        raise ValueError(f'Source node {source} is not in the graph.')
    if k < 1:
        raise ValueError('k must be >= 1.')
    if k > MAX_PATHWAY_K:
        raise ValueError(f'k must be <= {MAX_PATHWAY_K}: dense adjacency powers grow '
                         'with delta size (feature spec risk table).')

    nodelist = sorted(G.nodes)
    index = {n: i for i, n in enumerate(nodelist)}
    adjacency = nx.to_numpy_array(G, nodelist=nodelist, weight=None).astype(np.int64)

    walks = np.zeros(len(nodelist), dtype=np.int64)
    walks[index[source]] = 1
    cumulative = np.zeros(len(nodelist), dtype=np.int64)
    for _ in range(k):
        walks = walks @ adjacency
        cumulative += walks

    return {n: int(cumulative[index[n]]) for n in landscape_nodes(G)
            if cumulative[index[n]] > 0}


def _integral_index_of_connectivity(adjacency_graph, areas, nodes):
    """IIC over ``nodes``: sum of a_u * a_v / (1 + d_uv) over ordered pairs.

    ``d_uv`` is the patch-network hop distance (patch_adjacency edges only);
    pairs in different components contribute zero, and the diagonal (u == v,
    d = 0) is included, per the binary-graph IIC definition the spec cites.
    """
    subgraph = adjacency_graph.subgraph(nodes)
    lengths = dict(nx.all_pairs_shortest_path_length(subgraph))
    total = 0.0
    for u in nodes:
        area_u = areas[u]
        for v in nodes:
            if u == v:
                total += area_u * area_u
                continue
            hops = lengths.get(u, {}).get(v)
            if hops is not None:
                total += area_u * areas[v] / (1 + hops)
    return total


def patch_importance(G, index='iic'):
    """dI%: relative connectivity-index drop when each patch is removed.

    For each landscape node, removes it (with its adjacency links) and
    reports the percentage drop of the integral index of connectivity
    computed over the remaining patch network -- the spec's patch-importance
    metric, on the IIC basis for binary graphs. Distances run over
    ``patch_adjacency`` edges only: IIC is a landscape-connectivity index,
    and channel-mediated travel is :func:`distance_to_channel`'s job.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        A validated multiplex graph.
    index : str, optional
        Connectivity index; only ``'iic'`` is implemented (v1 is structural;
        the probability-of-connectivity basis needs areas beyond the graph).

    Returns
    -------
    dict
        Maps landscape node id to its dI% (0--100); 0.0 for every node when
        the full-graph index is 0. Ascending node-id order.

    Raises
    ------
    ValueError
        If an unsupported ``index`` is requested.
    """
    if index != 'iic':
        raise ValueError(f"Unsupported index {index!r}; only 'iic' is implemented.")
    patches = landscape_nodes(G)
    if not patches:
        return {}

    areas = {n: float(G.nodes[n]['area_m2']) for n in patches}
    # Undirected simple projection of the patch-adjacency pairs; a fresh
    # graph, so G is never touched.
    adjacency_graph = nx.Graph()
    adjacency_graph.add_nodes_from(patches)
    for node in patches:
        for _, v, d in G.edges(node, data=True):
            if d['edge_type'] == EdgeType.PATCH_ADJACENCY:
                adjacency_graph.add_edge(node, v)

    full = _integral_index_of_connectivity(adjacency_graph, areas, patches)
    if full <= 0:
        return {n: 0.0 for n in patches}
    return {n: 100.0 * (full - _integral_index_of_connectivity(adjacency_graph, areas,
                                                               set(patches) - {n})) / full
            for n in patches}


def cross_layer_components(G):
    """Weakly connected components that span more than one layer.

    The spec's joined-connectivity metric: components of the full multiplex
    graph (weak connectivity, since contact and adjacency pairs carry the
    undirected structure) that contain nodes from at least two distinct
    layers. Single-layer components -- e.g. an isolated lake -- are excluded;
    iterate ``nx.weakly_connected_components(G)`` for all of them.

    Returns
    -------
    tuple
        Frozensets of node ids, ordered by their smallest member.
    """
    spanning = []
    for component in nx.weakly_connected_components(G):
        layers = {G.nodes[n]['layer'] for n in component}
        if len(layers) > 1:
            spanning.append(frozenset(component))
    return tuple(sorted(spanning, key=min))

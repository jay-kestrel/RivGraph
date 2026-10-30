"""
Wetland and island patch layers (layers/patches.py)
===================================================

D3 of the multiplex-layers feature spec: extract wetland and island patches
from a landmass mask, attach them to the multiplex graph as landscape
layers, and wire them to the channel network and to each other.

Rules (feature spec, "Concrete node and edge definitions per layer"):

- **Patch nodes** are connected components of the morphologically closed
  landmass minus the channel mask. Closing (``im_utils.dilate`` then
  ``im_utils.erode``) fills narrow water intrusions in the landmass (channels
  outside the extracted network, lake arms, bay inlets) so they do not split
  or strand fragments; the extracted channel mask is then subtracted and keeps
  separating patches across genuine channels. Each patch carries ``area_m2``,
  ``perimeter_m``, and ``centroid_xy`` (map coordinates, i.e. ``(col, row)``).
- **Patch adjacency edges** (``patch_adjacency``) join two patches whose
  *shared channel bank* meets a length threshold: the number of channel-mask
  pixels 4-adjacent to both patches. This is the raster analogue of the
  published "8-connected at a reference water level" rule -- patches separated
  only by a short channel segment exchange water across it. Stored as
  bidirectional pairs.
- **Boundary edges** (``boundary``) join every patch touching the channel
  mask to the channel network: each contiguous (8-connected) group of
  channel-adjacent patch pixels becomes one edge pair, attached to the
  endpoint node of the nearest finalized channel link. Stored as
  bidirectional pairs.
- **Island special case** reuses the existing island machinery:
  ``mask_utils.surrounding_link_properties`` (the island-to-channel mapping
  behind ``rivnetwork.get_islands``) detects the channel loop around each
  bounded patch. When a loop is found, the surrounding links map directly to
  boundary edge pairs (one per link, attached to the link's nearest endpoint
  node). When loop detection returns its documented NaN fallback, the patch
  keeps island-ness but falls back to the mask-intersection rule for its
  wiring. Patches belonging to the border-attached (mainland) component of
  the channel-mask complement are wetlands.

Island detection needs the machinery's input label image (components of the
channel-mask complement, border blob removed). It is built here with the same
``im_utils.regionprops`` primitive ``mask_utils.get_island_properties`` uses
internally, minus polygonization: the polygon output is not consumed by patch
wiring, and ``get_island_properties`` currently cannot run under affine
versions without ``Affine.__matmul__`` (the pre-existing rasterio
island-polygon test failures). The island-to-channel mapping itself is the
untouched existing machinery.

All landscape adjacencies use the directed-pair convention from the spec
(NetworkX cannot mix directed and undirected edges in one graph): symmetric
``u->v`` / ``v->u`` edges with identical attributes; consumers filter by
``edge_type``.
"""
import ast
import warnings
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
from scipy.spatial import cKDTree

from rivgraph import im_utils, mask_utils
from rivgraph.layers.core import (
    EdgeType,
    LayerType,
    add_landscape_layer,
    validate_multiplex,
)
from rivgraph.ln_utils import network_ids_are_finalized


@dataclass(frozen=True)
class PatchExtraction:
    """Patch-side extraction result ready for multiplex attachment.

    Attributes
    ----------
    layer_nodes : dict
        Maps label-raster id to the feature spec for :func:`add_landscape_layer`
        (``centroid_xy``, ``area_m2``, ``perimeter_m``, optional ``mask_date``).
    island_labels : frozenset
        Labels of patches classified as islands (their enclosing complement
        component is not attached to the image border).
    boundary_edges : dict
        Maps label to a list of ``(channel_node_id, attrs)`` boundary contacts,
        already routed through the island machinery or the mask-intersection
        rule. ``attrs`` carry ``edge_type='boundary'``, ``interlayer=True``,
        ``weight``, ``channel_linkid`` and, for mask-rule contacts,
        ``snapped_xy``.
    adjacency_edges : tuple
        ``(label_a, label_b, attrs)`` patch-adjacency triples (``label_a <
        label_b``) whose shared channel bank met the threshold. ``attrs`` carry
        ``edge_type='patch_adjacency'``, ``interlayer=False``, ``weight``.
    island_link_ids : dict
        Maps island label to the machinery's surrounding link ids (the
        detected channel loop; empty when the wiring fell back to the
        mask-intersection rule).
    """

    layer_nodes: dict = field(default_factory=dict)
    island_labels: frozenset = frozenset()
    boundary_edges: dict = field(default_factory=dict)
    adjacency_edges: tuple = ()
    island_link_ids: dict = field(default_factory=dict)


def close_landmass(landmass, iterations=1, strel='square'):
    """Morphologically close a landmass mask (dilate then erode).

    Closing fills narrow water intrusions in the landmass so they do not
    split or strand island fragments (spec D3). The channel mask is
    subtracted afterwards and continues to separate patches across genuine
    channels; closing never bridges those.

    Parameters
    ----------
    landmass : np.array
        Binary landmass mask.
    iterations : int, optional
        Number of dilation/erosion passes; 0 disables closing. Default 1.
    strel : str, optional
        Structuring element passed to ``im_utils.dilate``/``erode``.

    Returns
    -------
    np.array
        The closed binary mask (same shape).
    """
    if iterations < 0:
        raise ValueError('iterations must be >= 0.')
    if iterations == 0:
        return np.array(landmass, dtype=bool)
    # scikit-image 0.25 (outside the repo's declared pin of <0.25) deprecates
    # morphology.square, which im_utils.dilate/erode construct internally, and
    # the pytest config escalates warnings to errors. Filter narrowly around
    # these calls; when scikit-image 0.27 removes square, im_utils.dilate
    # raises on its own and this filter does not mask that failure.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore',
            message=r'`square` is deprecated since version 0\.25',
            category=FutureWarning,
        )
        closed = im_utils.dilate(np.asarray(landmass, dtype=bool), n=iterations, strel=strel)
        closed = im_utils.erode(closed, n=iterations, strel=strel)
    return np.asarray(closed, dtype=bool)


def label_patches(landmass_closed, channel_mask, pixlen=1.0, pixarea=1.0, connectivity=2):
    """Label patch components of the closed landmass minus the channel mask.

    Parameters
    ----------
    landmass_closed : np.array
        Binary landmass mask after :func:`close_landmass`.
    channel_mask : np.array
        Binary channel mask (RivGraph's extracted network mask).
    pixlen : float, optional
        Pixel side length; scales ``perimeter_m``. Default 1.0.
    pixarea : float, optional
        Pixel area; scales ``area_m2``. Default 1.0.
    connectivity : int, optional
        1 for 4-connected components, 2 for 8-connected. Default 2.

    Returns
    -------
    Ipatch : np.array
        Label image; each patch's pixels carry its label-raster id.
    layer_nodes : dict
        Maps label to the feature spec for :func:`add_landscape_layer`.
    """
    channel = np.asarray(channel_mask, dtype=bool)
    patch_image = np.asarray(landmass_closed, dtype=bool) & ~channel
    props, Ipatch = im_utils.regionprops(
        patch_image,
        props=['label', 'area', 'perimeter', 'centroid'],
        connectivity=connectivity,
    )
    layer_nodes = {}
    for label, area, perim, centroid in zip(props['label'], props['area'], props['perimeter'], props['centroid']):
        # skimage centroids are (row, col); the graph contract uses map xy.
        layer_nodes[int(label)] = {
            'centroid_xy': [float(centroid[1]), float(centroid[0])],
            'area_m2': float(area) * float(pixarea),
            'perimeter_m': float(perim) * float(pixlen),
        }
    return Ipatch, layer_nodes


def shared_boundary_lengths(Ipatch, channel_mask):
    """Shared channel-bank length between each pair of distinct patches.

    Two distinct patch components never touch directly (they would be one
    component), so their shared boundary is measured through the channel that
    separates them: the number of channel-mask pixels 4-adjacent to both
    patches. This is the length of channel bank the two patches face each
    other across.

    Parameters
    ----------
    Ipatch : np.array
        Label image from :func:`label_patches`.
    channel_mask : np.array
        Binary channel mask.

    Returns
    -------
    dict
        Maps ``(label_a, label_b)`` (with ``label_a < label_b``) to the shared
        bank length in pixels. Patches sharing no channel pixel are absent.
    """
    channel = np.asarray(channel_mask, dtype=bool)
    counts = {}
    if not channel.any():
        return counts

    H, W = Ipatch.shape
    padded = np.pad(Ipatch, 1)
    neighbors = np.stack([
        padded[0:H, 1:W + 1],
        padded[2:H + 2, 1:W + 1],
        padded[1:H + 1, 0:W],
        padded[1:H + 1, 2:W + 2],
    ])
    rows, cols = np.nonzero(channel)
    stacked = np.sort(neighbors[:, rows, cols], axis=0)

    # Restrict the per-pixel loop to channel pixels bordering at least two
    # distinct patches; everything else cannot contribute a pair.
    distinct = (stacked[0] != 0).astype(int)
    distinct = distinct + np.count_nonzero((stacked[1:] != 0) & (stacked[1:] != stacked[:-1]), axis=0)
    for j in np.nonzero(distinct >= 2)[0]:
        values = sorted({int(v) for v in stacked[:, j] if v != 0})
        for i in range(len(values)):
            for k in range(i + 1, len(values)):
                key = (values[i], values[k])
                counts[key] = counts.get(key, 0) + 1
    return counts


def _dilate4(mask):
    """Dilate a binary mask within the 4-neighborhood (no diagonal spread)."""
    padded = np.pad(mask, 1)
    H, W = mask.shape
    out = mask.copy()
    out |= padded[0:H, 1:W + 1]
    out |= padded[2:H + 2, 1:W + 1]
    out |= padded[1:H + 1, 0:W]
    out |= padded[1:H + 1, 2:W + 2]
    return out


def _adjacent_pair_count(source, target):
    """Count ordered 4-adjacent (source pixel, target pixel) pairs."""
    H, W = source.shape
    s = np.pad(source, 1)
    t = np.pad(target, 1)
    center = s[1:H + 1, 1:W + 1]
    return int(
        np.count_nonzero(center & t[0:H, 1:W + 1])
        + np.count_nonzero(center & t[2:H + 2, 1:W + 1])
        + np.count_nonzero(center & t[1:H + 1, 0:W])
        + np.count_nonzero(center & t[1:H + 1, 2:W + 2])
    )


def _patch_contact_groups(patch_mask, channel_mask):
    """Contiguous groups of patch pixels 4-adjacent to the channel mask.

    One group per distinct contact location: per-pixel contacts would inflate
    the patch's channel degree, while merging groups would hide distinct
    entrances (the contiguity rule from the published lake recipe, applied to
    patches).

    Returns
    -------
    list
        ``(centroid_xy, weight_px)`` tuples, one per 8-connected group;
        ``weight_px`` counts the group's 4-adjacent (patch, channel) pixel
        pairs.
    """
    boundary = patch_mask & _dilate4(channel_mask)
    if not boundary.any():
        return []
    props, _ = im_utils.regionprops(boundary, props=['coords'], connectivity=2)
    groups = []
    for coords in props['coords']:
        group_image = np.zeros(boundary.shape, dtype=bool)
        group_image[coords[:, 0], coords[:, 1]] = True
        weight_px = _adjacent_pair_count(group_image, channel_mask)
        centroid = coords.mean(axis=0)  # (row, col)
        groups.append(([float(centroid[1]), float(centroid[0])], weight_px))
    return groups


class _ChannelGeometry:
    """Mask-space snapping index over the finalized channel network.

    Links contribute their pixels (``links['idx']``); contacts snap to the
    nearest link and then to that link's endpoint node nearest the contact.
    """

    def __init__(self, links, nodes, imshape):
        pixel_xy = []
        owners = []
        for lid, idx in zip(links['id'], links['idx']):
            flat = np.asarray(idx, dtype=int).ravel()
            rows, cols = np.unravel_index(flat, imshape)
            pixel_xy.append(np.column_stack([cols, rows]).astype(float))
            owners.append(np.full(len(flat), int(lid)))
        if not pixel_xy:
            raise ValueError("links contains no pixel indices ('idx' is empty); cannot snap patch contacts.")
        self._tree = cKDTree(np.concatenate(pixel_xy))
        self._owner = np.concatenate(owners)

        self._owner_image = np.zeros(imshape, dtype=int)
        for lid, idx in zip(links['id'], links['idx']):
            flat = np.asarray(idx, dtype=int).ravel()
            rows, cols = np.unravel_index(flat, imshape)
            self._owner_image[rows, cols] = int(lid)

        self._link_conn = {int(lid): [int(n) for n in conn] for lid, conn in zip(links['id'], links['conn'])}
        self._node_xy = {}
        for nid, idx in zip(nodes['id'], nodes['idx']):
            row, col = np.unravel_index(int(idx), imshape)
            self._node_xy[int(nid)] = np.array([col, row], dtype=float)

    def snap(self, xy):
        """Nearest channel link to a mask-space point -> ``(linkid, snapped_xy)``."""
        _, i = self._tree.query(np.asarray(xy, dtype=float))
        snapped = self._tree.data[i]
        return int(self._owner[i]), [float(snapped[0]), float(snapped[1])]

    def nearest_endpoint(self, linkid, xy):
        """Endpoint node of ``linkid`` nearest the mask-space point ``xy``."""
        xy = np.asarray(xy, dtype=float)
        return min(self._link_conn[linkid], key=lambda n: float(np.linalg.norm(self._node_xy[n] - xy)))

    def link_pair_counts(self, patch_mask):
        """4-adjacent (patch pixel, link pixel) pair counts per link id."""
        H, W = patch_mask.shape
        padded_owner = np.pad(self._owner_image, 1)
        counts = {}
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            neighbor_owner = padded_owner[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]
            touching = patch_mask & (neighbor_owner > 0)
            lids, lens = np.unique(neighbor_owner[touching], return_counts=True)
            for lid, cnt in zip(lids, lens):
                counts[int(lid)] = counts.get(int(lid), 0) + int(cnt)
        return counts


def _islands_from_mask(Imask, connectivity):
    """Build the machinery inputs: island components of ``~Imask``.

    Mirrors ``mask_utils.get_island_properties`` up to (not including)
    polygonization: components of the inverted mask via the same
    ``im_utils.regionprops`` primitive, with every border-attached component
    removed (the mainland/ocean blob).
    """
    inverted = ~np.asarray(Imask, dtype=bool)
    props, Iislands = im_utils.regionprops(inverted, props=['label'], connectivity=connectivity)
    border_labels = {int(v) for v in np.concatenate([
        Iislands[0, :], Iislands[-1, :], Iislands[:, 0], Iislands[:, -1]]) if v != 0}
    ids = [int(v) for v in props['label'] if int(v) not in border_labels]
    for label in border_labels:
        Iislands[Iislands == label] = 0
    return gpd.GeoDataFrame({'id': ids}), Iislands


def _parse_surrounding_link_ids(raw):
    """Parse ``surrounding_link_properties``' ``sur_link_ids`` output.

    The machinery stores a ``str(list)`` of link ids, or the empty string when
    loop detection returned its NaN fallback.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []
    return [int(lid) for lid in ast.literal_eval(raw)]


def _boundary_edge_attrs(weight, linkid, snapped_xy=None):
    attrs = {
        'edge_type': EdgeType.BOUNDARY,
        'interlayer': True,
        'weight': float(weight),
        'channel_linkid': int(linkid),
    }
    if snapped_xy is not None:
        attrs['snapped_xy'] = [float(snapped_xy[0]), float(snapped_xy[1])]
    return attrs


def extract_patches(links, nodes, Imask, landmass=None, closing_iterations=1,
                    connectivity=2, pixlen=1.0, pixarea=1.0,
                    min_shared_boundary_px=1, detect_islands=True, mask_date=None):
    """Extract wetland/island patches and their graph wiring from masks.

    Implements the spec's D3 rules: morphological closing on the landmass,
    patch components of (closed landmass) minus the channel mask, patch
    adjacency through shared channel banks, patch-to-channel boundary edges,
    and the island special case on the existing island machinery.

    Parameters
    ----------
    links, nodes : dict
        Finalized RivGraph channel dictionaries. Boundary edges may only
        reference finalized IDs. Island detection additionally requires the
        link widths/lengths (``wid_adj``/``len_adj``, as produced by
        ``compute_link_width_and_length``).
    Imask : np.array
        Binary channel mask.
    landmass : np.array, optional
        Binary landmass mask. Default: complement of the channel mask.
    closing_iterations : int, optional
        Closing passes on the landmass; 0 disables. Default 1.
    connectivity : int, optional
        1 for 4-connected components, 2 for 8-connected. Default 2.
    pixlen, pixarea : float, optional
        Pixel side length / area for metric scaling. Defaults 1.0.
    min_shared_boundary_px : int, optional
        Minimum shared channel bank (pixels) for a patch-adjacency edge;
        inclusive. Default 1.
    detect_islands : bool, optional
        Run the island machinery to classify bounded patches as islands and
        wire them from the detected surrounding links. Default True.
    mask_date : str, optional
        Stored on every patch node when provided (time-varying edge sets).

    Returns
    -------
    PatchExtraction
    """
    if not network_ids_are_finalized(links, nodes):
        raise RuntimeError(
            'Channel network IDs must be finalized (finalize_network_ids) before '
            'patch extraction: boundary edges may only reference finalized IDs.')

    channel = np.asarray(Imask) != 0
    if channel.ndim != 2:
        raise ValueError('Imask must be a 2-D binary mask.')
    if landmass is None:
        landmass = ~channel
    closed = close_landmass(landmass, iterations=closing_iterations)

    Ipatch, layer_nodes = label_patches(closed, channel, pixlen=pixlen, pixarea=pixarea,
                                        connectivity=connectivity)
    if mask_date is not None:
        for spec in layer_nodes.values():
            spec['mask_date'] = mask_date

    # Island machinery: components of the channel-mask complement, then the
    # surrounding-link loop for each (existing machinery, unmodified).
    Iislands = None
    sur_by_component = {}
    if detect_islands:
        for key in ('wid_adj', 'len_adj'):
            if key not in links:
                raise RuntimeError(
                    f'Island detection requires links[{key!r}] (compute_link_width_and_length); '
                    'pass detect_islands=False for wetland-only extraction.')
        islands, Iislands = _islands_from_mask(channel, connectivity=connectivity)
        islands = mask_utils.surrounding_link_properties(links, nodes, channel, islands,
                                                         Iislands, pixlen, pixarea)
        sur_by_component = {int(row.id): _parse_surrounding_link_ids(row.sur_link_ids)
                            for row in islands.itertuples()}

    geometry = _ChannelGeometry(links, nodes, channel.shape)
    shared = shared_boundary_lengths(Ipatch, channel)

    island_labels = set()
    island_link_ids = {}
    boundary_edges = {}
    for label in sorted(layer_nodes):
        patch_mask = Ipatch == label
        component = 0
        if Iislands is not None:
            values = np.unique(Iislands[patch_mask])
            values = values[values != 0]
            # A connected patch cannot straddle complement components.
            component = int(values[0]) if len(values) else 0
        sur_links = sur_by_component.get(component, [])

        if component:
            island_labels.add(label)
            island_link_ids[label] = list(sur_links)

        if sur_links:
            # Machinery wiring: the surrounding links map directly to boundary
            # edge pairs, one per link, at the link's nearest endpoint node.
            centroid = layer_nodes[label]['centroid_xy']
            counts = geometry.link_pair_counts(patch_mask)
            boundary_edges[label] = [
                (geometry.nearest_endpoint(lid, centroid),
                 _boundary_edge_attrs(counts.get(lid, 0) * pixlen, lid))
                for lid in sur_links
            ]
        else:
            # Wetlands, and islands whose loop detection fell back to NaN:
            # the mask-intersection rule. One edge per contiguous contact group.
            edges = []
            for centroid_xy, weight_px in _patch_contact_groups(patch_mask, channel):
                lid, snapped_xy = geometry.snap(centroid_xy)
                node = geometry.nearest_endpoint(lid, centroid_xy)
                edges.append((node, _boundary_edge_attrs(weight_px * pixlen, lid, snapped_xy)))
            boundary_edges[label] = edges

    adjacency_edges = tuple(
        (label_a, label_b, {
            'edge_type': EdgeType.PATCH_ADJACENCY,
            'interlayer': False,
            'weight': float(pixels * pixlen),
        })
        for (label_a, label_b), pixels in sorted(shared.items())
        if pixels >= min_shared_boundary_px
    )

    return PatchExtraction(
        layer_nodes=layer_nodes,
        island_labels=frozenset(island_labels),
        boundary_edges=boundary_edges,
        adjacency_edges=adjacency_edges,
        island_link_ids=island_link_ids,
    )


def _add_bidirectional_pair(G, u, v, **attrs):
    """Store one undirected adjacency as the spec's directed-pair convention."""
    G.add_edge(u, v, **dict(attrs))
    G.add_edge(v, u, **dict(attrs))


def add_patch_layer(G, links, nodes, Imask, landmass=None, **extraction_kwargs):
    """Extract patches and attach them to a multiplex graph.

    Island patches attach as one ``LayerType.ISLAND`` layer and wetlands as
    one ``LayerType.WETLAND`` layer (islands first, both in ascending
    label-raster order, via :func:`add_landscape_layer`'s deterministic ID
    assignment); adjacency and boundary edges attach as bidirectional pairs.
    The graph is validated after attachment.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Multiplex graph from :func:`rivgraph.layers.core.build_multiplex`.
    links, nodes, Imask, landmass : 
        Passed to :func:`extract_patches` (see its contract).
    **extraction_kwargs
        Optional :func:`extract_patches` parameters (``closing_iterations``,
        ``pixlen``, ``min_shared_boundary_px``, ``detect_islands``, ...).

    Returns
    -------
    G : networkx.MultiDiGraph
        The same graph, validated after the patch layers are attached.
    """
    extraction = extract_patches(links, nodes, Imask, landmass, **extraction_kwargs)

    existing = set(G.nodes)
    island_nodes = {label: extraction.layer_nodes[label] for label in sorted(extraction.island_labels)}
    wetland_nodes = {label: spec for label, spec in sorted(extraction.layer_nodes.items())
                     if label not in extraction.island_labels}
    if island_nodes:
        add_landscape_layer(G, LayerType.ISLAND, island_nodes, [])
    if wetland_nodes:
        add_landscape_layer(G, LayerType.WETLAND, wetland_nodes, [])

    label_to_node = {G.nodes[n]['label_id']: n for n in set(G.nodes) - existing}

    for label, link_ids in extraction.island_link_ids.items():
        G.nodes[label_to_node[label]]['sur_link_ids'] = list(link_ids)
    for label_a, label_b, attrs in extraction.adjacency_edges:
        _add_bidirectional_pair(G, label_to_node[label_a], label_to_node[label_b], **attrs)
    for label, edges in extraction.boundary_edges.items():
        for channel_node, attrs in edges:
            _add_bidirectional_pair(G, label_to_node[label], channel_node, **attrs)

    validate_multiplex(G)
    return G

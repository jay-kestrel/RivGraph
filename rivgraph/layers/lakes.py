# -*- coding: utf-8 -*-
"""
Lake layer extraction (layers/lakes.py)
=======================================

Spec D2: build the lake layer of the multiplex graph from binary masks.

Rules follow the published arctic-delta recipe (Vulis et al. 2020; see the
feature spec and the arctic research findings):

- **Lake nodes** are connected components of the open-water mask minus the
  channel mask (8-connectivity, via :func:`rivgraph.im_utils.regionprops`).
  Every component becomes a node -- degree-0 (never-connected) lakes are
  scientifically load-bearing, not noise.
- **Contact edges** group lake-boundary pixels lying within ``buffer_px`` of
  the channel mask (Euclidean distance transform, as in the island
  machinery's use of ``distance_transform_edt``). Contiguity is the published
  rule: per-pixel contacts would inflate degree and per-lake blobs would
  merge distinct entrances, so each 8-connected group of candidate pixels
  becomes exactly one contact.
- **Snapping**: each contact group snaps to the nearest finalized channel
  link (point-to-polyline); the graph edge attaches to the nearer of that
  link's two endpoint nodes, and the projected coordinate is stored as
  ``snapped_xy``. The buffer only gates which pixels are contact candidates;
  snapping always targets the nearest link.

All coordinates are in mask pixel space, expressed as ``(x=column, y=row)``
to match link/node geometries. ``pixel_size_m`` converts pixel measures to
metres exactly as ``mask_utils.get_island_properties`` scales island
properties by ``pixlen``/``pixarea``.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

import rivgraph.im_utils as iu
from rivgraph.layers.core import LayerType, add_landscape_layer
from rivgraph.ln_utils import network_ids_are_finalized


def _as_polyline(geom):
    """Link geometry as an (n, 2) array of (x, y); shapely or coord sequence."""
    if hasattr(geom, 'coords'):
        pts = np.asarray(list(geom.coords), dtype=float)
    else:
        pts = np.asarray(list(geom), dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        raise ValueError('Link geometry must be a sequence of (x, y) coordinates.')
    return pts


def _as_point(geom):
    """Node geometry as an (x, y) float pair; shapely Point or 2-sequence."""
    if hasattr(geom, 'x'):
        return np.array([geom.x, geom.y], dtype=float)
    pt = np.asarray(geom, dtype=float)
    if pt.shape != (2,):
        raise ValueError('Node geometry must be a single (x, y) coordinate.')
    return pt


def _closest_point_on_polyline(xy, polyline):
    """Distance and closest (x, y) point from ``xy`` to a polyline."""
    best_dist = None
    best_pt = None
    for a, b in zip(polyline[:-1], polyline[1:]):
        seg = b - a
        seg2 = seg @ seg
        if seg2 == 0:
            t = 0.0
        else:
            t = float(np.clip(((xy - a) @ seg) / seg2, 0.0, 1.0))
        proj = a + t * seg
        dist = float(np.hypot(*(xy - proj)))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_pt = proj
    return best_dist, best_pt


def _component_boundary(labels):
    """Boolean mask of labeled pixels with a 4-neighbor outside their blob."""
    padded = np.pad(labels, 1, mode='constant', constant_values=0)
    boundary = np.zeros(labels.shape, dtype=bool)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        neighbor = padded[1 + dr:1 + dr + labels.shape[0],
                          1 + dc:1 + dc + labels.shape[1]]
        boundary |= neighbor != labels
    return boundary & (labels > 0)


def _nearest_channel_node(nodes, node_ids, snapped_xy):
    """Endpoint node (of a link's conn pair) nearest to ``snapped_xy``.

    Ties resolve to the lowest node id so assignment is deterministic.
    """
    pos_by_id = {nid: i for i, nid in enumerate(nodes['id'])}
    best_id = None
    best_dist = None
    for nid in sorted(node_ids):
        if nid not in pos_by_id:
            raise RuntimeError(f'Link conn references node {nid}, '
                               'which is not in the nodes dictionary.')
        xy = _as_point(nodes['geometry'][pos_by_id[nid]])
        dist = float(np.hypot(xy[0] - snapped_xy[0], xy[1] - snapped_xy[1]))
        if best_dist is None or dist < best_dist:
            best_id = nid
            best_dist = dist
    return best_id


def _snap_group(group_xy, links):
    """Nearest link to a contact-group centroid -> (link position, snapped_xy)."""
    best = None
    for pos, geom in enumerate(links['geometry']):
        polyline = _as_polyline(geom)
        dist, snapped = _closest_point_on_polyline(group_xy, polyline)
        if best is None or dist < best[0]:
            best = (dist, pos, snapped)
    if best is None:
        raise RuntimeError('links must contain geometry for contact snapping.')
    return best[1], [float(best[2][0]), float(best[2][1])]


def extract_lakes(water_mask, channel_mask, links, nodes, buffer_px=1,
                  pixel_size_m=1.0, mask_date=None):
    """Extract lake nodes and channel contacts from binary masks.

    Implements the spec D2 rule set: connected components of the open-water
    mask minus the channel mask become lake nodes; contiguous groups of
    lake-boundary pixels within ``buffer_px`` of the channel mask become one
    contact each, snapped to the nearest finalized channel link. Degree-0
    lakes are kept as nodes.

    Parameters
    ----------
    water_mask : array-like
        Binary open-water mask (channel may or may not be included).
    channel_mask : array-like
        Binary channel mask, same shape as ``water_mask``.
    links : dict
        Finalized RivGraph links dictionary; requires ``id``, ``conn``, and
        ``geometry`` (one polyline per link, in the mask's pixel space as
        (x=column, y=row)).
    nodes : dict
        Finalized RivGraph nodes dictionary; requires ``id``, ``conn``, and
        ``geometry`` (one point per node, same coordinate space).
    buffer_px : float, optional
        Maximum Euclidean distance in pixels between a lake-boundary pixel
        and the channel mask for a contact candidate (default 1, i.e.
        orthogonally adjacent; diagonal-only adjacency is sqrt(2) away).
    pixel_size_m : float, optional
        Pixel edge length in metres for area/perimeter conversion
        (default 1). Follows the ``pixlen``/``pixarea`` convention of
        ``mask_utils.get_island_properties``.
    mask_date : str, optional
        Optional stamp written to lake nodes and contact edges (supports
        time-varying edge sets rebuilt per mask date).

    Returns
    -------
    layer_nodes : dict
        Maps label-raster id to a feature spec dict compatible with
        :func:`rivgraph.layers.core.add_landscape_layer` (``centroid_xy``,
        ``area_m2``, ``perimeter_m``, ``mask_date``).
    layer_edges : list
        ``(label_id, channel_node_id, attrs)`` contact tuples; ``attrs``
        carry ``weight`` (contact pixel count), ``buffer_px``, ``snapped_xy``,
        ``snapped_linkid``, and ``mask_date`` when stamped.

    Raises
    ------
    RuntimeError
        If the network IDs are not finalized, or geometry is missing.
    ValueError
        If the masks mismatch in shape or ``buffer_px`` is negative.
    """
    if not network_ids_are_finalized(links, nodes):
        raise RuntimeError('Channel network IDs must be finalized '
                           '(finalize_network_ids) before lake extraction: '
                           'contacts may only reference finalized IDs.')
    water = np.asarray(water_mask, dtype=bool)
    channel = np.asarray(channel_mask, dtype=bool)
    if water.shape != channel.shape:
        raise ValueError(f'water_mask and channel_mask must share one shape; '
                         f'got {water.shape} and {channel.shape}.')
    if buffer_px < 0:
        raise ValueError('buffer_px must be >= 0.')
    if 'geometry' not in links or len(links['geometry']) != len(links['id']):
        raise RuntimeError('links must carry one geometry per link for '
                           'contact snapping.')

    lake_mask = water & ~channel
    feats, lake_labels = iu.regionprops(lake_mask,
                                        props=['label', 'area', 'perimeter', 'centroid'],
                                        connectivity=2)
    layer_nodes = {}
    for i, lbl in enumerate(feats['label']):
        row_c, col_c = feats['centroid'][i]
        layer_nodes[int(lbl)] = {
            'centroid_xy': [float(col_c), float(row_c)],
            'area_m2': float(feats['area'][i]) * pixel_size_m ** 2,
            'perimeter_m': float(feats['perimeter'][i]) * pixel_size_m,
            'mask_date': mask_date,
        }

    # Contacts: boundary pixels of each lake component within buffer_px of
    # the channel mask. The channel mask itself is the EDT background, so a
    # channel-adjacent boundary pixel sits at distance 1.
    dist_to_channel = distance_transform_edt(~channel)
    candidates = _component_boundary(lake_labels) & (dist_to_channel <= buffer_px)

    groups, group_label_img = iu.regionprops(candidates,
                                             props=['label', 'area', 'centroid'],
                                             connectivity=2)
    layer_edges = []
    for g, glbl in enumerate(groups['label']):
        group_mask = group_label_img == int(glbl)
        lake_lbls = np.unique(lake_labels[group_mask])
        if lake_lbls.size != 1:
            # Cannot happen for 8-connected components, but refuse a silent
            # merge of two lakes into one contact if labeling ever changes.
            raise ValueError(f'Contact group {int(glbl)} spans lake labels '
                             f'{lake_lbls.tolist()}; groups must belong to one lake.')
        grow, gcol = groups['centroid'][g]
        link_pos, snapped_xy = _snap_group(np.array([gcol, grow], dtype=float), links)
        endpoint = _nearest_channel_node(nodes, links['conn'][link_pos], snapped_xy)

        attrs = {
            'weight': float(groups['area'][g]),
            'buffer_px': buffer_px,
            'snapped_xy': snapped_xy,
            'snapped_linkid': links['id'][link_pos],
        }
        if mask_date is not None:
            attrs['mask_date'] = mask_date
        layer_edges.append((int(lake_lbls[0]), endpoint, attrs))

    return layer_nodes, layer_edges


def add_lake_layer(G, water_mask, channel_mask, links, nodes, buffer_px=1,
                   pixel_size_m=1.0, mask_date=None):
    """Extract the lake layer from masks and attach it to a multiplex graph.

    Thin wrapper pairing :func:`extract_lakes` with
    :func:`rivgraph.layers.core.add_landscape_layer`, which assigns the
    deterministic landscape ids and stores each contact as a symmetric
    bidirectional edge pair.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        Multiplex graph from :func:`rivgraph.layers.core.build_multiplex`.
    water_mask, channel_mask : array-like
        Binary masks, as in :func:`extract_lakes`.
    links, nodes : dict
        Finalized channel links/nodes dictionaries (the same objects the
        graph was built from), used for contact snapping.
    buffer_px, pixel_size_m, mask_date : optional
        Passed through to :func:`extract_lakes`.

    Returns
    -------
    networkx.MultiDiGraph
        The same graph with the lake layer attached and re-validated.
    """
    layer_nodes, layer_edges = extract_lakes(water_mask, channel_mask, links, nodes,
                                             buffer_px=buffer_px,
                                             pixel_size_m=pixel_size_m,
                                             mask_date=mask_date)
    return add_landscape_layer(G, LayerType.LAKE, layer_nodes, layer_edges)

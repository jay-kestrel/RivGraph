# -*- coding: utf-8 -*-
"""
Landscape layers (layers package)
=================================

Landscape-layer extensions for RivGraph channel networks: lakes, wetlands,
and islands attached to the channel graph as one flat multiplex
``MultiDiGraph``. See the feature spec for the layer rules and acceptance
criteria.

D1 ships the multiplex core. Lake extraction (D2), patch extraction (D3),
layer metrics (D4), and node-link I/O (D5) extend this package in sibling
PRs -- keep exports here additive.
"""
from rivgraph.layers.core import (
    EdgeType,
    LayerType,
    NodeType,
    add_landscape_layer,
    build_multiplex,
    channel_view,
    validate_multiplex,
)
from rivgraph.layers.io import (
    multiplex_from_node_link_data,
    multiplex_to_node_link_data,
    read_multiplex_json,
    write_multiplex_json,
)

__all__ = [
    'EdgeType',
    'LayerType',
    'NodeType',
    'add_landscape_layer',
    'build_multiplex',
    'channel_view',
    'multiplex_from_node_link_data',
    'multiplex_to_node_link_data',
    'read_multiplex_json',
    'validate_multiplex',
    'write_multiplex_json',
]

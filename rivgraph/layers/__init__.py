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

__all__ = [
    'EdgeType',
    'LayerType',
    'NodeType',
    'add_landscape_layer',
    'build_multiplex',
    'channel_view',
    'validate_multiplex',
]

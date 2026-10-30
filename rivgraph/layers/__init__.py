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
from rivgraph.layers.lakes import (
    add_lake_layer,
    extract_lakes,
)
from rivgraph.layers.patches import (
    PatchExtraction,
    add_patch_layer,
    close_landmass,
    extract_patches,
    label_patches,
    shared_boundary_lengths,
)

__all__ = [
    'EdgeType',
    'LayerType',
    'NodeType',
    'PatchExtraction',
    'add_lake_layer',
    'add_landscape_layer',
    'add_patch_layer',
    'build_multiplex',
    'channel_view',
    'close_landmass',
    'extract_lakes',
    'extract_patches',
    'label_patches',
    'multiplex_from_node_link_data',
    'multiplex_to_node_link_data',
    'read_multiplex_json',
    'shared_boundary_lengths',
    'validate_multiplex',
    'write_multiplex_json',
]

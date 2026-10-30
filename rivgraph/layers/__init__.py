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
from rivgraph.layers.metrics import (
    MAX_PATHWAY_K,
    adjacency_degree,
    boundary_degree,
    boundary_weight,
    contact_counts_per_link,
    contact_degree,
    contact_degree_distribution,
    cross_layer_components,
    distance_to_channel,
    isolated_lakes,
    landscape_nodes,
    patch_degree,
    patch_importance,
    pathway_counts,
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
    'MAX_PATHWAY_K',
    'EdgeType',
    'LayerType',
    'NodeType',
    'PatchExtraction',
    'add_lake_layer',
    'add_landscape_layer',
    'add_patch_layer',
    'adjacency_degree',
    'boundary_degree',
    'boundary_weight',
    'build_multiplex',
    'channel_view',
    'close_landmass',
    'contact_counts_per_link',
    'contact_degree',
    'contact_degree_distribution',
    'cross_layer_components',
    'distance_to_channel',
    'extract_lakes',
    'extract_patches',
    'isolated_lakes',
    'label_patches',
    'landscape_nodes',
    'multiplex_from_node_link_data',
    'multiplex_to_node_link_data',
    'patch_degree',
    'patch_importance',
    'pathway_counts',
    'read_multiplex_json',
    'shared_boundary_lengths',
    'validate_multiplex',
    'write_multiplex_json',
]

"""Vessel scenario: the channel-network machinery on non-river imagery.

The mouse-brain demo (``examples/mouse_brain_example.py``) leaves the
``delta``/``river`` classes behind and drives the underlying functions
directly; this scenario follows that documented path on the committed scan:
binarize, keep the largest blob, fill holes, skeletonize, resolve the graph,
and measure link widths and lengths. The mask is un-georeferenced, so
RivGraph assigns default georeferencing and the widths are pixel widths.
The example's manual (Paint.NET) mask cleanup and its optional dilate/erode
smoothing are not reproducible steps, so they are intentionally excluded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenarios import datasets, runner

# Mask-preparation phases touch image space (extends the runner's map per
# its contract); the graph stages below are already mapped to links/s.
runner.PHASE_THROUGHPUT_UNITS.update(
    {
        "binarize": "pixels",
        "largest_blobs": "pixels",
        "fill_holes": "pixels",
    }
)

# Binarization threshold for the grayscale scan: the mouse-brain example's
# documented value ("pixel values less than 60 captures the blood vessels
# fairly well").
_VESSEL_THRESHOLD = 60

# Fill holes up to this many pixels (mouse_brain_example.py).
_MAX_HOLE_SIZE = 100


def _binarize(path: str) -> np.ndarray:
    """Read the grayscale scan and threshold it into a vessel mask."""
    from skimage import io

    image = io.imread(path, as_gray=True)
    return image < _VESSEL_THRESHOLD


def _vessel_outputs(links, nodes, **extra) -> dict:
    """Processed-network summary for the un-georeferenced mask.

    Default georeferencing gives each pixel a length of 1, so the widths
    and lengths ``link_widths_and_lengths`` computes are pixel values
    directly.
    """
    widths_px = [float(width) for width in links["wid_adj"]]
    outputs = {
        "n_links": len(links["id"]),
        "n_nodes": len(nodes["id"]),
        "total_channel_length_px": sum(float(length) for length in links["len"]),
        "width_mean_px": sum(widths_px) / len(widths_px),
        "width_max_px": max(widths_px),
    }
    outputs.update(extra)
    return outputs


@runner.register_scenario(
    name="brain_vessel_network",
    description=(
        "Vessel-network extraction on the un-georeferenced mouse-brain "
        "PNG: binarize, skeleton, network, and link widths"
    ),
    dataset="mouse_brain",
)
def brain_vessel_network(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("mouse_brain")
    with runner.scenario("brain_vessel_network", dataset, results_dir) as sc:
        from scipy.ndimage import distance_transform_edt
        from rivgraph import im_utils as iu
        from rivgraph import ln_utils as lnu
        from rivgraph import mask_to_graph as m2g

        mask = sc.phase("binarize", _binarize, str(dataset.mask_path))
        mask = sc.phase("largest_blobs", iu.largest_blobs, mask, action="keep")
        mask = sc.phase(
            "fill_holes", iu.fill_holes, mask, maxholesize=_MAX_HOLE_SIZE
        )
        Iskel = sc.phase("skeletonize", m2g.skeletonize_mask, mask)
        links, nodes = sc.phase("compute_network", m2g.skel_to_graph, Iskel)
        # Link-stage throughput counts the network the phases operate on.
        sc.count_links = lambda: len(links["id"])
        Idt = sc.phase("distance_transform", distance_transform_edt, mask)
        links = sc.phase(
            "link_widths_and_lengths", lnu.link_widths_and_lengths, links, Idt
        )
        sc.record_outputs(
            **_vessel_outputs(
                links,
                nodes,
                n_skeleton_pixels=int(np.count_nonzero(Iskel)),
            )
        )
    return sc.result

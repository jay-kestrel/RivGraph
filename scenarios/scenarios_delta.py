"""Delta scenarios: the Colville pipeline and its analysis-stage isolations.

Five scenarios over the committed Colville mask (Blueprint rows 1-4 and 9):
the full documented pipeline end to end, two analysis isolations (delta
metrics, steady-state fluxes), a raw image-stage throughput run, and a
skeletonize scaling sweep. Stage order and parameters follow
``examples/delta_example.py``; Colville's multiple inlets get the example's
explicit ``inlet='equal'`` policy (multi-inlet networks refuse implicit ones).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenarios import datasets, runner

# Delta-family phase names and the resource each actually touches: graph
# stages are attributed links/s, the mask-stage island pass pixels/s (extends
# the runner's map per its contract; per-scale skeletonize phases carry their
# own px/s in the scaling rows because scaled masks change the pixel count).
runner.PHASE_THROUGHPUT_UNITS.update(
    {
        "assign_flow_directions": "links",
        "junction_angles": "links",
        "islands": "pixels",
        "topo_metrics": "links",
        "delta_metrics": "links",
        "steady_state_fluxes": "links",
    }
)

# The nonlinear-entropy-rate family: the delta paper's headline metric set,
# isolated here for the metrics-only speed comparison against the full suite.
_NER_METRICS = ("nonlin_entropy_rate", "nER_prob_exceedence", "nER_randomized")

# Randomization trials (Blueprint: n_random=25 for the demo records).
_N_RANDOM = 25

_SCALES = (0.5, 0.75, 1.0)


def _instantiate(sc: runner.ScenarioContext, dataset: datasets.Dataset):
    """Instantiate the delta on the committed mask and wire link counting."""
    from rivgraph.classes import delta  # lazy: keeps --list light

    d = sc.phase(
        "instantiate",
        delta,
        "Colville",
        str(dataset.mask_path),
        results_folder=str(sc.work_dir),
    )
    # Link-stage throughput counts the network the phases operate on.
    sc.count_links = lambda: len(d.links["id"])
    return d


def _through_directions(sc: runner.ScenarioContext, dataset: datasets.Dataset):
    """The shared delta spine: instantiate through assign_flow_directions."""
    d = _instantiate(sc, dataset)
    sc.phase("skeletonize", d.skeletonize)
    sc.phase("compute_network", d.compute_network)
    sc.phase(
        "prune_network",
        d.prune_network,
        path_shoreline=str(dataset.aux["shoreline"]),
        path_inletnodes=str(dataset.aux["inlet_nodes"]),
    )
    sc.phase("link_widths_and_lengths", d.compute_link_width_and_length)
    sc.phase("assign_flow_directions", d.assign_flow_directions)
    return d


def _network_outputs(d, **extra) -> dict:
    """Processed-network summary: counts plus pixel-space width/length stats.

    Widths and lengths are in mask CRS units; pixel space is their ratio
    against the pixel length (pixlen == 1 for unprojected masks).
    """
    widths_px = [width / d.pixlen for width in d.links["wid_adj"]]
    outputs = {
        "n_links": len(d.links["id"]),
        "n_nodes": len(d.nodes["id"]),
        "n_inlets": len(d.nodes["inlets"]),
        "n_outlets": len(d.nodes["outlets"]),
        "total_channel_length_px": sum(d.links["len"]) / d.pixlen,
        "width_mean_px": sum(widths_px) / len(widths_px),
        "width_max_px": max(widths_px),
    }
    outputs.update(extra)
    return outputs


def _topo_subset(topo: dict) -> dict:
    """Scalar digest of the topologic metrics (arrays become means)."""
    subset: dict = {}
    if "nonlin_entropy_rate" in topo:
        subset["nonlin_entropy_rate"] = float(topo["nonlin_entropy_rate"])
    if "nER_prob_exceedence" in topo:
        subset["nER_prob_exceedence"] = float(topo["nER_prob_exceedence"])
    if "nER_randomized" in topo:
        subset["nER_randomized_mean"] = float(np.mean(topo["nER_randomized"]))
    if "n_alt_paths" in topo:
        paths = np.asarray(topo["n_alt_paths"], dtype=float)
        if paths.size:
            subset["n_alt_paths_mean"] = float(np.nanmean(paths[:, 1]))
    if "resistance_distance" in topo:
        subset["resistance_distance_mean"] = float(
            np.nanmean(np.asarray(topo["resistance_distance"], dtype=float))
        )
    return subset


def _flux_summary(links, nodes) -> dict:
    """Steady-state flux totals plus the split arriving at each outlet.

    ``compute_steady_state_link_fluxes`` returns the links dict with the
    ``flux_ss`` column attached rather than mutating its input, so callers
    pass the returned dict. ``nodes['conn']`` is parallel to ``nodes['id']``
    and lists the ids of the links touching each node; a directed outlet
    receives flow through them.
    """
    fluxes = [float(flux) for flux in links["flux_ss"]]
    link_ids = list(links["id"])
    node_ids = list(nodes["id"])
    outlet_split = {}
    for outlet_id in nodes["outlets"]:
        touching = set(nodes["conn"][node_ids.index(outlet_id)])
        outlet_split[str(outlet_id)] = sum(
            flux for link_id, flux in zip(link_ids, fluxes)
            if link_id in touching
        )
    return {
        "flux_total": sum(fluxes),
        "flux_min": min(fluxes),
        "flux_max": max(fluxes),
        "outlet_split": outlet_split,
    }


@runner.register_scenario(
    name="delta_colville_full",
    description=(
        "Full delta pipeline on the Colville mask: skeleton, network, prune, "
        "widths, flow directions, junction angles, islands, topologic "
        "metrics, and steady-state fluxes"
    ),
    dataset="colville",
)
def delta_colville_full(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("colville")
    with runner.scenario("delta_colville_full", dataset, results_dir) as sc:
        from rivgraph.deltas.delta_metrics import (
            compute_steady_state_link_fluxes,
        )

        d = _through_directions(sc, dataset)
        sc.phase("junction_angles", d.compute_junction_angles)
        islands = sc.phase("islands", d.get_islands)
        topo = sc.phase(
            "topo_metrics",
            d.compute_topologic_metrics,
            n_random=_N_RANDOM,
            inlet="equal",
            # The one-time experimental notice is asserted by the delta
            # metrics unit tests; the scenario measures compute, not warnings.
            warn_experimental=False,
        )
        flux_links = sc.phase(
            "steady_state_fluxes",
            compute_steady_state_link_fluxes,
            None,
            d.links,
            d.nodes,
            inlet="equal",
        )
        sc.record_outputs(
            **_network_outputs(
                d,
                n_islands=len(islands[0]),
                topo=_topo_subset(topo),
                flux=_flux_summary(flux_links, d.nodes),
            )
        )
    return sc.result


@runner.register_scenario(
    name="delta_colville_metrics_only",
    description=(
        "Isolates the delta analysis stage: the shared spine through flow "
        "directions, then compute_delta_metrics restricted to the "
        "nonlinear-entropy-rate family"
    ),
    dataset="colville",
)
def delta_colville_metrics_only(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("colville")
    with runner.scenario(
        "delta_colville_metrics_only", dataset, results_dir
    ) as sc:
        from rivgraph.deltas.delta_metrics import compute_delta_metrics

        d = _through_directions(sc, dataset)
        metrics = sc.phase(
            "delta_metrics",
            compute_delta_metrics,
            d.links,
            d.nodes,
            metrics=_NER_METRICS,
            n_random=_N_RANDOM,
            inlet="equal",
            warn_experimental=False,
        )
        sc.record_outputs(
            **_network_outputs(
                d,
                metrics_subset=list(_NER_METRICS),
                topo=_topo_subset(metrics),
            )
        )
    return sc.result


@runner.register_scenario(
    name="delta_colville_fluxes",
    description=(
        "Isolates the flux engine: the shared spine through flow directions, "
        "then width-routed steady-state link fluxes"
    ),
    dataset="colville",
)
def delta_colville_fluxes(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("colville")
    with runner.scenario("delta_colville_fluxes", dataset, results_dir) as sc:
        from rivgraph.deltas.delta_metrics import (
            compute_steady_state_link_fluxes,
        )

        d = _through_directions(sc, dataset)
        flux_links = sc.phase(
            "steady_state_fluxes",
            compute_steady_state_link_fluxes,
            None,
            d.links,
            d.nodes,
            inlet="equal",
        )
        sc.record_outputs(
            **_network_outputs(d, flux=_flux_summary(flux_links, d.nodes))
        )
    return sc.result


@runner.register_scenario(
    name="delta_colville_skeletonize",
    description=(
        "Raw image-stage throughput at full Colville mask scale: "
        "instantiate, skeletonize, distance transform"
    ),
    dataset="colville",
    quick=True,
    reports_network=False,
)
def delta_colville_skeletonize(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("colville")
    with runner.scenario(
        "delta_colville_skeletonize", dataset, results_dir
    ) as sc:
        d = _instantiate(sc, dataset)
        sc.phase("skeletonize", d.skeletonize)
        sc.phase("distance_transform", d.compute_distance_transform)
        n_skeleton_pixels = int(np.count_nonzero(d.Iskel))
        sc.record_outputs(
            n_skeleton_pixels=n_skeleton_pixels,
            skeleton_fraction=n_skeleton_pixels / sc.dataset["pixels"],
        )
    return sc.result


def _scaled_mask(dataset: datasets.Dataset, scale: float, out_path: Path) -> tuple:
    """Write the mask on a rescaled pixel grid; returns the binary shape.

    Pixel size scales by 1/scale so the georeferencing stays consistent with
    the original mask.
    """
    import rasterio
    from rasterio.transform import Affine
    from skimage.transform import rescale

    with rasterio.open(dataset.mask_path) as src:
        mask = src.read(1)
        profile = src.profile.copy()
        crs = src.crs
        transform = src.transform

    rescaled = rescale(
        mask.astype(np.float64),
        scale,
        order=1,
        anti_aliasing=False,
        preserve_range=True,
    )
    binary = (rescaled > 0.5).astype("uint8")

    profile.update(
        height=binary.shape[0],
        width=binary.shape[1],
        transform=transform * Affine.scale(1.0 / scale, 1.0 / scale),
        crs=crs,
        dtype="uint8",
        count=1,
        nodata=None,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(binary, 1)
    return binary.shape


def _skeletonize_count(inst) -> int:
    """Skeletonize and report the resulting skeleton pixel count."""
    inst.skeletonize()
    return int(np.count_nonzero(inst.Iskel))


@runner.register_scenario(
    name="delta_colville_scaling",
    description=(
        "Skeletonize the Colville mask at 0.5x, 0.75x, and 1.0x pixel "
        "scales: the efficiency curve of seconds and px/s vs pixels"
    ),
    dataset="colville",
    reports_network=False,
)
def delta_colville_scaling(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("colville")
    with runner.scenario("delta_colville_scaling", dataset, results_dir) as sc:
        from rivgraph.classes import delta

        scale_rows = []
        for scale in _SCALES:
            scaled_path = sc.work_dir / f"colville_mask_{scale}x.tif"
            sc.phase(
                f"prepare_{scale}x", _scaled_mask, dataset, scale, scaled_path
            )
            d = sc.phase(
                f"instantiate_{scale}x",
                delta,
                f"Colville_{scale}x",
                str(scaled_path),
                results_folder=str(sc.work_dir),
            )
            stats = sc.phase(
                f"skeletonize_{scale}x", _skeletonize_count, d
            )
            # The phase just appended is this skeletonize call; its seconds
            # feed the row's px/s (scaled masks change the pixel count, so
            # the dataset-level pixels/s attribution would be wrong here).
            seconds = sc.phases[-1].seconds
            rows, cols = int(d.Iskel.shape[0]), int(d.Iskel.shape[1])
            pixels = rows * cols
            scale_rows.append(
                {
                    "scale": scale,
                    "rows": rows,
                    "cols": cols,
                    "pixels": pixels,
                    "seconds": round(seconds, 4),
                    "pixels_per_s": pixels / seconds if seconds > 0 else None,
                    "n_skeleton_pixels": int(stats),
                }
            )
        sc.record_outputs(n_scales=len(scale_rows), scale_rows=scale_rows)
    return sc.result

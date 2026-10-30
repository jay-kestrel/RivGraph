"""River scenarios: the meander smoke, Brahmaputra full pipeline, and mesh sweep.

Three scenarios over the two committed river masks (Blueprint rows 5-7): the
fastest end-to-end smoke on the Meandering River, the full documented
braided-river pipeline on the Brahmaputra, and a mesh-sensitivity sweep at
two grid spacings. Stage order and parameters follow
``examples/braided_river_example.py``; the Brahmaputra enters from the North
and exits to the South (the example's explicit ``exit_sides``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scenarios import datasets, runner


def _through_widths(sc: runner.ScenarioContext, dataset: datasets.Dataset,
                    name: str):
    """The shared river spine: instantiate through link widths and lengths.

    Phase names and order follow ``examples/braided_river_example.py``
    (steps 1-5); pruning preserves the inlet/outlet links that the exit
    sides imply.
    """
    from rivgraph.classes import river  # lazy: keeps --list light

    d = sc.phase(
        "instantiate",
        river,
        name,
        str(dataset.mask_path),
        results_folder=str(sc.work_dir),
        exit_sides=dataset.exit_sides,
    )
    # Link-stage throughput counts the network the phases operate on.
    sc.count_links = lambda: len(d.links["id"])
    sc.phase("skeletonize", d.skeletonize)
    sc.phase("compute_network", d.compute_network)
    sc.phase("prune_network", d.prune_network)
    sc.phase("link_widths_and_lengths", d.compute_link_width_and_length)
    return d


def _network_outputs(d, **extra) -> dict:
    """Processed-network summary: counts plus pixel-space width/length stats.

    Same record contract as the delta family. Widths and lengths are in mask
    CRS units; pixel space is their ratio against the pixel length (pixlen
    == 1 for unprojected masks).
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


def _centerline_length_px(d) -> float:
    """Length of the computed centerline, converted to pixel space."""
    x, y = d.centerline
    length_crs_units = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
    return length_crs_units / d.pixlen


@runner.register_scenario(
    name="river_meander_smoke",
    description=(
        "Fastest end-to-end river smoke: skeleton, network, prune, and "
        "widths on the Meandering River mask"
    ),
    dataset="meandering",
    quick=True,
)
def river_meander_smoke(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("meandering")
    with runner.scenario("river_meander_smoke", dataset, results_dir) as sc:
        d = _through_widths(sc, dataset, "Meandering")
        sc.record_outputs(**_network_outputs(d))
    return sc.result


@runner.register_scenario(
    name="river_brahmaputra_full",
    description=(
        "Full braided-river pipeline on the Brahmaputra mask: skeleton, "
        "network, prune, widths, centerline, mesh, and flow directions"
    ),
    dataset="brahmaputra",
)
def river_brahmaputra_full(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("brahmaputra")
    with runner.scenario(
        "river_brahmaputra_full", dataset, results_dir
    ) as sc:
        d = _through_widths(sc, dataset, "Brahmaputra")
        sc.phase("centerline", d.compute_centerline)
        sc.phase("mesh", d.compute_mesh)
        sc.phase("assign_flow_directions", d.assign_flow_directions)
        sc.record_outputs(
            **_network_outputs(
                d,
                centerline_length_px=_centerline_length_px(d),
                n_meshlines=len(d.meshlines),
                n_meshpolys=len(d.meshpolys),
            )
        )
    return sc.result


# Mesh-sensitivity sweep: the documented default spacing (one average
# channel width) and a doubled spacing. Smoothing and buffer halfwidth stay
# at the example's defaults for both meshes so the comparison isolates
# grid_spacing.
_SPACING_FACTORS = (1.0, 2.0)


@runner.register_scenario(
    name="river_brahmaputra_mesh",
    description=(
        "Mesh sensitivity on the Brahmaputra: centerline plus meshes at the "
        "default and doubled grid spacing, with transect/polygon counts and "
        "time per spacing"
    ),
    dataset="brahmaputra",
)
def river_brahmaputra_mesh(results_dir: Path) -> runner.ScenarioResult:
    dataset = datasets.get_dataset("brahmaputra")
    with runner.scenario(
        "river_brahmaputra_mesh", dataset, results_dir
    ) as sc:
        d = _through_widths(sc, dataset, "Brahmaputra")
        sc.phase("centerline", d.compute_centerline)
        mesh_rows = []
        for factor in _SPACING_FACTORS:
            # The default (None) resolves to one average channel width
            # inside compute_mesh; the doubled spacing makes the comparison
            # explicit. avg_chan_width is set by the first mesh call.
            spacing = None if factor == 1.0 else factor * d.avg_chan_width
            sc.phase(f"mesh_{factor}x", d.compute_mesh, grid_spacing=spacing)
            seconds = sc.phases[-1].seconds
            mesh_rows.append(
                {
                    "spacing_factor": factor,
                    "grid_spacing": float(d.avg_chan_width * factor),
                    "n_meshlines": len(d.meshlines),
                    "n_meshpolys": len(d.meshpolys),
                    "seconds": round(seconds, 4),
                }
            )
        sc.record_outputs(
            **_network_outputs(d, n_spacings=len(mesh_rows), mesh_rows=mesh_rows)
        )
    return sc.result

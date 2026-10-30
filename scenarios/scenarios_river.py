"""River scenarios: the meander smoke pilot (fastest end-to-end pipeline).

More river scenarios (Brahmaputra full pipeline, mesh sensitivity) register
themselves here via the same decorator.
"""
from __future__ import annotations

from pathlib import Path

from scenarios import datasets, runner


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
        from rivgraph.classes import river  # lazy: keeps --list light

        d = sc.phase(
            "instantiate",
            river,
            "Meandering",
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

        # Lengths and widths are in mask CRS units; pixel space is their
        # ratio against the pixel length (pixlen == 1 for unprojected masks).
        widths_px = [width / d.pixlen for width in d.links["wid_adj"]]
        sc.record_outputs(
            n_links=len(d.links["id"]),
            n_nodes=len(d.nodes["id"]),
            n_inlets=len(d.nodes["inlets"]),
            n_outlets=len(d.nodes["outlets"]),
            total_channel_length_px=sum(d.links["len"]) / d.pixlen,
            width_mean_px=sum(widths_px) / len(widths_px),
            width_max_px=max(widths_px),
        )
    return sc.result

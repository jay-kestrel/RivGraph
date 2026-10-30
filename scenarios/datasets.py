"""Registry of the committed example datasets the scenarios run on.

Paths point into ``examples/data`` as committed in the repo; no downloads.
Shape and CRS metadata are read from the mask files at run time (not baked
into the registry) so the records reflect the actual datasets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "examples" / "data"


@dataclass(frozen=True)
class Dataset:
    """One committed example dataset.

    ``exit_sides`` is required to instantiate RivGraph's ``river`` class and
    is None for deltas and non-georeferenced imagery. ``aux`` carries the
    ancillary shapefiles some pipelines need (e.g. the Colville shoreline and
    inlet nodes).
    """

    key: str
    name: str
    mask_path: Path
    exit_sides: str | None = None
    aux: dict[str, Path] = field(default_factory=dict)


DATASETS: dict[str, Dataset] = {
    "colville": Dataset(
        key="colville",
        name="Colville Delta",
        mask_path=DATA_ROOT / "Colville_Delta" / "Colville_mask.tif",
        aux={
            "shoreline": DATA_ROOT / "Colville_Delta" / "Colville_shoreline.shp",
            "inlet_nodes": DATA_ROOT / "Colville_Delta" / "Colville_inlet_nodes.shp",
        },
    ),
    "brahmaputra": Dataset(
        key="brahmaputra",
        name="Brahmaputra Braided River",
        mask_path=DATA_ROOT / "Brahmaputra_Braided_River" / "Brahmaputra_mask.tif",
        # Enters from the North, exits to the South (braided_river_example.py).
        exit_sides="NS",
    ),
    "meandering": Dataset(
        key="meandering",
        name="Meandering River",
        mask_path=DATA_ROOT / "Meandering_River" / "meander_mask.tif",
        # The channel's two free ends lie on the image's North and South
        # extremes (verified against the mask; the East/West extremes are
        # mid-channel bend apexes, not tips).
        exit_sides="NS",
    ),
    "mouse_brain": Dataset(
        key="mouse_brain",
        name="Mouse Brain Vasculature",
        mask_path=DATA_ROOT / "Mouse_Brain" / "brain_scan.png",
    ),
}


def get_dataset(key: str) -> Dataset:
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(
            f"Unknown dataset {key!r}; known datasets: {sorted(DATASETS)}"
        ) from None


def describe_dataset(dataset: Dataset) -> dict:
    """Shape/CRS metadata read from the mask file, for the record's dataset."""
    import rasterio  # lazy: keep module import and --list light

    with rasterio.open(dataset.mask_path) as src:
        rows, cols = src.height, src.width
        crs = src.crs
    return {
        "name": dataset.name,
        "mask_path": str(dataset.mask_path.relative_to(REPO_ROOT)),
        "rows": rows,
        "cols": cols,
        "pixels": rows * cols,
        # None when the mask carries no CRS (RivGraph assigns default
        # georeferencing in that case; the record reports the dataset as-is).
        "crs": crs.to_string() if crs else None,
    }

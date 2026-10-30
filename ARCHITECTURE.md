# RivGraph architecture

This document describes the as-built architecture of the RivGraph fork at
`jay-kestrel/RivGraph`: how the package is organized, how a channel-network
mask becomes a directed graph, and where the Rust native extension sits in the
Python/Rust boundary. The fork delta relative to upstream is summarized in the
[changelog](CHANGELOG.md); algorithm background lives in the in-repo Sphinx
docs (`docs/source/index.rst`) and the papers linked from the README.

## Package layout

```
rivgraph/
├── classes.py           # public façade objects: rivnetwork, delta, river, centerline
├── directionality.py    # flow-direction assignment algorithms and dispatch
├── mask_to_graph.py     # mask → skeleton → graph; native/Python backend dispatch
├── mask_utils.py        # mask preparation and cleaning
├── im_utils.py          # image/skeleton operations (scikit-image based)
├── walk.py              # Python reference skeleton walker
├── ln_utils.py          # link/node utilities, incl. remove_duplicate_links
├── geo_utils.py         # geospatial helpers (CRS/units, coordinate transforms)
├── rasters.py           # raster I/O built on rasterio: open, metadata, GeoTIFF write
├── io_utils.py          # paths, pickling, shapefile/GeoPackage I/O
├── ordered_set.py       # OrderedSet container (backing ID determinism)
├── export_schema.py     # canonical vector export schemas (rg-v1)
├── cli.py               # rivgraph CLI (env, inspect, demo); stdlib-only at import
├── __main__.py          # python -m rivgraph
├── deltas/              # delta directionality, metrics, utilities
└── rivers/              # river directionality, utilities, centerlines
rust/                    # rivgraph-native crate: lib, graph, branchpoints, image, sets
bench/                   # pytest-benchmark suite pinned to committed Colville inputs
tests/                   # pytest suite incl. regression tests, run under both backends
```

The package `__init__.py` deliberately exposes only `__version__` (currently
`1.0`). The public surface is the classes in `rivgraph.classes` plus the
submodule functions documented in the API reference
(`docs/source/apiref/index.rst`).

Packaging and versioning: the build backend is maturin (`pyproject.toml`
declares `maturin>=1.9,<2`), with the crate manifest at the repository-root
`Cargo.toml` (`rivgraph-native` 1.0.0, pyo3 0.30.1, rust-numpy 0.30.0) and
crate sources under `rust/`. Installing the package compiles the Rust
extension as `rivgraph._native`. The Python version floor is 3.12; dependencies
are pinned through `environment.yml` (conda) and mirrored in `pyproject.toml`.

## The mask → graph pipeline

1. **Mask preparation.** `mask_utils` cleans and prepares the binary channel
   mask; `rasters` (rasterio-backed) and `geo_utils` handle georeferencing,
   units, and coordinate transforms. Downsampling and cropping helpers live in
   `rasters` (`downsample_binary_geotiff`, `crop_geotiff`).
2. **Skeletonization.** `im_utils` turns the mask into a one-pixel-wide
   skeleton (scikit-image morphology).
3. **Graph resolution.** `mask_to_graph.skel_to_graph` resolves the skeleton
   into links and nodes. This is the one place with a backend dispatch: the
   Rust extension (`rivgraph._native`) by default, or the Python reference
   walker in the same module when `PYDELTARCM_NATIVE=0`. Both paths end in
   `ln_utils.remove_duplicate_links`.
4. **Directionality.** `directionality.py` assigns flow directions
   (`add_directionality_trackers` plus the `dir_*` algorithm family:
   main-channel, shortest paths, known-link angles, bridges), with
   delta-specific and river-specific behavior in `deltas/` and `rivers/`.
5. **Metrics and analysis.** Morphologic metrics (widths, lengths, junction
   angles, islands) and delta topologic/dynamic metrics
   (`deltas.delta_metrics`) run over the directed graph.
6. **Export.** `io_utils` and `export_schema` write georeferenced vectors and
   rasters; GeoPackage is the preferred vector format.

The `delta`, `river`, and `centerline` classes orchestrate these steps as the
documented user-facing workflows.

## Class hierarchy

`rivgraph.classes` defines the object layer:

- **`rivnetwork`** — the base class: mask loading, `compute_network`,
  `compute_distance_transform`, `compute_link_width_and_length`,
  `compute_junction_angles`, `get_islands`, persistence
  (`save_network`/`load_network`), `adjacency_matrix`, and geospatial export
  (`to_geovectors`, `to_geotiff`).
- **`delta(rivnetwork)`** — adds `skeletonize`, shoreline/inlet-aware
  `prune_network`, `assign_flow_directions`, `plot_fluxes`, and
  `compute_topologic_metrics`.
- **`river(rivnetwork)`** — adds `skeletonize`, `prune_network`,
  `compute_centerline`, `compute_mesh`, and `assign_flow_directions`.
- **`centerline`** — a standalone smoothing/resampling/migration utility object
  (`smooth`, `resample`, `infs`, `intersection_points`, `mig_rate_*`, plots).

Typical workflows are shown in the notebooks under `examples/` and in
`docs/source/quickstart/index.rst`.

## The Python/Rust boundary

`mask_to_graph.skel_to_graph` checks the backend switch on **every call**:
with `PYDELTARCM_NATIVE=0` it runs the unchanged Python reference walker;
otherwise it imports `rivgraph._native` and hands the skeleton (as `uint8`) to
a single Rust call covering the walk, neighborhood analysis, branchpoint
pruning, and topology updates. The extension is validated for a 2-D binary
input and raises `ValueError` otherwise.

Two properties make the boundary safe:

- **Exact parity.** Tests compare IDs, pixel sequences, connectivity, and
  disconnected-network counts between backends without canonicalization.
- **ID determinism.** The Rust side consults interpreter-side `OrderedSet`s at
  ordering-sensitive decisions instead of iterating its own hash maps, so link
  and node IDs are identical across backends.

A missing or broken extension raises its import error rather than silently
falling back. The GIL is held during the native call — the port is a latency
optimization, not a parallelism claim. Details and benchmark numbers:
`docs/source/native/index.rst`.

## Geospatial I/O and export schemas

`rasters.py` centralizes raster opening (`open_raster`, with an explicit
default-georeferencing fallback for ungeoreferenced inputs), metadata access
(the `Raster` dataclass), coordinate/affine conversions, and GeoTIFF writing
(`write_geotiff`). `io_utils.py` handles paths, pickling, and vector I/O.

`export_schema.py` is the export contract: `EXPORT_SCHEMA_VERSION` (`rg-v1`),
the canonical `RG_LINK_SCHEMA_COLUMNS` / `RG_NODE_SCHEMA_COLUMNS` /
`SWORD_NODE_SCHEMA_COLUMNS` column sets, the `VectorFormat` dataclass (with a
`preserves_native_crs` flag per format), and helpers
(`normalize_geovector_format`, `get_driver_for_path`,
`get_extension_for_format`, `ordered_export_columns`). GeoPackage is the
recommended vector format; shapefile and GeoJSON remain available with
format-specific caveats.

**Security caution.** `io_utils.unpickle_links_and_nodes` (used by
`rivnetwork.load_network`) calls `pickle.load` on caller-supplied `.pkl`
paths. Loading an untrusted pickle executes arbitrary code — only load network
files you trust. This is a property of the pickle format, documented here
rather than changed.

## Determinism and reproducibility

- Link and node IDs are kept deterministic across backends through the
  `OrderedSet` consultation described above.
- The benchmark suite pins workloads to the committed Colville inputs in
  `examples/data/` and records backend, versions, and input checksums in the
  benchmark metadata (`bench/conftest.py`).
- CI runs the full test suite twice — native default, then
  `PYDELTARCM_NATIVE=0` — so behavior changes on either side of the boundary
  are caught.

## Testing and CI

Three workflows gate every PR to `master` (all three run on PRs):

- **`tests.yml`** — Rust toolchain + conda environment from
  `environment.yml`, editable install (compiles the extension), `cargo fmt
  --check`, `cargo clippy --locked -- -D warnings`, then the pytest suite
  twice: `python -m pytest -ra` (native default) and
  `PYDELTARCM_NATIVE=0 python -m pytest tests -ra` (pure-Python reference).
- **`docs.yml`** — same environment setup, then `make -C docs html` (Sphinx).
  The Pages deploy steps are guarded to upstream's repository, so the fork
  builds docs on every PR without deploying.
- **`benchmarks.yml`** — runs the `bench/` pytest-benchmark suite and uploads
  the baseline JSON as an artifact.

The regression tests pin delta and river workflows to committed Colville and
Brahmaputra data. The fork publishes no releases: there are no tags, no GitHub
releases, and no PyPI publish workflow; the conda-forge distribution is
upstream's (`jschwenk::rivgraph`) and does not include the fork additions.

## Fork orientation

This repository is a fork of
[VeinsOfTheEarth/RivGraph](https://github.com/VeinsOfTheEarth/RivGraph). The
fork delta over upstream v1.0: the Rust native skeleton walker (PR #3), the
pytest-benchmark harness (PR #1), core-graph and geo/raster/IO hardening fixes
(PRs #2, #4, #5), and the `rivgraph` CLI (PR #6). Package metadata
(`pyproject.toml` URLs) and the conda-forge distribution still refer to
upstream; the GitHub issue tracker is disabled on the fork. Development
workflow: see [CONTRIBUTING.md](CONTRIBUTING.md).

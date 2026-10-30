# Changelog

## Unreleased

Changes carried by this fork ahead of the next release. The versions below
summarize upstream history.

### Added

- Rust native skeleton walker: `mask_to_graph.skel_to_graph` dispatches to a
  compiled `rivgraph._native` extension (PyO3 + rust-numpy, built by maturin)
  with exact Python parity (#3). The Python reference walker remains available
  via `PYDELTARCM_NATIVE=0`.
- `rivgraph` command-line interface with `env`, `inspect`, and `demo`
  subcommands, also runnable as `python -m rivgraph` (#6).
- pytest-benchmark harness covering the walker, link widths, and flow
  directionality on the committed Colville inputs, plus a paired-backend
  Brahmaputra walker benchmark (#1).

### Changed

- Backend selection is re-evaluated on every `skel_to_graph` call; a missing or
  broken native extension raises its import error instead of silently falling
  back to the Python reference (#3).
- logging output normalized: continuity and cycle events log at `warning`,
  walker traces at `debug`, swallowed exceptions gain `debug` records with
  exception context, and tests capture output via capsys plus a loguru
  fixture instead of stdout hand-swaps (#7)

### Fixed

- core graph walk and directionality defects (#4)
- `im_utils` skeleton curvature walk, branchpoint indexing, and uint16 casts (#2)
- geo/raster/IO error handling, unit parsing, and downsample registration (#5)
- network pickle loading goes through a `RestrictedUnpickler` that allowlists
  only the built-in containers, OrderedDict, OrderedSet, and numpy
  array/dtype machinery; any other global raises `pickle.UnpicklingError`
  instead of executing arbitrary code (#11)

## Version 1.0

RivGraph v1 is a major modernization release focused on a cleaner geospatial stack, stronger testing and export contracts, and a more maintainable codebase.

### Highlights

- modernized the geospatial stack around rasterio, pyogrio, pyproj, shapely, and geopandas
- removed the remaining direct GDAL/`osgeo` bindings and the OpenCV dependency from the core codebase
- standardized Python 3.12 as the supported development and release target
- restored and refreshed automated testing and documentation builds with GitHub Actions
- speedups

### Added

- canonical vector export schema and contract tests for links, nodes, and SWORD (SWOT)-style exports
- deterministic ID support for links and nodes in modern workflows
- explicit inlet/outlet tagging in geovector exports
- raster backend contract tests and geospatial round-trip tests
- improved island polygonization tests based on rasterio-backed polygon generation
- refreshed example notebooks and documentation structure for the v1 release

### Changed

- GeoPackage is now the preferred vector export format for high-fidelity workflows
- documentation and examples were simplified around a smaller set of canonical notebooks
- source installation now uses a single canonical `environment.yml`
- packaging metadata and CI workflows were aligned with Python 3.12

### Fixed

- steady-state flux handling in directed delta graphs, including stronger DAG validation and better behavior around cycles and parallel edges
- island geometry generation and related export consistency
- directionality, export, and regression coverage across both river and delta workflows
- numerous documentation, install, and example inconsistencies accumulated since v0.5
- other open Issues have been addressed

### Notes for existing users

- RivGraph v1 targets Python 3.12.
- GeoPackage is the recommended vector export format; shapefile and GeoJSON remain available with format-specific caveats.
- A few export names and workflows were cleaned up for consistency in v1, so older scripts may need small updates.

## Version 0.5.0 (2022-08-08)

### New features and improvements

- testing suite overhaul

### Bug fixes

- delta metrics

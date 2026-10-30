[![DOI](https://joss.theoj.org/papers/10.21105/joss.02952/status.svg)](https://doi.org/10.21105/joss.02952)
[![tests](https://github.com/jay-kestrel/RivGraph/actions/workflows/tests.yml/badge.svg)](https://github.com/jay-kestrel/RivGraph/actions/workflows/tests.yml)
[![docs](https://github.com/jay-kestrel/RivGraph/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/jay-kestrel/RivGraph/actions/workflows/docs.yml)
[![benchmarks](https://github.com/jay-kestrel/RivGraph/actions/workflows/benchmarks.yml/badge.svg)](https://github.com/jay-kestrel/RivGraph/actions/workflows/benchmarks.yml)

[![RivGraph logo](docs/logos/rg_logo_full.png)](docs/source/index.rst "Go to documentation sources.")

# RivGraph

RivGraph is a Python package for converting a binary mask of a channel network into a directed, weighted graph of connected links and nodes. It is designed for river and delta channel networks derived from remote sensing imagery and geospatial masks.

![Core functionality of RivGraph](examples/images/rivgraph_overview_v1.png)

## Core capabilities

- morphologic metrics such as widths, lengths, and branching characteristics
- graph-based and algebraic representations of channel networks
- topologic and dynamic metrics such as alternative paths, flux sharing, and entropy-based measures
- tools for cleaning and preparing binary channel masks
- island detection, metrics, and filtering
- mesh generation for along-river analysis
- geospatial export of links, nodes, rasters, and derived products

RivGraph preserves georeferencing information throughout the workflow. If you start with a georeferenced mask, exported rasters and vectors remain aligned with the source data and can be used directly in GIS workflows.

The flow-directionality logic and validation are described in our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html). General package usage is described in our [JOSS paper](https://joss.theoj.org/papers/10.21105/joss.02952).

## Getting started

Start with the in-repo [documentation sources](docs/source/index.rst) and the notebooks in `examples/`. The upstream hosted site at <https://veinsoftheearth.github.io/RivGraph/> describes upstream behavior only — it does not cover this fork's native walker or CLI. RivGraph contains two primary classes, `delta` and `river`, that organize the main processing workflows. These notebooks show the end-to-end usage patterns:


- `examples/delta_example.ipynb`
- `examples/braided_river_example.ipynb`
- `examples/mouse_brain_example.ipynb`

## Documentation in this repository

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — as-built architecture: package layout, the mask→graph pipeline, the class hierarchy, and the Python/Rust backend boundary
- [`docs/source/install/index.rst`](docs/source/install/index.rst) — installation, including the Rust toolchain requirement for source builds of this fork
- [`docs/source/native/index.rst`](docs/source/native/index.rst) — the native skeleton walker: backend selection, parity guarantees, determinism, and performance
- [`docs/source/cli/index.rst`](docs/source/cli/index.rst) — the `rivgraph` command-line interface (`env`, `inspect`, `demo`)
- [`docs/source/apiref/index.rst`](docs/source/apiref/index.rst) — API reference generated from in-code docstrings
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development setup, dual-backend test invocation, and conventions
- [`CHANGELOG.md`](CHANGELOG.md) — release and fork history

### Fork vs upstream

This repository is a fork of [VeinsOfTheEarth/RivGraph](https://github.com/VeinsOfTheEarth/RivGraph). Beyond the upstream v1.0 codebase, the fork adds the Rust native skeleton walker (PR #3), a pytest-benchmark harness (PR #1), core-graph and geo/raster/IO hardening fixes (PRs #2, #4, #5), the `rivgraph` CLI (PR #6), and a restricted unpickler for network pickles (PR #11); the fork history is in the [changelog](CHANGELOG.md). Package metadata (`pyproject.toml` URLs) and the conda-forge distribution still refer to upstream, and the fork publishes no wheels or releases — builds from this repository are source-only.

## Native skeleton walker (fork build)

This fork builds `rivgraph._native` with maturin, PyO3, and rust-numpy. Install
Python 3.12 and a stable Rust toolchain (including Cargo), then build from source:

```bash
pip install -e '.[test,bench]'
python -m pytest tests
python -m pytest bench/ --benchmark-json=benchmark.json
```

NumPy `>=2,<2.3` is required: all existing tests, including the real delta and
river regression workflows, passed under NumPy 2.2.6 before porting. The upper
bound preserves compatibility with the existing SciPy 1.13 dependency. This
uses typed rust-numpy array interop instead of raw pointers. There is no wheel
build matrix in this change; source installation requires Rust, even if the
Python fallback will be selected at runtime.

`mask_to_graph.skel_to_graph` uses the native walker by default. Set
`PYDELTARCM_NATIVE=0` to run the unchanged Python reference instead (the switch
name is shared with the sibling pyDeltaRCM port). It is checked on every call.
Missing/broken native extensions raise their import error rather than silently
changing backends. The reference path itself does not import the extension.
The backend contract, parity guarantees, determinism model, and benchmark
tables are documented in [`docs/source/native/index.rst`](docs/source/native/index.rst).

The complete walk, neighborhood calculations, branchpoint pruning and topology
updates run in one Rust call. Python retains public `OrderedSet` containers and
the existing final duplicate-link cleanup. The Rust code deliberately uses
interpreter sets at ordering-sensitive decisions: arbitrary Rust hash iteration
or sorted indices would change node/link IDs. This keeps the GIL held during the
phase; it is a latency optimization, not a multi-threaded throughput claim.
Branchpoint results are memoized only within one call on its owned padded copy.

Exact parity tests compare IDs, pixel sequences, connectivity and disconnected
network counts without sorting/canonicalization on seed-42 synthetic, Colville
and Brahmaputra masks. Additional fixtures cover loops, boundaries, disconnected
networks, strided/read-only arrays, invalid inputs and 30 seeded branched masks.
The paired Brahmaputra benchmark uses the same staged skeleton for both
backends, with one warm-up and five measured rounds per backend. Versions,
checksums, medians and minima are included in the benchmark output.

## Demo scenarios

The [`scenarios/`](scenarios/) package runs RivGraph end to end on the four committed example datasets and reports what each run did and what it cost. Each named scenario drives the public `delta`/`river` workflows through explicit phases and emits one structured record per run: per-phase wall time, throughput, cumulative peak RSS, and the processed data (links, nodes, channel lengths, widths, islands, fluxes — whichever the scenario produces).

```bash
python -m scenarios --list                # the catalog: name + one-line description
python -m scenarios --quick               # fast, CI-safe subset
python -m scenarios --only delta_colville_full river_brahmaputra_full
python -m scenarios                       # the whole catalog
```

Every run writes `scenario_results.json` (machine-readable, one record per scenario) and `SCENARIOS.md` (human summary) into `scenarios_results/`. The suite is pytest-executable (`tests/test_scenarios.py`): the quick subset guards CI, records are contract-tested, and full pipelines run under the `slow` marker. The `scenarios.yml` workflow runs the quick subset on master and publishes both reports for each run — see the [demo scenarios documentation page](docs/source/scenarios/index.rst) for the catalog and record schema.

Prefer [`bench/`](bench/) when you need stage-level micro-benchmarks instead: it times three hot stages (`skel_to_graph`, `link_widths_and_lengths`, `assign_flow_directions`) with pytest-benchmark's statistical rounds, which is what you want when optimizing one stage. Reach for the scenarios when you want the end-to-end story — what a whole workflow processed and what it cost.

## Output and logging

RivGraph logs through [loguru](https://loguru.readthedocs.io); library code never uses `print()`.

- **Console**: With `verbose=True` (on `rivnetwork`, `river`, or `delta`), progress messages are written to stdout as bare text -- no timestamps or level prefixes. With `verbose=False`, the classes write nothing to stdout.
- **Log file**: Every `rivnetwork`, `river`, or `delta` run writes `<name>_log.log` beside the results folder, regardless of `verbose`. Each record is timestamped: `[YYYY-MM-DD at HH:mm:ss] | <message>`.
- **Levels**: The file and console sinks apply their own level filters. Routine progress is logged at `info`; operationally important conditions -- continuity violations, unfixable cycles -- at `warning`; diagnostic traces (e.g. the skeleton walker's per-step state) at `debug`.
- **Warnings**: I/O and export conditions and experimental-metric notices go through Python's `warnings` module. They appear on stderr regardless of `verbose` and respond to the standard `warnings` filtering machinery.

## COBOL walker backend (experimental)

A GnuCOBOL port of the walker's numeric march lives in `cobol/` as an
additional backend, parallel to the Rust one. `RIVGRAPH_COBOL=1` routes
`mask_to_graph.skel_to_graph` to it and takes precedence over
`PYDELTARCM_NATIVE`; unset behavior is unchanged. Build it from the repo root
(requires `gnucobol3`, e.g. `apt install gnucobol3`):

```bash
make -C cobol
RIVGRAPH_COBOL=1 python -m pytest tests
python -m pytest bench/ --benchmark-json=benchmark.json
```

The COBOL kernel performs numeric traversal only: it marches a link path
across the skeleton and stops at decision points, while Python keeps every
set-order-sensitive decision (branchpoint clustering, emanator ordering,
first-step selection) and hands the kernel canonicalized, sorted inputs —
never randomness. Parity with the reference is exact (ordered IDs, pixel
sequences, connectivity), held by the same fixture matrix as the Rust
walker in `tests/test_cobol_walker.py`. Requesting `RIVGRAPH_COBOL=1`
without a built backend raises rather than silently switching backends.
The Brahmaputra benchmark gains a third `cobol` row; COBOL timings are
feasibility measurements, not a performance gate. The kernel is
single-threaded.

## Use of AI

RivGraph was initially released before the advent of LLMs and thus v0.5.0 was written entirely without AI assistance. However, the v1.0 release was made possible through efficiencies provided by relying heavily on AI coding (mainly OpenAI tools). 

## Contributing

Documentation improvements, bug reports, and pull requests are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md) for environment setup, dual-backend test invocation, and conventions. The GitHub issue tracker is disabled on this fork, so please raise concerns in a pull request; the upstream [issue tracker](https://github.com/VeinsOfTheEarth/RivGraph/issues) remains the right place for upstream-only behavior.

## Citing RivGraph

If you use RivGraph in your research, please consider [citing it](https://joss.theoj.org/papers/10.21105/joss.02952).

If you use RivGraph's flow-directionality algorithms, please cite our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html).

If you use RivGraph's flux paritioning scheme that asssumes steady-state, width-weighted fluxes at junctions, you should be aware of [this work](https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2022GL097897).

## License

RivGraph is distributed under the [BSD 3-clause license](LICENSE.txt). Note that `LICENSE.txt` carries a BSD-3-Clause notice preceded by a Los Alamos National Laboratory government-contract header, so GitHub detects the repository license as "Other" rather than BSD-3-Clause; the header came with the upstream repository.

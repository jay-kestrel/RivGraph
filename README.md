[![DOI](https://joss.theoj.org/papers/10.21105/joss.02952/status.svg)](https://doi.org/10.21105/joss.02952)
[![tests](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/tests.yml/badge.svg)](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/tests.yml)
[![docs](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/docs.yml/badge.svg?branch=master)](https://github.com/VeinsOfTheEarth/RivGraph/actions/workflows/docs.yml)

[![RivGraph logo](docs/logos/rg_logo_full.png)](https://VeinsOfTheEarth.github.io/RivGraph/ "Go to documentation.")

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

Start with the [documentation](https://VeinsOfTheEarth.github.io/RivGraph/) and the notebooks in `examples/`. RivGraph contains two primary classes, `delta` and `river`, that organize the main processing workflows. These notebooks show the end-to-end usage patterns:


- `examples/delta_example.ipynb`
- `examples/braided_river_example.ipynb`
- `examples/mouse_brain_example.ipynb`


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

## Use of AI

RivGraph was initially released before the advent of LLMs and thus v0.5.0 was written entirely without AI assistance. However, the v1.0 release was made possible through efficiencies provided by relying heavily on AI coding (mainly OpenAI tools). 

## Contributing

Bug reports, documentation improvements, and pull requests are welcome. The easiest way to start is to open an issue in the [tracker](https://github.com/VeinsOfTheEarth/RivGraph/issues).

## Citing RivGraph

If you use RivGraph in your research, please consider [citing it](https://joss.theoj.org/papers/10.21105/joss.02952).

If you use RivGraph's flow-directionality algorithms, please cite our [ESurf Dynamics paper](https://www.earth-surf-dynam.net/8/87/2020/esurf-8-87-2020.html).

If you use RivGraph's flux paritioning scheme that asssumes steady-state, width-weighted fluxes at junctions, you should be aware of [this work](https://agupubs.onlinelibrary.wiley.com/doi/pdfdirect/10.1029/2022GL097897).

## License

RivGraph is distributed under the [BSD 3-clause license](LICENSE.txt).

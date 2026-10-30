# Contributing to RivGraph (fork)

Thanks for contributing. This fork carries its own additions on top of
[upstream RivGraph](https://github.com/VeinsOfTheEarth/RivGraph) — a Rust
native skeleton walker, a `rivgraph` CLI, and a benchmark harness — so
development setup differs slightly from upstream. The tone-setting community
notes in [`docs/source/contributing/index.rst`](docs/source/contributing/index.rst)
still apply.

The GitHub issue tracker is disabled on this fork: please raise questions or
proposals in a pull request (or in the upstream tracker for upstream-only
behavior).

## Environment setup

1. **Rust toolchain (required).** Installing the package compiles
   `rivgraph._native` through the maturin build backend. Install a stable Rust
   toolchain (including Cargo) with [rustup](https://rustup.rs) first.
2. **Python environment.** Use the canonical conda environment:

   ```bash
   conda env create -f environment.yml
   conda activate rivgraph
   pip install -e . --no-deps   # --no-deps: deps come from conda; still compiles _native
   ```

   A pure pip/uv route works too (`pip install -e .` in a Python 3.12+
   virtualenv) — see `docs/source/install/index.rst` for both variants.
3. **Rebuilding the extension.** After changing files under `rust/`, run
   `maturin develop` (or `maturin develop --release`) instead of reinstalling.

## Running the tests

CI runs the suite under **both** backends; do the same locally:

```bash
python -m pytest -ra                                # native walker (default)
PYDELTARCM_NATIVE=0 python -m pytest tests -ra      # pure-Python reference walker
```

A bare `python -m pytest -ra` collects `tests/` and `bench/`; the benchmark
module skips itself when `pytest-benchmark` is not installed. The regression
tests (`tests/regression/`) run real delta and river workflows against the
committed Colville and Brahmaputra data.

## Rust gates

The tests workflow checks Rust style and lints on every PR:

```bash
cargo fmt --check
PYO3_PYTHON=$(which python) cargo clippy --locked -- -D warnings
```

Keep `rust/` changes exercised by the parity tests: the native walker must
produce IDs, pixel sequences, connectivity, and disconnected-network counts
identical to the Python reference (no sorting or canonicalization in the
comparison).

## Benchmarks

The benchmark suite (`bench/`, pytest-benchmark) pins its workloads to the
committed Colville inputs under `examples/data/`, mirroring the documented
delta workflow (`skeletonize → compute_network → prune_network →
compute_link_width_and_length`), plus a paired-backend Brahmaputra walker
benchmark:

```bash
pip install "pytest-benchmark>=4,<6"
mkdir -p bench_results
python -m pytest bench/ -ra \
  --benchmark-columns=rounds,min,median,mean,stddev \
  --benchmark-sort=name \
  --benchmark-json=bench_results/baseline.json
```

CI (`.github/workflows/benchmarks.yml`) runs the same suite and uploads
`bench_results/baseline.json` as an artifact.

## Documentation

The Sphinx site builds from `docs/source/`:

```bash
pip install ".[docs]"
python -m sphinx -b html docs/source docs/build/html
```

New/changed docstrings should stay numpydoc-style (rendered by
`sphinx.ext.napoleon`); the API pages are generated from them
(`docs/source/apiref/`). Narrative pages: `docs/source/index.rst` holds the
toctree — new sections must be wired there.

## Logging and output conventions

RivGraph logs through [loguru](https://loguru.readthedocs.io). To keep output
consistent and testable:

- **Log via `logger.*`, never `print()`.** Console visibility is controlled by
  the class sinks (`verbose`), not the call site.
- **Match sibling levels.** The river and delta implementations of the same
  event must log at the same level.
- **Levels**: `debug` for diagnostic detail (e.g. per-step traces), `info` for
  routine progress, `warning` for conditions the user should know about that
  do not violate a contract.
- **Contract violations go through `warnings.warn()` or exceptions**, not log
  messages. Warnings fire regardless of `verbose` and are visible to any
  embedding application; use a custom `Warning` category for conditions a
  caller may want to filter.
- When you must swallow an exception, **log it at `debug` with the original
  error** so field failures stay diagnosable.

## Conventions

- **Commits:** Conventional Commits (`feat:`, `fix:`, `docs:`, `bench:`,
  `ci:`). PRs merge as squashes — one commit per PR.
- **Branches:** cut feature branches from `master` and target `master` in PRs.
- **Backend switches:** any Python-visible behavior change must keep both
  backends passing (see the dual test invocation above).
- **Security:** network persistence is JSON-only —
  `io_utils.save_links_and_nodes` / `load_links_and_nodes` (backing
  `save_network` / `load_network`) serialize plain dicts with type-tagged
  NumPy arrays and OrderedSets to `*_network.json`, and the loader refuses
  `.pkl` paths and pickle-magic content before parsing (#9). Unpickling
  untrusted files can execute arbitrary code, so don't reintroduce pickle
  entry points (`tests/test_network_io.py` covers the round-trip and the
  malicious-pickle refusal).

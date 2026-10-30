# Julia wave-1 benchmark artifacts

Re-baseline of the existing benchmark harness at branch HEAD
(`task/rivgraph-julia-skeleton-walker-widths-1CupRBew`, Python code identical
to `master` at `59627ac`) plus the matching Julia-port measurements.

## Environment

- CPU: sandbox vCPU; Python 3.13.14, pytest-benchmark 5.3.0 (4.0.0 crashes at
  session finish on Python 3.13 — `datetime.utcnow()` DeprecationWarning is
  escalated to an error while generating the JSON artifact).
- Julia 1.13.3 (see `julia/README.md`).
- Input hashes and versions are embedded in each JSON artifact.

## Conditions

- The Python run requires `PYDELTARCM_NATIVE=0` in this sandbox: the
  Rust `rivgraph._native` extension is not built here, and the session-scoped
  Colville fixture errors at setup through the native dispatch path otherwise.
  `0` selects the repo's documented reference path, which is also the oracle
  the Julia port is gated against — the correct comparison target.
- Native (Rust) and COBOL backend cases were deselected: neither toolchain is
  installed in this sandbox.
- `test_benchmark_assign_flow_directions` FAILS at this HEAD independent of
  this branch (this PR touches no Python source): `bridge_sides`
  (`rivgraph/graph_queries.py`) assumes `nx.immediate_dominators` includes the
  root sentinel in the returned parent map and dies with
  `KeyError: <object object ...>` on the current networkx when a node is
  dominated by the super-root. Flow directionality is out of scope for wave 1
  (reported `not_ported` in the Julia artifact); left untouched per the
  additive-only constraint of this track.

## Headline (Colville fixtures, 5 rounds, 1 warmup)

| Stage | Python reference (min) | Julia port (min) | Speedup |
|---|---|---|---|
| `skel_to_graph` | 3,726 ms | 196.5 ms | ~19x |
| `link_widths_and_lengths` | 1,520 ms | 3.5 ms | ~432x |

Medians: 3,804 ms / 0.207 s (walker) and 1,543 ms / 3.7 ms (widths).
The Julia walker number includes no warm JIT advantage beyond the single
warmup round; both sides run the same committed Colville fixtures.

Files:
- `python-baseline.json` / `python-baseline.log` — pytest-benchmark artifact
  and full log of the Python reference run.
- `julia-bench.json` / `julia-bench.log` — the Julia port's measurements
  (`julia/bench/benchmarks.jl` output; same stages, rounds, and input
  hashes).

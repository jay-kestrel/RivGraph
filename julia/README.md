# RivGraph Julia port (wave 1)

Sibling Julia package porting RivGraph's core numeric kernels with
**ordered-exact parity** against the Python reference — the same gate the Rust
native walker uses (`tests/test_native_walker.py`), applied as a third backend
consumed via committed fixtures.

## Scope (wave 1)

| Kernel | Python source | Contract |
|---|---|---|
| `skel_to_graph` | `mask_to_graph.py:19` / `walk.py` | `Bool[rows,cols] -> (links, nodes)` with the +40-pixel internal pad, `idx = row*ncols + col` flat C-order, order-sensitive container semantics (`CPythonSet`/`OrderedSet` emulation included) |
| `link_widths_and_lengths!` | `ln_utils.py:452` | appends `len`, `wid_pix`, `wid`, `wid_adj`, `wid_med`, `len_adj`, `sinuosity`; Amanatides–Woo `scan_mask` with the row-branch tie-break |
| `solve_adjacency_steady_state` | `_delta_metrics_core.py:379` | FIFO Kahn propagation, 0-based `apex`/`outlets` in the result, mass-balance check at `max(atol, 1e-10)` |
| `nl_entropy_rate` | `delta_metrics.py:778` | flux-normalized entropy rate |

Deliberately **not** ported (they change pixel semantics and cascade —
see the port spec): skeletonization (`skimage.morphology`), image I/O, and
flow directionality (networkx tie-breaking).

## Numerical equivalence contract

- **Bit-exact gates** where the Python reference is deterministic: walker
  outputs, per-node steady-state fluxes (compared through their IEEE-754
  bit patterns), and transition-matrix entries at 1e-12.
- **NumPy summation is emulated, not approximated**: `np.sum` on contiguous
  vectors is reproduced bit-for-bit by `_np_sum_contiguous` (pairwise
  unrolled-8 scheme); `np.sum(A, axis=0)` by sequential row accumulation.
  Julia's own `sum` reassociates and shifts low bits — do not "simplify" the
  port back to it.
- **Tie-breaking** follows CPython semantics via the `CPythonSet` emulation
  (replayed operation traces in `test/fixtures/settraces`).

## Usage

```julia
julia --project=julia -e 'using RivGraphJulia'
```

Run the parity suite:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

Benchmarks (5 rounds + 1 warmup, Colville fixtures, JSON artifact with
environment and input hashes, mirroring the Python harness contract):

```bash
julia --project=julia julia/bench/benchmarks.jl
# writes bench/results/julia-wave1/julia-bench.json
```

## Fixtures

All oracles are dumped from the Python reference path with
`PYDELTARCM_NATIVE=0` — see `test/fixtures/README.md` for provenance and
regeneration.

## CI

`.github/workflows/julia.yml` — no third-party Actions (repo policy): the
checkout is inlined, and the pinned Julia tarball (1.13.3, SHA-256-verified)
is downloaded in-script. No Python or Rust toolchain is required to run the
Julia suite.

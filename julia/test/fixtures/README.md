# Parity fixtures — provenance and regeneration

Every fixture here is an output of the **Python reference path**
(`PYDELTARCM_NATIVE=0`; the Rust native extension disabled), dumped with
CPython 3.12.16 from the environment recorded in `manifest.json` →
`versions`. The Julia tests are ordered-exact parity gates against these
dumps; nothing in this directory is hand-written.

## Layout

- `walker/<name>/` — `Iskel.npy` (packed-bits skeleton) + `links.json` /
  `nodes.json` (the ordered `skel_to_graph` oracle) + counts, for the
  synthetic masks, the 30 `branchedNN` combinatorial masks, and the real
  Colville / Brahmaputra skeletons (from `examples/data`).
- `widths/{synthetic,colville}/` — `Idt.npy.gz` distance transform +
  `links_in.json` (pruned network before widths) + `links_out.json`
  (oracle after `link_widths_and_lengths`).
- `delta/<name>/` — `A.npy.gz` adjacency + `expected.json` (F, SubN, P,
  per-node flux bit dumps) for 8 synthetic DAG families.
- `settraces/` — replayable CPython set-order operation traces (int and
  2-tuple keys) backing the `CPythonSet` emulation tests.
- `manifest.json` — per-input dtype/shape/SHA-256, generator versions,
  and the error-case registry (malformed adjacency the solver must reject).

## Regenerating

From the repo root, with the pinned Python environment from
`manifest.json`/`versions` (set-object iteration order is
CPython-version-dependent — do not mix):

```bash
python3 julia/test/fixtures/dump_fixtures.py     # walker/widths/delta dumps
python3 julia/test/fixtures/dump_set_traces.py   # CPython set traces
```

Both scripts write to `fixtures/generated/` (never over this directory
directly). After verifying `generated/manifest.json` hashes against the
committed fixtures, copy the changed inputs over the paths above and
re-run `julia --project=julia -e 'using Pkg; Pkg.test()'`.

`dump_fixtures.py` records the `rivgraph` commit it dumped from in
`manifest.json` → `versions` → `rivgraph_commit`. The committed fixtures
were dumped at `de1555aa`; the walker/widths/delta kernels are unchanged
since (verify with `git log <sha>..HEAD -- rivgraph/` before trusting
regenerated output).

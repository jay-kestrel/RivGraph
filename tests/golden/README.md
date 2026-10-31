# RivGraph golden oracle (spec PF-1, gate CHK-6)

One golden fixture for the RivGraph port effort, per the shared
port-fidelity and oracle spec (`PF-1` RivGraph row; gate `CHK-6`;
tolerances `PF-3`). RivGraph's graph-extraction core is RNG-free, so the
oracle is a frozen mask/geometry input set plus expected graph outputs --
no RNG vectors are needed for this repo.

## What is pinned

- **Reference commit:** the commit recorded in the fixture header
  (`reference_commit`). The mint ran the pipeline at that commit; this
  branch adds only `tests/golden/`, so the pipeline source is identical to
  master (`git diff <reference_commit>..HEAD -- rivgraph/` is empty).
- **Input dataset:** the `river_brahma_clipped` regression case
  (`tests/regression/data/river_brahma_clipped/inputs/`), frozen by sha256
  in the fixture header.
- **Environment:** Python and package versions recorded in the fixture
  header (`environment`). A dependency bump shifts the fixture digest and
  forces an explicit re-baseline.

**Why this dataset:** the brahma_clipped mask exercises the full river
pipeline -- skeletonization, pruning, link width/length, and directionality
over a braided network with cycles (1195 links, 794 nodes, 79 parallel link
pairs, bridge logic in the directionality path). It is the vetted
end-to-end case the existing river regression test already runs, so the
golden fixture anchors the same code a port must reproduce. The delta case
(`delta_mossy`) was rejected as the oracle because its tiny network
exercises far less graph topology.

## Files

- `golden.py` -- pure core: the reference-pipeline driver, fixture
  extraction/serialization, and the CHK-6 comparator.
- `mint_golden.py` -- the mint script (see below).
- `data/rivgraph_golden.json` -- the fixture (~559 KB, committed whole: it
  is under the spec's "few MB" storage bar, so the pygeoflood
  regenerate-in-CI rule does not apply).
- `data/rivgraph_golden.sha256` -- the pinned fixture digest (manifest).
- `../../test_golden_equivalence.py` -- the CHK-6 pytest gate.

## Mint procedure

From the repo root with the pinned environment active:

```bash
python -m tests.golden.mint_golden          # mint: two pipeline runs, byte-identical or hard-fail
python -m tests.golden.mint_golden --check  # verify a re-mint reproduces the committed digest
```

Byte-reproducibility enforcement (the spec's `26b3e64` pattern): the mint
runs the reference pipeline twice and hard-fails unless both serializations
are byte-identical. `--check` re-mints once and compares against the
committed manifest.

The pipeline runs the **pure-Python reference** (`PYDELTARCM_NATIVE=0`,
`RIVGRAPH_COBOL` cleared) -- the fixture binds to the reference
implementation, never to a backend build.

## CHK-6 gate semantics

`compare_graphs` implements the spec's value mode for graph outputs:

- **Tier A -- exact:** node/link id sequences, connectivity (`conn`
  node pairs per link; node `conn` incident links), topology
  (`link_conn`), inlets/outlets, directionality decisions (`guess`,
  `guess_alg`, `certain`, `certain_alg`, `certain_order`), parallels,
  `n_networks`, and finalization metadata.
- **Tier B -- PF-3 defaults:** link lengths (`len`, `len_adj`), widths
  (`wid`, `wid_adj`, `wid_med`), `sinuosity`, junction angles
  (`clangs`, `cldists`), and `maxang` within 1e-9 relative with a 1e-12
  absolute floor (`|a-b| <= max(1e-12, 1e-9 * max(|a|, |b|))`). NaN equals
  NaN; NaN vs number is a mismatch.
- First mismatch is reported with field and index, so a red gate points at
  the diverging element (the spec's triage entry point).

Excluded from the fixture (recorded in its `excluded_fields` header):
`links['idx']` per-link pixel chains and `links['wid_pix']` per-pixel width
diagnostics -- CHK-6 pins graph structure and attributes, not pixel
geometry, and excluding them keeps the fixture committable.

## Re-baselining

A digest may only change through an explicit re-baseline commit carrying a
recorded reason:

```bash
python -m tests.golden.mint_golden --rebaseline --reason "<why>"
```

Legitimate reasons: a pipeline behavior change (then the gate is re-pinned
to the new behavior), or a pinned-environment change. Speedups, refactors,
and backend changes that preserve the digest need nothing.

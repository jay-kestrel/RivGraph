.. _native:

======================
Native skeleton walker
======================

The fork builds a Rust implementation of the skeleton walker and ships it as the
compiled extension ``rivgraph._native`` (PyO3 + rust-numpy, packaged through
`maturin <https://www.maturin.rs>`_). It is the default backend for
``rivgraph.mask_to_graph.skel_to_graph`` — the step that resolves a skeleton
into links and nodes.

What runs natively
------------------

A single call into ``rivgraph._native`` covers the whole resolution phase: the
graph walk itself, neighborhood analysis, branchpoint identification and
pruning, and the topology updates. Python retains the public container types
(``OrderedSet``) and the final duplicate-link cleanup
(``ln_utils.remove_duplicate_links``), so the call's return shape is the same
for both backends.

Backend selection
-----------------

The backend is selected on **every** ``skel_to_graph`` call:

- If ``PYDELTARCM_NATIVE=0`` is set, the unchanged Python reference walker in
  ``mask_to_graph.py`` runs.
- Otherwise, the compiled ``rivgraph._native`` extension is required. A missing
  or broken extension raises its import error rather than silently falling
  back — rebuild it (``maturin develop``) or set ``PYDELTARCM_NATIVE=0``
  explicitly.

.. note::

   The environment-variable name is shared with the sibling pyDeltaRCM native
   port; setting it affects both packages if both are installed. The name is
   deliberate, and renaming it would be a code change rather than a
   documentation decision.

Exact parity with the Python reference
--------------------------------------

Exact-parity tests compare link and node IDs, pixel sequences, connectivity,
and disconnected-network counts **without sorting or canonicalization**, on a
seed-42 synthetic mask, the committed Colville and Brahmaputra masks, and
additional fixtures covering loops, image boundaries, disconnected networks,
strided and read-only arrays, invalid inputs, and 30 seeded branched masks.
CI runs the test suite twice — native by default, then with
``PYDELTARCM_NATIVE=0`` — to keep both paths exercised.

Determinism: OrderedSet tie-breaks
----------------------------------

The Rust code deliberately consults interpreter-side ``OrderedSet`` containers
at ordering-sensitive decisions instead of iterating its own hash maps or
sorting indices. Arbitrary Rust hash iteration or index sorting would change
node and link IDs; deferring those decisions to the Python-side containers
keeps IDs identical across backends. Branchpoint results are memoized only
within a single call, on its owned padded copy.

Threading
---------

The GIL is held for the duration of the native call. The port is a latency
optimization, not a multi-threaded throughput claim.

Performance
-----------

The benchmark suite includes a paired Brahmaputra walker benchmark: both
backends run on the same staged skeleton, with one warm-up and five measured
rounds per backend. Measured medians recorded when the walker was ported
(commit ``59627ac``):

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Mask
     - Python reference (median)
     - Native walker (median)
   * - Brahmaputra (largest paired mask)
     - 7.42 s
     - 0.128 s

Benchmark output includes interpreter and walker versions, input checksums,
and per-round medians and minima, so results stay comparable across runs. Run
the suite locally with ``python -m pytest bench/``; the setup is described in
:doc:`../contributing/index`.

Verifying an install
--------------------

``rivgraph env`` reports the effective backend (``native``, ``python``, or
``unavailable``, with the underlying import error when there is one), and
``rivgraph demo`` pushes a deterministic synthetic mask through
``skel_to_graph`` with the current backend — a quick end-to-end check that the
extension was built correctly. See :doc:`../cli/index`.

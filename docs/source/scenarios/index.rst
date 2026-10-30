.. _scenarios:

===============
Demo scenarios
===============

The ``scenarios/`` directory is a repository-level suite of named, repeatable
demonstrations. Each scenario runs the public ``delta``/``river`` workflows
end to end on one of the four committed example datasets and emits one
structured record per run: per-phase wall time, throughput, cumulative peak
RSS, and the data the run processed (links, nodes, channel lengths, widths,
islands, fluxes — or the scenario's own counts). The suite is
pytest-executable (``tests/test_scenarios.py``): the quick subset runs in the
default test suite and full pipelines run under the ``slow`` marker. It
complements the stage-level micro-benchmarks in ``bench/``; no RivGraph
algorithms are modified.

Running the suite
-----------------

.. code-block:: bash

   python -m scenarios --list                # catalog: name + one-line description
   python -m scenarios --quick               # fast, CI-safe subset
   python -m scenarios --only delta_colville_full river_brahmaputra_full
   python -m scenarios                       # the whole catalog

Every run writes two reports into ``scenarios_results/``:

- ``scenario_results.json`` — machine-readable, one record per scenario
- ``SCENARIOS.md`` — a human summary (overview table plus per-scenario detail)

Scenario names are the stable command-line surface; renaming one is a
breaking change and is avoided. ``--list`` is the authoritative catalog.

The scenario catalog
--------------------

.. list-table::
   :header-rows: 1
   :widths: 26, 22, 52

   * - Scenario
     - Dataset
     - What its record shows
   * - ``delta_colville_full``
     - Colville Delta
     - The flagship end-to-end pipeline: inlet/outlet counts, islands,
       topologic metrics, and steady-state fluxes
   * - ``delta_colville_metrics_only``
     - Colville Delta
     - Isolates the delta analysis stage for comparison against the full
       pipeline
   * - ``delta_colville_fluxes``
     - Colville Delta
     - Isolates the flux engine: flux min/max and the outlet split
   * - ``delta_colville_skeletonize``
     - Colville Delta
     - Raw image-stage throughput at full mask scale (quick subset)
   * - ``delta_colville_scaling``
     - Colville Delta, 0.5x/0.75x/1.0x
     - Efficiency curve: seconds and px/s against pixel count
   * - ``river_brahmaputra_full``
     - Brahmaputra Braided River
     - Full river pipeline including centerline, mesh, and flow directions
   * - ``river_brahmaputra_mesh``
     - Brahmaputra Braided River
     - Mesh sensitivity: counts and time per grid spacing
   * - ``river_meander_smoke``
     - Meandering River
     - Fastest end-to-end smoke — the CI canary (quick subset)
   * - ``brain_vessel_network``
     - Mouse Brain Vasculature
     - The same machinery on non-river imagery (un-georeferenced PNG)

All scenarios run headless on committed data only, with explicit parameters.

The run record
--------------

One JSON object per scenario (``result_to_dict`` in
``scenarios/runner.py`` is the schema source of truth):

.. code-block:: json

   {
     "scenario": "delta_colville_full",
     "description": "one-line description",
     "dataset": {"name", "mask_path", "rows", "cols", "pixels", "crs"},
     "phases": [{"name", "seconds", "throughput", "peak_rss_mb"}],
     "outputs": {"n_links", "n_nodes", "total_channel_length_px", "..."},
     "environment": {"python", "rivgraph", "platform", "cpu_count",
                     "utc_timestamp"},
     "status": "ok",
     "error": null
   }

Two measurement-honesty rules hold by construction:

- ``peak_rss_mb`` is the cumulative process peak after the phase
  (``resource.getrusage`` ``ru_maxrss``), never a per-phase delta.
- ``throughput`` is attributed only to the resource a phase actually touches:
  pixels/s for image stages, links/s for graph stages, ``null`` where neither
  applies.

Scenarios that extract a network always report ``n_links``, ``n_nodes``, and
``total_channel_length_px``; skeletonization and scaling scenarios report
their own counts (skeleton pixels, per-scale rows). A failing phase finalizes
its record as ``failed`` with the error text; the CLI keeps running the
remaining scenarios and exits 1 at the end if anything failed.

Reports and CI
--------------

The ``scenarios.yml`` workflow runs ``python -m scenarios --quick`` on pushes
to master and on manual dispatch. ``actions/upload-artifact`` is unavailable
under this repository's Actions policy, and Actions artifacts have no upload
REST API — the same constraint the docs workflow works around for the Pages
tar — so both reports are published as the run commit's pre-release assets,
individually downloadable per run and replaced when the same commit re-runs.

Scenarios versus micro-benchmarks
---------------------------------

Use ``bench/`` when optimizing one stage: pytest-benchmark times the hot
stages with statistical rounds and honest per-round state pinning. Use the
scenarios when you want the end-to-end story — what a whole workflow
processed and what it cost — or when a demo must survive contact with CI.

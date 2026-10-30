.. _cli:

=======================
Command-line interface
=======================

This fork ships a small housekeeping CLI alongside the Python API. It is
installed as the console script ``rivgraph`` (declared in ``pyproject.toml``)
and can also be invoked as ``python -m rivgraph``. There are three
subcommands: ``env``, ``inspect``, and ``demo``.

The module imports only the standard library at module level; every subcommand
imports RivGraph internals and third-party dependencies lazily, inside the
function that needs them. ``rivgraph env`` therefore works in bare
environments, and no command inherits the heavy ``rivgraph.classes`` import
chain (geopandas, shapely, matplotlib, networkx, contextily) unless it
actually needs it.

rivgraph env
------------

Reports the installed RivGraph version, Python version and executable,
platform, walker-backend state, and the installed version of each declared
dependency. Abbreviated, illustrative output (values depend on the
environment):

.. code-block:: console

   $ rivgraph env
   RivGraph 1.0
   Python 3.12.16 (/home/user/miniforge3/envs/rivgraph/bin/python)
   Platform: Linux-6.8.0-x86_64-with-glibc2.35
   Native walker backend: native (rivgraph._native importable)
   Dependency versions:
     numpy: 2.2.6
     scipy: 1.13.1
     ... (one line per declared dependency)

The backend line shows whether ``PYDELTARCM_NATIVE`` is set, whether
``rivgraph._native`` is importable, and — when the import fails — the
underlying error, plus a hint to build the extension (``maturin develop``) or
set ``PYDELTARCM_NATIVE=0``. With ``--json``, the same report is emitted as
machine-readable JSON.

rivgraph inspect
----------------

``rivgraph inspect MASK`` produces a read-only pre-flight report for a channel
mask raster. It reports shape, band count, and dtype; whether the raster is
georeferenced (with CRS, EPSG code, and WKT when available); the detected
unit; pixel size and pixel area; nodata; the unique pixel values (up to 20
listed); the channel-pixel count and fraction; and the number of channel
pixels touching the image border.

It emits warnings when:

- the mask is not binary (values other than 0/1, excluding nodata);
- the raster is not georeferenced (RivGraph then applies default 1×1 pixel
  georeferencing in memory);
- unit detection fails; or
- channel pixels touch the image edge — if edge pixels are no-data that was
  not set to land, skeletonization can produce a cyclic graph looping around
  the image edge (see :doc:`../issues/index`).

The command exits non-zero when the file does not exist or cannot be read.

rivgraph demo
-------------

``rivgraph demo`` builds a deterministic synthetic mask, runs it through
skeletonization and ``skel_to_graph`` with the current walker backend, and
prints the resolved graph size and timing:

.. code-block:: console

   $ rivgraph demo
   Synthetic mask: 400 x 400 (RandomState(42), gaussian sigma=7, threshold=85th percentile)
   Native walker backend: native
   Skeletonizing and resolving the graph (may take a few seconds)...
   Resolved graph in <elapsed>s: <links> links, <nodes> nodes

The mask construction is seeded (``RandomState(42)``): a Gaussian-smoothed
random field (sigma 7) thresholded at its 85th percentile. ``--size`` sets the
field width/height in pixels (minimum 32, default 400). ``--out DIR`` writes
the mask and its skeleton as ``demo_mask.tif`` and ``demo_skeleton.tif`` with
default 1×1 pixel georeferencing.

The demo requires a usable walker backend: it exits with an error when
``rivgraph._native`` is neither built nor bypassed via ``PYDELTARCM_NATIVE=0``.
It therefore doubles as a quick native-walker install verification — see
:doc:`../native/index`.

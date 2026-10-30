.. _install:

============
Installation
============

RivGraph can be installed either as a ready-to-use package from conda-forge or from source for development.
For source work, ``environment.yml`` is the canonical environment.

This fork (``jay-kestrel/RivGraph``) compiles a Rust native skeleton walker into
``rivgraph._native`` through the `maturin <https://www.maturin.rs>`_ build
backend, so **any source install requires a stable Rust toolchain** in addition
to Python. See :ref:`fork-source-build` below.

Install from conda
------------------

If you want to use RivGraph without modifying the code, install it into a fresh environment with conda:

::

   conda create -n rivgraph_env -c conda-forge "jschwenk::rivgraph"
   conda activate rivgraph_env

.. warning::

   The conda-forge package ``jschwenk::rivgraph`` is built from the upstream
   project (`VeinsOfTheEarth/RivGraph <https://github.com/VeinsOfTheEarth/RivGraph>`_)
   and does **not** contain the fork additions: the native skeleton walker
   (``rivgraph._native``), the ``rivgraph`` command-line interface, or the
   benchmark harness. To use them, build this repository from source
   (:ref:`fork-source-build`).

.. _fork-source-build:

Install from source
-------------------

For development, testing, or documentation work, start from the repository root.

**Rust toolchain prerequisite.** Installing the package compiles the Rust native
extension ``rivgraph._native`` — ``pyproject.toml`` declares the maturin build
backend (``maturin>=1.9,<2``), the crate sources live under ``rust/``, and the
Cargo manifest is the repository-root ``Cargo.toml``. Install a stable Rust
toolchain (including Cargo) with `rustup <https://rustup.rs>`_ before the first
install; without it the build fails.

Create the canonical source environment:

::

   conda env create -f environment.yml
   conda activate rivgraph

Then install RivGraph in editable mode:

::

   pip install -e . --no-deps

Using ``--no-deps`` is intentional here so that pip does not try to re-resolve compiled geospatial dependencies that are already managed by conda. It does not skip the native build: the editable install still compiles ``rivgraph._native``.

Pure pip (or uv) variant
~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer not to use conda, create a Python 3.12+ virtual environment and
let pip resolve the dependencies from PyPI:

::

   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e .

``uv`` works the same way (``uv venv``, then ``uv pip install -e .``). The
compiled geospatial stack (rasterio, pyogrio, shapely, geopandas, pyproj) comes
from PyPI wheels, and maturin compiles ``rivgraph._native`` as part of the
install.

Rebuild the native extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When iterating on the Rust sources in ``rust/``, rebuild the extension directly
with maturin instead of reinstalling the package:

::

   maturin develop            # debug build, fastest turnaround
   maturin develop --release  # optimized build

Verify the walker backend
~~~~~~~~~~~~~~~~~~~~~~~~~

``rivgraph env`` (see :doc:`../cli/index`) reports which walker backend is
effective: ``native`` when ``rivgraph._native`` imports cleanly, ``python`` when
``PYDELTARCM_NATIVE=0`` forces the pure-Python reference walker (see
:doc:`../native/index`), and ``unavailable`` when the extension is missing or
broken. ``rivgraph demo`` runs a deterministic synthetic mask through the full
mask-to-graph pipeline and doubles as a quick end-to-end check of the build.

Verify a source install
-----------------------

From the repository root, run the test suite:

::

   pytest -ra

If you want a faster smoke test focused on geospatial IO and regression coverage, run:

::

   pytest -ra tests/test_geospatial_roundtrip.py tests/regression

Build the documentation locally
-------------------------------

After activating the source environment, install the docs extras:

::

   pip install -e ".[docs]"

Then build the HTML docs from the repository root with a cross-platform command:

::

   python -m sphinx -b html docs/source docs/build/html

Open ``docs/build/html/index.html`` in your browser to review the rendered site.

Convenience commands
~~~~~~~~~~~~~~~~~~~~

You can also build from the ``docs`` directory:

**macOS / Linux**
::

   cd docs
   make html

**Windows**
::

   cd docs
   make.bat html

Use ``linkcheck`` separately when you want to validate external links:

::

   python -m sphinx -b linkcheck docs/source docs/build/linkcheck

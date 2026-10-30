"""
RivGraph Command-Line Interface (cli.py)
========================================

A small, additive housekeeping CLI: environment reporting, mask pre-flight
inspection, and a deterministic mask-to-graph demo.

The module imports only the standard library at module level. Every
subcommand imports RivGraph internals and third-party dependencies lazily,
inside the function that needs them. This keeps ``rivgraph env`` usable in
bare environments and prevents commands from inheriting the heavy
``rivgraph.classes`` import chain (geopandas, shapely, matplotlib, networkx,
contextily).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Distributions reported by `rivgraph env`, as accepted by importlib.metadata.
# Lookups read installed metadata only; no package is imported.
_DEPENDENCY_DISTRIBUTIONS = (
    'numpy', 'scipy', 'pandas', 'matplotlib', 'networkx', 'scikit-image',
    'shapely', 'geopandas', 'pyproj', 'rasterio', 'affine', 'pyogrio',
    'contextily', 'pillow', 'pyyaml', 'loguru', 'fastdtw',
)

# Pixel value cap when listing unique mask values in `rivgraph inspect`.
_MAX_SHOWN_VALUES = 20


def _native_backend_state() -> dict[str, Any]:
    """Report the native-walker backend state without importing RivGraph.

    Mirrors the selection in ``mask_to_graph.skel_to_graph``: the Python
    reference walker is used when ``PYDELTARCM_NATIVE=0``; otherwise the
    compiled ``rivgraph._native`` extension is required.
    """
    env_value = os.environ.get('PYDELTARCM_NATIVE')
    spec = importlib.util.find_spec('rivgraph._native')
    import_error: str | None = None
    if spec is not None:
        try:
            import rivgraph._native  # noqa: F401  (availability probe only)
        except Exception as exc:  # surfaced below, never raised
            import_error = f'{type(exc).__name__}: {exc}'

    available = spec is not None and import_error is None
    if env_value == '0':
        effective = 'python'
    elif available:
        effective = 'native'
    else:
        effective = 'unavailable'

    return {
        'effective': effective,
        'pydeltarcm_native_env': env_value,
        'native_extension_available': available,
        'native_import_error': import_error,
    }


def _dependency_versions() -> dict[str, str | None]:
    """Installed versions for the declared dependencies; None when absent."""
    versions: dict[str, str | None] = {}
    for distribution in _DEPENDENCY_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def cmd_env(args: argparse.Namespace) -> int:
    """Report RivGraph, Python, walker-backend, and dependency versions."""
    # The package __init__ only defines __version__, so this import stays
    # clean; with maturin dynamic versioning it is the authoritative source.
    from rivgraph import __version__

    backend = _native_backend_state()
    info: dict[str, Any] = {
        'rivgraph_version': __version__,
        'python_version': platform.python_version(),
        'python_executable': sys.executable,
        'platform': platform.platform(),
        'backend': backend,
        'dependencies': _dependency_versions(),
    }

    if args.json:
        print(json.dumps(info, indent=2))
        return 0

    print(f'RivGraph {info["rivgraph_version"]}')
    print(f'Python {info["python_version"]} ({info["python_executable"]})')
    print(f'Platform: {info["platform"]}')

    details = []
    if backend['pydeltarcm_native_env'] is not None:
        details.append(f'PYDELTARCM_NATIVE={backend["pydeltarcm_native_env"]}')
    if backend['native_extension_available']:
        details.append('rivgraph._native importable')
    elif backend['native_import_error'] is not None:
        details.append(
            f'rivgraph._native import failed ({backend["native_import_error"]})'
        )
    else:
        details.append('rivgraph._native not found (native extension not built)')
    print(f'Native walker backend: {backend["effective"]} ({"; ".join(details)})')
    if backend['effective'] == 'unavailable':
        print(
            '  Hint: build the native extension (maturin develop) or set '
            'PYDELTARCM_NATIVE=0 to use the Python reference walker.'
        )

    print('Dependency versions:')
    for name, version in info['dependencies'].items():
        print(f'  {name}: {version if version is not None else "(not installed)"}')
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Pre-flight report for a channel mask raster; read-only."""
    path = Path(args.mask)
    if not path.is_file():
        print(f'Error: no such file: {path}', file=sys.stderr)
        return 1

    # Lazy imports. rasters pulls rasterio/pyproj/scipy/skimage; geo_utils
    # adds the geopandas-backed io_utils. rivgraph.classes is intentionally
    # never imported here.
    import numpy as np
    import rivgraph.geo_utils as geo
    import rivgraph.rasters as rasters

    try:
        raster = rasters.open_raster(path)
    except Exception as exc:
        print(
            f'Error: could not read {path}: {type(exc).__name__}: {exc}',
            file=sys.stderr,
        )
        return 1

    data = np.asarray(raster.array)
    if data.dtype == bool:
        data = data.astype('uint8')

    warnings: list[str] = []
    nodata = raster.nodata
    values = np.unique(data)
    binary = all(
        float(v) in (0.0, 1.0)
        for v in values
        if nodata is None or float(v) != float(nodata)
    )
    if not binary:
        warnings.append('Mask is not binary (expected only 0/1 values).')

    if not raster.source_georeferenced:
        warnings.append(
            'Raster is not georeferenced; RivGraph applies default 1x1 pixel '
            'georeferencing in memory.'
        )

    unit: str | None = None
    if raster.source_georeferenced and raster.crs is not None:
        try:
            unit = geo.get_unit(raster.crs)
        except Exception as exc:  # reported, not swallowed
            warnings.append(
                f'Unit detection failed ({type(exc).__name__}: {exc}).'
            )

    total = int(data.size)
    nonzero = int(np.count_nonzero(data))
    border = np.concatenate(
        [data[0, :], data[-1, :], data[1:-1, 0], data[1:-1, -1]]
    )
    border_channel = int(np.count_nonzero(border))
    if border_channel > 0:
        warnings.append(
            'Channel pixels touch the image edge. If edge pixels are no-data '
            'that was not set to land (0), skeletonization can produce a '
            'cyclic graph looping around the image edge; see '
            'docs/source/issues/index.rst.'
        )

    print(f'Inspecting mask: {path}')
    print(
        f'  Shape: {raster.shape[0]} x {raster.shape[1]} (rows x cols), '
        f'bands: {raster.count}, dtype: {data.dtype}'
    )
    if raster.source_georeferenced:
        epsg = raster.crs.to_epsg() if raster.crs is not None else None
        print('  Georeferenced: yes')
        print(f'  CRS: EPSG:{epsg}' if epsg is not None else '  CRS: (no EPSG code)')
        print(f'  CRS WKT: {raster.crs_wkt}')
    else:
        print('  Georeferenced: no')
    print(f'  Unit: {unit if unit is not None else "(unknown)"}')
    print(
        f'  Pixel size: {raster.pixel_length} x {raster.pixel_length} {unit or ""};'
        f' pixel area: {raster.pixel_area} {(unit or "") + "^2"}'
    )
    print(f'  Nodata: {nodata if nodata is not None else "none"}')
    shown = ', '.join(str(v) for v in values[:_MAX_SHOWN_VALUES])
    if len(values) > _MAX_SHOWN_VALUES:
        shown += f', ... (+{len(values) - _MAX_SHOWN_VALUES} more)'
    print(f'  Unique values: {shown}')
    print(
        f'  Channel pixels: {nonzero} / {total} '
        f'({100.0 * nonzero / total if total else 0.0:.2f}%)'
    )
    print(f'  Border channel pixels: {border_channel}')
    if warnings:
        print('  Warnings:')
        for warning in warnings:
            print(f'    - {warning}')
    else:
        print('  Warnings: (none)')
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a deterministic synthetic mask through the mask-to-graph pipeline."""
    size = args.size
    if size < 32:
        print(f'Error: --size must be >= 32 (got {size}).', file=sys.stderr)
        return 1

    backend = _native_backend_state()
    if backend['effective'] == 'unavailable':
        print(
            'Error: the native walker extension (rivgraph._native) is not '
            'built and PYDELTARCM_NATIVE=0 is not set.',
            file=sys.stderr,
        )
        print(
            'Build it with `maturin develop` or run with PYDELTARCM_NATIVE=0 '
            'to use the Python reference walker.',
            file=sys.stderr,
        )
        return 1

    # Lazy imports: mask_to_graph transitively pulls the full geo stack.
    import numpy as np
    from scipy.ndimage import gaussian_filter

    import rivgraph.mask_to_graph as m2g
    from rivgraph import rasters

    # Same construction as tests/test_native_walker.py: seeded, smoothed
    # random field thresholded at its 85th percentile.
    field = gaussian_filter(np.random.RandomState(42).rand(size, size), 7)
    mask = field > np.percentile(field, 85)
    print(
        f'Synthetic mask: {size} x {size} '
        '(RandomState(42), gaussian sigma=7, threshold=85th percentile)'
    )
    print(f'Native walker backend: {backend["effective"]}')
    print('Skeletonizing and resolving the graph (may take a few seconds)...')

    start = time.perf_counter()
    skeleton = m2g.skeletonize_mask(mask)
    links, nodes = m2g.skel_to_graph(skeleton)
    elapsed = time.perf_counter() - start
    n_links = len(links['id'])
    n_nodes = len(nodes['id'])
    print(f'Resolved graph in {elapsed:.2f}s: {n_links} links, {n_nodes} nodes')

    out_dir = args.out
    if out_dir is not None:
        out_path = Path(out_dir)
        if out_path.exists() and not out_path.is_dir():
            print(f'Error: {out_path} exists and is not a directory.', file=sys.stderr)
            return 1
        out_path.mkdir(parents=True, exist_ok=True)
        transform = rasters._default_transform((size, size))
        gt = rasters.affine_to_geotransform(transform)
        rasters.write_geotiff(mask.astype('uint8'), gt, '', str(out_path / 'demo_mask.tif'), dtype='uint8')
        rasters.write_geotiff(skeleton.astype('uint8'), gt, '', str(out_path / 'demo_skeleton.tif'), dtype='uint8')
        print(f'Wrote demo_mask.tif and demo_skeleton.tif to {out_path}')
    return 0


def cmd_export_multiplex(args: argparse.Namespace) -> int:
    """Export a multiplex graph to canonical node-link JSON.

    Reads a node-link JSON document (as written by
    ``rivgraph.layers.io.write_multiplex_json``), validates the multiplex
    contract on import, and writes the canonical export either to ``--out``
    or to stdout.
    """
    # Lazy import: networkx and the layers package stay out of cli.py's
    # module-import time (see the import-cleanliness note in the module
    # docstring).
    from rivgraph.layers.io import (
        multiplex_to_node_link_data,
        read_multiplex_json,
        write_multiplex_json,
    )

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f'Error: no such file: {in_path}', file=sys.stderr)
        return 1

    try:
        multiplex = read_multiplex_json(in_path)
    except (ValueError, OSError) as exc:
        print(
            f'Error: could not read multiplex from {in_path}: {exc}',
            file=sys.stderr,
        )
        return 1

    try:
        if args.out is not None:
            write_multiplex_json(multiplex, args.out)
            print(f'Wrote multiplex node-link JSON: {args.out}')
        else:
            print(json.dumps(multiplex_to_node_link_data(multiplex), indent=2))
    except (ValueError, OSError) as exc:
        print(f'Error: could not write multiplex JSON: {exc}', file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='rivgraph',
        description=(
            'Housekeeping CLI for RivGraph: environment reporting, mask '
            'pre-flight inspection, a deterministic demo, and multiplex '
            'graph export. Run `rivgraph <command> --help` for command details.'
        ),
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    p_env = subparsers.add_parser(
        'env',
        help='report RivGraph, Python, walker-backend, and dependency versions',
    )
    p_env.add_argument(
        '--json',
        action='store_true',
        help='emit machine-readable JSON instead of plain text',
    )
    p_env.set_defaults(func=cmd_env)

    p_inspect = subparsers.add_parser(
        'inspect',
        help='pre-flight report for a channel mask raster (read-only)',
    )
    p_inspect.add_argument('mask', help='path to the mask GeoTIFF to inspect')
    p_inspect.set_defaults(func=cmd_inspect)

    p_demo = subparsers.add_parser(
        'demo',
        help='run a deterministic synthetic mask through mask-to-graph',
    )
    p_demo.add_argument(
        '--size',
        type=int,
        default=400,
        help='synthetic field width/height in pixels (default: 400)',
    )
    p_demo.add_argument(
        '--out',
        default=None,
        help='optional directory for demo_mask.tif and demo_skeleton.tif',
    )
    p_demo.set_defaults(func=cmd_demo)

    p_export = subparsers.add_parser(
        'export-multiplex',
        help='export a multiplex graph to canonical node-link JSON',
    )
    p_export.add_argument(
        'input',
        help='path to a node-link JSON document of a multiplex graph',
    )
    p_export.add_argument(
        '-o', '--out',
        default=None,
        help='output path for the canonical JSON (default: print to stdout)',
    )
    p_export.set_defaults(func=cmd_export_multiplex)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], int] = args.func
    return func(args)


if __name__ == '__main__':
    raise SystemExit(main())

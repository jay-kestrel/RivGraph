"""Tests for the ``rivgraph`` command-line interface.

The ``env`` tests and the import-cleanliness check run in bare environments
(they only need the stdlib plus pytest). The ``inspect`` and ``demo`` tests
require the geo stack and skip when ``rasterio``/``rivgraph.mask_to_graph``
are unavailable.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import rivgraph.cli as cli
from rivgraph import __version__

ROOT = Path(__file__).resolve().parents[1]

# Modules that cli.py must never import at module-import time: the heavy geo
# stack (geopandas, matplotlib, networkx, shapely, contextily) and even the
# light scientific stack (env must stay import-clean).
_FORBIDDEN_AT_IMPORT = (
    'geopandas', 'rasterio', 'matplotlib', 'networkx', 'shapely', 'contextily',
    'numpy', 'scipy', 'pandas', 'pyproj', 'affine', 'skimage',
)


def test_cli_module_is_import_clean() -> None:
    """cli.py must import only the stdlib so `rivgraph env` works anywhere."""
    code = (
        "import sys\n"
        "import rivgraph.cli\n"
        "heavy = [m for m in %r if m in sys.modules]\n"
        "print(','.join(heavy))\n"
    ) % (_FORBIDDEN_AT_IMPORT,)
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == '', (
        f'heavy modules imported at cli import time: {proc.stdout.strip()}'
    )


def test_env_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['env', '--json']) == 0
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert payload['rivgraph_version'] == __version__
    assert payload['python_version'] == platform.python_version()
    assert payload['backend']['effective'] in {'python', 'native', 'unavailable'}
    assert 'numpy' in payload['dependencies']
    assert 'rasterio' in payload['dependencies']


def test_env_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['env']) == 0
    out = capsys.readouterr().out
    assert f'RivGraph {__version__}' in out
    assert 'Native walker backend:' in out
    assert 'Dependency versions:' in out


def test_python_m_invocation() -> None:
    proc = subprocess.run(
        [sys.executable, '-m', 'rivgraph', 'env', '--json'],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload['rivgraph_version'] == __version__


def test_inspect_missing_file_returns_nonzero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['inspect', str(tmp_path / 'nope.tif')]) == 1
    assert 'no such file' in capsys.readouterr().err


@pytest.fixture()
def write_mask(tmp_path: Path) -> Any:
    """Writer for synthetic single-band GeoTIFFs; skips without rasterio."""
    rasterio = pytest.importorskip('rasterio', reason='rasterio is unavailable')
    pytest.importorskip('rivgraph.rasters', reason='rivgraph.rasters is unavailable')
    np = pytest.importorskip('numpy', reason='numpy is unavailable')
    from rasterio.transform import from_origin

    def _write(name: str, data: Any, *, crs: str | None = 'EPSG:32615', pixel: float = 30.0) -> Path:
        path = tmp_path / name
        kwargs: dict[str, Any] = {
            'driver': 'GTiff',
            'height': data.shape[0],
            'width': data.shape[1],
            'count': 1,
            'dtype': str(data.dtype),
            'transform': from_origin(500000.0, 4100000.0, pixel, pixel),
        }
        if crs is not None:
            kwargs['crs'] = crs
        with rasterio.open(path, 'w', **kwargs) as dst:
            dst.write(data, 1)
        return path

    return _write


def test_inspect_reports_projected_binary_mask(write_mask: Any, capsys: pytest.CaptureFixture[str]) -> None:
    np = pytest.importorskip('numpy', reason='numpy is unavailable')
    data = np.zeros((8, 8), dtype=np.uint8)
    data[3, 1:7] = 1
    path = write_mask('binary.tif', data)

    assert cli.main(['inspect', str(path)]) == 0
    out = capsys.readouterr().out
    assert '8 x 8' in out
    assert 'EPSG:32615' in out
    assert 'meter' in out
    assert 'not binary' not in out.lower()
    assert 'not georeferenced' not in out.lower()
    assert 'image edge' not in out.lower()


def test_inspect_warns_on_nonbinary_mask(write_mask: Any, capsys: pytest.CaptureFixture[str]) -> None:
    np = pytest.importorskip('numpy', reason='numpy is unavailable')
    data = np.zeros((8, 8), dtype=np.uint8)
    data[3, 1:7] = 2
    path = write_mask('nonbinary.tif', data)

    assert cli.main(['inspect', str(path)]) == 0
    out = capsys.readouterr().out
    assert 'not binary' in out
    assert '2' in out  # offending value is listed among unique values


def test_inspect_warns_on_unprojected_mask(write_mask: Any, capsys: pytest.CaptureFixture[str]) -> None:
    np = pytest.importorskip('numpy', reason='numpy is unavailable')
    data = np.zeros((8, 8), dtype=np.uint8)
    data[3, 1:7] = 1
    path = write_mask('unprojected.tif', data, crs=None)

    assert cli.main(['inspect', str(path)]) == 0
    out = capsys.readouterr().out
    assert 'Georeferenced: no' in out
    assert 'not georeferenced' in out


def test_inspect_warns_on_edge_channel_mask(write_mask: Any, capsys: pytest.CaptureFixture[str]) -> None:
    np = pytest.importorskip('numpy', reason='numpy is unavailable')
    data = np.zeros((8, 8), dtype=np.uint8)
    data[:, 0] = 1
    path = write_mask('edge.tif', data)

    assert cli.main(['inspect', str(path)]) == 0
    out = capsys.readouterr().out
    assert 'image edge' in out


def test_inspect_bundled_orinoco_mask(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip('rasterio', reason='rasterio is unavailable')
    pytest.importorskip('rivgraph.rasters', reason='rivgraph.rasters is unavailable')
    mask_path = ROOT / 'tests' / 'data' / 'Orinoco_filled.tif'

    assert cli.main(['inspect', str(mask_path)]) == 0
    out = capsys.readouterr().out
    assert 'Shape:' in out
    assert 'Unique values:' in out
    assert 'Channel pixels:' in out


def _demo_counts(output: str) -> tuple[int, int]:
    """Parse "Resolved graph in ...s: L links, N nodes" from demo output."""
    line = next(
        line for line in output.splitlines() if line.startswith('Resolved graph')
    )
    tail = line.split(':', 1)[1]
    links_part, nodes_part = tail.split(',')
    return (
        int(links_part.strip().split()[0]),
        int(nodes_part.strip().split()[0]),
    )


def test_demo_is_deterministic_and_writes_outputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip(
        'rivgraph.mask_to_graph', reason='mask_to_graph stack is unavailable'
    )

    out_dir = tmp_path / 'demo1'
    assert cli.main(['demo', '--size', '200', '--out', str(out_dir)]) == 0
    first = capsys.readouterr().out
    assert (out_dir / 'demo_mask.tif').is_file()
    assert (out_dir / 'demo_skeleton.tif').is_file()

    out_dir2 = tmp_path / 'demo2'
    assert cli.main(['demo', '--size', '200', '--out', str(out_dir2)]) == 0
    second = capsys.readouterr().out

    counts1 = _demo_counts(first)
    counts2 = _demo_counts(second)
    assert counts1 == counts2
    assert counts1[0] > 0  # links
    assert counts1[1] > 0  # nodes


def test_demo_rejects_tiny_size(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(['demo', '--size', '8']) == 1
    assert '--size must be >= 32' in capsys.readouterr().err

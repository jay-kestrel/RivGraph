"""Quick-subset tests for the scenarios suite (Blueprint AC1 subset, AC2-AC5).

These tests exercise the CLI surface, the record schema contract, and the
smoke scenario's sanity and determinism, so they stay inside the default
suite's budget; full-pipeline scenarios are exercised separately (``slow``
marker, ``scenarios.yml``).

``scenarios/`` is deliberately absent from the installed wheel (AC7), so the
harness is imported from the repo root source tree.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scenarios import runner  # noqa: E402

# The --quick budget (Blueprint: <= 5 minutes on CI) with headroom for slow
# CI runners; a breach shows up as a subprocess timeout failure.
_QUICK_TIMEOUT_SECONDS = 600


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "scenarios", *cli_args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=_QUICK_TIMEOUT_SECONDS,
    )


@pytest.fixture(scope="session", name="quick_run")
def fixture_quick_run(tmp_path_factory):
    """One shared headless ``--quick`` run for all record-reading tests."""
    results_dir = tmp_path_factory.mktemp("scenarios_quick")
    proc = _run_cli("--quick", "--results-dir", str(results_dir))
    return results_dir, proc


def test_list_shows_catalog():  # AC1 subset
    proc = _run_cli("--list")
    assert proc.returncode == 0, proc.stderr
    runner.discover_scenario_modules()
    assert runner.SCENARIOS, "the scenario registry is empty"
    for spec in runner.SCENARIOS.values():
        assert spec.name in proc.stdout
        assert spec.description in proc.stdout


def test_quick_exits_zero_and_writes_reports(quick_run):  # AC2
    results_dir, proc = quick_run
    assert proc.returncode == 0, proc.stdout + proc.stderr
    records = json.loads((results_dir / "scenario_results.json").read_text())
    assert isinstance(records, list)
    assert records
    assert (results_dir / "SCENARIOS.md").read_text().strip()


def test_quick_records_contract(quick_run):  # AC3
    results_dir, _ = quick_run
    records = json.loads((results_dir / "scenario_results.json").read_text())
    runner.discover_scenario_modules()
    for record in records:
        assert record["status"] == "ok", record.get("error")
        for key in ("name", "mask_path", "rows", "cols", "pixels", "crs"):
            assert key in record["dataset"]
        for key in ("python", "rivgraph", "platform", "cpu_count",
                    "utc_timestamp"):
            assert key in record["environment"]
        assert record["phases"]
        assert any(phase["seconds"] > 0 for phase in record["phases"])
        # Network scenarios always report the processed-network summary;
        # image-stage scenarios (skeletonize/scaling) report their own counts.
        if runner.SCENARIOS[record["scenario"]].reports_network:
            for key in ("n_links", "n_nodes", "total_channel_length_px"):
                assert key in record["outputs"]


def test_meander_smoke_sanity(quick_run):  # AC4
    results_dir, _ = quick_run
    records = json.loads((results_dir / "scenario_results.json").read_text())
    record = next(
        r for r in records if r["scenario"] == "river_meander_smoke"
    )
    outputs = record["outputs"]
    assert outputs["n_links"] > 0
    assert outputs["n_nodes"] > 0
    assert outputs["total_channel_length_px"] > 0
    diagonal_px = (
        record["dataset"]["rows"] ** 2 + record["dataset"]["cols"] ** 2
    ) ** 0.5
    assert 0 < outputs["width_mean_px"] <= diagonal_px


def test_in_process_determinism(tmp_path):  # AC5
    runner.discover_scenario_modules()
    spec = runner.SCENARIOS["river_meander_smoke"]
    first = runner.run_scenario(spec, tmp_path / "run1")
    second = runner.run_scenario(spec, tmp_path / "run2")
    assert first.status == second.status == "ok"
    assert [phase.name for phase in first.phases] == [
        phase.name for phase in second.phases
    ]
    assert first.outputs == second.outputs


# -- Delta family: full-pipeline runs (Blueprint AC6 and the scaling record).
# In-process runs through the public API; timing fields are the only thing
# they leave unasserted.


@pytest.mark.slow
def test_delta_colville_full_end_to_end(tmp_path):  # AC6
    runner.discover_scenario_modules()
    spec = runner.SCENARIOS["delta_colville_full"]
    result = runner.run_scenario(spec, tmp_path)
    assert result.status == "ok", result.error
    outputs = result.outputs
    for key in ("n_links", "n_nodes", "total_channel_length_px",
                "n_inlets", "n_outlets", "n_islands"):
        assert key in outputs
    assert outputs["n_links"] > 0
    assert outputs["n_nodes"] > 0
    assert outputs["total_channel_length_px"] > 0
    assert outputs["n_inlets"] > 0
    assert outputs["n_outlets"] > 0
    assert outputs["n_islands"] > 0
    # Topologic metrics digest and flux split are populated.
    assert outputs["topo"]["nonlin_entropy_rate"]
    assert sum(outputs["flux"]["outlet_split"].values()) > 0


@pytest.mark.slow
def test_delta_colville_scaling_rows(tmp_path):
    runner.discover_scenario_modules()
    spec = runner.SCENARIOS["delta_colville_scaling"]
    result = runner.run_scenario(spec, tmp_path)
    assert result.status == "ok", result.error
    rows = result.outputs["scale_rows"]
    assert result.outputs["n_scales"] == 3
    assert [row["scale"] for row in rows] == [0.5, 0.75, 1.0]
    for row in rows:
        assert row["seconds"] > 0
        assert row["pixels_per_s"] > 0
        assert row["n_skeleton_pixels"] > 0


# -- River and vessel family: full-pipeline runs (Blueprint rows 5, 6, and 8).
# In-process runs through the public API; timing fields are the only thing
# they leave unasserted.


@pytest.mark.slow
def test_river_brahmaputra_full_end_to_end(tmp_path):
    runner.discover_scenario_modules()
    spec = runner.SCENARIOS["river_brahmaputra_full"]
    result = runner.run_scenario(spec, tmp_path)
    assert result.status == "ok", result.error
    outputs = result.outputs
    for key in ("n_links", "n_nodes", "total_channel_length_px",
                "n_inlets", "n_outlets"):
        assert outputs[key] > 0
    # The record shows the centerline and mesh the direction-setting ran on.
    assert outputs["centerline_length_px"] > 0
    assert outputs["n_meshlines"] > 0
    assert outputs["n_meshpolys"] > 0


@pytest.mark.slow
def test_river_brahmaputra_mesh_sensitivity(tmp_path):
    runner.discover_scenario_modules()
    spec = runner.SCENARIOS["river_brahmaputra_mesh"]
    result = runner.run_scenario(spec, tmp_path)
    assert result.status == "ok", result.error
    rows = result.outputs["mesh_rows"]
    assert result.outputs["n_spacings"] == 2
    assert [row["spacing_factor"] for row in rows] == [1.0, 2.0]
    assert rows[1]["grid_spacing"] == 2 * rows[0]["grid_spacing"]
    for row in rows:
        assert row["seconds"] > 0
        assert row["n_meshlines"] > 0
        assert row["n_meshpolys"] > 0


@pytest.mark.slow
def test_brain_vessel_network_links(tmp_path):
    runner.discover_scenario_modules()
    spec = runner.SCENARIOS["brain_vessel_network"]
    result = runner.run_scenario(spec, tmp_path)
    assert result.status == "ok", result.error
    outputs = result.outputs
    assert outputs["n_links"] > 0
    assert outputs["n_nodes"] > 0
    assert outputs["total_channel_length_px"] > 0
    diagonal_px = (
        result.dataset["rows"] ** 2 + result.dataset["cols"] ** 2
    ) ** 0.5
    assert 0 < outputs["width_mean_px"] <= diagonal_px

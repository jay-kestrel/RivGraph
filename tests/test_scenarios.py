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
    for record in records:
        assert record["status"] == "ok", record.get("error")
        for key in ("name", "mask_path", "rows", "cols", "pixels", "crs"):
            assert key in record["dataset"]
        for key in ("python", "rivgraph", "platform", "cpu_count",
                    "utc_timestamp"):
            assert key in record["environment"]
        assert record["phases"]
        assert any(phase["seconds"] > 0 for phase in record["phases"])
        # Network scenarios always report the processed-network summary.
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

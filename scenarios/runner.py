"""Scenario harness: per-phase timing, structured records, and the registry.

Standard library only, per the Blueprint: wall time via ``time.perf_counter``
and memory via ``resource.getrusage``'s ``ru_maxrss`` -- the *cumulative
process peak*, never a per-phase delta. RivGraph itself is imported lazily
inside scenario functions so ``python -m scenarios --list`` stays cheap.

A scenario function receives a ``results_dir`` and returns a
``ScenarioResult``::

    @runner.register_scenario(name="...", description="...", dataset="...",
                              quick=True)
    def my_scenario(results_dir: Path) -> runner.ScenarioResult:
        with runner.scenario("my_scenario", datasets.get_dataset("..."),
                             results_dir) as sc:
            d = sc.phase("instantiate", delta, "Name", str(mask_path),
                         results_folder=str(sc.work_dir))
            sc.count_links = lambda: len(d.links["id"])
            sc.phase("skeletonize", d.skeletonize)
            sc.record_outputs(n_links=len(d.links["id"]))
        return sc.result  # return OUTSIDE the with-block

A failing phase finalizes the record as ``failed`` with the error text and
unwinds the scenario via ``PhaseFailure``, which the ``scenario()`` context
manager suppresses; the CLI keeps running the remaining scenarios and exits 1
at the end if anything failed.
"""
from __future__ import annotations

import dataclasses
import importlib.metadata
import json
import os
import pkgutil
import platform
import resource
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ru_maxrss is reported in KiB on Linux; /1024 gives MiB.
_RU_MAXRSS_TO_MIB = 1024.0


@dataclass
class PhaseMetric:
    """One timed pipeline phase.

    ``peak_rss_mb`` is the cumulative process peak after the phase (MiB,
    from ``resource.getrusage`` ``ru_maxrss``), not a per-phase delta.
    """

    name: str
    seconds: float
    throughput: dict | None
    peak_rss_mb: float


@dataclass
class ScenarioResult:
    """One record per scenario run: what ran, what it cost, what it produced."""

    scenario: str
    description: str
    dataset: dict
    phases: list[PhaseMetric]
    outputs: dict
    environment: dict
    status: str
    error: str | None = None


@dataclass(frozen=True)
class ScenarioSpec:
    """A registered scenario: stable CLI name, catalog metadata, function.

    ``reports_network`` declares whether the scenario extracts a channel
    network -- network scenarios always report ``n_links``/``n_nodes``/
    ``total_channel_length_px`` (the record contract); image-stage scenarios
    (skeletonize/scaling) report their own counts instead.
    """

    name: str
    description: str
    dataset: str
    quick: bool
    fn: Callable[[Path], ScenarioResult]
    reports_network: bool = True


SCENARIOS: dict[str, ScenarioSpec] = {}

# Throughput attribution by phase name: "pixels" for image-space stages,
# "links" for graph-space stages, absent or None where neither resource
# applies. Scenario modules extend this map when they add new phase names.
PHASE_THROUGHPUT_UNITS: dict[str, str | None] = {
    "instantiate": None,
    "skeletonize": "pixels",
    "distance_transform": "pixels",
    "compute_network": "links",
    "prune_network": "links",
    "link_widths_and_lengths": "links",
}


def register_scenario(
    name: str,
    description: str,
    dataset: str,
    quick: bool = False,
    reports_network: bool = True,
) -> Callable:
    """Register a scenario function under its stable CLI name."""

    def _register(fn: Callable[[Path], ScenarioResult]) -> Callable:
        if name in SCENARIOS:
            raise ValueError(f"Scenario {name!r} is already registered")
        SCENARIOS[name] = ScenarioSpec(
            name=name,
            description=description,
            dataset=dataset,
            quick=quick,
            fn=fn,
            reports_network=reports_network,
        )
        return fn

    return _register


def discover_scenario_modules() -> None:
    """Import every ``scenarios.scenarios_*`` module so the decorators run.

    Called before any registry access; importing a module twice is a no-op,
    so calling this again is harmless.
    """
    import scenarios as scenarios_pkg

    for module_info in pkgutil.iter_modules(scenarios_pkg.__path__):
        if module_info.name.startswith("scenarios_"):
            __import__(f"scenarios.{module_info.name}")


def _peak_rss_mib() -> float:
    """Cumulative process peak RSS in MiB (``resource.getrusage ru_maxrss``)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / _RU_MAXRSS_TO_MIB


def _environment() -> dict:
    try:
        version = importlib.metadata.version("rivgraph")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "python": platform.python_version(),
        "rivgraph": version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }


class PhaseFailure(Exception):
    """Internal: unwinds a scenario function after a failed phase."""


def _throughput(unit: str | None, dataset: dict, seconds: float,
                count_links: Callable[[], int] | None) -> dict | None:
    """Attribute throughput to the resource this phase actually touches."""
    if seconds <= 0:
        return None
    if unit == "pixels" and dataset.get("pixels"):
        return {"unit": "pixels/s", "value": dataset["pixels"] / seconds}
    if unit == "links" and count_links is not None:
        return {"unit": "links/s", "value": count_links() / seconds}
    return None


class ScenarioContext:
    """Collects phase metrics and outputs for one scenario run.

    ``phase()`` times its call with ``time.perf_counter`` and reads the
    cumulative process peak RSS after the phase. Throughput attribution uses
    ``PHASE_THROUGHPUT_UNITS``: pixel stages need only the dataset shape;
    link stages need the link count, so a scenario registers ``count_links``
    once its network object exists.
    """

    def __init__(self, spec: ScenarioSpec, dataset: dict, work_dir: Path) -> None:
        self.spec = spec
        self.dataset = dataset
        self.work_dir = work_dir
        self.phases: list[PhaseMetric] = []
        self.outputs: dict = {}
        self.count_links: Callable[[], int] | None = None
        self._status = "ok"
        self._error: str | None = None

    def phase(self, name: str, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Run ``fn(*args, **kwargs)`` as a timed phase; returns fn's result."""
        start = time.perf_counter()
        try:
            value = fn(*args, **kwargs)
        except Exception as exc:
            seconds = time.perf_counter() - start
            self.phases.append(
                PhaseMetric(
                    name=name,
                    seconds=seconds,
                    throughput=None,
                    peak_rss_mb=_peak_rss_mib(),
                )
            )
            self._status = "failed"
            self._error = f"{type(exc).__name__}: {exc}"
            raise PhaseFailure(
                f"phase {name!r} failed: {self._error}"
            ) from exc
        seconds = time.perf_counter() - start
        self.phases.append(
            PhaseMetric(
                name=name,
                seconds=seconds,
                throughput=_throughput(
                    PHASE_THROUGHPUT_UNITS.get(name),
                    self.dataset,
                    seconds,
                    self.count_links,
                ),
                peak_rss_mb=_peak_rss_mib(),
            )
        )
        return value

    def record_outputs(self, **outputs: Any) -> None:
        """Fold the scenario's processed-data summary into the record."""
        self.outputs.update(outputs)

    @property
    def result(self) -> ScenarioResult:
        return ScenarioResult(
            scenario=self.spec.name,
            description=self.spec.description,
            dataset=dict(self.dataset),
            phases=list(self.phases),
            outputs=dict(self.outputs),
            environment=_environment(),
            status=self._status,
            error=self._error,
        )


class _ScenarioRun:
    """Context-manager factory for one scenario run, per the Blueprint shape."""

    def __init__(self, name: str, dataset: Any, results_dir: Path) -> None:
        spec = SCENARIOS[name]
        from scenarios.datasets import describe_dataset

        work_root = Path(results_dir) / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        # A unique work dir per run keeps every run cold (no cached skeletons
        # or distance transforms leaking across runs).
        work_dir = Path(tempfile.mkdtemp(prefix=f"{name}_", dir=work_root))
        self.context = ScenarioContext(spec, describe_dataset(dataset), work_dir)

    def __enter__(self) -> ScenarioContext:
        return self.context

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            return False
        # Mark the failure on the context so the record keeps the phases
        # that did complete (don't overwrite a phase failure's error text).
        self.context._status = "failed"
        if self.context._error is None:
            self.context._error = f"{type(exc).__name__}: {exc}"
        if issubclass(exc_type, PhaseFailure):
            # A failed phase already finalized the failed record; suppress
            # its unwind so the scenario function can return the record.
            return True
        # A body failure (e.g. in record_outputs) still propagates; bridge
        # the context to run_scenario so the partial record survives.
        exc.scenario_partial = self.context  # type: ignore[attr-defined]
        return False


def scenario(name: str, dataset: Any, results_dir: Path) -> _ScenarioRun:
    """Open a scenario run: creates the work dir and the record context."""
    return _ScenarioRun(name, dataset, results_dir)


def _describe(spec: ScenarioSpec) -> dict:
    from scenarios.datasets import describe_dataset, get_dataset

    return describe_dataset(get_dataset(spec.dataset))


def _failed_record(spec: ScenarioSpec, error: str) -> ScenarioResult:
    try:
        dataset = _describe(spec)
    except Exception:
        dataset = {
            "name": spec.dataset,
            "mask_path": None,
            "rows": None,
            "cols": None,
            "pixels": None,
            "crs": None,
        }
    return ScenarioResult(
        scenario=spec.name,
        description=spec.description,
        dataset=dataset,
        phases=[],
        outputs={},
        environment=_environment(),
        status="failed",
        error=error,
    )


def run_scenario(spec: ScenarioSpec, results_dir: Path) -> ScenarioResult:
    """Run one scenario to completion, always returning a record."""
    try:
        result = spec.fn(Path(results_dir))
    except Exception as exc:
        # Failures outside any phase (imports, registry bugs, scenario-body
        # errors): the record must still exist so the CLI can continue and
        # report at the end. Prefer the context's partial record when the
        # scenario run bridged one -- it keeps the completed phases.
        partial = getattr(exc, "scenario_partial", None)
        if isinstance(partial, ScenarioContext):
            return partial.result
        return _failed_record(spec, f"{type(exc).__name__}: {exc}")
    if not isinstance(result, ScenarioResult):
        return _failed_record(
            spec,
            "scenario function returned "
            f"{type(result).__name__}, not ScenarioResult",
        )
    return result


def result_to_dict(result: ScenarioResult) -> dict:
    """JSON-serializable form of a record (machine-readable report schema)."""
    return dataclasses.asdict(result)


def write_reports(results: list[ScenarioResult], results_dir: Path) -> None:
    """Write the machine (JSON) and human (Markdown) reports."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = [result_to_dict(result) for result in results]
    with open(results_dir / "scenario_results.json", "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    with open(results_dir / "SCENARIOS.md", "w") as f:
        f.write(_markdown(results))


def _markdown(results: list[ScenarioResult]) -> str:
    """Human-readable summary: an overview table plus per-scenario detail."""
    lines = ["# RivGraph scenario results", ""]
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    lines.append(
        "| scenario | status | total (s) | peak RSS (MiB, cumulative) "
        "| n_links | n_nodes | channel length (px) |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for result in results:
        total = sum(phase.seconds for phase in result.phases)
        peak = max((phase.peak_rss_mb for phase in result.phases), default=0.0)
        lines.append(
            f"| {result.scenario} | {result.status} | {total:.1f} "
            f"| {peak:.1f} | {result.outputs.get('n_links', '-')} "
            f"| {result.outputs.get('n_nodes', '-')} "
            f"| {result.outputs.get('total_channel_length_px', '-')} |"
        )
    for result in results:
        ds = result.dataset
        lines += [
            "",
            f"## {result.scenario}",
            "",
            result.description,
            "",
            f"Dataset: {ds.get('name')} -- {ds.get('rows')}x{ds.get('cols')} "
            f"({ds.get('pixels')} px), crs: {ds.get('crs')}",
            "",
            "| phase | seconds | throughput | peak RSS (MiB) |",
            "|---|---|---|---|",
        ]
        for phase in result.phases:
            tp = phase.throughput
            tp_text = (
                f"{tp['value']:.1f} {tp['unit']}" if tp else "-"
            )
            lines.append(
                f"| {phase.name} | {phase.seconds:.3f} | {tp_text} "
                f"| {phase.peak_rss_mb:.1f} |"
            )
        lines += ["", f"status: {result.status}"]
        if result.error:
            lines.append(f"error: {result.error}")
        lines += ["", "outputs:", ""]
        lines += [f"- {key}: {value}" for key, value in result.outputs.items()]
        lines += [
            "",
            "environment: " + json.dumps(result.environment, sort_keys=True),
            "",
        ]
    return "\n".join(lines)

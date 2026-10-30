"""Scenario CLI: list the catalog, select scenarios, write the reports.

    python -m scenarios --list                # catalog: name + one-line description
    python -m scenarios --quick               # fast, CI-safe subset
    python -m scenarios --only NAME ...       # named scenarios
    python -m scenarios                       # the whole catalog

Writes ``scenario_results.json`` (machine-readable, one record per scenario)
and ``SCENARIOS.md`` (human summary) into the results directory. Exits 1 if
any scenario failed; the run loop continues past failures either way.

Stdlib-only at module level, mirroring ``rivgraph/cli.py``: RivGraph is
imported lazily inside scenario functions, so ``--list`` stays cheap.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scenarios import runner


def _select(args: argparse.Namespace) -> list:
    """Resolve the requested scenario selection; validates --only names."""
    runner.discover_scenario_modules()
    if args.only is not None:
        unknown = [name for name in args.only if name not in runner.SCENARIOS]
        if unknown:
            known = ", ".join(sorted(runner.SCENARIOS)) or "(none)"
            raise SystemExit(
                f"unknown scenario(s): {', '.join(unknown)}; known: {known}"
            )
        selected = [runner.SCENARIOS[name] for name in args.only]
        if args.quick:
            selected = [spec for spec in selected if spec.quick]
        return selected
    specs = list(runner.SCENARIOS.values())
    if args.quick:
        specs = [spec for spec in specs if spec.quick]
    return specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scenarios",
        description="Run RivGraph demo scenarios with per-phase metrics.",
    )
    parser.add_argument(
        "--list",
        dest="list_only",
        action="store_true",
        help="list the scenario catalog and exit",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run only the fast, CI-safe subset of scenarios",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help="run only these scenarios (combined with --quick, only their "
        "quick members)",
    )
    parser.add_argument(
        "--results-dir",
        default="scenarios_results",
        help="directory for reports and per-run work dirs "
        "(default: ./scenarios_results)",
    )
    args = parser.parse_args(argv)

    if args.list_only:
        runner.discover_scenario_modules()
        for spec in runner.SCENARIOS.values():
            print(f"{spec.name}: {spec.description}")
        return 0

    selected = _select(args)
    if not selected:
        print("no scenarios selected", file=sys.stderr)
        return 2

    results = []
    for spec in selected:
        print(f"=== {spec.name} ===", flush=True)
        result = runner.run_scenario(spec, Path(args.results_dir))
        total = sum(phase.seconds for phase in result.phases)
        line = f"{spec.name}: {result.status} in {total:.1f}s"
        if result.status == "failed":
            line += f" -- {result.error}"
        print(line, flush=True)
        results.append(result)

    runner.write_reports(results, Path(args.results_dir))
    print(f"reports written to {args.results_dir}")

    return 1 if any(result.status != "ok" for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())

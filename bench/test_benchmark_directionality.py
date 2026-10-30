"""Paired R4 search-core timings; Colville remains the end-to-end R1 gate."""
import pytest

from rivgraph import _directionality as query

pytest.importorskip("pytest_benchmark.plugin")


@pytest.mark.parametrize("backend", ["0", "1"], ids=["python", "native"])
@pytest.mark.parametrize("closed", [False, True], ids=["repairable", "no-terminals"])
def test_benchmark_cycle_search(benchmark, monkeypatch, backend, closed):
    monkeypatch.setenv("PYDELTARCM_NATIVE", backend)
    edges = [(i, (i + 1) % 10) for i in range(10)]
    if not closed:
        edges += [(10, 0), (9, 11)]
    args = (edges, list(range(10)), set() if closed else {10, 11},
            [] if closed else [10], [] if closed else [11])
    benchmark.extra_info.update(backend=backend, edges=edges, candidates=list(range(10)))
    benchmark.pedantic(query.select_cycle_flip, args=args, rounds=20, warmup_rounds=1)

"""Log-level convention tests for re-leveled events.

Pins the severity of events that must log identically across river and
delta code paths: continuity violations and unfixable cycles at WARNING
(mirroring ``rivers/river_directionality.py``), and the walker's per-step
trace at DEBUG. Uses the ``loguru_records`` fixture from
``tests/conftest.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests._helpers import require_raster_runtime


def _directionality_module():
    require_raster_runtime()
    from rivgraph import directionality

    return directionality


def _delta_directionality_module():
    require_raster_runtime()
    from rivgraph.deltas import delta_directionality

    return delta_directionality


def _mask_to_graph_module():
    require_raster_runtime()
    from rivgraph import mask_to_graph

    return mask_to_graph


def _walk_module():
    require_raster_runtime()
    from rivgraph import walk

    return walk


class TestDeltaLevels:
    """Delta-side events re-leveled to match their river twins."""

    def test_continuity_violation_logs_warning(self, loguru_records, monkeypatch):
        records = loguru_records()
        dd = _delta_directionality_module()
        di = _directionality_module()

        # Isolate the log site: skip the manual-directions csv and the
        # synthetic-DEM directionality pass, pretend the first continuity
        # check passes and the second reports node 55 (so the run reaches the
        # post-fix continuity report on this otherwise-empty network).
        monkeypatch.setattr(di, 'dir_set_manually', lambda links, nodes, csv: (links, nodes))
        monkeypatch.setattr(dd, 'set_initial_directionality', lambda links, nodes, imshape: (links, nodes))

        calls = []

        def _continuity_fake(links, nodes):
            calls.append(1)
            return [55] if len(calls) > 1 else []

        monkeypatch.setattr(di, 'check_continuity', _continuity_fake)

        links, nodes = dd.set_link_directions({'id': [], 'conn': []}, {'id': [], 'conn': []}, (10, 10))

        warnings = [r for r in records if r['level'].name == 'WARNING']
        assert len(warnings) == 1
        assert 'violate continuity' in warnings[0]['message']
        assert warnings[0]['function'] == 'set_link_directions'

    def test_unfixable_delta_cycle_logs_warning(self, loguru_records, monkeypatch):
        records = loguru_records()
        dd = _delta_directionality_module()

        # 3-link ring (0->1->2->0); the patched fixer never resolves it.
        links = {'id': [0, 1, 2], 'conn': [[0, 1], [1, 2], [2, 0]]}
        nodes = {'id': [0, 1, 2], 'conn': [[0, 2], [0, 1], [1, 2]]}

        monkeypatch.setattr(dd, 'fix_delta_cycle', lambda l, n, cfix, imshape: (l, n, 0))

        links, nodes, allfixed = dd.fix_delta_cycles(links, nodes, (10, 10))

        assert allfixed == 0
        warnings = [r for r in records if r['level'].name == 'WARNING']
        assert any('Could not fix the following cycles (links)' in r['message'] for r in warnings)

    def test_fixed_delta_cycle_logs_warning(self, loguru_records, monkeypatch):
        records = loguru_records()
        dd = _delta_directionality_module()

        links = {'id': [0, 1, 2], 'conn': [[0, 1], [1, 2], [2, 0]]}
        nodes = {'id': [0, 1, 2], 'conn': [[0, 2], [0, 1], [1, 2]]}

        monkeypatch.setattr(dd, 'fix_delta_cycle', lambda l, n, cfix, imshape: (l, n, 1))

        links, nodes, allfixed = dd.fix_delta_cycles(links, nodes, (10, 10))

        warnings = [r for r in records if r['level'].name == 'WARNING']
        assert any('were fixed, but should be manually checked' in r['message'] for r in warnings)
        assert not any('Could not fix' in r['message'] for r in warnings)


class TestGenericCycleLevels:
    """Shared (river + delta) fix_cycles cycle events log at warning."""

    def test_too_large_cycle_logs_warning(self, loguru_records):
        records = loguru_records()
        di = _directionality_module()

        # 11-link ring: fixing it requires checking 2**11 - 1 flip
        # combinations, which exceeds the 1024-combination attempt budget.
        n = 11
        links = {'id': list(range(n)), 'conn': [[i, (i + 1) % n] for i in range(n)]}
        nodes = {'id': list(range(n)), 'conn': [[(i - 1) % n, i] for i in range(n)]}

        links, nodes, remaining = di.fix_cycles(links, nodes)

        warnings = [r for r in records if r['level'].name == 'WARNING']
        assert any('is too large to attempt to fix automatically' in r['message'] for r in warnings)
        assert remaining == 1

    def test_unfixable_cycle_logs_warning(self, loguru_records):
        records = loguru_records()
        di = _directionality_module()

        # A 3-link ring: no flip configuration removes the cycle without
        # creating a sink/source or re-introducing the cycle, so no
        # combination qualifies.
        links = {'id': [0, 1, 2], 'conn': [[0, 1], [1, 2], [2, 0]], 'certain_alg': [0, 0, 0]}
        nodes = {'id': [0, 1, 2], 'conn': [[0, 2], [0, 1], [1, 2]]}

        links, nodes, remaining = di.fix_cycles(links, nodes)

        warnings = [r for r in records if r['level'].name == 'WARNING']
        assert any('Unfixable cycle found at links' in r['message'] for r in warnings)
        assert remaining == 1


class TestWalkerTraceLevel:
    """The walker's per-step diagnostic logs at debug, not info."""

    # Geometry shared by both tests: a diagonal tail (0,0)->(1,1)->(2,2)
    # feeding a junction whose only other exits are the NE (1,3) and SW (3,1)
    # diagonals. The walker's initialization and branchpoint handling resolve
    # or divert well-formed skeletons before either defensive branch can be
    # reached through geometry alone, so each test forces its branch's
    # precondition at the walk-helper seam (visited-pixel exclusions emptied;
    # branchpoint classification controlled) and pins that the site logs
    # DEBUG before raising. Flat indices in comments assume the walker's
    # fixed 20-pixel padding of the 5x5 input (45 columns).

    def _junction_skeleton(self):
        Iskel = np.zeros((5, 5))
        Iskel[0, 0] = 1  # flat 920: tail start (walks down-right)
        Iskel[1, 1] = 1  # flat 966: tail midpoint
        Iskel[2, 2] = 1  # flat 1012: junction
        Iskel[1, 3] = 1  # flat 968: diagonal arm tip (NE)
        Iskel[3, 1] = 1  # flat 1056: diagonal arm tip (SW)
        return Iskel

    def test_ambiguous_step_logs_debug_before_raise(self, loguru_records, monkeypatch):
        records = loguru_records()
        m2g = _mask_to_graph_module()
        walk = _walk_module()

        # With branchpoints disabled and no visited-pixel exclusions, the walk
        # crosses the junction and is left with two diagonal candidates: none
        # a branchpoint, none 4-connected, none in the no-turnaround
        # semicircle -- exactly the "critical flaw" state.
        monkeypatch.setattr(walk, 'cant_walk', lambda links, linkidx, nodes, Iskel: set())
        monkeypatch.setattr(walk, 'is_bp', lambda idx, Iskel: 0)

        with pytest.raises(RuntimeError, match='Ambiguous which step'):
            m2g._skel_to_graph_python(self._junction_skeleton())

        debug = [r for r in records if r['level'].name == 'DEBUG']
        assert len(debug) == 1
        assert debug[0]['message'].startswith('idx: ')
        assert 'poss_steps:' in debug[0]['message']

    def test_no_unique_branchpoint_logs_debug_before_raise(self, loguru_records, monkeypatch):
        records = loguru_records()
        m2g = _mask_to_graph_module()
        walk = _walk_module()

        # Same walk, but both junction candidates report as branchpoints.
        # Both are diagonal, so none is both branchpoint and 4-connected, and
        # the walk cannot choose a unique branchpoint step.
        monkeypatch.setattr(walk, 'cant_walk', lambda links, linkidx, nodes, Iskel: set())
        monkeypatch.setattr(walk, 'is_bp', lambda idx, Iskel: 1 if idx in (968, 1056) else 0)

        with pytest.raises(RuntimeError, match='unique branchpoint'):
            m2g._skel_to_graph_python(self._junction_skeleton())

        debug = [r for r in records if r['level'].name == 'DEBUG']
        assert len(debug) == 1
        assert debug[0]['message'].startswith('idx: ')
        assert 'poss_steps:' in debug[0]['message']

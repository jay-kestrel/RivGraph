"""Regression tests for per-instance logging (spec Set B).

Pins the de-globalized sink contract introduced when ``init_logger`` stopped
calling ``logger.configure``: constructing a RivGraph object must not disturb
sinks owned by others (B1), each instance's log file receives only its own
run's messages (B2), and ``centerline`` honors the same file+console contract
as its siblings without leaking through loguru's default stderr sink (B3).
"""
from __future__ import annotations

import os
import unittest.mock as mock

import numpy as np
from loguru import logger

from rivgraph.classes import centerline
from tests._helpers import REGRESSION_DATA_ROOT, require_rivgraph_classes

MASK = REGRESSION_DATA_ROOT / "delta_mossy" / "inputs" / "mask.tif"


def _read_log(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _compute_network(net):
    """Run ``net.compute_network()`` with the graph resolver stubbed."""
    net.Iskel = mock.MagicMock()
    net.skeletonize = mock.MagicMock()

    def _patched_skel_to_graph(_iskel):
        return {"id": [0]}, {"id": [1]}

    with mock.patch(
        "rivgraph.mask_to_graph.skel_to_graph", new=_patched_skel_to_graph
    ):
        net.compute_network()


class TestSinkOwnership:
    """B1: instantiation never resets sinks owned by others."""

    def test_second_instance_leaves_first_file_sink_working(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        net_a = rivnetwork("net_a", str(MASK), os.path.join(tmp_path, "results_a"))
        net_b = rivnetwork("net_b", str(MASK), os.path.join(tmp_path, "results_b"))

        # A's sink still receives A's messages after B's construction.
        _compute_network(net_a)

        assert "links and nodes have been resolved." in _read_log(net_a.paths["log"])
        assert os.path.isfile(net_b.paths["log"])

    def test_host_added_sink_survives_instantiation(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        host_messages = []
        sink_id = logger.add(host_messages.append, level="DEBUG")
        try:
            rivnetwork("net", str(MASK), os.path.join(tmp_path, "results"))
            logger.info("host sink still active")
        finally:
            logger.remove(sink_id)

        assert any("host sink still active" in str(m) for m in host_messages)


class TestRunRouting:
    """B2: each instance's log file contains only its own run's messages."""

    def test_log_files_receive_only_their_own_run(self, tmp_path):
        _, _, rivnetwork = require_rivgraph_classes()
        net_a = rivnetwork("net_a", str(MASK), os.path.join(tmp_path, "results_a"))

        _compute_network(net_a)
        log_a_before_b = _read_log(net_a.paths["log"])

        net_b = rivnetwork("net_b", str(MASK), os.path.join(tmp_path, "results_b"))
        _compute_network(net_b)

        # B's construction and B's compute never wrote to A's file.
        assert _read_log(net_a.paths["log"]) == log_a_before_b

        # B's own compute messages were routed to B's file.
        assert "links and nodes have been resolved." in _read_log(net_b.paths["log"])


class TestCenterlineParity:
    """B3: centerline honors the same file+console contract as its siblings."""

    def test_verbose_centerline_writes_log_and_keeps_stderr_clean(self, tmp_path, capfd):
        x = np.linspace(0.0, 10.0, 50)
        y = np.zeros_like(x)

        cl = centerline(x, y, verbose=True, results_folder=str(tmp_path))

        captured = capfd.readouterr()
        # No loguru default-format records on stderr.
        assert captured.err == ""
        # Console output is bare text: exactly the banner.
        assert captured.out == "---------- New Run ----------\n"
        # The log file exists, is non-empty, and carries the banner.
        assert os.path.basename(cl.paths["log"]) == "centerline_log.log"
        log_text = _read_log(cl.paths["log"])
        assert log_text and "---------- New Run ----------" in log_text

    def test_quiet_centerline_keeps_console_empty(self, tmp_path, capfd):
        x = np.linspace(0.0, 10.0, 50)
        y = np.zeros_like(x)

        cl = centerline(x, y, results_folder=str(tmp_path))

        captured = capfd.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert os.path.isfile(cl.paths["log"])
        assert os.path.getsize(cl.paths["log"]) > 0

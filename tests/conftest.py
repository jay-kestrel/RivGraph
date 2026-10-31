from __future__ import annotations

import pytest
from loguru import logger


@pytest.fixture
def loguru_records():
    """Capture loguru records so tests can assert levels and messages.

    Usage: ``records = loguru_records()`` -- callable at any point, before or
    after constructing a ``rivnetwork``/``delta``/``river``/``centerline``
    instance: RivGraph objects add their own per-instance sinks and never
    reset loguru's process-wide handler list, so a capture sink installed
    here survives instantiation. The callable installs a
    DEBUG-level capture sink and returns the list that collects one record
    dict per log call (``r["level"].name``, ``r["message"]``,
    ``r["exception"]``).
    """
    records = []
    sink_ids = []

    def _sink(message):
        records.append(message.record)

    def _start():
        sink_ids.append(logger.add(_sink, level="DEBUG"))
        return records

    yield _start

    for sink_id in sink_ids:
        logger.remove(sink_id)

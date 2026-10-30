"""Security tests for restricted unpickling of network pickles."""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest

from rivgraph.ordered_set import OrderedSet
from tests._helpers import require_io_utils


def _example_links_and_nodes():
    """Synthetic links/nodes shaped like the containers RivGraph pickles."""
    links = {
        'id': OrderedSet([1, 2]),
        'nconn': [[1], [1, 2]],
        'conn': [[2], [1]],
        'idx': [np.array([10, 11]), np.array([11, 12])],
        'geometry': ['LINESTRING (0 0, 1 1)', 'LINESTRING (1 1, 2 0)'],
        'widths': np.array([1.5, 2.5]),
    }
    nodes = {
        'id': OrderedSet([1, 2, 3]),
        'idx': OrderedSet([10, 11, 12]),
        'conn': [[2], [1, 3], [2]],
        'inlets': [1],
        'outlets': [3],
    }
    return links, nodes


def _assert_same_container(original, loaded):
    assert loaded.keys() == original.keys()
    for key, value in original.items():
        roundtripped = loaded[key]
        if isinstance(value, np.ndarray):
            assert np.array_equal(roundtripped, value)
        elif isinstance(value, OrderedSet):
            assert isinstance(roundtripped, OrderedSet)
            assert list(roundtripped) == list(value)
        elif isinstance(value, list) and any(isinstance(item, np.ndarray) for item in value):
            assert len(roundtripped) == len(value)
            for roundtripped_item, item in zip(roundtripped, value):
                assert np.array_equal(roundtripped_item, item)
        else:
            assert roundtripped == value


def test_network_pickle_roundtrip(tmp_path):
    """pickle_links_and_nodes -> unpickle_links_and_nodes preserves contents."""
    io_utils = require_io_utils()
    links, nodes = _example_links_and_nodes()
    path = str(tmp_path / 'network.pkl')

    io_utils.pickle_links_and_nodes(links, nodes, path)
    loaded_links, loaded_nodes = io_utils.unpickle_links_and_nodes(path)

    _assert_same_container(links, loaded_links)
    _assert_same_container(nodes, loaded_nodes)


def test_malicious_pickle_is_rejected_without_side_effects(tmp_path):
    """A pickle whose __reduce__ shells out must raise, not execute."""
    io_utils = require_io_utils()
    marker = tmp_path / 'pwned.marker'

    class MaliciousPayload:
        def __reduce__(self):
            return (os.system, ('touch {}'.format(marker),))

    path = str(tmp_path / 'malicious.pkl')
    with open(path, 'wb') as f:
        pickle.dump(MaliciousPayload(), f)

    with pytest.raises(pickle.UnpicklingError):
        io_utils.unpickle_links_and_nodes(path)

    assert not marker.exists()

"""CHK-6 golden-equivalence gate: the reference rebuild vs the pinned fixture.

Two checks against the committed fixture (``tests/golden/data/``):

- :func:`test_golden_graph_equivalence` -- the CHK-6 gate. The rebuilt graph
  matches the fixture with Tier A exact (id sets, connectivity, topology,
  directionality decisions) and Tier B within PF-3 tolerances (1e-9 relative,
  1e-12 absolute floor) on lengths/widths/float attributes.
- :func:`test_golden_fixture_byte_reproducible` -- a re-mint reproduces the
  committed fixture sha256. Byte drift means environment/input/pipeline drift;
  a legitimate behavior change re-baselines via an explicit commit
  (``python -m tests.golden.mint_golden --rebaseline --reason "<why>"``).

Both run the full reference pipeline (~40 s), so they carry the ``slow``
marker like the other full-pipeline tests; the default suite includes them.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.golden import golden
from tests.golden.mint_golden import FIXTURE_PATH, MANIFEST_PATH

pytestmark = pytest.mark.slow


def load_committed_fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"Golden fixture missing at {FIXTURE_PATH}; mint it with "
            "'python -m tests.golden.mint_golden' (see tests/golden/README.md)."
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt_graph(tmp_path_factory) -> dict:
    """One reference-pipeline rebuild shared by both gate tests."""
    links, nodes = golden.run_reference_pipeline(
        results_folder=tmp_path_factory.mktemp("rivgraph-golden")
    )
    return golden.extract_fixture(links, nodes)


def test_golden_graph_equivalence(rebuilt_graph: dict) -> None:
    """CHK-6: Tier A exact + Tier B within PF-3 against the pinned fixture."""
    committed = load_committed_fixture()
    violations = golden.compare_graphs(committed["graph"], rebuilt_graph)
    assert not violations, (
        "Golden-oracle gate failed against the fixture pinned at commit "
        f"{committed['reference_commit']}:\n  " + "\n  ".join(violations)
    )


def test_golden_fixture_byte_reproducible(rebuilt_graph: dict) -> None:
    """A re-mint reproduces the committed fixture digest (the 26b3e64 pattern)."""
    committed = load_committed_fixture()
    rebuild = golden.build_fixture(
        rebuilt_graph,
        reference_commit=committed["reference_commit"],
        env=golden.capture_environment(),
    )
    digest = hashlib.sha256(golden.serialize_fixture(rebuild)).hexdigest()

    expected = MANIFEST_PATH.read_text().split()[0]
    assert digest == expected, (
        "Re-minted fixture diverges from the committed manifest.\n"
        f"  committed: {expected}\n"
        f"  re-mint:   {digest}\n"
        "Causes: environment drift (pinned versions in the fixture header), "
        "input changes, or pipeline behavior change. If the change is a "
        "legitimate behavior change, re-baseline via "
        '\'python -m tests.golden.mint_golden --rebaseline --reason "<why>"\' '
        "in its own commit."
    )


def test_golden_inputs_unchanged() -> None:
    """The frozen input set still matches the hashes pinned in the fixture."""
    committed = load_committed_fixture()
    dataset = committed["dataset"]
    for key, path_key in (
        ("mask_sha256", "mask_path"),
        ("fixlinks_sha256", "fixlinks_path"),
    ):
        path = Path(golden.REPO_ROOT, dataset[path_key])
        assert path.exists(), f"Frozen input missing: {path}"
        digest = golden._sha256_file(path)
        assert digest == dataset[key], (
            f"Frozen input {dataset[path_key]} changed ({dataset[key]} -> "
            f"{digest}); the oracle's input set must stay frozen (spec PF-1)."
        )

"""Mint the RivGraph golden fixture (spec PF-1 RivGraph row).

Usage (from the repo root, with the pinned environment active):

    python -m tests.golden.mint_golden                 # mint: write fixture + manifest
    python -m tests.golden.mint_golden --check         # verify re-mint reproduces the digest
    python -m tests.golden.mint_golden --rebaseline --reason "<why>"

Byte-reproducibility enforcement (the spec's ``26b3e64`` pattern): the mint
runs the reference pipeline twice and hard-fails unless both serializations
are byte-identical. The committed manifest pins the fixture sha256;
``--check`` fails when a re-mint diverges from it. A digest may only change
through an explicit re-baseline commit carrying a recorded reason.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

from tests.golden.golden import (
    build_fixture,
    capture_environment,
    run_reference_pipeline,
    serialize_fixture,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "data" / "rivgraph_golden.json"
MANIFEST_PATH = Path(__file__).resolve().parent / "data" / "rivgraph_golden.sha256"


def reference_commit() -> str:
    """Full sha of the checkout the mint runs in; the fixture binds to it."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def mint_bytes() -> tuple[bytes, dict]:
    """Run the pipeline twice; return the byte-identical fixture and its digest.

    Raises on any nondeterminism -- the spec's mints are byte-reproducible or
    they are not mints.
    """
    env = capture_environment()
    digests = []
    for run in (1, 2):
        with tempfile.TemporaryDirectory(prefix="rivgraph-mint-") as workdir:
            links, nodes = run_reference_pipeline(results_folder=Path(workdir))
        payload = build_fixture({"links": links, "nodes": nodes}, reference_commit(), env)
        blob = serialize_fixture(payload)
        digests.append(hashlib.sha256(blob).hexdigest())
        if run == 1:
            first_blob = blob
    if digests[0] != digests[1]:
        raise SystemExit(
            "MINT FAILED: two consecutive pipeline runs produced different "
            f"fixtures ({digests[0]} vs {digests[1]}). The reference is not "
            "byte-reproducible; investigate before pinning anything."
        )
    return first_blob, {"sha256": digests[0], "environment": env}


def mint() -> None:
    blob, manifest = mint_bytes()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(blob)
    MANIFEST_PATH.write_text(f"{manifest['sha256']}  rivgraph_golden.json\n")
    counts = parse_counts(blob)
    print(f"fixture:  {FIXTURE_PATH} ({len(blob):,} bytes)")
    print(f"manifest: {MANIFEST_PATH}")
    print(f"sha256:   {manifest['sha256']}")
    print(f"graph:    {counts}")


def parse_counts(blob: bytes) -> str:
    import json

    header = json.loads(blob)
    counts = header["counts"]
    env = header["environment"]
    return (
        f"{counts['n_links']} links, {counts['n_nodes']} nodes "
        f"({counts['n_inlets']} inlets, {counts['n_outlets']} outlets, "
        f"{counts['n_parallels']} parallel pairs); python {env['python']}"
    )


def check() -> None:
    if not MANIFEST_PATH.exists():
        raise SystemExit("No committed manifest found; run a mint first.")
    expected = MANIFEST_PATH.read_text().split()[0]
    blob, manifest = mint_bytes()
    if manifest["sha256"] != expected:
        raise SystemExit(
            "RE-MINT DIVERGED from the committed manifest.\n"
            f"  committed: {expected}\n"
            f"  re-mint:   {manifest['sha256']}\n"
            "Causes: environment drift (pinned versions in the fixture "
            "header), input changes, or pipeline behavior change. If the "
            "change is a legitimate behavior change, re-mint with "
            "'--rebaseline --reason \"<why>\"' in its own commit."
        )
    print(f"OK: re-mint reproduces the pinned digest {expected}")


def rebaseline(reason: str) -> None:
    if not reason.strip():
        raise SystemExit("A re-baseline requires --reason (recorded in the commit).")
    blob, manifest = mint_bytes()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_bytes(blob)
    MANIFEST_PATH.write_text(f"{manifest['sha256']}  rivgraph_golden.json\n")
    print(f"re-baselined (reason: {reason})")
    print(f"sha256:   {manifest['sha256']}")
    print("Commit this with the reason in the message (spec PF-1 re-mint rule).")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--check", action="store_true", help="verify a re-mint reproduces the committed digest")
    group.add_argument("--rebaseline", action="store_true", help="mint a new fixture and manifest (requires --reason)")
    parser.add_argument("--reason", default="", help="why the re-baseline is legitimate")
    args = parser.parse_args(argv)
    if args.rebaseline:
        rebaseline(args.reason)
    elif args.check:
        check()
    else:
        mint()


if __name__ == "__main__":
    main(sys.argv[1:])

"""Integrity check: shipped release bundle bytes match the manifest (issue #470).

Validates every artifact catalogued in the wheel-shipped
``bucket_brigade/baselines/release/local/manifest.json`` against the
**tracked bytes** in the repo: each file must exist, its SHA-256 must
equal the manifest ``sha256``, and its size must equal ``size_bytes``.

Why this exists
---------------

The manifest was frozen in #401 (release 0.1.0). PR #420's full-grid
phase-diagram regeneration then overwrote 4 genome files inside the
shipped ``local/`` bundle without re-freezing, leaving the manifest's
integrity hashes silently false until issue #470 caught it during the
PR #469 evaluation. The pre-existing manifest tests
(``tests/test_release_manifest.py``) are schema-level only, so the
drift went undetected.

This test hashes the actual bundle (11 small JSON files, so it is
fast and unmarked) and makes any future post-freeze overwrite fail CI
loudly. Legitimate artifact changes must go through a re-freeze
(``python -m bucket_brigade.baselines.release.freeze``) or a manifest
round-trip repair (see #470) — never a bare file overwrite.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from bucket_brigade.baselines.release.manifest import (
    MANIFEST_FILENAME,
    load_manifest,
)

# Resolve the bundle relative to the repo checkout (tests/ lives at the
# repo root) so we always validate the *tracked* bytes, independent of
# any installed/downloaded copy or $BUCKET_BRIGADE_BASELINES_DIR.
_BUNDLE_DIR = (
    Path(__file__).resolve().parent.parent
    / "bucket_brigade"
    / "baselines"
    / "release"
    / "local"
)
_MANIFEST_PATH = _BUNDLE_DIR / MANIFEST_FILENAME

_ARTIFACTS = load_manifest(_MANIFEST_PATH).artifacts


def test_manifest_catalogs_at_least_one_artifact() -> None:
    """Guard against a silently-empty manifest making the suite vacuous."""
    assert len(_ARTIFACTS) > 0


@pytest.mark.parametrize("entry", _ARTIFACTS, ids=lambda e: e.filename)
def test_artifact_bytes_match_manifest(entry) -> None:
    """Every manifest entry's file exists with matching sha256 and size."""
    path = _BUNDLE_DIR / entry.filename
    assert path.is_file(), (
        f"Manifest entry {entry.filename!r} has no file at {path}. "
        "Either restore the artifact or re-freeze the bundle."
    )

    data = path.read_bytes()

    assert entry.sha256, (
        f"Manifest entry {entry.filename!r} has an empty sha256; the "
        "shipped bundle must be fully checksummed."
    )
    actual_sha = hashlib.sha256(data).hexdigest()
    assert actual_sha == entry.sha256, (
        f"sha256 drift for {entry.filename!r}: manifest says "
        f"{entry.sha256[:12]}..., tracked bytes hash to "
        f"{actual_sha[:12]}.... Post-freeze overwrites must update the "
        "manifest via the freeze script or a save_manifest round-trip "
        "(see issue #470); never overwrite bundle files in place."
    )
    assert len(data) == entry.size_bytes, (
        f"size_bytes drift for {entry.filename!r}: manifest says "
        f"{entry.size_bytes}, tracked file is {len(data)} bytes."
    )

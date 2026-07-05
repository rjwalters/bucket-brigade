"""Integrity check: shipped release bundle bytes match the manifest (issue #470).

Validates every artifact catalogued in the wheel-shipped
``bucket_brigade/baselines/release/local/manifest.json`` against the
**tracked bytes** in the repo: each file must exist, its SHA-256 must
equal the manifest ``sha256``, and its size must equal ``size_bytes``.
Also validates the inverse (#473): every file shipped inside the bundle
is catalogued in the manifest (or explicitly allowlisted), so nothing
can ship in the wheel while being unreachable via the loader API.

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


# Non-artifact files that legitimately live in the bundle without a
# manifest entry. Anything else on disk MUST be catalogued.
_UNCATALOGUED_ALLOWLIST = {
    MANIFEST_FILENAME,  # the manifest itself
    "README.md",  # bundle layout documentation
    ".gitkeep",  # keeps the dir present pre-freeze
}


def test_every_bundle_file_is_catalogued_in_manifest() -> None:
    """Inverse coverage check (issue #473).

    PR #420 wrote 29 genome files into the shipped bundle without
    manifest entries — they went out in the wheel unreachable via
    ``resolve_artifact_path()`` and with no integrity record. This test
    catches the next such overwrite: every file under the bundle dir
    must be either a manifest artifact or explicitly allowlisted above.
    """
    catalogued = {entry.filename for entry in _ARTIFACTS}
    on_disk = {
        p.relative_to(_BUNDLE_DIR).as_posix()
        for p in _BUNDLE_DIR.rglob("*")
        if p.is_file()
    }
    uncatalogued = sorted(on_disk - catalogued - _UNCATALOGUED_ALLOWLIST)
    assert not uncatalogued, (
        f"{len(uncatalogued)} file(s) ship inside the release bundle but "
        f"are not catalogued in the manifest: {uncatalogued}. Add manifest "
        "entries via a save_manifest round-trip (or re-freeze), or move "
        "the files out of the wheel-shipped local/ dir. See issue #473."
    )


def test_full_grid_phase_diagram_cell_loads_via_loader_api() -> None:
    """One #420 full-grid cell round-trips through the public loader.

    ``b0.10_k0.90_c0.50`` is the cell the #459/#466 anchor annotations
    cite; before #473 it shipped in the wheel but was unreachable
    because it had no manifest entry. Loading it end-to-end proves the
    retroactively-catalogued entries are wired into the loader API.
    """
    from bucket_brigade.baselines.release.loaders import load_nash, load_nash_genomes

    payload = load_nash(
        "minimal_specialization-v1",
        name="phase_diagram_b0.10_k0.90_c0.50",
        directory=_BUNDLE_DIR,
    )
    assert payload["scenario_id"] == "minimal_specialization-v1"
    assert len(payload["positions"]) == 4

    genomes = load_nash_genomes(
        "minimal_specialization-v1",
        name="phase_diagram_b0.10_k0.90_c0.50",
        directory=_BUNDLE_DIR,
    )
    assert genomes.shape == (4, 10)

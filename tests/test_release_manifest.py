"""Tests for the frozen-baseline release manifest schema (issue #373).

Covers :mod:`bucket_brigade.baselines.release.manifest`: round-trip
serialization, schema-version enforcement, and the ``find`` helper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bucket_brigade.baselines.release import (
    DEFAULT_HUGGINGFACE_REPO,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ArtifactEntry,
    Manifest,
    load_manifest,
    save_manifest,
)


def _example_manifest() -> Manifest:
    return Manifest(
        release_version="0.1.0",
        release_date="2026-06-08",
        source_commit="abc1234",
        artifacts=[
            ArtifactEntry(
                kind="archetype",
                name="hero",
                filename="archetypes/hero.pkl",
                sha256="deadbeef",
                size_bytes=42,
                notes="Hand-coded Hero archetype",
            ),
            ArtifactEntry(
                kind="nash",
                name="minimal_specialization",
                filename="nash/minimal_specialization-v1.json",
                scenario_id="minimal_specialization-v1",
                sha256="cafe1234",
                size_bytes=128,
            ),
        ],
        extra={"trained_with": "ppo"},
    )


def test_manifest_round_trip_through_file(tmp_path: Path) -> None:
    """Manifest -> JSON file -> Manifest should be byte-identical."""
    m = _example_manifest()
    p = tmp_path / MANIFEST_FILENAME
    save_manifest(m, p)

    # File exists, parses as JSON, and round-trips through load_manifest.
    assert p.exists()
    loaded = load_manifest(p)

    assert loaded == m
    # And the JSON serialization is stable / deterministic.
    raw = json.loads(p.read_text())
    assert raw["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert raw["release_version"] == "0.1.0"
    assert len(raw["artifacts"]) == 2


def test_manifest_find_matches_kind_name_scenario() -> None:
    m = _example_manifest()
    hero = m.find(kind="archetype", name="hero", scenario_id=None)
    assert hero is not None
    assert hero.filename == "archetypes/hero.pkl"

    nash = m.find(
        kind="nash",
        name="minimal_specialization",
        scenario_id="minimal_specialization-v1",
    )
    assert nash is not None

    # Mismatched scenario_id returns None — exact match only.
    assert m.find(kind="archetype", name="hero", scenario_id="anything-v1") is None
    # Unknown kind/name returns None.
    assert m.find(kind="archetype", name="nope") is None


def test_artifact_entry_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Unknown artifact kind"):
        ArtifactEntry(kind="bogus", name="x", filename="x.pkl")


def test_artifact_entry_rejects_backslash_path() -> None:
    with pytest.raises(ValueError, match="backslash"):
        ArtifactEntry(kind="archetype", name="x", filename="dir\\x.pkl")


def test_artifact_entry_rejects_empty_name_or_filename() -> None:
    with pytest.raises(ValueError, match="name"):
        ArtifactEntry(kind="archetype", name="", filename="x.pkl")
    with pytest.raises(ValueError, match="filename"):
        ArtifactEntry(kind="archetype", name="x", filename="")


def test_manifest_from_dict_rejects_future_schema_version() -> None:
    raw = {
        "schema_version": MANIFEST_SCHEMA_VERSION + 1,
        "release_version": "9.9.9",
        "release_date": "2999-01-01",
        "artifacts": [],
    }
    with pytest.raises(ValueError, match="newer than this client"):
        Manifest.from_dict(raw)


def test_manifest_from_dict_requires_release_version() -> None:
    raw = {"schema_version": MANIFEST_SCHEMA_VERSION}
    with pytest.raises(ValueError, match="release_version"):
        Manifest.from_dict(raw)


def test_default_repo_constant_is_set() -> None:
    # Sanity: the constant is non-empty so docs / scripts have
    # something to print.
    assert DEFAULT_HUGGINGFACE_REPO
    assert "/" in DEFAULT_HUGGINGFACE_REPO

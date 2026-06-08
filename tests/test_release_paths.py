"""Tests for the local-path resolver (issue #373).

Covers :mod:`bucket_brigade.baselines.release.paths`: directory
override, manifest loading, and artifact lookup against a synthetic
bundle written to ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

from bucket_brigade.baselines.release import (
    ArtifactEntry,
    LOCAL_RELEASE_DIR,
    MANIFEST_FILENAME,
    Manifest,
    iter_release_kinds,
    release_path,
    resolve_artifact_path,
    save_manifest,
)
from bucket_brigade.baselines.release.paths import (
    _ENV_OVERRIDE,
    load_release_manifest,
)


def _build_synthetic_bundle(root: Path) -> None:
    """Create a manifest + one artifact file under ``root``."""
    archetypes = root / "archetypes"
    archetypes.mkdir(parents=True, exist_ok=True)
    (archetypes / "hero.pkl").write_bytes(b"\x00fake-pickle\x00")

    manifest = Manifest(
        release_version="0.1.0-test",
        release_date="2026-06-08",
        artifacts=[
            ArtifactEntry(
                kind="archetype",
                name="hero",
                filename="archetypes/hero.pkl",
                size_bytes=13,
                notes="synthetic test bundle",
            )
        ],
    )
    save_manifest(manifest, root / MANIFEST_FILENAME)


@pytest.fixture()
def env_override_bundle(tmp_path: Path) -> Iterator[Path]:
    """Set $BUCKET_BRIGADE_BASELINES_DIR to a synthetic bundle dir."""
    _build_synthetic_bundle(tmp_path)
    old = os.environ.get(_ENV_OVERRIDE)
    os.environ[_ENV_OVERRIDE] = str(tmp_path)
    try:
        yield tmp_path
    finally:
        if old is None:
            os.environ.pop(_ENV_OVERRIDE, None)
        else:
            os.environ[_ENV_OVERRIDE] = old


def test_local_release_dir_is_inside_package() -> None:
    """The package-shipped local/ dir resolves to a path under bucket_brigade/."""
    # Even when empty, the path should be resolvable and ends in
    # baselines/release/local.
    assert str(LOCAL_RELEASE_DIR).endswith(str(Path("baselines") / "release" / "local"))


def test_iter_release_kinds_contract() -> None:
    kinds = list(iter_release_kinds())
    assert set(kinds) >= {"archetype", "nash", "ppo"}


def test_release_path_respects_env_override(env_override_bundle: Path) -> None:
    """When $BUCKET_BRIGADE_BASELINES_DIR is set, release_path() returns it."""
    assert release_path() == env_override_bundle.resolve()


def test_release_path_falls_back_when_override_missing(tmp_path: Path) -> None:
    """An override pointing at a nonexistent dir falls through to the package dir."""
    old = os.environ.get(_ENV_OVERRIDE)
    os.environ[_ENV_OVERRIDE] = str(tmp_path / "does-not-exist")
    try:
        # Should not raise; just fall back. Exact return is package-shipped
        # dir or user cache — both are non-None paths.
        out = release_path()
        assert isinstance(out, Path)
    finally:
        if old is None:
            os.environ.pop(_ENV_OVERRIDE, None)
        else:
            os.environ[_ENV_OVERRIDE] = old


def test_load_release_manifest_reads_synthetic_bundle(
    env_override_bundle: Path,
) -> None:
    m = load_release_manifest()
    assert m.release_version == "0.1.0-test"
    assert len(m.artifacts) == 1


def test_resolve_artifact_path_returns_existing_file(
    env_override_bundle: Path,
) -> None:
    p = resolve_artifact_path(kind="archetype", name="hero")
    assert p.exists()
    assert p.read_bytes() == b"\x00fake-pickle\x00"


def test_resolve_artifact_path_raises_keyerror_on_unknown(
    env_override_bundle: Path,
) -> None:
    with pytest.raises(KeyError, match="No artifact found"):
        resolve_artifact_path(kind="archetype", name="not-a-real-archetype")


def test_load_release_manifest_helpful_error_when_missing(tmp_path: Path) -> None:
    """When the override dir has no manifest, raise with a helpful hint."""
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No frozen-baseline manifest"):
        load_release_manifest(directory=empty)

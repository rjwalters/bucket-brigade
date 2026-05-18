"""
Tests for the Rust-extension staleness detector in ``tests/conftest.py``.

These tests exercise ``_rust_core_is_stale`` directly with monkeypatched
paths and mtimes. End-to-end testing of ``pytest_collection_modifyitems``
would require spawning a subprocess pytest run, which is brittle; here we
test the underlying detector instead and document a manual smoke test in
the PR body.

See issue #330 for context.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests import conftest as _conftest


def _touch_with_mtime(path: Path, mtime: float) -> None:
    """Create ``path`` (parents included) and set its mtime to ``mtime``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))


@pytest.fixture
def fake_rust_tree(tmp_path, monkeypatch):
    """Lay out a fake bucket-brigade-core tree and a fake installed .so.

    Layout::

        tmp_path/
            installed/
                bucket_brigade_core.cpython-darwin.so   <- the artifact
            source/
                src/
                    lib.rs
                    scenarios.rs
                Cargo.toml

    Returns ``(so_path, source_root)`` so tests can mutate mtimes freely.
    Monkeypatches ``_installed_rust_so_path`` and ``_rust_source_root`` so
    the detector sees this synthetic tree.
    """
    installed = tmp_path / "installed"
    source_root = tmp_path / "source"

    so_path = installed / "bucket_brigade_core.cpython-darwin.so"
    rs_files = [
        source_root / "src" / "lib.rs",
        source_root / "src" / "scenarios.rs",
    ]
    cargo_toml = source_root / "Cargo.toml"

    # Sources older than .so by default (fresh build state).
    for rs in rs_files:
        _touch_with_mtime(rs, mtime=1000.0)
    _touch_with_mtime(cargo_toml, mtime=1000.0)
    _touch_with_mtime(so_path, mtime=2000.0)

    monkeypatch.setattr(_conftest, "_installed_rust_so_path", lambda: so_path)
    monkeypatch.setattr(_conftest, "_rust_source_root", lambda: source_root)

    return so_path, source_root


class TestRustCoreIsStale:
    """Direct tests of the ``_rust_core_is_stale`` detector."""

    def test_fresh_build_is_not_stale(self, fake_rust_tree):
        """`.so` newer than all sources -> not stale."""
        assert _conftest._rust_core_is_stale() is False

    def test_newer_rs_file_triggers_stale(self, fake_rust_tree):
        """A `.rs` file newer than the `.so` -> stale."""
        _, source_root = fake_rust_tree
        os.utime(source_root / "src" / "scenarios.rs", (3000.0, 3000.0))
        assert _conftest._rust_core_is_stale() is True

    def test_newer_cargo_toml_triggers_stale(self, fake_rust_tree):
        """`Cargo.toml` newer than the `.so` -> stale (catches dep bumps)."""
        _, source_root = fake_rust_tree
        os.utime(source_root / "Cargo.toml", (3000.0, 3000.0))
        assert _conftest._rust_core_is_stale() is True

    def test_equal_mtime_is_not_stale(self, fake_rust_tree):
        """Source and `.so` at the same mtime -> not stale (boundary)."""
        so_path, source_root = fake_rust_tree
        os.utime(source_root / "src" / "lib.rs", (2000.0, 2000.0))
        os.utime(so_path, (2000.0, 2000.0))
        assert _conftest._rust_core_is_stale() is False

    def test_nested_rs_file_is_detected(self, fake_rust_tree, tmp_path):
        """Newly added deep-nested `.rs` files are picked up by rglob."""
        _, source_root = fake_rust_tree
        deep_rs = source_root / "src" / "agents" / "policy.rs"
        _touch_with_mtime(deep_rs, mtime=4000.0)
        assert _conftest._rust_core_is_stale() is True

    def test_missing_so_returns_false(self, monkeypatch):
        """No `.so` -> absence check handles this, detector returns False."""
        monkeypatch.setattr(_conftest, "_installed_rust_so_path", lambda: None)
        assert _conftest._rust_core_is_stale() is False

    def test_missing_source_root_returns_false(self, tmp_path, monkeypatch):
        """No source tree (e.g. wheel install) -> conservative False."""
        so_path = tmp_path / "fake.so"
        _touch_with_mtime(so_path, mtime=1000.0)
        monkeypatch.setattr(_conftest, "_installed_rust_so_path", lambda: so_path)
        monkeypatch.setattr(_conftest, "_rust_source_root", lambda: None)
        assert _conftest._rust_core_is_stale() is False

    def test_so_path_does_not_exist_returns_false(self, tmp_path, monkeypatch):
        """`.so` path returned but file doesn't exist -> conservative False."""
        ghost_so = tmp_path / "nonexistent.so"
        monkeypatch.setattr(_conftest, "_installed_rust_so_path", lambda: ghost_so)
        monkeypatch.setattr(_conftest, "_rust_source_root", lambda: tmp_path)
        assert _conftest._rust_core_is_stale() is False

    def test_empty_source_tree_returns_false(self, tmp_path, monkeypatch):
        """Source root with no `.rs` files and no `Cargo.toml` -> False."""
        installed = tmp_path / "installed"
        source_root = tmp_path / "source"
        (source_root / "src").mkdir(parents=True)
        so_path = installed / "fake.so"
        _touch_with_mtime(so_path, mtime=1000.0)

        monkeypatch.setattr(_conftest, "_installed_rust_so_path", lambda: so_path)
        monkeypatch.setattr(_conftest, "_rust_source_root", lambda: source_root)
        assert _conftest._rust_core_is_stale() is False


class TestRustSourceRootResolution:
    """Tests for ``_rust_source_root`` path resolution."""

    def test_finds_actual_source_root(self):
        """In the real repo checkout, the detector finds bucket-brigade-core/."""
        root = _conftest._rust_source_root()
        # We're running from a worktree of bucket-brigade — source root must exist.
        assert root is not None
        assert root.name == "bucket-brigade-core"
        assert (root / "src").is_dir()
        assert (root / "Cargo.toml").is_file()


class TestRustEscapeHatchConstants:
    """Sanity checks on the escape-hatch env var name constants."""

    def test_missing_hatch_name(self):
        assert _conftest.RUST_ESCAPE_HATCH_ENV == "BUCKET_BRIGADE_ALLOW_MISSING_RUST"

    def test_stale_hatch_name(self):
        assert (
            _conftest.RUST_STALE_ESCAPE_HATCH_ENV == "BUCKET_BRIGADE_ALLOW_STALE_RUST"
        )

    def test_build_script_path(self):
        assert _conftest.RUST_BUILD_SCRIPT == "bucket-brigade-core/build.sh"

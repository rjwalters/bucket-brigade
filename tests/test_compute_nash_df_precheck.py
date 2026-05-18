"""
Tests for the disk-space precheck helper used by Nash compute scripts.

Issue #269: ``compute_nash.py`` (and ``compute_nash_v2.py``) burned ~33 min of
simulation compute and then crashed when writing the final ``equilibrium.json``
because the filesystem was full. These tests exercise the startup-time
precheck that aborts BEFORE any simulation work runs.

The helper module ``experiments/scripts/_disk_precheck.py`` is imported by
path because the ``experiments/scripts`` directory is not a proper Python
package.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections import namedtuple
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PRECHECK_PATH = REPO_ROOT / "experiments" / "scripts" / "_disk_precheck.py"


def _load_precheck_module():
    """Load ``_disk_precheck.py`` as a standalone module for testing."""
    spec = importlib.util.spec_from_file_location("_disk_precheck", PRECHECK_PATH)
    assert spec is not None and spec.loader is not None, (
        f"Could not load spec for {PRECHECK_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_disk_precheck"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def precheck():
    """Provide the loaded ``_disk_precheck`` module."""
    return _load_precheck_module()


# ``os.statvfs`` returns an object with f_bavail (available blocks) and
# f_frsize (fragment size in bytes). We mock with a namedtuple that exposes
# both attributes.
FakeStatvfs = namedtuple("FakeStatvfs", ["f_bavail", "f_frsize"])


def _make_statvfs(free_bytes: int) -> FakeStatvfs:
    """Return a fake ``statvfs_result`` reporting ``free_bytes`` available."""
    # Use 4 KiB blocks; the precheck only cares about the product.
    frsize = 4096
    bavail = free_bytes // frsize
    return FakeStatvfs(f_bavail=bavail, f_frsize=frsize)


def test_passes_when_plenty_of_space(precheck, monkeypatch, tmp_path):
    """With 10 GiB free, the precheck should not exit."""
    monkeypatch.setattr(
        precheck.os,
        "statvfs",
        lambda _path: _make_statvfs(10 * 1024 * 1024 * 1024),  # 10 GiB
    )
    # Should return None and not raise SystemExit.
    assert precheck.check_free_space(tmp_path, min_free_mib=100) is None


def test_aborts_when_below_threshold(precheck, monkeypatch, tmp_path, capsys):
    """With only 5 MiB free and a 100 MiB threshold, the precheck must exit 1."""
    monkeypatch.setattr(
        precheck.os,
        "statvfs",
        lambda _path: _make_statvfs(5 * 1024 * 1024),  # 5 MiB
    )

    with pytest.raises(SystemExit) as excinfo:
        precheck.check_free_space(tmp_path, min_free_mib=100)

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "ERROR" in err
    assert "MiB free" in err
    assert "100 MiB" in err
    assert "Aborting before compute" in err


def test_aborts_when_zero_free_space(precheck, monkeypatch, tmp_path, capsys):
    """ENOSPC-equivalent: 0 bytes free should fail the precheck."""
    monkeypatch.setattr(
        precheck.os,
        "statvfs",
        lambda _path: _make_statvfs(0),
    )

    with pytest.raises(SystemExit) as excinfo:
        precheck.check_free_space(tmp_path, min_free_mib=100)

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "0.0 MiB free" in err


def test_threshold_at_exact_boundary_passes(precheck, monkeypatch, tmp_path):
    """Free bytes exactly at the threshold should be considered sufficient."""
    threshold_mib = 100
    free_bytes = threshold_mib * 1024 * 1024
    monkeypatch.setattr(
        precheck.os,
        "statvfs",
        lambda _path: _make_statvfs(free_bytes),
    )
    # Equal-to-threshold passes (the check is strict `<`).
    assert precheck.check_free_space(tmp_path, min_free_mib=threshold_mib) is None


def test_resolves_nonexistent_output_dir_to_parent(
    precheck, monkeypatch, tmp_path, capsys
):
    """If output-dir does not exist, the precheck statvfs's its nearest ancestor."""
    target = tmp_path / "does" / "not" / "exist" / "yet"
    seen_paths: list[Path] = []

    def fake_statvfs(path):
        seen_paths.append(Path(path))
        return _make_statvfs(10 * 1024 * 1024 * 1024)  # 10 GiB

    monkeypatch.setattr(precheck.os, "statvfs", fake_statvfs)

    precheck.check_free_space(target, min_free_mib=100)

    # Should have statvfs'd an existing ancestor (tmp_path itself or one of its
    # ancestors), not the non-existent target path.
    assert seen_paths, "statvfs should have been called"
    used = seen_paths[0]
    assert used.exists(), f"precheck must call statvfs on an existing path, got {used}"


def test_error_message_names_the_volume(precheck, monkeypatch, tmp_path, capsys):
    """The error message should include the resolved filesystem path."""
    monkeypatch.setattr(
        precheck.os,
        "statvfs",
        lambda _path: _make_statvfs(1024),  # 1 KiB free
    )

    with pytest.raises(SystemExit):
        precheck.check_free_space(tmp_path, min_free_mib=100)

    err = capsys.readouterr().err
    # The resolved path is the nearest existing ancestor of tmp_path
    # (which is tmp_path itself, since pytest created it).
    assert str(tmp_path.resolve()) in err


def test_statvfs_oserror_aborts(precheck, monkeypatch, tmp_path, capsys):
    """If statvfs itself fails, abort with a clear error rather than crashing later."""

    def raising_statvfs(_path):
        raise OSError("simulated stat failure")

    monkeypatch.setattr(precheck.os, "statvfs", raising_statvfs)

    with pytest.raises(SystemExit) as excinfo:
        precheck.check_free_space(tmp_path, min_free_mib=100)

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "could not stat filesystem" in err
    assert "Aborting before compute" in err


def test_default_threshold_is_100_mib(precheck):
    """The conservative default threshold should be 100 MiB (per issue #269)."""
    assert precheck.DEFAULT_MIN_FREE_MIB == 100


def test_real_filesystem_has_free_space(precheck, tmp_path):
    """Smoke test: a real tmp dir on a normal CI box should pass the precheck.

    This guards against breakage in ``_resolve_existing_parent`` / argument
    plumbing without mocking ``os.statvfs``. If the host running tests really
    has less than 100 MiB free, we skip rather than fail.
    """
    stats = os.statvfs(tmp_path)
    free_bytes = stats.f_bavail * stats.f_frsize
    if free_bytes < 100 * 1024 * 1024:
        pytest.skip("host has <100 MiB free; cannot run unmocked precheck smoke test")

    # Should not raise.
    assert precheck.check_free_space(tmp_path, min_free_mib=100) is None

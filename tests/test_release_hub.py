"""Tests for the HuggingFace integration (issue #373).

These tests do NOT make any real network requests. They:

1. Verify dry-run upload validates the manifest and prints a summary
   without touching the network.
2. Verify the upload path detects missing files and raises a clear
   error.
3. Mock :mod:`huggingface_hub` to exercise the real-upload code path
   without contacting HF.
4. Verify the optional-import error message points users at the right
   install extra when :mod:`huggingface_hub` is missing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest

from bucket_brigade.baselines.release import (
    ArtifactEntry,
    MANIFEST_FILENAME,
    Manifest,
    save_manifest,
)
from bucket_brigade.baselines.release import hub as hub_module
from bucket_brigade.baselines.release.hub import (
    _INSTALL_HINT,
    upload_release,
)


def _build_valid_bundle(root: Path) -> Manifest:
    (root / "archetypes").mkdir(parents=True, exist_ok=True)
    (root / "archetypes" / "hero.pkl").write_bytes(b"fake-pickle")
    m = Manifest(
        release_version="0.1.0",
        release_date="2026-06-08",
        artifacts=[
            ArtifactEntry(
                kind="archetype",
                name="hero",
                filename="archetypes/hero.pkl",
                size_bytes=11,
            )
        ],
    )
    save_manifest(m, root / MANIFEST_FILENAME)
    return m


def test_upload_release_dry_run_succeeds_without_huggingface(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Dry-run must NOT import huggingface_hub or hit the network."""
    _build_valid_bundle(tmp_path)
    out = upload_release(source_dir=tmp_path, dry_run=True)
    assert out == "DRY-RUN"
    captured = capsys.readouterr()
    assert "DRY RUN" in captured.out
    assert "archetypes/hero.pkl" in captured.out


def test_upload_release_detects_missing_files(tmp_path: Path) -> None:
    """A manifest referencing a missing file should fail validation."""
    # Build a manifest pointing at a file we never create.
    m = Manifest(
        release_version="0.1.0",
        release_date="2026-06-08",
        artifacts=[
            ArtifactEntry(
                kind="archetype",
                name="ghost",
                filename="archetypes/ghost.pkl",
            )
        ],
    )
    save_manifest(m, tmp_path / MANIFEST_FILENAME)

    with pytest.raises(ValueError, match="missing file"):
        upload_release(source_dir=tmp_path, dry_run=True)


def test_upload_release_no_manifest_is_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no manifest.json"):
        upload_release(source_dir=tmp_path, dry_run=True)


def test_upload_release_real_path_uses_huggingface_hub(tmp_path: Path) -> None:
    """When dry_run=False, we call huggingface_hub.upload_folder.

    We monkey-patch ``_import_hf`` so the test does not need
    :mod:`huggingface_hub` installed and never hits the network.
    """
    _build_valid_bundle(tmp_path)

    fake_hub = types.SimpleNamespace(
        upload_folder=mock.MagicMock(return_value="https://example/commit"),
    )
    with mock.patch.object(hub_module, "_import_hf", return_value=fake_hub):
        out = upload_release(
            source_dir=tmp_path,
            repo_id="org/repo",
            dry_run=False,
        )

    fake_hub.upload_folder.assert_called_once()
    call_kwargs: Dict[str, Any] = fake_hub.upload_folder.call_args.kwargs
    assert call_kwargs["repo_id"] == "org/repo"
    assert call_kwargs["repo_type"] == "model"
    assert "0.1.0" in call_kwargs["commit_message"]
    assert out == "https://example/commit"


def test_download_release_validates_snapshot_has_manifest(tmp_path: Path) -> None:
    """download_release should raise if the snapshot has no manifest."""
    # Pretend snapshot_download returns an empty dir.
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()

    fake_hub = types.SimpleNamespace(
        snapshot_download=mock.MagicMock(return_value=str(snapshot_dir)),
    )
    with mock.patch.object(hub_module, "_import_hf", return_value=fake_hub):
        with pytest.raises(FileNotFoundError, match="no manifest.json"):
            hub_module.download_release(cache_dir=snapshot_dir)


def test_import_hf_raises_helpful_error_when_missing() -> None:
    """If huggingface_hub is not installed, raise with the install hint.

    We simulate "missing module" by injecting a sentinel ImportError via
    sys.modules and calling _import_hf in a context where the real import
    is shadowed.
    """
    # Hide huggingface_hub from importlib by forcing an ImportError.
    sentinel = "__hub_test_blocker__"
    saved = sys.modules.pop("huggingface_hub", None)
    sys.modules["huggingface_hub"] = sentinel  # type: ignore[assignment]
    try:
        # Calling _import_hf with a non-module entry in sys.modules will
        # succeed (Python returns whatever you put in sys.modules), so we
        # need a stronger guard: directly monkey-patch the import.
        with mock.patch(
            "builtins.__import__",
            side_effect=ImportError("simulated missing huggingface_hub"),
        ):
            with pytest.raises(ImportError) as exc_info:
                hub_module._import_hf()
        assert _INSTALL_HINT in str(exc_info.value)
    finally:
        # Restore module table.
        if saved is None:
            sys.modules.pop("huggingface_hub", None)
        else:
            sys.modules["huggingface_hub"] = saved

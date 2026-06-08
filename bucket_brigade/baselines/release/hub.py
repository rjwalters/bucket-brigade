"""HuggingFace Hub integration for frozen baseline release bundles (#373).

Thin wrappers around :mod:`huggingface_hub` for two operations:

- :func:`download_release`: fetch a release bundle (manifest + all
  artifacts) into the local cache, so subsequent calls to
  :func:`bucket_brigade.baselines.release.resolve_artifact_path` work
  offline.
- :func:`upload_release`: push a release bundle from a local directory
  to a HuggingFace repo. **Operator-only** — never invoked
  automatically.

This module is gated by the optional ``[huggingface]`` install extra
(see ``pyproject.toml``). If :mod:`huggingface_hub` is not installed,
any function call raises :class:`ImportError` with a hint pointing at
the install command.

Why this is the *only* place the package touches HuggingFace
------------------------------------------------------------

The wider package never imports :mod:`huggingface_hub` at module
import time. That keeps a vanilla
``pip install bucket-brigade && python -c "import bucket_brigade"``
working with **zero** network dependency and minimal install
footprint. The HuggingFace path is opt-in: users who want it install
the ``[huggingface]`` extra and explicitly call
:func:`download_release`.

Slice #371 will produce real artifacts and operators will then run
``python -m scripts.release.upload_to_hf --confirm`` (see that script
for the full operator workflow). Until #371 lands, these functions
are wired up but produce no real artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .manifest import (
    DEFAULT_HUGGINGFACE_REPO,
    MANIFEST_FILENAME,
    Manifest,
    load_manifest,
)
from .paths import _user_cache_dir  # noqa: PLC2701 — intentional internal share


_INSTALL_HINT: str = (
    "huggingface_hub is required for this operation. "
    "Install it with: pip install 'bucket-brigade[huggingface]'"
)


def _import_hf():  # type: ignore[no-untyped-def]
    """Import :mod:`huggingface_hub` lazily with a friendly error.

    Returns the module, or raises :class:`ImportError` with
    :data:`_INSTALL_HINT` if the optional dep is missing.
    """
    try:
        import huggingface_hub  # lazy import is the point
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(_INSTALL_HINT) from exc
    return huggingface_hub


def download_release(
    repo_id: str = DEFAULT_HUGGINGFACE_REPO,
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    *,
    token: Optional[str] = None,
) -> Path:
    """Download a release bundle from a HuggingFace repo into a local dir.

    The full repo snapshot is fetched (manifest + all artifacts) so
    subsequent ``resolve_artifact_path`` calls work offline. By default
    the snapshot lands in
    ``~/.cache/bucket_brigade/baselines/<release_version>/``.

    Args:
        repo_id: HuggingFace repo ID (``"org/repo"``). Defaults to
            :data:`DEFAULT_HUGGINGFACE_REPO`.
        revision: Git revision (branch, tag, or commit). ``None`` uses
            the default branch.
        cache_dir: Directory to put the snapshot in. If ``None`` we
            place it under
            ``~/.cache/bucket_brigade/baselines/<release_version>/``
            once we know the version from the downloaded manifest. If
            you pre-specify a directory, the contents of the snapshot
            are placed directly inside.
        token: Optional HuggingFace API token (only needed for private
            repos; the project repo is public).

    Returns:
        Absolute path to the directory containing ``manifest.json``.

    Raises:
        ImportError: If :mod:`huggingface_hub` is not installed.
        OSError: For network / disk errors propagated from
            :func:`huggingface_hub.snapshot_download`.
    """
    hf = _import_hf()
    # First-stage: download to a temp-ish location keyed by repo_id +
    # revision. We do this in two stages so we can rename to a
    # release_version-keyed dir once we've read the manifest.
    initial_dest = cache_dir
    if initial_dest is None:
        initial_dest = _user_cache_dir() / "_snapshots" / repo_id.replace("/", "__")
    initial_dest = Path(initial_dest)
    initial_dest.mkdir(parents=True, exist_ok=True)

    snapshot_dir = hf.snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(initial_dest),
        token=token,
        repo_type="model",
    )
    snapshot_dir = Path(snapshot_dir)

    manifest_path = snapshot_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Downloaded snapshot at {snapshot_dir} has no {MANIFEST_FILENAME}. "
            f"Repo {repo_id!r} does not look like a Bucket Brigade release "
            "bundle. Did you pass the right repo_id?"
        )

    if cache_dir is None:
        # User did not override → move to version-keyed cache dir for
        # reproducibility across processes.
        manifest = load_manifest(manifest_path)
        version_dir = _user_cache_dir(manifest.release_version)
        if not version_dir.exists():
            version_dir.parent.mkdir(parents=True, exist_ok=True)
            # On most platforms the snapshot dir lives inside the cache
            # tree already, so a rename is cheap. If rename fails
            # (cross-device), fall back to copy.
            try:
                snapshot_dir.rename(version_dir)
            except OSError:  # pragma: no cover - cross-device fallback
                import shutil

                shutil.copytree(snapshot_dir, version_dir)
        return version_dir

    return snapshot_dir


def upload_release(
    source_dir: Path,
    repo_id: str = DEFAULT_HUGGINGFACE_REPO,
    *,
    token: Optional[str] = None,
    commit_message: Optional[str] = None,
    dry_run: bool = True,
) -> str:
    """Upload a local release bundle to a HuggingFace repo.

    **Operator-only** — this function is invoked by the
    ``scripts/release/upload_to_hf.py`` CLI under explicit
    ``--confirm`` from a human. It is **never** called by tests or
    automation in this repo.

    Args:
        source_dir: Directory containing ``manifest.json`` and all the
            artifact files referenced by the manifest. Must exist and
            contain a valid manifest.
        repo_id: HuggingFace repo ID. Defaults to
            :data:`DEFAULT_HUGGINGFACE_REPO`.
        token: HuggingFace API token with write access. If ``None``,
            :mod:`huggingface_hub` will look in
            ``$HF_TOKEN`` / ``~/.cache/huggingface/token``.
        commit_message: Git commit message for the HF repo. Defaults to
            ``"Upload bucket-brigade baselines <release_version>"``.
        dry_run: If ``True`` (default), validate inputs and print what
            *would* be uploaded but do NOT push anything. The CLI
            forces this to ``False`` only when the operator passes
            ``--confirm``.

    Returns:
        The commit URL on success, or the string ``"DRY-RUN"`` when
        ``dry_run=True``.

    Raises:
        ImportError: If :mod:`huggingface_hub` is not installed.
        FileNotFoundError: If ``source_dir`` lacks a manifest.
        ValueError: If the manifest references files that don't exist
            in ``source_dir``.
    """
    source_dir = Path(source_dir).resolve()
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"upload_release: no {MANIFEST_FILENAME} in {source_dir}."
        )
    manifest = load_manifest(manifest_path)
    _validate_manifest_files_exist(manifest, source_dir)

    if commit_message is None:
        commit_message = f"Upload bucket-brigade baselines {manifest.release_version}"

    if dry_run:
        # Print a summary for the operator and return.
        print(
            f"[DRY RUN] Would upload {len(manifest.artifacts)} artifacts "
            f"from {source_dir} to {repo_id} (commit: {commit_message!r})"
        )
        for a in manifest.artifacts:
            print(f"  - {a.kind:9s} {a.name:30s} {a.filename}")
        return "DRY-RUN"

    hf = _import_hf()
    commit = hf.upload_folder(
        folder_path=str(source_dir),
        repo_id=repo_id,
        token=token,
        commit_message=commit_message,
        repo_type="model",
    )
    return str(commit)


def _validate_manifest_files_exist(manifest: Manifest, source_dir: Path) -> None:
    """Raise :class:`ValueError` if any manifest entry has no file on disk."""
    missing = []
    for a in manifest.artifacts:
        if not (source_dir / a.filename).exists():
            missing.append(a.filename)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Manifest references {len(missing)} missing file(s) in "
            f"{source_dir}: {joined}"
        )

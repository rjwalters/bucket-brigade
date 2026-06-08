"""Local path resolution for frozen baseline release artifacts (#373).

This module knows **where** release artifacts live on the local
filesystem. It does NOT know how to deserialize them (that's #371's
job once it lands). It also does not fetch from HuggingFace — see
:mod:`bucket_brigade.baselines.release.hub` for that.

Layout convention
-----------------

Inside the package:

::

    bucket_brigade/baselines/release/local/
        manifest.json
        archetypes/
            hero.pkl
            firefighter.pkl
            ...
        nash/
            minimal_specialization-v1.json
            ...
        ppo/
            minimal_specialization-v1.pt
            ...

The ``local/`` directory is shipped inside the pip wheel (declared via
``hatch.build.targets.wheel.packages`` in ``pyproject.toml``).
Slice #371 populates it with the actual artifact files. Until then it
contains only ``.gitkeep`` and a ``README.md`` describing the
intended contents.

Lookup order
------------

The lookup order is:

1. Environment override: ``BUCKET_BRIGADE_BASELINES_DIR`` (absolute
   path). Used by CI / operators who want to point at a different
   bundle without reinstalling the wheel.
2. Package-shipped ``bucket_brigade/baselines/release/local/``.
3. ``~/.cache/bucket_brigade/baselines/<release_version>/`` — the
   default destination of :func:`hub.download_release` so that
   downloads survive across processes.

Callers who want HuggingFace fallback should use
:mod:`.hub.snapshot_into_cache` first, then call
:func:`resolve_artifact_path` — i.e., the local resolver never makes
network requests on its own.
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path
from typing import Iterator, Optional

from .manifest import (
    ARTIFACT_KINDS,
    MANIFEST_FILENAME,
    Manifest,
    load_manifest,
)


# Environment variable for overriding the local release directory.
_ENV_OVERRIDE: str = "BUCKET_BRIGADE_BASELINES_DIR"


def _package_local_dir() -> Path:
    """Path to the wheel-shipped ``local/`` directory.

    Uses :mod:`importlib.resources` so it works whether the package
    is installed (zipped or unzipped) or running from an editable
    source checkout.
    """
    # importlib.resources.files is the modern (3.9+) entry point.
    pkg_files = importlib.resources.files("bucket_brigade.baselines.release")
    return Path(str(pkg_files)) / "local"


def _user_cache_dir(release_version: Optional[str] = None) -> Path:
    """Default per-user cache for downloaded release bundles."""
    base = Path.home() / ".cache" / "bucket_brigade" / "baselines"
    if release_version:
        return base / release_version
    return base


# Public name for the wheel-shipped local directory. Tests and
# operators import this to know where #371's artifacts should land.
LOCAL_RELEASE_DIR: Path = _package_local_dir()


def iter_release_kinds() -> Iterator[str]:
    """Yield the canonical artifact kinds (``archetype``, ``nash``, ``ppo``).

    Exposed here so callers can iterate without importing
    :mod:`.manifest` directly.
    """
    yield from ARTIFACT_KINDS


def release_path() -> Path:
    """Return the directory that should contain ``manifest.json``.

    Resolution order (see module docstring):

    1. ``$BUCKET_BRIGADE_BASELINES_DIR`` if set and pointing at an
       existing directory.
    2. The wheel-shipped ``bucket_brigade/baselines/release/local/``.
    3. ``~/.cache/bucket_brigade/baselines/`` (top-level cache;
       per-release-version subdirs live underneath).

    The returned directory is **not** guaranteed to contain a
    manifest yet — callers must handle ``FileNotFoundError`` from
    :func:`load_release_manifest` if the bundle has not been
    populated or downloaded.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        candidate = Path(override).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        # Fall through to package-shipped path; we deliberately do NOT
        # raise on a missing override directory so test environments
        # can set the variable optimistically.

    pkg_dir = _package_local_dir()
    if pkg_dir.exists():
        return pkg_dir

    return _user_cache_dir()


def load_release_manifest(directory: Optional[Path] = None) -> Manifest:
    """Load the manifest from ``directory`` (or :func:`release_path`).

    Raises:
        FileNotFoundError: If no manifest is present.
        ValueError: If the manifest is malformed (see
            :class:`bucket_brigade.baselines.release.manifest.Manifest`).
    """
    directory = directory if directory is not None else release_path()
    manifest_path = Path(directory) / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No frozen-baseline manifest found at {manifest_path}. "
            "Either (a) install a release bundle into "
            f"{directory}, (b) set ${_ENV_OVERRIDE} to a directory "
            "containing one, or (c) call "
            "bucket_brigade.baselines.release.hub.download_release() "
            "to fetch from HuggingFace."
        )
    return load_manifest(manifest_path)


def resolve_artifact_path(
    kind: str,
    name: str,
    scenario_id: Optional[str] = None,
    directory: Optional[Path] = None,
) -> Path:
    """Locate the on-disk file for a single artifact.

    Args:
        kind: One of :data:`ARTIFACT_KINDS`.
        name: Artifact name (see :class:`ArtifactEntry`).
        scenario_id: Optional scenario ID. Must match exactly
            (including ``None``) — there is no fuzzy matching.
        directory: Directory containing ``manifest.json``. Defaults
            to :func:`release_path`.

    Returns:
        Absolute :class:`pathlib.Path` to the artifact file.

    Raises:
        FileNotFoundError: If the manifest or the artifact file is
            missing.
        KeyError: If no manifest entry matches.
    """
    directory = directory if directory is not None else release_path()
    manifest = load_release_manifest(directory)
    entry = manifest.find(kind=kind, name=name, scenario_id=scenario_id)
    if entry is None:
        raise KeyError(
            f"No artifact found for (kind={kind!r}, name={name!r}, "
            f"scenario_id={scenario_id!r}) in manifest at "
            f"{directory / MANIFEST_FILENAME}."
        )
    abs_path = Path(directory) / entry.filename
    if not abs_path.exists():
        raise FileNotFoundError(
            f"Manifest entry refers to {entry.filename!r} but file is "
            f"missing at {abs_path}. Re-download the release bundle "
            "or rebuild the wheel."
        )
    return abs_path

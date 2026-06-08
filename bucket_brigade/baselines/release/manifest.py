"""Manifest schema for frozen baseline release bundles (issue #373).

A **manifest** is a single ``manifest.json`` file that catalogs every
artifact in a release bundle (archetypes, Nash vectors, PPO
checkpoints). It serves two purposes:

1. **Local discovery**: when a user calls
   ``bucket_brigade.baselines.release.resolve_artifact_path(...)``, the
   loader reads the manifest to find the file on disk.
2. **HuggingFace mirroring**: the manifest is uploaded alongside the
   artifacts to a public HuggingFace repo. Downstream users (or CI
   smoke tests) can fetch the bundle via
   :func:`bucket_brigade.baselines.release.hub.download_release` and
   the manifest tells them which file is which.

Schema (``manifest.json``)
--------------------------

::

    {
      "schema_version": 1,
      "release_version": "0.1.0",
      "release_date": "2026-06-08",
      "source_commit": "abc123",
      "artifacts": [
        {
          "kind": "archetype",
          "name": "hero",
          "filename": "archetypes/hero.pkl",
          "scenario_id": null,
          "sha256": "deadbeef...",
          "size_bytes": 12345,
          "notes": "Hand-coded Hero archetype, see archetypes.py"
        },
        ...
      ],
      "huggingface_repo": "rjwalters/bucket-brigade-baselines",
      "extra": {}
    }

The schema is intentionally small and forward-compatible. Adding new
top-level fields or new ``ArtifactEntry`` fields is OK; **renaming or
removing** fields requires bumping :data:`MANIFEST_SCHEMA_VERSION`.

The current ``schema_version=1`` is finalised by this slice (#373).
Slice #371 (frozen baselines) will populate the artifact list and
ship the first ``manifest.json``. Until then,
:data:`MANIFEST_FILENAME` and this schema are the **contract** between
the two slices.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Bumped only on incompatible schema changes (renames, removals,
# semantic changes to existing fields). Additive changes do not bump.
MANIFEST_SCHEMA_VERSION: int = 1

# Canonical manifest filename. Hardcoded across local + HF copies so
# the loader can find it without configuration.
MANIFEST_FILENAME: str = "manifest.json"

# Default HuggingFace repo ID used by the upload/download helpers when
# the caller does not override. The repo does not need to exist for
# this constant to be useful — it documents the *intent*, and the
# operator running the upload script gets to confirm before push.
#
# Operators publishing the first real bundle should:
#   1. Create the repo on https://huggingface.co under the project org.
#   2. Confirm this constant matches the actual repo ID; bump if not.
#   3. Run `python -m scripts.release.upload_to_hf --confirm`.
DEFAULT_HUGGINGFACE_REPO: str = "rjwalters/bucket-brigade-baselines"


# Accepted artifact kinds. Kept as a string-typed module constant
# rather than an Enum so manifests written by older versions are still
# parseable.
ARTIFACT_KINDS = ("archetype", "nash", "ppo")


@dataclass(frozen=True)
class ArtifactEntry:
    """One row in the manifest's ``artifacts`` list.

    Attributes:
        kind: One of :data:`ARTIFACT_KINDS`. The loader uses this to
            dispatch to the right deserializer in
            :mod:`bucket_brigade.baselines` (when #371 lands).
        name: Stable identifier within (kind, scenario_id). For
            archetypes this is the archetype name (``"hero"``); for
            Nash/PPO it is typically the scenario base name.
        filename: Path **relative to the manifest's directory**. The
            loader joins ``manifest_dir / filename`` to get the
            absolute path. Use forward slashes for portability.
        scenario_id: Optional versioned scenario ID (e.g.
            ``"minimal_specialization-v1"``) the artifact applies to.
            ``None`` for scenario-agnostic artifacts (e.g. archetypes).
        sha256: Hex SHA-256 of the file contents. Used by the
            downloader to verify integrity after fetching from
            HuggingFace. Empty string when unknown / not yet computed.
        size_bytes: File size in bytes, for progress reporting.
        notes: Free-form human-readable description.
    """

    kind: str
    name: str
    filename: str
    scenario_id: Optional[str] = None
    sha256: str = ""
    size_bytes: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ARTIFACT_KINDS:
            raise ValueError(
                f"Unknown artifact kind {self.kind!r}; expected one of {ARTIFACT_KINDS}"
            )
        if not self.name:
            raise ValueError("ArtifactEntry.name must be non-empty")
        if not self.filename:
            raise ValueError("ArtifactEntry.filename must be non-empty")
        if "\\" in self.filename:
            raise ValueError(
                f"ArtifactEntry.filename {self.filename!r} contains a "
                "backslash; use forward slashes for portability."
            )


@dataclass(frozen=True)
class Manifest:
    """Top-level manifest object."""

    release_version: str
    release_date: str  # ISO-8601 date (YYYY-MM-DD)
    artifacts: List[ArtifactEntry] = field(default_factory=list)
    source_commit: str = ""
    huggingface_repo: str = DEFAULT_HUGGINGFACE_REPO
    schema_version: int = MANIFEST_SCHEMA_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-ready dict.

        The output round-trips cleanly through :func:`json.dumps` /
        :func:`json.loads` and is consumed by :func:`save_manifest`.
        """
        d: Dict[str, Any] = asdict(self)
        # Preserve the field order users expect at the top of the file.
        return {
            "schema_version": d["schema_version"],
            "release_version": d["release_version"],
            "release_date": d["release_date"],
            "source_commit": d["source_commit"],
            "huggingface_repo": d["huggingface_repo"],
            "artifacts": d["artifacts"],
            "extra": d["extra"],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Manifest":
        """Construct a Manifest from a parsed JSON dict.

        Raises:
            ValueError: If ``schema_version`` is newer than this
                module's :data:`MANIFEST_SCHEMA_VERSION` (forward
                incompatibility), or if required fields are missing.
        """
        if "schema_version" not in data:
            raise ValueError("manifest is missing required key 'schema_version'")
        sv = int(data["schema_version"])
        if sv > MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"manifest schema_version={sv} is newer than this client "
                f"supports (MANIFEST_SCHEMA_VERSION={MANIFEST_SCHEMA_VERSION}). "
                "Upgrade bucket-brigade to read this release."
            )
        for required in ("release_version", "release_date"):
            if required not in data:
                raise ValueError(f"manifest is missing required key {required!r}")
        artifacts_raw = data.get("artifacts", []) or []
        artifacts = [
            ArtifactEntry(
                kind=str(a["kind"]),
                name=str(a["name"]),
                filename=str(a["filename"]),
                scenario_id=a.get("scenario_id"),
                sha256=str(a.get("sha256", "")),
                size_bytes=int(a.get("size_bytes", 0)),
                notes=str(a.get("notes", "")),
            )
            for a in artifacts_raw
        ]
        return cls(
            release_version=str(data["release_version"]),
            release_date=str(data["release_date"]),
            artifacts=artifacts,
            source_commit=str(data.get("source_commit", "")),
            huggingface_repo=str(
                data.get("huggingface_repo", DEFAULT_HUGGINGFACE_REPO)
            ),
            schema_version=sv,
            extra=dict(data.get("extra", {}) or {}),
        )

    def find(
        self,
        kind: str,
        name: str,
        scenario_id: Optional[str] = None,
    ) -> Optional[ArtifactEntry]:
        """Look up a single artifact by ``(kind, name, scenario_id)``.

        Returns the first matching entry, or ``None`` if nothing matches.
        """
        for a in self.artifacts:
            if a.kind == kind and a.name == name and a.scenario_id == scenario_id:
                return a
        return None


def load_manifest(path: Path) -> Manifest:
    """Load a manifest from a JSON file on disk.

    Args:
        path: Path to ``manifest.json``.

    Returns:
        Parsed :class:`Manifest`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the manifest is malformed (see
            :meth:`Manifest.from_dict`).
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Manifest.from_dict(data)


def save_manifest(manifest: Manifest, path: Path) -> None:
    """Write a manifest to disk as pretty-printed JSON.

    Args:
        manifest: The :class:`Manifest` to serialize.
        path: Destination ``manifest.json`` path. Parent directories
            are created if missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=False)
        f.write("\n")

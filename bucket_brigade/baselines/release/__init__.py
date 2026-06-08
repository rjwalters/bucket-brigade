"""Frozen baseline distribution surface (issue #373, slice 5/5 of #365).

This subpackage is the **plumbing** for distributing frozen baseline
artifacts (archetype pickles, Nash equilibrium vectors, PPO checkpoints)
to downstream users — both as files shipped inside the pip wheel and as
files mirrored to a public HuggingFace repo.

The artifacts themselves (and the loader APIs that interpret them) are
the responsibility of the **#371 frozen baselines** slice. This module
defines only:

1. The manifest schema (:mod:`.manifest`) — what metadata travels with
   each artifact set, so the same files can be served from a local
   ``site-packages`` install, an in-repo dev path, or a HuggingFace
   download URL.
2. Local-path resolution (:mod:`.paths`) — given a scenario ID and a
   kind (``archetype`` / ``nash`` / ``ppo``), where on disk should we
   look first?
3. HuggingFace hub helpers (:mod:`.hub`) — thin wrappers around
   ``huggingface_hub.snapshot_download`` and ``upload_folder``, behind
   the optional ``[huggingface]`` install extra.

The plumbing is deliberately **artifact-agnostic** — it does not assume
any particular pickle format, network architecture, or scenario list,
so #371 can ship whatever shape it lands on without touching this layer.

See :mod:`bucket_brigade.baselines.release.manifest` for the schema and
``docs/RELEASE.md`` for the full operator workflow (wheel build,
PyPI publish, HuggingFace upload).
"""

from __future__ import annotations

from .loaders import (
    list_archetypes,
    list_nash_scenarios,
    list_ppo_scenarios,
    load_archetype,
    load_nash,
    load_nash_genomes,
    load_ppo,
)
from .manifest import (
    DEFAULT_HUGGINGFACE_REPO,
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA_VERSION,
    ArtifactEntry,
    Manifest,
    load_manifest,
    save_manifest,
)
from .paths import (
    LOCAL_RELEASE_DIR,
    iter_release_kinds,
    release_path,
    resolve_artifact_path,
)


__all__ = [
    # Manifest schema
    "ArtifactEntry",
    "Manifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "DEFAULT_HUGGINGFACE_REPO",
    "load_manifest",
    "save_manifest",
    # Local path resolution
    "LOCAL_RELEASE_DIR",
    "release_path",
    "resolve_artifact_path",
    "iter_release_kinds",
    # Loader API (#371)
    "load_archetype",
    "load_nash",
    "load_nash_genomes",
    "load_ppo",
    "list_archetypes",
    "list_nash_scenarios",
    "list_ppo_scenarios",
]

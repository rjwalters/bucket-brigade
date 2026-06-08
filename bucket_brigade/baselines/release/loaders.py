"""High-level loader API for frozen baseline release artifacts (issue #371).

This module sits one layer above
:func:`bucket_brigade.baselines.release.resolve_artifact_path` (which
gives you a :class:`pathlib.Path` to a file) and gives you back the
*deserialised* artifact: a numpy array for archetypes, a structured
dict for Nash equilibria, a path-or-state-dict for PPO checkpoints.

The three loaders satisfy the issue #371 acceptance criterion
"loader APIs return working agents/parameters" without forcing the
import of heavy dependencies (no torch import unless you call
:func:`load_ppo`).

Schema discovery
----------------

The artifacts must be produced by
:mod:`bucket_brigade.baselines.release.freeze` (or any tool that
writes the same JSON schemas — see that module's docstring). Once
frozen, the loaders here are stable: the JSON schemas are versioned
implicitly via :data:`bucket_brigade.baselines.release.manifest.
MANIFEST_SCHEMA_VERSION`.

Example
-------

::

    import bucket_brigade.baselines.release as release

    hero = release.load_archetype("hero")           # np.ndarray, shape (10,)
    ne = release.load_nash("minimal_specialization-v1")
    ne["team_payoff"]                                # -756.36
    ne["positions"][0]["genome"]                    # 10-element list[float]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .paths import resolve_artifact_path


def load_archetype(name: str, directory: Optional[Path] = None) -> np.ndarray:
    """Load a frozen archetype's parameter vector.

    Args:
        name: Archetype name (e.g. ``"hero"``, ``"firefighter"``).
            Matches the keys of
            :data:`bucket_brigade.agents.archetypes.ARCHETYPES`.
        directory: Directory containing ``manifest.json``. Defaults to
            :func:`bucket_brigade.baselines.release.paths.release_path`.

    Returns:
        ``np.ndarray`` of shape ``(10,)`` and dtype ``float64`` —
        identical (modulo dtype) to the in-memory vector in
        :mod:`bucket_brigade.agents.archetypes`.

    Raises:
        FileNotFoundError: If no manifest is present (the bundle hasn't
            been frozen / downloaded yet).
        KeyError: If ``name`` is not in the manifest.
    """
    path = resolve_artifact_path(kind="archetype", name=name, directory=directory)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return np.asarray(payload["genome"], dtype=np.float64)


def load_nash(
    scenario_id: str,
    name: Optional[str] = None,
    directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load a frozen Nash equilibrium for ``scenario_id``.

    Args:
        scenario_id: Versioned scenario ID, e.g.
            ``"minimal_specialization-v1"``.
        name: Manifest entry name when multiple NEs share a
            ``scenario_id`` (e.g. phase-diagram cells, where every
            entry has ``scenario_id="minimal_specialization-v1"`` but
            different per-(β, κ, c) names). Defaults to the
            scenario's base name (``scenario_id.rsplit("-v", 1)[0]``)
            so the common single-NE case "just works".
        directory: Override the manifest directory.

    Returns:
        Dict with the Nash payload, e.g.::

            {
                "scenario_id": "minimal_specialization-v1",
                "param_names": ["honesty_bias", ...],
                "team_payoff": -756.36,
                "per_position_payoffs": [-756.36, ...],
                "symmetric_profile": True,
                "profile_label": "hero | hero | hero | hero",
                "iterations": 17,
                "positions": [
                    {"position": 0, "closest_archetype": "hero(d=0.00)",
                     "genome": [1.0, 1.0, 1.0, ...]},
                    ...
                ],
                "source_parameters": {...},
            }

    Raises:
        KeyError: If no entry matches ``(scenario_id, name)``.
        FileNotFoundError: If no manifest is present.
    """
    lookup_name = name if name is not None else scenario_id.rsplit("-v", 1)[0]
    path = resolve_artifact_path(
        kind="nash",
        name=lookup_name,
        scenario_id=scenario_id,
        directory=directory,
    )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_nash_genomes(
    scenario_id: str,
    name: Optional[str] = None,
    directory: Optional[Path] = None,
) -> np.ndarray:
    """Return just the per-position genomes for a frozen Nash equilibrium.

    Convenience wrapper around :func:`load_nash` for the common case
    where the caller only wants the strategy vectors (shape
    ``(num_positions, 10)``).
    """
    payload = load_nash(scenario_id=scenario_id, name=name, directory=directory)
    positions: List[Dict[str, Any]] = list(payload.get("positions", []))
    if not positions:
        raise ValueError(
            f"Nash payload for {scenario_id!r} has no positions; "
            "the source freeze likely encountered a non-converged NE."
        )
    # Sort by position index to guarantee a stable agent ordering.
    positions.sort(key=lambda p: int(p.get("position", 0)))
    return np.asarray(
        [[float(x) for x in p["genome"]] for p in positions],
        dtype=np.float64,
    )


def load_ppo(
    scenario_id: str,
    name: Optional[str] = None,
    directory: Optional[Path] = None,
) -> Path:
    """Return the on-disk path to a frozen PPO checkpoint.

    We intentionally return a :class:`~pathlib.Path` rather than a
    deserialised ``torch`` state dict so the loader has zero torch
    dependency (and thus zero import cost when the caller is only
    after archetypes / Nash vectors). Callers wanting the state dict
    do::

        import torch
        from bucket_brigade.baselines.release import load_ppo

        ckpt_path = load_ppo("minimal_specialization-v1")
        state = torch.load(ckpt_path, map_location="cpu")

    Args:
        scenario_id: Versioned scenario ID.
        name: Manifest entry name; defaults to the scenario base name.
        directory: Override the manifest directory.

    Returns:
        Absolute :class:`pathlib.Path` to the ``.pt`` checkpoint.

    Raises:
        KeyError: If no PPO entry exists for the scenario yet (likely
            because issue #384 has not produced and committed the
            checkpoint).
        FileNotFoundError: If the manifest references a file that
            does not exist on disk.
    """
    lookup_name = name if name is not None else scenario_id.rsplit("-v", 1)[0]
    return resolve_artifact_path(
        kind="ppo",
        name=lookup_name,
        scenario_id=scenario_id,
        directory=directory,
    )


def list_archetypes(directory: Optional[Path] = None) -> List[str]:
    """List archetype names present in the frozen manifest.

    Returns:
        Sorted list of archetype names. Empty if no manifest is found
        or no archetype entries are present.
    """
    return _list_kind("archetype", directory=directory)


def list_nash_scenarios(directory: Optional[Path] = None) -> List[str]:
    """List versioned scenario IDs that have at least one frozen NE entry."""
    from .paths import load_release_manifest

    try:
        manifest = load_release_manifest(directory)
    except FileNotFoundError:
        return []
    scenarios = {
        a.scenario_id for a in manifest.artifacts if a.kind == "nash" and a.scenario_id
    }
    return sorted(scenarios)


def list_ppo_scenarios(directory: Optional[Path] = None) -> List[str]:
    """List versioned scenario IDs with a frozen PPO checkpoint."""
    from .paths import load_release_manifest

    try:
        manifest = load_release_manifest(directory)
    except FileNotFoundError:
        return []
    scenarios = {
        a.scenario_id for a in manifest.artifacts if a.kind == "ppo" and a.scenario_id
    }
    return sorted(scenarios)


def _list_kind(kind: str, directory: Optional[Path] = None) -> List[str]:
    """Sorted list of ``ArtifactEntry.name`` values for a given kind."""
    from .paths import load_release_manifest

    try:
        manifest = load_release_manifest(directory)
    except FileNotFoundError:
        return []
    return sorted(a.name for a in manifest.artifacts if a.kind == kind)

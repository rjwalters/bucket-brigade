"""Build a frozen-baseline release bundle from in-repo sources (issue #371).

This module is the **producer** for the release plumbing that landed in
#373 (``bucket_brigade/baselines/release/``). It walks the canonical
artifact sources in the repo and writes:

- ``archetypes/<name>.json`` — one JSON file per hand-coded archetype
  in :mod:`bucket_brigade.agents.archetypes`.
- ``nash/<scenario_id>.json`` — one JSON file per Nash equilibrium
  vector from the heterogeneous-oracle sweeps (currently
  ``minimal_specialization``, ``rest_trap``) plus the per-cell NE
  vectors from the phase-diagram preview (``experiments/nash/
  phase_diagram/preview/``).
- ``ppo/`` — **empty** until #384 (the operator-driven PPO baseline
  training task) lands real checkpoints. The freeze script reports
  this as a known gap rather than failing.
- ``manifest.json`` — a :class:`~bucket_brigade.baselines.release
  .manifest.Manifest` cataloguing every artifact above, conforming to
  the ``schema_version=1`` schema fixed by #373.

Why JSON, not pickle?
---------------------

Both archetypes and Nash strategy profiles are dense 10-element float
vectors with a small dict wrapper. JSON is:

- Language-agnostic (the wasm frontend / TypeScript code in ``web/``
  can read it directly).
- Diffable in code review (a parameter change shows up as a clean
  textual diff).
- Pickle-safe by construction (no arbitrary code execution risk for
  downstream users running ``load_archetype("hero")``).

PPO checkpoints will still be ``.pt`` (torch state-dicts) when #384
lands them — that's a binary format choice driven by the producer
(``experiments/p3_specialization/train.py``).

Usage
-----

From the repo root::

    python -m bucket_brigade.baselines.release.freeze

writes the bundle into the wheel-shipped
``bucket_brigade/baselines/release/local/`` directory. Use
``--dry-run`` to print the manifest without touching disk, or
``--output-dir`` to stage into a separate directory before review.

The freeze step is deterministic given the same source files — running
it twice produces byte-identical output. Operators committing a new
release should:

1. Run ``python -m bucket_brigade.baselines.release.freeze`` to update
   ``bucket_brigade/baselines/release/local/``.
2. Verify ``git diff bucket_brigade/baselines/release/local/`` looks
   like the intended change.
3. Commit, build a wheel (``uv build``), and ship.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess  # nosec B404 (only used to stamp manifest with git short SHA; argv is hardcoded)
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from bucket_brigade.agents.archetypes import ARCHETYPES

from .manifest import (
    MANIFEST_FILENAME,
    ArtifactEntry,
    Manifest,
    save_manifest,
)


# Default release version stamped into newly-frozen manifests when the
# operator does not pass --release-version. Bumped manually before a
# real release (the version controls the per-release HF cache subdir
# in :mod:`.paths`). 0.1.0 -> 0.1.1: annotation-only manifest update for
# the (kappa=0.90, c=0.50) phase-diagram cells (#459/#466); artifact
# bytes unchanged. 0.1.1 -> 0.1.2: manifest-only hash re-sync (#470)
# for the 4 (c=0.50) phase-diagram cells that #420's full-grid
# regeneration rewrote after the 0.1.0 freeze; shipped bytes unchanged
# since #420. 0.1.2 -> 0.1.3: manifest-only addition (#473) of the 29
# full-grid phase-diagram genomes #420 shipped in local/nash/
# phase_diagram/ without cataloguing; artifact bytes unchanged. Those
# 29 cells are NOT reproducible by this script (the per-cell solver
# outputs were never committed — only the aggregated summary at
# experiments/nash/phase_diagram/results.json, which carries no genome
# vectors), so :func:`freeze_release` now refuses to clear a bundle
# containing files it cannot regenerate unless ``--force`` is passed.
# NOTE: re-running this freeze script regenerates the 4 preview-derived
# (c=0.50) cells from the preview sources (restoring the
# source_parameters / swept_parameters blocks #420 dropped) — review
# issues #470/#473 before re-freezing.
DEFAULT_RELEASE_VERSION: str = "0.1.3"


# Default subdirectory layout inside the release bundle. Mirrors the
# layout documented in ``local/README.md`` and the schema docstring in
# :mod:`.manifest`.
ARCHETYPES_SUBDIR: str = "archetypes"
NASH_SUBDIR: str = "nash"
PPO_SUBDIR: str = "ppo"


# Canonical (heterogeneous-NE source directory, scenario_id) pairs the
# freeze script knows how to ingest. New scenarios are added by
# appending tuples here — no code change required elsewhere.
#
# The scenario_id is the frozen versioned ID from
# :mod:`bucket_brigade.envs.registry`; if a scenario doesn't have a
# versioned ID yet, leave a TODO comment and skip it.
_HETEROGENEOUS_NE_SOURCES: Tuple[Tuple[str, str], ...] = (
    (
        "experiments/nash/heterogeneous/minimal_specialization",
        "minimal_specialization-v1",
    ),
    ("experiments/nash/heterogeneous/rest_trap", "rest_trap-v1"),
)


# Phase-diagram preview root. Each subdirectory (``alc-*``) is one
# cluster-host's contribution to the partial sweep; each contains
# ``cells/<tag>/results.json`` files (see PR #387 / #391).
_PHASE_DIAGRAM_PREVIEW_ROOT: str = "experiments/nash/phase_diagram/preview"


# Per-cell provenance annotations appended to the manifest ``notes`` of
# specific phase-diagram cells, so re-freezing does not lose them. The
# artifact files themselves stay byte-identical to the solver record;
# only the manifest note carries the annotation (issue #466).
_PHASE_DIAGRAM_CELL_NOTE_SUFFIXES: Dict[str, str] = {
    tag: (
        " Anchor update (#459/#466): this solver-selected profile "
        "(hero|FF|FF|FF, solver team_payoff 72.0095, winner's-curse-"
        "biased) is an epsilon-NE at epsilon=50, but the FF|hero|hero|FF "
        "profile frozen at b0.10_k0.90_c0.50.json (beta-inert, same game) "
        "is also an epsilon-NE and decisively better (CRN paired "
        "+9.55 +/- 2.73/episode; CRN team payoff 55.36 +/- 3.44 vs "
        "45.80 +/- 3.58). Cite b0.10_k0.90_c0.50.json as this cell's NE "
        "anchor; this file is retained unchanged as the historical "
        "solver record. See experiments/nash/phase_diagram/"
        "exploitability/RESULTS.md."
    )
    for tag in ("b0.50_k0.90_c0.50", "b0.90_k0.90_c0.50")
}


class UnknownBundleFilesError(RuntimeError):
    """Raised when a freeze would delete bundle files it cannot regenerate.

    Issue #470/#473 background: PR #420 wrote 29 full-grid phase-diagram
    genome files directly into the shipped ``local/nash/phase_diagram/``
    bundle without committing the per-cell solver outputs they came
    from. The freeze script only knows the preview sources, so a naive
    re-freeze would silently ``rmtree`` those cells. Rather than lose
    unreproducible artifacts, :func:`freeze_release` refuses up front
    and lists the orphaned files; the operator can pass ``force=True``
    (CLI ``--force``) after moving/backing up the files or deciding the
    deletion is intended.
    """

    def __init__(self, bundle_dir: Path, unknown_files: Sequence[str]) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.unknown_files = list(unknown_files)
        listing = "\n".join(f"  - {f}" for f in self.unknown_files)
        super().__init__(
            f"Refusing to freeze into {self.bundle_dir}: it contains "
            f"{len(self.unknown_files)} artifact file(s) this freeze run "
            "cannot regenerate from its known sources and would therefore "
            f"delete:\n{listing}\n"
            "These are likely unreproducible artifacts written directly "
            "into the bundle (see issues #470/#473 — e.g. the #420 "
            "full-grid phase-diagram genomes, whose per-cell solver "
            "outputs were never committed). Either move them out of the "
            "bundle, stage their sources so the freeze can regenerate "
            "them, or re-run with --force to delete them anyway."
        )


@dataclass(frozen=True)
class FreezeStats:
    """Summary of what a freeze run produced.

    Returned by :func:`freeze_release` so callers (CLI / tests) can
    report a one-line summary without re-reading the manifest.
    """

    num_archetypes: int
    num_nash_scenarios: int
    num_nash_phase_diagram_cells: int
    num_ppo_checkpoints: int
    bundle_dir: Path
    manifest_path: Path

    def short_summary(self) -> str:
        return (
            f"Wrote release bundle to {self.bundle_dir}\n"
            f"  archetypes: {self.num_archetypes}\n"
            f"  nash scenarios (heterogeneous): {self.num_nash_scenarios}\n"
            f"  nash cells (phase-diagram preview): {self.num_nash_phase_diagram_cells}\n"
            f"  ppo checkpoints: {self.num_ppo_checkpoints}\n"
            f"  manifest: {self.manifest_path}"
        )


def _git_short_sha(repo_root: Path) -> str:
    """Return the short HEAD SHA, or empty string if git is unavailable.

    Used purely to stamp the manifest with the source commit so a
    downstream consumer can map a release bundle back to the code that
    produced it.
    """
    try:
        out = subprocess.run(  # nosec B603 B607 (git rev-parse — argv is hardcoded, not user input)
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        return ""


def _write_json_atomic(payload: Dict[str, Any], path: Path) -> None:
    """Write ``payload`` as pretty-printed JSON to ``path``.

    Pretty-prints with two-space indent and a trailing newline so the
    file is friendly to ``git diff``. The write is *not* literally
    atomic (no tmpfile + rename) because all callers either delete the
    bundle dir up front or write each filename exactly once per run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def _sha256_of(path: Path) -> str:
    """Return hex SHA-256 of the file at ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_artifact_entry(
    bundle_dir: Path,
    relative_path: str,
    kind: str,
    name: str,
    scenario_id: Optional[str] = None,
    notes: str = "",
) -> ArtifactEntry:
    """Build an :class:`ArtifactEntry` for a file already written.

    Computes ``sha256`` and ``size_bytes`` on the fly so the manifest
    is self-validating for downstream consumers.
    """
    abs_path = bundle_dir / relative_path
    return ArtifactEntry(
        kind=kind,
        name=name,
        filename=relative_path,
        scenario_id=scenario_id,
        sha256=_sha256_of(abs_path),
        size_bytes=abs_path.stat().st_size,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Archetype freezing
# ---------------------------------------------------------------------------

# Names of the 10 parameter slots in archetype / NE genomes. Kept in
# sync with :mod:`bucket_brigade.agents.archetypes` (see the module
# docstring there for the canonical ordering). Duplicated here so the
# JSON output is self-describing; downstream consumers don't need to
# import the agents module just to map indices to names.
_ARCHETYPE_PARAM_NAMES: Tuple[str, ...] = (
    "honesty_bias",
    "work_tendency",
    "neighbor_help_bias",
    "own_house_priority",
    "risk_aversion",
    "coordination_weight",
    "exploration_rate",
    "fatigue_memory",
    "rest_reward_bias",
    "altruism_factor",
)


def _archetype_payload(name: str, vector: Sequence[float]) -> Dict[str, Any]:
    """Build the JSON payload for one archetype.

    Schema (intentionally small and stable)::

        {
            "name": "hero",
            "param_names": ["honesty_bias", ..., "altruism_factor"],
            "genome": [1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.0, 0.9, 0.0, 1.0],
            "source": "bucket_brigade/agents/archetypes.py"
        }
    """
    if len(vector) != len(_ARCHETYPE_PARAM_NAMES):
        raise ValueError(
            f"Archetype {name!r} has {len(vector)} params; expected "
            f"{len(_ARCHETYPE_PARAM_NAMES)} (see _ARCHETYPE_PARAM_NAMES)."
        )
    return {
        "name": name,
        "param_names": list(_ARCHETYPE_PARAM_NAMES),
        "genome": [float(x) for x in vector],
        "source": "bucket_brigade/agents/archetypes.py",
    }


def _freeze_archetypes(bundle_dir: Path) -> List[ArtifactEntry]:
    """Write one JSON file per archetype and return their manifest entries.

    Iterates :data:`bucket_brigade.agents.archetypes.ARCHETYPES` in a
    sorted order for deterministic output.
    """
    entries: List[ArtifactEntry] = []
    for name in sorted(ARCHETYPES.keys()):
        payload = _archetype_payload(name, ARCHETYPES[name])
        rel_path = f"{ARCHETYPES_SUBDIR}/{name}.json"
        _write_json_atomic(payload, bundle_dir / rel_path)
        entries.append(
            _make_artifact_entry(
                bundle_dir=bundle_dir,
                relative_path=rel_path,
                kind="archetype",
                name=name,
                scenario_id=None,
                notes="Hand-coded archetype from bucket_brigade.agents.archetypes",
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Nash equilibrium vector freezing
# ---------------------------------------------------------------------------


def _select_best_converged_ne(
    equilibria: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the converged NE with the highest team payoff, or None.

    The heterogeneous double-oracle sweeps return many candidates per
    scenario; the "best" one (for downstream consumers wanting *one*
    canonical NE per scenario) is the converged equilibrium maximising
    team payoff. If nothing converged we return ``None`` and the
    caller skips that scenario rather than freezing a stale candidate.
    """
    converged = [eq for eq in equilibria if eq.get("converged")]
    if not converged:
        return None
    return max(converged, key=lambda eq: float(eq.get("team_payoff", float("-inf"))))


def _nash_payload(
    scenario_id: str,
    source_results: Dict[str, Any],
    best_ne: Dict[str, Any],
    swept_parameters: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build the JSON payload for one frozen Nash equilibrium.

    Captures just enough metadata that a loader can reconstruct a
    per-position strategy profile (the 10-element genomes) plus the
    measured team payoff and a pointer back to the source run.
    """
    profile = best_ne.get("strategy_profile") or []
    positions = []
    for pos_entry in profile:
        positions.append(
            {
                "position": int(pos_entry.get("position", 0)),
                "closest_archetype": str(pos_entry.get("closest_archetype", "")),
                "genome": [float(x) for x in pos_entry.get("genome", [])],
            }
        )
    payload: Dict[str, Any] = {
        "scenario_id": scenario_id,
        "algorithm": str(
            source_results.get("algorithm", "heterogeneous_double_oracle")
        ),
        "param_names": list(_ARCHETYPE_PARAM_NAMES),
        "team_payoff": float(best_ne.get("team_payoff", 0.0)),
        "per_position_payoffs": [
            float(x) for x in best_ne.get("per_position_payoffs", [])
        ],
        "symmetric_profile": bool(best_ne.get("symmetric_profile", False)),
        "profile_label": str(best_ne.get("profile_label", "")),
        "iterations": int(best_ne.get("iterations", 0)),
        "positions": positions,
        "source_parameters": dict(source_results.get("parameters", {})),
    }
    if swept_parameters is not None:
        payload["swept_parameters"] = {k: float(v) for k, v in swept_parameters.items()}
    return payload


def _freeze_heterogeneous_ne(repo_root: Path, bundle_dir: Path) -> List[ArtifactEntry]:
    """Freeze one NE vector per scenario from heterogeneous-oracle results."""
    entries: List[ArtifactEntry] = []
    for rel_source, scenario_id in _HETEROGENEOUS_NE_SOURCES:
        src = repo_root / rel_source / "results.json"
        if not src.exists():
            # Skip — the source sweep hasn't been run / committed yet.
            # We don't fail the whole freeze because callers may want
            # to ship a partial bundle.
            continue
        with src.open("r", encoding="utf-8") as f:
            source_results = json.load(f)
        best = _select_best_converged_ne(source_results.get("equilibria", []))
        if best is None:
            # No converged NE in the source — skip rather than freeze a
            # non-converged candidate.
            continue
        payload = _nash_payload(scenario_id, source_results, best)
        rel_path = f"{NASH_SUBDIR}/{scenario_id}.json"
        _write_json_atomic(payload, bundle_dir / rel_path)
        entries.append(
            _make_artifact_entry(
                bundle_dir=bundle_dir,
                relative_path=rel_path,
                kind="nash",
                name=scenario_id.rsplit("-v", 1)[0],
                scenario_id=scenario_id,
                notes=f"Best converged NE from {rel_source}/results.json",
            )
        )
    return entries


def _freeze_phase_diagram_ne(repo_root: Path, bundle_dir: Path) -> List[ArtifactEntry]:
    """Freeze per-cell NE vectors from the phase-diagram preview.

    Each cell lives at
    ``experiments/nash/phase_diagram/preview/<host>/cells/<tag>/results.json``
    and corresponds to one (β, κ, c) point on the phase-diagram grid
    (see PR #387 for the canonical layout). The cells share the
    ``minimal_specialization`` substrate, swept along β / κ / c.
    """
    entries: List[ArtifactEntry] = []
    preview_root = repo_root / _PHASE_DIAGRAM_PREVIEW_ROOT
    if not preview_root.exists():
        return entries
    # Dedupe by cell tag — if two hosts ran the same cell we keep the
    # first one (typically alc-2 / alc-4 / alc-6 in lexicographic
    # order) for deterministic output.
    seen_tags: set[str] = set()
    for host_dir in sorted(p for p in preview_root.iterdir() if p.is_dir()):
        cells_dir = host_dir / "cells"
        if not cells_dir.is_dir():
            continue
        for cell_dir in sorted(p for p in cells_dir.iterdir() if p.is_dir()):
            tag = cell_dir.name
            if tag in seen_tags:
                continue
            src = cell_dir / "results.json"
            if not src.exists():
                continue
            with src.open("r", encoding="utf-8") as f:
                source_results = json.load(f)
            best = _select_best_converged_ne(source_results.get("equilibria", []))
            if best is None:
                continue
            swept = source_results.get("swept_parameters", {})
            # Scenario ID for phase-diagram cells inherits the base
            # scenario but the per-cell sweep parameters live in
            # ``swept_parameters`` inside the payload.
            scenario_id = "minimal_specialization-v1"
            payload = _nash_payload(scenario_id, source_results, best, swept)
            rel_path = f"{NASH_SUBDIR}/phase_diagram/{tag}.json"
            _write_json_atomic(payload, bundle_dir / rel_path)
            entries.append(
                _make_artifact_entry(
                    bundle_dir=bundle_dir,
                    relative_path=rel_path,
                    kind="nash",
                    # Use the cell tag as the unique name so multiple
                    # cells coexist in the manifest under
                    # scenario_id=minimal_specialization-v1.
                    name=f"phase_diagram_{tag}",
                    scenario_id=scenario_id,
                    notes=(
                        f"Phase-diagram cell {tag} from {host_dir.name}; "
                        "best converged NE for the (β, κ, c) point."
                        + _PHASE_DIAGRAM_CELL_NOTE_SUFFIXES.get(tag, "")
                    ),
                )
            )
            seen_tags.add(tag)
    return entries


# ---------------------------------------------------------------------------
# PPO checkpoint freezing (scaffolding only until #384 produces files)
# ---------------------------------------------------------------------------


def _freeze_ppo_checkpoints(repo_root: Path, bundle_dir: Path) -> List[ArtifactEntry]:
    """Freeze any pre-staged PPO checkpoints in the source tree.

    Looks at ``bucket_brigade/baselines/release/ppo/`` in the source
    tree (NOT in the wheel, which is the destination). If #384 has
    landed checkpoints there, we copy them into the bundle. If not,
    we return an empty list and the manifest simply has no ``ppo``
    entries — downstream PPO loaders raise a clean :class:`KeyError`
    until the operator populates the directory.

    The two-step flow (#384 stages files here -> this freeze script
    copies + checksums them) means PPO follows the same manifest /
    SHA-256 discipline as archetypes and NE vectors.
    """
    # Source directory operators stage PPO files into. Kept separate
    # from the destination (bundle_dir / PPO_SUBDIR) so re-running the
    # freeze script is idempotent even when the destination IS
    # bundle_dir.
    stage_dir = repo_root / "experiments" / "p3_specialization" / "baselines"
    if not stage_dir.exists():
        return []
    entries: List[ArtifactEntry] = []
    # Glob for `*/best.pt` and `*/checkpoint.pt` to match the layout
    # documented in #384's operator workflow. Skip if nothing matches.
    candidates: List[Path] = []
    for pattern in ("*/checkpoint.pt", "*/best.pt"):
        candidates.extend(sorted(stage_dir.glob(pattern)))
    for src in candidates:
        scenario_name = src.parent.name
        # Map scenario name -> versioned ID. If we can't find a match,
        # skip rather than emit a half-baked entry.
        scenario_id = f"{scenario_name}-v1"
        dest_rel = f"{PPO_SUBDIR}/{scenario_id}.pt"
        dest_abs = bundle_dir / dest_rel
        dest_abs.parent.mkdir(parents=True, exist_ok=True)
        dest_abs.write_bytes(src.read_bytes())
        entries.append(
            _make_artifact_entry(
                bundle_dir=bundle_dir,
                relative_path=dest_rel,
                kind="ppo",
                name=scenario_name,
                scenario_id=scenario_id,
                notes=f"PPO checkpoint staged from {src.relative_to(repo_root)}",
            )
        )
    return entries


# ---------------------------------------------------------------------------
# Top-level freeze entry point
# ---------------------------------------------------------------------------


def _produce_artifacts(
    repo_root: Path,
    bundle_dir: Path,
    include_phase_diagram: bool,
    include_ppo: bool,
) -> Tuple[
    List[ArtifactEntry],
    List[ArtifactEntry],
    List[ArtifactEntry],
    List[ArtifactEntry],
]:
    """Write every artifact file into ``bundle_dir`` and return entries.

    Shared by the real freeze and the pre-flight staging pass in
    :func:`find_unreproducible_files`, so both agree exactly on which
    files a freeze run produces.
    """
    archetype_entries = _freeze_archetypes(bundle_dir)
    nash_entries = _freeze_heterogeneous_ne(repo_root, bundle_dir)
    pd_entries: List[ArtifactEntry] = []
    if include_phase_diagram:
        pd_entries = _freeze_phase_diagram_ne(repo_root, bundle_dir)
    ppo_entries: List[ArtifactEntry] = []
    if include_ppo:
        ppo_entries = _freeze_ppo_checkpoints(repo_root, bundle_dir)
    return archetype_entries, nash_entries, pd_entries, ppo_entries


def _existing_artifact_files(bundle_dir: Path) -> set[str]:
    """Relative POSIX paths of all files under the artifact subdirs."""
    found: set[str] = set()
    for sub in (ARCHETYPES_SUBDIR, NASH_SUBDIR, PPO_SUBDIR):
        target = bundle_dir / sub
        if not target.is_dir():
            continue
        for p in target.rglob("*"):
            if p.is_file():
                found.add(p.relative_to(bundle_dir).as_posix())
    return found


def find_unreproducible_files(
    repo_root: Path,
    bundle_dir: Path,
    include_phase_diagram: bool = True,
    include_ppo: bool = True,
) -> List[str]:
    """Files in ``bundle_dir``'s artifact subdirs a freeze would NOT rewrite.

    Stages a throwaway freeze into a temporary directory to compute the
    exact set of files this run would produce, then returns the sorted
    relative paths of existing bundle files outside that set. A
    non-empty result means a destructive freeze would permanently
    delete artifacts that cannot be regenerated from the known sources
    (see :class:`UnknownBundleFilesError`).
    """
    existing = _existing_artifact_files(Path(bundle_dir))
    if not existing:
        return []
    with tempfile.TemporaryDirectory() as tmp:
        groups = _produce_artifacts(
            repo_root=Path(repo_root),
            bundle_dir=Path(tmp),
            include_phase_diagram=include_phase_diagram,
            include_ppo=include_ppo,
        )
    producible = {entry.filename for group in groups for entry in group}
    return sorted(existing - producible)


def freeze_release(
    repo_root: Path,
    bundle_dir: Path,
    release_version: str = DEFAULT_RELEASE_VERSION,
    release_date: Optional[str] = None,
    include_phase_diagram: bool = True,
    include_ppo: bool = True,
    force: bool = False,
) -> FreezeStats:
    """Gather every artifact and write the manifest into ``bundle_dir``.

    Args:
        repo_root: Path to the bucket-brigade repo root. Used to
            resolve experiment result directories.
        bundle_dir: Destination for the release bundle. Created if it
            doesn't exist. Existing ``manifest.json`` and any
            ``archetypes/``, ``nash/``, ``ppo/`` subdirectories are
            **overwritten** (older entries removed) so re-runs are
            deterministic.
        release_version: Version string stamped into the manifest.
        release_date: ISO-8601 date stamped into the manifest. Defaults
            to today (UTC).
        include_phase_diagram: If True, freeze per-cell NE vectors from
            the phase-diagram preview. Most callers want this.
        include_ppo: If True, copy any staged PPO checkpoints into the
            bundle. No-op if none are staged.
        force: If True, skip the pre-flight unreproducible-files check
            and clear the artifact subdirs unconditionally (the pre-#473
            behaviour).

    Returns:
        :class:`FreezeStats` summarising what was written.

    Raises:
        UnknownBundleFilesError: If ``bundle_dir`` contains artifact
            files this run cannot regenerate and ``force`` is False.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Pre-flight (#473): refuse to delete artifact files this run
    # cannot regenerate. PR #420 demonstrated the failure mode — genome
    # files written straight into the shipped bundle with no committed
    # source, which a naive re-freeze would silently rmtree.
    if not force:
        unknown = find_unreproducible_files(
            repo_root=repo_root,
            bundle_dir=bundle_dir,
            include_phase_diagram=include_phase_diagram,
            include_ppo=include_ppo,
        )
        if unknown:
            raise UnknownBundleFilesError(bundle_dir, unknown)

    # Clear out any prior artifact subdirs so stale entries don't
    # linger across runs. The .gitkeep + README that ship in the
    # wheel-shipped local/ dir live at the bundle root, not in the
    # subdirs, so this is safe.
    for sub in (ARCHETYPES_SUBDIR, NASH_SUBDIR, PPO_SUBDIR):
        target = bundle_dir / sub
        if target.exists():
            _rmtree(target)

    archetype_entries, nash_entries, pd_entries, ppo_entries = _produce_artifacts(
        repo_root=repo_root,
        bundle_dir=bundle_dir,
        include_phase_diagram=include_phase_diagram,
        include_ppo=include_ppo,
    )

    if release_date is None:
        release_date = datetime.date.today().isoformat()

    manifest = Manifest(
        release_version=release_version,
        release_date=release_date,
        source_commit=_git_short_sha(repo_root),
        artifacts=[*archetype_entries, *nash_entries, *pd_entries, *ppo_entries],
        extra={
            # Document the producer so a manifest read in isolation
            # points back at the script that wrote it.
            "frozen_by": "bucket_brigade.baselines.release.freeze",
        },
    )
    manifest_path = bundle_dir / MANIFEST_FILENAME
    save_manifest(manifest, manifest_path)

    return FreezeStats(
        num_archetypes=len(archetype_entries),
        num_nash_scenarios=len(nash_entries),
        num_nash_phase_diagram_cells=len(pd_entries),
        num_ppo_checkpoints=len(ppo_entries),
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
    )


def _rmtree(path: Path) -> None:
    """Recursively remove a directory. Kept here to avoid importing
    :mod:`shutil` at module load time and to keep the freeze module
    self-contained for ease of reading."""
    import shutil

    shutil.rmtree(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_repo_root() -> Path:
    """Best-effort guess at the repo root when invoked from anywhere.

    Walks up from this file's location to find the top-level
    ``bucket_brigade`` directory. Falls back to CWD if nothing is
    found (the caller can pass ``--repo-root`` explicitly).
    """
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "bucket_brigade").is_dir() and (
            parent / "pyproject.toml"
        ).exists():
            return parent
    return Path.cwd()


def _default_bundle_dir(repo_root: Path) -> Path:
    """The wheel-shipped ``local/`` dir under :data:`repo_root`."""
    return repo_root / "bucket_brigade" / "baselines" / "release" / "local"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bucket_brigade.baselines.release.freeze",
        description=(
            "Build a frozen-baseline release bundle from in-repo sources "
            "(archetypes, Nash equilibrium vectors, PPO checkpoints if staged)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Path to the bucket-brigade repo root (default: auto-detect).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Destination directory for the release bundle (default: "
            "bucket_brigade/baselines/release/local/ under --repo-root)."
        ),
    )
    parser.add_argument(
        "--release-version",
        default=DEFAULT_RELEASE_VERSION,
        help="Release version string stamped into the manifest.",
    )
    parser.add_argument(
        "--release-date",
        default=None,
        help="ISO-8601 release date (default: today UTC).",
    )
    parser.add_argument(
        "--no-phase-diagram",
        action="store_true",
        help="Skip per-cell NE vectors from the phase-diagram preview.",
    )
    parser.add_argument(
        "--no-ppo",
        action="store_true",
        help="Skip staged PPO checkpoints (use when PPO source dir is stale).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the manifest that *would* be written and exit without touching disk."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Clear the destination artifact subdirs even if they contain "
            "files this freeze cannot regenerate (skips the #473 "
            "unreproducible-files guard). DESTRUCTIVE: review the guard's "
            "file list (run without --force first) before using."
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    repo_root = (args.repo_root or _default_repo_root()).resolve()
    bundle_dir = (args.output_dir or _default_bundle_dir(repo_root)).resolve()

    if args.dry_run:
        # In dry-run mode we still construct the bundle in a temp dir
        # so we exercise the same code path, then report and exit.
        with tempfile.TemporaryDirectory() as tmp:
            stats = freeze_release(
                repo_root=repo_root,
                bundle_dir=Path(tmp),
                release_version=args.release_version,
                release_date=args.release_date,
                include_phase_diagram=not args.no_phase_diagram,
                include_ppo=not args.no_ppo,
            )
            print("(dry-run)")
            print(stats.short_summary())
            print(f"  intended destination: {bundle_dir}")
        # Also run the #473 pre-flight against the *intended*
        # destination so operators see what a real (non---force) run
        # would refuse to delete.
        unknown = find_unreproducible_files(
            repo_root=repo_root,
            bundle_dir=bundle_dir,
            include_phase_diagram=not args.no_phase_diagram,
            include_ppo=not args.no_ppo,
        )
        if unknown:
            print(
                f"  WARNING: {len(unknown)} file(s) in the destination "
                "cannot be regenerated by this freeze; a real run will "
                "refuse without --force:"
            )
            for f in unknown:
                print(f"    - {f}")
        return 0

    try:
        stats = freeze_release(
            repo_root=repo_root,
            bundle_dir=bundle_dir,
            release_version=args.release_version,
            release_date=args.release_date,
            include_phase_diagram=not args.no_phase_diagram,
            include_ppo=not args.no_ppo,
            force=args.force,
        )
    except UnknownBundleFilesError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(stats.short_summary())
    return 0


if __name__ == "__main__":  # pragma: no cover — module CLI entry point
    sys.exit(main())

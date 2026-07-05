"""Tests for the frozen-baseline freeze script (issue #371).

Covers :mod:`bucket_brigade.baselines.release.freeze`:

- Archetype JSON files match the in-memory archetype vectors.
- Heterogeneous-oracle NE freezing pulls the best converged
  equilibrium and round-trips through the loader API.
- Manifest is well-formed and self-consistent (every entry's sha256 +
  size match the file on disk).
- The script is idempotent — running it twice into the same directory
  produces byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from bucket_brigade.agents.archetypes import ARCHETYPES
from bucket_brigade.baselines.release import (
    MANIFEST_FILENAME,
    load_manifest,
)
from bucket_brigade.baselines.release.freeze import (
    DEFAULT_RELEASE_VERSION,
    _ARCHETYPE_PARAM_NAMES,
    _select_best_converged_ne,
    UnknownBundleFilesError,
    find_unreproducible_files,
    freeze_release,
)


def _fake_heterogeneous_results(
    scenario: str, num_positions: int = 4, with_converged: bool = True
) -> Dict[str, Any]:
    """Build a minimal results.json matching the heterogeneous-oracle shape."""
    genome = [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8]
    strategy_profile = [
        {
            "position": i,
            "closest_archetype": "firefighter(d=0.00)",
            "parameters": dict(zip(_ARCHETYPE_PARAM_NAMES, genome)),
            "genome": genome,
        }
        for i in range(num_positions)
    ]
    equilibria = []
    if with_converged:
        equilibria.append(
            {
                "converged": True,
                "iterations": 17,
                "team_payoff": -700.0,
                "per_position_payoffs": [-700.0] * num_positions,
                "symmetric_profile": True,
                "profile_label": " | ".join(["firefighter"] * num_positions),
                "strategy_profile": strategy_profile,
            }
        )
    # Add a non-converged candidate with a *higher* payoff to verify
    # the selector picks the best *converged* one, not just best overall.
    equilibria.append(
        {
            "converged": False,
            "iterations": 25,
            "team_payoff": -500.0,
            "per_position_payoffs": [-500.0] * num_positions,
            "symmetric_profile": False,
            "profile_label": " | ".join(["hero"] * num_positions),
            "strategy_profile": strategy_profile,
        }
    )
    return {
        "scenario": scenario,
        "algorithm": "heterogeneous_double_oracle",
        "parameters": {
            "num_simulations": 1000,
            "max_iterations": 25,
            "epsilon": 50.0,
            "num_restarts": 20,
            "seed": 42,
        },
        "equilibria": equilibria,
    }


def _stage_fake_repo(repo_root: Path) -> None:
    """Populate ``repo_root`` with the minimal source layout the freeze
    script expects (one heterogeneous NE scenario + one phase-diagram cell).
    Real-repo paths are reproduced exactly so tests exercise the same
    glob patterns as production."""
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("# fake\n")
    (repo_root / "bucket_brigade").mkdir(parents=True, exist_ok=True)

    # Heterogeneous-NE source for minimal_specialization.
    hetero_dir = (
        repo_root / "experiments" / "nash" / "heterogeneous" / "minimal_specialization"
    )
    hetero_dir.mkdir(parents=True, exist_ok=True)
    (hetero_dir / "results.json").write_text(
        json.dumps(_fake_heterogeneous_results("minimal_specialization"))
    )

    # One phase-diagram preview cell on the canonical layout
    # preview/<host>/cells/<tag>/results.json.
    cell_dir = (
        repo_root
        / "experiments"
        / "nash"
        / "phase_diagram"
        / "preview"
        / "alc-fake"
        / "cells"
        / "b0.50_k0.50_c0.50"
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    pd_results = _fake_heterogeneous_results("minimal_specialization@cell")
    pd_results["swept_parameters"] = {"beta": 0.5, "kappa": 0.5, "c": 0.5}
    (cell_dir / "results.json").write_text(json.dumps(pd_results))


def test_freeze_writes_archetype_json_matching_in_memory(tmp_path: Path) -> None:
    """Each archetype gets a JSON file whose ``genome`` matches the
    in-memory ``ARCHETYPES`` numpy vector elementwise."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)
    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")

    for name, expected in ARCHETYPES.items():
        path = bundle / "archetypes" / f"{name}.json"
        assert path.exists(), f"missing archetype {name}"
        payload = json.loads(path.read_text())
        assert payload["name"] == name
        assert payload["param_names"] == list(_ARCHETYPE_PARAM_NAMES)
        assert len(payload["genome"]) == len(expected)
        for got, exp in zip(payload["genome"], expected):
            assert got == pytest.approx(float(exp))


def test_freeze_writes_well_formed_manifest_with_checksums(tmp_path: Path) -> None:
    """Manifest is loadable AND every entry's sha256 + size_bytes match
    the file on disk."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)
    stats = freeze_release(
        repo_root=repo,
        bundle_dir=bundle,
        release_date="2026-06-08",
        release_version="9.9.9",
    )

    manifest = load_manifest(bundle / MANIFEST_FILENAME)
    assert manifest.release_version == "9.9.9"
    assert manifest.release_date == "2026-06-08"
    assert manifest.extra.get("frozen_by") == "bucket_brigade.baselines.release.freeze"

    # Counts match the stats helper.
    assert stats.num_archetypes == len(ARCHETYPES)
    assert stats.num_nash_scenarios == 1  # only minimal_specialization staged
    assert stats.num_nash_phase_diagram_cells == 1
    assert stats.num_ppo_checkpoints == 0  # no PPO staged in fake repo

    # Per-entry checksum + size verification.
    for entry in manifest.artifacts:
        path = bundle / entry.filename
        assert path.exists(), f"manifest references missing file {entry.filename}"
        data = path.read_bytes()
        assert entry.size_bytes == len(data)
        assert entry.sha256 == hashlib.sha256(data).hexdigest()


def test_freeze_selects_best_converged_ne_only() -> None:
    """The selector ignores non-converged equilibria even if they have
    a higher team payoff. This is the contract that prevents shipping
    a stale candidate as the canonical NE."""
    equilibria: List[Dict[str, Any]] = [
        {"converged": False, "team_payoff": 100.0, "iterations": 25},
        {"converged": True, "team_payoff": -500.0, "iterations": 17},
        {"converged": True, "team_payoff": -300.0, "iterations": 20},
    ]
    best = _select_best_converged_ne(equilibria)
    assert best is not None
    assert best["team_payoff"] == -300.0


def test_freeze_returns_none_when_no_converged_ne() -> None:
    """If nothing converged the selector returns ``None`` so the freeze
    script can skip the scenario rather than emit a bad entry."""
    equilibria: List[Dict[str, Any]] = [
        {"converged": False, "team_payoff": 100.0},
        {"converged": False, "team_payoff": -500.0},
    ]
    assert _select_best_converged_ne(equilibria) is None


def test_freeze_is_idempotent(tmp_path: Path) -> None:
    """Running the freeze script twice into the same directory produces
    byte-identical output for every artifact. This guarantees that a
    re-freeze with no upstream changes shows up as an empty git diff."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)

    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    first_snapshot = _snapshot_bundle(bundle)

    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    second_snapshot = _snapshot_bundle(bundle)

    assert first_snapshot == second_snapshot


def test_freeze_handles_missing_nash_sources_gracefully(tmp_path: Path) -> None:
    """Bundle freezes successfully with only archetypes when no NE
    source files are present. Operators on a fresh checkout (no Nash
    runs yet) should still be able to ship archetype-only bundles."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text("# fake\n")
    (repo / "bucket_brigade").mkdir(parents=True, exist_ok=True)

    stats = freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    assert stats.num_archetypes == len(ARCHETYPES)
    assert stats.num_nash_scenarios == 0
    assert stats.num_nash_phase_diagram_cells == 0


def test_freeze_refuses_to_delete_unreproducible_files(tmp_path: Path) -> None:
    """A freeze into a bundle containing artifact files it cannot
    regenerate refuses loudly instead of rmtree-ing them (issue #473 —
    the #420 full-grid genomes were exactly such files). Nothing in the
    destination is modified by the refused run."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)
    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    snapshot_before = _snapshot_bundle(bundle)

    # Inject a file the freeze sources cannot reproduce (analogous to
    # a #420-style direct write into the shipped bundle).
    orphan = bundle / "nash" / "phase_diagram" / "b9.99_k9.99_c9.99.json"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("{}")

    with pytest.raises(UnknownBundleFilesError) as excinfo:
        freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    assert "b9.99_k9.99_c9.99.json" in str(excinfo.value)
    assert excinfo.value.unknown_files == ["nash/phase_diagram/b9.99_k9.99_c9.99.json"]

    # The refused run must not have touched the bundle.
    assert orphan.exists()
    snapshot_after = _snapshot_bundle(bundle)
    snapshot_after.pop("nash/phase_diagram/b9.99_k9.99_c9.99.json")
    assert snapshot_after == snapshot_before


def test_freeze_force_clears_stale_artifact_subdirs(tmp_path: Path) -> None:
    """With ``force=True`` the freeze keeps the pre-#473 behaviour:
    prior subdir contents are removed so an artifact renamed or removed
    upstream doesn't linger as a stale file in ``local/``."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)
    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")

    # Inject a fake stale file under archetypes/. A forced freeze must
    # remove it.
    stale = bundle / "archetypes" / "ghost.json"
    stale.write_text("{}")
    assert stale.exists()

    freeze_release(
        repo_root=repo, bundle_dir=bundle, release_date="2026-06-08", force=True
    )
    assert not stale.exists()


def test_find_unreproducible_files_reports_only_orphans(tmp_path: Path) -> None:
    """The pre-flight helper returns exactly the files a freeze cannot
    regenerate: empty for a bundle the sources fully cover, and the
    orphan's relative path once one is injected."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)
    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")

    assert find_unreproducible_files(repo_root=repo, bundle_dir=bundle) == []

    orphan = bundle / "nash" / "orphan.json"
    orphan.write_text("{}")
    assert find_unreproducible_files(repo_root=repo, bundle_dir=bundle) == [
        "nash/orphan.json"
    ]


def test_freeze_uses_default_release_version_when_unspecified(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_fake_repo(repo)
    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    manifest = load_manifest(bundle / MANIFEST_FILENAME)
    assert manifest.release_version == DEFAULT_RELEASE_VERSION


def _snapshot_bundle(bundle: Path) -> Dict[str, bytes]:
    """Return a {relative_path: bytes} snapshot for byte-comparison."""
    out: Dict[str, bytes] = {}
    for p in sorted(bundle.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(bundle))] = p.read_bytes()
    return out

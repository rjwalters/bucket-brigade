"""Tests for the frozen-baseline loader API (issue #371).

Covers :mod:`bucket_brigade.baselines.release.loaders`:

- ``load_archetype`` returns a numpy vector matching the in-memory
  archetype.
- ``load_nash`` returns a structured dict; ``load_nash_genomes``
  flattens it to a (num_positions, 10) array.
- ``load_ppo`` raises ``KeyError`` when no PPO entry exists (the
  expected state pre-#384).
- ``list_*`` helpers return empty lists on a missing manifest rather
  than raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from bucket_brigade.agents.archetypes import ARCHETYPES, HERO_PARAMS
from bucket_brigade.baselines.release import (
    list_archetypes,
    list_nash_scenarios,
    list_ppo_scenarios,
    load_archetype,
    load_nash,
    load_nash_genomes,
    load_ppo,
)
from bucket_brigade.baselines.release.freeze import (
    _ARCHETYPE_PARAM_NAMES,
    freeze_release,
)


def _stage_minimal_repo(repo_root: Path) -> None:
    """Stage a fake repo with one heterogeneous NE source for
    minimal_specialization. Mirrors the fixture in test_release_freeze
    but kept local to avoid cross-test coupling."""
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("# fake\n")
    (repo_root / "bucket_brigade").mkdir(parents=True, exist_ok=True)
    genome = [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8]
    strategy_profile = [
        {
            "position": i,
            "closest_archetype": "firefighter(d=0.00)",
            "parameters": dict(zip(_ARCHETYPE_PARAM_NAMES, genome)),
            "genome": genome,
        }
        for i in range(4)
    ]
    payload: Dict[str, Any] = {
        "scenario": "minimal_specialization",
        "algorithm": "heterogeneous_double_oracle",
        "parameters": {"num_simulations": 1000, "seed": 42},
        "equilibria": [
            {
                "converged": True,
                "iterations": 17,
                "team_payoff": -700.0,
                "per_position_payoffs": [-700.0] * 4,
                "symmetric_profile": True,
                "profile_label": "firefighter | firefighter | firefighter | firefighter",
                "strategy_profile": strategy_profile,
            }
        ],
    }
    hetero_dir = (
        repo_root / "experiments" / "nash" / "heterogeneous" / "minimal_specialization"
    )
    hetero_dir.mkdir(parents=True, exist_ok=True)
    (hetero_dir / "results.json").write_text(json.dumps(payload))


@pytest.fixture()
def frozen_bundle(tmp_path: Path) -> Path:
    """Build a freshly frozen bundle in tmp_path for the loader tests."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_minimal_repo(repo)
    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")
    return bundle


def test_load_archetype_returns_numpy_vector_matching_source(
    frozen_bundle: Path,
) -> None:
    arr = load_archetype("hero", directory=frozen_bundle)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10,)
    np.testing.assert_allclose(arr, HERO_PARAMS)


def test_load_archetype_supports_every_archetype(frozen_bundle: Path) -> None:
    for name in ARCHETYPES:
        arr = load_archetype(name, directory=frozen_bundle)
        np.testing.assert_allclose(arr, ARCHETYPES[name])


def test_load_archetype_raises_keyerror_for_unknown_name(
    frozen_bundle: Path,
) -> None:
    with pytest.raises(KeyError, match="archetype"):
        load_archetype("nonexistent_archetype", directory=frozen_bundle)


def test_load_nash_returns_dict_with_expected_keys(frozen_bundle: Path) -> None:
    payload = load_nash("minimal_specialization-v1", directory=frozen_bundle)
    assert payload["scenario_id"] == "minimal_specialization-v1"
    assert payload["param_names"] == list(_ARCHETYPE_PARAM_NAMES)
    assert payload["team_payoff"] == pytest.approx(-700.0)
    assert payload["symmetric_profile"] is True
    assert len(payload["positions"]) == 4
    assert "genome" in payload["positions"][0]
    assert len(payload["positions"][0]["genome"]) == 10


def test_load_nash_genomes_returns_2d_array(frozen_bundle: Path) -> None:
    g = load_nash_genomes("minimal_specialization-v1", directory=frozen_bundle)
    assert g.shape == (4, 10)
    assert g.dtype == np.float64
    # All four positions are identical in the staged fixture.
    for i in range(1, 4):
        np.testing.assert_allclose(g[i], g[0])


def test_load_nash_raises_keyerror_for_unknown_scenario(frozen_bundle: Path) -> None:
    with pytest.raises(KeyError):
        load_nash("nonexistent-v1", directory=frozen_bundle)


def test_load_ppo_raises_keyerror_when_no_ppo_entry(frozen_bundle: Path) -> None:
    """Pre-#384, no PPO checkpoints are staged so PPO loads should
    raise ``KeyError`` — distinct from FileNotFoundError (which would
    mean the manifest itself is missing)."""
    with pytest.raises(KeyError):
        load_ppo("minimal_specialization-v1", directory=frozen_bundle)


def test_list_helpers_return_expected_contents(frozen_bundle: Path) -> None:
    archetypes = list_archetypes(directory=frozen_bundle)
    assert set(archetypes) == set(ARCHETYPES.keys())

    nash_scenarios = list_nash_scenarios(directory=frozen_bundle)
    assert "minimal_specialization-v1" in nash_scenarios

    ppo_scenarios = list_ppo_scenarios(directory=frozen_bundle)
    assert ppo_scenarios == []  # nothing staged in the test fixture


def test_list_helpers_return_empty_lists_on_missing_manifest(
    tmp_path: Path,
) -> None:
    """When the manifest is missing entirely (fresh wheel without an
    artifact bundle), the list helpers degrade to empty lists rather
    than raising — keeps downstream code that wants to enumerate
    available baselines simple."""
    empty_dir = tmp_path / "no_manifest_here"
    empty_dir.mkdir()
    assert list_archetypes(directory=empty_dir) == []
    assert list_nash_scenarios(directory=empty_dir) == []
    assert list_ppo_scenarios(directory=empty_dir) == []


def test_load_nash_with_explicit_name_finds_phase_diagram_cell(
    tmp_path: Path,
) -> None:
    """When multiple NE entries share a scenario_id (the phase-diagram
    cell case), the caller can disambiguate by passing the manifest
    ``name`` directly."""
    repo = tmp_path / "repo"
    bundle = tmp_path / "bundle"
    _stage_minimal_repo(repo)
    # Stage one phase-diagram cell.
    cell_dir = (
        repo
        / "experiments"
        / "nash"
        / "phase_diagram"
        / "preview"
        / "alc-fake"
        / "cells"
        / "b0.50_k0.50_c0.50"
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    genome = [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8]
    strategy_profile = [
        {
            "position": i,
            "closest_archetype": "firefighter(d=0.00)",
            "parameters": dict(zip(_ARCHETYPE_PARAM_NAMES, genome)),
            "genome": genome,
        }
        for i in range(4)
    ]
    (cell_dir / "results.json").write_text(
        json.dumps(
            {
                "scenario": "minimal_specialization@cell",
                "algorithm": "heterogeneous_double_oracle",
                "parameters": {"num_simulations": 1000, "seed": 1},
                "swept_parameters": {"beta": 0.5, "kappa": 0.5, "c": 0.5},
                "equilibria": [
                    {
                        "converged": True,
                        "iterations": 17,
                        "team_payoff": -600.0,
                        "per_position_payoffs": [-600.0] * 4,
                        "symmetric_profile": True,
                        "profile_label": "firefighter | firefighter | firefighter | firefighter",
                        "strategy_profile": strategy_profile,
                    }
                ],
            }
        )
    )

    freeze_release(repo_root=repo, bundle_dir=bundle, release_date="2026-06-08")

    payload = load_nash(
        "minimal_specialization-v1",
        name="phase_diagram_b0.50_k0.50_c0.50",
        directory=bundle,
    )
    assert payload["scenario_id"] == "minimal_specialization-v1"
    assert payload.get("swept_parameters") == {
        "beta": 0.5,
        "kappa": 0.5,
        "c": 0.5,
    }

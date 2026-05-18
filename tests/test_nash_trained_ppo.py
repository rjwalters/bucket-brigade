"""Tests for the trained-PPO Nash pipeline (issue #275).

Covers:
- :class:`TrainedPolicyArchetype` checkpoint round-trip and deterministic
  action reproducibility.
- Above-random filter logic.
- ``compute_nash_trained.py`` pool expansion + mocked-Nash smoke (so the
  test stays fast on CI / a laptop).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

# Skip everything if torch isn't installed (the helper module imports it
# unconditionally for the policy wrapper).
torch = pytest.importorskip("torch")

from bucket_brigade.agents import TrainedPolicyArchetype  # noqa: E402
from bucket_brigade.agents.archetypes import ARCHETYPES  # noqa: E402
from bucket_brigade.training.joint_trainer import flatten_dict_obs  # noqa: E402
from bucket_brigade.training.networks import PolicyNetwork  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "experiments" / "scripts" / "compute_nash_trained.py"


def _load_script_module():
    """Load ``compute_nash_trained.py`` as a standalone module for testing."""
    spec = importlib.util.spec_from_file_location("compute_nash_trained", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["compute_nash_trained"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module():
    return _load_script_module()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dummy_obs(num_agents: int = 4, num_houses: int = 10) -> dict:
    """Build a dict-obs matching :func:`flatten_dict_obs`'s shape contract."""
    return {
        "houses": np.zeros(num_houses, dtype=np.int8),
        "signals": np.zeros(num_agents, dtype=np.int8),
        "locations": np.zeros(num_agents, dtype=np.int8),
        "last_actions": np.zeros((num_agents, 2), dtype=np.int8),
        "scenario_info": np.zeros(10, dtype=np.float32),
    }


def _save_random_policy(
    tmp_path: Path,
    obs_dim: int,
    action_dims: list,
    hidden_size: int = 16,
    seed: int = 0,
) -> Path:
    """Save a freshly-initialized PolicyNetwork to a state_dict on disk."""
    torch.manual_seed(seed)
    policy = PolicyNetwork(
        obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size
    )
    path = tmp_path / f"agent_{seed}.pt"
    torch.save(policy.state_dict(), path)
    return path


# ---------------------------------------------------------------------------
# TrainedPolicyArchetype: round-trip
# ---------------------------------------------------------------------------


def test_trained_policy_archetype_roundtrip_matches_source(tmp_path):
    """Loading a saved policy and acting on the same obs should match the
    deterministic forward pass of the source policy."""
    num_agents = 4
    obs = _make_dummy_obs(num_agents=num_agents)
    flat = flatten_dict_obs(obs, agent_id=2, num_agents=num_agents)
    obs_dim = flat.shape[0]
    action_dims = [10, 2, 2]

    # Save a freshly-initialized policy.
    torch.manual_seed(123)
    source = PolicyNetwork(obs_dim=obs_dim, action_dims=action_dims, hidden_size=16)
    path = tmp_path / "agent_2.pt"
    torch.save(source.state_dict(), path)

    # Load via the wrapper.
    archetype = TrainedPolicyArchetype(
        state_dict_path=path,
        agent_id=2,
        num_agents=num_agents,
        deterministic=True,
    )

    # Compute the deterministic action from the source.
    source.eval()
    with torch.no_grad():
        x = torch.from_numpy(flat).unsqueeze(0)
        actions, _, _ = source.get_action(x, deterministic=True)
    expected = np.array([int(a[0].item()) for a in actions], dtype=np.int8)

    actual = archetype.act(obs)
    assert actual.shape == (3,)
    assert np.array_equal(actual, expected), (
        f"wrapper action {actual} != source action {expected}"
    )


def test_trained_policy_archetype_infers_architecture_from_state_dict(tmp_path):
    """Loading without explicit ``obs_dim`` / ``hidden_size`` should still work."""
    num_agents = 4
    obs = _make_dummy_obs(num_agents=num_agents)
    obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=num_agents).shape[0]

    path = _save_random_policy(
        tmp_path, obs_dim=obs_dim, action_dims=[10, 2, 2], hidden_size=32
    )
    archetype = TrainedPolicyArchetype(
        state_dict_path=path, agent_id=0, num_agents=num_agents
    )
    assert archetype.obs_dim == obs_dim
    assert archetype.hidden_size == 32
    assert archetype.action_dims == [10, 2, 2]


def test_trained_policy_archetype_obs_dim_mismatch_raises(tmp_path):
    """A checkpoint trained on a different ``num_agents`` should raise."""
    obs_dim = 50  # arbitrary mismatch
    path = _save_random_policy(tmp_path, obs_dim=obs_dim, action_dims=[10, 2, 2])
    archetype = TrainedPolicyArchetype(state_dict_path=path, agent_id=0, num_agents=4)
    obs = _make_dummy_obs(num_agents=4)
    with pytest.raises(ValueError, match="obs dim"):
        archetype.act(obs)


def test_trained_policy_archetype_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TrainedPolicyArchetype(
            state_dict_path=tmp_path / "does_not_exist.pt",
            agent_id=0,
            num_agents=4,
        )


# ---------------------------------------------------------------------------
# Above-random filter logic
# ---------------------------------------------------------------------------


def test_above_random_threshold_basic(script_module):
    """Filter must return True iff reward strictly beats random + margin."""
    above = script_module.above_random_threshold
    assert above(300.0, random_baseline=251.23, margin=0.0) is True
    assert above(251.23, random_baseline=251.23, margin=0.0) is False
    assert above(251.24, random_baseline=251.23, margin=0.0) is True
    # Margin shifts threshold up.
    assert above(260.0, random_baseline=251.23, margin=20.0) is False
    assert above(300.0, random_baseline=251.23, margin=20.0) is True


# ---------------------------------------------------------------------------
# Pool expansion: solver must not crash with fake archetypes
# ---------------------------------------------------------------------------


def test_solver_accepts_augmented_pool():
    """Adding 2 trained-stub strategies to the heuristic pool must produce a
    valid Nash distribution (sums to 1, K-dim, non-negative)."""
    from bucket_brigade.equilibrium.nash_solver import solve_symmetric_nash

    K = len(ARCHETYPES) + 2  # 5 heuristics + 2 trained stubs
    # Build a synthetic but plausible payoff matrix:
    # - everyone gets baseline 100 against themselves
    # - random off-diagonal in [-30, +30]
    rng = np.random.RandomState(7)
    matrix = 100.0 + rng.uniform(-30.0, 30.0, size=(K, K))
    distribution = solve_symmetric_nash(matrix)
    assert distribution.shape == (K,)
    assert np.all(distribution >= -1e-9)
    np.testing.assert_allclose(distribution.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Script: dry-run + mocked payoff matrix
# ---------------------------------------------------------------------------


def test_compute_nash_trained_dry_run_heuristics_only(tmp_path, script_module, capsys):
    """``--dry-run`` without ``--include-trained`` should produce a 5-strategy
    partial verdict and exit 0."""
    out = tmp_path / "nash_out"
    rc = script_module.main(
        [
            "--scenario",
            "default",
            "--simulations",
            "1",
            "--output-dir",
            str(out),
            "--dry-run",
            "--quiet",
        ]
    )
    assert rc == 0
    partial_path = out / "default" / "equilibrium_partial.json"
    assert partial_path.exists(), capsys.readouterr().out
    import json

    partial = json.loads(partial_path.read_text())
    assert partial["pool_size"] == len(ARCHETYPES)
    assert partial["dry_run"] is True


def test_compute_nash_trained_smoke_mocked_payoff(tmp_path, script_module, capsys):
    """End-to-end smoke: patch the payoff-evaluator to return a precomputed
    matrix, then verify the solver runs and writes equilibrium.json."""
    out = tmp_path / "nash_out"
    K = len(ARCHETYPES)
    # Diagonal-dominant matrix so the solver puts mass on a single strategy.
    fake_matrix = np.eye(K) * 100.0 - 50.0 * (1.0 - np.eye(K))
    with mock.patch.object(
        script_module, "evaluate_mixed_payoff_matrix", return_value=fake_matrix
    ):
        rc = script_module.main(
            [
                "--scenario",
                "default",
                "--simulations",
                "1",
                "--output-dir",
                str(out),
                "--quiet",
            ]
        )
    assert rc == 0, capsys.readouterr().out
    eq_path = out / "default" / "equilibrium.json"
    assert eq_path.exists()
    import json

    eq = json.loads(eq_path.read_text())
    assert eq["pool_size"] == K
    # No trained policies in pool by default -> verdict reflects that.
    assert eq["trained_mass"] == 0.0
    assert "no-trained-policies-in-pool" in eq["verdict"]
    assert len(eq["distribution"]) == K
    np.testing.assert_allclose(sum(eq["distribution"]), 1.0, atol=1e-6)


def test_compute_nash_trained_unknown_scenario_exits(tmp_path, script_module):
    rc = script_module.main(
        [
            "--scenario",
            "not_a_real_scenario",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2


def test_discover_checkpoint_dirs_returns_empty_when_no_match(tmp_path, script_module):
    """Globbing a non-existent path must yield an empty list, not crash."""
    pattern = str(tmp_path / "nonexistent" / "{scenario}" / "policies")
    matches = script_module.discover_checkpoint_dirs(pattern, scenario="default")
    assert matches == []

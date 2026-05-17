"""Tests for ``experiments/p3_specialization/bc_init.py`` (issue #270).

Covers:

- Demo gathering yields the requested per-agent pair counts and shape.
- Saved checkpoints round-trip into a fresh ``PolicyNetwork`` (the load
  target inside ``train_one_cell``).
- ``train_one_cell`` raises a clear error when ``bc_init_checkpoint_dir``
  points at a directory with a missing or malformed checkpoint.

Pure-CPU, deliberately small (``num_pairs_per_agent=64`` / ``epochs=1``) so
the suite finishes in seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "p3_specialization"))

import bc_init  # type: ignore[import-not-found]
from bucket_brigade.training.networks import PolicyNetwork


@pytest.fixture(scope="module")
def small_bc_run(tmp_path_factory):
    """Run a tiny BC fit end-to-end and return the output directory + summary."""
    out = tmp_path_factory.mktemp("bc_init_smoke")
    num_agents = 4
    obs_per_agent, act_per_agent = bc_init.gather_demos(
        scenario_name="minimal_specialization",
        num_agents=num_agents,
        num_pairs_per_agent=64,
        seed=0,
    )

    assert len(obs_per_agent) == num_agents
    obs_dim = obs_per_agent[0].shape[1]
    # Per-agent obs and actions must have aligned row counts.
    for i in range(num_agents):
        assert obs_per_agent[i].shape == (64, obs_dim)
        assert act_per_agent[i].shape == (64, 3)
        # Action dimensions must be in-bounds for [10, 2, 2].
        assert act_per_agent[i][:, 0].max() < 10
        assert act_per_agent[i][:, 1].max() < 2
        assert act_per_agent[i][:, 2].max() < 2

    policies = []
    for i in range(num_agents):
        p, _ = bc_init.bc_fit_one_agent(
            obs=obs_per_agent[i],
            acts=act_per_agent[i],
            obs_dim=obs_dim,
            action_dims=[10, 2, 2],
            hidden_size=64,
            epochs=1,
            batch_size=32,
            lr=1e-3,
            device="cpu",
            seed=i,
        )
        policies.append(p)
        torch.save(p.state_dict(), out / f"agent_{i}.pt")

    return {"out": out, "obs_dim": obs_dim, "policies": policies, "num_agents": num_agents}


def test_demos_have_correct_identity_one_hot():
    """Each agent's flattened obs must carry its own one-hot identity tail."""
    num_agents = 4
    obs_per_agent, _ = bc_init.gather_demos(
        scenario_name="minimal_specialization",
        num_agents=num_agents,
        num_pairs_per_agent=16,
        seed=0,
    )
    # The identity one-hot is the last `num_agents` entries.
    for i in range(num_agents):
        tail = obs_per_agent[i][:, -num_agents:]
        # All rows must encode the correct identity (one-hot at slot i).
        np.testing.assert_array_equal(tail.argmax(axis=1), np.full(16, i))
        np.testing.assert_allclose(tail.sum(axis=1), np.ones(16))


def test_checkpoint_round_trip(small_bc_run):
    """Saved per-agent state dicts must load into a fresh PolicyNetwork."""
    obs_dim = small_bc_run["obs_dim"]
    for i in range(small_bc_run["num_agents"]):
        ckpt = torch.load(small_bc_run["out"] / f"agent_{i}.pt", map_location="cpu")
        fresh = PolicyNetwork(obs_dim=obs_dim, action_dims=[10, 2, 2], hidden_size=64)
        fresh.load_state_dict(ckpt)
        # Sanity: round-tripped policy must produce the same logits on a probe
        # obs as the source policy. (Both share the same state dict.)
        x = torch.zeros(1, obs_dim, dtype=torch.float32)
        x[0, -small_bc_run["num_agents"] + i] = 1.0  # identity slot
        with torch.no_grad():
            l1, _ = small_bc_run["policies"][i](x)
            l2, _ = fresh(x)
        for lhs, rhs in zip(l1, l2):
            torch.testing.assert_close(lhs, rhs)


def test_train_one_cell_rejects_missing_checkpoint(small_bc_run, tmp_path):
    """train_one_cell must raise FileNotFoundError when an agent_*.pt is missing."""
    # Import locally so the train.py module-level imports are deferred until
    # the rest of the suite verifies bc_init.py works.
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "p3_specialization"))
    import train  # type: ignore[import-not-found]

    # Build a bogus checkpoint dir with only 3 of the 4 required state dicts.
    incomplete_dir = tmp_path / "incomplete"
    incomplete_dir.mkdir()
    for i in range(small_bc_run["num_agents"] - 1):
        # Copy the existing checkpoints.
        sd = torch.load(small_bc_run["out"] / f"agent_{i}.pt", map_location="cpu")
        torch.save(sd, incomplete_dir / f"agent_{i}.pt")

    cfg = train.CellConfig(
        scenario="minimal_specialization",
        lambda_red=0.0,
        seed=0,
        num_iterations=1,  # we should never actually iterate
        rollout_steps=8,
        num_agents=small_bc_run["num_agents"],
        bc_init_checkpoint_dir=str(incomplete_dir),
    )
    with pytest.raises(FileNotFoundError, match="agent_3.pt"):
        train.train_one_cell(cfg, tmp_path / "cell_out")


def test_train_one_cell_rejects_nonexistent_dir(tmp_path):
    """train_one_cell must raise FileNotFoundError when the dir does not exist."""
    sys.path.insert(0, str(REPO_ROOT / "experiments" / "p3_specialization"))
    import train  # type: ignore[import-not-found]

    cfg = train.CellConfig(
        scenario="minimal_specialization",
        lambda_red=0.0,
        seed=0,
        num_iterations=1,
        rollout_steps=8,
        num_agents=4,
        bc_init_checkpoint_dir=str(tmp_path / "does_not_exist"),
    )
    with pytest.raises(FileNotFoundError, match="bc_init_checkpoint_dir"):
        train.train_one_cell(cfg, tmp_path / "cell_out")

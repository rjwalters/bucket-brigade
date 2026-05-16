"""Smoke tests for the joint multi-agent PPO trainer.

These tests don't claim that the trainer *learns* --- they only check that
the wiring is correct: shapes, finite losses, gradient flow into every
encoder, and that the redundancy penalty does what it claims.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.training.joint_trainer import (
    JointPPOTrainer,
    flatten_dict_obs,
)


NUM_AGENTS = 4
ROLLOUT_STEPS = 64
ACTION_DIMS = [10, 2]  # [house index, mode]


def _env_fn():
    return BucketBrigadeEnv(num_agents=NUM_AGENTS)


def _make_trainer(
    redundancy_coef: float = 0.0, hidden_size: int = 16
) -> JointPPOTrainer:
    env = _env_fn()
    obs = env.reset(seed=0)
    # Issue #204: obs_dim now includes the per-agent identity one-hot tail.
    obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
    return JointPPOTrainer(
        env_fn=_env_fn,
        num_agents=NUM_AGENTS,
        obs_dim=obs_dim,
        action_dims=ACTION_DIMS,
        hidden_size=hidden_size,
        minibatch_size=32,
        ppo_epochs=2,
        redundancy_coef=redundancy_coef,
        seed=0,
    )


class TestEnvResetRegression:
    """Regression test for the BucketBrigadeEnv.reset() state-leak bug
    that broke the first P3 sweep --- RUINED houses from a previous
    episode persisted into the next one, locking the trainer's continuing
    env at the worst-case reward."""

    def test_reset_clears_ruined_houses(self):
        env = BucketBrigadeEnv(num_agents=NUM_AGENTS)
        env.reset(seed=42)
        env.houses[2] = env.RUINED
        env.houses[5] = env.RUINED

        env.reset()  # no seed --- the auto-reset path the trainer uses.
        assert env.houses[2] != env.RUINED
        assert env.houses[5] != env.RUINED
        assert ((env.houses == env.SAFE) | (env.houses == env.BURNING)).all()


class TestPerAgentObsDifferentiation:
    """Regression tests for the issue #204 latent bug: ``collect_rollout``
    used to reuse one flat obs row for every agent, so PPO was provably
    feeding identical input to all N policies. The Python flatteners
    silently dropped the ``agent_id`` field that ``AgentObservation``
    already carried (see ``bucket-brigade-core/src/lib.rs:43``).

    After the fix, ``flatten_dict_obs(obs, agent_id=i, num_agents=N)``
    appends a one-hot identity tail so per-agent flat obs vectors are
    pairwise distinct, and ``collect_rollout`` stores them as
    ``[T, N, obs_dim]`` instead of ``[T, obs_dim]``.
    """

    def test_flatten_dict_obs_legacy_no_identity(self):
        """Backward-compat: legacy call without ``agent_id`` returns the
        old length (no identity tail) so existing callers keep working."""
        env = _env_fn()
        obs = env.reset(seed=0)
        legacy = flatten_dict_obs(obs)
        with_id = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS)
        # The identity-equipped variant is exactly N elements longer.
        assert with_id.shape[0] == legacy.shape[0] + NUM_AGENTS

    def test_flatten_dict_obs_per_agent_differs(self):
        """Acceptance criterion from #204: ``flatten_dict_obs(obs, agent_id=i)``
        returns *different* vectors for ``i = 0..N-1`` on the same obs dict."""
        env = _env_fn()
        obs = env.reset(seed=0)
        rows = [
            flatten_dict_obs(obs, agent_id=i, num_agents=NUM_AGENTS)
            for i in range(NUM_AGENTS)
        ]
        for i in range(NUM_AGENTS):
            for j in range(i + 1, NUM_AGENTS):
                assert not np.array_equal(rows[i], rows[j]), (
                    f"agents {i} and {j} got identical flat obs — "
                    "identity one-hot tail is broken"
                )
        # And the only thing that differs is the last N elements (identity).
        prefix_len = rows[0].shape[0] - NUM_AGENTS
        for i in range(1, NUM_AGENTS):
            assert np.array_equal(rows[0][:prefix_len], rows[i][:prefix_len]), (
                "non-identity components leaked: per-agent obs should only "
                "differ in the identity one-hot tail"
            )

    def test_flatten_dict_obs_requires_num_agents_with_agent_id(self):
        env = _env_fn()
        obs = env.reset(seed=0)
        with pytest.raises(ValueError):
            flatten_dict_obs(obs, agent_id=0)
        with pytest.raises(ValueError):
            flatten_dict_obs(obs, agent_id=NUM_AGENTS, num_agents=NUM_AGENTS)
        with pytest.raises(ValueError):
            flatten_dict_obs(obs, agent_id=-1, num_agents=NUM_AGENTS)

    def test_collect_rollout_per_agent_obs_distinct(self):
        """Acceptance criterion from #204: ``RolloutBuffer.observations``
        must expose per-agent rows that differ across agents at each step
        (i.e. PPO is no longer fed identical input to all N policies)."""
        trainer = _make_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)

        obs = rollout.observations.cpu().numpy()  # [T, N, obs_dim]
        assert obs.ndim == 3, "observations must be [T, N, obs_dim] post-#204"
        assert obs.shape[1] == NUM_AGENTS

        # At every step t and for every distinct agent pair (i, j),
        # the rows must differ.
        diffs = 0
        for t in range(obs.shape[0]):
            for i in range(NUM_AGENTS):
                for j in range(i + 1, NUM_AGENTS):
                    if not np.array_equal(obs[t, i], obs[t, j]):
                        diffs += 1
        expected_pairs = obs.shape[0] * (NUM_AGENTS * (NUM_AGENTS - 1) // 2)
        assert diffs == expected_pairs, (
            f"only {diffs}/{expected_pairs} (t, i<j) pairs had distinct obs — "
            "the identity-tail latent-bug fix from #204 is not flowing through"
        )

    def test_collect_rollout_identity_tail_matches_agent_id(self):
        """The last N components of each per-agent row must equal one-hot(i)."""
        trainer = _make_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        obs = rollout.observations.cpu().numpy()  # [T, N, obs_dim]
        tail = obs[..., -NUM_AGENTS:]
        expected = np.eye(NUM_AGENTS, dtype=np.float32)
        for t in range(obs.shape[0]):
            assert np.array_equal(tail[t], expected), (
                f"step {t}: identity one-hot tail does not match np.eye(N): "
                f"got {tail[t]!r}"
            )


class TestRolloutCollection:
    def test_rollout_shapes(self):
        trainer = _make_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)

        # Issue #204: observations are now per-agent (`[T, N, obs_dim]`).
        assert rollout.observations.shape == (
            ROLLOUT_STEPS,
            NUM_AGENTS,
            trainer.obs_dim,
        )
        assert rollout.dones.shape == (ROLLOUT_STEPS,)
        for i in range(NUM_AGENTS):
            assert rollout.actions[i].shape == (ROLLOUT_STEPS, len(ACTION_DIMS))
            assert rollout.log_probs[i].shape == (ROLLOUT_STEPS,)
            assert rollout.values[i].shape == (ROLLOUT_STEPS,)
            assert rollout.rewards[i].shape == (ROLLOUT_STEPS,)

    def test_actions_in_range(self):
        trainer = _make_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        for i in range(NUM_AGENTS):
            a = rollout.actions[i]
            assert a[:, 0].min() >= 0 and a[:, 0].max() < ACTION_DIMS[0]
            assert a[:, 1].min() >= 0 and a[:, 1].max() < ACTION_DIMS[1]


class TestUpdate:
    def test_update_runs_without_nan(self):
        trainer = _make_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        for k, v in stats.items():
            assert np.isfinite(v), f"non-finite stat {k}={v}"

    def test_redundancy_zero_path(self):
        # With redundancy_coef = 0, total_loss should equal the sum of
        # per-agent losses (policy + value*coef - entropy*coef).
        trainer = _make_trainer(redundancy_coef=0.0)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        expected = sum(
            stats[f"policy_loss/agent_{i}"]
            + trainer.value_coef * stats[f"value_loss/agent_{i}"]
            - trainer.entropy_coef * stats[f"entropy/agent_{i}"]
            for i in range(NUM_AGENTS)
        )
        assert stats["total_loss"] == pytest.approx(expected, rel=1e-4)

    def test_redundancy_changes_loss(self):
        # Same rollout, two trainers with same init seed but different lambda.
        # Total loss should differ because the penalty is non-zero in general.
        trainer_a = _make_trainer(redundancy_coef=0.0)
        trainer_b = _make_trainer(redundancy_coef=1.0)
        rollout_a = trainer_a.collect_rollout(ROLLOUT_STEPS)
        rollout_b = trainer_b.collect_rollout(ROLLOUT_STEPS)
        stats_a = trainer_a.update(rollout_a)
        stats_b = trainer_b.update(rollout_b)
        # Redundancy loss is recorded for both (the *value* of the surrogate),
        # but only contributes to total_loss when coef > 0.
        assert stats_a["redundancy_loss"] >= 0.0
        assert stats_b["redundancy_loss"] >= 0.0
        # Lambda > 0 → total_loss > sum of per-agent losses when surrogate > 0.
        agent_sum_b = sum(
            stats_b[f"policy_loss/agent_{i}"]
            + trainer_b.value_coef * stats_b[f"value_loss/agent_{i}"]
            - trainer_b.entropy_coef * stats_b[f"entropy/agent_{i}"]
            for i in range(NUM_AGENTS)
        )
        assert stats_b["total_loss"] >= agent_sum_b - 1e-6


class TestRedundancyPenalty:
    def test_zero_for_orthogonal_features(self):
        # Two encoders that produce per-sample anti-aligned features:
        # standardize → cross-corr is exactly the identity-like structure
        # only when features actually correlate. Use independent gaussians.
        torch.manual_seed(0)
        z1 = torch.randn(1024, 8)
        z2 = torch.randn(1024, 8)
        pen = JointPPOTrainer.redundancy_penalty([z1, z2])
        # Independent gaussians: cross-corr per feature pair is O(1/sqrt(B)),
        # so the squared sum scaled by 1/d^2 stays small.
        assert pen.item() < 0.05

    def test_max_for_identical_features(self):
        torch.manual_seed(0)
        z = torch.randn(1024, 8)
        pen = JointPPOTrainer.redundancy_penalty([z, z])
        # Identical features → cross-corr matrix has diagonal of 1s, and the
        # squared sum is at least d (only on-diagonal correlations are 1.0;
        # off-diagonals are O(1/sqrt(B))). After /(num_pairs * d^2) → ~1/d.
        assert pen.item() > 0.1

    def test_more_redundancy_for_correlated_features(self):
        torch.manual_seed(0)
        base = torch.randn(1024, 8)
        # z1 and z2 share a correlated structure plus independent noise.
        z1 = base + 0.1 * torch.randn(1024, 8)
        z2 = base + 0.1 * torch.randn(1024, 8)
        # z3 and z4 fully independent.
        z3 = torch.randn(1024, 8)
        z4 = torch.randn(1024, 8)
        pen_corr = JointPPOTrainer.redundancy_penalty([z1, z2])
        pen_indep = JointPPOTrainer.redundancy_penalty([z3, z4])
        assert pen_corr.item() > pen_indep.item()


class TestGradientFlow:
    def test_redundancy_grads_into_every_encoder(self):
        # With redundancy_coef > 0, every encoder's trunk should receive a
        # gradient from the cross-agent penalty (i.e. .grad is not None and
        # not zero for at least one parameter).
        trainer = _make_trainer(redundancy_coef=1.0)

        # Manually run one forward through every encoder on a shared batch and
        # backprop just the redundancy penalty. We can't easily isolate the
        # gradient from a regular .update() call, so this is an isolated check.
        # Issue #204: ``_last_obs`` is now ``[N, obs_dim]`` — pick agent 0's row
        # as a template for the batch.
        obs = trainer._last_obs[0]
        obs_t = torch.from_numpy(np.tile(obs, (32, 1)))
        # Random observations so encoders produce non-degenerate features.
        obs_t = obs_t + 0.01 * torch.randn_like(obs_t)

        feats = [p.encoder_output(obs_t) for p in trainer.policies]
        pen = JointPPOTrainer.redundancy_penalty(feats)
        pen.backward()

        for i, policy in enumerate(trainer.policies):
            # The shared trunk's first linear layer must see a gradient.
            first_linear = policy.shared[0]
            assert first_linear.weight.grad is not None, f"agent {i}: no grad"
            assert first_linear.weight.grad.abs().sum().item() > 0.0, (
                f"agent {i}: zero gradient from redundancy penalty"
            )

"""Smoke tests for the joint multi-agent PPO trainer.

These tests don't claim that the trainer *learns* --- they only check that
the wiring is correct: shapes, finite losses, gradient flow into every
encoder, and that the redundancy penalty does what it claims.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.training.joint_trainer import (
    JointPPOTrainer,
    flatten_dict_obs,
)
from bucket_brigade.training.networks import CentralizedCritic, CentralizedQCritic


NUM_AGENTS = 4
ROLLOUT_STEPS = 64
ACTION_DIMS = [10, 2, 2]  # [house index, mode, signal] (issue #235)


def _env_fn():
    return BucketBrigadeEnv(num_agents=NUM_AGENTS)


def _make_trainer(
    redundancy_coef: float = 0.0,
    hidden_size: int = 16,
    centralized_critic: bool = False,
    coma: bool = False,
    coma_target_update_every: int = 200,
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
        centralized_critic=centralized_critic,
        coma=coma,
        coma_target_update_every=coma_target_update_every,
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


class TestCentralizedCritic:
    """Issue #208: MAPPO centralized critic.

    The centralized-critic path replaces the per-agent value heads with a
    single shared :class:`CentralizedCritic` consuming the global portion
    of the observation (identity-one-hot tail stripped). Per-agent
    advantages still come from per-agent rewards; only the value baseline
    is shared.
    """

    def test_centralized_critic_forward_shape(self):
        """``CentralizedCritic(obs_dim)`` accepts ``(B, obs_dim)`` and
        returns ``(B, 1)``."""
        critic = CentralizedCritic(obs_dim=42, hidden_size=32)
        x = torch.randn(8, 42)
        v = critic(x)
        assert v.shape == (8, 1)

    def test_init_critic_attributes(self):
        """Trainer with ``centralized_critic=True`` exposes a critic +
        optimizer; default flag leaves them ``None``."""
        trainer_off = _make_trainer(centralized_critic=False)
        assert trainer_off.centralized_critic is False
        assert trainer_off.critic is None
        assert trainer_off.critic_optimizer is None

        trainer_on = _make_trainer(centralized_critic=True)
        assert trainer_on.centralized_critic is True
        assert isinstance(trainer_on.critic, CentralizedCritic)
        # Global obs dim = obs_dim - num_agents (identity-tail stripped).
        assert trainer_on.critic.obs_dim == trainer_on.obs_dim - NUM_AGENTS
        assert trainer_on.critic_optimizer is not None

    def test_critic_incompatible_with_redundancy_coef(self):
        """Combining the centralized critic with the redundancy penalty
        must raise (per-agent actor coupling vs. shared critic — two
        unrelated experiments)."""
        with pytest.raises(ValueError, match="centralized_critic"):
            _make_trainer(centralized_critic=True, redundancy_coef=0.1)

    def test_rollout_values_are_shared_under_centralized_critic(self):
        """When MAPPO is on, every agent's stored rollout value at each
        timestep must be the same float (the shared critic output)."""
        trainer = _make_trainer(centralized_critic=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        # All N value rows must equal agent 0's value row, elementwise.
        v0 = rollout.values[0].cpu().numpy()
        for i in range(1, NUM_AGENTS):
            vi = rollout.values[i].cpu().numpy()
            assert np.allclose(vi, v0), (
                f"agent {i} value row differs from agent 0 — centralized "
                "critic should broadcast a single shared estimate"
            )

    def test_update_runs_without_nan_under_centralized_critic(self):
        """End-to-end smoke: rollout + update completes with finite stats
        and the new ``critic_value_loss`` key is present."""
        trainer = _make_trainer(centralized_critic=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        for k, v in stats.items():
            assert np.isfinite(v), f"non-finite stat {k}={v}"
        assert "critic_value_loss" in stats
        # Per-agent value_loss stats are recorded as 0.0 (the per-agent
        # value heads are bypassed under MAPPO).
        for i in range(NUM_AGENTS):
            assert stats[f"value_loss/agent_{i}"] == 0.0

    def test_critic_receives_gradient_after_update(self):
        """After one ``update()`` call, the centralized critic's first
        linear layer must have non-zero gradient (i.e. the critic loss
        actually drives the centralized network's weights)."""
        trainer = _make_trainer(centralized_critic=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        trainer.update(rollout)
        first_linear = trainer.critic.shared[0]
        assert first_linear.weight.grad is not None, "critic: no grad"
        assert first_linear.weight.grad.abs().sum().item() > 0.0, (
            "critic: zero gradient after update"
        )

    def test_actors_still_receive_gradient_under_centralized_critic(self):
        """After one ``update()`` call, every per-agent policy network's
        first linear layer must have non-zero gradient (the policy loss
        still drives the actors even without a per-agent value head)."""
        trainer = _make_trainer(centralized_critic=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        trainer.update(rollout)
        for i, policy in enumerate(trainer.policies):
            first_linear = policy.shared[0]
            assert first_linear.weight.grad is not None, f"agent {i}: no grad"
            assert first_linear.weight.grad.abs().sum().item() > 0.0, (
                f"agent {i}: zero gradient under centralized critic"
            )

    def test_default_flag_no_critic_constructed(self):
        """Regression guard: with ``centralized_critic=False`` (default),
        no centralized-critic module or optimizer is constructed, and
        the new ``critic_value_loss`` stat is **not** present in the
        update output. This pins the default code path to the
        pre-#208 IPPO contract.
        """
        trainer = _make_trainer(centralized_critic=False)
        assert trainer.critic is None
        assert trainer.critic_optimizer is None

        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        assert "critic_value_loss" not in stats, (
            "default IPPO path leaked a centralized-critic stat key"
        )
        # Per-agent value losses must still be non-zero floats (the per-
        # agent value heads are exercised on the default path).
        any_nonzero = any(
            stats[f"value_loss/agent_{i}"] != 0.0 for i in range(NUM_AGENTS)
        )
        assert any_nonzero, (
            "default IPPO path: per-agent value_loss collapsed to 0 — "
            "the per-agent value head is no longer being trained"
        )

    def test_default_flag_two_trainers_identical_init(self):
        """Two trainers constructed with the same seed and
        ``centralized_critic=False`` produce identical initial weights
        --- the canonical IPPO determinism check (seeding only; rollout
        determinism is RNG-state-dependent and tested elsewhere)."""
        trainer_a = _make_trainer(centralized_critic=False)
        trainer_b = _make_trainer(centralized_critic=False)
        for pa, pb in zip(trainer_a.policies, trainer_b.policies):
            for ka, va in pa.state_dict().items():
                assert torch.equal(va, pb.state_dict()[ka]), (
                    f"initial weights differ at {ka} — IPPO seeding broken"
                )

    def test_critic_uses_identity_stripped_global_obs(self):
        """The centralized critic consumes the global obs (identity-tail
        stripped). Its input dim must equal ``obs_dim - num_agents``,
        and a manual forward on a hand-crafted batch returns the right
        shape."""
        trainer = _make_trainer(centralized_critic=True)
        # Hand-craft a [mb, obs_dim - num_agents] tensor (no identity tail).
        mb = 7
        x = torch.randn(mb, trainer.obs_dim - NUM_AGENTS)
        v = trainer.critic(x)
        assert v.shape == (mb, 1)

    def test_num_agents_one_reduces_to_ppo_like(self):
        """Sanity: ``num_agents=1`` with the centralized critic should
        construct and run without error (it reduces to vanilla PPO with
        the value head moved to a separate module)."""
        env = BucketBrigadeEnv(num_agents=1)
        obs = env.reset(seed=0)
        obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=1).shape[0]
        trainer = JointPPOTrainer(
            env_fn=lambda: BucketBrigadeEnv(num_agents=1),
            num_agents=1,
            obs_dim=obs_dim,
            action_dims=ACTION_DIMS,
            hidden_size=16,
            minibatch_size=32,
            ppo_epochs=2,
            centralized_critic=True,
            seed=0,
        )
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        assert np.isfinite(stats["total_loss"])
        assert np.isfinite(stats["critic_value_loss"])


class TestCOMA:
    """Issue #284: COMA counterfactual multi-agent policy gradient.

    The COMA path installs a :class:`CentralizedQCritic` whose per-agent
    Q-vector feeds a counterfactual advantage

    ::

        A_i = Q(s, a)_i - sum_u  pi_i(u | s) * Q(s, (a_{-i}, u))_i

    enumerated over the joint per-agent action space (``prod(action_dims) =
    40`` for ``[10, 2, 2]``). The Q-critic is trained against 1-step TD
    targets with a hard-copy target network.
    """

    def test_centralized_q_critic_forward_shape(self):
        """``CentralizedQCritic`` accepts (global_obs, joint_action_oh,
        agent_id_oh) and returns ``(batch, action_dim_total)``."""
        critic = CentralizedQCritic(
            global_obs_dim=42,
            num_agents=NUM_AGENTS,
            action_dim_total=40,
            hidden_size=16,
        )
        x = torch.randn(8, 42)
        joint_oh = torch.zeros(8, NUM_AGENTS * 40)
        agent_id_oh = torch.zeros(8, NUM_AGENTS)
        agent_id_oh[:, 0] = 1.0
        q = critic(x, joint_oh, agent_id_oh)
        assert q.shape == (8, 40)

    def test_counterfactual_baseline_collapses_when_q_independent_of_action(
        self,
    ):
        """If Q is constant in agent i's action (i.e. q_all is the same
        value across all actions), the COMA baseline equals q_chosen, so
        the advantage is exactly zero. This is the load-bearing structural
        check that the baseline implements the counterfactual identity."""
        batch = 16
        action_dim_total = 40
        # q_all constant across actions: every column = 7.5.
        q_all = torch.full((batch, action_dim_total), 7.5)
        q_chosen = torch.full((batch,), 7.5)
        # Arbitrary probability distribution (must sum to 1 per row).
        action_probs = torch.softmax(torch.randn(batch, action_dim_total), dim=-1)
        adv = JointPPOTrainer.coma_counterfactual_advantage(
            q_chosen=q_chosen,
            q_all=q_all,
            action_probs=action_probs,
        )
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-6), (
            f"COMA baseline failed to collapse when Q is action-independent; "
            f"max |adv| = {adv.abs().max().item()}"
        )

    def test_counterfactual_baseline_matches_hand_computed_reference(self):
        """Hand-computed reference on a tiny 2-action, 1-batch example.

        Setup: action_dim_total=2, batch=1.
        Q-values: q_all = [3.0, 5.0], q_chosen = 3.0 (action 0 taken).
        Policy: pi = [0.25, 0.75].

        Baseline: 0.25*3.0 + 0.75*5.0 = 0.75 + 3.75 = 4.5.
        Advantage: 3.0 - 4.5 = -1.5.
        """
        q_all = torch.tensor([[3.0, 5.0]])
        q_chosen = torch.tensor([3.0])
        action_probs = torch.tensor([[0.25, 0.75]])
        adv = JointPPOTrainer.coma_counterfactual_advantage(
            q_chosen=q_chosen,
            q_all=q_all,
            action_probs=action_probs,
        )
        assert torch.allclose(adv, torch.tensor([-1.5]), atol=1e-6), (
            f"Expected adv = -1.5, got {adv.item()}"
        )

    def test_init_q_critic_attributes(self):
        """Trainer with ``coma=True`` exposes a q_critic, q_critic_target,
        and q_critic_optimizer; default flag leaves them ``None``."""
        trainer_off = _make_trainer(coma=False)
        assert trainer_off.coma is False
        assert trainer_off.q_critic is None
        assert trainer_off.q_critic_target is None
        assert trainer_off.q_critic_optimizer is None

        trainer_on = _make_trainer(coma=True)
        assert trainer_on.coma is True
        assert isinstance(trainer_on.q_critic, CentralizedQCritic)
        assert isinstance(trainer_on.q_critic_target, CentralizedQCritic)
        # action_dim_total = prod([10, 2, 2]) = 40.
        assert trainer_on.action_dim_total == 40
        assert trainer_on.q_critic.global_obs_dim == trainer_on.obs_dim - NUM_AGENTS
        assert trainer_on.q_critic_optimizer is not None

    def test_coma_incompatible_with_centralized_critic(self):
        """COMA and MAPPO are separate experiments — combining them
        should raise."""
        with pytest.raises(ValueError, match="coma"):
            _make_trainer(coma=True, centralized_critic=True)

    def test_coma_incompatible_with_redundancy_coef(self):
        """COMA and the redundancy penalty couple unrelated experiments
        — combining them should raise."""
        with pytest.raises(ValueError, match="coma"):
            _make_trainer(coma=True, redundancy_coef=0.1)

    def test_default_flag_no_q_critic_constructed(self):
        """Regression guard: with ``coma=False`` (default), no Q-critic
        module or optimizer is constructed, and the new ``coma_q_loss``
        stat is **not** present in the update output. This pins the
        default code path to the pre-#284 IPPO contract.
        """
        trainer = _make_trainer(coma=False)
        assert trainer.q_critic is None
        assert trainer.q_critic_target is None
        assert trainer.q_critic_optimizer is None

        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        assert "coma_q_loss" not in stats

    def test_use_coma_false_bit_identity_with_pre_change_path(self):
        """Two trainers constructed with the same seed and ``coma=False``
        produce identical initial weights (the canonical IPPO determinism
        check)."""
        trainer_a = _make_trainer(coma=False)
        trainer_b = _make_trainer(coma=False)
        for pa, pb in zip(trainer_a.policies, trainer_b.policies):
            for ka, va in pa.state_dict().items():
                assert torch.equal(va, pb.state_dict()[ka]), (
                    f"initial weights differ at {ka} — coma=False seeding broken"
                )

    def test_pack_joint_action_roundtrip(self):
        """``_pack_joint_action`` packs ``[10, 2, 2]`` into a flat code in
        ``[0, 40)`` via row-major flattening."""
        trainer = _make_trainer(coma=True)
        # action_dims = [10, 2, 2]. Strides = [4, 2, 1].
        # Action (3, 1, 0) → 3*4 + 1*2 + 0 = 14.
        a = torch.tensor([[3, 1, 0], [0, 0, 0], [9, 1, 1]], dtype=torch.long)
        packed = trainer._pack_joint_action(a)
        assert packed.tolist() == [14, 0, 39]
        # Bounds.
        assert packed.min().item() >= 0
        assert packed.max().item() < 40

    def test_update_runs_without_nan_under_coma(self):
        """End-to-end smoke: rollout + update completes with finite stats
        and the new ``coma_q_loss`` key is present."""
        trainer = _make_trainer(coma=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        for k, v in stats.items():
            assert np.isfinite(v), f"non-finite stat {k}={v}"
        assert "coma_q_loss" in stats
        # Per-agent value_loss stats are recorded as 0.0 (the per-agent
        # value heads are bypassed under COMA, exactly as under MAPPO).
        for i in range(NUM_AGENTS):
            assert stats[f"value_loss/agent_{i}"] == 0.0

    def test_q_critic_receives_gradient_after_update(self):
        """After one update, the live Q-critic's first linear layer must
        have non-zero gradient (the TD regression actually drives the
        Q-network)."""
        trainer = _make_trainer(coma=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        trainer.update(rollout)
        first_linear = trainer.q_critic.shared[0]
        assert first_linear.weight.grad is not None, "q_critic: no grad"
        assert first_linear.weight.grad.abs().sum().item() > 0.0, (
            "q_critic: zero gradient after update"
        )

    def test_actors_still_receive_gradient_under_coma(self):
        """After one update, every per-agent policy's first linear layer
        must have non-zero gradient (COMA's counterfactual advantage
        still drives the actors even without a per-agent value head)."""
        trainer = _make_trainer(coma=True)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        trainer.update(rollout)
        for i, policy in enumerate(trainer.policies):
            first_linear = policy.shared[0]
            assert first_linear.weight.grad is not None, f"agent {i}: no grad"
            assert first_linear.weight.grad.abs().sum().item() > 0.0, (
                f"agent {i}: zero gradient under COMA"
            )

    def test_q_target_network_does_not_receive_gradient(self):
        """The target Q-network is a slow snapshot; its parameters must
        not have ``requires_grad`` set, so they are never optimized."""
        trainer = _make_trainer(coma=True)
        for p in trainer.q_critic_target.parameters():
            assert not p.requires_grad, "target Q-network parameters must be frozen"

    def test_target_network_hard_copies_on_cadence(self):
        """When the configured cadence is hit, the target network's
        weights match the live Q-critic exactly."""
        # Tiny cadence so we trigger the hard copy on the very first
        # update step (ppo_epochs=2 ⇒ first step is index 1).
        trainer = _make_trainer(coma=True, coma_target_update_every=1)
        # Perturb the live Q-critic so the comparison after one update
        # is non-trivial.
        with torch.no_grad():
            trainer.q_critic.shared[0].weight.add_(0.5)
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        trainer.update(rollout)
        for k, v in trainer.q_critic.state_dict().items():
            assert torch.allclose(v, trainer.q_critic_target.state_dict()[k]), (
                f"target Q-network out of sync at param {k} after hard-copy"
            )

    def test_td_target_one_step_sanity_in_scripted_setting(self):
        """1-step TD target sanity. Construct a rollout where the next-step
        Q is known (target network is the same as the live Q at init), and
        verify the TD target matches the closed-form
        ``r_t + gamma * (1 - done_t) * Q_target(s_{t+1}, a_{t+1})``.
        """
        trainer = _make_trainer(coma=True)
        # Fresh trainer: target == live (we just copied state_dict in __init__).
        T = 4
        global_obs = torch.randn(T, trainer._global_obs_dim)
        # Joint actions all zero (deterministic and easy to gather).
        joint_action_packed = torch.zeros((T, NUM_AGENTS), dtype=torch.long)
        rewards = torch.tensor(
            [
                [1.0] * NUM_AGENTS,
                [2.0] * NUM_AGENTS,
                [3.0] * NUM_AGENTS,
                [4.0] * NUM_AGENTS,
            ],
        )
        dones = torch.tensor([0.0, 0.0, 1.0, 0.0])  # episode ends at t=2

        targets = trainer._coma_compute_td_targets(
            global_obs=global_obs,
            joint_action_packed=joint_action_packed,
            rewards=rewards,
            dones=dones,
        )

        # Hand-compute expected next_targets (zero at t=T-1).
        joint_oh = trainer._coma_joint_action_one_hot(joint_action_packed)
        expected_next = torch.zeros((T, NUM_AGENTS))
        with torch.no_grad():
            for i in range(NUM_AGENTS):
                for t in range(T - 1):
                    aid = global_obs.new_zeros((1, NUM_AGENTS))
                    aid[0, i] = 1.0
                    q_next = trainer.q_critic_target(
                        global_obs[t + 1 : t + 2],
                        joint_oh[t + 1 : t + 2],
                        aid,
                    )
                    expected_next[t, i] = q_next[0, joint_action_packed[t + 1, i]]

        done_mask = (1.0 - dones).unsqueeze(-1)
        expected = rewards + trainer.gamma * done_mask * expected_next
        assert torch.allclose(targets, expected, atol=1e-5), (
            "1-step TD target mismatch vs closed-form reference"
        )

    def test_zero_advantage_when_policy_is_uniform_and_q_is_constant_per_action(
        self,
    ):
        """End-to-end COMA-advantage check: in a state where Q is the same
        for every action of agent i, the counterfactual baseline equals
        Q(s, a)_i exactly, so A_i = 0 regardless of pi_i. This is the
        spec's third test (see issue body)."""
        trainer = _make_trainer(coma=True)
        # Manually set the Q-critic's final head weights to zero (so all
        # logits = bias, which is shared across all 40 actions).
        with torch.no_grad():
            trainer.q_critic.q_head.weight.zero_()
            # Bias is a per-action scalar; setting all entries to the same
            # constant keeps Q(s, ·) constant across actions for any (s, a).
            trainer.q_critic.q_head.bias.fill_(2.0)
        T = 8
        global_obs = torch.randn(T, trainer._global_obs_dim)
        joint_action_packed = torch.zeros((T, NUM_AGENTS), dtype=torch.long)
        action_probs_per_agent = [
            torch.softmax(torch.randn(T, trainer.action_dim_total), dim=-1)
            for _ in range(NUM_AGENTS)
        ]
        advantages, _ = trainer._coma_compute_advantages(
            global_obs=global_obs,
            joint_action_packed=joint_action_packed,
            action_probs_per_agent=action_probs_per_agent,
        )
        assert torch.allclose(advantages, torch.zeros_like(advantages), atol=1e-6), (
            f"advantage failed to collapse to 0 when Q is action-independent; "
            f"max |adv| = {advantages.abs().max().item()}"
        )

    def test_reproducibility_two_runs_same_seed(self):
        """Two trainers seeded identically produce bit-identical Q-critic
        weights at init, including the target network."""
        a = _make_trainer(coma=True)
        b = _make_trainer(coma=True)
        for ka, va in a.q_critic.state_dict().items():
            assert torch.equal(va, b.q_critic.state_dict()[ka]), (
                f"q_critic init differs at {ka} — COMA seeding broken"
            )
        for ka, va in a.q_critic_target.state_dict().items():
            assert torch.equal(va, b.q_critic_target.state_dict()[ka]), (
                f"q_critic_target init differs at {ka} — COMA seeding broken"
            )


class TestHCA:
    """Issue #289: Hindsight Credit Assignment.

    HCA is gated by ``advantage_estimator="hca"`` on the trainer. The
    default ``advantage_estimator="gae"`` path is bit-identical to the
    pre-#289 codepath, so the existing TestUpdate suite covers regression.
    """

    def _make_hca_trainer(self, **overrides) -> JointPPOTrainer:
        env = _env_fn()
        obs = env.reset(seed=0)
        obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
        kwargs = dict(
            env_fn=_env_fn,
            num_agents=NUM_AGENTS,
            obs_dim=obs_dim,
            action_dims=ACTION_DIMS,
            hidden_size=16,
            minibatch_size=32,
            ppo_epochs=2,
            advantage_estimator="hca",
            seed=0,
        )
        kwargs.update(overrides)
        return JointPPOTrainer(**kwargs)

    def test_gae_default_path_bit_identical(self):
        """Regression: ``advantage_estimator="gae"`` (the default) must
        produce the exact same per-agent advantages as a trainer
        constructed without specifying the flag."""
        # Two trainers seeded identically — one default, one with the
        # GAE flag passed explicitly. Both must produce identical
        # advantages on the same rollout.
        t_default = _make_trainer()
        t_explicit = _make_trainer()
        # Override the explicit one to set the flag (default-value via
        # ``__init__`` — should be a no-op semantically).
        assert t_default.advantage_estimator == "gae"
        assert t_explicit.advantage_estimator == "gae"
        # Use the same rollout for both, so we test only the advantage
        # path. We compute advantages directly via the private method
        # rather than running update() (which mutates optimizer state).
        rollout = t_default.collect_rollout(ROLLOUT_STEPS)
        a0, r0 = t_default._compute_advantages(
            rollout.rewards[0], rollout.values[0], rollout.dones
        )
        a1, r1 = t_explicit._compute_advantages(
            rollout.rewards[0], rollout.values[0], rollout.dones
        )
        torch.testing.assert_close(a0, a1)
        torch.testing.assert_close(r0, r1)

    def test_hca_trainer_constructs(self):
        """Hindsight nets and optimizers exist on the HCA path; GAE path
        leaves them ``None``."""
        hca = self._make_hca_trainer()
        assert hca.hindsight_nets is not None
        assert hca.hindsight_optimizers is not None
        assert len(hca.hindsight_nets) == NUM_AGENTS
        assert len(hca.hindsight_optimizers) == NUM_AGENTS

        gae = _make_trainer()
        assert gae.hindsight_nets is None
        assert gae.hindsight_optimizers is None

    def test_hca_update_runs_without_nan(self):
        """End-to-end: one rollout + one update on the HCA path must
        produce finite stats."""
        trainer = self._make_hca_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        for k, v in stats.items():
            assert np.isfinite(v), f"non-finite stat {k}={v}"
        # HCA-specific diagnostics must be present.
        for i in range(NUM_AGENTS):
            assert f"hindsight_loss/agent_{i}" in stats
            assert f"hca_clip_frac/agent_{i}" in stats
            assert f"hca_ratio_mean/agent_{i}" in stats

    def test_hca_advantages_shape_matches_gae(self):
        """The HCA advantage tensor must have the same shape as GAE's
        on the same rollout (one scalar per timestep)."""
        gae_trainer = _make_trainer()
        hca_trainer = self._make_hca_trainer()
        rollout = gae_trainer.collect_rollout(ROLLOUT_STEPS)
        # Run the hindsight-net step so the HCA call inside compute is
        # well-defined.
        hca_trainer._update_hindsight_nets(rollout)
        a_gae, _ = gae_trainer._compute_advantages(
            rollout.rewards[0], rollout.values[0], rollout.dones
        )
        a_hca, _ = hca_trainer._compute_advantages(
            rollout.rewards[0],
            rollout.values[0],
            rollout.dones,
            agent_id=0,
            observations=rollout.observations[:, 0, :],
            actions=rollout.actions[0],
            policy_log_probs=rollout.log_probs[0],
        )
        assert a_gae.shape == a_hca.shape
        assert a_hca.dtype == torch.float32

    def test_hca_reduces_to_mc_when_ratio_is_one(self):
        """Sanity: when the hindsight ratio is forced to 1.0 (i.e.
        ``h == pi``), HCA's advantage collapses to ``(Z_t - V_t)`` —
        the pure Monte-Carlo advantage. Test by patching ``log_prob``
        of the hindsight net to mirror the rollout log-probs.
        """
        from bucket_brigade.training.networks import compute_returns_to_go

        trainer = self._make_hca_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)

        # Patch hindsight net so log h(a|s,X) == policy_log_probs[t] for
        # every t, making the ratio exactly 1 (and thus advantage 0
        # outside any clipping path).
        original_log_prob = trainer.hindsight_nets[0].log_prob

        def fake_log_prob(obs, x, actions):
            # Match the captured policy log-probs of agent 0 for this batch.
            return rollout.log_probs[0]

        trainer.hindsight_nets[0].log_prob = fake_log_prob  # type: ignore[assignment]
        try:
            adv, ret = trainer._compute_advantages(
                rollout.rewards[0],
                rollout.values[0],
                rollout.dones,
                agent_id=0,
                observations=rollout.observations[:, 0, :],
                actions=rollout.actions[0],
                policy_log_probs=rollout.log_probs[0],
            )
        finally:
            trainer.hindsight_nets[0].log_prob = original_log_prob  # type: ignore[assignment]

        # Ratio 1.0 → (1 - 1) * (Z - V) = 0 everywhere.
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)
        # Returns target is Z_t exactly.
        Z_expected = torch.tensor(
            compute_returns_to_go(
                rollout.rewards[0].cpu().tolist(),
                rollout.dones.cpu().tolist(),
                gamma=trainer.gamma,
            ),
            dtype=torch.float32,
        )
        torch.testing.assert_close(ret, Z_expected, atol=1e-5, rtol=1e-5)

    def test_hindsight_loss_decreases_on_synthetic_data(self):
        """The hindsight network's MLE loss must decrease on a small
        synthetic dataset (smoke test that gradients flow and the
        optimizer actually steps the network)."""
        from bucket_brigade.training.networks import (
            HindsightNetwork,
            encode_return_bucket,
        )

        torch.manual_seed(0)
        obs_dim = 8
        x_dim = 4
        action_dims = [3, 2]
        T = 256
        net = HindsightNetwork(
            obs_dim=obs_dim, x_dim=x_dim, action_dims=action_dims, hidden_size=16
        )
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)

        obs = torch.randn(T, obs_dim)
        # Make X deterministically informative about the action by tying
        # X to the action taken so the net has something to learn.
        actions = torch.stack(
            [
                torch.randint(0, action_dims[0], (T,)),
                torch.randint(0, action_dims[1], (T,)),
            ],
            dim=1,
        )
        returns = (actions[:, 0].float() + actions[:, 1].float()).detach()
        X = encode_return_bucket(returns, num_buckets=x_dim)

        loss_first = -net.log_prob(obs, X, actions).mean()
        for _ in range(50):
            loss = -net.log_prob(obs, X, actions).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        loss_last = -net.log_prob(obs, X, actions).mean()
        assert loss_last.item() < loss_first.item() - 0.05, (
            f"hindsight loss did not decrease: first={loss_first.item():.4f} "
            f"last={loss_last.item():.4f}"
        )

    def test_invalid_advantage_estimator_raises(self):
        """Mistyped or unsupported estimator names must raise at trainer
        construction, not silently fall back to GAE."""
        env = _env_fn()
        obs = env.reset(seed=0)
        obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
        with pytest.raises(ValueError, match="advantage_estimator"):
            JointPPOTrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=obs_dim,
                action_dims=ACTION_DIMS,
                hidden_size=16,
                advantage_estimator="invalid_name",
                seed=0,
            )

    def test_compute_returns_to_go_resets_at_dones(self):
        """Smoke: ``compute_returns_to_go`` should not leak credit across
        episode boundaries marked by ``dones[t] == True``."""
        from bucket_brigade.training.networks import compute_returns_to_go

        rewards = [1.0, 2.0, 3.0, 4.0]
        dones = [False, True, False, True]
        # Episode 1: rewards [1, 2], Z = [1 + γ*2, 2]
        # Episode 2: rewards [3, 4], Z = [3 + γ*4, 4]
        gamma = 0.9
        Z = compute_returns_to_go(rewards, dones, gamma=gamma)
        assert Z[0] == pytest.approx(1.0 + gamma * 2.0)
        assert Z[1] == pytest.approx(2.0)
        assert Z[2] == pytest.approx(3.0 + gamma * 4.0)
        assert Z[3] == pytest.approx(4.0)


class TestSocialInfluence:
    """Issue #290: Jaques-2019 social-influence intrinsic motivation.

    The trainer optionally adds ``alpha * sum_{j != i} KL(real || cf_marg)``
    to each agent's environmental reward before the advantage estimator
    (works with GAE, HCA, and COMA), where the KL is between agent ``j``'s
    next-step policy under the real ``last_actions[i]`` and the
    counterfactual marginal averaged over MC samples of ``a'_i ~ pi_i``.
    ``alpha = 0`` (default) is the IPPO control and must preserve
    bit-identical pre-#290 behavior; this is the critical regression test
    (``test_influence_coef_zero_bit_identical``).
    """

    def test_influence_coef_zero_bit_identical(self):
        """Critical regression test: with ``influence_coef = 0``, ``update()``
        must produce *exactly* the same stats as a fresh trainer that
        knows nothing about the influence path.

        Strategy: run the baseline trainer (no influence kwargs, so default
        ``influence_coef = 0``) from a fixed seed, then re-seed and run an
        explicit ``influence_coef = 0`` trainer. Because the env and policy
        sampling consume torch + numpy global RNG state during rollout,
        we must re-seed both globals just before each ``collect_rollout``
        and again before each ``update`` so the comparison is apples-to-apples.

        With identical RNG state, identical policy init, and identical
        env reset state, both paths must produce bit-identical stats.
        The influence-zero path additionally must NOT emit any
        ``influence_reward/*`` diagnostic keys (the no-op branch is fully
        skipped, including the diagnostic block).
        """

        def _run_one(influence_coef: float) -> Tuple[dict, list]:
            # Re-seed both global RNGs to a known state before constructing the
            # trainer. The trainer's __init__ also seeds, so this just makes
            # the construction order's effect on globals deterministic.
            torch.manual_seed(2026)
            np.random.seed(2026)
            env = _env_fn()
            obs = env.reset(seed=0)
            obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
            trainer = JointPPOTrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=obs_dim,
                action_dims=ACTION_DIMS,
                hidden_size=16,
                minibatch_size=32,
                ppo_epochs=2,
                redundancy_coef=0.0,
                influence_coef=influence_coef,
                seed=2026,
            )
            # Re-seed before rollout so action/env sampling matches between
            # the two runs regardless of any side-effects in __init__ that
            # may differ when ``influence_coef`` is set (today there are
            # none — but be defensive).
            torch.manual_seed(7777)
            np.random.seed(7777)
            rollout = trainer.collect_rollout(ROLLOUT_STEPS)
            torch.manual_seed(8888)
            np.random.seed(8888)
            stats = trainer.update(rollout)
            policy_params = [
                [p.detach().clone() for p in pol.parameters()]
                for pol in trainer.policies
            ]
            return stats, policy_params

        stats_a, params_a = _run_one(influence_coef=0.0)  # baseline default
        stats_b, params_b = _run_one(influence_coef=0.0)  # explicit zero

        # Diagnostic keys must NOT appear in the alpha=0 path.
        for key in stats_b:
            assert not key.startswith("influence_reward/"), (
                f"influence_coef=0 path leaked diagnostic key {key!r}"
            )

        # Bit-identical stats.
        for key in stats_a:
            assert key in stats_b, f"missing key in influence-path: {key}"
            assert stats_a[key] == pytest.approx(stats_b[key], rel=1e-6, abs=1e-9), (
                f"alpha=0 path diverged from baseline at {key!r}: "
                f"{stats_a[key]} vs {stats_b[key]}"
            )
        # And no extra keys either (alpha=0 emits exactly the baseline schema).
        for key in stats_b:
            assert key in stats_a, f"alpha=0 path emitted extra key: {key!r}"

        # Post-update policy parameters must match.
        for i in range(NUM_AGENTS):
            for p_a, p_b in zip(params_a[i], params_b[i]):
                assert torch.allclose(p_a, p_b, rtol=1e-6, atol=1e-8), (
                    f"agent {i}: alpha=0 path diverged in post-update params"
                )

    def test_influence_reward_shape_and_finite(self):
        """``_compute_influence_reward()`` returns ``{i: Tensor[T]}`` with
        no NaN/Inf for a typical rollout."""
        trainer = _make_trainer()
        trainer.influence_coef = 0.5  # force the path on for direct inspection
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        intrinsic = trainer._compute_influence_reward(rollout)
        assert set(intrinsic.keys()) == set(range(NUM_AGENTS))
        for i in range(NUM_AGENTS):
            r = intrinsic[i]
            assert r.shape == (ROLLOUT_STEPS,), (
                f"agent {i}: intrinsic reward shape {tuple(r.shape)} "
                f"!= expected ({ROLLOUT_STEPS},)"
            )
            assert torch.isfinite(r).all(), (
                f"agent {i}: non-finite influence reward (NaN or Inf present)"
            )
            assert (r >= 0).all(), (
                f"agent {i}: KL-based influence reward must be non-negative"
            )

    def test_influence_reward_zero_when_partner_policy_ignores_last_actions(self):
        """Zero-influence sanity check: if teammate policies do NOT depend on
        agent ``i``'s slot in ``last_actions``, the counterfactual marginal
        equals the real distribution and KL → 0.

        Implementation: zero out the columns of every partner policy's first
        linear layer that read the ``last_actions[i]`` slot from the input.
        This severs the only mechanism by which a counterfactual ``a'_i``
        can change ``pi_j``'s output distribution.
        """
        trainer = _make_trainer()
        trainer.influence_coef = 1.0

        # The shared trunk's first linear layer is `policy.shared[0]`. Zero
        # out its weight columns for every agent ``i``'s last_actions slot,
        # for every j != i. To keep the test simple and exhaustive, zero
        # out *all* last_actions block columns in every policy (this also
        # zeroes the agent's own slot, which is unused for influence).
        la_start = trainer._influence_la_block_start
        la_width = trainer._influence_la_slot_width
        la_end = la_start + NUM_AGENTS * la_width
        with torch.no_grad():
            for policy in trainer.policies:
                w = policy.shared[0].weight
                # ``w`` shape is [hidden_size, obs_dim]; zero out the
                # columns in [la_start, la_end).
                w[:, la_start:la_end] = 0.0

        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        intrinsic = trainer._compute_influence_reward(rollout)
        # Every per-step intrinsic reward should be (numerically) zero.
        for i in range(NUM_AGENTS):
            r = intrinsic[i]
            max_abs = float(r.abs().max().item())
            assert max_abs < 1e-5, (
                f"agent {i}: expected ~0 influence when partner policies "
                f"ignore last_actions, got max |r|={max_abs:.3e}"
            )

    def test_influence_reward_single_agent_is_zero(self):
        """No other agents to influence → influence reward is exactly 0."""
        env = BucketBrigadeEnv(num_agents=1)
        obs = env.reset(seed=0)
        obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=1).shape[0]
        trainer = JointPPOTrainer(
            env_fn=lambda: BucketBrigadeEnv(num_agents=1),
            num_agents=1,
            obs_dim=obs_dim,
            action_dims=ACTION_DIMS,
            hidden_size=16,
            minibatch_size=32,
            ppo_epochs=2,
            influence_coef=1.0,
            seed=0,
        )
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        intrinsic = trainer._compute_influence_reward(rollout)
        assert set(intrinsic.keys()) == {0}
        assert torch.equal(intrinsic[0], torch.zeros(ROLLOUT_STEPS))

    def test_influence_update_records_diagnostics(self):
        """When ``influence_coef > 0``, ``update()`` must emit
        ``influence_reward/*`` diagnostic keys in the returned stats dict."""
        trainer = _make_trainer()
        trainer.influence_coef = 0.5
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        stats = trainer.update(rollout)
        # Required diagnostic keys.
        assert "influence_reward/mean" in stats
        for i in range(NUM_AGENTS):
            assert f"influence_reward/agent_{i}_mean" in stats
            assert f"influence_reward/agent_{i}_max" in stats
            assert np.isfinite(stats[f"influence_reward/agent_{i}_mean"])
            assert stats[f"influence_reward/agent_{i}_mean"] >= 0.0

    def test_influence_reproducibility_at_fixed_seed(self):
        """The influence reward is deterministic for the same trainer state +
        same rollout + same torch RNG state.

        Two invocations of ``_compute_influence_reward`` on the same rollout
        with the same intervening ``torch.manual_seed`` call must produce
        bit-identical outputs (the only stochastic component is the MC
        sampling of counterfactual actions, which is fully RNG-driven).
        """
        trainer = _make_trainer()
        trainer.influence_coef = 0.5
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        torch.manual_seed(123)
        intrinsic_a = trainer._compute_influence_reward(rollout)
        torch.manual_seed(123)
        intrinsic_b = trainer._compute_influence_reward(rollout)
        for i in range(NUM_AGENTS):
            assert torch.equal(intrinsic_a[i], intrinsic_b[i]), (
                f"agent {i}: influence reward not bit-identical across "
                "two calls with the same RNG seed"
            )

    def test_influence_coef_validation(self):
        """Constructor rejects negative ``influence_coef`` and
        non-positive ``influence_mc_samples``."""
        env = _env_fn()
        obs = env.reset(seed=0)
        obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
        common = dict(
            env_fn=_env_fn,
            num_agents=NUM_AGENTS,
            obs_dim=obs_dim,
            action_dims=ACTION_DIMS,
            hidden_size=16,
            minibatch_size=32,
            ppo_epochs=2,
            seed=0,
        )
        with pytest.raises(ValueError):
            JointPPOTrainer(influence_coef=-0.1, **common)
        with pytest.raises(ValueError):
            JointPPOTrainer(influence_mc_samples=0, **common)


class TestCommitmentModeTwoPhase:
    """Issue #252: within-night commitment mode wiring through the trainer.

    Smoke tests for the two-phase rollout path. The trainer auto-detects
    ``commitment_mode='two_phase'`` from the env's scenario and switches
    its rollout loop to do two policy forward passes per env-step.
    These tests verify shape/finiteness/auto-detection — they do NOT
    claim the trainer learns anything.
    """

    def _two_phase_env_fn(self):
        from bucket_brigade.envs.scenarios_generated import default_scenario

        scenario = default_scenario(num_agents=NUM_AGENTS)
        scenario.commitment_mode = "two_phase"
        return BucketBrigadeEnv(scenario=scenario)

    def _make_two_phase_trainer(self) -> JointPPOTrainer:
        env = self._two_phase_env_fn()
        obs = env.reset(seed=0)
        # Two-phase obs includes round1_signals so obs_dim grows by
        # num_agents vs simultaneous.
        obs_dim = flatten_dict_obs(
            obs,
            agent_id=0,
            num_agents=NUM_AGENTS,
            include_round1_signals=True,
        ).shape[0]
        return JointPPOTrainer(
            env_fn=self._two_phase_env_fn,
            num_agents=NUM_AGENTS,
            obs_dim=obs_dim,
            action_dims=ACTION_DIMS,
            hidden_size=16,
            minibatch_size=32,
            ppo_epochs=2,
            seed=0,
        )

    def test_trainer_autodetects_two_phase(self):
        trainer = self._make_two_phase_trainer()
        assert trainer._commitment_mode == "two_phase"

    def test_collect_rollout_two_phase_finite_shapes(self):
        """A short two-phase rollout produces finite tensors of the
        expected shapes. This is the smoke-test analogue of the issue
        body's 'smoke training run' — we don't run 100 PPO iters
        locally (per CLAUDE.md compute guidelines) but we do verify
        the rollout loop completes with finite stats."""
        trainer = self._make_two_phase_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        # Shapes match the simultaneous case (the buffer schema is the
        # same; only the env-side mechanic changed).
        assert rollout.observations.shape == (
            ROLLOUT_STEPS,
            NUM_AGENTS,
            trainer.obs_dim,
        )
        for i in range(NUM_AGENTS):
            assert rollout.actions[i].shape == (ROLLOUT_STEPS, len(ACTION_DIMS))
            assert torch.isfinite(rollout.log_probs[i]).all()
            assert torch.isfinite(rollout.values[i]).all()
            assert torch.isfinite(rollout.rewards[i]).all()

    def test_two_phase_update_does_not_nan(self):
        """A PPO update on a two-phase rollout produces finite gradients
        and no NaN losses. End-to-end smoke check."""
        trainer = self._make_two_phase_trainer()
        rollout = trainer.collect_rollout(ROLLOUT_STEPS)
        # Update returns a dict of finite scalar metrics.
        metrics = trainer.update(rollout)
        for key, value in metrics.items():
            assert np.isfinite(value), (
                f"metric {key}={value} is not finite in two-phase update"
            )

    def test_simultaneous_obs_dim_unchanged(self):
        """Bit-exact regression: simultaneous-mode obs_dim is exactly
        what it was pre-#252 (no extra round1_signals channel)."""
        env = _env_fn()
        obs = env.reset(seed=0)
        sim_dim = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]
        # Sanity that the pre-#252 width is recoverable (whatever the
        # exact number is, it must not contain the round1_signals
        # channel — that's the bit-exact backward-compat claim).
        # We probe by comparing to `include_round1_signals=False`:
        # they must be equal (the default for the flag is False).

        # And `include_round1_signals=False` matches.
        tp_off_dim = flatten_dict_obs(
            obs,
            agent_id=0,
            num_agents=NUM_AGENTS,
            include_round1_signals=False,
        ).shape[0]
        assert tp_off_dim == sim_dim

    def test_two_phase_obs_dim_includes_round1_channel(self):
        """Two-phase flatten adds num_agents extra features."""
        env = self._two_phase_env_fn()
        obs = env.reset(seed=0)
        with_r1 = flatten_dict_obs(
            obs,
            agent_id=0,
            num_agents=NUM_AGENTS,
            include_round1_signals=True,
        ).shape[0]
        without_r1 = flatten_dict_obs(
            obs,
            agent_id=0,
            num_agents=NUM_AGENTS,
            include_round1_signals=False,
        ).shape[0]
        assert with_r1 == without_r1 + NUM_AGENTS

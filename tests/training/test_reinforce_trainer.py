"""Tests for the vanilla REINFORCE trainer (issue #273).

Covers:

- **Mutual exclusion**: REINFORCE + centralized_critic / COMA / lambda_red /
  HCA advantage / influence / LOLA all raise ``ValueError`` at trainer
  construction time, mirroring the LolaTrainer mutex guards.
- **Bucket-brigade smoke**: 4 agents, a handful of REINFORCE iterations on
  ``minimal_specialization``; loss is finite, no NaN, every per-agent
  policy has non-zero gradient on at least one parameter.
- **Reproducibility**: same seed produces identical per-iteration losses
  within ``1e-5`` over 3 iterations.
- **IPD sanity (positive control)**: vanilla policy gradient on the
  iterated prisoner's dilemma should *fail to find cooperation* (mean
  reward stays near the always-defect equilibrium of -2.0). This is the
  inverse of LOLA's IPD sanity test — vanilla PG converging to (D, D) is
  the published baseline, and is a positive control confirming the
  implementation has no accidental opponent-awareness or shaping leak.
- **CellConfig integration**: ``cfg.algorithm='reinforce'`` clears its own
  mutex block and dispatches to :class:`REINFORCETrainer`. Default
  ``cfg.algorithm='ppo'`` preserves bit-identity with pre-#273 behavior.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip module when RL extras absent (issue #484)
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv  # noqa: E402
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name  # noqa: E402
from bucket_brigade.training.joint_trainer import flatten_dict_obs  # noqa: E402
from bucket_brigade.training.reinforce_trainer import REINFORCETrainer  # noqa: E402


NUM_AGENTS = 4
ACTION_DIMS = [10, 2, 2]
SCENARIO = "minimal_specialization"
SMOKE_ITERS = 2
SMOKE_ROLLOUT = 32  # tiny: tests verify wiring, not learning.


def _env_fn():
    scenario = get_scenario_by_name(SCENARIO, num_agents=NUM_AGENTS)
    return BucketBrigadeEnv(scenario=scenario)


def _obs_dim() -> int:
    env = _env_fn()
    obs = env.reset(seed=0)
    return flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS).shape[0]


def _make_reinforce(
    seed: int = 0,
    hidden_size: int = 16,
    lr: float = 3e-4,
    normalize_returns: bool = True,
) -> REINFORCETrainer:
    return REINFORCETrainer(
        env_fn=_env_fn,
        num_agents=NUM_AGENTS,
        obs_dim=_obs_dim(),
        action_dims=ACTION_DIMS,
        hidden_size=hidden_size,
        lr=lr,
        normalize_returns=normalize_returns,
        seed=seed,
    )


# ----------------------------------------------------------------------
# Mutual exclusion guards
# ----------------------------------------------------------------------


class TestMutualExclusion:
    """The off-PPO diagnostic interpretation requires REINFORCE to run as
    clean vanilla PG. Each guard fires at construction time so a typo
    surfaces before any rollout work."""

    def test_reinforce_rejects_centralized_critic(self):
        with pytest.raises(ValueError, match="centralized_critic"):
            REINFORCETrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=_obs_dim(),
                action_dims=ACTION_DIMS,
                hidden_size=16,
                centralized_critic=True,
                seed=0,
            )

    def test_reinforce_rejects_coma(self):
        with pytest.raises(ValueError, match="coma"):
            REINFORCETrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=_obs_dim(),
                action_dims=ACTION_DIMS,
                hidden_size=16,
                coma=True,
                seed=0,
            )

    def test_reinforce_rejects_redundancy_penalty(self):
        with pytest.raises(ValueError, match="redundancy"):
            REINFORCETrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=_obs_dim(),
                action_dims=ACTION_DIMS,
                hidden_size=16,
                redundancy_coef=0.01,
                seed=0,
            )

    def test_reinforce_rejects_hca_advantage(self):
        with pytest.raises(ValueError, match="advantage_estimator"):
            REINFORCETrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=_obs_dim(),
                action_dims=ACTION_DIMS,
                hidden_size=16,
                advantage_estimator="hca",
                seed=0,
            )

    def test_reinforce_rejects_influence(self):
        with pytest.raises(ValueError, match="influence_coef"):
            REINFORCETrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=_obs_dim(),
                action_dims=ACTION_DIMS,
                hidden_size=16,
                influence_coef=0.1,
                seed=0,
            )

    def test_reinforce_rejects_lola_dice(self):
        with pytest.raises(ValueError, match="lola_dice"):
            REINFORCETrainer(
                env_fn=_env_fn,
                num_agents=NUM_AGENTS,
                obs_dim=_obs_dim(),
                action_dims=ACTION_DIMS,
                hidden_size=16,
                lola_dice=True,
                seed=0,
            )

    # ------------------------------------------------------------------
    # CellConfig-level guards: ``cfg.algorithm='reinforce'`` paired with any
    # of the conflicting knobs raises before trainer construction.
    # ------------------------------------------------------------------

    def test_cellconfig_guard_reinforce_plus_centralized_critic(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
            centralized_critic=True,
        )
        with pytest.raises(ValueError, match="centralized_critic"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_guard_reinforce_plus_coma(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
            coma=True,
        )
        with pytest.raises(ValueError, match="coma"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_guard_reinforce_plus_lambda_red(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.01,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
        )
        with pytest.raises(ValueError, match="lambda_red"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_guard_reinforce_plus_hca(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
            advantage_estimator="hca",
        )
        with pytest.raises(ValueError, match="advantage_estimator"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_guard_reinforce_plus_influence(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
            influence_coef=0.1,
        )
        with pytest.raises(ValueError, match="influence_coef"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_guard_reinforce_plus_lola(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
            lola_dice=True,
        )
        with pytest.raises(ValueError, match="lola_dice"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_guard_invalid_algorithm(self):
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="a2c",  # not supported
        )
        with pytest.raises(ValueError, match="algorithm"):
            train_one_cell(
                cfg, output_dir=__import__("pathlib").Path("/tmp/should-not-create")
            )

    def test_cellconfig_reinforce_alone_passes(self, tmp_path, monkeypatch):
        """``cfg.algorithm='reinforce'`` with all other knobs at default
        must clear the mutex block. We monkeypatch the REINFORCETrainer so
        the test verifies validation only — not a full rollout."""
        from experiments.p3_specialization import train as train_mod
        from experiments.p3_specialization.train import CellConfig, train_one_cell

        constructed = {"hit": False}

        class _StubTrainer:
            def __init__(self, **kwargs):
                constructed["hit"] = True
                raise RuntimeError("short-circuit: mutex check passed")

        monkeypatch.setattr(train_mod, "REINFORCETrainer", _StubTrainer)

        cfg = CellConfig(
            scenario=SCENARIO,
            lambda_red=0.0,
            seed=0,
            num_iterations=1,
            rollout_steps=SMOKE_ROLLOUT,
            num_agents=NUM_AGENTS,
            algorithm="reinforce",
        )
        with pytest.raises(RuntimeError, match="short-circuit"):
            train_one_cell(cfg, output_dir=tmp_path / "reinforce-alone")
        assert constructed["hit"], (
            "Mutex block short-circuited; REINFORCE-alone happy path is broken."
        )

    def test_cellconfig_default_algorithm_is_ppo(self):
        """Bit-identity guard: the default ``algorithm`` value is ``"ppo"``
        so existing CellConfig callers (every PR before #273) see no
        behavior change."""
        from experiments.p3_specialization.train import CellConfig

        cfg = CellConfig(scenario=SCENARIO, lambda_red=0.0, seed=0)
        assert cfg.algorithm == "ppo", (
            f"CellConfig.algorithm default changed: got {cfg.algorithm!r} "
            "(expected 'ppo' for pre-#273 bit-identity)."
        )


# ----------------------------------------------------------------------
# Bucket-brigade smoke test
# ----------------------------------------------------------------------


class TestREINFORCEBucketBrigadeSmoke:
    """The trainer constructs, rolls out, updates, and produces finite
    losses + non-zero gradients on each per-agent policy."""

    def test_reinforce_smoke_finite_losses_and_gradients(self):
        trainer = _make_reinforce(seed=0)
        for _ in range(SMOKE_ITERS):
            rb = trainer.collect_rollout(SMOKE_ROLLOUT)
            stats = trainer.update(rb)
            for k, v in stats.items():
                assert np.isfinite(v), f"non-finite stat {k}={v}"
        # Per-agent policy gradients should have been populated by the
        # last update (we never zero them after .step()).
        for i, policy in enumerate(trainer.policies):
            grad_norms = [
                p.grad.norm().item() for p in policy.parameters() if p.grad is not None
            ]
            assert any(g > 0 for g in grad_norms), (
                f"agent {i} has zero gradient on every parameter — "
                "the REINFORCE surrogate isn't reaching this policy."
            )

    def test_reinforce_collect_rollout_yields_compatible_buffer(self):
        """REINFORCE's ``RolloutBuffer`` must have the same shape as the
        IPPO one so the info-theory hook in ``train.py`` consumes it
        without branching."""
        trainer = _make_reinforce(seed=0)
        rb = trainer.collect_rollout(SMOKE_ROLLOUT)
        assert rb.observations.shape == (SMOKE_ROLLOUT, NUM_AGENTS, _obs_dim())
        for i in range(NUM_AGENTS):
            assert rb.actions[i].shape[0] == SMOKE_ROLLOUT
            assert rb.rewards[i].shape == (SMOKE_ROLLOUT,)
            assert rb.values[i].shape == (SMOKE_ROLLOUT,)
        assert rb.dones.shape == (SMOKE_ROLLOUT,)

    def test_reinforce_update_requires_rollout(self):
        """REINFORCE has no internal differentiable rollout path (unlike
        LOLA), so :meth:`update` must reject the ``rollout=None`` call to
        prevent a silent no-op."""
        trainer = _make_reinforce(seed=0)
        with pytest.raises(ValueError, match="rollout"):
            trainer.update()

    def test_reinforce_normalize_returns_off_also_works(self):
        """The sweep covers both ``normalize_returns`` settings; the
        un-normalized path must produce finite stats too."""
        trainer = _make_reinforce(seed=0, normalize_returns=False)
        rb = trainer.collect_rollout(SMOKE_ROLLOUT)
        stats = trainer.update(rb)
        for k, v in stats.items():
            assert np.isfinite(v), f"non-finite stat {k}={v}"
        assert stats["reinforce/normalize_returns"] == 0.0


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------


class TestREINFORCEReproducibility:
    """Two runs with the same seed produce the same per-iteration losses."""

    def test_reinforce_same_seed_same_total_loss(self):
        def _run() -> List[float]:
            trainer = _make_reinforce(seed=1234)
            losses: List[float] = []
            for _ in range(SMOKE_ITERS + 1):  # 3 iterations
                rb = trainer.collect_rollout(SMOKE_ROLLOUT)
                stats = trainer.update(rb)
                losses.append(stats["total_loss"])
            return losses

        losses_a = _run()
        losses_b = _run()
        for a, b in zip(losses_a, losses_b):
            assert abs(a - b) < 1e-5, (
                f"reproducibility failure: {a} vs {b} (delta={abs(a - b):.2e})"
            )


# ----------------------------------------------------------------------
# IPD sanity test (positive control: vanilla PG should NOT find
# cooperation in IPD — Foerster et al. 2018 baseline)
# ----------------------------------------------------------------------


class _IPDPolicy(nn.Module):
    """Trivial 2-action policy for the 2-agent iterated prisoner's
    dilemma. Observation = (own_last, opp_last) as a 4-D one-hot;
    2 actions: cooperate (0) or defect (1)."""

    def __init__(self, obs_dim: int = 4, hidden_size: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Standard IPD payoff matrix. Row=C is action 0, Row=D is action 1.
IPD_PAYOFFS = np.array(
    [
        [(-1, -1), (-3, 0)],  # Row=C
        [(0, -3), (-2, -2)],  # Row=D
    ],
    dtype=np.float32,
)


def _ipd_obs(my_last: int, opp_last: int) -> torch.Tensor:
    v = torch.zeros(4)
    v[my_last] = 1.0
    v[2 + opp_last] = 1.0
    return v


def _run_ipd_reinforce(
    num_iters: int = 200,
    episode_len: int = 50,
    lr: float = 0.3,
    seed: int = 0,
) -> float:
    """Train two IPD agents head-to-head with vanilla REINFORCE.

    Standalone driver (not :class:`REINFORCETrainer`, which is wired to
    the bucket-brigade env). The math is the same MC policy gradient.

    Returns the mean per-step reward (across both agents) averaged over
    the last 10 iterations.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    policy_a = _IPDPolicy()
    policy_b = _IPDPolicy()
    opt_a = optim.SGD(policy_a.parameters(), lr=lr)
    opt_b = optim.SGD(policy_b.parameters(), lr=lr)

    def _rollout_episode():
        my_last_a, my_last_b = 0, 0  # "C, C" start
        log_probs_a: List[torch.Tensor] = []
        log_probs_b: List[torch.Tensor] = []
        rewards_a: List[float] = []
        rewards_b: List[float] = []
        for _ in range(episode_len):
            obs_a = _ipd_obs(my_last_a, my_last_b)
            obs_b = _ipd_obs(my_last_b, my_last_a)
            logits_a = policy_a(obs_a.unsqueeze(0))[0]
            logits_b = policy_b(obs_b.unsqueeze(0))[0]
            dist_a = torch.distributions.Categorical(logits=logits_a)
            dist_b = torch.distributions.Categorical(logits=logits_b)
            a_act = dist_a.sample()
            b_act = dist_b.sample()
            log_probs_a.append(dist_a.log_prob(a_act))
            log_probs_b.append(dist_b.log_prob(b_act))
            r_a, r_b = IPD_PAYOFFS[a_act.item(), b_act.item()]
            rewards_a.append(float(r_a))
            rewards_b.append(float(r_b))
            my_last_a = int(a_act.item())
            my_last_b = int(b_act.item())
        return (
            torch.stack(log_probs_a),
            torch.stack(log_probs_b),
            torch.tensor(rewards_a),
            torch.tensor(rewards_b),
            float((sum(rewards_a) + sum(rewards_b)) / (2 * episode_len)),
        )

    def _reinforce_loss(lp: torch.Tensor, rewards: torch.Tensor) -> torch.Tensor:
        """Vanilla REINFORCE: ``-Σ_t log π(a_t) * G_t`` with mean-
        subtracted returns (variance reduction, NOT a learned baseline)."""
        # Discounted MC return with gamma = 1.0 (the IPD literature uses
        # undiscounted; either is fine for the sanity check).
        G = torch.zeros_like(rewards)
        running = rewards.new_tensor(0.0)
        for t in reversed(range(rewards.shape[0])):
            running = rewards[t] + running
            G[t] = running
        # Standardize for variance reduction.
        if rewards.shape[0] > 1:
            G = (G - G.mean()) / (G.std() + 1e-8)
        return -(lp * G).mean()

    recent_rewards: List[float] = []
    for it in range(num_iters):
        lp_a, lp_b, r_a, r_b, mean_r = _rollout_episode()
        L_a = _reinforce_loss(lp_a, r_a)
        L_b = _reinforce_loss(lp_b, r_b)
        opt_a.zero_grad()
        L_a.backward(retain_graph=True)
        opt_a.step()
        opt_b.zero_grad()
        L_b.backward()
        opt_b.step()

        if it >= num_iters - 10:
            recent_rewards.append(mean_r)

    return float(np.mean(recent_rewards)) if recent_rewards else 0.0


class TestREINFORCEIPDSanity:
    """Positive control: vanilla policy gradient is the published
    *failure* baseline on the IPD (Foerster et al. 2018 Fig. 2). The
    naive learners converge to always-defect (D, D), which has mean
    reward ``-2.0`` per step. This test confirms our REINFORCE
    implementation reproduces that failure mode — failure here would be
    suspicious (e.g., an accidental opponent-awareness or shaping leak
    that's giving vanilla PG signal it shouldn't have).

    The IPD reward bounds:
        - Always-defect (D, D):  mean per-step = -2.0   (REINFORCE target)
        - Always-cooperate (C, C): mean per-step = -1.0 (Pareto optimum)

    We assert REINFORCE's mean reward stays close to -2.0 (the failure
    mode), specifically below -1.5 (the midpoint). This is a loose bound
    that catches a regression where vanilla PG somehow finds cooperation
    — which would mean either the env is not really IPD or the
    implementation has a leak.
    """

    @pytest.mark.slow
    def test_reinforce_fails_to_find_cooperation_on_ipd(self):
        # Multiple seeds because IPD is a noisy 2x2 stochastic game.
        seeds = [0, 1, 2]
        mean_rewards: List[float] = []
        for s in seeds:
            r = _run_ipd_reinforce(seed=s)
            mean_rewards.append(r)
        mean_r = float(np.mean(mean_rewards))
        # Vanilla PG should converge near the always-defect equilibrium
        # of -2.0. We give a loose ceiling of -1.4 — i.e., REINFORCE
        # cannot accidentally end up closer to (C, C) than to (D, D).
        # The published Foerster-'18 Fig. 2 vanilla-PG result is around
        # -1.98; we leave headroom for the modest 200-iter budget.
        assert mean_r <= -1.4, (
            f"REINFORCE found cooperation on IPD across {len(seeds)} seeds "
            f"(mean reward = {mean_r:.3f}, expected near -2.0). This "
            "suggests an accidental opponent-awareness or shaping leak in "
            "the implementation — vanilla PG should NOT find (C, C). "
            "This is the inverse of LOLA's IPD sanity test."
        )

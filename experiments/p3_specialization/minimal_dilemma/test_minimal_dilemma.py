"""Tests for the minimal-dilemma toy reproduction (issue #292).

Eight test classes covering the 8 testable items from the curator spec:

1. Env step contract: action shape, reward sign, episode length.
2. Env payoff matrix matches the analytic table (Option A, m=1.6).
3. Specialist policies (always_cooperate, always_defect, tit_for_tat) are
   correct in isolation.
4. ``flatten_dict_obs`` integration: the env dict shape produces a 7-d
   per-agent obs (5-d base + 2 identity-tail), as required by the curator
   spec.
5. 7-arm baseline harness: rolling each arm against itself reproduces the
   expected per-step reward within reasonable variance.
6. K=20 → K=200 stability gate logic (synthetic data; passes when phase-2
   is stable, fails when phase-2 drifts upward).
7. IPPO smoke run: 1-2 iters complete without exception and emit a
   well-formed metrics.json.
8. Determinism: two trainings with identical seed produce identical
   final actor weights.

These tests are local-only and run in <2 min. Full sweeps go on
COMPUTE_HOST_PRIMARY per CLAUDE.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Repo-root sys.path shim — pytest is invoked from inside the worktree but
# the experiments package isn't always on sys.path by default in a uv venv.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_brigade.training.joint_trainer import flatten_dict_obs  # noqa: E402

from experiments.p3_specialization.minimal_dilemma.env import (  # noqa: E402
    ACTION_COOPERATE,
    ACTION_DEFECT,
    EPISODE_LENGTH,
    MinimalDilemmaEnv,
    NUM_AGENTS,
    REWARD_MUTUAL_COOPERATE,
    REWARD_MUTUAL_DEFECT,
    REWARD_UNILATERAL_COOPERATE,
    REWARD_UNILATERAL_DEFECT,
    step_reward,
)
from experiments.p3_specialization.minimal_dilemma.specialists import (  # noqa: E402
    always_cooperate,
    always_defect,
    always_random,
    tit_for_tat,
)
from experiments.p3_specialization.minimal_dilemma.best_of_n import (  # noqa: E402
    stability_gate,
)
from experiments.p3_specialization.minimal_dilemma.train_ippo import (  # noqa: E402
    TrainConfig,
    train_one_cell,
)
from experiments.p3_specialization.minimal_dilemma.verdict import (  # noqa: E402
    classify_verdict,
    compute_verdict,
    gate1_ippo_defects,
    gate2_specialist_dominates,
    gate3_bc_holds_basin,
    gate4_bridge_fails,
)


# ----------------------------------------------------------------------------
# 1) Env step contract.
# ----------------------------------------------------------------------------
class TestEnvStepContract:
    def test_reset_returns_dict_with_required_keys(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        # ``flatten_dict_obs`` reads these five keys; missing any would
        # silently break the JointPPOTrainer integration.
        for key in ("houses", "signals", "locations", "last_actions", "scenario_info"):
            assert key in obs, f"missing obs key: {key}"
        assert obs["last_actions"].shape == (NUM_AGENTS, 2)
        assert obs["scenario_info"].shape == (1,)
        assert env.done is False

    def test_step_action_shape_validation(self):
        env = MinimalDilemmaEnv()
        env.reset(seed=0)
        with pytest.raises(ValueError, match="shape"):
            env.step(np.array([[0]], dtype=np.int64))  # wrong shape

    def test_step_action_value_validation(self):
        env = MinimalDilemmaEnv()
        env.reset(seed=0)
        with pytest.raises(ValueError, match="actions must be"):
            env.step(np.array([[0], [3]], dtype=np.int64))

    def test_step_rewards_shape_and_dtype(self):
        env = MinimalDilemmaEnv()
        env.reset(seed=0)
        joint = np.array([[1], [1]], dtype=np.int64)
        obs, rewards, dones, info = env.step(joint)
        assert rewards.shape == (NUM_AGENTS,)
        assert rewards.dtype == np.float32
        assert dones.shape == (NUM_AGENTS,)
        assert dones.dtype == np.bool_
        assert info == {}

    def test_episode_length_exact(self):
        """Episode must terminate at exactly step EPISODE_LENGTH (50)."""
        env = MinimalDilemmaEnv()
        env.reset(seed=0)
        joint = np.array([[0], [0]], dtype=np.int64)
        for step in range(EPISODE_LENGTH - 1):
            _, _, dones, _ = env.step(joint)
            assert not dones.any(), f"episode ended early at step {step}"
        _, _, dones, _ = env.step(joint)
        assert dones.all(), "episode did not terminate at EPISODE_LENGTH"
        assert env.done is True

    def test_step_after_done_raises(self):
        env = MinimalDilemmaEnv(episode_length=2)
        env.reset(seed=0)
        joint = np.array([[0], [0]], dtype=np.int64)
        env.step(joint)
        env.step(joint)
        assert env.done
        with pytest.raises(RuntimeError, match="terminated"):
            env.step(joint)


# ----------------------------------------------------------------------------
# 2) Payoff matrix matches the analytic table.
# ----------------------------------------------------------------------------
class TestPayoffMatrix:
    def test_pure_function_payoff_matrix(self):
        assert step_reward(0, 0) == pytest.approx(
            (REWARD_MUTUAL_DEFECT, REWARD_MUTUAL_DEFECT)
        )
        assert step_reward(1, 1) == pytest.approx(
            (REWARD_MUTUAL_COOPERATE, REWARD_MUTUAL_COOPERATE)
        )
        # (C, D) → agent 0 = cooperator (loses), agent 1 = defector (gains).
        assert step_reward(1, 0) == pytest.approx(
            (REWARD_UNILATERAL_COOPERATE, REWARD_UNILATERAL_DEFECT)
        )
        assert step_reward(0, 1) == pytest.approx(
            (REWARD_UNILATERAL_DEFECT, REWARD_UNILATERAL_COOPERATE)
        )

    def test_dilemma_invariants(self):
        """Public-goods dilemma requires 1 < m < 2; check each inequality."""
        # Mutual-C strictly beats mutual-D.
        assert REWARD_MUTUAL_COOPERATE > REWARD_MUTUAL_DEFECT
        # Unilateral-D strictly beats mutual-C for the defector (incentive
        # to defect at every step → dominant-strategy equilibrium).
        assert REWARD_UNILATERAL_DEFECT > REWARD_MUTUAL_COOPERATE
        # Mutual-D strictly beats unilateral-C for the cooperator
        # (incentive to defect even when partner cooperates).
        assert REWARD_MUTUAL_DEFECT > REWARD_UNILATERAL_COOPERATE

    def test_total_reward_over_10_steps(self):
        """10 steps of mutual-C → sum = 6.0; 10 steps of mutual-D → 0.0 (curator spec)."""
        env = MinimalDilemmaEnv(episode_length=10)
        env.reset(seed=0)
        total_a0 = 0.0
        total_a1 = 0.0
        joint = np.array([[1], [1]], dtype=np.int64)
        for _ in range(10):
            _, rewards, _, _ = env.step(joint)
            total_a0 += float(rewards[0])
            total_a1 += float(rewards[1])
        assert total_a0 == pytest.approx(6.0)
        assert total_a1 == pytest.approx(6.0)

        env = MinimalDilemmaEnv(episode_length=10)
        env.reset(seed=0)
        total_a0 = 0.0
        joint = np.array([[0], [0]], dtype=np.int64)
        for _ in range(10):
            _, rewards, _, _ = env.step(joint)
            total_a0 += float(rewards[0])
        assert total_a0 == pytest.approx(0.0)


# ----------------------------------------------------------------------------
# 3) Specialists are correct.
# ----------------------------------------------------------------------------
class TestSpecialists:
    def test_always_cooperate_emits_C_for_any_obs(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        joint = always_cooperate(obs)
        assert joint.shape == (NUM_AGENTS, 1)
        assert joint.dtype == np.int64
        assert (joint == ACTION_COOPERATE).all()
        # Mid-episode obs — also all-C.
        for _ in range(5):
            obs, _, _, _ = env.step(joint)
            joint = always_cooperate(obs)
            assert (joint == ACTION_COOPERATE).all()

    def test_always_defect_emits_D_for_any_obs(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        joint = always_defect(obs)
        assert (joint == ACTION_DEFECT).all()

    def test_tit_for_tat_starts_cooperate(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        joint = tit_for_tat(obs)
        # Step 0: no prior actions → both agents cooperate.
        assert (joint == ACTION_COOPERATE).all()

    def test_tit_for_tat_copies_opponent(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        # Step 0: TFT cooperates (smoke check; full assertion is in
        # test_tit_for_tat_starts_cooperate above).
        _ = tit_for_tat(obs)
        # Inject an asymmetric move so the previous-action obs is interesting.
        injected = np.array([[ACTION_DEFECT], [ACTION_COOPERATE]], dtype=np.int64)
        obs, _, _, _ = env.step(injected)
        joint1 = tit_for_tat(obs)
        # Agent 0 should now copy opponent's previous action = ACTION_COOPERATE.
        # Agent 1 should copy opponent's previous action = ACTION_DEFECT.
        assert int(joint1[0, 0]) == ACTION_COOPERATE
        assert int(joint1[1, 0]) == ACTION_DEFECT

    def test_always_random_distribution(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        rng = np.random.default_rng(42)
        joints = np.stack([always_random(obs, rng=rng) for _ in range(2000)])
        coop_frac = float((joints == ACTION_COOPERATE).mean())
        # Loose bounds: with 4000 samples (2 agents × 2000 steps) the
        # cooperation rate should be near 0.5 — well inside [0.45, 0.55].
        assert 0.45 < coop_frac < 0.55


# ----------------------------------------------------------------------------
# 4) flatten_dict_obs integration.
# ----------------------------------------------------------------------------
class TestObsFlatteningIntegration:
    def test_obs_dim_is_7_with_identity_tail(self):
        """5-d base (4-d last_actions + 1-d step) + 2-d identity = 7-d."""
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        flat = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS)
        assert flat.shape == (7,)
        assert flat.dtype == np.float32

    def test_per_agent_flatten_differs_only_in_identity_tail(self):
        env = MinimalDilemmaEnv()
        obs = env.reset(seed=0)
        flat0 = flatten_dict_obs(obs, agent_id=0, num_agents=NUM_AGENTS)
        flat1 = flatten_dict_obs(obs, agent_id=1, num_agents=NUM_AGENTS)
        # First 5 entries (the shared global obs) match exactly.
        np.testing.assert_array_equal(flat0[:5], flat1[:5])
        # Last 2 entries (identity one-hot) differ.
        np.testing.assert_array_equal(flat0[5:], np.array([1.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(flat1[5:], np.array([0.0, 1.0], dtype=np.float32))


# ----------------------------------------------------------------------------
# 5) 7-arm baseline harness expected per-step rewards.
# ----------------------------------------------------------------------------
def _self_play_per_agent_reward(
    arm_a, arm_b, num_episodes=10, episode_length=50, seed=0
) -> float:
    """Roll two policies against each other and return mean per-step per-agent reward.

    Each ``arm`` is a callable ``policy(obs, num_agents=2) -> [2, 1]`` joint
    action. We pick row 0 of arm_a's output for agent 0 and row 1 of arm_b's
    output for agent 1 so each arm controls only its own agent.
    """
    env = MinimalDilemmaEnv(episode_length=episode_length)
    totals: list[float] = []
    rng = np.random.default_rng(seed)
    for ep in range(num_episodes):
        obs = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        ep_total = 0.0
        while not env.done:
            ja = arm_a(obs)
            jb = arm_b(obs)
            joint = np.array([[int(ja[0, 0])], [int(jb[1, 0])]], dtype=np.int64)
            obs, rewards, _, _ = env.step(joint)
            ep_total += float(rewards.sum())
        totals.append(ep_total / (episode_length * NUM_AGENTS))
    return float(np.mean(totals))


class TestSevenArmBaselines:
    """Curator's expected per-step per-agent rewards (Option A, m=1.6).

    Arm                                   | Expected
    --------------------------------------+----------
    Mutual defect (AllD vs AllD)          | 0.0
    Random vs Random                      | 0.3
    AllC vs AllC                          | 0.6
    AllC vs AllD                          | 0.3 (= -0.2/2 + 0.8/2)
    TFT vs TFT                            | 0.6 (mutual cooperation forever)
    TFT vs AllD                           | very small (TFT defects step 1+)
    AllD vs Random                        | ~0.4 (D gets 0.5*0.8=0.4)

    "Random_seeded" uses a fixed RNG so the test is deterministic.
    """

    def test_alld_self_play_is_zero(self):
        r = _self_play_per_agent_reward(always_defect, always_defect)
        assert r == pytest.approx(0.0, abs=1e-6)

    def test_allc_self_play_is_cooperative(self):
        r = _self_play_per_agent_reward(always_cooperate, always_cooperate)
        assert r == pytest.approx(REWARD_MUTUAL_COOPERATE, abs=1e-6)

    def test_tft_self_play_is_cooperative(self):
        r = _self_play_per_agent_reward(tit_for_tat, tit_for_tat)
        assert r == pytest.approx(REWARD_MUTUAL_COOPERATE, abs=1e-6)

    def test_allc_vs_alld_midpoint(self):
        r = _self_play_per_agent_reward(always_cooperate, always_defect)
        # (-0.2 + 0.8) / 2 = 0.3
        assert r == pytest.approx(0.3, abs=1e-6)

    def test_random_self_play_near_quarter_point(self):
        # E[r] under uniform random = 0.3 (= m/4); finite-sample tolerance 0.1.
        def rand_fn(obs, num_agents=NUM_AGENTS, _rng=np.random.default_rng(0)):
            return always_random(obs, num_agents=num_agents, rng=_rng)

        r = _self_play_per_agent_reward(rand_fn, rand_fn, num_episodes=200)
        assert 0.20 < r < 0.40


# ----------------------------------------------------------------------------
# 6) K=20 → K=200 stability gate logic on synthetic data.
# ----------------------------------------------------------------------------
class TestStabilityGate:
    def test_stable_top_passes_gate(self):
        # Phase 1 and phase 2 means agree → bridge "failed" (basin-trap consistent).
        top_phase1 = [
            {"mean_per_step": 0.05},
            {"mean_per_step": 0.04},
            {"mean_per_step": 0.03},
        ]
        top_phase2 = [
            {"mean_per_step": 0.05},
            {"mean_per_step": 0.04},
            {"mean_per_step": 0.03},
        ]
        gate = stability_gate(top_phase1, top_phase2, slack=0.05)
        assert gate["passed"] is True
        assert gate["drift"] == pytest.approx(0.0, abs=1e-9)

    def test_drifting_upward_fails_gate(self):
        # Phase 2 mean much higher than phase 1 → bridge succeeded → fail.
        # Use a drift larger than the slack to ensure the gate flags it.
        top_phase1 = [
            {"mean_per_step": 0.10},
            {"mean_per_step": 0.05},
        ]
        top_phase2 = [
            {"mean_per_step": 0.50},
            {"mean_per_step": 0.40},
        ]
        gate = stability_gate(top_phase1, top_phase2, slack=0.05)
        assert gate["passed"] is False
        assert gate["drift"] > 0.05

    def test_drifting_downward_passes_gate(self):
        # The #271 failure mode: phase-1 looked good, phase-2 collapsed.
        # Bridge fails (= basin-trap consistent).
        top_phase1 = [{"mean_per_step": 0.45}]
        top_phase2 = [{"mean_per_step": 0.05}]
        gate = stability_gate(top_phase1, top_phase2, slack=0.05)
        assert gate["passed"] is True
        assert gate["drift"] < 0.0


# ----------------------------------------------------------------------------
# 7) IPPO smoke run.
# ----------------------------------------------------------------------------
class TestIPPOSmoke:
    def test_two_iter_training_produces_metrics(self, tmp_path: Path):
        cfg = TrainConfig(
            seed=0,
            num_iterations=2,
            rollout_steps=128,
            hidden_size=32,
            ppo_epochs=2,
            minibatch_size=64,
        )
        metrics = train_one_cell(cfg, tmp_path)
        assert (tmp_path / "metrics.json").exists()
        assert (tmp_path / "config.json").exists()
        assert (tmp_path / "policies" / "agent_0.pt").exists()
        assert (tmp_path / "policies" / "agent_1.pt").exists()
        # Metrics list has one entry per iteration with the expected keys.
        assert len(metrics) == 2
        for record in metrics:
            assert "mean_step_reward_per_agent" in record
            assert "cooperation_fraction" in record
            assert 0.0 <= record["cooperation_fraction"] <= 1.0
        # JSON round-trip works.
        with (tmp_path / "metrics.json").open() as f:
            loaded = json.load(f)
        assert len(loaded) == 2

    def test_mappo_arm_runs(self, tmp_path: Path):
        cfg = TrainConfig(
            seed=0,
            num_iterations=1,
            rollout_steps=128,
            hidden_size=32,
            ppo_epochs=1,
            minibatch_size=64,
            centralized_critic=True,
        )
        train_one_cell(cfg, tmp_path)
        assert (tmp_path / "metrics.json").exists()


# ----------------------------------------------------------------------------
# 8) Determinism at fixed seed.
# ----------------------------------------------------------------------------
class TestDeterminism:
    def test_same_seed_same_final_policy(self, tmp_path: Path):
        cfg_a = TrainConfig(
            seed=7,
            num_iterations=2,
            rollout_steps=128,
            hidden_size=32,
            ppo_epochs=1,
            minibatch_size=64,
        )
        cfg_b = TrainConfig(
            seed=7,
            num_iterations=2,
            rollout_steps=128,
            hidden_size=32,
            ppo_epochs=1,
            minibatch_size=64,
        )
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        train_one_cell(cfg_a, dir_a)
        train_one_cell(cfg_b, dir_b)
        # Compare final state dicts byte-for-byte.
        sd_a0 = torch.load(dir_a / "policies" / "agent_0.pt", weights_only=True)
        sd_b0 = torch.load(dir_b / "policies" / "agent_0.pt", weights_only=True)
        for k in sd_a0:
            torch.testing.assert_close(sd_a0[k], sd_b0[k], rtol=0, atol=0)


# ----------------------------------------------------------------------------
# 9) Verdict classifier — full 4-gate truth table.
# ----------------------------------------------------------------------------
class TestVerdictClassifier:
    """Each named verdict is reachable from a synthetic fixture set."""

    def _make_ippo_summaries(self, reward: float, coop: float, n: int = 5):
        return [
            {"final_per_agent_reward": reward, "final_coop_fraction": coop}
            for _ in range(n)
        ]

    def _passing_bestofn_gate(self):
        return {
            "passed": True,
            "phase1_mean": 0.05,
            "phase2_mean": 0.05,
            "phase2_max": 0.06,
            "drift": 0.0,
            "slack": 0.05,
        }

    def test_basin_trap_replicated(self):
        out = compute_verdict(
            ippo_seed_summaries=self._make_ippo_summaries(reward=0.0, coop=0.02),
            specialist_per_agent_reward=0.6,
            bc_summary={"min_test_accuracy": 0.99},
            ppo_continuation_seed_summaries=self._make_ippo_summaries(
                reward=0.5, coop=0.95, n=3
            ),
            bestofn_stability_gate=self._passing_bestofn_gate(),
        )
        assert out["verdict"] == "basin_trap_replicated"

    def test_anti_attractor_in_toy_when_bc_decays(self):
        # Gate 3 fails (BC reward collapses to defect under PPO continuation).
        out = compute_verdict(
            ippo_seed_summaries=self._make_ippo_summaries(reward=0.0, coop=0.02),
            specialist_per_agent_reward=0.6,
            bc_summary={"min_test_accuracy": 0.99},
            ppo_continuation_seed_summaries=self._make_ippo_summaries(
                reward=0.0, coop=0.02, n=3
            ),
            bestofn_stability_gate=self._passing_bestofn_gate(),
        )
        assert out["verdict"] == "anti_attractor_in_toy"

    def test_null_result_when_ippo_does_not_defect(self):
        # Gate 1 fails (IPPO above defect threshold).
        out = compute_verdict(
            ippo_seed_summaries=self._make_ippo_summaries(reward=0.4, coop=0.7),
            specialist_per_agent_reward=0.6,
            bc_summary={"min_test_accuracy": 0.99},
            ppo_continuation_seed_summaries=self._make_ippo_summaries(
                reward=0.5, coop=0.95, n=3
            ),
            bestofn_stability_gate=self._passing_bestofn_gate(),
        )
        assert out["verdict"] == "null_result"

    def test_unit_gate_helpers(self):
        # Direct unit coverage of each gate helper.
        g1 = gate1_ippo_defects(
            [{"final_per_agent_reward": 0.01, "final_coop_fraction": 0.05}]
        )
        assert g1["passed"]
        g2 = gate2_specialist_dominates(0.6)
        assert g2["passed"]
        g3 = gate3_bc_holds_basin(
            {"min_test_accuracy": 0.99},
            [{"final_per_agent_reward": 0.5, "final_coop_fraction": 0.95}],
        )
        assert g3["passed"]
        g4 = gate4_bridge_fails(
            {"passed": True, "phase2_mean": 0.05}, ippo_mean_reward=0.0
        )
        assert g4["passed"]
        assert classify_verdict(g1, g2, g3, g4) == "basin_trap_replicated"

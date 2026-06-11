"""Unit + smoke tests for the issue #291 single-agent joint-action wrapper.

Covers:

- Wrapper action-space sanity: joint action space equals product of
  per-agent action sizes; joint actions split correctly into per-agent
  actions and reach the underlying env unmodified.
- Reward aggregation: wrapped scalar reward equals the sum of per-agent
  rewards on every step.
- Specialist-ceiling smoke: driving the wrapper with the hand-tuned
  specialist policy beats the random floor by a wide margin (the upper
  bound is representable under the new action space).
- Random-floor smoke: uniform-random joint actions produce reward in a
  known floor range.
- K=200 re-eval stability gate: the analyzer enforces a K=200 gate
  before reporting a verdict; we test the gate's existence (it triggers
  on missing ``k200_gap_closed`` field) without running K=200.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bucket_brigade.baselines import specialist_action_joint
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.envs.single_agent_wrapper import SingleAgentJointWrapper


SCENARIO = "minimal_specialization"
NUM_SUBAGENTS = 4
ACTION_DIM_PER_AGENT = 3
SMOKE_EPISODES = 5  # tiny budget so the unit-test suite stays fast


def _build_wrapper(seed: int = 0) -> SingleAgentJointWrapper:
    scenario = get_scenario_by_name(SCENARIO, num_agents=NUM_SUBAGENTS)
    env = BucketBrigadeEnv(scenario=scenario)
    wrapper = SingleAgentJointWrapper(env)
    wrapper.reset(seed=seed)
    return wrapper


class TestActionSpace:
    """The wrapper's joint action space must equal the product of per-agent sizes."""

    def test_joint_action_size_equals_product(self):
        wrapper = _build_wrapper()
        per_agent = int(np.prod(wrapper.action_dims_per_agent))
        # 10 houses, 2 modes, 2 signals -> 40 per sub-agent.
        assert per_agent == 40
        # 4 sub-agents -> 40 ** 4 = 2_560_000.
        assert wrapper.joint_action_size == 40**NUM_SUBAGENTS

    def test_joint_action_dims_is_concatenation(self):
        wrapper = _build_wrapper()
        # Should be 12 dims = [house, mode, signal] * 4.
        assert len(wrapper.joint_action_dims) == NUM_SUBAGENTS * ACTION_DIM_PER_AGENT
        for i in range(NUM_SUBAGENTS):
            offset = i * ACTION_DIM_PER_AGENT
            assert wrapper.joint_action_dims[offset] == wrapper.action_dims_per_agent[0]
            assert wrapper.joint_action_dims[offset + 1] == 2
            assert wrapper.joint_action_dims[offset + 2] == 2

    def test_action_dims_match_scenario_num_houses(self):
        """For minimal_specialization (10 houses) the per-agent house dim is 10."""
        wrapper = _build_wrapper()
        assert wrapper.action_dims_per_agent[0] == 10


class TestSplitJointAction:
    """The wrapper decodes a flat joint action into the right per-agent slots."""

    def test_split_flat_to_per_agent(self):
        wrapper = _build_wrapper()
        # Construct a deterministic joint action: agent i goes to house i,
        # toggles mode/signal as i & 1.
        flat = []
        for i in range(NUM_SUBAGENTS):
            flat.extend([i, i & 1, i & 1])
        per_agent = wrapper._split_joint_action(np.asarray(flat))
        assert per_agent.shape == (NUM_SUBAGENTS, ACTION_DIM_PER_AGENT)
        for i in range(NUM_SUBAGENTS):
            assert per_agent[i, 0] == i
            assert per_agent[i, 1] == (i & 1)
            assert per_agent[i, 2] == (i & 1)

    def test_split_2d_shape_passes_through(self):
        wrapper = _build_wrapper()
        per_agent_in = np.zeros((NUM_SUBAGENTS, ACTION_DIM_PER_AGENT), dtype=np.int64)
        per_agent_in[0] = [3, 1, 1]
        per_agent_out = wrapper._split_joint_action(per_agent_in)
        np.testing.assert_array_equal(per_agent_in, per_agent_out)

    def test_split_wrong_size_raises(self):
        wrapper = _build_wrapper()
        with pytest.raises(ValueError):
            wrapper._split_joint_action(
                np.zeros(NUM_SUBAGENTS * ACTION_DIM_PER_AGENT + 1)
            )
        with pytest.raises(ValueError):
            wrapper._split_joint_action(
                np.zeros((NUM_SUBAGENTS + 1, ACTION_DIM_PER_AGENT))
            )
        with pytest.raises(ValueError):
            wrapper._split_joint_action(np.zeros((2, 2, 2)))


class TestStepSemantics:
    """Wrapped step must match the underlying env on observation and reward."""

    def test_reward_equals_per_agent_sum(self):
        """The wrapped scalar reward must equal ``sum(per_agent_rewards)``."""
        wrapper = _build_wrapper(seed=123)
        # A bunch of random actions; verify the aggregate every step.
        rng = np.random.default_rng(0)
        for _ in range(20):
            # Build a fresh flat joint action with the right shape.
            flat = []
            for _i in range(NUM_SUBAGENTS):
                flat.extend(
                    [
                        int(rng.integers(wrapper.action_dims_per_agent[0])),
                        int(rng.integers(2)),
                        int(rng.integers(2)),
                    ]
                )
            obs, team_reward, done, info = wrapper.step(np.asarray(flat))
            per_agent = info["per_agent_rewards"]
            assert pytest.approx(float(per_agent.sum()), rel=1e-6) == team_reward
            if done:
                wrapper.reset()

    def test_obs_shape_consistent_across_reset_and_step(self):
        wrapper = _build_wrapper(seed=42)
        stacked0 = wrapper.reset(seed=42)
        assert stacked0.shape == (NUM_SUBAGENTS, wrapper.obs_dim_per_agent)
        # Default neutral action to advance one step.
        zero = np.zeros(NUM_SUBAGENTS * ACTION_DIM_PER_AGENT, dtype=np.int64)
        stacked1, _r, _d, _info = wrapper.step(zero)
        assert stacked1.shape == stacked0.shape


class TestSpecialistCeilingSmoke:
    """Driving the wrapper with the hand-tuned specialist must clear the random floor."""

    def test_specialist_beats_random_floor(self):
        """Specialist should reach per-step team reward >> random floor.

        We do not assert ``gap_closed >= 0.85`` here — that needs the
        full K=200 re-eval pass. The unit-test budget is ``SMOKE_EPISODES``
        episodes, which is enough to show specialist >> random but not
        enough to estimate the converged ceiling. The K=200 evaluation
        lives in the analyzer + run driver.
        """
        wrapper = _build_wrapper(seed=7)
        scenario = wrapper.env.scenario
        num_houses = int(getattr(scenario, "num_houses", 10))
        per_step_team_rewards = []
        wrapper.reset(seed=7)
        episodes_done = 0
        while episodes_done < SMOKE_EPISODES:
            # Specialist needs the raw dict obs, not the flattened stack.
            obs_dict = wrapper.env._get_observation()
            per_agent_actions = specialist_action_joint(
                obs_dict, num_agents=NUM_SUBAGENTS, num_houses=num_houses
            )
            _stacked, team_reward, done, _info = wrapper.step(per_agent_actions)
            per_step_team_rewards.append(team_reward)
            if done:
                wrapper.reset()
                episodes_done += 1
        mean_team = float(np.mean(per_step_team_rewards))
        # Random baseline on minimal_specialization is -87.72 (or -96.07
        # under the pre-#246 frozen analyzers); specialist is -28.38
        # (canonical n=10k value from issue #416; was -22.07 under PR #243
        # n=50). Specialist should clear the random floor by a wide margin
        # even on a tiny sample.
        assert mean_team > -60.0, (
            f"specialist mean per-step team reward = {mean_team:.2f} did "
            "not clear -60 (random floor is -96.07). Either the wrapper "
            "is corrupting the action plumbing or the specialist baseline "
            "regressed."
        )


class TestRandomFloorSmoke:
    """Uniform-random joint actions must produce reward near the documented floor."""

    def test_random_actions_near_floor(self):
        """Random actions should sit near (but not collapse below) the
        analyze_270.py random baseline (-96.07).

        We accept a wide band because ``SMOKE_EPISODES`` is small and
        per-episode variance is high; the test is a sanity floor, not
        a statistical estimate.
        """
        wrapper = _build_wrapper(seed=2026)
        rng = np.random.default_rng(2026)
        per_step_team_rewards = []
        wrapper.reset(seed=2026)
        episodes_done = 0
        while episodes_done < SMOKE_EPISODES:
            flat = []
            for _i in range(NUM_SUBAGENTS):
                flat.extend(
                    [
                        int(rng.integers(wrapper.action_dims_per_agent[0])),
                        int(rng.integers(2)),
                        int(rng.integers(2)),
                    ]
                )
            _s, team_reward, done, _info = wrapper.step(np.asarray(flat))
            per_step_team_rewards.append(team_reward)
            if done:
                wrapper.reset()
                episodes_done += 1
        mean_team = float(np.mean(per_step_team_rewards))
        # Random baseline reference is -96.07. Allow a wide ±60 band on
        # this short smoke sample.
        assert -180.0 < mean_team < -20.0, (
            f"random-action mean per-step team reward = {mean_team:.2f} "
            "outside the smoke-test sanity band (-180, -20). Either the "
            "env semantics shifted or the wrapper is corrupting actions."
        )


class TestK200StabilityGate:
    """The analyzer must refuse to emit a verdict before K=200 re-eval runs.

    This test does NOT run K=200 (that requires the full sweep + a
    separate re-eval pass). It only asserts the gate exists and trips
    when the per-seed metrics lack a ``k200_gap_closed`` field.
    """

    def test_analyzer_blocks_without_k200(self, tmp_path: Path) -> None:
        # Synthesize a phase-1-only metrics.json (no k200_gap_closed) and
        # check the analyzer's gate function rejects it.
        from experiments.p3_specialization.analyze_291 import (
            _enforce_k200_gate,
            aggregate_arm,
        )

        seed_dir = tmp_path / "seed_42"
        seed_dir.mkdir()
        # Two iterations of fake training-time metrics.
        (seed_dir / "metrics.json").write_text(
            json.dumps(
                [
                    {"iteration": 0, "mean_step_reward_team": -90.0},
                    {"iteration": 1, "mean_step_reward_team": -85.0},
                ]
            )
        )
        arm = aggregate_arm([seed_dir], "joint_control_ppo")
        # Default mode (gate enforced): should produce a blocking message.
        failure = _enforce_k200_gate(arm, allow_phase1_only=False)
        assert failure is not None
        assert "K=200" in failure

    def test_analyzer_bypass_flag_disables_gate(self, tmp_path: Path) -> None:
        from experiments.p3_specialization.analyze_291 import (
            _enforce_k200_gate,
            aggregate_arm,
        )

        seed_dir = tmp_path / "seed_42"
        seed_dir.mkdir()
        (seed_dir / "metrics.json").write_text(
            json.dumps([{"iteration": 0, "mean_step_reward_team": -90.0}])
        )
        arm = aggregate_arm([seed_dir], "joint_control_ppo")
        # Explicit opt-out: gate should pass.
        failure = _enforce_k200_gate(arm, allow_phase1_only=True)
        assert failure is None

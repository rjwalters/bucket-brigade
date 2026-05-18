"""Tests for the MacroActionEnv wrapper (issue #286).

The wrapper coarsens the decision frequency by committing each agent to a
multi-step option (PATROL, DEFEND_OWN, REST_UNTIL_FIRE, FOLLOW_j) for
``commit_steps`` base-env steps. These tests verify the contract,
per-option primitive semantics, reward accumulation, and edge cases.
"""

import dataclasses

import numpy as np
import pytest

from bucket_brigade.envs import (
    BucketBrigadeEnv,
    MacroActionEnv,
    OPT_DEFEND_OWN,
    OPT_FOLLOW_BASE,
    OPT_PATROL,
    OPT_REST_UNTIL_FIRE,
    trivial_cooperation_scenario,
)
from bucket_brigade.envs.scenarios_generated import minimal_specialization_scenario


def _make_env(num_agents: int = 4, commit_steps: int = 3) -> MacroActionEnv:
    scenario = minimal_specialization_scenario(num_agents=num_agents)
    base = BucketBrigadeEnv(scenario=scenario)
    return MacroActionEnv(base, commit_steps=commit_steps)


def _make_env_no_fire(num_agents: int = 4, commit_steps: int = 3) -> MacroActionEnv:
    """Build an env with ignition probability zeroed out (for REST_UNTIL_FIRE)."""
    scenario = minimal_specialization_scenario(num_agents=num_agents)
    scenario = dataclasses.replace(scenario, prob_house_catches_fire=0.0)
    base = BucketBrigadeEnv(scenario=scenario)
    return MacroActionEnv(base, commit_steps=commit_steps)


class TestMacroActionEnvContract:
    """Wrapper preserves the BucketBrigadeEnv contract."""

    def test_step_contract_matches_base(self):
        env = _make_env(num_agents=4, commit_steps=3)
        obs = env.reset(seed=42)

        # Same obs dict structure as BucketBrigadeEnv.
        for key in ("signals", "locations", "houses", "last_actions", "scenario_info"):
            assert key in obs, f"missing obs key: {key}"
        assert obs["signals"].shape == (4,)
        assert obs["locations"].shape == (4,)
        assert obs["houses"].shape == (env.num_houses,)
        assert obs["last_actions"].shape == (4, 2)

        # Step with all PATROL.
        macro_actions = np.zeros(4, dtype=np.int64)  # OPT_PATROL
        obs2, rewards, dones, info = env.step(macro_actions)

        # Return shapes match the base env.
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert isinstance(info, dict)
        assert "base_steps" in info
        assert info["base_steps"] <= 3
        # The base env returns one boolean for done broadcast across agents
        # so all four entries agree.
        assert bool(dones[0]) == bool(dones[1]) == bool(dones[2]) == bool(dones[3])

    def test_accepts_2d_macro_action_shape(self):
        """Trainer passes ``[N, 1]`` for a single Discrete head — accept it."""
        env = _make_env(num_agents=4, commit_steps=2)
        env.reset(seed=0)
        macro_actions = np.zeros((4, 1), dtype=np.int64)  # all PATROL
        obs, rewards, dones, info = env.step(macro_actions)
        assert rewards.shape == (4,)

    def test_action_space_size(self):
        env = _make_env(num_agents=4)
        # num_options = 3 + (num_agents - 1) = 3 + 3 = 6.
        assert env.num_options == 6


class TestPatrolOption:
    def test_patrol_advances_position(self):
        """After PATROL for N=3 steps, location = (start + 3) mod num_houses."""
        env = _make_env(num_agents=4, commit_steps=3)
        env.reset(seed=42)
        start_locations = env.locations.copy()

        macro_actions = np.full(4, OPT_PATROL, dtype=np.int64)
        _, _, _, info = env.step(macro_actions)

        # If the episode early-terminated, base_steps may be < commit_steps;
        # use that as the ground truth for the position advance.
        n_steps = int(info["base_steps"])
        expected = (start_locations.astype(np.int64) + n_steps) % env.num_houses
        np.testing.assert_array_equal(env.locations, expected.astype(np.int8))


class TestDefendOwnOption:
    def test_defend_own_parks_at_home(self):
        """DEFEND_OWN sets target house to agent_home_positions[i] every step."""
        env = _make_env(num_agents=4, commit_steps=3)
        env.reset(seed=42)
        home_positions = env.agent_home_positions.copy()

        macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
        _, _, _, info = env.step(macro_actions)

        # After the macro-step, every agent's location should be its home.
        np.testing.assert_array_equal(env.locations, home_positions.astype(np.int8))

        # Each primitive action emitted should also park at home.
        for prim in info["primitive_actions"]:
            np.testing.assert_array_equal(prim[:, 0], home_positions.astype(np.int64))


class TestRestUntilFireOption:
    def test_rest_until_fire_stays_rest_with_no_fires(self):
        """With prob_house_catches_fire=0 and no initial fires, REST_UNTIL_FIRE
        emits only REST primitives."""
        env = _make_env_no_fire(num_agents=4, commit_steps=3)
        env.reset(seed=42)

        # Sanity: no houses are burning at the start of the window.
        # (The wrapper checks burning *after* the first base step, but
        # ignition probability is zero so this stays the case.)
        assert not np.any(env.houses == BucketBrigadeEnv.BURNING)

        macro_actions = np.full(4, OPT_REST_UNTIL_FIRE, dtype=np.int64)
        _, _, _, info = env.step(macro_actions)

        # Every primitive emitted should have mode=REST (=0).
        for prim in info["primitive_actions"]:
            np.testing.assert_array_equal(
                prim[:, 1], np.zeros(env.num_agents, dtype=np.int64)
            )

    def test_rest_until_fire_transitions_to_patrol(self):
        """When a fire is visible at the start of a step, REST_UNTIL_FIRE
        switches to the PATROL primitive (target = (loc+1) % H) for the
        remainder of the commit window."""
        env = _make_env_no_fire(num_agents=4, commit_steps=4)
        env.reset(seed=42)

        # Force-inject a fire BEFORE the first step. The wrapper checks
        # ``base_env.houses`` for BURNING *before* building each primitive,
        # so step 0 should already emit the PATROL primitive.
        env.base_env.houses[0] = BucketBrigadeEnv.BURNING

        macro_actions = np.full(4, OPT_REST_UNTIL_FIRE, dtype=np.int64)
        _, _, _, info = env.step(macro_actions)

        prims = info["primitive_actions"]
        # Step 0 must use the PATROL branch: target = (loc+1) % H. Initial
        # loc is 0, so target = 1.
        assert np.all(prims[0][:, 0] == 1), (
            f"Expected PATROL target=(loc+1)%H=1 on step 0, got {prims[0][:, 0]}"
        )

        # Sticky: the fire-seen flag stays True after burn-out. Even after
        # house 0 -> RUINED, subsequent steps still PATROL.
        if len(prims) >= 2:
            # After step 0, agents are at location 1. Step 1's PATROL
            # target = (1 + 1) % H = 2.
            assert np.all(prims[1][:, 0] == 2), (
                f"Expected PATROL target=2 on step 1, got {prims[1][:, 0]}"
            )


class TestFollowOption:
    def test_follow_copies_target_primitive_action(self):
        """FOLLOW_j emits obs['last_actions'][j] as its primitive [house, mode]."""
        env = _make_env(num_agents=4, commit_steps=2)
        env.reset(seed=42)

        # Drive one base step directly to seed last_actions with non-zero values.
        # Use the base env: agent 0 -> house 3 work, agent 1 -> house 5 rest, etc.
        seed_primitive = np.array(
            [
                [3, BucketBrigadeEnv.WORK, BucketBrigadeEnv.WORK],
                [5, BucketBrigadeEnv.REST, BucketBrigadeEnv.REST],
                [7, BucketBrigadeEnv.WORK, BucketBrigadeEnv.WORK],
                [9, BucketBrigadeEnv.REST, BucketBrigadeEnv.REST],
            ],
            dtype=np.int64,
        )
        env.base_env.step(seed_primitive)

        # Now ask agent 0 to FOLLOW agent 1 (j=1). bundle_idx = 1 for agent 0
        # because the FOLLOW bundle for agent 0 lists [1, 2, 3] -> opts [3, 4, 5].
        # So FOLLOW_1 from agent 0 is opt = OPT_FOLLOW_BASE + 0 = 3? No:
        # bundle[0] = agent 1 -> opt 3. bundle[1] = agent 2 -> opt 4.
        # bundle[2] = agent 3 -> opt 5. So FOLLOW_1 from agent 0 = opt 3.
        follow_1_from_0 = OPT_FOLLOW_BASE + 0  # agent 1 is bundle index 0

        macro_actions = np.array(
            [follow_1_from_0, OPT_PATROL, OPT_PATROL, OPT_PATROL],
            dtype=np.int64,
        )
        _, _, _, info = env.step(macro_actions)

        # First primitive emitted by agent 0 should equal agent 1's seeded
        # last_action: [house=5, mode=REST, signal=REST].
        first_prim = info["primitive_actions"][0]
        assert int(first_prim[0, 0]) == 5
        assert int(first_prim[0, 1]) == int(BucketBrigadeEnv.REST)

    def test_follow_skips_self(self):
        """FOLLOW bundle for agent i never targets j == i."""
        env = _make_env(num_agents=4)
        # For agent 2, the bundle is [agent 0, agent 1, agent 3] (skipping self).
        # opt 3 -> j=0, opt 4 -> j=1, opt 5 -> j=3.
        assert env._follow_target(agent_i=2, opt=OPT_FOLLOW_BASE + 0) == 0
        assert env._follow_target(agent_i=2, opt=OPT_FOLLOW_BASE + 1) == 1
        assert env._follow_target(agent_i=2, opt=OPT_FOLLOW_BASE + 2) == 3
        # And for agent 0, the bundle is [1, 2, 3].
        assert env._follow_target(agent_i=0, opt=OPT_FOLLOW_BASE + 0) == 1
        assert env._follow_target(agent_i=0, opt=OPT_FOLLOW_BASE + 2) == 3


class TestRewardAggregation:
    def test_reward_aggregation_undiscounted(self):
        """Macro-step reward equals the sum of per-base-step rewards over the
        commit window.

        Compare against driving ``BucketBrigadeEnv`` directly with the same
        primitive sequence the wrapper emits."""
        # Wrapper-driven rollout.
        env_w = _make_env(num_agents=4, commit_steps=3)
        env_w.reset(seed=123)
        macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
        _, macro_rewards, _, info = env_w.step(macro_actions)
        primitives = info["primitive_actions"]
        n_steps = info["base_steps"]

        # Replay the same primitive sequence against a freshly-seeded base env.
        scenario = minimal_specialization_scenario(num_agents=4)
        base = BucketBrigadeEnv(scenario=scenario)
        base.reset(seed=123)
        sum_rewards = np.zeros(4, dtype=np.float32)
        for k in range(n_steps):
            _, rewards, _, _ = base.step(primitives[k])
            sum_rewards += rewards.astype(np.float32)

        np.testing.assert_allclose(macro_rewards, sum_rewards, atol=1e-5)


class TestEarlyTermination:
    def test_early_termination_done_short_circuits(self):
        """If base env terminates at step k < commit_steps, macro-step returns
        at that boundary with dones=True."""
        # Trivial cooperation scenario terminates fast with all-fire / all-safe.
        # Force termination by using min_nights=1 via dataclasses.replace.
        scenario = trivial_cooperation_scenario(num_agents=4)
        scenario = dataclasses.replace(scenario, min_nights=1)
        base = BucketBrigadeEnv(scenario=scenario)
        env = MacroActionEnv(base, commit_steps=20)
        env.reset(seed=42)

        macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
        _, _, dones, info = env.step(macro_actions)

        # With commit_steps=20 and short min_nights, episode must end
        # before the window closes — wrapper short-circuits.
        assert info["base_steps"] < 20
        assert bool(dones.any())

    def test_commit_steps_one_collapses_to_primitive_behavior(self):
        """commit_steps=1 means each macro-action is exactly one base step."""
        env = _make_env(num_agents=4, commit_steps=1)
        env.reset(seed=7)
        macro_actions = np.full(4, OPT_PATROL, dtype=np.int64)
        _, _, _, info = env.step(macro_actions)
        assert info["base_steps"] == 1
        assert len(info["primitive_actions"]) == 1


class TestEdgeCases:
    def test_invalid_option_raises(self):
        env = _make_env(num_agents=4, commit_steps=3)
        env.reset(seed=0)
        # num_options=6 for N=4, so 6 is out of range.
        bad = np.array([6, 0, 0, 0], dtype=np.int64)
        with pytest.raises(ValueError):
            env.step(bad)

    def test_commit_steps_must_be_positive(self):
        base = BucketBrigadeEnv(scenario=minimal_specialization_scenario(num_agents=4))
        with pytest.raises(ValueError):
            MacroActionEnv(base, commit_steps=0)

    def test_commit_steps_exceeds_min_nights_no_state_leak(self):
        """Episode ends mid-commit; verify no IndexError and _prev_houses_state
        is reset cleanly across reset (regression check for
        bucket_brigade_env.py:130-132)."""
        scenario = minimal_specialization_scenario(num_agents=4)
        scenario = dataclasses.replace(scenario, min_nights=2)
        base = BucketBrigadeEnv(scenario=scenario)
        env = MacroActionEnv(base, commit_steps=30)

        # Run several macro-steps with auto-resets to check state hygiene.
        env.reset(seed=42)
        for _ in range(3):
            macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
            _, _, dones, _ = env.step(macro_actions)
            if bool(dones.any()):
                env.reset()  # caller-driven auto-reset
                # After reset, _prev_houses_state must be all zeros (no stale
                # RUINED bits leaking from the prior episode).
                assert np.all(env.base_env._prev_houses_state == 0)


class TestSmokeRollout:
    def test_smoke_rollout(self):
        """Instantiate env via wrapper, step a few rollouts, sanity-check."""
        env = _make_env(num_agents=4, commit_steps=3)
        obs = env.reset(seed=42)
        assert obs is not None

        rng = np.random.RandomState(0)
        for _ in range(10):
            # Sample a random option per agent.
            macro_actions = rng.randint(0, env.num_options, size=4).astype(np.int64)
            obs, rewards, dones, info = env.step(macro_actions)
            assert rewards.shape == (4,)
            assert dones.shape == (4,)
            assert 1 <= info["base_steps"] <= 3
            if bool(dones.any()):
                env.reset()

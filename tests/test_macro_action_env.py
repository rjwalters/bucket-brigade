"""Tests for the MacroActionEnv wrapper (issue #286).

The wrapper coarsens the decision frequency by committing each agent to a
multi-step option (PATROL, DEFEND_OWN, REST_UNTIL_FIRE, FOLLOW_j) for
``commit_steps`` base-env steps. These tests verify the contract,
per-option primitive semantics, reward accumulation, and edge cases.
"""

import dataclasses
from typing import List

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


# ----------------------------------------------------------------------
# Issue #344: MacroActionEnv + commitment_mode="two_phase" composition.
# Architect ratified Option 2a: round-1 signal sampled per base step
# inside the macro window, round-2 macro committed for the whole window.
# ----------------------------------------------------------------------


def _make_two_phase_env(num_agents: int = 4, commit_steps: int = 3) -> MacroActionEnv:
    scenario = minimal_specialization_scenario(num_agents=num_agents)
    scenario = dataclasses.replace(scenario, commitment_mode="two_phase")
    base = BucketBrigadeEnv(scenario=scenario)
    return MacroActionEnv(base, commit_steps=commit_steps)


def _make_two_phase_env_no_fire(
    num_agents: int = 4, commit_steps: int = 3
) -> MacroActionEnv:
    scenario = minimal_specialization_scenario(num_agents=num_agents)
    scenario = dataclasses.replace(
        scenario,
        commitment_mode="two_phase",
        prob_house_catches_fire=0.0,
    )
    base = BucketBrigadeEnv(scenario=scenario)
    return MacroActionEnv(base, commit_steps=commit_steps)


class TestMacroTwoPhaseComposition:
    """Issue #344: smoke + contract tests for MacroActionEnv x two_phase."""

    def test_construction_no_longer_raises_on_two_phase(self):
        """Pre-#344 the wrapper gated two-phase with NotImplementedError;
        post-#344 the constructor accepts two-phase scenarios."""
        scenario = minimal_specialization_scenario(num_agents=4)
        scenario = dataclasses.replace(scenario, commitment_mode="two_phase")
        base = BucketBrigadeEnv(scenario=scenario)
        # Must not raise.
        env = MacroActionEnv(base, commit_steps=3)
        assert env.num_options == 6  # 3 + (4 - 1)

    def test_step_raises_on_two_phase_scenario(self):
        """The simultaneous :meth:`step` path now raises on two-phase
        scenarios — callers must use :meth:`step_two_phase` instead.
        Mirrors the base env's :meth:`BucketBrigadeEnv.step` guardrail."""
        env = _make_two_phase_env(num_agents=4, commit_steps=3)
        env.reset(seed=42)
        macro_actions = np.zeros(4, dtype=np.int64)
        with pytest.raises(RuntimeError, match="two_phase"):
            env.step(macro_actions)

    def test_step_two_phase_raises_on_simultaneous_scenario(self):
        """Conversely, :meth:`step_two_phase` requires two-phase mode."""
        env = _make_env(num_agents=4, commit_steps=3)
        env.reset(seed=42)
        r1 = np.zeros((3, 4), dtype=np.int8)
        macro_actions = np.zeros(4, dtype=np.int64)
        with pytest.raises(RuntimeError, match="two_phase"):
            env.step_two_phase(r1, macro_actions)

    def test_smoke_step_two_phase_executes_full_window(self):
        """Composition smoke: wrapper + two_phase constructs and steps
        cleanly through one macro window."""
        env = _make_two_phase_env(num_agents=4, commit_steps=3)
        obs = env.reset(seed=42)
        assert obs is not None

        # Round-1 signals: shape (commit_steps, num_agents). All zeros
        # (Uncommitted/Rest signals) — should not interfere with macro.
        r1 = np.zeros((3, 4), dtype=np.int8)
        macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
        obs2, rewards, dones, info = env.step_two_phase(r1, macro_actions)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert info["base_steps"] <= 3
        # Wrapper records the round-1 signals it actually fed to the env.
        assert "round1_signals_used" in info
        assert len(info["round1_signals_used"]) == info["base_steps"]
        for r in info["round1_signals_used"]:
            np.testing.assert_array_equal(r, np.zeros(4, dtype=np.int8))

    def test_step_two_phase_accepts_1d_round1_broadcast(self):
        """A 1D round-1 vector is broadcast to all base steps (Option 2b
        convenience — the deception channel is throttled to 1/N but the
        signature is convenient for tests / direct callers)."""
        env = _make_two_phase_env(num_agents=4, commit_steps=3)
        env.reset(seed=7)
        r1 = np.array([1, 0, 1, 0], dtype=np.int8)
        macro_actions = np.full(4, OPT_PATROL, dtype=np.int64)
        _, _, _, info = env.step_two_phase(r1, macro_actions)
        # All recorded r1 vectors should equal the broadcast input.
        for r in info["round1_signals_used"]:
            np.testing.assert_array_equal(r, r1)

    def test_round1_signals_property_passthrough(self):
        """The wrapper exposes a ``round1_signals`` pass-through so the
        trainer's existing two-phase rollout code (which mutates
        ``env.round1_signals`` directly) keeps working uniformly."""
        env = _make_two_phase_env(num_agents=4, commit_steps=3)
        env.reset(seed=0)
        np.testing.assert_array_equal(env.round1_signals, np.zeros(4, dtype=np.int8))
        env.round1_signals = np.array([1, 0, 1, 0], dtype=np.int8)
        np.testing.assert_array_equal(
            env.base_env.round1_signals, np.array([1, 0, 1, 0], dtype=np.int8)
        )


class TestMacroTwoPhaseDeceptionChannel:
    """**PR-GATE**: deception-channel preservation under composition.

    The composition's analog of
    ``tests/test_environment.py::TestCommitmentMode::test_can_still_lie``.
    Under Option 2a the round-1 signal is sampled per base step; under
    Option 1 or Option 2b it would be throttled to 1/commit_steps. This
    test asserts the per-base-step rate is preserved — if a hardcoded
    Liar policy emits round-1=WORK on every base step and chooses a
    REST_UNTIL_FIRE macro (whose round-2 mode is REST until first fire),
    the lie-rate stays at 100% per base step, not 100%/commit_steps.

    Architect-flagged as the PR-gate (issue #344). If this fails the
    composition has silently collapsed to Option 1's deception profile
    and the PR must NOT merge.
    """

    def test_can_still_lie_under_macro_per_base_step(self):
        """Per-base-step lie-rate >= 50% with a hardcoded liar (the
        threshold discriminates Option 2a from Option 2b: a 1/N=20%
        cap would fail the >= 50% gate even in the worst case)."""
        env = _make_two_phase_env_no_fire(num_agents=4, commit_steps=5)
        env.reset(seed=42)

        # Hardcoded liar: round-1 signal=WORK (1) on every base step.
        # Macro = REST_UNTIL_FIRE → round-2 mode=REST (0) while no fire
        # has been observed (and the env is constructed with
        # prob_house_catches_fire=0 so no fire ever ignites).
        r1_lie = np.ones((5, 4), dtype=np.int8)  # WORK on every base step
        macro_actions = np.full(4, OPT_REST_UNTIL_FIRE, dtype=np.int64)

        lying_count = 0
        total_count = 0
        # Drive several macro-steps (each = commit_steps base steps).
        for _ in range(4):
            if env.done:
                env.reset()
            obs, _, _, info = env.step_two_phase(r1_lie, macro_actions)
            # Inspect every (round-1 signal, round-2 mode) pair from the
            # primitives the wrapper emitted across the commit window.
            r1_used = info["round1_signals_used"]
            primitives = info["primitive_actions"]
            assert len(r1_used) == len(primitives)
            for r1_k, prim_k in zip(r1_used, primitives):
                for i in range(env.num_agents):
                    total_count += 1
                    r2_mode_i = int(prim_k[i, 1])
                    r1_i = int(r1_k[i])
                    if r1_i != r2_mode_i:
                        lying_count += 1

        assert total_count > 0
        lie_rate = lying_count / total_count
        # The hardcoded liar lies on every base step (rate == 1.0).
        # PR-gate threshold: >= 50% (much higher than 1/N=20%) to
        # discriminate Option 2a from Option 2b.
        assert lie_rate >= 0.5, (
            f"PR-GATE (#344): expected per-base-step lie-rate >= 50% with "
            f"a hardcoded liar + REST_UNTIL_FIRE macro; got {lie_rate:.4f} "
            f"({lying_count}/{total_count} pairs inconsistent). If this "
            f"rate is throttled to ~1/commit_steps the composition has "
            f"silently collapsed to Option 1's deception profile and the "
            f"deception channel has been destroyed. DO NOT MERGE."
        )

    def test_per_base_step_signals_not_broadcast(self):
        """Discriminator between Option 2a and Option 2b: each base step
        inside the macro window must consume the r1 row at its position
        in the [commit_steps, N] input — NOT a single broadcast value.

        If the wrapper silently broadcast a single r1 vector to all base
        steps (Option 2b), the recorded ``round1_signals_used`` would
        be identical across all base steps. This test sends a strictly
        varying matrix and asserts row k is used at base step k."""
        env = _make_two_phase_env_no_fire(num_agents=4, commit_steps=3)
        env.reset(seed=0)
        # Distinct rows so 2b broadcast would fail the assertion.
        r1 = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=np.int8,
        )
        macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
        _, _, _, info = env.step_two_phase(r1, macro_actions)
        r1_used = info["round1_signals_used"]
        assert len(r1_used) == info["base_steps"]
        for k in range(info["base_steps"]):
            (
                np.testing.assert_array_equal(r1_used[k], r1[k]),
                (
                    f"base step {k} used r1={r1_used[k]} but input row "
                    f"{k} was {r1[k]}; the wrapper has silently broadcast a "
                    f"single r1 vector (Option 2b), throttling the deception "
                    f"channel to 1/commit_steps"
                ),
            )

    def test_round1_signals_visible_in_obs(self):
        """The base env's ``round1_signals`` slot reflects the LAST base
        step's round-1 signal after the macro window closes (matches the
        per-step ``base_env.step_two_phase`` contract)."""
        env = _make_two_phase_env_no_fire(num_agents=4, commit_steps=3)
        env.reset(seed=0)
        # Last base step's signal: [0, 1, 0, 1].
        r1 = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 1, 0],
                [0, 1, 0, 1],
            ],
            dtype=np.int8,
        )
        macro_actions = np.full(4, OPT_DEFEND_OWN, dtype=np.int64)
        obs, _, _, info = env.step_two_phase(r1, macro_actions)
        if info["base_steps"] == 3:
            np.testing.assert_array_equal(
                obs["round1_signals"], np.array([0, 1, 0, 1], dtype=np.int8)
            )


class TestMacroSinglePhaseRegression:
    """Issue #344: single-phase regression — the wrapper's behavior on
    ``commitment_mode="simultaneous"`` scenarios must be bit-identical
    to the pre-#344 path. The new construction logic and the new
    ``step_two_phase`` method must not perturb anything on the
    single-phase path."""

    def test_single_phase_step_bit_identical(self):
        """Wrapper-driven single-phase rollout produces the same
        (rewards, locations, houses) as a direct re-play of the wrapper-
        emitted primitives against a fresh base env. This is the same
        invariant as :class:`TestRewardAggregation` but exercised across
        several macro-steps + random options to maximize the chance of
        catching a regression."""
        env_w = _make_env(num_agents=4, commit_steps=3)
        env_w.reset(seed=42)
        rng = np.random.RandomState(0)
        all_primitives: List[np.ndarray] = []
        all_macro_rewards: List[np.ndarray] = []
        # Save the n_steps actually executed each macro-step.
        all_n_steps: List[int] = []
        for _ in range(5):
            macro_actions = rng.randint(0, env_w.num_options, size=4).astype(np.int64)
            _, macro_rewards, dones, info = env_w.step(macro_actions)
            all_primitives.extend(info["primitive_actions"])
            all_macro_rewards.append(macro_rewards.copy())
            all_n_steps.append(int(info["base_steps"]))
            if bool(dones.any()):
                env_w.reset()

        # Replay all primitives directly against a fresh base env (with
        # the same seed + auto-reset pattern). Compare per-macro-step
        # accumulated rewards.
        scenario = minimal_specialization_scenario(num_agents=4)
        base = BucketBrigadeEnv(scenario=scenario)
        base.reset(seed=42)
        primitive_idx = 0
        for macro_idx, n_steps in enumerate(all_n_steps):
            sum_rew = np.zeros(4, dtype=np.float32)
            for _ in range(n_steps):
                _, rewards, dones, _ = base.step(all_primitives[primitive_idx])
                sum_rew += rewards.astype(np.float32)
                primitive_idx += 1
                if bool(dones.any()):
                    base.reset()
                    break
            np.testing.assert_allclose(all_macro_rewards[macro_idx], sum_rew, atol=1e-5)

    def test_single_phase_construction_still_works(self):
        """Constructing the wrapper on a default (simultaneous) scenario
        does not raise after the #344 gate removal."""
        env = _make_env(num_agents=4, commit_steps=3)
        assert env.num_options == 6
        assert env._is_two_phase is False

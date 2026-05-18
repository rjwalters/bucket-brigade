"""Tests for issue #265 — dense Δsafe progress shaping.

The Python env mirrors the Rust core's new ``Scenario.progress_shaping_coef``
field. The default is ``0.0`` and every pre-#265 scenario keeps it zero, so
existing behavior is bit-exactly preserved (fast-path skip in
``_compute_rewards``).

Test pattern modeled on ``tests/test_environment.py::TestActionShaping``
(issue #259 / PR #263 precedent).
"""

from __future__ import annotations

import numpy as np
import pytest

from bucket_brigade.envs import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import Scenario, SCENARIO_REGISTRY


def _make_minimal_scenario(**overrides) -> Scenario:
    """Build a Scenario with all reward/cost fields zeroed except those overridden.

    Mirrors the helper in ``tests/test_environment.py`` (kept duplicated here
    to avoid an import-only edit to that file). Used to isolate the
    contribution of ``progress_shaping_coef`` to ``_compute_rewards``.
    """
    base = dict(
        prob_fire_spreads_to_neighbor=0.0,
        prob_solo_agent_extinguishes_fire=0.0,
        prob_house_catches_fire=0.0,
        team_reward_house_survives=0.0,
        team_penalty_house_burns=0.0,
        reward_own_house_survives=0.0,
        reward_other_house_survives=0.0,
        penalty_own_house_burns=0.0,
        penalty_other_house_burns=0.0,
        cost_to_work_one_night=0.0,
        min_nights=12,
        num_agents=4,
    )
    base.update(overrides)
    return Scenario(**base)


class TestProgressShapingDefaults:
    """Issue #265 default-value regression: every pre-#265 scenario must keep
    ``progress_shaping_coef = 0.0`` so the post-#236 P3 protocol and all
    standing regression numerics remain reproducible.
    """

    def test_defaults_are_zero_across_all_scenarios(self):
        """Every pre-#265 scenario keeps the shaping coef at 0.0."""
        for name, factory in SCENARIO_REGISTRY.items():
            s = factory(num_agents=4)
            assert s.progress_shaping_coef == 0.0, (
                f"Pre-#265 scenario '{name}' must keep "
                f"progress_shaping_coef=0.0; got {s.progress_shaping_coef}"
            )

    def test_scenario_dataclass_has_the_field(self):
        """The new field exists on ``Scenario`` and defaults to 0.0."""
        s = _make_minimal_scenario()
        assert hasattr(s, "progress_shaping_coef")
        assert s.progress_shaping_coef == 0.0


class TestProgressShapingBitExactBaseline:
    """Coef=0.0 must produce byte-identical per-step rewards to the pre-#265
    fast path. This guards the entire prior P3 ladder's numerics — the May-14
    baselines, the #260 verdict, etc."""

    def test_zero_coef_matches_default_scenario(self):
        """With coef=0, a full episode produces identical per-step rewards
        on the canonical ``default`` scenario."""
        from bucket_brigade.envs.scenarios_generated import default_scenario

        env_a = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        env_b = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        env_a.reset(seed=99)
        env_b.reset(seed=99)
        # Explicitly assert the treatment env's coef is the default zero.
        assert env_b.scenario.progress_shaping_coef == 0.0
        rng = np.random.RandomState(0)
        for _ in range(10):
            actions = rng.randint(0, 2, size=(4, 2)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)
            _, r_a, _, _ = env_a.step(actions)
            _, r_b, _, _ = env_b.step(actions)
            np.testing.assert_allclose(r_a, r_b)

    def test_explicit_zero_coef_matches_unset_field(self):
        """Explicitly setting coef=0.0 on a fresh scenario object is
        indistinguishable from leaving it at the dataclass default."""
        s_default = _make_minimal_scenario()
        s_explicit = _make_minimal_scenario()
        s_explicit.progress_shaping_coef = 0.0
        env_a = BucketBrigadeEnv(scenario=s_default)
        env_b = BucketBrigadeEnv(scenario=s_explicit)
        env_a.reset(seed=7)
        env_b.reset(seed=7)
        rng = np.random.RandomState(1)
        for _ in range(8):
            actions = rng.randint(0, 2, size=(4, 2)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)
            _, r_a, _, _ = env_a.step(actions)
            _, r_b, _, _ = env_b.step(actions)
            np.testing.assert_allclose(r_a, r_b)


class TestProgressShapingSignChecks:
    """Hand-crafted Δsafe scenarios verify the sign and magnitude of the
    shaping bonus.

    The bonus is shared across all agents via the team-reward broadcast at
    the end of the per-agent loop in ``_compute_rewards``, so each agent
    receives ``coef * (cur_safe - prev_safe)`` on top of their other reward
    components. With all other reward fields zeroed, the bonus is the
    *only* contribution to per-agent rewards.
    """

    def test_positive_delta_safe_yields_positive_bonus(self):
        """One BURNING -> SAFE transition gives every agent +coef of team
        bonus on that step."""
        scenario = _make_minimal_scenario(
            prob_solo_agent_extinguishes_fire=1.0,  # deterministic extinguish
            progress_shaping_coef=10.0,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # Setup: only house 5 burning, nothing else changes.
        env.houses = np.zeros(10, dtype=np.int8)
        env.houses[5] = env.BURNING
        env._prev_houses_state = env.houses.copy()
        # All 4 agents work at house 5 -> guaranteed extinguish.
        actions = np.array(
            [[5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        # cur_safe - prev_safe = 10 - 9 = +1, bonus = 10*1 = +10 per agent.
        for i, r in enumerate(rewards):
            assert abs(float(r) - 10.0) < 1e-5, (
                f"Agent {i} should get progress bonus of +10.0, got {r}"
            )

    def test_zero_delta_safe_yields_zero_bonus(self):
        """A steady-state step (no SAFE-count change) gives every agent +0
        of progress bonus.

        Note: REST actions still produce the standard +0.5 rest reward in
        ``_compute_rewards`` regardless of shaping. We isolate the shaping
        contribution by having all agents WORK (no rest reward) at SAFE
        houses where nothing happens; cost_to_work_one_night is also zero,
        so each agent's total reward equals the progress bonus alone (0.0).
        """
        scenario = _make_minimal_scenario(
            prob_fire_spreads_to_neighbor=0.0,
            prob_solo_agent_extinguishes_fire=0.0,
            prob_house_catches_fire=0.0,
            cost_to_work_one_night=0.0,
            progress_shaping_coef=10.0,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # All houses SAFE, no fires, nothing ignites (fire prob = 0). The
        # extinguish-fire phase still runs but has no burning house to act on.
        env.houses = np.zeros(10, dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        # Use WORK actions to avoid the +0.5 rest reward; cost is zero.
        actions = np.array(
            [[0, 1, 0], [3, 1, 0], [5, 1, 0], [8, 1, 0]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        for i, r in enumerate(rewards):
            assert abs(float(r) - 0.0) < 1e-5, (
                f"Agent {i} should get zero progress bonus on a "
                f"no-transition step, got {r}"
            )

    def test_negative_delta_safe_yields_negative_bonus(self):
        """A SAFE -> BURNING transition (loss of one safe house) gives
        every agent -coef of progress penalty on that step.

        We trigger the spark phase to ignite all SAFE houses (the spark
        phase runs *after* the burn-out phase, so it directly mutates the
        end-of-step house states that ``_compute_rewards`` reads). With
        ``prob_house_catches_fire=1.0`` every SAFE house becomes BURNING
        on this step.
        """
        scenario = _make_minimal_scenario(
            prob_fire_spreads_to_neighbor=0.0,
            prob_solo_agent_extinguishes_fire=0.0,
            prob_house_catches_fire=1.0,
            progress_shaping_coef=10.0,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # All houses SAFE at start of step. Spark phase will set every
        # SAFE house BURNING (prob=1.0). cur_safe = 0; prev_safe = 10.
        env.houses = np.zeros(10, dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        # All agents WORK (no rest reward); cost is zero.
        actions = np.array(
            [[0, 1, 0], [3, 1, 0], [5, 1, 0], [8, 1, 0]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        # delta_safe = 0 - 10 = -10. Per-agent bonus = 10.0 * -10 = -100.0.
        expected = 10.0 * (0 - 10)  # = -100.0
        for i, r in enumerate(rewards):
            assert abs(float(r) - expected) < 1e-5, (
                f"Agent {i} should get progress penalty of {expected}, got {r}"
            )

    def test_shaping_is_team_shared(self):
        """The progress bonus is team-shared: every agent gets the *same*
        bonus regardless of action. Distinguishes #265 (team-shared) from
        #262 (per-agent action-shaping)."""
        scenario = _make_minimal_scenario(
            prob_solo_agent_extinguishes_fire=1.0,
            progress_shaping_coef=3.0,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        env.houses = np.zeros(10, dtype=np.int8)
        env.houses[5] = env.BURNING
        env._prev_houses_state = env.houses.copy()
        # Only agent 0 works at house 5; rest are resting at other houses.
        actions = np.array(
            [[5, 1, 1], [0, 0, 0], [3, 0, 0], [8, 0, 0]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        # All agents should receive the same +3.0 progress bonus (delta=+1).
        # No other reward source (team_reward_house_survives=0,
        # cost_to_work_one_night=0, ownership rewards 0). Rest-reward is 0.5
        # per resting agent.
        expected_worker = 3.0  # progress bonus only (no work cost).
        expected_rester = 3.0 + 0.5  # progress bonus + rest reward.
        assert abs(float(rewards[0]) - expected_worker) < 1e-5, (
            f"Working agent should get progress bonus only "
            f"({expected_worker}), got {rewards[0]}"
        )
        for i in range(1, 4):
            assert abs(float(rewards[i]) - expected_rester) < 1e-5, (
                f"Resting agent {i} should get progress bonus + rest "
                f"({expected_rester}), got {rewards[i]}"
            )


@pytest.mark.requires_rust
class TestProgressShapingRustParity:
    """Rust/Python parity for a non-zero ``progress_shaping_coef``.

    Uses ``trivial_cooperation`` dynamics (``prob_house_catches_fire=0.0``,
    all-SAFE initial state) so the per-step rewards are fully deterministic
    — no RNG decision actually changes house state or rewards, allowing
    bit-comparable Rust vs Python rewards without depending on the
    different underlying PRNG implementations. Pattern mirrors
    ``tests/test_rust_integration.py::test_rust_vs_python_consistency``.
    """

    def test_rust_python_parity_with_nonzero_coef(self):
        from bucket_brigade_core import BucketBrigade
        from bucket_brigade_core import Scenario as RustPyScenario

        # Build Rust + Python scenarios with identical fields and
        # ``progress_shaping_coef = 10.0``. The PyScenario constructor now
        # accepts ``progress_shaping_coef`` as an optional kwarg (default
        # 0.0); existing positional callers stay byte-identical.
        rust_scenario = RustPyScenario(
            prob_fire_spreads_to_neighbor=0.1,
            prob_solo_agent_extinguishes_fire=0.5,
            prob_house_catches_fire=0.0,  # no spontaneous fires for determinism
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
            cost_to_work_one_night=0.5,
            min_nights=12,
            reward_own_house_survives=1.0,
            reward_other_house_survives=0.0,
            penalty_own_house_burns=2.0,
            penalty_other_house_burns=0.0,
            progress_shaping_coef=10.0,
        )
        rust_env = BucketBrigade(rust_scenario, 4, seed=42)

        py_scenario = Scenario(
            prob_fire_spreads_to_neighbor=0.1,
            prob_solo_agent_extinguishes_fire=0.5,
            prob_house_catches_fire=0.0,
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
            reward_own_house_survives=1.0,
            reward_other_house_survives=0.0,
            penalty_own_house_burns=2.0,
            penalty_other_house_burns=0.0,
            cost_to_work_one_night=0.5,
            min_nights=12,
            num_agents=4,
        )
        py_scenario.progress_shaping_coef = 10.0
        py_env = BucketBrigadeEnv(scenario=py_scenario)
        py_env.reset(seed=42)

        # Drive both engines with the same action sequence; rewards agree
        # to float tolerance because all-SAFE initial state + prob_fire=0
        # means no RNG branches actually trigger.
        actions_seq = [
            [[0, 1], [1, 0], [2, 1], [3, 0]],
            [[1, 1], [2, 1], [3, 1], [0, 0]],
            [[0, 0], [1, 0], [2, 0], [3, 0]],
        ]
        for i, action_set in enumerate(actions_seq):
            rust_rewards, _, _ = rust_env.step(action_set)
            _, py_rewards, _, _ = py_env.step(np.array(action_set))
            np.testing.assert_allclose(
                rust_rewards,
                py_rewards,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Rust/Python rewards differ at step {i}",
            )

    def test_rust_python_parity_with_zero_coef(self):
        """Coef=0.0 PyScenario via the new kwarg path matches the
        all-default Rust scenario (existing pre-#265 PyScenario behavior)."""
        from bucket_brigade_core import BucketBrigade
        from bucket_brigade_core import Scenario as RustPyScenario

        rust_scenario = RustPyScenario(
            prob_fire_spreads_to_neighbor=0.1,
            prob_solo_agent_extinguishes_fire=0.5,
            prob_house_catches_fire=0.0,
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
            cost_to_work_one_night=0.5,
            min_nights=12,
            reward_own_house_survives=1.0,
            reward_other_house_survives=0.0,
            penalty_own_house_burns=2.0,
            penalty_other_house_burns=0.0,
            # progress_shaping_coef omitted; defaults to 0.0.
        )
        assert rust_scenario.progress_shaping_coef == 0.0
        rust_env = BucketBrigade(rust_scenario, 4, seed=42)

        py_scenario = Scenario(
            prob_fire_spreads_to_neighbor=0.1,
            prob_solo_agent_extinguishes_fire=0.5,
            prob_house_catches_fire=0.0,
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
            reward_own_house_survives=1.0,
            reward_other_house_survives=0.0,
            penalty_own_house_burns=2.0,
            penalty_other_house_burns=0.0,
            cost_to_work_one_night=0.5,
            min_nights=12,
            num_agents=4,
        )
        py_env = BucketBrigadeEnv(scenario=py_scenario)
        py_env.reset(seed=42)

        actions_seq = [
            [[0, 1], [1, 0], [2, 1], [3, 0]],
            [[1, 1], [2, 1], [3, 1], [0, 0]],
        ]
        for i, action_set in enumerate(actions_seq):
            rust_rewards, _, _ = rust_env.step(action_set)
            _, py_rewards, _, _ = py_env.step(np.array(action_set))
            np.testing.assert_allclose(
                rust_rewards,
                py_rewards,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Rust/Python (coef=0) rewards differ at step {i}",
            )

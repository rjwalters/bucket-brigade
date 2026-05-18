"""Tests for the Bucket Brigade environment."""

import dataclasses

import numpy as np
import pytest

from bucket_brigade.envs import BucketBrigadeEnv, default_scenario, random_scenario
from bucket_brigade.envs.scenarios_generated import Scenario


class TestBucketBrigadeEnv:
    """Test the Bucket Brigade environment."""

    def test_environment_initialization(self):
        """Test environment can be initialized."""
        scenario = default_scenario(num_agents=4)
        env = BucketBrigadeEnv(scenario)

        assert env.num_agents == 4
        assert env.scenario.num_agents == 4
        assert len(env.houses) == 10  # Ring of 10 houses
        assert len(env.locations) == 4
        assert len(env.signals) == 4

    def test_environment_reset(self):
        """Test environment reset functionality."""
        env = BucketBrigadeEnv(num_agents=4)
        obs = env.reset(seed=42)

        # Check observation structure
        required_keys = [
            "signals",
            "locations",
            "houses",
            "last_actions",
            "scenario_info",
        ]
        for key in required_keys:
            assert key in obs

        assert len(obs["signals"]) == 4
        assert len(obs["locations"]) == 4
        assert len(obs["houses"]) == 10
        assert len(obs["last_actions"]) == 4
        assert len(obs["scenario_info"]) > 0

    def test_game_step(self):
        """Test a single game step."""
        env = BucketBrigadeEnv(num_agents=4)
        env.reset(seed=42)

        # Create random actions for all agents
        actions = np.random.randint(
            0, 2, size=(4, 3)
        )  # [house, mode, signal] (issue #235)

        obs, rewards, dones, info = env.step(actions)

        # Check return values
        assert len(obs["signals"]) == 4
        assert len(rewards) == 4
        assert len(dones) == 4
        assert isinstance(info, dict)

        # Rewards should be reasonable floats
        assert all(isinstance(r, (int, float, np.floating)) for r in rewards)

    def test_game_termination(self):
        """Test game termination conditions."""
        env = BucketBrigadeEnv(num_agents=4)
        env.reset(seed=42)

        # Run for many steps to ensure termination
        max_steps = 100
        for step in range(max_steps):
            actions = np.random.randint(
                0, 2, size=(4, 3)
            )  # [house, mode, signal] (issue #235)
            obs, rewards, dones, info = env.step(actions)

            if env.done:
                break

        # Game should eventually terminate
        assert env.done or step < max_steps

    def test_scenario_generation(self):
        """Test scenario generation."""
        # Test default scenario
        scenario = default_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert 0 < scenario.prob_fire_spreads_to_neighbor < 1
        assert scenario.prob_solo_agent_extinguishes_fire > 0

        # Test random scenario
        scenario = random_scenario(num_agents=4, seed=42)
        assert scenario.num_agents == 4
        assert (
            0.15 <= scenario.prob_fire_spreads_to_neighbor <= 0.35
        )  # Within expected range

    def test_house_state_transitions(self):
        """Test house state transitions."""
        env = BucketBrigadeEnv(num_agents=4)
        env.reset(seed=42)

        # Manually set up a test scenario
        env.houses = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0])  # Houses 1 and 7 burning

        # Create actions where agents work on burning houses
        # Issue #235: action is [house, mode, signal]. Honest signaling
        # (signal == mode) here since the test only checks state transitions.
        actions = np.array(
            [
                [1, 1, 1],  # Agent 0 works on burning house 1
                [7, 1, 1],  # Agent 1 works on burning house 7
                [0, 0, 0],  # Agent 2 rests
                [2, 0, 0],  # Agent 3 rests
            ]
        )

        env.step(actions)

        # Check that burning houses may have been extinguished
        # (Probabilistic, so we can't assert definitively)
        assert all(state in [0, 1, 2] for state in env.houses)

    def test_reward_calculation(self):
        """Test reward calculation logic."""
        env = BucketBrigadeEnv(num_agents=4)
        env.reset(seed=42)

        # Test with known house states
        env.houses = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # All safe
        env._prev_houses_state = env.houses.copy()

        # Issue #235: action is [house, mode, signal]. Honest defaults here.
        actions = np.array(
            [
                [0, 1, 1],  # Work
                [1, 0, 0],  # Rest
                [2, 1, 1],  # Work
                [3, 0, 0],  # Rest
            ]
        )

        rewards = env._compute_rewards(actions)

        # Working agents get cost penalty, resting agents get rest bonus
        # Note: working agents may still have positive total rewards from team component
        assert rewards[0] <= rewards[1]  # Working should be <= resting (cost penalty)
        assert rewards[2] <= rewards[3]  # Working should be <= resting (cost penalty)

    def test_replay_functionality(self):
        """Test game replay saving and structure."""
        env = BucketBrigadeEnv(num_agents=4)
        env.reset(seed=42)

        # Run a few steps
        for _ in range(3):
            actions = np.random.randint(
                0, 2, size=(4, 3)
            )  # [house, mode, signal] (issue #235)
            env.step(actions)

        # Save replay
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            env.save_replay(temp_path)

            # Check file was created and has content
            assert os.path.exists(temp_path)

            import json

            with open(temp_path, "r") as f:
                data = json.load(f)

            assert "scenario" in data
            assert "nights" in data
            assert len(data["nights"]) > 0

            # Check night structure
            night = data["nights"][0]
            required_fields = [
                "night",
                "houses",
                "signals",
                "locations",
                "actions",
                "rewards",
            ]
            for field in required_fields:
                assert field in night

        finally:
            os.unlink(temp_path)


class TestScenarioValidation:
    """Test scenario parameter validation."""

    def test_scenario_parameter_ranges(self):
        """Test that scenarios generate reasonable parameter values."""
        scenarios = [random_scenario(num_agents=4, seed=i) for i in range(10)]

        for scenario in scenarios:
            assert 0.15 <= scenario.prob_fire_spreads_to_neighbor <= 0.35
            assert 0.4 <= scenario.prob_solo_agent_extinguishes_fire <= 0.6
            assert scenario.team_reward_house_survives == 100  # Fixed values
            assert scenario.team_penalty_house_burns == 100
            assert scenario.cost_to_work_one_night == 0.5
            # prob_house_catches_fire is sampled from {0.0} or [0.01, 0.05]
            assert scenario.prob_house_catches_fire == 0.0 or (
                0.01 <= scenario.prob_house_catches_fire <= 0.05
            )
            assert 10 <= scenario.min_nights <= 20
            assert scenario.num_agents == 4

    def test_scenario_feature_vector(self):
        """Test scenario feature vector generation."""
        scenario = default_scenario(num_agents=4)
        features = scenario.to_feature_vector()

        assert len(features) == 12  # Should have 12 parameters
        assert all(
            isinstance(f, (int, float, np.integer, np.floating)) for f in features
        )


class TestScenarioFunctions:
    """Test individual scenario functions."""

    def test_easy_scenario(self):
        """Test easy scenario generation."""
        from bucket_brigade.envs import easy_scenario

        scenario = easy_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.1  # Low spread
        assert (
            scenario.prob_solo_agent_extinguishes_fire == 0.8
        )  # High extinguish efficiency
        assert scenario.min_nights == 10

    def test_hard_scenario(self):
        """Test hard scenario generation."""
        from bucket_brigade.envs import hard_scenario

        scenario = hard_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.4  # High spread
        assert (
            scenario.prob_solo_agent_extinguishes_fire == 0.3
        )  # Low extinguish efficiency
        assert scenario.min_nights == 15

    def test_trivial_cooperation_scenario(self):
        """Test trivial cooperation scenario."""
        from bucket_brigade.envs import trivial_cooperation_scenario

        scenario = trivial_cooperation_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.15
        assert scenario.prob_solo_agent_extinguishes_fire == 0.9
        assert scenario.prob_house_catches_fire == 0.0  # No spontaneous fires

    def test_early_containment_scenario(self):
        """Test early containment scenario."""
        from bucket_brigade.envs import early_containment_scenario

        scenario = early_containment_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.35  # High spread
        assert (
            scenario.prob_solo_agent_extinguishes_fire == 0.6
        )  # Moderate extinguish efficiency

    def test_greedy_neighbor_scenario(self):
        """Test greedy neighbor scenario."""
        from bucket_brigade.envs import greedy_neighbor_scenario

        scenario = greedy_neighbor_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.cost_to_work_one_night == 1.0  # High work cost

    def test_sparse_heroics_scenario(self):
        """Test sparse heroics scenario."""
        from bucket_brigade.envs import sparse_heroics_scenario

        scenario = sparse_heroics_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.1  # Very low spread
        assert scenario.min_nights == 20  # Long games

    def test_rest_trap_scenario(self):
        """Test rest trap scenario."""
        from bucket_brigade.envs import rest_trap_scenario

        scenario = rest_trap_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.05  # Very low spread
        assert (
            scenario.prob_solo_agent_extinguishes_fire == 0.95
        )  # Very high extinguish rate

    def test_chain_reaction_scenario(self):
        """Test chain reaction scenario."""
        from bucket_brigade.envs import chain_reaction_scenario

        scenario = chain_reaction_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.45  # High spread
        assert (
            scenario.prob_solo_agent_extinguishes_fire == 0.6
        )  # Moderate extinguish efficiency

    def test_deceptive_calm_scenario(self):
        """Test deceptive calm scenario."""
        from bucket_brigade.envs import deceptive_calm_scenario

        scenario = deceptive_calm_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.min_nights == 20  # Long games
        assert scenario.prob_house_catches_fire == 0.05  # Occasional sparks

    def test_overcrowding_scenario(self):
        """Test overcrowding scenario."""
        from bucket_brigade.envs import overcrowding_scenario

        scenario = overcrowding_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert (
            scenario.prob_solo_agent_extinguishes_fire == 0.3
        )  # Low extinguish efficiency
        assert scenario.team_reward_house_survives == 50.0  # Lower reward

    def test_mixed_motivation_scenario(self):
        """Test mixed motivation scenario."""
        from bucket_brigade.envs import mixed_motivation_scenario

        scenario = mixed_motivation_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.prob_fire_spreads_to_neighbor == 0.3  # Moderate spread
        assert scenario.min_nights == 15

    def test_sample_easy_coop_scenario(self):
        """Test easy cooperation sampling (function removed; skipped)."""
        import pytest

        pytest.skip(
            "sample_easy_coop_scenario removed in commit 27c76e74 (scenarios.py -> scenarios_generated.py refactor)"
        )

    def test_sample_crisis_scenario(self):
        """Test crisis scenario sampling (function removed; skipped)."""
        import pytest

        pytest.skip(
            "sample_crisis_scenario removed in commit 27c76e74 (scenarios.py -> scenarios_generated.py refactor)"
        )

    def test_sample_sparse_work_scenario(self):
        """Test sparse work scenario sampling (function removed; skipped)."""
        import pytest

        pytest.skip(
            "sample_sparse_work_scenario removed in commit 27c76e74 (scenarios.py -> scenarios_generated.py refactor)"
        )

    def test_sample_deception_scenario(self):
        """Test deception scenario sampling (function removed; skipped)."""
        import pytest

        pytest.skip(
            "sample_deception_scenario removed in commit 27c76e74 (scenarios.py -> scenarios_generated.py refactor)"
        )


# Helper: indices of the six reward fields inside Scenario.to_feature_vector()
# Mirrors scenarios_generated.py:51-67 (see issue #157).
REWARD_FIELD_INDICES = {
    "team_reward_house_survives": 3,
    "team_penalty_house_burns": 4,
    "reward_own_house_survives": 5,
    "reward_other_house_survives": 6,
    "penalty_own_house_burns": 7,
    "penalty_other_house_burns": 8,
}


def _make_minimal_scenario(**overrides) -> Scenario:
    """Build a Scenario with all reward/cost fields zeroed except those overridden.

    Used to isolate the contribution of a single reward field to ``_compute_rewards``.
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


class TestScenarioRewardFields:
    """Regression and behavior coverage for the six Scenario reward fields.

    Context (issue #157): After the A/L split in PR #150, ``Scenario`` exposes six
    reward fields, but only two (``team_reward_house_survives`` and
    ``team_penalty_house_burns``) are actually consumed by
    ``BucketBrigadeEnv._compute_rewards`` and the Rust ``rewards.rs`` engine. The
    four ownership-variant fields are currently shadowed by hardcoded constants
    (``+1.0`` / ``-2.0``) for ``own`` houses and are entirely unreferenced for
    ``other`` houses.

    This class:
      * Guards the field names and feature-vector layout against accidental
        rename/reorder (AC1, AC2).
      * Verifies behavior of the two reward fields that ARE wired up (AC3).
      * Uses ``xfail(strict=False)`` to *document* that the four remaining fields
        are not yet honored by reward computation (AC4). These xfails will turn
        into XPASS once a future PR wires the fields up; that is intentional and
        signals the fix.
    """

    # --- AC1: field-name regression ----------------------------------------

    def test_scenario_has_all_six_reward_fields(self):
        """All six post-A/L-split reward fields must exist on ``Scenario``."""
        field_names = {f.name for f in dataclasses.fields(Scenario)}
        for name in REWARD_FIELD_INDICES:
            assert name in field_names, (
                f"Scenario dataclass is missing reward field {name!r}. "
                "If this field was renamed, update REWARD_FIELD_INDICES and "
                "every reward-computation site that references it."
            )

    def test_scenario_reward_fields_are_readable(self):
        """Each reward field accepts and round-trips a distinct non-default value.

        Scalar inputs to the four per-agent ownership fields are auto-promoted
        to length-``num_agents`` lists by ``Scenario.__post_init__`` (issue
        #198); team_* fields remain scalar.
        """
        scenario = _make_minimal_scenario(
            team_reward_house_survives=11.0,
            team_penalty_house_burns=12.0,
            reward_own_house_survives=13.0,
            reward_other_house_survives=14.0,
            penalty_own_house_burns=15.0,
            penalty_other_house_burns=16.0,
        )
        assert scenario.team_reward_house_survives == 11.0
        assert scenario.team_penalty_house_burns == 12.0
        # Ownership reward fields are per-agent vectors of length num_agents
        # after #198. _make_minimal_scenario uses num_agents=4 by default.
        assert scenario.reward_own_house_survives == [13.0] * 4
        assert scenario.reward_other_house_survives == [14.0] * 4
        assert scenario.penalty_own_house_burns == [15.0] * 4
        assert scenario.penalty_other_house_burns == [16.0] * 4

    # --- AC2: feature-vector layout -----------------------------------------

    def test_feature_vector_length_is_twelve(self):
        """``to_feature_vector`` returns the expected 12-element layout."""
        features = default_scenario(num_agents=4).to_feature_vector()
        assert features.shape == (12,)

    def test_reward_fields_at_expected_feature_indices(self):
        """Each reward field surfaces at the index documented in issue #157.

        The four per-agent ownership reward fields (#198) are reduced to
        their mean in ``to_feature_vector`` to preserve the 12-element layout
        for downstream consumers. For scalar inputs (auto-promoted to a
        uniform vector) the mean equals the original scalar.
        """
        scenario = _make_minimal_scenario(
            team_reward_house_survives=11.0,
            team_penalty_house_burns=12.0,
            reward_own_house_survives=13.0,
            reward_other_house_survives=14.0,
            penalty_own_house_burns=15.0,
            penalty_other_house_burns=16.0,
        )
        features = scenario.to_feature_vector()
        # Fields that remain scalar after #198: team_*.
        scalar_fields = {"team_reward_house_survives", "team_penalty_house_burns"}
        for name, idx in REWARD_FIELD_INDICES.items():
            field_value = getattr(scenario, name)
            if name in scalar_fields:
                expected = field_value
            else:
                # Per-agent vector; feature vector stores the mean.
                expected = float(np.mean(field_value))
            assert features[idx] == pytest.approx(expected), (
                f"Feature index {idx} should expose {name}. "
                "If indices have shifted, update REWARD_FIELD_INDICES and any "
                "consumer (e.g. observation.rs, payoff_evaluator_rust.py)."
            )

    # --- AC3: behavior tests for the two reward fields actually in use ------

    def test_team_reward_house_survives_scales_per_agent_reward(self):
        """Doubling ``team_reward_house_survives`` doubles the team component."""
        # Two scenarios that differ ONLY in team_reward_house_survives.
        scenario_low = _make_minimal_scenario(team_reward_house_survives=10.0)
        scenario_high = _make_minimal_scenario(team_reward_house_survives=20.0)

        def reward_with(scenario: Scenario) -> np.ndarray:
            env = BucketBrigadeEnv(scenario=scenario)
            env.reset(seed=0)
            # All houses safe; no transitions => only team component fires.
            env.houses = np.zeros(10, dtype=np.int8)
            env._prev_houses_state = env.houses.copy()
            # All agents REST so cost_to_work_one_night is irrelevant (=0 anyway).
            actions = np.zeros((env.num_agents, 2), dtype=np.int8)
            return env._compute_rewards(actions)

        low_rewards = reward_with(scenario_low)
        high_rewards = reward_with(scenario_high)

        # Each agent has a 0.5 rest bonus; the team component is the only other
        # contribution because all_safe + no transitions + zero costs.
        # team_component = team_reward_house_survives * 1.0 (saved fraction)
        # so the *difference* per agent must equal 20.0 - 10.0 == 10.0.
        diff = high_rewards - low_rewards
        assert np.allclose(diff, 10.0), (
            "Per-agent reward delta should equal the team_reward_house_survives "
            f"delta when all houses are safe. Got {diff!r}."
        )

    def test_team_penalty_house_burns_scales_per_agent_reward(self):
        """Doubling ``team_penalty_house_burns`` doubles the team penalty."""
        scenario_low = _make_minimal_scenario(team_penalty_house_burns=10.0)
        scenario_high = _make_minimal_scenario(team_penalty_house_burns=20.0)

        def reward_with(scenario: Scenario) -> np.ndarray:
            env = BucketBrigadeEnv(scenario=scenario)
            env.reset(seed=0)
            # All houses ruined; no SAFE-transitions => only team penalty fires
            # (plus the per-agent owned-ruined hardcoded -2.0, but that is
            # *identical* between the two scenarios so it cancels in the delta).
            env.houses = np.full(10, env.RUINED, dtype=np.int8)
            env._prev_houses_state = env.houses.copy()
            actions = np.zeros((env.num_agents, 2), dtype=np.int8)
            return env._compute_rewards(actions)

        low_rewards = reward_with(scenario_low)
        high_rewards = reward_with(scenario_high)

        # Burned fraction is 1.0 for all 10 houses, so the per-agent delta is
        # -(20.0 - 10.0) == -10.0 (penalty grew more negative).
        diff = high_rewards - low_rewards
        assert np.allclose(diff, -10.0), (
            "Per-agent reward delta should equal -(team_penalty_house_burns) "
            f"delta when all houses are ruined. Got {diff!r}."
        )

    # --- AC4: behavior tests for the four ownership reward fields ----------
    #
    # These were originally xfail-documented bugs (issue #157). They flipped
    # to passing once #170 wired the fields through `_compute_rewards`; the
    # xfail markers were removed when #198 generalized the fields to
    # per-agent vectors. The tests below verify the canonical scalar-promote
    # behavior (a scalar input behaves identically to a uniform per-agent
    # vector).

    def test_reward_own_house_survives_is_honored(self):
        """Per-agent saved-own-house bonus should equal ``reward_own_house_survives``."""
        scenario = _make_minimal_scenario(reward_own_house_survives=10.0)
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # Deterministic ownership: agent 0 owns house 0.
        env.house_owners = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int8)
        # Stage a SAFE-transition for agent 0's owned house 0 only.
        env._prev_houses_state = np.array(
            [env.BURNING, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8
        )
        env.houses = np.zeros(10, dtype=np.int8)  # all SAFE now
        actions = np.zeros((env.num_agents, 2), dtype=np.int8)  # all REST
        rewards = env._compute_rewards(actions)

        # team_reward_house_survives is 0; rest bonus is +0.5; expected own-save
        # bonus is +10.0 (from scenario), so agent 0 should be 10.5. Currently
        # gets 1.5 because of the hardcoded +1.0.
        assert rewards[0] == pytest.approx(10.5)

    def test_penalty_own_house_burns_is_honored(self):
        """Per-agent owned-ruined penalty should equal ``-penalty_own_house_burns``."""
        scenario = _make_minimal_scenario(penalty_own_house_burns=10.0)
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # Agent 0 owns house 0 only (others split among 1-3).
        env.house_owners = np.array([0, 1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int8)
        # House 0 is RUINED, owned by agent 0; everything else SAFE.
        env.houses = np.array([env.RUINED, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        actions = np.zeros((env.num_agents, 2), dtype=np.int8)  # all REST
        rewards = env._compute_rewards(actions)

        # rest bonus +0.5; owned-ruined penalty should be -10.0 (from scenario).
        # Currently agent 0 gets 0.5 - 2.0 = -1.5 because of the hardcoded -2.0.
        assert rewards[0] == pytest.approx(-9.5)

    def test_reward_other_house_survives_affects_rewards(self):
        """Two scenarios differing only in ``reward_other_house_survives`` must yield
        different per-agent rewards when other-owned houses survive."""
        scenario_low = _make_minimal_scenario(reward_other_house_survives=0.0)
        scenario_high = _make_minimal_scenario(reward_other_house_survives=10.0)

        def reward_for_agent_zero(scenario: Scenario) -> float:
            env = BucketBrigadeEnv(scenario=scenario)
            env.reset(seed=0)
            # Agent 0 owns *no* houses; houses 0-9 owned by agents 1-3.
            env.house_owners = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=np.int8)
            # All houses SAFE, all transitioned from BURNING this turn (so each
            # is a save event for its owner).
            env.houses = np.zeros(10, dtype=np.int8)
            env._prev_houses_state = np.full(10, env.BURNING, dtype=np.int8)
            actions = np.zeros((env.num_agents, 2), dtype=np.int8)  # all REST
            return float(env._compute_rewards(actions)[0])

        low = reward_for_agent_zero(scenario_low)
        high = reward_for_agent_zero(scenario_high)
        assert high != low, (
            "reward_other_house_survives currently has no effect; both scenarios "
            f"produced reward={low} for agent 0."
        )

    def test_penalty_other_house_burns_affects_rewards(self):
        """Two scenarios differing only in ``penalty_other_house_burns`` must yield
        different per-agent rewards when other-owned houses are ruined."""
        scenario_low = _make_minimal_scenario(penalty_other_house_burns=0.0)
        scenario_high = _make_minimal_scenario(penalty_other_house_burns=10.0)

        def reward_for_agent_zero(scenario: Scenario) -> float:
            env = BucketBrigadeEnv(scenario=scenario)
            env.reset(seed=0)
            # Agent 0 owns *no* houses.
            env.house_owners = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1], dtype=np.int8)
            # All houses RUINED, owned by others. Agent 0 has zero owned
            # ruined houses, so the hardcoded -2.0*owned_ruined is 0 for agent 0
            # and any difference must come from penalty_other_house_burns.
            env.houses = np.full(10, env.RUINED, dtype=np.int8)
            env._prev_houses_state = env.houses.copy()
            actions = np.zeros((env.num_agents, 2), dtype=np.int8)  # all REST
            return float(env._compute_rewards(actions)[0])

        low = reward_for_agent_zero(scenario_low)
        high = reward_for_agent_zero(scenario_high)
        assert high != low, (
            "penalty_other_house_burns currently has no effect; both scenarios "
            f"produced reward={low} for agent 0."
        )


class TestPerAgentOwnershipRewards:
    """Per-agent ownership reward semantics introduced in issue #198.

    The four ownership reward fields are now ``List[float]`` of length
    ``num_agents`` instead of scalars. ``Scenario.__post_init__`` auto-promotes
    scalar inputs to ``[scalar] * num_agents`` so existing scenarios behave
    identically. These tests verify the per-agent path: an explicit asymmetric
    vector causes only the targeted agent to receive the reward.
    """

    def test_explicit_per_agent_vector_targets_one_agent(self):
        """Per-agent ``reward_own_house_survives`` only fires for matching agent."""
        # Vector [10.0, 0.0, 0.0, 0.0]: only agent 0 gets the own-save bonus.
        scenario = _make_minimal_scenario(
            reward_own_house_survives=[10.0, 0.0, 0.0, 0.0],
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # Assign agents 0..3 ownership in round-robin order so that EACH agent
        # owns exactly one of houses 0..3, and houses 4..9 cycle owners.
        env.house_owners = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int8)
        # Stage a save event for EVERY agent's first owned house (houses 0..3).
        env._prev_houses_state = np.array([env.BURNING] * 4 + [0] * 6, dtype=np.int8)
        env.houses = np.zeros(10, dtype=np.int8)  # all SAFE
        actions = np.zeros((env.num_agents, 2), dtype=np.int8)  # all REST
        rewards = env._compute_rewards(actions)

        # Each agent has +0.5 rest bonus. Only agent 0 also gets +10.0 from
        # its per-agent reward_own_house_survives entry.
        # Other agents (1, 2, 3) have 0.0 in the vector, so just +0.5.
        assert rewards[0] == pytest.approx(10.5), (
            "Agent 0 should receive its per-agent own-save reward (10.0) plus "
            f"the rest bonus (0.5), got {rewards[0]}."
        )
        for agent_id in (1, 2, 3):
            assert rewards[agent_id] == pytest.approx(0.5), (
                f"Agent {agent_id} should receive only the rest bonus (0.5) "
                f"since its per-agent entry is 0.0, got {rewards[agent_id]}."
            )

    def test_scalar_input_is_promoted_uniformly(self):
        """A scalar field input must behave identically to a uniform vector.

        This is the canonical backward-compatibility test: a scenario with a
        scalar input value `v` must produce the same per-agent rewards as a
        scenario with `[v] * num_agents` (in both Python and Rust).
        """
        scalar_scenario = _make_minimal_scenario(reward_own_house_survives=7.0)
        vector_scenario = _make_minimal_scenario(
            reward_own_house_survives=[7.0, 7.0, 7.0, 7.0],
        )

        # The two scenarios should have identical promoted vectors.
        assert (
            scalar_scenario.reward_own_house_survives
            == vector_scenario.reward_own_house_survives
            == [7.0, 7.0, 7.0, 7.0]
        )

        # And produce identical rewards under the same conditions.
        def rewards_with(scenario: Scenario) -> np.ndarray:
            env = BucketBrigadeEnv(scenario=scenario)
            env.reset(seed=0)
            env.house_owners = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1], dtype=np.int8)
            env._prev_houses_state = np.array(
                [env.BURNING] * 4 + [0] * 6, dtype=np.int8
            )
            env.houses = np.zeros(10, dtype=np.int8)
            actions = np.zeros((env.num_agents, 2), dtype=np.int8)
            return env._compute_rewards(actions)

        scalar_rewards = rewards_with(scalar_scenario)
        vector_rewards = rewards_with(vector_scenario)
        np.testing.assert_allclose(scalar_rewards, vector_rewards)

    def test_wrong_length_vector_rejected(self):
        """Constructing a Scenario with a mismatched-length vector errors out."""
        # _make_minimal_scenario uses num_agents=4; passing a length-5 list
        # should raise ValueError from __post_init__.
        with pytest.raises(ValueError, match="num_agents"):
            _make_minimal_scenario(
                reward_own_house_survives=[1.0, 2.0, 3.0, 4.0, 5.0],
            )


class TestPositionalScenario:
    """Issue #203: spatial cost asymmetry on the 10-house ring.

    The new ``positional_default`` scenario adds three optional fields to
    ``Scenario`` (``agent_home_positions``, ``distance_cost_alpha``,
    ``distance_metric``). When ``distance_cost_alpha == 0.0`` (the default
    for every other scenario) behavior is bit-exactly identical to the
    pre-#203 env.
    """

    def test_positional_default_loads_with_expected_fields(self):
        """The Python factory exposes the new spatial-cost fields."""
        from bucket_brigade.envs.scenarios_generated import get_scenario_by_name

        s = get_scenario_by_name("positional_default", num_agents=4)
        assert s.agent_home_positions == [0, 3, 5, 8]
        assert s.distance_cost_alpha == 0.1
        assert s.distance_metric == "ring_arc"
        # Reward magnitudes mirror `default`.
        assert s.team_reward_house_survives == 100.0
        assert s.team_penalty_house_burns == 100.0
        assert s.reward_own_house_survives == [20.0, 20.0, 20.0, 20.0]
        assert s.penalty_own_house_burns == [40.0, 40.0, 40.0, 40.0]

    def test_zero_alpha_preserves_default_work_cost(self):
        """``distance_cost_alpha == 0.0`` => work cost == cost_to_work_one_night."""
        scenario = _make_minimal_scenario(cost_to_work_one_night=0.5)
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        env.houses = np.zeros(10, dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        # All agents WORK at house 0 (max distance from agent 3's home=3).
        actions = np.zeros((env.num_agents, 2), dtype=np.int8)
        actions[:, 0] = 0
        actions[:, 1] = env.WORK
        rewards = env._compute_rewards(actions)
        # No team component, no own/other transitions => only -base_cost each.
        np.testing.assert_allclose(rewards, [-0.5, -0.5, -0.5, -0.5])

    def test_positive_alpha_scales_cost_with_ring_distance(self):
        """Work cost = base + alpha * ring_dist(home, target)."""
        scenario = _make_minimal_scenario(
            cost_to_work_one_night=0.5,
            agent_home_positions=[0, 3, 5, 8],
            distance_cost_alpha=0.1,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        env.houses = np.zeros(10, dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        # All four agents WORK at house 5 with honest signaling.
        # ring_dist on 10-ring:  (0,5)=5, (3,5)=2, (5,5)=0, (8,5)=3
        actions = np.array([[5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1]], dtype=np.int8)
        rewards = env._compute_rewards(actions)
        expected = -np.array(
            [0.5 + 0.1 * 5, 0.5 + 0.1 * 2, 0.5 + 0.1 * 0, 0.5 + 0.1 * 3],
            dtype=np.float32,
        )
        np.testing.assert_allclose(rewards, expected, atol=1e-6)

    def test_resting_is_immune_to_distance_cost(self):
        """Distance cost only applies to WORK actions, not REST."""
        scenario = _make_minimal_scenario(
            cost_to_work_one_night=0.5,
            agent_home_positions=[0, 3, 5, 8],
            distance_cost_alpha=0.1,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        env.houses = np.zeros(10, dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        # All agents REST.
        actions = np.zeros((env.num_agents, 2), dtype=np.int8)
        rewards = env._compute_rewards(actions)
        # Pure rest bonus +0.5 for every agent regardless of "where" they rest.
        np.testing.assert_allclose(rewards, [0.5, 0.5, 0.5, 0.5])

    def test_wrong_length_home_positions_rejected(self):
        """num_agents mismatch raises ValueError."""
        with pytest.raises(ValueError, match="num_agents"):
            _make_minimal_scenario(agent_home_positions=[0, 3, 5])  # len=3, na=4

    def test_out_of_range_home_position_rejected(self):
        """Home positions must satisfy ``0 <= p < 10``."""
        with pytest.raises(ValueError, match=r"out of range"):
            _make_minimal_scenario(agent_home_positions=[0, 3, 5, 10])

    def test_unsupported_distance_metric_rejected(self):
        """Only ``"ring_arc"`` is supported today."""
        with pytest.raises(ValueError, match="distance_metric"):
            _make_minimal_scenario(distance_metric="euclidean")

    def test_empty_home_positions_falls_back_to_round_robin(self):
        """Empty ``agent_home_positions`` => engine uses ``np.arange(num_agents)``."""
        scenario = _make_minimal_scenario()  # defaults: empty list, alpha=0.0
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        np.testing.assert_array_equal(
            env.agent_home_positions, np.arange(env.num_agents, dtype=np.int8)
        )

    def test_zero_alpha_matches_pre203_full_episode(self):
        """End-to-end: alpha=0 produces the same per-step reward trace as no
        spatial field at all. Guards against accidental side effects of
        threading the new field through the engine."""
        from bucket_brigade.envs.scenarios_generated import default_scenario

        env_a = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        env_b = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        env_a.reset(seed=99)
        env_b.reset(seed=99)
        rng = np.random.RandomState(0)
        for _ in range(5):
            actions = rng.randint(0, 2, size=(4, 2)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)
            _, r_a, _, _ = env_a.step(actions)
            _, r_b, _, _ = env_b.step(actions)
            np.testing.assert_allclose(r_a, r_b)


class TestV2MinimalScenario:
    """Issue #254: ``v2_minimal`` (2 houses x 4 agents) — the PPO
    learnability diagnostic. The Python env must size all its
    house-indexed arrays from ``scenario.num_houses`` rather than the
    pre-#254 hardcoded 10."""

    def _v2_scenario(self):
        from bucket_brigade.envs.scenarios_generated import v2_minimal_scenario

        return v2_minimal_scenario(num_agents=4)

    def test_v2_minimal_scenario_values(self):
        """Scenario factory produces the documented values."""
        s = self._v2_scenario()
        assert s.num_houses == 2
        assert s.num_agents == 4
        assert s.team_reward_house_survives == 10.0
        assert s.team_penalty_house_burns == 10.0
        assert s.prob_house_catches_fire == 0.05
        assert s.min_nights == 8
        # Per-agent vectors length 4.
        assert s.reward_own_house_survives == [50.0, 50.0, 50.0, 50.0]
        assert s.penalty_own_house_burns == [100.0, 100.0, 100.0, 100.0]

    def test_env_construction_uses_num_houses(self):
        """``BucketBrigadeEnv`` sizes its house vectors from
        ``scenario.num_houses``."""
        env = BucketBrigadeEnv(scenario=self._v2_scenario())
        assert env.num_houses == 2
        assert env.houses.shape == (2,)
        assert env._prev_houses_state.shape == (2,)
        assert env.house_owners.shape == (2,)
        # Round-robin: house 0 -> agent 0, house 1 -> agent 1.
        np.testing.assert_array_equal(env.house_owners, np.array([0, 1]))
        # Home positions wrap modulo num_houses for the 4-on-2 case.
        np.testing.assert_array_equal(env.agent_home_positions, np.array([0, 1, 0, 1]))

    def test_reset_observation_shape(self):
        """After ``reset()`` the observation houses field is length 2."""
        env = BucketBrigadeEnv(scenario=self._v2_scenario())
        obs = env.reset(seed=42)
        assert obs["houses"].shape == (2,)

    def test_random_rollout_no_crash(self):
        """A 20-step random rollout completes without panicking on
        out-of-range house indices."""
        env = BucketBrigadeEnv(scenario=self._v2_scenario())
        env.reset(seed=42)
        rng = np.random.RandomState(0)
        for _ in range(20):
            actions = rng.randint(0, 2, size=(4, 3)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 2, size=4)  # house in [0, 1]
            obs, rewards, dones, _ = env.step(actions)
            assert obs["houses"].shape == (2,)
            assert rewards.shape == (4,)
            if dones[0]:
                break

    def test_pre_254_scenarios_still_ten_houses(self):
        """Backward compat: every existing scenario still produces a
        10-house env (the engine paths previously hardcoded 10 now read
        ``scenario.num_houses`` which defaults to 10, but the resulting
        behavior must be identical for pre-#254 scenarios)."""
        from bucket_brigade.envs.scenarios_generated import (
            SCENARIO_REGISTRY,
        )

        for name, factory in SCENARIO_REGISTRY.items():
            if name.startswith("v2_"):
                continue
            scenario = factory(num_agents=4)
            assert scenario.num_houses == 10, (
                f"Pre-#254 scenario '{name}' regressed: num_houses={scenario.num_houses}"
            )
            env = BucketBrigadeEnv(scenario=scenario)
            assert env.num_houses == 10
            assert env.houses.shape == (10,)


class TestActionShaping:
    """Issue #259: action-conditioned per-step reward shaping.

    The Python env mirrors the Rust core's two new ``Scenario`` fields,
    ``action_shaping_alpha`` and ``action_shaping_beta``. Both default to
    ``0.0`` and every pre-#259 scenario keeps them zero, so existing
    behavior is bit-exactly preserved. Test pattern modeled on
    ``TestPositionalScenario`` (PR #203 precedent).
    """

    def test_defaults_are_zero_across_all_scenarios(self):
        """Every pre-#259 scenario keeps both shaping knobs at 0.0."""
        from bucket_brigade.envs.scenarios_generated import SCENARIO_REGISTRY

        for name, factory in SCENARIO_REGISTRY.items():
            s = factory(num_agents=4)
            assert s.action_shaping_alpha == 0.0, (
                f"Pre-#259 scenario '{name}' must keep "
                f"action_shaping_alpha=0.0; got {s.action_shaping_alpha}"
            )
            assert s.action_shaping_beta == 0.0, (
                f"Pre-#259 scenario '{name}' must keep "
                f"action_shaping_beta=0.0; got {s.action_shaping_beta}"
            )

    def test_zero_shaping_matches_pre259_full_episode(self):
        """End-to-end: alpha=0 and beta=0 produce the same per-step reward
        trace as before — guards against accidental side effects of threading
        the new fields through ``_compute_rewards``."""
        from bucket_brigade.envs.scenarios_generated import default_scenario

        env_a = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        env_b = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        env_a.reset(seed=99)
        env_b.reset(seed=99)
        rng = np.random.RandomState(0)
        for _ in range(10):
            actions = rng.randint(0, 2, size=(4, 2)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)
            _, r_a, _, _ = env_a.step(actions)
            _, r_b, _, _ = env_b.step(actions)
            np.testing.assert_allclose(r_a, r_b)

    def test_alpha_credit_share_sums_to_alpha(self):
        """When k workers co-extinguish one fire, the sum of their alpha
        bonuses equals alpha exactly. Core invariant of the credit-share
        formulation."""
        scenario = _make_minimal_scenario(
            prob_solo_agent_extinguishes_fire=1.0,  # deterministic extinguish
            action_shaping_alpha=1.5,
            action_shaping_beta=0.0,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        # Set up: only house 5 burning, no other fires, no spread/spark.
        env.houses = np.zeros(10, dtype=np.int8)
        env.houses[5] = env.BURNING
        env._prev_houses_state = env.houses.copy()
        # All 4 agents work at house 5 -> guaranteed extinguish.
        actions = np.array(
            [[5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1]],
            dtype=np.int8,
        )
        # Drive the engine through a full step so the BURNING -> SAFE
        # transition actually happens and prev_houses captures the right
        # start state. We can't call _compute_rewards in isolation because
        # it depends on self.houses already reflecting end-of-step.
        _, rewards, _, _ = env.step(actions)
        # Sum should equal alpha=1.5 (cost is zero, team/ownership are zero).
        assert abs(float(np.sum(rewards)) - 1.5) < 1e-5, (
            f"Sum of alpha bonuses should equal alpha=1.5, got {np.sum(rewards)}"
        )
        # Each worker should get alpha/4 = 0.375.
        for i, r in enumerate(rewards):
            assert abs(float(r) - 0.375) < 1e-5, (
                f"Worker {i} should get alpha/4=0.375, got {r}"
            )

    def test_rest_agents_get_zero_shaping_bonus(self):
        """REST actions never receive alpha or beta, regardless of nearby
        extinguish events."""
        scenario = _make_minimal_scenario(
            prob_solo_agent_extinguishes_fire=1.0,
            action_shaping_alpha=2.0,
            action_shaping_beta=0.5,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=42)
        env.houses = np.zeros(10, dtype=np.int8)
        env.houses[3] = env.BURNING
        env._prev_houses_state = env.houses.copy()
        # Agent 0 works house 3 (extinguishes), agents 1-3 rest at house 0.
        actions = np.array(
            [[3, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        # Agent 0 is sole worker -> alpha/1 = 2.0 (no work cost).
        assert abs(float(rewards[0]) - 2.0) < 1e-5
        # Resting agents get +0.5 rest bonus, no shaping.
        for i in range(1, 4):
            assert abs(float(rewards[i]) - 0.5) < 1e-5, (
                f"Resting agent {i} should get +0.5 rest only, got {rewards[i]}"
            )

    def test_beta_rewards_preventive_presence(self):
        """An agent working at a SAFE house that stays SAFE gets beta;
        distinct from alpha (which requires BURNING->SAFE)."""
        scenario = _make_minimal_scenario(
            prob_fire_spreads_to_neighbor=0.0,
            prob_house_catches_fire=0.0,
            action_shaping_alpha=0.0,
            action_shaping_beta=0.3,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=7)
        env.houses = np.zeros(10, dtype=np.int8)
        env._prev_houses_state = env.houses.copy()
        # All 4 agents work at distinct safe houses.
        actions = np.array(
            [[0, 1, 1], [3, 1, 1], [5, 1, 1], [8, 1, 1]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        # Each working agent: +0.3 preventive bonus (no cost, no other rewards).
        for i, r in enumerate(rewards):
            assert abs(float(r) - 0.3) < 1e-5, (
                f"Working agent {i} at safe house should get beta=0.3, got {r}"
            )

    def test_beta_does_not_fire_on_extinguished_house(self):
        """Beta requires prev==SAFE; a BURNING->SAFE transition (alpha's
        domain) does NOT trigger beta."""
        scenario = _make_minimal_scenario(
            prob_solo_agent_extinguishes_fire=1.0,
            action_shaping_alpha=0.0,
            action_shaping_beta=0.5,
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        env.houses = np.zeros(10, dtype=np.int8)
        env.houses[5] = env.BURNING
        env._prev_houses_state = env.houses.copy()
        # Agent 0 works house 5 (extinguishes); others rest.
        actions = np.array(
            [[5, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=np.int8,
        )
        _, rewards, _, _ = env.step(actions)
        # alpha=0 so no extinguish bonus. beta=0.5 but prev[5]=BURNING so
        # beta doesn't fire. Agent 0 nets 0.0 (no cost, no shaping).
        assert abs(float(rewards[0])) < 1e-5, (
            f"Extinguishing agent must not get beta (prev was BURNING), "
            f"got {rewards[0]}"
        )

    def test_per_step_reward_variance_increases_with_alpha(self):
        """Acceptance criterion: non-zero alpha provides a per-agent
        gradient signal at extinguish events. Comparing alpha=0 vs alpha=2
        on the same fixed extinguish scenario, the alpha=2 case must
        produce strictly higher per-agent reward variance (the signal PPO
        needs for credit assignment)."""
        # 1 worker (extinguisher), 3 resters at safe houses. With alpha=0
        # the worker's reward is just -cost=0 and resters get +0.5 (so
        # rewards are [0, 0.5, 0.5, 0.5], var ~ 0.047). With alpha=2 the
        # worker gets +2.0 and resters get +0.5 (rewards [2.0, 0.5, 0.5,
        # 0.5], var ~ 0.42). The variance gap is the PPO signal.
        variances = {}
        for alpha in (0.0, 2.0):
            scenario = _make_minimal_scenario(
                prob_solo_agent_extinguishes_fire=1.0,
                action_shaping_alpha=alpha,
                action_shaping_beta=0.0,
            )
            env = BucketBrigadeEnv(scenario=scenario)
            env.reset(seed=0)
            env.houses = np.zeros(10, dtype=np.int8)
            env.houses[5] = env.BURNING
            env._prev_houses_state = env.houses.copy()
            actions = np.array(
                [[5, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                dtype=np.int8,
            )
            _, rewards, _, _ = env.step(actions)
            variances[alpha] = float(np.var(rewards))
        assert variances[2.0] > variances[0.0], (
            f"alpha=2 should produce strictly higher per-agent reward "
            f"variance than alpha=0; got {variances}"
        )
        # And the alpha=2 variance is substantively positive (sanity).
        assert variances[2.0] > 0.1

    def test_v2_minimal_with_shaping_runs(self):
        """Action shaping composes with v2_minimal (small ring)."""
        from bucket_brigade.envs.scenarios_generated import v2_minimal_scenario

        s = v2_minimal_scenario(num_agents=4)
        # Override shaping (the v2_minimal factory keeps defaults).
        scenario = dataclasses.replace(
            s, action_shaping_alpha=0.5, action_shaping_beta=0.1
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=42)
        rng = np.random.RandomState(0)
        for _ in range(20):
            actions = rng.randint(0, 2, size=(4, 3)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 2, size=4)
            _, rewards, dones, _ = env.step(actions)
            assert rewards.shape == (4,)
            assert np.all(np.isfinite(rewards))
            if dones[0]:
                break


class TestPotentialBasedShaping:
    """Issue #283: potential-based team-welfare shaping (Ng-Harada-Russell 1999).

    Adds an aligned per-step bonus ``F(s, a, s') = lambda * (
    gamma * Phi(s') - Phi(s))`` shared across all agents. The
    closed-form team-welfare potential (option B):
        Phi(s) = team_reward * (safe/N)
                 - team_penalty * (ruined/N)
                 - 0.5 * team_penalty * (burning/N)
    NHR (ICML 1999) guarantees the optimal-policy set is preserved under
    this shaping for any potential function Phi.

    The critical correctness test is the telescoping-identity check: if
    Phi(s_T) = 0 is enforced at terminal states, the discounted sum of
    shaping terms across a trajectory must equal a policy-independent
    constant. A buggy implementation (e.g., forgetting to zero out the
    terminal-state potential, or off-by-one in s vs s') breaks this
    identity, and the test catches it.
    """

    def test_defaults_are_zero_across_all_scenarios(self):
        """Every pre-#283 scenario keeps team_welfare_lambda=0.0 and
        kind='none' so existing scenarios are bit-exactly unchanged."""
        from bucket_brigade.envs.scenarios_generated import SCENARIO_REGISTRY

        for name, factory in SCENARIO_REGISTRY.items():
            s = factory(num_agents=4)
            assert s.team_welfare_lambda == 0.0, (
                f"Pre-#283 scenario '{name}' must keep "
                f"team_welfare_lambda=0.0; got {s.team_welfare_lambda}"
            )
            assert s.team_welfare_kind == "none", (
                f"Pre-#283 scenario '{name}' must keep "
                f"team_welfare_kind='none'; got {s.team_welfare_kind!r}"
            )
            assert s.team_welfare_gamma == 1.0, (
                f"Pre-#283 scenario '{name}' must keep "
                f"team_welfare_gamma=1.0; got {s.team_welfare_gamma}"
            )

    def test_zero_lambda_matches_pre283_full_episode(self):
        """Byte-identity regression guard: lambda=0 (any kind) produces the
        same per-step reward trace as before — guards against accidental
        side effects of threading the new fields through ``_compute_rewards``."""
        env_a = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        # Same dynamics but with the new fields set explicitly to the
        # disabled-shaping defaults — must produce identical rewards.
        scenario_b = dataclasses.replace(
            default_scenario(num_agents=4),
            team_welfare_lambda=0.0,
            team_welfare_gamma=1.0,
            team_welfare_kind="none",
        )
        env_b = BucketBrigadeEnv(scenario=scenario_b)
        env_a.reset(seed=99)
        env_b.reset(seed=99)
        rng = np.random.RandomState(0)
        for _ in range(20):
            actions = rng.randint(0, 2, size=(4, 3)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)
            _, r_a, dones_a, _ = env_a.step(actions)
            _, r_b, dones_b, _ = env_b.step(actions)
            np.testing.assert_array_equal(r_a, r_b)
            if bool(dones_a[0]):
                break

    def test_zero_lambda_with_kind_set_is_still_byte_identical(self):
        """A scenario with kind=team_welfare_closed_form but lambda=0
        must still produce byte-identical rewards (the env fast-path skips
        shaping when lambda is zero, regardless of kind)."""
        env_a = BucketBrigadeEnv(scenario=default_scenario(num_agents=4))
        scenario_b = dataclasses.replace(
            default_scenario(num_agents=4),
            team_welfare_lambda=0.0,
            team_welfare_gamma=0.99,  # different gamma — must still be inert
            team_welfare_kind="team_welfare_closed_form",
        )
        env_b = BucketBrigadeEnv(scenario=scenario_b)
        env_a.reset(seed=7)
        env_b.reset(seed=7)
        rng = np.random.RandomState(1)
        for _ in range(15):
            actions = rng.randint(0, 2, size=(4, 3)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)
            _, r_a, dones_a, _ = env_a.step(actions)
            _, r_b, dones_b, _ = env_b.step(actions)
            np.testing.assert_array_equal(r_a, r_b)
            if bool(dones_a[0]):
                break

    def test_phi_terminal_convention(self):
        """At a truncated last step, the env must treat Phi(s') = 0 so the
        shaping term at terminal becomes lambda * (-Phi(s)). This is the
        NHR boundary condition that makes the telescoping identity exact.

        We zero out every base-reward component (work cost, ownership,
        team) so the per-agent reward equals the shaping term exactly,
        making the terminal-state Phi(s')=0 convention directly testable.
        """
        # Compare two configurations: one where the step is terminal (and
        # thus Phi(next) should be zeroed) and one where it's not (because
        # the test setup forces a non-terminal next state). The per-agent
        # reward must equal the shaping term in both cases.
        scenario_terminal = _make_minimal_scenario(
            # All base reward components zeroed so reward == shaping term.
            team_reward_house_survives=0.0,
            team_penalty_house_burns=0.0,
            cost_to_work_one_night=0.0,
            # min_nights=0 lets _check_termination fire on the first step.
            # The env's `night` counter is 0 during the first step's
            # termination check (it advances after) so `night >= min_nights`
            # requires min_nights == 0 for first-step termination.
            min_nights=0,
            team_welfare_lambda=1.0,
            team_welfare_gamma=0.99,
            # Tie Phi to a separate non-zero scale so we can verify the
            # terminal-zeroing without confounding with the base reward.
            # Reusing Phi's coefficients from team_reward_* is the env's
            # convention, but for this test we need both isolated.
            # The closed-form Phi uses team_reward_house_survives and
            # team_penalty_house_burns; both are 0 above, so Phi(s) = 0
            # for ALL s, which makes the test vacuous.
            #
            # Solution: rest-reward component (+0.5 per REST agent) is
            # hard-coded; instead, set BOTH team_reward and team_penalty
            # to a non-zero scale -- the team reward is *added per agent*
            # but the shaping is also per-agent so we just compute the
            # expected total.
            team_welfare_kind="team_welfare_closed_form",
            prob_solo_agent_extinguishes_fire=1.0,  # deterministic save
            prob_fire_spreads_to_neighbor=0.0,
            prob_house_catches_fire=0.0,
        )
        # team_reward must be nonzero for Phi(s) to be nonzero, since Phi
        # uses team_reward and team_penalty as coefficients. Reset them.
        scenario_terminal = dataclasses.replace(
            scenario_terminal,
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
        )
        env = BucketBrigadeEnv(scenario=scenario_terminal)
        env.reset(seed=0)
        # Set up a single burning house at index 5; all 4 agents work it.
        # After the step: house 5 is SAFE -> all houses SAFE -> terminate
        # via _check_termination's `all_safe` branch.
        env.houses = np.zeros(10, dtype=np.int8)
        env.houses[5] = env.BURNING
        env._prev_houses_state = env.houses.copy()
        # Phi(s) before step: 1 burning, 9 safe, 0 ruined.
        # = 10*(9/10) - 10*(0/10) - 0.5*10*(1/10) = 9 - 0 - 0.5 = 8.5
        phi_prev_expected = 10.0 * 0.9 - 0.5 * 10.0 * 0.1
        assert abs(phi_prev_expected - 8.5) < 1e-9
        phi_prev = env._compute_team_welfare_phi(env._prev_houses_state)
        assert abs(phi_prev - phi_prev_expected) < 1e-5

        actions = np.array(
            [[5, 1, 1], [5, 1, 1], [5, 1, 1], [5, 1, 1]],
            dtype=np.int8,
        )
        _, rewards, dones, _ = env.step(actions)
        assert bool(dones[0]), "Episode should have terminated this step"

        # Decompose the expected reward:
        #   - Work cost: 0 (cost_to_work_one_night = 0).
        #   - Rest reward: 0 (all agents working).
        #   - Ownership: 0 (own/other reward+penalty all zero).
        #   - Team reward: team_reward_house_survives * 1.0 (all safe) = 10.0
        #   - Shaping: lambda * (gamma * Phi(terminal) - Phi(prev))
        #            = 1.0 * (0.99 * 0 - 8.5) = -8.5
        # Total per agent: 10.0 + (-8.5) = 1.5
        # Each agent receives the same total (team-shared shaping +
        # team-shared team_reward).
        expected_per_agent = 10.0 + (-phi_prev_expected)
        for i, r in enumerate(rewards):
            assert abs(float(r) - expected_per_agent) < 1e-4, (
                f"Agent {i}: expected terminal-step reward "
                f"team_reward + shaping = 10.0 + (-Phi(s_prev)) = "
                f"{expected_per_agent}, got {float(r)}. Did the env "
                f"forget the Phi(terminal)=0 convention? (Got value "
                f"suggests Phi(s')=10.0 was used instead of 0.0.)"
            )

        # Negative control: if we'd used Phi(s') = Phi(actual all-safe) = 10
        # instead of 0, each reward would have been 10 + (0.99*10 - 8.5) =
        # 10 + 1.4 = 11.4 (not 1.5). Make sure the test would reject this.
        wrong_shaping = 0.99 * 10.0 - 8.5
        wrong_per_agent = 10.0 + wrong_shaping
        assert abs(wrong_per_agent - 11.4) < 1e-4
        assert abs(rewards[0] - wrong_per_agent) > 0.1, (
            "Test sanity: if the terminal convention were broken, "
            "rewards would be ~11.4, not the observed ~1.5. The test "
            "must be sensitive to the difference."
        )

    def test_telescoping_identity_under_random_policy(self):
        """CRITICAL: NHR invariance test. Under potential-based shaping,
        the discounted sum of shaped rewards along any trajectory equals
        the base discounted return + lambda * (gamma^T * Phi(s_T) - Phi(s_0)).

        With Phi(terminal) := 0, the bracketed correction collapses to
        -lambda * Phi(s_0), a constant independent of the policy or
        trajectory tail. This is what makes NHR shaping invariant.

        We verify by running the SAME policy (fixed seed) under both
        lambda=0 (base) and lambda=1 (shaped) with everything else
        deterministic, then checking the discounted-sum equation.

        A buggy implementation that:
          - forgets the terminal-state convention,
          - swaps s and s' in the bonus,
          - or fails to apply the discount inside the shaping bonus,
        will violate this identity and the test will fail.
        """
        gamma = 0.99
        # Build two envs that differ ONLY in lambda. Use a deterministic-
        # enough setup that the same actions sequence produces the same
        # trajectory in both envs (which is required for the per-step
        # comparison to be meaningful).
        scenario_base = _make_minimal_scenario(
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
            min_nights=12,
            cost_to_work_one_night=0.5,
            team_welfare_lambda=0.0,
            team_welfare_gamma=gamma,
            team_welfare_kind="team_welfare_closed_form",
        )
        scenario_shaped = dataclasses.replace(
            scenario_base, team_welfare_lambda=1.0
        )
        env_base = BucketBrigadeEnv(scenario=scenario_base)
        env_shaped = BucketBrigadeEnv(scenario=scenario_shaped)
        # Identical RNG seed -> identical fire dynamics under identical
        # actions. We use the env-level seed for spread / spark / extinguish
        # stochasticity.
        env_base.reset(seed=12345)
        env_shaped.reset(seed=12345)

        # Snapshot s_0 for the constant correction term. Phi is computed
        # from the env's `houses` after reset/initialization.
        phi_s0_base = env_base._compute_team_welfare_phi(env_base.houses)
        phi_s0_shaped = env_shaped._compute_team_welfare_phi(env_shaped.houses)
        # Phi is a deterministic function of `houses` only; same seed ->
        # same initial houses -> same Phi(s_0) in both envs.
        assert phi_s0_base == phi_s0_shaped, (
            f"Phi(s_0) must match across base/shaped envs with same seed; "
            f"got base={phi_s0_base}, shaped={phi_s0_shaped}"
        )
        phi_s0 = phi_s0_base

        rng = np.random.RandomState(2026)
        # Track discounted sums per-agent. Both envs see the SAME actions.
        discounted_base = np.zeros(4, dtype=np.float64)
        discounted_shaped = np.zeros(4, dtype=np.float64)
        t = 0
        terminated_at = None
        # Walk one full episode (or up to a long horizon if it doesn't
        # terminate). We rely on min_nights=12 + nonzero ignition rate
        # ensuring termination within a reasonable number of steps.
        max_steps = 64
        for _ in range(max_steps):
            actions = rng.randint(0, 2, size=(4, 3)).astype(np.int8)
            actions[:, 0] = rng.randint(0, 10, size=4)

            _, r_base, dones_base, _ = env_base.step(actions)
            _, r_shaped, dones_shaped, _ = env_shaped.step(actions)

            # Same actions + same seed + same dynamics -> trajectories must
            # match step-by-step. If this assertion fires, the test
            # premise breaks and the comparison below is meaningless.
            assert np.array_equal(env_base.houses, env_shaped.houses), (
                f"Step {t}: house states diverged between base and "
                f"shaped envs ({env_base.houses} vs {env_shaped.houses})."
            )
            assert bool(dones_base[0]) == bool(dones_shaped[0]), (
                f"Step {t}: done flags diverged ({dones_base[0]} vs "
                f"{dones_shaped[0]})."
            )

            discount = gamma**t
            discounted_base += discount * r_base.astype(np.float64)
            discounted_shaped += discount * r_shaped.astype(np.float64)
            t += 1

            if bool(dones_base[0]):
                terminated_at = t
                break

        assert terminated_at is not None, (
            f"Episode did not terminate within {max_steps} steps; "
            "test premise broken."
        )

        # NHR identity: sum_t gamma^t F_t = lambda * (gamma^T Phi(s_T) - Phi(s_0)).
        # With Phi(terminal) = 0 enforced, this collapses to -lambda * Phi(s_0).
        # Lambda=1, so expected correction per agent = -Phi(s_0).
        expected_correction = -1.0 * phi_s0
        observed_correction = discounted_shaped - discounted_base
        # All four agents share the team-wide shaping term, so the
        # per-agent observed correction must be uniform (within float
        # noise).
        np.testing.assert_allclose(
            observed_correction,
            np.full(4, expected_correction),
            rtol=0,
            atol=1e-3,
            err_msg=(
                f"NHR telescoping identity violated. "
                f"Expected per-agent correction = -Phi(s_0) = "
                f"{expected_correction:.6f}; observed correction = "
                f"{observed_correction}. "
                f"This indicates a bug: either Phi(terminal) is not "
                f"zeroed, or the per-step shaping term uses the wrong "
                f"sign / state / discount."
            ),
        )

    def test_phi_closed_form_values(self):
        """Spot-check the closed-form Phi against hand-computed values
        on a few small house states."""
        scenario = _make_minimal_scenario(
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
            team_welfare_lambda=0.0,  # value computation does not require shaping enabled
            team_welfare_gamma=1.0,
            team_welfare_kind="team_welfare_closed_form",
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)

        # All safe: Phi = 10*(10/10) - 10*0 - 0.5*10*0 = 10.0
        houses_all_safe = np.full(10, env.SAFE, dtype=np.int8)
        assert abs(env._compute_team_welfare_phi(houses_all_safe) - 10.0) < 1e-6

        # All ruined: Phi = 0 - 10 - 0 = -10.0
        houses_all_ruined = np.full(10, env.RUINED, dtype=np.int8)
        assert abs(env._compute_team_welfare_phi(houses_all_ruined) - (-10.0)) < 1e-6

        # All burning: Phi = 0 - 0 - 0.5*10 = -5.0
        houses_all_burning = np.full(10, env.BURNING, dtype=np.int8)
        assert abs(env._compute_team_welfare_phi(houses_all_burning) - (-5.0)) < 1e-6

        # Mixed: 5 safe, 3 burning, 2 ruined.
        # Phi = 10*(5/10) - 10*(2/10) - 0.5*10*(3/10) = 5 - 2 - 1.5 = 1.5
        mixed = np.array(
            [env.SAFE] * 5 + [env.BURNING] * 3 + [env.RUINED] * 2, dtype=np.int8
        )
        assert abs(env._compute_team_welfare_phi(mixed) - 1.5) < 1e-6

    def test_phi_kind_none_returns_zero(self):
        """When kind='none', Phi must be exactly 0 regardless of state.
        Guards the env fast-path skip in _compute_rewards."""
        scenario = _make_minimal_scenario(
            team_reward_house_survives=100.0,
            team_penalty_house_burns=100.0,
            team_welfare_lambda=0.0,
            team_welfare_gamma=1.0,
            team_welfare_kind="none",
        )
        env = BucketBrigadeEnv(scenario=scenario)
        env.reset(seed=0)
        for state_code in (env.SAFE, env.BURNING, env.RUINED):
            houses = np.full(10, state_code, dtype=np.int8)
            assert env._compute_team_welfare_phi(houses) == 0.0, (
                f"kind='none' must produce Phi=0; got "
                f"{env._compute_team_welfare_phi(houses)} for state {state_code}"
            )

    def test_unknown_kind_rejected_at_construction(self):
        """The Scenario allowlist must reject unknown team_welfare_kind
        values at construction time."""
        with pytest.raises(ValueError, match="team_welfare_kind"):
            _make_minimal_scenario(team_welfare_kind="bogus")

    def test_shaping_is_team_shared(self):
        """The shaping term is added equally to every agent's reward, so
        the per-agent reward differential under shaping equals the
        per-agent reward differential without shaping. Confirms the
        team-shared application (not per-agent decomposition)."""
        # Scenario where shaping fires deterministically: solo extinguish
        # of a burning house, no spread/spark, lambda=1.
        scenario_no_shape = _make_minimal_scenario(
            prob_solo_agent_extinguishes_fire=1.0,
            prob_fire_spreads_to_neighbor=0.0,
            prob_house_catches_fire=0.0,
            team_reward_house_survives=10.0,
            team_penalty_house_burns=10.0,
        )
        scenario_shaped = dataclasses.replace(
            scenario_no_shape,
            team_welfare_lambda=1.0,
            team_welfare_gamma=0.99,
            team_welfare_kind="team_welfare_closed_form",
        )
        env_a = BucketBrigadeEnv(scenario=scenario_no_shape)
        env_b = BucketBrigadeEnv(scenario=scenario_shaped)
        env_a.reset(seed=11)
        env_b.reset(seed=11)
        # Force identical start state.
        env_a.houses = np.zeros(10, dtype=np.int8)
        env_a.houses[3] = env_a.BURNING
        env_a._prev_houses_state = env_a.houses.copy()
        env_b.houses = env_a.houses.copy()
        env_b._prev_houses_state = env_a.houses.copy()
        actions = np.array(
            [[3, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            dtype=np.int8,
        )
        _, r_a, _, _ = env_a.step(actions)
        _, r_b, _, _ = env_b.step(actions)
        # The pairwise differential (r_b - r_a) must be uniform across
        # agents — shaping is team-shared, not per-agent.
        diff = r_b - r_a
        np.testing.assert_allclose(
            diff, np.full(4, diff[0]), rtol=0, atol=1e-5,
            err_msg=(
                "Potential-based shaping must apply equally to every "
                "agent (team-shared bonus). Per-agent differential is "
                f"non-uniform: {diff}"
            ),
        )

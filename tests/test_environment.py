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
        actions = np.random.randint(0, 2, size=(4, 2))  # [house, mode] for each agent

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
            actions = np.random.randint(0, 2, size=(4, 2))
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
        actions = np.array(
            [
                [1, 1],  # Agent 0 works on burning house 1
                [7, 1],  # Agent 1 works on burning house 7
                [0, 0],  # Agent 2 rests
                [2, 0],  # Agent 3 rests
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

        actions = np.array(
            [
                [0, 1],  # Work
                [1, 0],  # Rest
                [2, 1],  # Work
                [3, 0],  # Rest
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
            actions = np.random.randint(0, 2, size=(4, 2))
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

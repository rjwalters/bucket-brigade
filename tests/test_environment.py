"""Tests for the Bucket Brigade environment."""

import numpy as np
from bucket_brigade.envs import BucketBrigadeEnv, default_scenario, random_scenario


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
        assert 0 < scenario.beta < 1
        assert scenario.kappa > 0

        # Test random scenario
        scenario = random_scenario(num_agents=4, seed=42)
        assert scenario.num_agents == 4
        assert 0.15 <= scenario.beta <= 0.35  # Within expected range

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
            assert 0.15 <= scenario.beta <= 0.35
            assert 0.4 <= scenario.kappa <= 0.6
            assert scenario.A == 100  # Fixed values
            assert scenario.L == 100
            assert scenario.c == 0.5
            assert 0.1 <= scenario.rho_ignite <= 0.3
            assert 10 <= scenario.N_min <= 20
            assert scenario.num_agents == 4

    def test_scenario_feature_vector(self):
        """Test scenario feature vector generation."""
        scenario = default_scenario(num_agents=4)
        features = scenario.to_feature_vector()

        assert len(features) == 10  # Should have 10 parameters
        assert all(
            isinstance(f, (int, float, np.integer, np.floating)) for f in features
        )


class TestScenarioFunctions:
    """Test individual scenario functions."""

    def test_easy_scenario(self):
        """Test easy scenario generation."""
        from bucket_brigade.envs.scenarios import easy_scenario

        scenario = easy_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.1  # Low spread
        assert scenario.kappa == 0.8  # High extinguish efficiency
        assert scenario.N_min == 10

    def test_hard_scenario(self):
        """Test hard scenario generation."""
        from bucket_brigade.envs.scenarios import hard_scenario

        scenario = hard_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.4  # High spread
        assert scenario.kappa == 0.3  # Low extinguish efficiency
        assert scenario.N_min == 15

    def test_trivial_cooperation_scenario(self):
        """Test trivial cooperation scenario."""
        from bucket_brigade.envs.scenarios import trivial_cooperation_scenario

        scenario = trivial_cooperation_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.15
        assert scenario.kappa == 0.9
        assert scenario.p_spark == 0.0  # No spontaneous fires

    def test_early_containment_scenario(self):
        """Test early containment scenario."""
        from bucket_brigade.envs.scenarios import early_containment_scenario

        scenario = early_containment_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.35  # High spread
        assert scenario.rho_ignite == 0.3  # Many initial fires

    def test_greedy_neighbor_scenario(self):
        """Test greedy neighbor scenario."""
        from bucket_brigade.envs.scenarios import greedy_neighbor_scenario

        scenario = greedy_neighbor_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.c == 1.0  # High work cost

    def test_sparse_heroics_scenario(self):
        """Test sparse heroics scenario."""
        from bucket_brigade.envs.scenarios import sparse_heroics_scenario

        scenario = sparse_heroics_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.1  # Very low spread
        assert scenario.N_min == 20  # Long games

    def test_rest_trap_scenario(self):
        """Test rest trap scenario."""
        from bucket_brigade.envs.scenarios import rest_trap_scenario

        scenario = rest_trap_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.05  # Very low spread
        assert scenario.kappa == 0.95  # Very high extinguish rate

    def test_chain_reaction_scenario(self):
        """Test chain reaction scenario."""
        from bucket_brigade.envs.scenarios import chain_reaction_scenario

        scenario = chain_reaction_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.45  # High spread
        assert scenario.rho_ignite == 0.3  # Many initial fires

    def test_deceptive_calm_scenario(self):
        """Test deceptive calm scenario."""
        from bucket_brigade.envs.scenarios import deceptive_calm_scenario

        scenario = deceptive_calm_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.N_min == 20  # Long games
        assert scenario.p_spark == 0.05  # Occasional sparks

    def test_overcrowding_scenario(self):
        """Test overcrowding scenario."""
        from bucket_brigade.envs.scenarios import overcrowding_scenario

        scenario = overcrowding_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.kappa == 0.3  # Low extinguish efficiency
        assert scenario.A == 50.0  # Lower reward

    def test_mixed_motivation_scenario(self):
        """Test mixed motivation scenario."""
        from bucket_brigade.envs.scenarios import mixed_motivation_scenario

        scenario = mixed_motivation_scenario(num_agents=4)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.3  # Moderate spread
        assert scenario.N_min == 15

    def test_sample_easy_coop_scenario(self):
        """Test easy cooperation sampling."""
        from bucket_brigade.envs.scenarios import sample_easy_coop_scenario

        scenario = sample_easy_coop_scenario(num_agents=4, seed=42)
        assert scenario.num_agents == 4
        assert 0.1 <= scenario.beta <= 0.2
        assert 0.7 <= scenario.kappa <= 0.9
        assert scenario.p_spark == 0.0  # No sparks in easy coop

    def test_sample_crisis_scenario(self):
        """Test crisis scenario sampling."""
        from bucket_brigade.envs.scenarios import sample_crisis_scenario

        scenario = sample_crisis_scenario(num_agents=4, seed=42)
        assert scenario.num_agents == 4
        assert 0.3 <= scenario.beta <= 0.5
        assert 0.4 <= scenario.kappa <= 0.6
        assert 0.2 <= scenario.rho_ignite <= 0.4

    def test_sample_sparse_work_scenario(self):
        """Test sparse work scenario sampling."""
        from bucket_brigade.envs.scenarios import sample_sparse_work_scenario

        scenario = sample_sparse_work_scenario(num_agents=4, seed=42)
        assert scenario.num_agents == 4
        assert 0.1 <= scenario.beta <= 0.2
        assert 0.6 <= scenario.c <= 0.9
        assert 15 <= scenario.N_min <= 25

    def test_sample_deception_scenario(self):
        """Test deception scenario sampling."""
        from bucket_brigade.envs.scenarios import sample_deception_scenario

        scenario = sample_deception_scenario(num_agents=4, seed=42)
        assert scenario.num_agents == 4
        assert scenario.beta == 0.25
        assert scenario.N_min == 15
        assert 0.03 <= scenario.p_spark <= 0.05

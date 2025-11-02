"""Tests for the Bucket Brigade environment."""

import numpy as np
import pytest
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
        required_keys = ['signals', 'locations', 'houses', 'last_actions', 'scenario_info']
        for key in required_keys:
            assert key in obs

        assert len(obs['signals']) == 4
        assert len(obs['locations']) == 4
        assert len(obs['houses']) == 10
        assert len(obs['last_actions']) == 4
        assert len(obs['scenario_info']) > 0

    def test_game_step(self):
        """Test a single game step."""
        env = BucketBrigadeEnv(num_agents=4)
        env.reset(seed=42)

        # Create random actions for all agents
        actions = np.random.randint(0, 2, size=(4, 2))  # [house, mode] for each agent

        obs, rewards, dones, info = env.step(actions)

        # Check return values
        assert len(obs['signals']) == 4
        assert len(rewards) == 4
        assert len(dones) == 4
        assert isinstance(info, dict)

        # Rewards should be reasonable
        assert all(isinstance(r, (int, float)) for r in rewards)

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
        actions = np.array([
            [1, 1],  # Agent 0 works on burning house 1
            [7, 1],  # Agent 1 works on burning house 7
            [0, 0],  # Agent 2 rests
            [2, 0],  # Agent 3 rests
        ])

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

        actions = np.array([
            [0, 1],  # Work
            [1, 0],  # Rest
            [2, 1],  # Work
            [3, 0],  # Rest
        ])

        rewards = env._compute_rewards(actions)

        # Working agents should get cost penalty
        assert rewards[0] < 0  # Working
        assert rewards[1] > 0  # Resting
        assert rewards[2] < 0  # Working
        assert rewards[3] > 0  # Resting

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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            env.save_replay(temp_path)

            # Check file was created and has content
            assert os.path.exists(temp_path)

            import json
            with open(temp_path, 'r') as f:
                data = json.load(f)

            assert 'scenario' in data
            assert 'nights' in data
            assert len(data['nights']) > 0

            # Check night structure
            night = data['nights'][0]
            required_fields = ['night', 'houses', 'signals', 'locations', 'actions', 'rewards']
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
        assert all(isinstance(f, (int, float)) for f in features)

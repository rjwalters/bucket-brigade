"""
Unit tests for observation utilities.

Tests the conversion of Rust observations to numpy arrays for neural network input.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from bucket_brigade.training.observation_utils import (
    flatten_observation,
    get_observation_dim,
    create_scenario_info,
)


class TestFlattenObservation:
    """Tests for flatten_observation function."""

    def test_flatten_observation_basic(self):
        """Test basic observation flattening with standard 4-agent, 10-house game."""
        # Create mock observation
        obs = Mock()
        obs.houses = [0.0, 0.5, 1.0, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.1]  # 10 houses
        obs.signals = [0.1, 0.2, 0.3, 0.4]  # 4 agents
        obs.locations = [0, 2, 5, 7]  # 4 agents
        obs.last_actions = [[0, 1], [2, 0], [5, 2], [7, 1]]  # 4 agents x 2 values

        # Flatten
        flat_obs = flatten_observation(obs)

        # Expected dimensions: 10 houses + 4 signals + 4 locations + 8 last_actions + 10 scenario_info = 36
        assert flat_obs.shape == (36,)
        assert flat_obs.dtype == np.float32

        # Verify components are in correct order
        assert np.allclose(flat_obs[:10], obs.houses)  # houses
        assert np.allclose(flat_obs[10:14], obs.signals)  # signals
        assert np.allclose(flat_obs[14:18], obs.locations)  # locations
        assert np.allclose(flat_obs[18:26], [0, 1, 2, 0, 5, 2, 7, 1])  # last_actions flattened
        assert np.allclose(flat_obs[26:], np.zeros(10))  # scenario_info (zeros when None)

    def test_flatten_observation_with_scenario_info(self):
        """Test flattening with provided scenario info."""
        # Create mock observation
        obs = Mock()
        obs.houses = [0.5] * 10
        obs.signals = [0.1] * 4
        obs.locations = [0, 1, 2, 3]
        obs.last_actions = [[0, 0], [1, 1], [2, 2], [3, 0]]

        # Custom scenario info
        scenario_info = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Flatten
        flat_obs = flatten_observation(obs, scenario_info)

        # Verify scenario info is included
        assert np.allclose(flat_obs[26:], scenario_info)

    def test_flatten_observation_different_sizes(self):
        """Test flattening with different game configurations."""
        # 2 agents, 5 houses
        obs = Mock()
        obs.houses = [0.0, 0.1, 0.2, 0.3, 0.4]  # 5 houses
        obs.signals = [0.5, 0.6]  # 2 agents
        obs.locations = [0, 2]  # 2 agents
        obs.last_actions = [[0, 1], [2, 0]]  # 2 agents x 2 values

        flat_obs = flatten_observation(obs)

        # Expected: 5 houses + 2 signals + 2 locations + 4 last_actions + 10 scenario_info = 23
        assert flat_obs.shape == (23,)
        assert flat_obs.dtype == np.float32

    def test_flatten_observation_dtype_consistency(self):
        """Test that output is always float32."""
        obs = Mock()
        obs.houses = [0, 1, 2]  # int list
        obs.signals = [0.0, 1.0]  # float list
        obs.locations = [0, 1]  # int list
        obs.last_actions = [[0, 1], [1, 0]]  # int nested list

        flat_obs = flatten_observation(obs)
        assert flat_obs.dtype == np.float32

    def test_flatten_observation_empty_last_actions(self):
        """Test handling of edge case with minimal observations."""
        obs = Mock()
        obs.houses = [0.5]
        obs.signals = [0.1]
        obs.locations = [0]
        obs.last_actions = [[0, 0]]

        flat_obs = flatten_observation(obs)

        # Expected: 1 house + 1 signal + 1 location + 2 last_actions + 10 scenario_info = 15
        assert flat_obs.shape == (15,)


class TestGetObservationDim:
    """Tests for get_observation_dim function."""

    def test_standard_game_config(self):
        """Test dimension calculation for standard 4-agent, 10-house game."""
        dim = get_observation_dim(num_houses=10, num_agents=4)
        # 10 houses + 4 signals + 4 locations + 8 last_actions + 10 scenario_info = 36
        assert dim == 36

    def test_small_game_config(self):
        """Test dimension calculation for small game."""
        dim = get_observation_dim(num_houses=5, num_agents=2)
        # 5 houses + 2 signals + 2 locations + 4 last_actions + 10 scenario_info = 23
        assert dim == 23

    def test_large_game_config(self):
        """Test dimension calculation for large game."""
        dim = get_observation_dim(num_houses=20, num_agents=8)
        # 20 houses + 8 signals + 8 locations + 16 last_actions + 10 scenario_info = 62
        assert dim == 62

    def test_single_agent(self):
        """Test dimension calculation for single agent game."""
        dim = get_observation_dim(num_houses=5, num_agents=1)
        # 5 houses + 1 signal + 1 location + 2 last_actions + 10 scenario_info = 19
        assert dim == 19

    def test_consistency_with_flatten(self):
        """Test that get_observation_dim matches actual flattened observation size."""
        # Create observation with 3 agents, 7 houses
        obs = Mock()
        obs.houses = [0.0] * 7
        obs.signals = [0.0] * 3
        obs.locations = [0] * 3
        obs.last_actions = [[0, 0]] * 3

        flat_obs = flatten_observation(obs)
        expected_dim = get_observation_dim(num_houses=7, num_agents=3)

        assert flat_obs.shape[0] == expected_dim


class TestCreateScenarioInfo:
    """Tests for create_scenario_info function."""

    def test_create_scenario_info_standard(self):
        """Test scenario info creation with standard scenario."""
        # Mock scenario object
        scenario = Mock()
        scenario.prob_fire_spreads_to_neighbor = 0.1
        scenario.prob_solo_agent_extinguishes_fire = 0.2
        scenario.prob_house_catches_fire = 0.3
        scenario.team_reward_house_survives = 10.0
        scenario.team_penalty_house_burns = -5.0
        scenario.cost_to_work_one_night = -1.0
        scenario.min_nights = 3

        scenario_info = create_scenario_info(scenario)

        # Check shape and dtype
        assert scenario_info.shape == (10,)
        assert scenario_info.dtype == np.float32

        # Check values
        assert scenario_info[0] == pytest.approx(0.1)  # prob_fire_spreads_to_neighbor
        assert scenario_info[1] == pytest.approx(0.2)  # prob_solo_agent_extinguishes_fire
        assert scenario_info[2] == pytest.approx(0.3)  # prob_house_catches_fire
        assert scenario_info[3] == pytest.approx(10.0)  # team_reward_house_survives
        assert scenario_info[4] == pytest.approx(-5.0)  # team_penalty_house_burns
        assert scenario_info[5] == pytest.approx(-1.0)  # cost_to_work_one_night
        assert scenario_info[6] == pytest.approx(3.0)  # min_nights (converted to float)

        # Check padding
        assert scenario_info[7] == pytest.approx(0.0)
        assert scenario_info[8] == pytest.approx(0.0)
        assert scenario_info[9] == pytest.approx(0.0)

    def test_create_scenario_info_edge_values(self):
        """Test scenario info with extreme values."""
        scenario = Mock()
        scenario.prob_fire_spreads_to_neighbor = 0.0
        scenario.prob_solo_agent_extinguishes_fire = 1.0
        scenario.prob_house_catches_fire = 0.0
        scenario.team_reward_house_survives = 100.0
        scenario.team_penalty_house_burns = -100.0
        scenario.cost_to_work_one_night = 0.0
        scenario.min_nights = 1

        scenario_info = create_scenario_info(scenario)

        assert scenario_info[0] == 0.0
        assert scenario_info[1] == 1.0
        assert scenario_info[2] == 0.0
        assert scenario_info[3] == 100.0
        assert scenario_info[4] == -100.0
        assert scenario_info[5] == 0.0
        assert scenario_info[6] == 1.0

    def test_create_scenario_info_dtype(self):
        """Test that scenario info is always float32."""
        scenario = Mock()
        scenario.prob_fire_spreads_to_neighbor = 0.1
        scenario.prob_solo_agent_extinguishes_fire = 0.2
        scenario.prob_house_catches_fire = 0.3
        scenario.team_reward_house_survives = 10.0
        scenario.team_penalty_house_burns = -5.0
        scenario.cost_to_work_one_night = -1.0
        scenario.min_nights = 3  # int

        scenario_info = create_scenario_info(scenario)

        # All values should be float32
        assert scenario_info.dtype == np.float32
        assert isinstance(scenario_info[6], np.float32)  # min_nights converted


class TestObservationUtilsIntegration:
    """Integration tests combining multiple utility functions."""

    def test_full_pipeline_consistency(self):
        """Test that dimension calculation and flattening are consistent."""
        num_houses = 8
        num_agents = 3

        # Create observation
        obs = Mock()
        obs.houses = [0.1 * i for i in range(num_houses)]
        obs.signals = [0.2 * i for i in range(num_agents)]
        obs.locations = list(range(num_agents))
        obs.last_actions = [[i, i % 3] for i in range(num_agents)]

        # Create scenario
        scenario = Mock()
        scenario.prob_fire_spreads_to_neighbor = 0.15
        scenario.prob_solo_agent_extinguishes_fire = 0.25
        scenario.prob_house_catches_fire = 0.05
        scenario.team_reward_house_survives = 15.0
        scenario.team_penalty_house_burns = -7.5
        scenario.cost_to_work_one_night = -0.5
        scenario.min_nights = 5

        # Get expected dimension
        expected_dim = get_observation_dim(num_houses, num_agents)

        # Flatten observation with scenario info
        scenario_info = create_scenario_info(scenario)
        flat_obs = flatten_observation(obs, scenario_info)

        # Verify consistency
        assert flat_obs.shape[0] == expected_dim
        assert flat_obs.dtype == np.float32

        # Verify scenario info is correctly embedded
        assert np.array_equal(flat_obs[-10:], scenario_info)

    def test_multiple_observations_same_game(self):
        """Test that multiple observations from same game have consistent dimensions."""
        num_houses = 10
        num_agents = 4

        # Create scenario info once
        scenario = Mock()
        scenario.prob_fire_spreads_to_neighbor = 0.1
        scenario.prob_solo_agent_extinguishes_fire = 0.2
        scenario.prob_house_catches_fire = 0.3
        scenario.team_reward_house_survives = 10.0
        scenario.team_penalty_house_burns = -5.0
        scenario.cost_to_work_one_night = -1.0
        scenario.min_nights = 3
        scenario_info = create_scenario_info(scenario)

        expected_dim = get_observation_dim(num_houses, num_agents)

        # Create multiple observations with different values
        observations = []
        for _ in range(5):
            obs = Mock()
            obs.houses = np.random.rand(num_houses).tolist()
            obs.signals = np.random.rand(num_agents).tolist()
            obs.locations = np.random.randint(0, num_houses, num_agents).tolist()
            obs.last_actions = np.random.randint(0, 3, (num_agents, 2)).tolist()
            observations.append(obs)

        # Flatten all observations
        flat_observations = [flatten_observation(obs, scenario_info) for obs in observations]

        # All should have same dimension
        for flat_obs in flat_observations:
            assert flat_obs.shape[0] == expected_dim
            assert flat_obs.dtype == np.float32

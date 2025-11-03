"""Tests for agent implementations."""

import numpy as np
import pytest
from bucket_brigade.agents import (
    RandomAgent,
    HeuristicAgent,
    create_random_agent,
    create_archetype_agent,
)


class TestAgentBase:
    """Test the base agent class."""

    def test_agent_initialization(self):
        """Test agent initialization."""
        agent = RandomAgent(0, "TestAgent")
        assert agent.agent_id == 0
        assert agent.name == "TestAgent"

    def test_agent_reset(self):
        """Test agent reset functionality."""
        agent = RandomAgent(0)
        agent.reset()  # Should not raise any errors

    def test_random_agent_actions(self):
        """Test random agent produces valid actions."""
        agent = RandomAgent(0)
        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        action = agent.act(obs)

        # Check action format
        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert 0 <= action[0] <= 9  # House index
        assert action[1] in [0, 1]  # Mode (REST or WORK)


class TestHeuristicAgent:
    """Test the heuristic agent implementation."""

    def test_heuristic_agent_initialization(self):
        """Test heuristic agent initialization."""
        params = np.array([0.5] * 10)  # Neutral parameters
        agent = HeuristicAgent(params, 0, "HeuristicAgent")

        assert agent.agent_id == 0
        assert agent.name == "HeuristicAgent"
        assert len(agent.params) == 10

    def test_parameter_validation(self):
        """Test parameter validation."""
        with pytest.raises(AssertionError):
            HeuristicAgent(np.array([0.5] * 9), 0)  # Wrong number of parameters

    def test_action_generation(self):
        """Test heuristic agent action generation."""
        # Create an agent that tends to work
        params = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.5, 0.1, 0.0, 0.0, 0.5])
        agent = HeuristicAgent(params, 0)

        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array(
                [0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
            ),  # Houses 1 and 7 burning
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        action = agent.act(obs)

        assert isinstance(action, np.ndarray)
        assert action.shape == (2,)
        assert 0 <= action[0] <= 9
        assert action[1] in [0, 1]

    def test_agent_memory(self):
        """Test agent memory and fatigue effects."""
        params = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.9, 0.5, 0.5]
        )  # High fatigue
        agent = HeuristicAgent(params, 0)

        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        # First action
        action1 = agent.act(obs)

        # Second action should have some chance to repeat (fatigue effect)
        action2 = agent.act(obs)

        # Actions should be valid regardless
        assert isinstance(action1, np.ndarray)
        assert isinstance(action2, np.ndarray)

    def test_scenario_adaptation(self):
        """Test agent adapts to different scenarios."""
        params = np.array(
            [0.5, 0.5, 0.5, 0.5, 0.8, 0.5, 0.5, 0.0, 0.5, 0.5]
        )  # High risk aversion
        agent = HeuristicAgent(params, 0)

        # High fire scenario
        high_fire_obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),  # All burning
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        # Low fire scenario
        low_fire_obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),  # All safe
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        # Agent should potentially behave differently (though stochastic)
        action_high = agent.act(high_fire_obs)
        action_low = agent.act(low_fire_obs)

        assert isinstance(action_high, np.ndarray)
        assert isinstance(action_low, np.ndarray)


# TODO: Implement proper agent loader tests when API is finalized
# class TestAgentLoader:
#     """Test agent loading functionality."""
#     pass


class TestAgentFactory:
    """Test agent factory functions."""

    def test_create_random_agent(self):
        """Test random agent creation."""
        agent = create_random_agent(0, "RandomTest")

        assert agent.agent_id == 0
        assert agent.name == "RandomTest"
        assert len(agent.params) == 10
        assert all(0 <= p <= 1 for p in agent.params)

    def test_create_archetype_agent(self):
        """Test archetype agent creation."""
        archetypes = ["firefighter", "liar", "free_rider", "hero", "coordinator"]

        for archetype in archetypes:
            agent = create_archetype_agent(archetype, 0)

            assert agent.agent_id == 0
            assert archetype.title() in agent.name
            assert len(agent.params) == 10
            assert all(0 <= p <= 1 for p in agent.params)

    def test_invalid_archetype(self):
        """Test invalid archetype raises error."""
        with pytest.raises(ValueError):
            create_archetype_agent("invalid_archetype", 0)

    def test_agent_action_consistency(self):
        """Test agents produce consistent action formats."""
        agents = [
            RandomAgent(0),
            create_random_agent(1),
            create_archetype_agent("firefighter", 2),
            create_archetype_agent("free_rider", 3),
        ]

        obs = {
            "signals": np.array([0, 1, 0, 1]),
            "locations": np.array([0, 1, 2, 3]),
            "houses": np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
            "last_actions": np.array([[0, 0], [1, 1], [2, 0], [3, 1]]),
            "scenario_info": np.array([0.25, 0.5, 100, 100, 0.5, 0.2, 12, 0.02, 12, 4]),
        }

        for agent in agents:
            action = agent.act(obs)

            assert isinstance(action, np.ndarray)
            assert action.shape == (2,)
            assert isinstance(action[0], (int, np.integer))  # House index
            assert isinstance(action[1], (int, np.integer))  # Mode
            assert 0 <= action[0] <= 9
            assert action[1] in [0, 1]

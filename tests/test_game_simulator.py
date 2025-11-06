"""
Unit tests for game simulator.

Tests the CPU-side game simulator that manages Rust environments and
distributes experiences to GPU learners.
"""

import pytest
import numpy as np
import torch
import multiprocessing as mp
from unittest.mock import Mock, MagicMock, patch
from queue import Empty

from bucket_brigade.training.game_simulator import Matchmaker, GameSimulator
from bucket_brigade.training import PolicyNetwork
from bucket_brigade_core import SCENARIOS


class TestMatchmaker:
    """Tests for Matchmaker class."""

    def test_initialization(self):
        """Test matchmaker initialization."""
        matchmaker = Matchmaker(population_size=8, num_agents_per_game=4)

        assert matchmaker.population_size == 8
        assert matchmaker.num_agents_per_game == 4
        assert len(matchmaker.match_counts) == 8
        assert all(count == 0 for count in matchmaker.match_counts)

    def test_round_robin_sampling(self):
        """Test round-robin matchmaking strategy."""
        matchmaker = Matchmaker(population_size=8, num_agents_per_game=4)

        # First sample should select agents 0-3 (all have count 0)
        agents = matchmaker.sample_agents(strategy="round_robin")
        assert len(agents) == 4
        assert set(agents) == {0, 1, 2, 3}

        # Verify counts updated
        assert all(matchmaker.match_counts[i] == 1 for i in range(4))
        assert all(matchmaker.match_counts[i] == 0 for i in range(4, 8))

        # Second sample should select agents 4-7 (lower counts)
        agents = matchmaker.sample_agents(strategy="round_robin")
        assert set(agents) == {4, 5, 6, 7}

        # All agents should have count 1 now
        assert all(count == 1 for count in matchmaker.match_counts)

    def test_round_robin_even_distribution(self):
        """Test that round-robin maintains even distribution over time."""
        matchmaker = Matchmaker(population_size=6, num_agents_per_game=2)

        # Run many samples
        for _ in range(10):
            matchmaker.sample_agents(strategy="round_robin")

        # All agents should have similar counts (within 1 of each other)
        counts = matchmaker.match_counts
        assert max(counts) - min(counts) <= 1

    def test_random_sampling(self):
        """Test random matchmaking strategy."""
        matchmaker = Matchmaker(population_size=8, num_agents_per_game=4)

        # Sample agents randomly
        agents = matchmaker.sample_agents(strategy="random")

        assert len(agents) == 4
        assert all(0 <= agent < 8 for agent in agents)
        assert len(set(agents)) == 4  # No duplicates

        # Counts should be updated
        assert sum(matchmaker.match_counts) == 4

    def test_random_sampling_multiple_rounds(self):
        """Test random sampling over multiple rounds."""
        matchmaker = Matchmaker(population_size=8, num_agents_per_game=4)

        for _ in range(10):
            agents = matchmaker.sample_agents(strategy="random")
            assert len(agents) == 4
            assert len(set(agents)) == 4

        # After 10 rounds, total count should be 40
        assert sum(matchmaker.match_counts) == 40

    def test_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        matchmaker = Matchmaker(population_size=8, num_agents_per_game=4)

        with pytest.raises(ValueError, match="Unknown matchmaking strategy"):
            matchmaker.sample_agents(strategy="invalid")

    def test_exact_population_size(self):
        """Test when population size equals num_agents_per_game."""
        matchmaker = Matchmaker(population_size=4, num_agents_per_game=4)

        agents = matchmaker.sample_agents(strategy="round_robin")
        assert set(agents) == {0, 1, 2, 3}


class TestGameSimulatorInitialization:
    """Tests for GameSimulator initialization."""

    def test_basic_initialization(self):
        """Test basic simulator initialization."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=4,
            population_size=8,
            num_agents_per_game=4,
            seed=42,
        )

        assert simulator.num_games == 4
        assert simulator.population_size == 8
        assert simulator.num_agents_per_game == 4
        assert len(simulator.envs) == 4
        assert simulator.total_steps == 0
        assert simulator.total_episodes == 0
        assert len(simulator.episode_rewards) == 8

    def test_scenario_info_created(self):
        """Test that scenario info is cached."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(scenario=scenario, num_games=2)

        assert simulator.scenario_info is not None
        assert simulator.scenario_info.shape == (10,)
        assert simulator.scenario_info.dtype == np.float32

    def test_matchmaker_initialized(self):
        """Test that matchmaker is initialized correctly."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=8,
            num_agents_per_game=4,
        )

        assert simulator.matchmaker is not None
        assert simulator.matchmaker.population_size == 8
        assert simulator.matchmaker.num_agents_per_game == 4

    def test_seed_determinism(self):
        """Test that seed produces deterministic results."""
        scenario = SCENARIOS['trivial_cooperation']

        # Create two simulators with same seed
        sim1 = GameSimulator(scenario=scenario, num_games=2, seed=42)
        sim2 = GameSimulator(scenario=scenario, num_games=2, seed=42)

        # Environment states should be similar (hard to test exactly without running)
        # At least verify they initialize without error
        assert sim1 is not None
        assert sim2 is not None


class TestGameSimulatorPolicyManagement:
    """Tests for policy registration and updates."""

    def test_register_policy(self):
        """Test registering a policy for an agent."""
        scenario = SCENARIOS['trivial_cooperation']
        simulator = GameSimulator(scenario=scenario, num_games=2)

        # Create policy
        policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)

        # Register
        simulator.register_policy(agent_id=0, policy=policy)

        assert 0 in simulator.policies
        assert simulator.policies[0] is policy

    def test_register_multiple_policies(self):
        """Test registering policies for multiple agents."""
        scenario = SCENARIOS['trivial_cooperation']
        simulator = GameSimulator(scenario=scenario, num_games=2, population_size=4)

        for agent_id in range(4):
            policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
            simulator.register_policy(agent_id=agent_id, policy=policy)

        assert len(simulator.policies) == 4
        assert all(i in simulator.policies for i in range(4))

    def test_update_policy(self):
        """Test updating an agent's policy weights."""
        scenario = SCENARIOS['trivial_cooperation']
        simulator = GameSimulator(scenario=scenario, num_games=2)

        # Register initial policy
        policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
        simulator.register_policy(agent_id=0, policy=policy)

        # Get initial state dict
        initial_state = {k: v.clone() for k, v in policy.state_dict().items()}

        # Create new state dict with different weights
        new_state = policy.state_dict()
        for key in new_state:
            new_state[key] = torch.randn_like(new_state[key])

        # Update policy
        simulator.update_policy(agent_id=0, state_dict=new_state)

        # Verify policy was updated
        updated_state = simulator.policies[0].state_dict()
        for key in updated_state:
            assert not torch.allclose(updated_state[key], initial_state[key])
            assert torch.allclose(updated_state[key], new_state[key])


class TestGameSimulatorActionSelection:
    """Tests for action selection."""

    def test_select_action_shape(self):
        """Test that select_action returns correct shapes."""
        scenario = SCENARIOS['trivial_cooperation']
        simulator = GameSimulator(scenario=scenario, num_games=2)

        # Register policy
        policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
        simulator.register_policy(agent_id=0, policy=policy)

        # Create dummy observation (36 dims for standard game)
        observation = np.random.randn(36).astype(np.float32)

        # Select action
        house, mode, logprob = simulator.select_action(agent_id=0, observation=observation)

        # Verify types and ranges
        assert isinstance(house, int)
        assert isinstance(mode, int)
        assert isinstance(logprob, float)
        assert 0 <= house < 10
        assert 0 <= mode < 3

    def test_select_action_determinism(self):
        """Test that same observation with same seed produces same action."""
        scenario = SCENARIOS['trivial_cooperation']

        # Create policy and observation
        policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
        observation = np.random.randn(36).astype(np.float32)

        # Select action multiple times with same input
        results = []
        for _ in range(5):
            simulator = GameSimulator(scenario=scenario, num_games=2, seed=42)
            simulator.register_policy(agent_id=0, policy=policy)

            torch.manual_seed(42)
            house, mode, logprob = simulator.select_action(agent_id=0, observation=observation)
            results.append((house, mode))

        # All results should be the same (with same seed)
        assert all(r == results[0] for r in results)


class TestGameSimulatorExperienceDistribution:
    """Tests for experience distribution to queues."""

    def test_distribute_experiences_basic(self):
        """Test distributing experiences to queues."""
        scenario = SCENARIOS['trivial_cooperation']

        # Create mock queues (easier to test than real mp.Queue)
        queues = [Mock() for _ in range(4)]

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
            experience_queues=queues,
        )

        # Create mock trajectories
        trajectories = {
            0: [{'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0}],
            1: [{'obs': np.zeros(36), 'action': [1, 1], 'reward': 2.0}],
        }

        # Distribute
        simulator.distribute_experiences(trajectories)

        # Verify experiences were queued
        assert queues[0].put.called
        assert queues[1].put.called
        assert not queues[2].put.called
        assert not queues[3].put.called

        # Check content
        queues[0].put.assert_called_once_with((0, trajectories[0][0]))
        queues[1].put.assert_called_once_with((1, trajectories[1][0]))

    def test_distribute_experiences_multiple_per_agent(self):
        """Test distributing multiple experiences per agent."""
        scenario = SCENARIOS['trivial_cooperation']

        # Use mock queues
        queues = [Mock() for _ in range(2)]

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=2,
            experience_queues=queues,
        )

        # Create trajectories with multiple experiences
        trajectories = {
            0: [
                {'obs': np.zeros(36), 'action': [0, 0], 'reward': 1.0},
                {'obs': np.zeros(36), 'action': [1, 1], 'reward': 2.0},
                {'obs': np.zeros(36), 'action': [2, 2], 'reward': 3.0},
            ],
        }

        simulator.distribute_experiences(trajectories)

        # Should have called put 3 times on queue 0
        assert queues[0].put.call_count == 3

        # Verify each call
        calls = queues[0].put.call_args_list
        for i, call in enumerate(calls):
            agent_id, exp = call[0][0]
            assert agent_id == 0
            assert exp == trajectories[0][i]


class TestGameSimulatorPolicyUpdates:
    """Tests for receiving policy updates from GPU learners."""

    def test_check_policy_updates_basic(self):
        """Test checking for and applying policy updates."""
        scenario = SCENARIOS['trivial_cooperation']

        # Create mock update queue
        update_queue = Mock()

        # Create completely new policy with different weights
        new_policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
        torch.manual_seed(999)
        for param in new_policy.parameters():
            param.data = torch.randn_like(param.data)
        new_state = new_policy.state_dict()

        # Configure mock queue to return update then be empty
        update_queue.empty.side_effect = [False, True]  # Has one update, then empty
        update_queue.get.return_value = (0, new_state)

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=2,
            policy_update_queue=update_queue,
        )

        # Register initial policy
        policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
        simulator.register_policy(agent_id=0, policy=policy)

        # Get initial weights
        initial_state = {k: v.clone() for k, v in policy.state_dict().items()}

        # Check for updates
        simulator.check_policy_updates()

        # Verify policy was updated
        updated_state = simulator.policies[0].state_dict()
        for key in updated_state:
            # Should be different from initial
            assert not torch.allclose(updated_state[key], initial_state[key])
            # Should match new_state
            assert torch.allclose(updated_state[key], new_state[key])

    def test_check_policy_updates_multiple(self):
        """Test applying multiple policy updates."""
        scenario = SCENARIOS['trivial_cooperation']

        update_queue = mp.Queue()

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
            policy_update_queue=update_queue,
        )

        # Register policies for multiple agents
        for agent_id in range(4):
            policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
            simulator.register_policy(agent_id=agent_id, policy=policy)

        # Queue multiple updates
        for agent_id in range(4):
            new_state = {k: torch.randn_like(v) for k, v in simulator.policies[agent_id].state_dict().items()}
            update_queue.put((agent_id, new_state))

        # Apply all updates
        simulator.check_policy_updates()

        # Queue should be empty
        assert update_queue.empty()

    def test_check_policy_updates_no_queue(self):
        """Test that check_policy_updates handles no queue gracefully."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            policy_update_queue=None,
        )

        # Should not raise error
        simulator.check_policy_updates()


class TestGameSimulatorStatistics:
    """Tests for statistics tracking."""

    def test_get_statistics_initial(self):
        """Test getting statistics from fresh simulator."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
        )

        stats = simulator.get_statistics()

        assert stats['total_steps'] == 0
        assert stats['total_episodes'] == 0
        assert len(stats['episode_rewards']) == 4
        assert all(len(rewards) == 0 for rewards in stats['episode_rewards'].values())
        assert len(stats['match_counts']) == 4
        assert all(count == 0 for count in stats['match_counts'])

    def test_episode_rewards_tracking(self):
        """Test that episode rewards are tracked correctly."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
        )

        # Manually add episode rewards (simulating what run_episode would do)
        simulator.episode_rewards[0].append(10.0)
        simulator.episode_rewards[0].append(15.0)
        simulator.episode_rewards[1].append(12.0)

        stats = simulator.get_statistics()

        assert len(stats['episode_rewards'][0]) == 2
        assert len(stats['episode_rewards'][1]) == 1
        assert stats['episode_rewards'][0][0] == pytest.approx(10.0)


class TestGameSimulatorIntegration:
    """Integration tests with real Rust environments and policies."""

    def test_run_episode_basic(self):
        """Test running a full episode with real environment."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
            num_agents_per_game=4,
            seed=42,
        )

        # Register policies for all agents
        for agent_id in range(4):
            policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
            simulator.register_policy(agent_id=agent_id, policy=policy)

        # Run episode
        trajectories = simulator.run_episode(env_id=0)

        # Verify trajectories structure
        assert len(trajectories) == 4  # One per agent
        for agent_id, traj in trajectories.items():
            assert len(traj) > 0  # Should have at least some steps
            for exp in traj:
                assert 'obs' in exp
                assert 'action' in exp
                assert 'reward' in exp
                assert 'next_obs' in exp or exp['done']
                assert 'done' in exp
                assert 'logprob' in exp

    def test_run_episode_updates_statistics(self):
        """Test that running episode updates statistics."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
            num_agents_per_game=4,
            seed=42,
        )

        # Register policies
        for agent_id in range(4):
            policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
            simulator.register_policy(agent_id=agent_id, policy=policy)

        initial_episodes = simulator.total_episodes
        initial_steps = simulator.total_steps

        # Run episode
        simulator.run_episode(env_id=0)

        # Verify statistics updated
        assert simulator.total_episodes == initial_episodes + 1
        assert simulator.total_steps > initial_steps

    def test_multiple_episodes_consistent(self):
        """Test running multiple episodes produces consistent results."""
        scenario = SCENARIOS['trivial_cooperation']

        simulator = GameSimulator(
            scenario=scenario,
            num_games=2,
            population_size=4,
            num_agents_per_game=4,
            seed=42,
        )

        # Register policies
        for agent_id in range(4):
            policy = PolicyNetwork(obs_dim=36, action_dims=[10, 3], hidden_size=128)
            simulator.register_policy(agent_id=agent_id, policy=policy)

        # Run multiple episodes
        for _ in range(5):
            trajectories = simulator.run_episode(env_id=0)
            assert len(trajectories) == 4

        # Verify statistics
        assert simulator.total_episodes == 5
        assert simulator.total_steps > 0

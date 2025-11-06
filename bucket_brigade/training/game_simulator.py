"""
CPU Game Simulator for Population-Based Training.

This module implements the CPU-side game simulator that:
1. Runs Rust BucketBrigade environments in parallel
2. Manages matchmaking (assigning agents to games)
3. Collects trajectories from gameplay
4. Distributes experiences to GPU learners via queues
5. Receives and updates agent policies from GPU learners

Architecture:
    CPU Process (this) → Experience Queues → GPU Learners
    GPU Learners → Policy Update Queue → CPU Process (this)
"""

import multiprocessing as mp
from typing import Dict, List, Optional, Tuple
import time
import random

import torch
import numpy as np

from bucket_brigade_core import BucketBrigade, Scenario
from bucket_brigade.training.observation_utils import flatten_observation, create_scenario_info


class Matchmaker:
    """Handles agent selection for games (matchmaking)."""

    def __init__(self, population_size: int, num_agents_per_game: int = 4):
        self.population_size = population_size
        self.num_agents_per_game = num_agents_per_game
        self.match_counts = [0] * population_size  # Track how many times each agent has played

    def sample_agents(self, strategy: str = "round_robin") -> List[int]:
        """
        Sample agent IDs for a game.

        Args:
            strategy: Matchmaking strategy ("round_robin", "random", "fitness_based")

        Returns:
            List of agent IDs to play in the game
        """
        if strategy == "round_robin":
            # Sample agents with lowest play counts to ensure even distribution
            sorted_agents = sorted(range(self.population_size), key=lambda i: self.match_counts[i])
            agents = sorted_agents[:self.num_agents_per_game]
            for agent_id in agents:
                self.match_counts[agent_id] += 1
            return agents

        elif strategy == "random":
            # Completely random sampling
            agents = random.sample(range(self.population_size), k=self.num_agents_per_game)
            for agent_id in agents:
                self.match_counts[agent_id] += 1
            return agents

        else:
            raise ValueError(f"Unknown matchmaking strategy: {strategy}")


class GameSimulator:
    """
    CPU-based game simulator for population training.

    Manages multiple parallel environments, coordinates agent matchmaking,
    collects experiences, and distributes them to GPU learners.
    """

    def __init__(
        self,
        scenario: Scenario,
        num_games: int = 64,
        population_size: int = 8,
        num_agents_per_game: int = 4,
        experience_queues: Optional[List[mp.Queue]] = None,
        policy_update_queue: Optional[mp.Queue] = None,
        matchmaking_strategy: str = "round_robin",
        seed: Optional[int] = None,
    ):
        """
        Initialize the game simulator.

        Args:
            scenario: BucketBrigade scenario configuration
            num_games: Number of parallel games to run
            population_size: Number of agents in the population
            num_agents_per_game: Number of agents per game (typically 4)
            experience_queues: List of queues for sending experiences to GPU learners
            policy_update_queue: Queue for receiving policy updates from GPU learners
            matchmaking_strategy: Strategy for selecting agents ("round_robin", "random")
            seed: Random seed for reproducibility
        """
        self.scenario = scenario
        self.num_games = num_games
        self.population_size = population_size
        self.num_agents_per_game = num_agents_per_game
        self.matchmaking_strategy = matchmaking_strategy

        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Create Rust environments
        print(f"Creating {num_games} Rust environments...")
        self.envs = [
            BucketBrigade(scenario, num_agents_per_game, seed=seed + i if seed else None)
            for i in range(num_games)
        ]

        # Cache scenario info for observation flattening
        self.scenario_info = create_scenario_info(scenario)

        # Initialize matchmaker
        self.matchmaker = Matchmaker(population_size, num_agents_per_game)

        # Communication queues
        self.experience_queues = experience_queues or []
        self.policy_update_queue = policy_update_queue

        # Policy repository (CPU-side copies of all agent policies)
        self.policies: Dict[int, torch.nn.Module] = {}

        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.episode_rewards: Dict[int, List[float]] = {i: [] for i in range(population_size)}

    def register_policy(self, agent_id: int, policy: torch.nn.Module):
        """Register a policy for an agent in the population."""
        self.policies[agent_id] = policy
        print(f"Registered policy for agent {agent_id}")

    def update_policy(self, agent_id: int, state_dict: dict):
        """Update an agent's policy with new weights from GPU learner."""
        if agent_id in self.policies:
            self.policies[agent_id].load_state_dict(state_dict)

    def select_action(self, agent_id: int, observation: np.ndarray) -> Tuple[int, int, float]:
        """
        Select action for an agent given observation.

        Args:
            agent_id: ID of the agent
            observation: Flattened observation array

        Returns:
            Tuple of (house_action, mode_action, log_prob)
        """
        policy = self.policies[agent_id]
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            action_logits, value = policy(obs_tensor)

            # Sample actions
            house_probs = torch.softmax(action_logits[0], dim=-1)
            mode_probs = torch.softmax(action_logits[1], dim=-1)

            house_dist = torch.distributions.Categorical(house_probs)
            mode_dist = torch.distributions.Categorical(mode_probs)

            house = house_dist.sample().item()
            mode = mode_dist.sample().item()

            # Calculate log probability
            house_logprob = house_dist.log_prob(torch.tensor(house))
            mode_logprob = mode_dist.log_prob(torch.tensor(mode))
            total_logprob = (house_logprob + mode_logprob).item()

        return house, mode, total_logprob

    def run_episode(self, env_id: int) -> Dict[int, List]:
        """
        Run one episode in a specific environment.

        Args:
            env_id: Index of environment to use

        Returns:
            Dictionary mapping agent_id to list of (obs, action, reward, next_obs, done, logprob) tuples
        """
        env = self.envs[env_id]

        # Matchmaking: select which agents will play
        agent_ids = self.matchmaker.sample_agents(self.matchmaking_strategy)

        # Reset environment
        env.reset()
        # Get observations for all agents (as flat arrays)
        observations = [
            flatten_observation(env.get_observation(i), self.scenario_info)
            for i in range(self.num_agents_per_game)
        ]

        # Storage for each agent's trajectory
        trajectories = {agent_id: [] for agent_id in agent_ids}

        done = False
        step = 0
        max_steps = 1000  # Safety limit

        while not done and step < max_steps:
            # Each agent selects an action
            actions = []
            logprobs = []

            for i, agent_id in enumerate(agent_ids):
                obs = observations[i]
                house, mode, logprob = self.select_action(agent_id, obs)
                actions.append([house, mode])
                logprobs.append(logprob)

            # Step environment (returns tuple: rewards_list, done, info)
            rewards_list, done, info = env.step(actions)
            next_observations = [
                flatten_observation(env.get_observation(i), self.scenario_info)
                for i in range(self.num_agents_per_game)
            ]

            # Record experience for each agent
            for i, agent_id in enumerate(agent_ids):
                experience = {
                    'obs': observations[i],
                    'action': actions[i],
                    'reward': rewards_list[i],
                    'next_obs': next_observations[i] if not done else None,
                    'done': done,
                    'logprob': logprobs[i],
                }
                trajectories[agent_id].append(experience)

            observations = next_observations
            step += 1

        # Update statistics
        self.total_steps += step
        self.total_episodes += 1

        for agent_id in agent_ids:
            episode_reward = sum(exp['reward'] for exp in trajectories[agent_id])
            self.episode_rewards[agent_id].append(episode_reward)

        return trajectories

    def distribute_experiences(self, trajectories: Dict[int, List]):
        """
        Send experiences to appropriate GPU learner queues.

        Args:
            trajectories: Dictionary mapping agent_id to list of experiences
        """
        for agent_id, experiences in trajectories.items():
            if agent_id < len(self.experience_queues):
                queue = self.experience_queues[agent_id]
                for exp in experiences:
                    queue.put((agent_id, exp))

    def check_policy_updates(self):
        """Check for and apply any policy updates from GPU learners."""
        if self.policy_update_queue is None:
            return

        updated = 0
        while not self.policy_update_queue.empty():
            agent_id, state_dict = self.policy_update_queue.get()
            self.update_policy(agent_id, state_dict)
            updated += 1

        if updated > 0:
            print(f"Applied {updated} policy updates from GPU learners")

    def run(self, num_episodes: int, update_interval: int = 10):
        """
        Main simulation loop.

        Args:
            num_episodes: Number of episodes to run
            update_interval: Check for policy updates every N episodes
        """
        print(f"\n{'='*60}")
        print(f"🎮 Starting Game Simulator")
        print(f"{'='*60}")
        print(f"Num games: {self.num_games}")
        print(f"Population size: {self.population_size}")
        print(f"Agents per game: {self.num_agents_per_game}")
        print(f"Matchmaking: {self.matchmaking_strategy}")
        print(f"Target episodes: {num_episodes}")
        print(f"{'='*60}\n")

        start_time = time.time()
        last_report = start_time

        for episode in range(num_episodes):
            # Run episode in a random environment
            env_id = random.randint(0, self.num_games - 1)
            trajectories = self.run_episode(env_id)

            # Distribute experiences to GPU learners
            self.distribute_experiences(trajectories)

            # Periodically check for policy updates
            if (episode + 1) % update_interval == 0:
                self.check_policy_updates()

                # Report progress
                elapsed = time.time() - last_report
                eps_per_sec = update_interval / elapsed

                # Calculate average rewards
                avg_rewards = {
                    agent_id: np.mean(rewards[-10:]) if rewards else 0.0
                    for agent_id, rewards in self.episode_rewards.items()
                }
                mean_reward = np.mean(list(avg_rewards.values()))

                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Speed: {eps_per_sec:.1f} eps/s | "
                      f"Avg Reward: {mean_reward:.3f} | "
                      f"Total Steps: {self.total_steps:,}")

                last_report = time.time()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Simulation Complete!")
        print(f"Total episodes: {self.total_episodes:,}")
        print(f"Total steps: {self.total_steps:,}")
        print(f"Time: {total_time/60:.1f} minutes")
        print(f"Speed: {self.total_episodes/total_time:.1f} eps/s")
        print(f"{'='*60}\n")

    def get_statistics(self) -> Dict:
        """Get simulation statistics."""
        return {
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'episode_rewards': self.episode_rewards,
            'match_counts': self.matchmaker.match_counts,
        }

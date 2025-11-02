"""
PufferLib-compatible environment wrapper for Bucket Brigade.

This wraps the BucketBrigadeEnv to work with PufferLib's multi-agent training framework.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, List, Any
import logging

from .bucket_brigade_env import BucketBrigadeEnv
from .scenarios import Scenario, default_scenario
from ..agents import create_random_agent, create_archetype_agent

logger = logging.getLogger(__name__)


class PufferBucketBrigade(gym.Env):
    """
    PufferLib-compatible wrapper for Bucket Brigade multi-agent environment.

    This environment trains a single agent while other agents are controlled
    by fixed policies (random, expert, etc.) assigned randomly each episode.
    """

    def __init__(
        self,
        scenario: Optional[Scenario] = None,
        num_opponents: int = 3,
        opponent_policies: Optional[List[str]] = None,
        max_steps: int = 50,
    ):
        """
        Initialize the PufferLib environment.

        Args:
            scenario: Game scenario (uses default if None)
            num_opponents: Number of opponent agents
            opponent_policies: List of opponent policy types
            max_steps: Maximum steps per episode
        """
        super().__init__()

        self.num_opponents = num_opponents
        self.num_agents = num_opponents + 1  # +1 for trained agent
        self.max_steps = max_steps
        self.steps_taken = 0

        # Default opponent policies
        if opponent_policies is None:
            opponent_policies = ["random", "random", "firefighter", "coordinator"]
        self.opponent_policies = opponent_policies[:num_opponents]

        # Create scenario
        if scenario is None:
            scenario = default_scenario(self.num_agents)
        self.scenario = scenario

        # Initialize underlying environment
        self.env = BucketBrigadeEnv(scenario)

        # Create opponent agents (will be refreshed each episode)
        self.opponent_agents: List[Any] = []

        # PufferLib observation/action spaces
        # Observation: [houses(10), agent_signals(N), agent_locations(N), last_actions(N,2), scenario_info(10)]
        obs_size = 10 + self.num_agents + self.num_agents + (self.num_agents * 2) + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: [house_index, mode] where house_index ∈ [0,9], mode ∈ [0,1]
        self.action_space = spaces.MultiDiscrete([10, 2])

        # Track episode statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        # Reset step counter
        self.steps_taken = 0

        # Create new opponent agents for this episode
        self._create_opponent_agents()

        # Reset underlying environment
        obs = self.env.reset(seed=seed)

        # Convert observation to flat array for PufferLib
        flat_obs = self._flatten_observation(obs, agent_id=0)

        return flat_obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.steps_taken += 1

        # Convert action to [house, mode] format
        agent_action = [int(action[0]), int(action[1])]

        # Get actions from all agents
        all_actions = [agent_action]  # Trained agent first

        # Get opponent actions
        current_obs = self.env._get_observation()
        for opponent in self.opponent_agents:
            opponent_action = opponent.act(current_obs)
            all_actions.append(opponent_action)

        # Step the environment
        obs, rewards, dones, info = self.env.step(np.array(all_actions))

        # Extract reward for trained agent (first agent)
        reward = float(rewards[0])

        # Check termination conditions
        terminated = self.env.done
        truncated = self.steps_taken >= self.max_steps

        # Convert observation for trained agent
        flat_obs = self._flatten_observation(obs, agent_id=0)

        return flat_obs, reward, terminated, truncated, info

    def _create_opponent_agents(self) -> None:
        """Create opponent agents for this episode."""
        self.opponent_agents = []

        for i, policy_type in enumerate(self.opponent_policies):
            agent_id = i + 1  # 0 is trained agent

            if policy_type == "random":
                agent = create_random_agent(agent_id)
            elif policy_type in [
                "firefighter",
                "coordinator",
                "free_rider",
                "liar",
                "hero",
            ]:
                agent = create_archetype_agent(policy_type, agent_id)
            else:
                # Default to random
                agent = create_random_agent(agent_id)

            self.opponent_agents.append(agent)

    def _flatten_observation(
        self, obs: Dict[str, np.ndarray], agent_id: int
    ) -> np.ndarray:
        """Convert observation dict to flat array for PufferLib."""
        houses = obs["houses"].astype(np.float32)
        signals = obs["signals"].astype(np.float32)
        locations = obs["locations"].astype(np.float32)
        last_actions = obs["last_actions"].flatten().astype(np.float32)
        scenario_info = obs["scenario_info"].astype(np.float32)

        # Concatenate all observation components
        flat_obs = np.concatenate(
            [
                houses,  # 10 values
                signals,  # N values
                locations,  # N values
                last_actions,  # N*2 values
                scenario_info,  # 10 values
            ]
        )

        return flat_obs

    def render(self) -> None:
        """Render the environment."""
        self.env.render()

    def close(self) -> None:
        """Close the environment."""
        pass


class PufferBucketBrigadeVectorized(PufferBucketBrigade):
    """
    Vectorized version for multiple parallel environments.
    """

    def __init__(self, num_envs: int = 1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.num_envs = num_envs

        # Vectorized observation space
        obs_size = 10 + self.num_agents + self.num_agents + (self.num_agents * 2) + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_envs, obs_size), dtype=np.float32
        )

        self.action_space = spaces.MultiDiscrete([num_envs, 10, 2])

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset all environments."""
        if seed is not None:
            np.random.seed(seed)

        # For simplicity, just reset the single environment
        # In a full implementation, this would handle multiple envs
        obs, info = super().reset(seed, options)
        return np.array([obs]), info

    def step(  # type: ignore[override]
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all environments."""
        # For simplicity, just step the single environment
        # In a full implementation, this would handle multiple envs
        obs, reward, terminated, truncated, info = super().step(actions[0])
        return (
            np.array([obs]),
            np.array([reward]),
            np.array([terminated]),
            np.array([truncated]),
            [info],
        )


# Factory functions for easy environment creation
def make_env(
    scenario_name: str = "default", num_opponents: int = 3
) -> PufferBucketBrigade:
    """Create a PufferLib environment with specified scenario."""

    # Import scenario functions locally to avoid circular imports
    from .scenarios import (
        trivial_cooperation_scenario,
        early_containment_scenario,
        greedy_neighbor_scenario,
        sparse_heroics_scenario,
        default_scenario,
    )

    # Map scenario names to functions
    scenario_map = {
        "default": default_scenario,
        "trivial_cooperation": trivial_cooperation_scenario,
        "early_containment": early_containment_scenario,
        "greedy_neighbor": greedy_neighbor_scenario,
        "sparse_heroics": sparse_heroics_scenario,
    }

    scenario_func = scenario_map.get(scenario_name, default_scenario)
    scenario = scenario_func(num_opponents + 1)  # +1 for trained agent

    return PufferBucketBrigade(scenario, num_opponents)


def make_vectorized_env(
    num_envs: int = 8, scenario_name: str = "default", num_opponents: int = 3
) -> PufferBucketBrigadeVectorized:
    """Create a vectorized environment for parallel training."""
    return PufferBucketBrigadeVectorized(
        num_envs=num_envs, scenario_name=scenario_name, num_opponents=num_opponents
    )

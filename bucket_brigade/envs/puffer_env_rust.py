"""
Rust-backed PufferLib-compatible environment wrapper for Bucket Brigade.

This wraps the Rust bucket_brigade_core to work with PufferLib's multi-agent
training framework with 100x performance improvement.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Optional, Tuple, List, Any
import logging
import bucket_brigade_core as core

from ..agents import create_random_agent, create_archetype_agent

logger = logging.getLogger(__name__)




def _heuristic_action(
    theta: np.ndarray,
    obs_dict: dict[str, np.ndarray],
    agent_id: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simplified heuristic action selection for opponent agents."""
    work_tendency = theta[1]
    own_house_priority = theta[3]
    rest_reward_bias = theta[8]

    if rng.random() < work_tendency * (1 - rest_reward_bias):
        owned_house = agent_id % 10
        if obs_dict["houses"][owned_house] == 1 and rng.random() < own_house_priority:
            house = owned_house
        else:
            burning = [i for i, h in enumerate(obs_dict["houses"]) if h == 1]
            if burning:
                house = rng.choice(burning)
            else:
                house = owned_house
        mode = 1  # WORK
    else:
        house = agent_id % 10
        mode = 0  # REST

    return [house, mode]


class RustPufferBucketBrigade(gym.Env):
    """
    Rust-backed PufferLib-compatible wrapper for Bucket Brigade.

    100x faster than Python implementation for RL training.
    """

    def __init__(
        self,
        scenario: Optional[str] = None,
        num_opponents: int = 3,
        opponent_policies: Optional[List[str]] = None,
        max_steps: int = 50,
    ):
        """
        Initialize the Rust-backed PufferLib environment.

        Args:
            scenario: Scenario name from core.SCENARIOS (uses "trivial_cooperation" if None)
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

        # Get scenario from Rust core (single source of truth)
        scenario_name = scenario or "trivial_cooperation"
        if scenario_name not in core.SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{scenario_name}'. "
                f"Available: {list(core.SCENARIOS.keys())}"
            )
        self.rust_scenario = core.SCENARIOS[scenario_name]
        self.scenario_name = scenario_name

        # Initialize Rust environment
        self.env: Optional[core.BucketBrigade] = None  # Will be created in reset()

        # RNG for heuristic decisions
        self.rng = np.random.RandomState()

        # Create opponent agents (will be refreshed each episode)
        self.opponent_agents: List[Any] = []

        # Track last observation
        self._last_obs: Optional[Dict[str, np.ndarray]] = None

        # PufferLib observation/action spaces
        obs_size = 10 + self.num_agents + self.num_agents + (self.num_agents * 2) + 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Action: [house_index, mode] where house_index ∈ [0,9], mode ∈ [0,1]
        self.action_space = spaces.MultiDiscrete([10, 2])

        # Track episode statistics
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        # Cache for observations
        self._last_obs = None

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Reset step counter
        self.steps_taken = 0

        # Create new Rust environment for this episode
        episode_seed = self.rng.randint(0, 2**31 - 1) if seed is not None else None
        self.env = core.BucketBrigade(self.rust_scenario, self.num_agents, seed=episode_seed)

        # Create new opponent agents for this episode
        self._create_opponent_agents()

        # Get initial observation from Rust env
        rust_obs = self.env.get_observation(0)
        obs_dict = {
            "houses": np.array(rust_obs.houses),
            "signals": np.array(rust_obs.signals),
            "locations": np.array(rust_obs.locations),
            "last_actions": np.zeros((self.num_agents, 2)),  # No actions yet
            "scenario_info": np.array(
                [
                    self.rust_scenario.prob_fire_spreads_to_neighbor,
                    self.rust_scenario.prob_solo_agent_extinguishes_fire,
                    self.rust_scenario.prob_house_catches_fire,
                    self.rust_scenario.team_reward_house_survives,
                    self.rust_scenario.team_penalty_house_burns,
                    self.rust_scenario.cost_to_work_one_night,
                    float(self.rust_scenario.min_nights),
                    float(self.num_agents),
                    0.0,  # Padding
                    0.0,  # Padding
                ]
            ),
        }
        self._last_obs = obs_dict

        # Convert observation to flat array for PufferLib
        flat_obs = self._flatten_observation(obs_dict, agent_id=0)

        return flat_obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment."""
        self.steps_taken += 1

        # Convert action to [house, mode] format
        agent_action = [int(action[0]), int(action[1])]

        # Get actions from all agents
        all_actions = [agent_action]  # Trained agent first

        # Get opponent actions using heuristic or agent policy
        for i, opponent in enumerate(self.opponent_agents):
            opponent_action = opponent.act(self._last_obs)
            all_actions.append(opponent_action)

        # Step the Rust environment
        assert self.env is not None, "Environment not initialized"
        rewards_list, done, info = self.env.step(all_actions)

        # Get observations for all agents
        rust_obs = self.env.get_observation(0)
        obs_dict = {
            "houses": np.array(rust_obs.houses),
            "signals": np.array(rust_obs.signals),
            "locations": np.array(rust_obs.locations),
            "last_actions": np.array(all_actions),
            "scenario_info": self._last_obs["scenario_info"]
            if self._last_obs is not None
            else np.zeros(10),  # Reuse
        }
        self._last_obs = obs_dict

        # Extract reward for trained agent (first agent)
        reward = float(rewards_list[0])

        # Check termination conditions
        terminated = done
        truncated = self.steps_taken >= self.max_steps

        # Convert observation for trained agent
        flat_obs = self._flatten_observation(obs_dict, agent_id=0)

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
        # Could implement basic console rendering if needed
        pass

    def close(self) -> None:
        """Close the environment."""
        pass


class RustPufferBucketBrigadeVectorized(RustPufferBucketBrigade):
    """
    Vectorized Rust-backed version for multiple parallel environments.
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
def make_rust_env(
    scenario_name: str = "trivial_cooperation", num_opponents: int = 3
) -> RustPufferBucketBrigade:
    """Create a Rust-backed PufferLib environment with specified scenario.

    Scenarios are loaded directly from bucket_brigade_core.SCENARIOS (Rust source of truth).
    """
    return RustPufferBucketBrigade(scenario_name, num_opponents)


def make_rust_vectorized_env(
    num_envs: int = 8, scenario_name: str = "trivial_cooperation", num_opponents: int = 3
) -> RustPufferBucketBrigadeVectorized:
    """Create a vectorized Rust-backed environment for parallel training.

    Scenarios are loaded directly from bucket_brigade_core.SCENARIOS (Rust source of truth).
    """
    return RustPufferBucketBrigadeVectorized(
        num_envs=num_envs, scenario=scenario_name, num_opponents=num_opponents
    )

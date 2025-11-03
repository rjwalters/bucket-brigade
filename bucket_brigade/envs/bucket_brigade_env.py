"""
Bucket Brigade multi-agent environment implementation.
"""

import numpy as np
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .scenarios import Scenario


class BucketBrigadeEnv:
    """
    Multi-agent environment implementing the Bucket Brigade firefighting game.

    Agents cooperate on a ring of 10 houses to prevent fires from spreading.
    The game ends when all fires are extinguished or all houses are ruined.
    """

    # House states
    SAFE = 0
    BURNING = 1
    RUINED = 2

    # Agent modes
    REST = 0
    WORK = 1

    def __init__(self, scenario: Optional[Scenario] = None, num_agents: int = 4):
        """
        Initialize the Bucket Brigade environment.

        Args:
            scenario: Game scenario parameters. If None, uses default scenario.
            num_agents: Number of agents (4-10). Ignored if scenario is provided.
        """
        if scenario is None:
            from .scenarios import default_scenario

            scenario = default_scenario(num_agents)

        self.scenario = scenario
        self.num_agents = scenario.num_agents

        # Environment state
        self.houses = np.zeros(10, dtype=np.int8)  # House states: SAFE, BURNING, RUINED
        self.locations = np.zeros(self.num_agents, dtype=np.int8)  # Agent positions
        self.signals = np.zeros(
            self.num_agents, dtype=np.int8
        )  # Agent signals: REST, WORK
        self.last_actions = np.zeros(
            (self.num_agents, 2), dtype=np.int8
        )  # [house, mode]

        # Game state
        self.night = 0
        self.done = False
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)

        # Trajectory recording for replays
        self.trajectory: List[Dict] = []

        # Previous house state for reward computation
        self._prev_houses_state: np.ndarray = np.zeros(10, dtype=np.int8)

        # House ownership: assign agents to houses in round-robin fashion
        self.house_owners = np.arange(10) % self.num_agents

        # Initialize random state for reproducibility
        self.rng = np.random.RandomState()

    def reset(self, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Reset the environment to start a new game.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Initial observation dictionary
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Reset game state
        self.night = 0
        self.done = False
        self.rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.trajectory = []

        # Initialize house states
        self._initialize_houses()

        # Initialize agent positions and signals
        self.locations = np.zeros(self.num_agents, dtype=np.int8)
        self.signals = np.zeros(self.num_agents, dtype=np.int8)
        self.last_actions = np.zeros((self.num_agents, 2), dtype=np.int8)

        # Record initial state
        self._record_night()

        return self._get_observation()

    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, Dict]:
        """
        Execute one night of the game.

        Args:
            actions: Per-agent actions, shape (N, 2) with [house_index, mode_flag]

        Returns:
            observation, rewards, dones, info
        """
        if self.done:
            raise RuntimeError("Game is already finished")

        # 1. Signal phase (signals are implicit in actions for now)
        # In this simplified version, we assume agents signal their intended mode
        self.signals = actions[:, 1].copy()  # mode_flag becomes the signal

        # 2. Action phase: update agent locations
        self.last_actions = actions.copy()
        self.locations = actions[:, 0].copy()

        # 3. Extinguish phase
        # Agents respond to fires visible at start of turn
        self._extinguish_fires(actions)

        # 4. Burn-out phase
        # Unextinguished fires become ruined houses
        self._burn_out_houses()

        # 5. Spread phase
        # Fires spread to neighbors (visible next turn)
        self._spread_fires()

        # 6. Spark phase (if active)
        # New fires ignite (visible next turn)
        if self.night < self.scenario.N_spark:
            self._spark_fires()

        # 7. Compute rewards
        self.rewards = self._compute_rewards(actions)

        # 8. Check termination
        self.done = self._check_termination()

        # 9. Record this night
        self._record_night()

        # 10. Advance to next night
        self.night += 1

        return (
            self._get_observation(),
            self.rewards.copy(),
            np.full(self.num_agents, self.done),
            {},
        )

    def _initialize_houses(self) -> None:
        """Initialize houses with some burning based on rho_ignite."""
        num_burning = int(np.round(self.scenario.rho_ignite * 10))
        burning_indices = self.rng.choice(10, size=num_burning, replace=False)
        self.houses = np.zeros(10, dtype=np.int8)
        self.houses[burning_indices] = self.BURNING

    def _extinguish_fires(self, actions: np.ndarray) -> None:
        """Extinguish fires based on worker presence and kappa parameter."""
        for house_idx in range(10):
            if self.houses[house_idx] != self.BURNING:
                continue

            # Count workers at this house
            workers_here = np.sum(
                (actions[:, 0] == house_idx) & (actions[:, 1] == self.WORK)
            )

            # Probability of extinguishing: 1 - exp(-kappa * k)
            p_extinguish = 1.0 - np.exp(-self.scenario.kappa * workers_here)

            if self.rng.random() < p_extinguish:
                self.houses[house_idx] = self.SAFE

    def _spread_fires(self) -> None:
        """Spread fires to neighboring safe houses."""
        # Burning houses try to ignite their safe neighbors
        for house_idx in range(10):
            if self.houses[house_idx] != self.BURNING:
                continue

            # Check both neighbors
            for neighbor_offset in [-1, 1]:
                neighbor_idx = (house_idx + neighbor_offset) % 10
                if self.houses[neighbor_idx] == self.SAFE:
                    if self.rng.random() < self.scenario.beta:
                        self.houses[neighbor_idx] = self.BURNING

    def _burn_out_houses(self) -> None:
        """Burning houses that neither extinguished nor spread become ruined."""
        # Any remaining burning houses become ruined
        burning_mask = self.houses == self.BURNING
        self.houses[burning_mask] = self.RUINED

    def _spark_fires(self) -> None:
        """Add spontaneous fires if spark phase is active."""
        for house_idx in range(10):
            if (
                self.houses[house_idx] == self.SAFE
                and self.rng.random() < self.scenario.p_spark
            ):
                self.houses[house_idx] = self.BURNING

    def _compute_rewards(self, actions: np.ndarray) -> np.ndarray:
        """Compute rewards for all agents."""
        # Store previous house states to compute changes
        prev_houses = (
            np.array(self._prev_houses_state)
            if hasattr(self, "_prev_houses_state")
            else self.houses.copy()
        )
        self._prev_houses_state = self.houses.copy()

        # Count final outcomes
        saved_houses = np.sum(self.houses == self.SAFE)
        ruined_houses = np.sum(self.houses == self.RUINED)
        total_saved_fraction = saved_houses / 10.0
        total_burned_fraction = ruined_houses / 10.0

        # Team reward component (shared by all)
        team_reward = (
            self.scenario.A * total_saved_fraction
            - self.scenario.L * total_burned_fraction
        )

        # Individual rewards
        individual_rewards = np.zeros(self.num_agents, dtype=np.float32)

        for agent_idx in range(self.num_agents):
            # Work/rest component
            if actions[agent_idx, 1] == self.WORK:
                individual_rewards[agent_idx] -= self.scenario.c  # Cost of working
            else:
                individual_rewards[agent_idx] += 0.5  # Rest reward

            # Ownership changes: bonus for owned houses that become safe
            owned_houses = np.where(self.house_owners == agent_idx)[0]
            for house_idx in owned_houses:
                if (
                    prev_houses[house_idx] != self.SAFE
                    and self.houses[house_idx] == self.SAFE
                ):
                    individual_rewards[agent_idx] += 1.0  # Bonus for saving owned house

            # Penalty for owned houses that are ruined
            owned_ruined = np.sum(self.houses[owned_houses] == self.RUINED)
            individual_rewards[agent_idx] -= 2.0 * owned_ruined

            # Team reward component (gamma = 0.1)
            individual_rewards[agent_idx] += 0.1 * team_reward

        return individual_rewards

    def _check_termination(self) -> bool:
        """Check if the game should end."""
        # Must play at least N_min nights
        if self.night < self.scenario.N_min:
            return False

        # End if all houses are safe or all are ruined
        all_safe = np.all(self.houses == self.SAFE)
        all_ruined = np.all(self.houses == self.RUINED)

        return bool(all_safe or all_ruined)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation for all agents."""
        return {
            "signals": self.signals.copy(),
            "locations": self.locations.copy(),
            "houses": self.houses.copy(),
            "last_actions": self.last_actions.copy(),
            "scenario_info": self.scenario.to_feature_vector(),
        }

    def _record_night(self) -> None:
        """Record current night state for replay."""
        night_data = {
            "night": self.night,
            "houses": self.houses.tolist(),
            "signals": self.signals.tolist(),
            "locations": self.locations.tolist(),
            "actions": self.last_actions.tolist(),
            "rewards": self.rewards.tolist(),
        }
        self.trajectory.append(night_data)

    def save_replay(self, path: str) -> None:
        """
        Save complete game trajectory as JSON.

        Args:
            path: File path to save replay
        """
        replay_data = {
            "scenario": {
                "beta": self.scenario.beta,
                "kappa": self.scenario.kappa,
                "A": self.scenario.A,
                "L": self.scenario.L,
                "c": self.scenario.c,
                "rho_ignite": self.scenario.rho_ignite,
                "N_min": self.scenario.N_min,
                "p_spark": self.scenario.p_spark,
                "N_spark": self.scenario.N_spark,
                "num_agents": self.scenario.num_agents,
            },
            "nights": self.trajectory,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(replay_data, f, indent=2)

    def render(self) -> None:
        """Simple text-based rendering for debugging."""
        house_symbols = ["□", "🔥", "💀"]  # SAFE, BURNING, RUINED

        print(f"Night {self.night}:")
        print("Houses:", "".join(house_symbols[state] for state in self.houses))
        print("Signals:", "".join("R" if s == self.REST else "W" for s in self.signals))
        print("Locations:", self.locations)
        print("Rewards:", self.rewards)
        print()

    @property
    def observation_space(self) -> Any:
        """Gym/PufferLib compatible observation space."""
        # This would need pufferlib import, simplified for now
        return None

    @property
    def action_space(self) -> Any:
        """Gym/PufferLib compatible action space."""
        # This would need pufferlib import, simplified for now
        return None

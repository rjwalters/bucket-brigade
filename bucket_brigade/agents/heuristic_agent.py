"""
Heuristic scripted agents with parameterized behaviors for Bucket Brigade.
"""

import numpy as np
from typing import Dict, Optional
from .agent_base import AgentBase


class HeuristicAgent(AgentBase):
    """
    Parameterized heuristic agent with diverse behavioral traits.

    The agent uses ~10 parameters to control signaling, house selection, and risk preferences.
    """

    def __init__(self, params: np.ndarray, agent_id: int, name: Optional[str] = None):
        """
        Initialize heuristic agent with parameter vector.

        Args:
            params: Parameter vector of length ~10 controlling behavior
            agent_id: Unique agent identifier
            name: Optional agent name
        """
        super().__init__(agent_id, name)

        # Parameter vector (length 10)
        self.params = params.copy()
        assert len(params) == 10, f"Expected 10 parameters, got {len(params)}"

        # Unpack parameters with descriptive names
        self.honesty_bias = params[0]           # Probability of truthful signaling (0-1)
        self.work_tendency = params[1]          # Base tendency to work (0-1)
        self.neighbor_help_bias = params[2]     # Preference for helping neighbor houses (0-1)
        self.own_house_priority = params[3]     # Priority for own house (0-1)
        self.risk_aversion = params[4]          # Sensitivity to burning houses (0-1)
        self.coordination_weight = params[5]    # Trust in others' signals (0-1)
        self.exploration_rate = params[6]       # Randomness in decisions (0-1)
        self.fatigue_memory = params[7]         # Inertia to repeat actions (0-1)
        self.rest_reward_bias = params[8]       # Preference for resting (0-1)
        self.altruism_factor = params[9]        # Willingness to help others (0-1)

        # Internal state
        self.last_action = None

    def reset(self):
        """Reset agent state between games."""
        self.last_action = None

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose action based on observations and parameters.

        Returns:
            Action [house_index, mode_flag]
        """
        # Extract observation components
        signals = obs['signals']           # (N,) - current signals
        locations = obs['locations']       # (N,) - current locations
        houses = obs['houses']             # (10,) - house states
        last_actions = obs['last_actions'] # (N,2) - previous actions
        scenario_info = obs['scenario_info']  # Scenario parameters

        # Determine owned house
        owned_house = self.agent_id % 10

        # 1. Signal selection (internal intent)
        work_intent = self._compute_work_intent(houses, scenario_info)
        signal = self._choose_signal(work_intent)

        # 2. Action selection
        house_choice = self._choose_house(houses, signals, locations, owned_house, scenario_info)
        mode_choice = self._choose_mode(signal, work_intent)

        # 3. Apply exploration
        if np.random.random() < self.exploration_rate:
            house_choice = np.random.randint(10)
            mode_choice = np.random.randint(2)

        # 4. Apply fatigue memory (tendency to repeat)
        if self.last_action is not None and np.random.random() < self.fatigue_memory:
            house_choice, mode_choice = self.last_action

        # Store for next time
        self.last_action = (house_choice, mode_choice)

        return np.array([house_choice, mode_choice], dtype=np.int8)

    def _compute_work_intent(self, houses: np.ndarray, scenario_info: np.ndarray) -> float:
        """Compute internal tendency to work this night."""
        burning_fraction = np.mean(houses == 1)  # Fraction of burning houses
        owned_house = self.agent_id % 10
        own_house_burning = houses[owned_house] == 1

        # Base work tendency
        intent = self.work_tendency

        # Adjust for risk (more burning houses -> more likely to work)
        intent += self.risk_aversion * burning_fraction

        # Adjust for own house
        if own_house_burning:
            intent += self.own_house_priority * 0.5

        # Adjust for rest preference
        intent *= (1.0 - self.rest_reward_bias)

        return np.clip(intent, 0.0, 1.0)

    def _choose_signal(self, work_intent: float) -> int:
        """Choose whether to signal WORK or REST."""
        if np.random.random() < self.honesty_bias:
            # Honest signaling
            return 1 if work_intent > 0.5 else 0
        else:
            # Dishonest signaling - opposite of intent
            return 0 if work_intent > 0.5 else 1

    def _choose_house(self, houses: np.ndarray, signals: np.ndarray, locations: np.ndarray,
                     owned_house: int, scenario_info: np.ndarray) -> int:
        """Choose which house to target."""
        candidates = []

        # Evaluate each house
        for house_idx in range(10):
            score = 0.0

            # Distance from owned house (prefer closer houses)
            distance = min(abs(house_idx - owned_house),
                          10 - abs(house_idx - owned_house))
            proximity_bonus = 1.0 / (1.0 + distance)
            score += proximity_bonus * 0.1

            # Own house priority
            if house_idx == owned_house:
                score += self.own_house_priority

            # Burning house priority
            if houses[house_idx] == 1:  # Burning
                score += 0.5

            # Neighbor help bias
            neighbors = [(house_idx - 1) % 10, (house_idx + 1) % 10]
            neighbor_burning = any(houses[n] == 1 for n in neighbors)
            if neighbor_burning:
                score += self.neighbor_help_bias * 0.3

            # Coordination: houses where others are currently located (simplified coordination)
            agents_here = np.sum(locations == house_idx)
            score += self.coordination_weight * agents_here * 0.1

            # Risk aversion: avoid houses near many fires
            nearby_fires = sum(1 for n in neighbors if houses[n] == 1)
            score -= self.risk_aversion * nearby_fires * 0.2

            candidates.append((house_idx, score))

        # Select house proportionally to scores
        houses_list, scores = zip(*candidates)
        houses_array = np.array(houses_list)
        scores = np.array(scores)
        scores = np.maximum(scores, 0.01)  # Ensure positive probabilities

        # Normalize to probabilities
        probs = scores / np.sum(scores)

        return int(np.random.choice(houses_array, p=probs))

    def _choose_mode(self, signal: int, work_intent: float) -> int:
        """Choose whether to actually work or rest."""
        if signal == 1:  # Signaled work
            # Follow signal with honesty probability, otherwise rest
            return 1 if np.random.random() < self.honesty_bias else 0
        else:  # Signaled rest
            # Sometimes work anyway (false modesty) based on altruism
            return 1 if np.random.random() < (work_intent * self.altruism_factor) else 0


def create_random_agent(agent_id: int, name: Optional[str] = None) -> HeuristicAgent:
    """Create a random heuristic agent."""
    params = np.random.uniform(0, 1, 10)
    return HeuristicAgent(params, agent_id, name)


def create_archetype_agent(archetype: str, agent_id: int) -> HeuristicAgent:
    """
    Create an agent with predefined archetypal parameters.

    Args:
        archetype: One of 'firefighter', 'liar', 'free_rider', 'hero', 'coordinator'
        agent_id: Agent ID
    """
    archetypes = {
        'firefighter': [1.0, 0.9, 0.5, 0.8, 0.5, 0.7, 0.1, 0.0, 0.0, 0.8],  # Honest, works a lot, helps neighbors
        'liar': [0.1, 0.7, 0.0, 0.9, 0.2, 0.8, 0.3, 0.0, 0.4, 0.2],          # Dishonest, selfish
        'free_rider': [0.7, 0.2, 0.0, 0.9, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0],    # Rarely works, protects own house
        'hero': [1.0, 1.0, 1.0, 0.5, 0.1, 0.5, 0.0, 0.9, 0.0, 1.0],          # Always works, helps everyone
        'coordinator': [0.9, 0.6, 0.7, 0.6, 0.8, 1.0, 0.05, 0.0, 0.2, 0.6]   # Coordinates with others
    }

    if archetype not in archetypes:
        raise ValueError(f"Unknown archetype: {archetype}")

    params = np.array(archetypes[archetype])
    name = f"{archetype.title()}-{agent_id}"
    return HeuristicAgent(params, agent_id, name)

"""
Optimal agent for Early Containment scenario.

In this scenario, fires start aggressively but can be stopped with early
coordination. Delay causes cascading failure. The optimal strategy is to
work early and focus on containing fire clusters.
"""

import numpy as np
from typing import Dict
from ..agent_base import AgentBase


class EarlyContainmentAgent(AgentBase):
    """
    Works aggressively early in the game to contain fire clusters.

    Monitors fire spread patterns and prioritizes containing clusters
    of connected burning houses.
    """

    def __init__(self, agent_id: int, name: str = "EarlyContainment"):
        super().__init__(agent_id, name)
        self.night_count = 0

    def reset(self):
        """Reset night counter."""
        self.night_count = 0

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Early game: Always work on burning houses.
        Later game: Rest if no fires remain.
        """
        self.night_count += 1
        houses = obs['houses']

        # Find burning houses
        burning_houses = np.where(houses == 1)[0]

        if len(burning_houses) > 0:
            # Work on burning houses, prioritizing clusters
            target_house = self._select_best_burning_house(houses, burning_houses)
            return np.array([target_house, 1])  # [house, WORK]
        else:
            # No fires - rest to save costs for potential future fires
            own_house = self.agent_id % 10
            return np.array([own_house, 0])  # [house, REST]

    def _select_best_burning_house(self, houses: np.ndarray, burning_houses: np.ndarray) -> int:
        """
        Select the best burning house to work on.

        Prioritizes houses that are part of clusters (have burning neighbors)
        to contain fire spread.
        """
        if len(burning_houses) == 1:
            return burning_houses[0]

        # Score each burning house by cluster size
        best_house = burning_houses[0]
        best_score = 0

        for house_idx in burning_houses:
            # Count burning neighbors
            neighbors = [(house_idx - 1) % 10, (house_idx + 1) % 10]
            cluster_size = 1  # Count self

            for neighbor in neighbors:
                if houses[neighbor] == 1:  # Burning neighbor
                    cluster_size += 1

            if cluster_size > best_score:
                best_score = cluster_size
                best_house = house_idx

        return best_house


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return EarlyContainmentAgent(agent_id, **kwargs)

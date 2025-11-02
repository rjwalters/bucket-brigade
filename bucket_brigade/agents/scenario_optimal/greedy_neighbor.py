"""
Optimal agent for Greedy Neighbor scenario.

In this scenario, fires spread slowly but helping neighbors costs valuable
rest time. Selfish agents can rely on others to work. The optimal strategy
balances self-interest with necessary cooperation.
"""

import numpy as np
from typing import Dict
from ..agent_base import AgentBase


class GreedyNeighborAgent(AgentBase):
    """
    Selfish agent that prioritizes own house but helps neighbors when necessary.

    Only helps others when adjacent fires threaten personal property.
    Otherwise focuses on maximizing own rewards.
    """

    def __init__(self, agent_id: int, name: str = "GreedyNeighbor"):
        super().__init__(agent_id, name)
        self.own_house = agent_id % 10

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Prioritize own house, but help neighbors if they threaten you.
        """
        houses = obs['houses']

        # Check if own house is burning
        if houses[self.own_house] == 1:
            return np.array([self.own_house, 1])  # Work on own house

        # Check if neighbor houses are burning (could spread to you)
        neighbors = [(self.own_house - 1) % 10, (self.own_house + 1) % 10]
        threatening_neighbors = [n for n in neighbors if houses[n] == 1]

        if threatening_neighbors:
            # Help neighbor to prevent spread to own house
            return np.array([threatening_neighbors[0], 1])  # Work on first threatening neighbor

        # No immediate threats - rest to save costs
        return np.array([self.own_house, 0])  # Rest at own house


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return GreedyNeighborAgent(agent_id, **kwargs)

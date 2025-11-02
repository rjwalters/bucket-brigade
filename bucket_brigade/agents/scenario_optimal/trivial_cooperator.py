"""
Optimal agent for Trivial Cooperation scenario.

In this scenario, fires are rare and extinguish easily. The optimal strategy
is to always work honestly and help extinguish any fires that appear.
"""

import numpy as np
from typing import Dict
from ..agent_base import AgentBase


class TrivialCooperator(AgentBase):
    """
    Always works honestly on any burning houses.

    This is optimal in scenarios where work costs are low relative to
    the benefits of cooperation, and fires are manageable.
    """

    def __init__(self, agent_id: int, name: str = "TrivialCooperator"):
        super().__init__(agent_id, name)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Always signal WORK and work on the first available burning house.
        """
        houses = obs["houses"]

        # Find first burning house (0=Safe, 1=Burning, 2=Ruined)
        burning_houses = np.where(houses == 1)[0]

        if len(burning_houses) > 0:
            # Work on first burning house
            target_house = burning_houses[0]
            return np.array([target_house, 1])  # [house, WORK]
        else:
            # No fires, work on own house (house index = agent_id % 10)
            own_house = self.agent_id % 10
            return np.array([own_house, 1])  # [house, WORK]


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return TrivialCooperator(agent_id, **kwargs)

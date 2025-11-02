"""
Optimal agent for Sparse Heroics scenario.

In this scenario, few workers can make the difference. Fires spread slowly
but do not stop spontaneously. The optimal strategy dispatches minimal
firefighters efficiently.
"""

import numpy as np
from typing import Dict
from ..agent_base import AgentBase


class SparseHeroAgent(AgentBase):
    """
    Efficiently allocates work when fires are present, rests when not needed.

    Monitors fire count and only works when necessary to contain spread.
    Avoids overworking while ensuring fires don't get out of control.
    """

    def __init__(self, agent_id: int, name: str = "SparseHero"):
        super().__init__(agent_id, name)
        self.own_house = agent_id % 10

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Work only when fires need attention, rest otherwise.
        """
        houses = obs['houses']
        burning_count = np.sum(houses == 1)

        # If no fires, rest
        if burning_count == 0:
            return np.array([self.own_house, 0])  # Rest

        # If fires exist, work on the most critical one
        burning_houses = np.where(houses == 1)[0]

        # Prioritize own house if burning
        if houses[self.own_house] == 1:
            return np.array([self.own_house, 1])

        # Otherwise work on first burning house (simple strategy)
        # In a more sophisticated version, could consider distance or cluster size
        target_house = burning_houses[0]
        return np.array([target_house, 1])  # Work on fire


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return SparseHeroAgent(agent_id, **kwargs)

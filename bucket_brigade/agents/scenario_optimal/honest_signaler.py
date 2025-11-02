"""
Optimal agent for Deceptive Calm scenario.

In this scenario, honest signaling provides advantage during occasional
flare-ups. The optimal strategy signals truthfully and reacts quickly
to new fires.
"""

import numpy as np
from typing import Dict
from ..agent_base import AgentBase


class HonestSignaler(AgentBase):
    """
    Always signals truthfully and responds quickly to fire changes.

    Monitors house states across nights to detect new fires and responds
    immediately to contain them.
    """

    def __init__(self, agent_id: int, name: str = "HonestSignaler"):
        super().__init__(agent_id, name)
        self.own_house = agent_id % 10
        self.last_houses = None

    def reset(self):
        """Reset house state tracking."""
        self.last_houses = None

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Signal truthfully and work on any new or existing fires.
        """
        houses = obs['houses']

        # Check for new fires since last turn
        new_fires = []
        if self.last_houses is not None:
            for i, (current, last) in enumerate(zip(houses, self.last_houses)):
                if current == 1 and last != 1:  # New fire
                    new_fires.append(i)

        self.last_houses = houses.copy()

        # Find all current fires
        burning_houses = np.where(houses == 1)[0]

        if len(burning_houses) > 0:
            # Prioritize new fires, then any burning house
            target_house = new_fires[0] if new_fires else burning_houses[0]

            # Honest signaling: signal matches action
            return np.array([target_house, 1])  # Signal WORK, work
        else:
            # No fires - rest honestly
            return np.array([self.own_house, 0])  # Signal REST, rest


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return HonestSignaler(agent_id, **kwargs)

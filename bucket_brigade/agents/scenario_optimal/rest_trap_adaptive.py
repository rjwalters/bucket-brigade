"""
Optimal agent for Rest Trap scenario.

In this scenario, fires almost always extinguish by themselves, but rare
long fires can destroy the town if everyone rests too long. The optimal
strategy rests initially but mobilizes when persistent fires are detected.
"""

import numpy as np
from typing import Dict
from ..agent_base import AgentBase


class RestTrapAdaptiveAgent(AgentBase):
    """
    Rests initially but adapts when persistent fires are detected.

    Monitors fire persistence across nights and mobilizes when fires
    survive longer than expected.
    """

    def __init__(self, agent_id: int, name: str = "RestTrapAdaptive"):
        super().__init__(agent_id, name)
        self.own_house = agent_id % 10
        self.fire_persistence = {}  # Track how long fires have been burning

    def reset(self):
        """Reset fire tracking state."""
        self.fire_persistence = {}

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Rest unless fires show unusual persistence.
        """
        houses = obs["houses"]
        current_burning = set(np.where(houses == 1)[0])

        # Update persistence tracking
        # Remove fires that are no longer burning
        self.fire_persistence = {
            house: nights
            for house, nights in self.fire_persistence.items()
            if house in current_burning
        }

        # Increment persistence for current fires
        for house in current_burning:
            if house not in self.fire_persistence:
                self.fire_persistence[house] = 0
            self.fire_persistence[house] += 1

        # Check for persistent fires (burning > 2 nights)
        persistent_fires = [
            house for house, nights in self.fire_persistence.items() if nights > 2
        ]

        if persistent_fires:
            # Emergency: work on persistent fire
            target_house = persistent_fires[0]  # Work on first persistent fire
            return np.array([target_house, 1])  # Work
        else:
            # Normal case: rest and let fires extinguish naturally
            return np.array([self.own_house, 0])  # Rest


# For agent submission system
def create_agent(agent_id: int, **kwargs):
    """Factory function for creating this agent type."""
    return RestTrapAdaptiveAgent(agent_id, **kwargs)

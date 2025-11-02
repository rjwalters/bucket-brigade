"""
Base class for all agents in the Bucket Brigade environment.
Provides a unified interface compatible with both scripted heuristics and learned policies.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class AgentBase(ABC):
    """
    Abstract base class for all agents participating in Bucket Brigade games.

    Agents must implement the act() method to choose actions based on observations.
    The reset() method allows agents to clear internal state between games.
    """

    def __init__(self, agent_id: int, name: Optional[str] = None):
        """
        Initialize the agent.

        Args:
            agent_id: Unique identifier for this agent
            name: Optional human-readable name for the agent
        """
        self.agent_id = agent_id
        self.name = name or f"Agent-{agent_id}"

    def reset(self) -> None:
        """
        Reset agent internal state between games.
        Default implementation does nothing; subclasses can override.
        """
        pass

    @abstractmethod
    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose an action based on the current observation.

        Args:
            obs: Dictionary containing observation data:
                - "signals": np.ndarray[int8], shape (N,), current signals
                - "locations": np.ndarray[int8], shape (N,), current agent positions
                - "houses": np.ndarray[int8], shape (10,), house states
                - "last_actions": np.ndarray[int8], shape (N,2), previous-night actions
                - "scenario_info": np.ndarray[float32], shape (k,), scenario vector

        Returns:
            Action as np.ndarray[int8], shape (2,):
                [house_index, mode_flag]
                - house_index ∈ [0..9]
                - mode_flag ∈ {0=REST, 1=WORK}
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, name='{self.name}')"


class RandomAgent(AgentBase):
    """
    Baseline agent that chooses actions randomly.
    Useful for testing and as a performance baseline.
    """

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Choose a random house and random mode (work/rest).
        """
        house_index = np.random.randint(0, 10)  # Random house 0-9
        mode_flag = np.random.randint(0, 2)      # Random mode 0=REST, 1=WORK
        return np.array([house_index, mode_flag], dtype=np.int8)

"""
Base class for all agents in the Bucket Brigade environment.
Provides a unified interface compatible with both scripted heuristics and learned policies.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional


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
            Action as np.ndarray[int8], shape (3,) per issue #235:
                ``[house_index, mode_flag, signal]``
                - ``house_index`` in [0..9]
                - ``mode_flag`` in {0=REST, 1=WORK} — the action actually taken
                - ``signal`` in {0=REST, 1=WORK} — the broadcast intent
                  (may differ from ``mode_flag``; i.e. a lie)

            Honest agents simply emit ``signal == mode_flag``.
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

        Issue #235: the random baseline is **honest** by default —
        ``signal == mode_flag``. A truly random-over-the-full-action-space
        baseline would lie 25% of the time, which is research-meaningful;
        keeping the random baseline honest preserves the prior-PR semantic
        of "uniform-random house, uniform-random mode" and isolates the
        signal channel as a deliberate degree of freedom available to
        strategic agents.
        """
        house_index = np.random.randint(0, 10)  # Random house 0-9
        mode_flag = np.random.randint(0, 2)  # Random mode 0=REST, 1=WORK
        signal = int(mode_flag)  # Honest signal — see docstring above.
        return np.array([house_index, mode_flag, signal], dtype=np.int8)

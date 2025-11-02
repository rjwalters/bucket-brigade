"""
Agent Submission Template for Bucket Brigade

This template shows the required interface for submitting agents to the
Bucket Brigade platform. Your agent must implement the functions below.

IMPORTANT SECURITY NOTES:
- Your code will be executed in isolated environments
- Malicious code will be detected and rejected
- Only standard library imports are allowed
- No network access, file system access, or system calls

AGENT INTERFACE:
Your agent must be a class that inherits from or implements the same
interface as AgentBase. It must have:

1. __init__(self, agent_id: int, name: str = "MyAgent")
2. reset(self) -> None  # Called between games
3. act(self, obs: Dict[str, np.ndarray]) -> np.ndarray  # Return [house, mode]

OBSERVATION FORMAT:
obs = {
    'signals': np.ndarray[int8], shape (N,) - Other agents' signals
    'locations': np.ndarray[int8], shape (N,) - Other agents' locations
    'houses': np.ndarray[int8], shape (10,) - House states (0=Safe, 1=Burning, 2=Ruined)
    'last_actions': np.ndarray[int8], shape (N,2) - Last night actions [house, mode]
    'scenario_info': np.ndarray[float32], shape (k,) - Scenario parameters
}

ACTION FORMAT:
Return np.array([house_index, mode_flag])
- house_index: int 0-9 (which house to target)
- mode_flag: int 0=REST, 1=WORK

EXAMPLE IMPLEMENTATION:
"""

import numpy as np
from typing import Dict
from .agent_base import AgentBase


class MyCustomAgent(AgentBase):
    """
    Example custom agent implementation.

    Replace this with your own agent logic.
    """

    def __init__(self, agent_id: int, name: str = "MyAgent"):
        super().__init__(agent_id, name)
        # Initialize your agent's state here
        self.my_state_variable = 0

    def reset(self):
        """Reset agent state between games."""
        self.my_state_variable = 0

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Implement your agent's decision logic here.

        Args:
            obs: Current observation dictionary

        Returns:
            Action as [house_index, mode_flag]
        """
        houses = obs["houses"]
        own_house = self.agent_id % 10

        # Example strategy: Work on own house if burning, otherwise rest
        if houses[own_house] == 1:  # Own house is burning
            return np.array([own_house, 1])  # Work on own house
        else:
            return np.array([own_house, 0])  # Rest at own house


# REQUIRED: Factory function for agent creation
def create_agent(agent_id: int, **kwargs):
    """
    Factory function that creates an instance of your agent.

    Args:
        agent_id: Unique identifier for this agent
        **kwargs: Additional parameters (name, etc.)

    Returns:
        Instance of your agent class
    """
    return MyCustomAgent(agent_id, **kwargs)


# OPTIONAL: Agent metadata for the submission system
AGENT_METADATA = {
    "name": "My Custom Agent",
    "author": "Your Name",
    "description": "Brief description of your agent strategy",
    "version": "1.0.0",
    "tags": ["example", "template"],  # Categories for your agent
}

# OPTIONAL: Parameter ranges for hyperparameter optimization
AGENT_PARAMETERS = {
    # If your agent has configurable parameters, define them here
    # 'learning_rate': {'min': 0.001, 'max': 0.1, 'default': 0.01},
}

"""
SUBMISSION INSTRUCTIONS:

1. Copy this template to a new file (e.g., my_agent.py)
2. Implement your agent class with the required interface
3. Update the create_agent() function
4. Fill in AGENT_METADATA with your information
5. Test your agent locally before submission
6. Submit your .py file through the platform interface

TESTING YOUR AGENT:

from bucket_brigade.envs import BucketBrigadeEnv, default_scenario
from my_agent import create_agent

# Create environment
env = BucketBrigadeEnv(default_scenario(4))
obs = env.reset()

# Create your agent
agent = create_agent(0, name="TestAgent")

# Test action generation
action = agent.act(obs)
print(f"Action: {action}")  # Should be [house, mode]

VALIDATION CHECKLIST:
- [ ] Agent inherits from AgentBase or implements same interface
- [ ] create_agent() function returns agent instance
- [ ] act() returns numpy array of shape (2,) with valid values
- [ ] No external imports beyond standard library + numpy
- [ ] Agent handles observation format correctly
- [ ] Agent works with different agent_id values
- [ ] reset() method properly clears state
"""

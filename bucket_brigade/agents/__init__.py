"""
Agents for the Bucket Brigade environment.
"""

from .agent_base import AgentBase, RandomAgent
from .heuristic_agent import HeuristicAgent, create_random_agent, create_archetype_agent

__all__ = [
    # Base classes
    "AgentBase",
    "RandomAgent",
    "HeuristicAgent",
    "TrainedPolicyArchetype",
    # Factory functions
    "create_random_agent",
    "create_archetype_agent",
]


def __getattr__(name):
    if name == "TrainedPolicyArchetype":
        from .trained_policy_archetype import TrainedPolicyArchetype

        return TrainedPolicyArchetype
    raise AttributeError(f"module 'bucket_brigade.agents' has no attribute {name!r}")

"""Reinforcement learning training utilities.

This package provides modular components for training agents using various
RL algorithms and curriculum learning strategies.
"""

from .networks import (
    HindsightNetwork,
    PolicyNetwork,
    TransformerPolicyNetwork,
    compute_gae,
    compute_hca_advantages,
    compute_returns_to_go,
    encode_return_bucket,
)
from .game_simulator import GameSimulator, Matchmaker
from .policy_learner import PolicyLearner, learner_process
from .population_trainer import PopulationTrainer
from .observation_utils import (
    flatten_observation,
    get_observation_dim,
    create_scenario_info,
)

__all__ = [
    "PolicyNetwork",
    "TransformerPolicyNetwork",
    "HindsightNetwork",
    "compute_gae",
    "compute_hca_advantages",
    "compute_returns_to_go",
    "encode_return_bucket",
    "GameSimulator",
    "Matchmaker",
    "PolicyLearner",
    "learner_process",
    "PopulationTrainer",
    "flatten_observation",
    "get_observation_dim",
    "create_scenario_info",
]

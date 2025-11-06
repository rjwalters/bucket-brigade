"""Reinforcement learning training utilities.

This package provides modular components for training agents using various
RL algorithms and curriculum learning strategies.
"""

from .curriculum import CurriculumStage, CurriculumTrainer
from .networks import PolicyNetwork, TransformerPolicyNetwork, compute_gae
from .game_simulator import GameSimulator, Matchmaker
from .policy_learner import PolicyLearner, learner_process
from .population_trainer import PopulationTrainer
from .observation_utils import flatten_observation, get_observation_dim, create_scenario_info

__all__ = [
    "PolicyNetwork",
    "TransformerPolicyNetwork",
    "CurriculumTrainer",
    "CurriculumStage",
    "compute_gae",
    "GameSimulator",
    "Matchmaker",
    "PolicyLearner",
    "learner_process",
    "PopulationTrainer",
    "flatten_observation",
    "get_observation_dim",
    "create_scenario_info",
]

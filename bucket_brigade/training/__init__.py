"""Reinforcement learning training utilities.

This package provides modular components for training agents using various
RL algorithms and curriculum learning strategies.
"""

from .curriculum import CurriculumStage, CurriculumTrainer
from .networks import PolicyNetwork, compute_gae
from .game_simulator import GameSimulator, Matchmaker
from .policy_learner import PolicyLearner, learner_process
from .population_trainer import PopulationTrainer

__all__ = [
    "PolicyNetwork",
    "CurriculumTrainer",
    "CurriculumStage",
    "compute_gae",
    "GameSimulator",
    "Matchmaker",
    "PolicyLearner",
    "learner_process",
    "PopulationTrainer",
]

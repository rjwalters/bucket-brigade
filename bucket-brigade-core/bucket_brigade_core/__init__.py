"""
Bucket Brigade Core - High-performance Rust implementation

This module provides a fast Rust implementation of the Bucket Brigade
multi-agent cooperation environment, compatible with Python through PyO3.
"""

from .bucket_brigade_core import (
    PyBucketBrigade as BucketBrigade,
    PyScenario as Scenario,
    PyAgentObservation as AgentObservation,
    PyGameState as GameState,
    PyGameResult as GameResult,
    PyVectorEnv as VectorEnv,
    SCENARIOS,
    run_heuristic_episode,
    run_heuristic_episode_focal,
)

__all__ = [
    "BucketBrigade",
    "Scenario",
    "AgentObservation",
    "GameState",
    "GameResult",
    "VectorEnv",
    "SCENARIOS",
    "run_heuristic_episode",
    "run_heuristic_episode_focal",
]

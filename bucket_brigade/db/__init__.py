"""
Lightweight experiment tracking for RL training runs.

This module provides SQLite-based tracking of training experiments,
separate from the agent registry functionality.
"""

from .experiments import (
    init_experiments_db,
    get_experiments_session,
    create_experiment_run,
    complete_experiment_run,
    log_training_metric,
    log_evaluation_result,
    find_run_by_model_path,
)
from .models import ExperimentRun, TrainingMetric, EvaluationResult

__all__ = [
    "init_experiments_db",
    "get_experiments_session",
    "create_experiment_run",
    "complete_experiment_run",
    "log_training_metric",
    "log_evaluation_result",
    "find_run_by_model_path",
    "ExperimentRun",
    "TrainingMetric",
    "EvaluationResult",
]

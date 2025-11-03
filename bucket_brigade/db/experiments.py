"""
Lightweight experiment tracking for RL training runs.

Provides SQLite-based tracking of training runs, metrics, and evaluation results.
Separate from the main PostgreSQL agent registry database.
"""

import os
from pathlib import Path
from typing import Generator, Optional
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base, ExperimentRun, TrainingMetric, EvaluationResult

# Default SQLite database location
DEFAULT_EXPERIMENTS_DB = "data/experiments.db"


def get_experiments_db_path() -> str:
    """Get the path to the experiments database."""
    db_path = os.getenv("EXPERIMENTS_DB", DEFAULT_EXPERIMENTS_DB)
    return db_path


def init_experiments_db(db_path: Optional[str] = None) -> Session:
    """
    Initialize experiments database and return a session.

    Args:
        db_path: Path to SQLite database file (defaults to EXPERIMENTS_DB env var)

    Returns:
        SQLAlchemy Session instance
    """
    if db_path is None:
        db_path = get_experiments_db_path()

    # Create parent directory if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create engine (SQLite doesn't need connection pooling)
    engine = create_engine(
        f"sqlite:///{db_path}",
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
    )

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    return SessionLocal()


def get_experiments_session(db_path: Optional[str] = None) -> Generator[Session, None, None]:
    """
    Get a database session for experiment tracking.

    Args:
        db_path: Path to SQLite database file (defaults to EXPERIMENTS_DB env var)

    Yields:
        SQLAlchemy Session instance

    Usage:
        with get_experiments_session() as session:
            # Use session here
            pass
    """
    session = init_experiments_db(db_path)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_experiment_run(
    session: Session,
    run_name: str,
    scenario: str,
    hyperparameters: dict,
    model_path: str,
) -> ExperimentRun:
    """
    Create a new experiment run record.

    Args:
        session: Database session
        run_name: Unique name for this run
        scenario: Scenario name
        hyperparameters: Dict of training hyperparameters
        model_path: Path where model will be saved

    Returns:
        Created ExperimentRun instance
    """
    run = ExperimentRun(
        run_name=run_name,
        scenario=scenario,
        hyperparameters=hyperparameters,
        model_path=model_path,
        started_at=datetime.utcnow(),
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def complete_experiment_run(
    session: Session,
    run_id: int,
    final_stats: dict,
) -> ExperimentRun:
    """
    Mark experiment run as completed with final statistics.

    Args:
        session: Database session
        run_id: Experiment run ID
        final_stats: Dict of final training statistics

    Returns:
        Updated ExperimentRun instance
    """
    run = session.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
    if run:
        run.completed_at = datetime.utcnow()
        run.final_stats = final_stats
        session.commit()
        session.refresh(run)
    return run


def log_training_metric(
    session: Session,
    run_id: int,
    step: int,
    avg_reward: Optional[float] = None,
    episode_length: Optional[float] = None,
) -> TrainingMetric:
    """
    Log a training metric at a specific step.

    Args:
        session: Database session
        run_id: Experiment run ID
        step: Training step number
        avg_reward: Average reward (if available)
        episode_length: Average episode length (if available)

    Returns:
        Created TrainingMetric instance
    """
    metric = TrainingMetric(
        run_id=run_id,
        step=step,
        avg_reward=avg_reward,
        episode_length=episode_length,
        timestamp=datetime.utcnow(),
    )
    session.add(metric)
    session.commit()
    return metric


def log_evaluation_result(
    session: Session,
    run_id: int,
    eval_scenario: str,
    mean_reward: float,
    std_reward: float,
    num_episodes: int,
) -> EvaluationResult:
    """
    Log evaluation results for a trained model.

    Args:
        session: Database session
        run_id: Experiment run ID
        eval_scenario: Scenario used for evaluation
        mean_reward: Mean reward across episodes
        std_reward: Standard deviation of rewards
        num_episodes: Number of evaluation episodes

    Returns:
        Created EvaluationResult instance
    """
    result = EvaluationResult(
        run_id=run_id,
        eval_scenario=eval_scenario,
        mean_reward=mean_reward,
        std_reward=std_reward,
        num_episodes=num_episodes,
        evaluated_at=datetime.utcnow(),
    )
    session.add(result)
    session.commit()
    return result


def find_run_by_model_path(session: Session, model_path: str) -> Optional[ExperimentRun]:
    """
    Find an experiment run by its model path.

    Args:
        session: Database session
        model_path: Path to model file

    Returns:
        ExperimentRun instance if found, None otherwise
    """
    return session.query(ExperimentRun).filter(ExperimentRun.model_path == model_path).first()

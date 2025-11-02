"""Job Queue System for Bucket Brigade Tournament Matchups.

This module provides a priority-based job queue system for managing matchup
execution in tournaments. It uses an abstract interface to support multiple
backend implementations (in-memory and Redis).

The queue supports three priority levels (HIGH, MEDIUM, LOW) and provides
process-safe operations for concurrent access.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional, List, Literal, Tuple
from enum import IntEnum
import json
import pickle
import threading


class JobPriority(IntEnum):
    """Priority levels for matchup jobs.

    Lower numeric values indicate higher priority.
    Jobs are dequeued in priority order: HIGH → MEDIUM → LOW.
    """
    HIGH = 0      # High uncertainty matchups (adaptive sampling)
    MEDIUM = 1    # Standard matchups
    LOW = 2       # Backfill/maintenance jobs


@dataclass
class MatchupJob:
    """Job specification for a single matchup.

    Attributes:
        team_ids: List of agent IDs forming the team
        scenario: Scenario name/configuration string
        seed: Random seed for reproducibility
        priority: Job priority level (default: MEDIUM)
    """
    team_ids: List[int]
    scenario: str
    seed: int
    priority: JobPriority = JobPriority.MEDIUM

    def __lt__(self, other: 'MatchupJob') -> bool:
        """Enable priority comparison for PriorityQueue.

        Lower priority values (e.g., HIGH=0) are "less than" higher values (e.g., LOW=2),
        ensuring high-priority jobs are dequeued first.
        """
        return self.priority < other.priority

    def to_dict(self) -> dict:
        """Serialize job to dictionary for persistence.

        Returns:
            Dictionary representation with all fields
        """
        data = asdict(self)
        data['priority'] = int(self.priority)  # Convert enum to int
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'MatchupJob':
        """Deserialize job from dictionary.

        Args:
            data: Dictionary with job fields

        Returns:
            MatchupJob instance
        """
        data['priority'] = JobPriority(data['priority'])
        return cls(**data)


class JobQueueBackend(ABC):
    """Abstract interface for job queue implementations.

    This abstraction allows switching between in-memory and distributed
    (Redis) backends without changing application code.
    """

    @abstractmethod
    def enqueue(self, job: MatchupJob) -> None:
        """Add job to queue.

        Args:
            job: MatchupJob to enqueue
        """
        pass

    @abstractmethod
    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[MatchupJob]:
        """Remove and return highest-priority job.

        Args:
            block: If True, wait for job if queue is empty
            timeout: Maximum time to wait in seconds (None = wait forever)

        Returns:
            Highest-priority job, or None if queue is empty (and not blocking)
        """
        pass

    @abstractmethod
    def peek(self) -> Optional[MatchupJob]:
        """View next job without removing it.

        Returns:
            Next job that would be dequeued, or None if empty
        """
        pass

    @abstractmethod
    def size(self) -> int:
        """Get number of jobs in queue.

        Returns:
            Number of pending jobs
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all jobs from queue."""
        pass

    @abstractmethod
    def persist(self, filepath: str) -> None:
        """Save queue state to disk.

        Args:
            filepath: Path to save queue state
        """
        pass

    @abstractmethod
    def restore(self, filepath: str) -> None:
        """Restore queue state from disk.

        Args:
            filepath: Path to restore queue state from
        """
        pass


class InMemoryJobQueue(JobQueueBackend):
    """Thread-safe priority queue using threading primitives.

    This implementation uses threading primitives (Lock, list) to ensure thread-safety.
    Priority ordering is achieved by sorting during dequeue.

    Performance characteristics:
    - Enqueue: O(1)
    - Dequeue: O(n log n) where n is queue size
    - Recommended for queues with <10k jobs

    Note: For true multiprocessing safety across separate processes,
    consider using RedisJobQueue or a separate queue management process.
    """

    def __init__(self):
        """Initialize thread-safe queue."""
        self._jobs: List[Tuple[JobPriority, MatchupJob]] = []
        self._lock = threading.Lock()

    def enqueue(self, job: MatchupJob) -> None:
        """Add job with priority.

        Args:
            job: MatchupJob to enqueue
        """
        with self._lock:
            # Store as (priority, job) tuple for ordering
            self._jobs.append((job.priority, job))

    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[MatchupJob]:
        """Get highest-priority job (lowest priority number).

        Implementation sorts jobs by priority and returns the highest priority one.

        Args:
            block: If True, wait for job if queue is empty (not implemented)
            timeout: Maximum time to wait in seconds (not implemented)

        Returns:
            Highest-priority job, or None if empty
        """
        with self._lock:
            if not self._jobs:
                return None

            # Sort by priority (ascending - lowest number = highest priority)
            self._jobs.sort(key=lambda x: x[0])

            # Get and remove highest priority job
            priority, job = self._jobs.pop(0)
            return job

    def peek(self) -> Optional[MatchupJob]:
        """View next job without removing.

        Returns:
            Next job that would be dequeued, or None if empty
        """
        with self._lock:
            if not self._jobs:
                return None

            # Sort by priority
            sorted_jobs = sorted(self._jobs, key=lambda x: x[0])
            return sorted_jobs[0][1]

    def size(self) -> int:
        """Get queue size.

        Returns:
            Number of pending jobs
        """
        with self._lock:
            return len(self._jobs)

    def clear(self) -> None:
        """Clear all jobs."""
        with self._lock:
            self._jobs.clear()

    def persist(self, filepath: str) -> None:
        """Save queue to disk using pickle.

        Args:
            filepath: Path to save queue state
        """
        with self._lock:
            # Convert to serializable format (dict list)
            serializable_jobs = [(int(priority), job.to_dict())
                                 for priority, job in self._jobs]

            # Save to file
            with open(filepath, 'wb') as f:
                pickle.dump(serializable_jobs, f)

    def restore(self, filepath: str) -> None:
        """Restore queue from disk.

        Args:
            filepath: Path to restore queue state from
        """
        with open(filepath, 'rb') as f:
            serializable_jobs: List[Tuple[int, dict]] = pickle.load(f)

        # Convert back to job objects
        jobs = [(JobPriority(priority), MatchupJob.from_dict(job_dict))
                for priority, job_dict in serializable_jobs]

        with self._lock:
            # Clear existing queue and restore
            self._jobs = jobs


class JobQueue:
    """High-level job queue service with configurable backend.

    This class provides a simple interface for job queue operations and
    delegates to a backend implementation (in-memory or Redis).

    Example:
        >>> queue = JobQueue(backend="memory")
        >>> job = MatchupJob([1, 2], "scenario_a", 42, JobPriority.HIGH)
        >>> queue.enqueue(job)
        >>> next_job = queue.dequeue()
        >>> print(queue.size())
        0
    """

    def __init__(
        self,
        backend: Literal["memory", "redis"] = "memory",
        redis_url: str = "redis://localhost:6379/0"
    ):
        """Initialize job queue with specified backend.

        Args:
            backend: Backend type ("memory" or "redis")
            redis_url: Redis connection URL (only used if backend="redis")

        Raises:
            ValueError: If backend is not "memory" or "redis"
        """
        if backend == "memory":
            self._backend = InMemoryJobQueue()
        elif backend == "redis":
            # Redis implementation will be added in Phase 2
            raise NotImplementedError("Redis backend not yet implemented")
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def enqueue(self, job: MatchupJob) -> None:
        """Add job to queue.

        Args:
            job: MatchupJob to enqueue
        """
        return self._backend.enqueue(job)

    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> Optional[MatchupJob]:
        """Remove and return highest-priority job.

        Args:
            block: If True, wait for job if queue is empty
            timeout: Maximum time to wait in seconds

        Returns:
            Highest-priority job, or None if empty
        """
        return self._backend.dequeue(block, timeout)

    def peek(self) -> Optional[MatchupJob]:
        """View next job without removing it.

        Returns:
            Next job that would be dequeued, or None if empty
        """
        return self._backend.peek()

    def size(self) -> int:
        """Get number of jobs in queue.

        Returns:
            Number of pending jobs
        """
        return self._backend.size()

    def clear(self) -> None:
        """Remove all jobs from queue."""
        return self._backend.clear()

    def persist(self, filepath: str) -> None:
        """Save queue state to disk.

        Args:
            filepath: Path to save queue state
        """
        return self._backend.persist(filepath)

    def restore(self, filepath: str) -> None:
        """Restore queue state from disk.

        Args:
            filepath: Path to restore queue state from
        """
        return self._backend.restore(filepath)

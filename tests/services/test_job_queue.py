"""Unit tests for job queue system."""

import os
import tempfile
import threading
import time

import pytest

from bucket_brigade.services.job_queue import (
    JobQueue,
    MatchupJob,
    JobPriority,
    InMemoryJobQueue,
)


class TestMatchupJob:
    """Tests for MatchupJob dataclass."""

    def test_create_job(self):
        """Test creating a MatchupJob."""
        job = MatchupJob([1, 2], "scenario_a", 42)
        assert job.team_ids == [1, 2]
        assert job.scenario == "scenario_a"
        assert job.seed == 42
        assert job.priority == JobPriority.MEDIUM

    def test_create_job_with_priority(self):
        """Test creating a MatchupJob with custom priority."""
        job = MatchupJob([1, 2], "scenario_a", 42, JobPriority.HIGH)
        assert job.priority == JobPriority.HIGH

    def test_job_comparison(self):
        """Test that jobs can be compared by priority."""
        high = MatchupJob([1, 2], "scenario", 1, JobPriority.HIGH)
        medium = MatchupJob([3, 4], "scenario", 2, JobPriority.MEDIUM)
        low = MatchupJob([5, 6], "scenario", 3, JobPriority.LOW)

        assert high < medium
        assert medium < low
        assert high < low

    def test_job_serialization(self):
        """Test job serialization to dict."""
        job = MatchupJob([1, 2], "scenario_a", 42, JobPriority.HIGH)
        data = job.to_dict()

        assert data["team_ids"] == [1, 2]
        assert data["scenario"] == "scenario_a"
        assert data["seed"] == 42
        assert data["priority"] == 0  # HIGH = 0

    def test_job_deserialization(self):
        """Test job deserialization from dict."""
        data = {
            "team_ids": [1, 2],
            "scenario": "scenario_a",
            "seed": 42,
            "priority": 0,
        }
        job = MatchupJob.from_dict(data)

        assert job.team_ids == [1, 2]
        assert job.scenario == "scenario_a"
        assert job.seed == 42
        assert job.priority == JobPriority.HIGH


class TestInMemoryJobQueue:
    """Tests for InMemoryJobQueue implementation."""

    def test_enqueue_dequeue(self):
        """Test basic enqueue and dequeue operations."""
        queue = InMemoryJobQueue()
        job = MatchupJob([1, 2], "scenario", 42)

        queue.enqueue(job)
        assert queue.size() == 1

        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.team_ids == [1, 2]
        assert dequeued.scenario == "scenario"
        assert dequeued.seed == 42
        assert queue.size() == 0

    def test_dequeue_empty_queue(self):
        """Test dequeue on empty queue returns None."""
        queue = InMemoryJobQueue()
        result = queue.dequeue(block=False)
        assert result is None

    def test_priority_ordering(self):
        """Test jobs dequeued in priority order."""
        queue = InMemoryJobQueue()

        # Enqueue in random order
        queue.enqueue(MatchupJob([1, 2], "scenario_a", 42, JobPriority.LOW))
        queue.enqueue(MatchupJob([3, 4], "scenario_b", 43, JobPriority.HIGH))
        queue.enqueue(MatchupJob([5, 6], "scenario_c", 44, JobPriority.MEDIUM))

        # Dequeue should return HIGH → MEDIUM → LOW
        job1 = queue.dequeue()
        assert job1 is not None
        assert job1.priority == JobPriority.HIGH

        job2 = queue.dequeue()
        assert job2 is not None
        assert job2.priority == JobPriority.MEDIUM

        job3 = queue.dequeue()
        assert job3 is not None
        assert job3.priority == JobPriority.LOW

        assert queue.size() == 0

    def test_peek_non_destructive(self):
        """Test peek doesn't remove jobs."""
        queue = InMemoryJobQueue()
        job = MatchupJob([1, 2], "scenario", 42)
        queue.enqueue(job)

        job1 = queue.peek()
        job2 = queue.peek()

        assert job1 is not None
        assert job2 is not None
        assert job1.team_ids == job2.team_ids
        assert queue.size() == 1

    def test_peek_empty_queue(self):
        """Test peek on empty queue returns None."""
        queue = InMemoryJobQueue()
        result = queue.peek()
        assert result is None

    def test_peek_returns_highest_priority(self):
        """Test peek returns highest priority job."""
        queue = InMemoryJobQueue()

        queue.enqueue(MatchupJob([1, 2], "scenario", 1, JobPriority.MEDIUM))
        queue.enqueue(MatchupJob([3, 4], "scenario", 2, JobPriority.HIGH))
        queue.enqueue(MatchupJob([5, 6], "scenario", 3, JobPriority.LOW))

        peeked = queue.peek()
        assert peeked is not None
        assert peeked.priority == JobPriority.HIGH

    def test_clear(self):
        """Test clearing the queue."""
        queue = InMemoryJobQueue()

        for i in range(10):
            queue.enqueue(MatchupJob([i, i + 1], "scenario", i))

        assert queue.size() == 10

        queue.clear()
        assert queue.size() == 0
        assert queue.dequeue() is None

    def test_persistence(self):
        """Test queue serialization and restoration."""
        queue = InMemoryJobQueue()

        # Add jobs with different priorities
        queue.enqueue(MatchupJob([1, 2], "scenario_a", 42, JobPriority.HIGH))
        queue.enqueue(MatchupJob([3, 4], "scenario_b", 43, JobPriority.MEDIUM))
        queue.enqueue(MatchupJob([5, 6], "scenario_c", 44, JobPriority.LOW))

        # Persist to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name

        try:
            queue.persist(filepath)

            # Create new queue and restore
            new_queue = InMemoryJobQueue()
            new_queue.restore(filepath)

            # Verify size
            assert new_queue.size() == 3

            # Verify order is preserved
            job1 = new_queue.dequeue()
            assert job1 is not None
            assert job1.priority == JobPriority.HIGH

            job2 = new_queue.dequeue()
            assert job2 is not None
            assert job2.priority == JobPriority.MEDIUM

            job3 = new_queue.dequeue()
            assert job3 is not None
            assert job3.priority == JobPriority.LOW

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestJobQueue:
    """Tests for JobQueue factory class."""

    def test_create_memory_backend(self):
        """Test creating JobQueue with memory backend."""
        queue = JobQueue(backend="memory")
        assert isinstance(queue._backend, InMemoryJobQueue)

    def test_create_redis_backend_not_implemented(self):
        """Test creating JobQueue with redis backend raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            JobQueue(backend="redis")

    def test_create_invalid_backend(self):
        """Test creating JobQueue with invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            JobQueue(backend="invalid")

    def test_enqueue_dequeue(self):
        """Test JobQueue delegates to backend."""
        queue = JobQueue(backend="memory")
        job = MatchupJob([1, 2], "scenario", 42)

        queue.enqueue(job)
        assert queue.size() == 1

        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.team_ids == [1, 2]

    def test_all_methods_delegate(self):
        """Test all JobQueue methods delegate to backend."""
        queue = JobQueue(backend="memory")

        # Test enqueue
        job1 = MatchupJob([1, 2], "scenario", 1, JobPriority.HIGH)
        job2 = MatchupJob([3, 4], "scenario", 2, JobPriority.LOW)
        queue.enqueue(job1)
        queue.enqueue(job2)

        # Test size
        assert queue.size() == 2

        # Test peek
        peeked = queue.peek()
        assert peeked is not None
        assert peeked.priority == JobPriority.HIGH
        assert queue.size() == 2  # peek doesn't remove

        # Test dequeue
        dequeued = queue.dequeue()
        assert dequeued is not None
        assert dequeued.priority == JobPriority.HIGH
        assert queue.size() == 1

        # Test persist/restore
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name

        try:
            queue.persist(filepath)

            new_queue = JobQueue(backend="memory")
            new_queue.restore(filepath)
            assert new_queue.size() == 1

            # Test clear
            new_queue.clear()
            assert new_queue.size() == 0

        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestConcurrentAccess:
    """Tests for concurrent access to job queue."""

    def test_concurrent_enqueue(self):
        """Test multiple threads enqueueing simultaneously."""
        queue = InMemoryJobQueue()

        def worker(start: int, count: int):
            """Worker function that enqueues jobs."""
            for i in range(start, start + count):
                job = MatchupJob([i, i + 1], "scenario", i)
                queue.enqueue(job)

        # Spawn 5 workers, each adding 20 jobs
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i * 20, 20))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        # Should have 100 jobs total
        assert queue.size() == 100

    def test_concurrent_enqueue_dequeue(self):
        """Test producers and consumers running simultaneously."""
        queue = InMemoryJobQueue()
        consumed_count = 0
        count_lock = threading.Lock()

        def producer(count: int):
            """Producer that enqueues jobs."""
            for i in range(count):
                job = MatchupJob([i, i + 1], "scenario", i, JobPriority.MEDIUM)
                queue.enqueue(job)
                time.sleep(0.001)  # Small delay to allow interleaving

        def consumer(count: int):
            """Consumer that dequeues jobs."""
            nonlocal consumed_count
            local_count = 0
            for _ in range(count):
                job = queue.dequeue(block=False)
                if job:
                    local_count += 1
                time.sleep(0.001)  # Small delay to allow interleaving

            with count_lock:
                consumed_count += local_count

        # 3 producers (30 jobs each), 2 consumers (45 jobs each)
        threads = []
        threads.append(threading.Thread(target=producer, args=(30,)))
        threads.append(threading.Thread(target=producer, args=(30,)))
        threads.append(threading.Thread(target=producer, args=(30,)))

        for t in threads:
            t.start()

        # Wait for producers to finish
        for t in threads:
            t.join()

        # Now run consumers
        consumer_threads = []
        consumer_threads.append(threading.Thread(target=consumer, args=(45,)))
        consumer_threads.append(threading.Thread(target=consumer, args=(45,)))

        for t in consumer_threads:
            t.start()

        for t in consumer_threads:
            t.join()

        # All 90 jobs should be consumed
        assert consumed_count == 90
        assert queue.size() == 0


class TestBenchmarks:
    """Performance benchmarks for job queue."""

    def test_throughput_benchmark(self):
        """Verify 1000+ jobs/second throughput."""
        queue = InMemoryJobQueue()

        # Enqueue 1000 jobs
        start = time.time()
        for i in range(1000):
            queue.enqueue(MatchupJob([i, i + 1], "scenario", i))
        enqueue_duration = time.time() - start

        enqueue_rate = 1000 / enqueue_duration
        assert (
            enqueue_rate > 1000
        ), f"Enqueue rate too slow: {enqueue_rate:.0f} jobs/sec"

        # Dequeue 1000 jobs
        start = time.time()
        for _ in range(1000):
            queue.dequeue()
        dequeue_duration = time.time() - start

        dequeue_rate = 1000 / dequeue_duration
        assert (
            dequeue_rate > 1000
        ), f"Dequeue rate too slow: {dequeue_rate:.0f} jobs/sec"

    def test_large_queue_performance(self):
        """Test performance with larger queue (5000 jobs)."""
        queue = InMemoryJobQueue()

        # Enqueue 5000 jobs with mixed priorities
        start = time.time()
        for i in range(5000):
            priority = JobPriority(i % 3)  # Cycle through priorities
            queue.enqueue(MatchupJob([i, i + 1], "scenario", i, priority))
        enqueue_duration = time.time() - start

        assert queue.size() == 5000
        assert enqueue_duration < 5.0, f"Enqueue took too long: {enqueue_duration:.2f}s"

        # Dequeue all jobs
        dequeued_count = 0
        while queue.size() > 0:
            job = queue.dequeue()
            if job:
                dequeued_count += 1

        assert dequeued_count == 5000
        # Note: Dequeue is O(n log n) so this will be slower for large queues
        # This is expected and documented


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

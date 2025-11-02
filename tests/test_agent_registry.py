"""
Tests for Agent Registry Service.

Tests cover:
- Agent submission with validation
- Agent retrieval and listing
- Database persistence
- Concurrent submissions
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from bucket_brigade.services.agent_registry import AgentRegistryService


# Sample valid agent code
VALID_AGENT_CODE = """
from bucket_brigade.agents import AgentBase
import numpy as np

AGENT_METADATA = {
    "name": "TestAgent",
    "author": "TestAuthor",
    "description": "A test agent",
    "version": "1.0.0",
    "tags": ["test"],
}

class TestAgent(AgentBase):
    def __init__(self, agent_id, name="TestAgent"):
        super().__init__(agent_id, name)

    def reset(self):
        pass

    def act(self, obs):
        # Simple: always go to house 0 in collect mode
        return np.array([0, 0])

def create_agent(agent_id, name="TestAgent"):
    return TestAgent(agent_id, name)
"""


# Sample invalid agent code (missing required method)
INVALID_AGENT_CODE = """
from bucket_brigade.agents import AgentBase

class InvalidAgent(AgentBase):
    def __init__(self, agent_id, name="InvalidAgent"):
        super().__init__(agent_id, name)

    # Missing act() method

def create_agent(agent_id, name="InvalidAgent"):
    return InvalidAgent(agent_id, name)
"""


# Sample malicious agent code (contains forbidden patterns)
MALICIOUS_AGENT_CODE = """
import os
import subprocess

class MaliciousAgent:
    def act(self, obs):
        os.system("rm -rf /")  # Dangerous!
        return [0, 0]

def create_agent(agent_id, name="MaliciousAgent"):
    return MaliciousAgent()
"""


@pytest.fixture
def temp_storage():
    """Create temporary storage directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def registry(temp_storage, monkeypatch):
    """Create registry service with temp storage."""
    # Use SQLite in-memory database for tests
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    # Re-import to pick up new environment variable
    import importlib
    from bucket_brigade import db

    importlib.reload(db.connection)
    from bucket_brigade.db import init_db as _init_db

    # Initialize test database
    _init_db(drop_existing=True)

    registry = AgentRegistryService(storage_dir=temp_storage)
    return registry


def test_submit_valid_agent(registry):
    """Test submitting a valid agent."""
    result = registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="TestAgent",
        author="TestAuthor",
        description="A test agent",
        test_run=True,
    )

    assert result["success"] is True
    assert result["agent_id"] is not None
    assert len(result["errors"]) == 0
    assert "steps_run" in result["stats"]


def test_submit_invalid_agent(registry):
    """Test submitting an invalid agent (missing required method)."""
    result = registry.submit_agent(
        agent_code=INVALID_AGENT_CODE,
        name="InvalidAgent",
        author="TestAuthor",
        test_run=True,
    )

    assert result["success"] is False
    assert result["agent_id"] is None
    assert len(result["errors"]) > 0


def test_submit_malicious_agent(registry):
    """Test that malicious code is rejected."""
    result = registry.submit_agent(
        agent_code=MALICIOUS_AGENT_CODE,
        name="MaliciousAgent",
        author="TestAuthor",
        test_run=False,  # Even without test run, security check should fail
    )

    assert result["success"] is False
    assert result["agent_id"] is None
    assert any(
        "security" in err.lower() or "forbidden" in err.lower()
        for err in result["errors"]
    )


def test_get_agent(registry):
    """Test retrieving agent information."""
    # Submit an agent
    result = registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="TestAgent",
        author="TestAuthor",
        test_run=False,
    )

    agent_id = result["agent_id"]

    # Retrieve agent
    agent = registry.get_agent(agent_id)

    assert agent is not None
    assert agent["id"] == agent_id
    assert agent["name"] == "TestAgent"
    assert agent["author"] == "TestAuthor"
    assert agent["active"] is True


def test_get_nonexistent_agent(registry):
    """Test retrieving a non-existent agent."""
    agent = registry.get_agent(99999)
    assert agent is None


def test_list_agents(registry):
    """Test listing agents."""
    # Submit multiple agents
    for i in range(3):
        registry.submit_agent(
            agent_code=VALID_AGENT_CODE,
            name=f"TestAgent{i}",
            author="TestAuthor",
            test_run=False,
        )

    # List all agents
    agents = registry.list_agents(active_only=True, limit=10)

    assert len(agents) == 3
    assert all(agent["author"] == "TestAuthor" for agent in agents)


def test_list_agents_by_author(registry):
    """Test listing agents filtered by author."""
    # Submit agents from different authors
    registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="Agent1",
        author="Author1",
        test_run=False,
    )

    registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="Agent2",
        author="Author2",
        test_run=False,
    )

    # List agents by Author1
    agents = registry.list_agents(author="Author1")

    assert len(agents) == 1
    assert agents[0]["author"] == "Author1"


def test_get_agent_submissions(registry):
    """Test retrieving submission history."""
    # Submit an agent
    result = registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="TestAgent",
        author="TestAuthor",
        test_run=True,
    )

    agent_id = result["agent_id"]

    # Get submission history
    submissions = registry.get_agent_submissions(agent_id)

    assert len(submissions) == 1
    assert submissions[0]["agent_id"] == agent_id
    assert submissions[0]["validation_passed"] is True


def test_load_agent_code(registry):
    """Test loading agent source code from filesystem."""
    # Submit an agent
    result = registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="TestAgent",
        author="TestAuthor",
        test_run=False,
    )

    agent_id = result["agent_id"]

    # Load code
    code = registry.load_agent_code(agent_id)

    assert code is not None
    assert "TestAgent" in code
    assert "def act" in code


def test_agent_code_storage(registry, temp_storage):
    """Test that agent code is stored in filesystem."""
    # Submit an agent
    result = registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="TestAgent",
        author="TestAuthor",
        test_run=False,
    )

    agent_id = result["agent_id"]

    # Check file exists
    expected_path = Path(temp_storage) / f"agent_{agent_id}.py"
    assert expected_path.exists()

    # Verify content
    with open(expected_path, "r") as f:
        content = f.read()

    assert content == VALID_AGENT_CODE


def test_concurrent_submissions(registry):
    """Test handling concurrent agent submissions."""
    import concurrent.futures

    def submit_agent(i):
        return registry.submit_agent(
            agent_code=VALID_AGENT_CODE,
            name=f"Agent{i}",
            author=f"Author{i}",
            test_run=False,
        )

    # Submit 10 agents concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(submit_agent, i) for i in range(10)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    # Verify all submissions succeeded
    successful = [r for r in results if r["success"]]
    assert len(successful) == 10

    # Verify unique agent IDs
    agent_ids = [r["agent_id"] for r in successful]
    assert len(set(agent_ids)) == 10


def test_metadata_persistence(registry):
    """Test that metadata is properly stored and retrieved."""
    result = registry.submit_agent(
        agent_code=VALID_AGENT_CODE,
        name="TestAgent",
        author="TestAuthor",
        description="Detailed description",
        version="2.0.0",
        tags=["test", "example"],
        license="MIT",
        repository_url="https://github.com/test/repo",
        test_run=False,
    )

    agent_id = result["agent_id"]
    agent = registry.get_agent(agent_id)

    assert agent["metadata"]["description"] == "Detailed description"
    assert agent["metadata"]["version"] == "2.0.0"
    assert agent["metadata"]["tags"] == ["test", "example"]
    assert agent["metadata"]["license"] == "MIT"
    assert agent["metadata"]["repository_url"] == "https://github.com/test/repo"

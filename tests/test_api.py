"""
Tests for Agent Registry REST API.

Tests cover:
- POST /agents/submit
- GET /agents/list
- GET /agents/{id}
- GET /agents/{id}/submissions
- GET /agents/{id}/code
"""

import pytest
from fastapi.testclient import TestClient

from bucket_brigade.services.api import app


# Sample valid agent code
VALID_AGENT_CODE = """
from bucket_brigade.agents import AgentBase
import numpy as np

class TestAgent(AgentBase):
    def __init__(self, agent_id, name="TestAgent"):
        super().__init__(agent_id, name)

    def reset(self):
        pass

    def act(self, obs):
        return np.array([0, 0])

def create_agent(agent_id, name="TestAgent"):
    return TestAgent(agent_id, name)
"""


@pytest.fixture
def client(monkeypatch):
    """Create test client with in-memory database."""
    # Use SQLite in-memory database for tests
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    # Re-import to pick up new environment variable
    import importlib
    from bucket_brigade import db

    importlib.reload(db.connection)
    from bucket_brigade.db import init_db as _init_db

    # Initialize test database
    _init_db(drop_existing=True)

    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    """Test root endpoint returns API information."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["service"] == "Bucket Brigade Agent Registry"
    assert "endpoints" in data


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"


def test_submit_agent(client):
    """Test submitting an agent via API."""
    payload = {
        "agent_code": VALID_AGENT_CODE,
        "name": "TestAgent",
        "author": "TestAuthor",
        "description": "A test agent",
        "version": "1.0.0",
        "tags": ["test"],
        "test_run": False,  # Skip behavioral tests for speed
    }

    response = client.post("/agents/submit", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert data["agent_id"] is not None
    assert len(data["errors"]) == 0


def test_submit_invalid_agent(client):
    """Test submitting an invalid agent."""
    payload = {
        "agent_code": "invalid python code!",
        "name": "InvalidAgent",
        "author": "TestAuthor",
        "test_run": False,
    }

    response = client.post("/agents/submit", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is False
    assert data["agent_id"] is None
    assert len(data["errors"]) > 0


def test_list_agents(client):
    """Test listing agents."""
    # Submit a few agents first
    for i in range(3):
        payload = {
            "agent_code": VALID_AGENT_CODE,
            "name": f"Agent{i}",
            "author": "TestAuthor",
            "test_run": False,
        }
        client.post("/agents/submit", json=payload)

    # List agents
    response = client.get("/agents/list")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 3
    assert all(agent["author"] == "TestAuthor" for agent in data)


def test_list_agents_with_filters(client):
    """Test listing agents with filters."""
    # Submit agents from different authors
    client.post(
        "/agents/submit",
        json={
            "agent_code": VALID_AGENT_CODE,
            "name": "Agent1",
            "author": "Author1",
            "test_run": False,
        },
    )

    client.post(
        "/agents/submit",
        json={
            "agent_code": VALID_AGENT_CODE,
            "name": "Agent2",
            "author": "Author2",
            "test_run": False,
        },
    )

    # Filter by author
    response = client.get("/agents/list?author=Author1")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["author"] == "Author1"


def test_get_agent(client):
    """Test retrieving agent details."""
    # Submit an agent
    submit_response = client.post(
        "/agents/submit",
        json={
            "agent_code": VALID_AGENT_CODE,
            "name": "TestAgent",
            "author": "TestAuthor",
            "test_run": False,
        },
    )

    agent_id = submit_response.json()["agent_id"]

    # Get agent details
    response = client.get(f"/agents/{agent_id}")
    assert response.status_code == 200

    data = response.json()
    assert data["id"] == agent_id
    assert data["name"] == "TestAgent"
    assert data["author"] == "TestAuthor"


def test_get_nonexistent_agent(client):
    """Test retrieving a non-existent agent returns 404."""
    response = client.get("/agents/99999")
    assert response.status_code == 404


def test_get_agent_submissions(client):
    """Test retrieving agent submission history."""
    # Submit an agent
    submit_response = client.post(
        "/agents/submit",
        json={
            "agent_code": VALID_AGENT_CODE,
            "name": "TestAgent",
            "author": "TestAuthor",
            "test_run": True,
        },
    )

    agent_id = submit_response.json()["agent_id"]

    # Get submission history
    response = client.get(f"/agents/{agent_id}/submissions")
    assert response.status_code == 200

    data = response.json()
    assert len(data) == 1
    assert data[0]["agent_id"] == agent_id
    assert data[0]["validation_passed"] is True


def test_get_agent_code(client):
    """Test retrieving agent source code."""
    # Submit an agent
    submit_response = client.post(
        "/agents/submit",
        json={
            "agent_code": VALID_AGENT_CODE,
            "name": "TestAgent",
            "author": "TestAuthor",
            "test_run": False,
        },
    )

    agent_id = submit_response.json()["agent_id"]

    # Get source code
    response = client.get(f"/agents/{agent_id}/code")
    assert response.status_code == 200

    data = response.json()
    assert data["agent_id"] == agent_id
    assert "def act" in data["code"]


def test_pagination(client):
    """Test pagination parameters."""
    # Submit 10 agents
    for i in range(10):
        client.post(
            "/agents/submit",
            json={
                "agent_code": VALID_AGENT_CODE,
                "name": f"Agent{i}",
                "author": "TestAuthor",
                "test_run": False,
            },
        )

    # Get first 5
    response = client.get("/agents/list?limit=5&offset=0")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5

    # Get next 5
    response = client.get("/agents/list?limit=5&offset=5")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 5

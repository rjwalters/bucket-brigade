"""
FastAPI REST API for Bucket Brigade Agent Registry.

Provides HTTP endpoints for:
- POST /agents/submit - Submit new agent
- GET /agents/list - List all agents
- GET /agents/{id} - Get agent details
- GET /agents/{id}/submissions - Get submission history
- GET /agents/{id}/code - Get agent source code
"""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field

from .agent_registry import AgentRegistryService


# Pydantic models for request/response
class AgentSubmitRequest(BaseModel):
    """Request model for agent submission."""

    agent_code: str = Field(..., description="Python source code as string")
    name: str = Field(..., description="Agent display name")
    author: str = Field(..., description="Agent creator")
    description: Optional[str] = Field(None, description="Agent description")
    version: str = Field("1.0.0", description="Semantic version")
    tags: Optional[List[str]] = Field(None, description="List of tags/keywords")
    license: Optional[str] = Field(None, description="License identifier")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    test_run: bool = Field(True, description="Run behavioral validation tests")


class AgentSubmitResponse(BaseModel):
    """Response model for agent submission."""

    success: bool
    agent_id: Optional[int]
    errors: List[str]
    warnings: List[str]
    stats: dict


class AgentInfo(BaseModel):
    """Model for agent information."""

    id: int
    name: str
    author: str
    created_at: str
    updated_at: str
    active: bool
    description: Optional[str]
    version: str
    tags: List[str]


class AgentDetail(BaseModel):
    """Detailed agent information including metadata."""

    id: int
    name: str
    author: str
    code_path: str
    created_at: str
    updated_at: str
    active: bool
    metadata: dict


class SubmissionRecord(BaseModel):
    """Model for submission history record."""

    id: int
    agent_id: int
    validation_passed: bool
    validation_errors: Optional[List[str]]
    validation_warnings: Optional[List[str]]
    test_stats: Optional[dict]
    submitted_at: str


# Initialize FastAPI app
app = FastAPI(
    title="Bucket Brigade Agent Registry API",
    description="REST API for submitting and managing Bucket Brigade agents",
    version="1.0.0",
)

# Initialize registry service
registry = AgentRegistryService()


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "service": "Bucket Brigade Agent Registry",
        "version": "1.0.0",
        "endpoints": {
            "submit": "POST /agents/submit",
            "list": "GET /agents/list",
            "get": "GET /agents/{id}",
            "submissions": "GET /agents/{id}/submissions",
            "code": "GET /agents/{id}/code",
        },
    }


@app.post("/agents/submit", response_model=AgentSubmitResponse)
def submit_agent(request: AgentSubmitRequest):
    """
    Submit a new agent for validation and storage.

    Args:
        request: Agent submission request with code and metadata

    Returns:
        Submission result with success status, agent ID, and validation results
    """
    result = registry.submit_agent(
        agent_code=request.agent_code,
        name=request.name,
        author=request.author,
        description=request.description,
        version=request.version,
        tags=request.tags,
        license=request.license,
        repository_url=request.repository_url,
        test_run=request.test_run,
    )

    return AgentSubmitResponse(**result)


@app.post("/agents/submit_file")
async def submit_agent_file(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    author: Optional[str] = Form(None),
    test_run: bool = Form(True),
):
    """
    Submit an agent from an uploaded file.

    Args:
        file: Uploaded Python file
        name: Optional agent name
        author: Optional author name
        test_run: Whether to run behavioral tests

    Returns:
        Submission result
    """
    # Read file content
    content = await file.read()
    agent_code = content.decode("utf-8")

    result = registry.submit_agent(
        agent_code=agent_code,
        name=name or file.filename.replace(".py", ""),
        author=author or "Unknown",
        test_run=test_run,
    )

    return AgentSubmitResponse(**result)


@app.get("/agents/list", response_model=List[AgentInfo])
def list_agents(
    active_only: bool = True,
    author: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    List agents with optional filtering.

    Args:
        active_only: Only return active agents (default: True)
        author: Filter by author name
        limit: Maximum number of results (default: 100)
        offset: Number of results to skip (default: 0)

    Returns:
        List of agent information
    """
    agents = registry.list_agents(
        active_only=active_only, author=author, limit=limit, offset=offset
    )

    return [AgentInfo(**agent) for agent in agents]


@app.get("/agents/{agent_id}", response_model=AgentDetail)
def get_agent(agent_id: int):
    """
    Get detailed information about a specific agent.

    Args:
        agent_id: Agent identifier

    Returns:
        Detailed agent information

    Raises:
        404: Agent not found
    """
    agent = registry.get_agent(agent_id)

    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    return AgentDetail(**agent)


@app.get("/agents/{agent_id}/submissions", response_model=List[SubmissionRecord])
def get_agent_submissions(agent_id: int):
    """
    Get submission history for an agent.

    Args:
        agent_id: Agent identifier

    Returns:
        List of submission records

    Raises:
        404: Agent not found
    """
    # Verify agent exists
    agent = registry.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    submissions = registry.get_agent_submissions(agent_id)

    return [SubmissionRecord(**sub) for sub in submissions]


@app.get("/agents/{agent_id}/code")
def get_agent_code(agent_id: int):
    """
    Get agent source code.

    Args:
        agent_id: Agent identifier

    Returns:
        Agent source code as plain text

    Raises:
        404: Agent not found or code file missing
    """
    code = registry.load_agent_code(agent_id)

    if code is None:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} code not found")

    return {"agent_id": agent_id, "code": code}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "agent-registry"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

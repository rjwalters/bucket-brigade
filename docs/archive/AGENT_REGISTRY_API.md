# Agent Registry API Documentation

REST API for submitting and managing Bucket Brigade agents.

## Base URL

```
http://localhost:8000
```

## Endpoints

### GET /

Get API information and available endpoints.

**Response:**

```json
{
  "service": "Bucket Brigade Agent Registry",
  "version": "1.0.0",
  "endpoints": {
    "submit": "POST /agents/submit",
    "list": "GET /agents/list",
    "get": "GET /agents/{id}",
    "submissions": "GET /agents/{id}/submissions",
    "code": "GET /agents/{id}/code"
  }
}
```

### POST /agents/submit

Submit a new agent for validation and storage.

**Request Body:**

```json
{
  "agent_code": "string (Python source code)",
  "name": "string (agent display name)",
  "author": "string (agent creator)",
  "description": "string (optional, agent description)",
  "version": "string (optional, default: 1.0.0)",
  "tags": ["array", "of", "strings"] (optional),
  "license": "string (optional, e.g., MIT)",
  "repository_url": "string (optional, repo URL)",
  "test_run": true (optional, default: true)
}
```

**Response:**

```json
{
  "success": true,
  "agent_id": 42,
  "errors": [],
  "warnings": [],
  "stats": {
    "steps_run": 20,
    "total_reward": 15.5,
    "avg_reward": 0.775,
    "game_completed": false
  }
}
```

**Example:**

```bash
curl -X POST http://localhost:8000/agents/submit \
  -H "Content-Type: application/json" \
  -d '{
    "agent_code": "from bucket_brigade.agents import AgentBase\nimport numpy as np\n\nclass MyAgent(AgentBase):\n    def act(self, obs):\n        return np.array([0, 0])\n\ndef create_agent(agent_id, name=\"MyAgent\"):\n    return MyAgent(agent_id, name)",
    "name": "MyAgent",
    "author": "Alice",
    "description": "A simple test agent"
  }'
```

### POST /agents/submit_file

Submit an agent from an uploaded file.

**Request:**

Multipart form data:
- `file`: Python file (required)
- `name`: Agent name (optional)
- `author`: Author name (optional)
- `test_run`: Boolean (optional, default: true)

**Response:**

Same as `/agents/submit`

**Example:**

```bash
curl -X POST http://localhost:8000/agents/submit_file \
  -F "file=@my_agent.py" \
  -F "name=MyAgent" \
  -F "author=Alice"
```

### GET /agents/list

List agents with optional filtering and pagination.

**Query Parameters:**

- `active_only` (boolean, default: true) - Only return active agents
- `author` (string, optional) - Filter by author name
- `limit` (integer, default: 100) - Maximum number of results
- `offset` (integer, default: 0) - Number of results to skip

**Response:**

```json
[
  {
    "id": 42,
    "name": "MyAgent",
    "author": "Alice",
    "created_at": "2025-11-02T14:30:00",
    "updated_at": "2025-11-02T14:30:00",
    "active": true,
    "description": "A simple test agent",
    "version": "1.0.0",
    "tags": ["test", "example"]
  }
]
```

**Example:**

```bash
# List all active agents
curl http://localhost:8000/agents/list

# List agents by author
curl http://localhost:8000/agents/list?author=Alice

# Paginated results
curl http://localhost:8000/agents/list?limit=10&offset=20
```

### GET /agents/{agent_id}

Get detailed information about a specific agent.

**Path Parameters:**

- `agent_id` (integer) - Agent identifier

**Response:**

```json
{
  "id": 42,
  "name": "MyAgent",
  "author": "Alice",
  "code_path": "agents/submitted/agent_42.py",
  "created_at": "2025-11-02T14:30:00",
  "updated_at": "2025-11-02T14:30:00",
  "active": true,
  "metadata": {
    "description": "A simple test agent",
    "version": "1.0.0",
    "tags": ["test", "example"],
    "license": "MIT",
    "repository_url": "https://github.com/alice/my-agent"
  }
}
```

**Example:**

```bash
curl http://localhost:8000/agents/42
```

**Error Response (404):**

```json
{
  "detail": "Agent 42 not found"
}
```

### GET /agents/{agent_id}/submissions

Get submission history for an agent.

**Path Parameters:**

- `agent_id` (integer) - Agent identifier

**Response:**

```json
[
  {
    "id": 1,
    "agent_id": 42,
    "validation_passed": true,
    "validation_errors": null,
    "validation_warnings": ["Warning message"],
    "test_stats": {
      "steps_run": 20,
      "total_reward": 15.5,
      "avg_reward": 0.775,
      "game_completed": false
    },
    "submitted_at": "2025-11-02T14:30:00"
  }
]
```

**Example:**

```bash
curl http://localhost:8000/agents/42/submissions
```

### GET /agents/{agent_id}/code

Get agent source code.

**Path Parameters:**

- `agent_id` (integer) - Agent identifier

**Response:**

```json
{
  "agent_id": 42,
  "code": "from bucket_brigade.agents import AgentBase\nimport numpy as np\n\nclass MyAgent(AgentBase):\n    def act(self, obs):\n        return np.array([0, 0])\n\ndef create_agent(agent_id, name=\"MyAgent\"):\n    return MyAgent(agent_id, name)"
}
```

**Example:**

```bash
curl http://localhost:8000/agents/42/code
```

### GET /health

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "service": "agent-registry"
}
```

**Example:**

```bash
curl http://localhost:8000/health
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK` - Success
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `500 Internal Server Error` - Server error

Error responses include a `detail` field:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Rate Limiting

Currently no rate limiting is enforced. For production deployments, consider:

- Rate limiting by IP address
- Authentication with API keys
- Request throttling for submission endpoints

## Authentication

Currently no authentication is required. For production deployments, consider:

- API key authentication
- OAuth 2.0 integration
- Per-user submission limits

## Running the API Server

### Development

```bash
# Start development server
python -m bucket_brigade.services.api

# Or with uvicorn directly
uvicorn bucket_brigade.services.api:app --reload
```

### Production

```bash
# Start production server with multiple workers
uvicorn bucket_brigade.services.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info

# Or with Gunicorn
gunicorn bucket_brigade.services.api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

### Docker

```bash
# Build image
docker build -t bucket-brigade-api .

# Run container
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL="postgresql://..." \
  bucket-brigade-api
```

## Client Examples

### Python

```python
import requests

# Submit agent
with open("my_agent.py", "r") as f:
    agent_code = f.read()

response = requests.post(
    "http://localhost:8000/agents/submit",
    json={
        "agent_code": agent_code,
        "name": "MyAgent",
        "author": "Alice",
        "test_run": True,
    }
)

if response.json()["success"]:
    agent_id = response.json()["agent_id"]
    print(f"Agent submitted successfully! ID: {agent_id}")
else:
    print("Submission failed:")
    for error in response.json()["errors"]:
        print(f"  - {error}")

# List agents
response = requests.get("http://localhost:8000/agents/list")
agents = response.json()

for agent in agents:
    print(f"{agent['id']}: {agent['name']} by {agent['author']}")
```

### JavaScript

```javascript
// Submit agent
const response = await fetch('http://localhost:8000/agents/submit', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    agent_code: agentCodeString,
    name: 'MyAgent',
    author: 'Alice',
    test_run: true
  })
});

const result = await response.json();

if (result.success) {
  console.log(`Agent submitted! ID: ${result.agent_id}`);
} else {
  console.error('Submission failed:', result.errors);
}

// List agents
const listResponse = await fetch('http://localhost:8000/agents/list');
const agents = await listResponse.json();

agents.forEach(agent => {
  console.log(`${agent.id}: ${agent.name} by ${agent.author}`);
});
```

### cURL

```bash
# Submit agent from file
AGENT_CODE=$(cat my_agent.py)
curl -X POST http://localhost:8000/agents/submit \
  -H "Content-Type: application/json" \
  -d "{\"agent_code\": \"$AGENT_CODE\", \"name\": \"MyAgent\", \"author\": \"Alice\"}"

# List agents
curl http://localhost:8000/agents/list | jq .

# Get agent details
curl http://localhost:8000/agents/42 | jq .

# Get agent code
curl http://localhost:8000/agents/42/code | jq -r .code
```

## Interactive API Documentation

FastAPI provides automatic interactive API documentation:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

These interfaces allow you to:
- Explore all endpoints
- View request/response schemas
- Test API calls interactively
- Download OpenAPI specification

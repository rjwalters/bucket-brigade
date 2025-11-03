# MCP Remote SSH Server - Setup Guide

## Quick Start

The MCP Remote SSH server allows Claude Code to execute commands on remote machines (like GPU servers) directly.

### 1. Configure Connection

Create a `.env` file with your SSH connection details:

**Option A: Automatic Setup (Recommended)**
```bash
./scripts/setup-mcp-env.sh
```

**Option B: Manual Setup**
```bash
# Copy the example file
cp .env.example .env

# Find your SSH connection details
ssh -G <your-cluster-name> | grep -E "^(hostname|port|user)"

# Edit .env with your values
# SSH_HOST=root@localhost
# SSH_PORT=10022
```

### 2. Configure Claude Code

The `.mcp.json` file is already configured to load the MCP server:

```json
{
  "mcpServers": {
    "remote-ssh": {
      "command": "node",
      "args": ["${PWD}/mcp-server/dist/index.js"]
    }
  }
}
```

The server will automatically load settings from `.env` in the project root.

### 3. Restart Claude Code

After creating/updating `.env`, restart Claude Code to load the MCP server:

```bash
# Exit current session (Ctrl+D or type 'exit')
# Then start a new session in this directory
cd /path/to/bucket-brigade
claude
```

### 4. Test Connection

Once Claude Code restarts, test that the MCP tools are available:

```typescript
// Test basic connection
remote_bash({
  command: "hostname && whoami",
  description: "Test remote connection"
})

// Check GPU
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU availability"
})
```

## Configuration Details

### .env File Format

```bash
# Required: SSH connection details
SSH_HOST=user@hostname    # e.g., root@localhost
SSH_PORT=10022           # SSH port number

# Optional: Custom SSH key path
# SSH_KEY_PATH=~/.ssh/custom-key
```

### Getting Configuration for SkyPilot

If you're using SkyPilot, the SSH configuration is automatically created when you launch a cluster:

```bash
# 1. Check your cluster status
sky status

# 2. Get SSH configuration (replace 'your-cluster' with actual name)
ssh -G your-cluster | grep -E "^(hostname|port|user)"

# Example output:
#   hostname localhost
#   port 10022
#   user root

# 3. Use these values in .env:
#   SSH_HOST=root@localhost
#   SSH_PORT=10022
```

## Available MCP Tools

Once configured, Claude Code will have access to:

### `remote_bash`
Execute commands on the remote machine.

**Parameters:**
- `command` (string): The bash command to execute
- `description` (string): Human-readable description
- `run_in_background` (boolean): Run as background job
- `timeout` (number): Timeout in milliseconds (default: 120000)

**Example:**
```typescript
remote_bash({
  command: "cd /workspace && ls -la",
  description: "List workspace directory"
})
```

### `remote_file_read`
Read files from the remote machine.

**Parameters:**
- `file_path` (string): Absolute path to file on remote
- `offset` (number): Line number to start reading from
- `limit` (number): Number of lines to read

**Example:**
```typescript
remote_file_read({
  file_path: "/workspace/logs/training.log",
  limit: 50
})
```

### `remote_bash_output`
Get output from a background job.

**Parameters:**
- `bash_id` (string): Job ID from run_in_background

**Example:**
```typescript
// Start background job
const result = remote_bash({
  command: "sleep 5 && echo 'Done'",
  description: "Long running task",
  run_in_background: true
})

// Check output later
remote_bash_output({
  bash_id: result.job_id
})
```

## Troubleshooting

### MCP Server Not Loading

Check if the server is running:
```bash
# Look for the remote-ssh MCP server in Claude Code startup logs
# It should show: "✓ SSH connected to user@hostname"
```

If not loading:
1. Verify `.env` file exists and has correct format
2. Check `.mcp.json` uses `${PWD}` (not hardcoded path)
3. Ensure `mcp-server/dist/index.js` exists (run `pnpm build` if missing)
4. Restart Claude Code completely

### Connection Fails

Test SSH connection manually:
```bash
# Use values from your .env file
ssh -p <SSH_PORT> <SSH_HOST> "echo 'Connection works'"
```

Common issues:
- **Port forwarding stopped**: SkyPilot may need `sky start <cluster>`
- **Wrong credentials**: Check SSH key permissions (`chmod 600 ~/.ssh/key`)
- **Firewall**: Ensure port is accessible from your machine

### Commands Timeout

Increase timeout for long-running commands:
```typescript
remote_bash({
  command: "long_running_task",
  description: "This takes a while",
  timeout: 600000  // 10 minutes
})
```

Or use background jobs:
```typescript
remote_bash({
  command: "very_long_task",
  description: "This takes hours",
  run_in_background: true  // Don't wait for completion
})
```

## Security Notes

- `.env` is gitignored - never commit SSH credentials
- `.env.example` provides a template without sensitive data
- MCP server uses SSH key authentication (respects SSH config)
- Connections use your system's SSH agent and known_hosts

## Next Steps

After setup:
1. See `TEST_PLAN.md` for testing procedures
2. See `AGENT_PROMPT.md` for usage workflow
3. See `SANDBOX_GUIDE.md` for remote training examples

---

**Created:** 2025-11-03
**Issue:** #10 - MCP Remote SSH for GPU Training

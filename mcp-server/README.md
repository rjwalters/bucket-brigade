# MCP Remote SSH Server

MCP server that allows Claude to execute commands on remote machines via SSH.

## Features

- **Remote Bash Execution**: Run commands on remote hosts
- **Background Jobs**: Start long-running tasks and monitor them
- **File Reading**: Read files from remote filesystem
- **Full SSH Config Support**: Uses native SSH command with ProxyCommand, ControlMaster, etc.
- **SSH Config Aliases**: Reference hosts by SSH config alias name

## Installation

```bash
cd mcp-server
npm install
npm run build
```

## Configuration

### Option 1: Project-level .env (Recommended)

Create a `.env` file in your project root:

```bash
# SSH_HOST can be an SSH config alias or user@host format
SSH_HOST=my-gpu-server
```

Add `.mcp.json` to your project root:

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

The MCP server will automatically load the `.env` file and use your `~/.ssh/config` settings.

### Option 2: Global MCP Settings

Add to your Claude Code MCP settings (`~/.claude.json`):

```json
{
  "mcpServers": {
    "remote-ssh": {
      "command": "node",
      "args": ["/absolute/path/to/bucket-brigade/mcp-server/dist/index.js"],
      "env": {
        "SSH_HOST": "my-gpu-server"
      }
    }
  }
}
```

### Environment Variables

- `SSH_HOST`: SSH config alias (e.g., `my-server`) or user@host format (e.g., `user@host.com`)
  - Uses your `~/.ssh/config` settings including ProxyCommand, ControlMaster, etc.
  - Supports all SSH configuration options

### SSH Config Example

In `~/.ssh/config`:

```
Host my-gpu-server
  HostName localhost
  Port 10022
  User root
  ProxyCommand ssh -W %h:%p jump-host.example.com
  ControlMaster auto
  ControlPath ~/.ssh/control-%r@%h:%p
  ControlPersist 10m
```

Then set `SSH_HOST=my-gpu-server` in your `.env` file.

## Usage

Once configured, Claude will have access to these tools:

### `remote_bash`

Execute commands on the remote host.

```typescript
// Example: Check GPU status
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU utilization"
})

// Example: Start training in background
remote_bash({
  command: "./scripts/sandbox.sh train 500000",
  description: "Start 500K step training",
  run_in_background: true
})
```

### `remote_bash_output`

Get output from background jobs.

```typescript
remote_bash_output({
  bash_id: "remote-1"
})
```

### `remote_file_read`

Read files from remote host.

```typescript
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training_500000.log",
  offset: 0,
  limit: 100
})
```

## Development

```bash
# Watch mode during development
npm run watch

# Run directly with tsx
npm run dev
```

## Architecture

This MCP server uses the native SSH command (`ssh`) instead of a Node.js SSH library. This design provides:

- **Full SSH Config Support**: All `~/.ssh/config` directives work (ProxyCommand, ControlMaster, etc.)
- **Jump Host Support**: Automatic support for bastion/jump hosts via ProxyCommand
- **Connection Multiplexing**: ControlMaster settings are respected for faster connections
- **Standard Behavior**: Works exactly like your normal SSH commands

## Security Notes

- Uses SSH key authentication (configured in `~/.ssh/config`)
- Leverages your existing SSH configuration and keys
- Commands execute with permissions of SSH user
- Be cautious about running untrusted commands

## Example Workflow

1. Configure MCP server with remote GPU machine
2. Claude can now:
   - Check GPU availability: `nvidia-smi`
   - Start training: `./scripts/sandbox.sh train 500000`
   - Monitor progress: read log files
   - Check status: `./scripts/sandbox.sh status`
   - Attach to tmux sessions
   - Pull trained models

## Troubleshooting

**Connection fails:**
- Test SSH manually: `ssh $SSH_HOST` (should work without password)
- Check SSH config: `ssh -G $SSH_HOST` to see effective configuration
- Verify SSH config alias exists in `~/.ssh/config`
- For ProxyCommand setups, test jump host connectivity first

**Commands timeout:**
- Increase timeout parameter in tool call
- Use `run_in_background: true` for long commands
- Check if ControlMaster is working: `ssh -O check $SSH_HOST`

**Permission denied:**
- Ensure SSH user has necessary permissions
- Check file paths are absolute
- Verify SSH key authentication is configured correctly

**MCP server not starting:**
- Check that `.env` file exists and has `SSH_HOST` set
- Rebuild the server: `npm run build`
- Test server directly: `node mcp-server/dist/index.js` (should show connection test)

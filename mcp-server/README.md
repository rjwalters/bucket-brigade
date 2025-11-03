# MCP Remote SSH Server

MCP server that allows Claude to execute commands on remote machines via SSH.

## Features

- **Remote Bash Execution**: Run commands on remote hosts
- **Background Jobs**: Start long-running tasks and monitor them
- **File Reading**: Read files from remote filesystem
- **SSH Key Authentication**: Secure connection using SSH keys

## Installation

```bash
cd mcp-server
npm install
npm run build
```

## Configuration

Add to your Claude Code MCP settings (`~/.config/claude-code/mcp_settings.json`):

```json
{
  "mcpServers": {
    "remote-ssh": {
      "command": "node",
      "args": ["/absolute/path/to/bucket-brigade/mcp-server/dist/index.js"],
      "env": {
        "SSH_HOST": "user@remote-host.com",
        "SSH_PORT": "22",
        "SSH_KEY_PATH": "/Users/you/.ssh/id_rsa"
      }
    }
  }
}
```

### Environment Variables

- `SSH_HOST`: Remote host in format `user@hostname`
- `SSH_PORT`: SSH port (default: 22)
- `SSH_KEY_PATH`: Path to SSH private key (optional, uses default if not specified)

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

## Security Notes

- Uses SSH key authentication (no passwords)
- SSH connection is established once and reused
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
- Verify SSH key has correct permissions (`chmod 600 ~/.ssh/id_rsa`)
- Test SSH manually: `ssh user@host`
- Check SSH_HOST format: `user@hostname`

**Commands timeout:**
- Increase timeout parameter
- Use `run_in_background: true` for long commands

**Permission denied:**
- Ensure SSH user has necessary permissions
- Check file paths are absolute

# MCP Server Setup Guide

This repository includes an MCP (Model Context Protocol) server for remote SSH access, allowing Claude to execute commands on remote machines.

## Quick Setup

### 1. Configure SSH Host

Create a `.env` file in the repository root:

```bash
echo "SSH_HOST=your-remote-host" > .env
```

Replace `your-remote-host` with:
- An SSH config alias (e.g., `my-gpu-server`)
- Or a direct host (e.g., `user@host.com`)

### 2. Ensure MCP Server is Built

The built MCP server (`mcp-server/dist/`) is committed to the repository, so it should work immediately.

If you need to rebuild:

```bash
cd mcp-server
npm install
npm run build
```

### 3. Verify Configuration

Check that `.mcp.json` exists in the repository root:

```bash
cat .mcp.json
```

Should contain:
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

### 4. Restart Claude Code

For MCP servers to load, restart your Claude Code session:
- Exit Claude Code
- Start a new session in the repository

## Usage

Once configured, Claude will have access to these MCP tools:

### `mcp__remote-ssh__remote_bash`
Execute commands on the remote host.

```typescript
mcp__remote-ssh__remote_bash({
  command: "nvidia-smi",
  description: "Check GPU status"
})
```

### `mcp__remote-ssh__remote_bash_output`
Get output from background jobs.

```typescript
mcp__remote-ssh__remote_bash_output({
  bash_id: "remote-1"
})
```

### `mcp__remote-ssh__remote_file_read`
Read files from the remote host.

```typescript
mcp__remote-ssh__remote_file_read({
  file_path: "/path/to/file.log",
  offset: 0,
  limit: 100
})
```

## SSH Configuration

The MCP server uses your standard SSH configuration. Configure your remote host in `~/.ssh/config`:

```ssh-config
Host my-gpu-server
  HostName remote.example.com
  User your-username
  Port 22
  IdentityFile ~/.ssh/id_rsa
  # Optional: Connection multiplexing for faster connections
  ControlMaster auto
  ControlPath ~/.ssh/control-%r@%h:%p
  ControlPersist 10m
```

Then set `SSH_HOST=my-gpu-server` in your `.env` file.

## Troubleshooting

### MCP Tools Not Available

**Check if Claude Code loaded the MCP server:**
- The tools should appear with the `mcp__remote-ssh__` prefix
- If not, verify `.mcp.json` exists and restart Claude Code

**Verify .env file:**
```bash
cat .env
# Should show: SSH_HOST=your-host
```

### Connection Errors

**Test SSH manually first:**
```bash
ssh your-remote-host
# Should connect without prompting for password
```

**Check effective SSH configuration:**
```bash
ssh -G your-remote-host
```

**Verify the host is in SSH config:**
```bash
grep -A 10 "^Host your-remote-host" ~/.ssh/config
```

### MCP Server Not Starting

**Rebuild the server:**
```bash
cd mcp-server
npm install
npm run build
```

**Test server directly:**
```bash
cd mcp-server
node dist/index.js
# Should show: "Testing SSH connection to: your-host"
```

## Security Notes

- Uses SSH key authentication (no passwords stored)
- Leverages your existing SSH configuration and keys
- Commands execute with permissions of the SSH user
- The `.env` file is gitignored (not committed)

## Per-Branch/Worktree Setup

The MCP server configuration is global to the repository:
- `.mcp.json` is committed and shared across branches
- `mcp-server/dist/` is committed for immediate availability
- `.env` is gitignored and needs to be created per clone

When setting up a new worktree:
1. The `.mcp.json` and `mcp-server/dist/` are already there
2. Just create the `.env` file: `echo "SSH_HOST=your-host" > .env`
3. Restart Claude Code in that worktree

## Architecture

The MCP server uses the native `ssh` command instead of a Node.js SSH library. This provides:

- **Full SSH Config Support**: All `~/.ssh/config` directives work
- **Jump Host Support**: ProxyCommand setups work automatically
- **Connection Multiplexing**: ControlMaster settings are respected
- **Standard Behavior**: Works exactly like your normal SSH commands

## Additional Resources

- [MCP Server README](../mcp-server/README.md) - Detailed server documentation
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- SSH Config Manual: `man ssh_config`

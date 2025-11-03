# Instructions for Agent After Claude Code Restart

## Current Status

**FIXED**: MCP configuration now uses the correct `.mcp.json` file format (previously was using wrong file `.claude/mcp_settings.json`).

The MCP server is built and configured. After restarting Claude Code, the tools (`remote_bash`, `remote_file_read`, `remote_bash_output`) should be available.

## Critical: Start in the Correct Directory

**You MUST start Claude Code in the Issue #10 worktree:**

```bash
cd /Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10
```

**DO NOT** start in the main repo directory. Claude Code loads MCP servers from `.mcp.json` in the current working directory.

## Step 1: Verify You're in the Right Place

```bash
pwd
# Expected output: /Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10

git branch --show-current
# Expected output: feature/issue-10
```

## Step 2: Verify MCP Tools Are Loaded

After Claude Code restarts, check if you have access to these tools:
- `remote_bash`
- `remote_bash_output`
- `remote_file_read`

**Test by trying to use one:**

```typescript
remote_bash({
  command: "echo 'Hello from remote!'",
  description: "Test MCP connection"
})
```

### If Tools ARE Available ✅

Great! Follow `TEST_PLAN.md` to verify functionality:
1. Test basic command execution
2. Check GPU status with `nvidia-smi`
3. Start the 500K training run

### If Tools Are NOT Available ❌

The MCP server didn't load. Possible causes:

#### Issue 1: Wrong Working Directory
- You started Claude Code in the wrong directory
- **Fix**: Restart Claude Code from the worktree path above

#### Issue 2: Wrong Configuration File
- **Old (incorrect)**: `.claude/mcp_settings.json`
- **New (correct)**: `.mcp.json` at project root
- **Fix**: The correct `.mcp.json` file has been created with this content:

```json
{
  "mcpServers": {
    "remote-ssh": {
      "command": "node",
      "args": ["${PWD}/mcp-server/dist/index.js"],
      "env": {
        "SSH_HOST": "rwalters-sandbox-1",
        "SSH_PORT": "22"
      }
    }
  }
}
```

The `${PWD}` variable expands to the current working directory when Claude Code starts, making the configuration portable across worktrees.

Then restart Claude Code again.

#### Issue 3: Multiple Claude Code Instances
- Other instances shouldn't interfere, but to be safe:
- Close ALL Claude Code instances
- Start only one instance in the worktree directory

#### Issue 4: MCP Server Fails to Start
Check if the MCP server can run manually:

```bash
cd /Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10
SSH_HOST=rwalters-sandbox-1 SSH_PORT=22 node mcp-server/dist/index.js
```

Expected output: "MCP Remote SSH Server running on stdio"

If it fails, check:
- Node.js is installed: `node --version`
- Dependencies are installed: `cd mcp-server && npm install`
- Server is built: `ls -la mcp-server/dist/index.js`

## Step 3: Once MCP Tools Work

Follow the workflow in `TEST_PLAN.md`:

1. **Test basic connectivity**
   ```typescript
   remote_bash({
     command: "echo 'Connection test'",
     description: "Test remote connection"
   })
   ```

2. **Check GPU**
   ```typescript
   remote_bash({
     command: "nvidia-smi",
     description: "Verify GPU availability"
   })
   ```

3. **Pull latest code on remote**
   ```typescript
   remote_bash({
     command: "cd /workspace/bucket-brigade && git pull",
     description: "Sync code to remote"
   })
   ```

4. **Start 500K training**
   ```typescript
   remote_bash({
     command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh train 500000",
     description: "Start 500K step training",
     run_in_background: true
   })
   ```

5. **Monitor progress** (see TEST_PLAN.md for details)

## Quick Reference

- **Worktree path**: `/Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10`
- **MCP config**: `.claude/mcp_settings.json` (in worktree)
- **MCP server**: `mcp-server/dist/index.js` (in worktree)
- **Remote host**: `rwalters-sandbox-1` (in SSH config)
- **Remote repo**: `/workspace/bucket-brigade/`
- **Test plan**: `TEST_PLAN.md`
- **Sandbox guide**: `SANDBOX_GUIDE.md`

## Troubleshooting Session Summary

**What we checked:**
- ✅ MCP config exists in worktree: `.claude/mcp_settings.json`
- ✅ MCP server built and executable: `mcp-server/dist/index.js`
- ✅ MCP server runs manually
- ✅ Config syntax is correct
- ❌ MCP tools not loaded in Claude Code session

**Hypothesis:**
Claude Code needs to start in the worktree directory for the MCP server to load with the correct relative paths.

**Solution Applied:**
Updated `.claude/mcp_settings.json` to use `${PWD}` environment variable expansion for portable path resolution. This allows the MCP server to load correctly regardless of which worktree you're in, as long as the server files exist relative to the starting directory.

**Next Action:**
Restart Claude Code from the worktree directory and verify tools are available.

---

**Created**: 2025-11-03
**Session**: Issue #10 MCP troubleshooting

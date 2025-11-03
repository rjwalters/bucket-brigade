# Sandbox Development Guide

## Overview

This guide explains how to work with remote sandbox environments (GPU machines) via MCP Remote SSH tools. It's designed for Claude Code agents managing long-running tasks like training runs.

## Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  Local Machine  │         │  GitHub (sync)   │         │ Remote Sandbox  │
│  (Development)  │ ◄─────► │                  │ ◄─────► │   (GPU/Compute) │
│                 │         │                  │         │                 │
│  Claude Code    │         │  git push/pull   │         │  /workspace/    │
│  + MCP Server   │         │                  │         │  bucket-brigade │
└─────────────────┘         └──────────────────┘         └─────────────────┘
         │                                                         ▲
         └─────────────────────────────────────────────────────────┘
                          MCP Remote SSH Tools
                    (remote_bash, remote_file_read, etc.)
```

## Core Principles

### 1. **GitHub is the Source of Truth**

All code changes must flow through GitHub:
- **Local → GitHub**: `git push` from local worktree
- **GitHub → Remote**: `git pull` on remote sandbox
- **Never edit directly on remote** - Changes will be lost!

### 2. **Remote Directory Convention**

Remote repositories should be located at:
```
/workspace/{repo-name}/
```

For this project:
```
/workspace/bucket-brigade/
```

### 3. **MCP Tools for Remote Access**

Use MCP Remote SSH tools instead of manual SSH:
- ✅ Better for automation and monitoring
- ✅ Integrated with Claude Code
- ✅ Automatic logging and error handling
- ❌ Don't use `Bash(ssh ...)` commands

## MCP Remote SSH Tools

### Available Tools

After Claude Code restarts with MCP server loaded, you'll have:

#### 1. `remote_bash` - Execute commands remotely

```typescript
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU status",
  timeout: 30000,  // Optional: milliseconds
  run_in_background: false  // Optional: for long-running tasks
})
```

**Use cases:**
- Run training scripts
- Check system status
- Git operations
- Process management

#### 2. `remote_bash_output` - Read background job output

```typescript
remote_bash_output({
  bash_id: "remote-1",  // From background job
  filter: "epoch"  // Optional: regex filter
})
```

**Use cases:**
- Monitor training progress
- Check logs from background jobs
- Track long-running processes

#### 3. `remote_file_read` - Read remote files

```typescript
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training.log",
  offset: 0,  // Optional: start line
  limit: 100  // Optional: number of lines
})
```

**Use cases:**
- Read training logs
- Check configuration files
- Verify output files
- Monitor results

## Workflow Patterns

### Pattern 1: Code Change → Remote Execution

When you make code changes locally:

```typescript
// 1. Commit and push changes locally
Bash("git add . && git commit -m 'Update training script'")
Bash("git push")

// 2. Pull changes on remote
remote_bash({
  command: "cd /workspace/bucket-brigade && git pull",
  description: "Pull latest changes"
})

// 3. Run updated code
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/train.sh",
  description: "Run training with new code"
})
```

### Pattern 2: Start Long-Running Task

For tasks that take hours (like training):

```typescript
// 1. Ensure code is up to date
remote_bash({
  command: "cd /workspace/bucket-brigade && git pull",
  description: "Sync latest code"
})

// 2. Start background task
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/train_simple.py --steps 500000",
  description: "Start 500K step training",
  run_in_background: true
})

// 3. Monitor periodically
remote_bash_output({
  bash_id: "remote-1"
})

// 4. Or read log files
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training_500k.log",
  limit: 50
})
```

### Pattern 3: Check Status and Troubleshoot

```typescript
// Check what's running
remote_bash({
  command: "ps aux | grep python",
  description: "Check Python processes"
})

// Check GPU usage
remote_bash({
  command: "nvidia-smi",
  description: "GPU utilization"
})

// Check disk space
remote_bash({
  command: "df -h /workspace",
  description: "Check available space"
})

// Read recent errors
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/error.log",
  limit: 100
})
```

### Pattern 4: Retrieve Results

```typescript
// 1. Check if training completed
remote_bash({
  command: "ls -lh /workspace/bucket-brigade/models/",
  description: "List trained models"
})

// 2. Read training summary
remote_file_read({
  file_path: "/workspace/bucket-brigade/results/summary.json",
  limit: 1000
})

// 3. Copy results back via git (if applicable)
remote_bash({
  command: "cd /workspace/bucket-brigade && git add models/ results/ && git commit -m 'Add training results' && git push",
  description: "Push results to GitHub"
})

// 4. Pull results locally
Bash("git pull")
```

## Common Tasks

### Start Training Run

```typescript
// Sync code
remote_bash({
  command: "cd /workspace/bucket-brigade && git pull",
  description: "Pull latest training code"
})

// Verify environment
remote_bash({
  command: "cd /workspace/bucket-brigade && source .venv/bin/activate && python --version",
  description: "Check Python environment"
})

// Start training
remote_bash({
  command: "cd /workspace/bucket-brigade && source .venv/bin/activate && python scripts/train_simple.py --num-steps 500000 --save-path models/policy_500k.pt",
  description: "Start 500K training",
  run_in_background: true,
  timeout: 3600000  // 1 hour timeout
})
```

### Monitor Training Progress

```typescript
// Check recent output
remote_bash_output({
  bash_id: "remote-1"
})

// Read log tail
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training.log",
  limit: 50
})

// Check GPU usage
remote_bash({
  command: "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv",
  description: "GPU metrics"
})
```

### Debug Issues

```typescript
// Check if process is still running
remote_bash({
  command: "ps aux | grep train_simple",
  description: "Check training process"
})

// Read error logs
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/error.log",
  limit: 100
})

// Check Python traceback
remote_bash({
  command: "cd /workspace/bucket-brigade && tail -200 logs/training.log | grep -A 10 'Traceback'",
  description: "Find Python errors"
})
```

## Best Practices

### ✅ DO

1. **Always sync code through GitHub**
   ```typescript
   // Local: git push
   // Remote: git pull
   ```

2. **Use descriptive command descriptions**
   ```typescript
   remote_bash({
     command: "...",
     description: "Clear description of what this does"
   })
   ```

3. **Check status before starting new tasks**
   ```typescript
   // Verify nothing is running
   remote_bash({
     command: "ps aux | grep train",
     description: "Check for existing training"
   })
   ```

4. **Set appropriate timeouts**
   ```typescript
   remote_bash({
     command: "quick_check.sh",
     timeout: 5000  // 5 seconds for quick tasks
   })

   remote_bash({
     command: "long_training.sh",
     timeout: 3600000  // 1 hour for long tasks
   })
   ```

5. **Use background mode for long tasks**
   ```typescript
   remote_bash({
     command: "./train.sh",
     run_in_background: true  // Don't block
   })
   ```

### ❌ DON'T

1. **Don't edit files directly on remote**
   - Changes will be lost on next `git pull`
   - Makes debugging harder
   - Breaks reproducibility

2. **Don't use manual SSH via Bash tool**
   ```typescript
   // ❌ Bad
   Bash("ssh remote 'command'")

   // ✅ Good
   remote_bash({
     command: "command",
     description: "What it does"
   })
   ```

3. **Don't forget to sync code**
   ```typescript
   // ❌ Bad - using old code
   remote_bash({ command: "./train.sh" })

   // ✅ Good - sync first
   remote_bash({ command: "git pull && ./train.sh" })
   ```

4. **Don't leave zombie processes**
   ```typescript
   // Always check if you need to kill old processes
   remote_bash({
     command: "pkill -f train_simple.py",
     description: "Kill old training processes"
   })
   ```

5. **Don't assume file paths**
   ```typescript
   // ❌ Bad - might be wrong location
   remote_file_read({ file_path: "~/logs/training.log" })

   // ✅ Good - use full path
   remote_file_read({ file_path: "/workspace/bucket-brigade/logs/training.log" })
   ```

## Troubleshooting

### MCP Tools Not Available

**Problem**: `remote_bash` and other tools not found

**Solution**: Claude Code needs to restart to load MCP server
1. Save your work
2. Exit Claude Code
3. Restart Claude Code
4. Tools should now be available

### Connection Failures

**Problem**: SSH connection times out or fails

**Solution**: Check SSH configuration
```bash
# Test SSH manually
ssh rwalters-sandbox-1 "echo 'Connection works'"

# Verify SSH config
cat ~/.ssh/config | grep -A 5 "rwalters-sandbox-1"
```

### Remote Directory Not Found

**Problem**: `/workspace/bucket-brigade/` doesn't exist

**Solution**: Clone repository on remote
```typescript
remote_bash({
  command: "cd /workspace && git clone https://github.com/rjwalters/bucket-brigade.git",
  description: "Clone repository to sandbox"
})
```

### Git Pull Conflicts

**Problem**: `git pull` fails due to conflicts

**Solution**: Remote should never have local changes, so hard reset
```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && git fetch origin && git reset --hard origin/main",
  description: "Hard reset to remote main"
})
```

### Background Job Lost

**Problem**: Can't find `bash_id` for background job

**Solution**: Check process manually
```typescript
remote_bash({
  command: "ps aux | grep train_simple",
  description: "Find training process"
})

// Or use tmux for persistent sessions
remote_bash({
  command: "tmux list-sessions",
  description: "List tmux sessions"
})
```

## Environment Setup (First Time)

When setting up a new sandbox:

```typescript
// 1. Clone repository
remote_bash({
  command: "cd /workspace && git clone https://github.com/rjwalters/bucket-brigade.git",
  description: "Clone repository"
})

// 2. Install dependencies
remote_bash({
  command: "cd /workspace/bucket-brigade && uv sync",
  description: "Install Python dependencies"
})

// 3. Verify GPU
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU availability"
})

// 4. Create logs directory
remote_bash({
  command: "mkdir -p /workspace/bucket-brigade/logs",
  description: "Create logs directory"
})

// 5. Test basic functionality
remote_bash({
  command: "cd /workspace/bucket-brigade && uv run python -c 'import torch; print(torch.cuda.is_available())'",
  description: "Test PyTorch CUDA"
})
```

## Example: Complete Training Workflow

Here's a full example of running a training session:

```typescript
// 1. Commit local changes
Bash("git add scripts/train_simple.py && git commit -m 'Improve training script'")
Bash("git push")

// 2. Sync remote
remote_bash({
  command: "cd /workspace/bucket-brigade && git pull",
  description: "Pull latest changes"
})

// 3. Check GPU availability
remote_bash({
  command: "nvidia-smi",
  description: "Verify GPU is free"
})

// 4. Start training
remote_bash({
  command: "cd /workspace/bucket-brigade && source .venv/bin/activate && python scripts/train_simple.py --num-steps 500000 --save-path models/policy_500k.pt 2>&1 | tee logs/training_500k.log",
  description: "Start 500K step training",
  run_in_background: true
})

// 5. Wait a bit, then check progress
// (wait 5 minutes)

// 6. Monitor
remote_bash_output({
  bash_id: "remote-1",
  filter: "step"  // Only show lines with "step" in them
})

remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training_500k.log",
  limit: 50
})

// 7. Check GPU usage
remote_bash({
  command: "nvidia-smi",
  description: "GPU utilization"
})

// 8. After training completes, commit results
remote_bash({
  command: "cd /workspace/bucket-brigade && git add models/policy_500k.pt && git commit -m 'Add 500K trained policy' && git push",
  description: "Save trained model"
})

// 9. Pull results locally
Bash("git pull")
```

## Security Notes

- SSH keys should be properly configured and secured
- Never commit secrets to the repository
- Use environment variables for sensitive configuration
- The MCP server reads `SSH_HOST` from `.claude/mcp_settings.json`

## Related Files

- `.claude/mcp_settings.json` - MCP server configuration
- `mcp-server/` - MCP Remote SSH server implementation
- `TEST_PLAN.md` - Testing checklist for MCP server
- `TODO.md` - Current task status (per issue worktree)

---

**Created**: 2025-11-03
**For**: Issue #10 - GPU Training Orchestration

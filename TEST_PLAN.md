# MCP Remote SSH Server - Test Plan

## Current Status

✅ **MCP Server built** - Available at `mcp-server/dist/index.js`
✅ **MCP configuration** - `.mcp.json` loads server automatically
✅ **.env support** - SSH connection configured via `.env` file
⏳ **Requires Claude Code restart** - To load MCP tools with new .env

## ⚠️ IMPORTANT: Setup & Restart Required

### 1. Configure SSH Connection

**Option A: Automatic (Recommended)**
```bash
./scripts/setup-mcp-env.sh
```

**Option B: Manual**
```bash
# Copy template
cp .env.example .env

# Get your SSH settings (for SkyPilot)
ssh -G <your-cluster-name> | grep -E "^(hostname|port|user)"

# Edit .env with your values
vim .env
```

See `MCP_SETUP.md` for detailed configuration instructions.

### 2. Restart Claude Code

After creating `.env`, restart Claude Code:
```bash
exit
claude
```

The MCP server will load automatically and connect using `.env` settings.

### 3. Verify MCP Server is Loaded

In a new Claude Code session, check that you have access to these tools:
- `remote_bash`
- `remote_bash_output`
- `remote_file_read`

📖 **See `MCP_SETUP.md` for setup details and `SANDBOX_GUIDE.md` for usage patterns**

## Test Cases

### Test 1: Basic Command Execution

```typescript
remote_bash({
  command: "echo 'Hello from remote!'",
  description: "Test basic command execution"
})
```

**Expected:** Returns "Hello from remote!"

### Test 2: Check GPU Status

```typescript
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU availability"
})
```

**Expected:** Shows NVIDIA L4 GPU info

### Test 3: Check Training Status

```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh status",
  description: "Check if training is running"
})
```

**Expected:** Shows active training sessions and processes

### Test 4: Read Training Logs

```typescript
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training_500000.log",
  limit: 50
})
```

**Expected:** Returns last 50 lines of training log

### Test 5: Background Job

```typescript
// Start a background job
remote_bash({
  command: "sleep 5 && echo 'Done sleeping'",
  description: "Test background job",
  run_in_background: true
})

// Then check output (wait a few seconds)
remote_bash_output({
  bash_id: "remote-1"
})
```

**Expected:** First call returns job ID, second call shows output after sleep completes

## Full Workflow Test

Once MCP server is working, test the full GPU training workflow:

### 1. Pull Latest Code on Remote

```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && git pull",
  description: "Pull latest training code"
})
```

### 2. Start Training

```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh train 500000",
  description: "Start 500K step training"
})
```

### 3. Monitor Progress

```typescript
// Check status
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh status",
  description: "Check training status"
})

// Read logs
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training_500000.log",
  limit: 100
})

// Check GPU utilization
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU usage"
})
```

### 4. Attach to Tmux Session

```typescript
// List tmux sessions
remote_bash({
  command: "tmux list-sessions",
  description: "List training sessions"
})
```

## Troubleshooting

### MCP Server Not Loading

Check Claude Code logs:
```bash
tail -f ~/.config/claude-code/logs/main.log
```

### SSH Connection Fails

Test SSH manually:
```bash
ssh rwalters-sandbox-1 "echo 'Connection works'"
```

Verify SSH config in `~/.ssh/config`:
```
Host rwalters-sandbox-1
    HostName <ip-address>
    User root
    IdentityFile ~/.ssh/your-key
```

### Permission Issues

Ensure SSH key has correct permissions:
```bash
chmod 600 ~/.ssh/your-key
```

## Issue #10 - GPU Training

Once MCP server is confirmed working, we can directly manage the GPU training:

1. **Verify GPU is available** - `nvidia-smi`
2. **Start training with GPU support** - Training script now has full CUDA support
3. **Monitor progress** - Every 1,000 steps
4. **Check logs** - Real-time log viewing
5. **Pull trained models** - When complete

The training script improvements from this session:
- ✅ Full GPU/CUDA support added
- ✅ Progress logging every 1,000 steps (was 10,000)
- ✅ Verbose debug output
- ✅ Auto-flush for real-time visibility
- ✅ sandbox.sh auto-setup and tmux integration

## Next Steps After Testing

1. Verify MCP server works with all test cases
2. Start actual 500K training run
3. Monitor until completion (should be much faster on GPU!)
4. Run 1M training
5. Evaluate models and compare results
6. Document findings in issue #10
7. Create PR with trained models

---

**Created:** 2025-11-03
**Session:** Issue #10 GPU training workflow

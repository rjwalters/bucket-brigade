# Welcome Back! MCP Remote SSH Server Setup

## Your Mission

You are continuing work on **Issue #10**: implementing MCP Remote SSH server for running GPU training on a remote machine. The MCP server has been built and configured. Your job is to **verify it works** and then **start the 500K training run**.

## Critical First Step: Verify MCP Tools Are Loaded

**Check if you have these three tools available:**
- `remote_bash` - Execute commands on remote GPU machine
- `remote_file_read` - Read files from remote machine
- `remote_bash_output` - Monitor background processes

**Test immediately:**
```typescript
remote_bash({
  command: "echo 'Hello from remote GPU machine!'",
  description: "Test MCP connection"
})
```

### If MCP Tools Work ✅

Great! Skip to **"Your Workflow"** section below.

### If MCP Tools DON'T Work ❌

Something went wrong with the MCP server loading. Read `RESTART_INSTRUCTIONS.md` for troubleshooting steps. The most common issues are:

1. **Wrong directory**: You must start Claude Code from `/Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10`
2. **Config issue**: Check `.claude/mcp_settings.json` has `${PWD}` in the path
3. **Server not built**: Verify `mcp-server/dist/index.js` exists

## Your Workflow (Once MCP Tools Work)

Follow the detailed test plan in `TEST_PLAN.md`. Here's the quick version:

### 1. Test Basic Connectivity
```typescript
remote_bash({
  command: "hostname && whoami",
  description: "Verify remote connection"
})
```

### 2. Check GPU Status
```typescript
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU availability"
})
```

Expected: Should show NVIDIA GPU information.

### 3. Sync Latest Code to Remote
```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && git fetch && git status",
  description: "Check remote repo status"
})
```

If remote is behind, pull latest changes:
```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && git pull origin main",
  description: "Sync code to remote"
})
```

### 4. Start 500K Training Run
```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh train 500000",
  description: "Start 500K step training run",
  run_in_background: true
})
```

**IMPORTANT**: Use `run_in_background: true` so the training runs in the background and doesn't block.

### 5. Monitor Training Progress

The training will take 1-2 hours. Monitor it with:

```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && tail -20 logs/training_500k.log",
  description: "Check training progress"
})
```

Look for:
- ✅ Training steps progressing (should show step numbers)
- ✅ Loss values decreasing over time
- ✅ No error messages or crashes
- ✅ Regular checkpoint saves

**Check every 15-30 minutes** during the run.

### 6. When Training Completes

After 500K steps complete successfully:

```typescript
remote_bash({
  command: "cd /workspace/bucket-brigade && ls -lh checkpoints/",
  description: "Verify checkpoint was saved"
})
```

Expected: Should see `checkpoint_500k.pt` file (around 100-500 MB).

### 7. Document Results

Create a summary in `TRAINING_RESULTS.md`:
- Training duration
- Final loss value
- Any issues encountered
- Checkpoint file size
- GPU utilization stats

## Key Files Reference

- **`TEST_PLAN.md`**: Detailed testing procedures
- **`SANDBOX_GUIDE.md`**: How the remote sandbox works
- **`RESTART_INSTRUCTIONS.md`**: Troubleshooting if MCP tools don't load
- **`.claude/mcp_settings.json`**: MCP configuration
- **`TODO.md`**: Overall issue tracking

## Remote Machine Details

- **SSH Host**: `rwalters-sandbox-1` (configured in `~/.ssh/config`)
- **Repository**: `/workspace/bucket-brigade/`
- **GPU**: NVIDIA GPU (verify with `nvidia-smi`)
- **Training Script**: `./scripts/sandbox.sh train <steps>`
- **Log Location**: `logs/training_500k.log`
- **Checkpoints**: `checkpoints/`

## What NOT to Do

- ❌ Don't use local `Bash` tool for remote commands - use `remote_bash` instead
- ❌ Don't start training without `run_in_background: true` - it will timeout
- ❌ Don't forget to monitor progress - training can fail silently
- ❌ Don't skip GPU verification - training will fail without GPU
- ❌ Don't assume code is synced - always check/pull latest on remote

## Success Criteria

Issue #10 is complete when:
1. ✅ MCP Remote SSH tools work correctly
2. ✅ 500K training run completes without errors
3. ✅ Checkpoint file is saved successfully
4. ✅ Results are documented
5. ✅ PR is ready for review

## If You Get Stuck

1. **Read** `TEST_PLAN.md` for detailed procedures
2. **Read** `SANDBOX_GUIDE.md` to understand the remote setup
3. **Read** `RESTART_INSTRUCTIONS.md` if MCP tools aren't working
4. **Check** `TODO.md` for the overall task list
5. **Ask** the user if something is unclear

## Ready?

Start by verifying the MCP tools are loaded, then follow the workflow above. Good luck! 🚀

---

**Created**: 2025-11-03
**Issue**: #10 - MCP Remote SSH for GPU Training
**Branch**: `feature/issue-10`
**Status**: MCP server configured, ready for testing

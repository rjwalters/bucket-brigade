# Issue #10 - Longer Training Runs (GPU)

## Goal

Improve trained policy performance by running longer training sessions on GPU hardware.

## Background

Initial training runs used only 50K steps, which resulted in:
- Mean reward: -79.82 ± 177.70
- High variance in episode performance
- Inconsistent cooperation behavior

Longer training (500K-1M steps) should allow the policy to:
- Learn better coordination strategies
- Reduce variance in performance
- Achieve more consistent positive rewards

## Current Status

### ✅ Completed

1. **MCP Remote SSH Server** - Built and committed to main
   - `mcp-server/` - TypeScript MCP server for remote SSH access
   - `.mcp.json` - MCP configuration (FIXED: was using wrong file)
   - Tools: `remote_bash`, `remote_bash_output`, `remote_file_read`
   - Target: `rwalters-sandbox-1` (GPU machine)

2. **Training Infrastructure**
   - GPU/CUDA support verified in training scripts
   - Progress logging every 1,000 steps
   - Auto-flush for real-time log visibility
   - Verbose debug output

3. **MCP Configuration Fix**
   - ❌ Previous attempt used wrong file: `.claude/mcp_settings.json`
   - ✅ **FIXED**: Created correct `.mcp.json` file at project root
   - ✅ Updated documentation to reflect correct configuration
   - ✅ Committed changes to both main and feature/issue-10 branches
   - **Root cause**: Claude Code uses `.mcp.json`, not `.claude/mcp_settings.json`

### ⏳ Next Steps - AFTER RESTART

**🚨 CRITICAL: Start Claude Code in the worktree directory:**
```bash
cd /Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10
```

1. **Verify MCP tools loaded** - Try using `remote_bash` tool
2. **If NOT loaded** - See `RESTART_INSTRUCTIONS.md` for troubleshooting
3. **Once loaded** - Follow `TEST_PLAN.md`:
   - Test basic connectivity
   - Check GPU availability (`nvidia-smi`)
   - Pull latest code on remote
   - Start 500K training run
   - Monitor progress
4. **After 500K completes** - Run 1M training if results are good
5. **Evaluate models** and compare performance metrics
6. **Document findings** and optimal training duration

## Commands

### Start Training (500K steps)

```bash
# Via MCP (once loaded):
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh train 500000",
  description: "Start 500K step training"
})
```

### Monitor Training

```bash
# Check status
remote_bash({
  command: "cd /workspace/bucket-brigade && ./scripts/sandbox.sh status",
  description: "Check training status"
})

# View logs
remote_file_read({
  file_path: "/workspace/bucket-brigade/logs/training_500000.log",
  limit: 100
})

# GPU utilization
remote_bash({
  command: "nvidia-smi",
  description: "Check GPU usage"
})
```

### Manual SSH (fallback)

```bash
ssh rwalters-sandbox-1 "cd /workspace/bucket-brigade && ./scripts/sandbox.sh train 500000"
ssh rwalters-sandbox-1 "nvidia-smi"
ssh rwalters-sandbox-1 "tail -100 /workspace/bucket-brigade/logs/training_500000.log"
```

## Success Criteria

- Mean reward > 0 (positive net benefit)
- Standard deviation < 100 (more consistent)
- At least 70% of episodes have positive rewards
- Training completes without errors
- Models saved to `models/` directory

## Files

- `scripts/train_simple.py` - Training script with GPU support
- `scripts/sandbox.sh` - Remote training orchestration
- `mcp-server/` - MCP Remote SSH server
- `TEST_PLAN.md` - Testing checklist for MCP server
- `.mcp.json` - MCP configuration (correct file location)
- `SANDBOX_GUIDE.md` - **Complete guide for working with remote environments**
- `RESTART_INSTRUCTIONS.md` - Troubleshooting guide for MCP setup

## Related Issues

- See `TRAINING_GUIDE.md` for training details
- Issue #11 - Hyperparameter tuning (blocked on this)

---

**Status**: Waiting for Claude Code restart to load MCP server
**Last Updated**: 2025-11-03

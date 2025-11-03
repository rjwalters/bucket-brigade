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

1. **MCP Remote SSH Server** - Built and committed
   - `mcp-server/` - TypeScript MCP server for remote SSH access
   - `.mcp.json` - MCP configuration at project root
   - Tools: `remote_bash`, `remote_bash_output`, `remote_file_read`
   - Target: Remote GPU machine via SSH

2. **Training Infrastructure**
   - GPU/CUDA support verified in training scripts
   - Progress logging every 1,000 steps
   - Auto-flush for real-time log visibility
   - Verbose debug output

3. **MCP Configuration Evolution**
   - ❌ Initial attempt: Wrong file `.claude/mcp_settings.json`
   - ✅ Fix #1: Created correct `.mcp.json` file at project root
   - ✅ Fix #2: Added `.env` support for easier SSH configuration
   - ✅ **Current**: `.env` file + setup script for convenience
   - **Benefits**: No hardcoded paths, works with SkyPilot port forwarding

4. **.env Configuration System**
   - `.env.example` - Template with detailed instructions
   - `.env` - Local configuration (gitignored)
   - `scripts/setup-mcp-env.sh` - Auto-configure from SSH config
   - `MCP_SETUP.md` - Comprehensive setup guide
   - Supports SkyPilot clusters with port forwarding

### ⏳ Next Steps - AFTER RESTART

**🚨 SETUP REQUIRED BEFORE RESTART:**
```bash
# 1. Configure SSH connection (if not done)
./scripts/setup-mcp-env.sh

# 2. Restart Claude Code from project directory
exit
cd /Users/rwalters/GitHub/bucket-brigade/.loom/worktrees/issue-10
claude
```

**After restart:**

1. **Verify MCP tools loaded** - Try using `remote_bash` tool
2. **If NOT loaded** - See `MCP_SETUP.md` for troubleshooting
3. **Once loaded** - Follow `TEST_PLAN.md`:
   - Test basic connectivity
   - Check GPU availability (`nvidia-smi`)
   - Pull latest code on remote
   - Start 500K training run with `run_in_background: true`
   - Monitor progress regularly
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
- `scripts/setup-mcp-env.sh` - Auto-configure .env from SSH config
- `mcp-server/` - MCP Remote SSH server (TypeScript)
- `.mcp.json` - MCP server configuration
- `.env.example` - Template for SSH connection settings
- `.env` - Local SSH configuration (gitignored)
- `MCP_SETUP.md` - **Complete MCP setup and troubleshooting guide**
- `TEST_PLAN.md` - Testing checklist for MCP server
- `AGENT_PROMPT.md` - Quick start guide for new sessions
- `SANDBOX_GUIDE.md` - Complete guide for working with remote environments

## Related Issues

- See `TRAINING_GUIDE.md` for training details
- Issue #11 - Hyperparameter tuning (blocked on this)

---

**Status**: Ready for restart - .env configuration complete
**Last Updated**: 2025-11-03
**Branch**: `feature/issue-10`

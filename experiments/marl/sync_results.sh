#!/bin/bash
# Sync training results from remote GPU instance
# Usage: ./sync_results.sh [remote-host] [run-name]

REMOTE_HOST="${1:-rwalters-sandbox-2}"
RUN_NAME="$2"

echo "📦 Syncing PPO training results from $REMOTE_HOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Find bucket-brigade directory on remote
REMOTE_DIR=$(ssh "$REMOTE_HOST" 'if [ -d ~/bucket-brigade ]; then echo "bucket-brigade"; elif [ -d ~/projects/bucket-brigade ]; then echo "projects/bucket-brigade"; elif [ -d bucket-brigade ]; then echo "bucket-brigade"; else echo ""; fi')

if [ -z "$REMOTE_DIR" ]; then
    echo "❌ Cannot find bucket-brigade directory on $REMOTE_HOST"
    exit 1
fi

echo "✓ Found remote repo at: ~/$REMOTE_DIR"

LOCAL_DIR="$(pwd)"
if [[ ! "$LOCAL_DIR" =~ "bucket-brigade" ]]; then
    echo "⚠️  Warning: Not in bucket-brigade directory. Results will sync to: $LOCAL_DIR"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Function to sync a directory
sync_dir() {
    local src="$1"
    local desc="$2"

    echo ""
    echo "📁 Syncing $desc..."

    if ssh "$REMOTE_HOST" "[ -d ~/$REMOTE_DIR/$src ]"; then
        rsync -avz --progress \
            "$REMOTE_HOST:~/$REMOTE_DIR/$src/" \
            "./$src/"

        echo "✅ $desc synced"
    else
        echo "⚠️  Remote directory not found: $src"
    fi
}

# Sync TensorBoard runs
if [ -n "$RUN_NAME" ]; then
    echo "🎯 Syncing specific run: $RUN_NAME"
    sync_dir "experiments/marl/runs/$RUN_NAME" "TensorBoard logs ($RUN_NAME)"
    sync_dir "experiments/marl/checkpoints/$RUN_NAME" "Checkpoints ($RUN_NAME)"
else
    echo "🎯 Syncing all runs"
    sync_dir "experiments/marl/runs" "All TensorBoard logs"
    sync_dir "experiments/marl/checkpoints" "All checkpoints"
fi

# Sync final models
sync_dir "experiments/marl/*.pt" "Final models"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Sync complete!"
echo ""
echo "📊 View results:"
echo "   tensorboard --logdir experiments/marl/runs/"
echo ""
echo "💡 Tips:"
echo "   - Sync specific run: ./sync_results.sh $REMOTE_HOST run_name"
echo "   - View checkpoints: ls -lh experiments/marl/checkpoints/"
echo "   - View models: ls -lh experiments/marl/*.pt"
echo ""

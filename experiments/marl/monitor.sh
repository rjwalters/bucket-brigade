#!/bin/bash
# Local monitoring script for remote GPU training
# Usage: ./monitor.sh [remote-host]

REMOTE_HOST="${1:-rwalters-sandbox-2}"

echo "📊 Monitoring PPO training on $REMOTE_HOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Function to check if port forwarding is already running
check_port_forward() {
    if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "✅ TensorBoard port (6006) already forwarded"
        return 0
    else
        return 1
    fi
}

# Setup port forwarding for TensorBoard if not already running
if ! check_port_forward; then
    echo "🔌 Setting up TensorBoard port forwarding..."
    echo "   Running: ssh -f -N -L 6006:localhost:6006 $REMOTE_HOST"
    ssh -f -N -L 6006:localhost:6006 "$REMOTE_HOST"

    if check_port_forward; then
        echo "✅ Port forwarding established"
    else
        echo "❌ Failed to setup port forwarding"
        exit 1
    fi
fi

echo ""
echo "📈 TensorBoard URLs:"
echo "   Local:  http://localhost:6006"
echo "   To stop port forwarding: pkill -f 'ssh.*6006'"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🖥️  Remote Training Status:"
echo ""

# Check training process
ssh "$REMOTE_HOST" << 'EOF'
# Find bucket-brigade directory
if [ -d ~/bucket-brigade ]; then
    cd ~/bucket-brigade
elif [ -d ~/projects/bucket-brigade ]; then
    cd ~/projects/bucket-brigade
elif [ -d bucket-brigade ]; then
    cd bucket-brigade
else
    echo "❌ Cannot find bucket-brigade directory"
    exit 1
fi

# Check for running Python processes
if pgrep -f "train_gpu.py" > /dev/null; then
    echo "✅ Training process is running"

    # Show process info
    ps aux | grep "train_gpu.py" | grep -v grep | awk '{print "   PID: "$2" | CPU: "$3"% | Mem: "$4"%"}'

    # Check GPU usage
    echo ""
    if command -v nvidia-smi &> /dev/null; then
        echo "🎮 GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader |
            awk -F, '{printf "   GPU %s: %s | Util: %s | Mem: %s/%s\n", $1, $2, $3, $4, $5}'
    fi
else
    echo "⚠️  No training process found"
fi

echo ""
echo "📁 Recent checkpoints:"
if [ -d "experiments/marl/checkpoints" ]; then
    ls -lth experiments/marl/checkpoints/*/*.pt 2>/dev/null | head -3 |
        awk '{print "   "$9" ("$6" "$7" "$8")"}'
else
    echo "   No checkpoints directory found"
fi

echo ""
echo "📊 Latest logs:"
if [ -f "experiments/marl/latest_training.log" ]; then
    tail -10 experiments/marl/latest_training.log | sed 's/^/   /'
else
    echo "   No training log found"
fi
EOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 Monitoring Tips:"
echo "   - View TensorBoard: open http://localhost:6006"
echo "   - Tail logs: ssh $REMOTE_HOST 'tail -f ~/bucket-brigade/experiments/marl/latest_training.log'"
echo "   - Check GPU: ssh $REMOTE_HOST 'watch -n 1 nvidia-smi'"
echo "   - Sync results: ./sync_results.sh $REMOTE_HOST"
echo ""

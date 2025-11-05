#!/bin/bash
# Monitor evolution experiment progress

REMOTE="rwalters-sandbox-1"

echo "Evolution Experiment Monitor"
echo "============================"
echo

# Check if tmux session exists
if ! ssh "$REMOTE" "tmux has-session -t evolution 2>/dev/null"; then
    echo "❌ No evolution tmux session found on $REMOTE"
    exit 1
fi

echo "📊 Recent output from evolution session:"
echo "----------------------------------------"
ssh "$REMOTE" "tmux capture-pane -t evolution -p | tail -40"

echo
echo "----------------------------------------"
echo

# Check active scenarios
echo "🔥 Active scenario processes:"
ssh "$REMOTE" "ps aux | grep 'run_evolution.py' | grep -v grep | awk '{print \$2, \$11, \$12, \$13, \$14}'"

echo
echo "📁 Completed scenarios:"
ssh "$REMOTE" "ls -1 bucket-brigade/experiments/scenarios/*/evolved/evolution_trace.json 2>/dev/null | wc -l | xargs echo '  scenarios with evolution data:'"

echo
echo "💾 Log files:"
ssh "$REMOTE" "ls -lth bucket-brigade/logs/evolution/ 2>/dev/null | head -10"

echo
echo "========================================"
echo "Commands:"
echo "  Attach to session:  ssh $REMOTE -t 'tmux attach -t evolution'"
echo "  Kill session:       ssh $REMOTE 'tmux kill-session -t evolution'"
echo "  Watch live:         watch -n 10 ./scripts/monitor_evolution.sh"
echo "========================================"

#!/bin/bash
# Setup MCP Remote SSH .env configuration
#
# This script helps configure the .env file for connecting to your
# SkyPilot or other remote GPU machine.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE="$PROJECT_ROOT/.env.example"

echo "=== MCP Remote SSH Configuration Setup ==="
echo ""

# Check if .env already exists
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  .env file already exists at: $ENV_FILE"
    read -p "Do you want to overwrite it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting. Edit $ENV_FILE manually if you need to change settings."
        exit 0
    fi
fi

# Get cluster name
echo "What is your SSH host/cluster name?"
echo "(e.g., 'rwalters-sandbox-1' for SkyPilot, or 'my-gpu-server')"
read -p "SSH host: " SSH_ALIAS

if [ -z "$SSH_ALIAS" ]; then
    echo "❌ SSH host cannot be empty"
    exit 1
fi

echo ""
echo "Getting SSH configuration for '$SSH_ALIAS'..."

# Try to get SSH config
if ! ssh -G "$SSH_ALIAS" &>/dev/null; then
    echo "❌ SSH host '$SSH_ALIAS' not found in SSH config"
    echo ""
    echo "For SkyPilot clusters, run: sky status"
    echo "For other hosts, check: ~/.ssh/config"
    exit 1
fi

# Extract connection details
HOSTNAME=$(ssh -G "$SSH_ALIAS" | grep "^hostname " | awk '{print $2}')
PORT=$(ssh -G "$SSH_ALIAS" | grep "^port " | awk '{print $2}')
USER=$(ssh -G "$SSH_ALIAS" | grep "^user " | awk '{print $2}')

echo "✓ Found SSH configuration:"
echo "  User: $USER"
echo "  Hostname: $HOSTNAME"
echo "  Port: $PORT"
echo ""

# Test connection
echo "Testing SSH connection..."
if ssh -p "$PORT" "$USER@$HOSTNAME" -o ConnectTimeout=5 "echo 'Connection successful!'" &>/dev/null; then
    echo "✓ SSH connection test passed!"
else
    echo "⚠️  SSH connection test failed (but continuing anyway)"
    echo "   You may need to start your SkyPilot cluster or check credentials"
fi

echo ""

# Create .env file
cat > "$ENV_FILE" <<EOF
# MCP Remote SSH Server Configuration
# Auto-configured for $SSH_ALIAS on $(date)

SSH_HOST=$USER@$HOSTNAME
SSH_PORT=$PORT

# Optional: Uncomment and set if using a non-default SSH key
# SSH_KEY_PATH=~/.ssh/sky-key
EOF

echo "✓ Created .env file at: $ENV_FILE"
echo ""
echo "Configuration:"
echo "  SSH_HOST=$USER@$HOSTNAME"
echo "  SSH_PORT=$PORT"
echo ""
echo "Next steps:"
echo "  1. Restart Claude Code to load the new configuration"
echo "  2. Test the connection with: remote_bash({command: 'hostname'})"
echo ""
echo "For manual configuration, see: $ENV_EXAMPLE"

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
HOSTNAME=$(ssh -G "$SSH_ALIAS" 2>/dev/null | grep "^hostname " | awk '{print $2}')
PORT=$(ssh -G "$SSH_ALIAS" 2>/dev/null | grep "^port " | awk '{print $2}')
USER=$(ssh -G "$SSH_ALIAS" 2>/dev/null | grep "^user " | awk '{print $2}')
PROXYCOMMAND=$(ssh -G "$SSH_ALIAS" 2>/dev/null | grep "^proxycommand " | sed 's/^proxycommand //')

# Check if using ProxyCommand (common for SkyPilot)
if [ -n "$PROXYCOMMAND" ]; then
    echo "✓ Detected ProxyCommand (SkyPilot/jump host configuration)"
    echo ""
    echo "MCP remote-ssh doesn't support ProxyCommand. You need to set up an SSH tunnel."
    echo ""
    echo "Run this command in a separate terminal (keep it running):"
    echo "  ssh -N -L $PORT:$HOSTNAME:$PORT $SSH_ALIAS"
    echo ""
    echo "This creates a tunnel from localhost:$PORT to the remote machine."
    echo ""

    # Extract SSH key from ProxyCommand if present
    SSH_KEY=$(echo "$PROXYCOMMAND" | grep -oE '\-i [^ ]+' | awk '{print $2}')

    # For ProxyCommand, MCP will connect through the tunnel on localhost
    FINAL_HOST="localhost"
    FINAL_PORT=$PORT

    echo "Configuration will use SSH tunnel:"
    echo "  Local tunnel: localhost:$PORT"
    echo "  Target: $USER@$HOSTNAME:$PORT (via tunnel)"
    if [ -n "$SSH_KEY" ]; then
        echo "  SSH Key: $SSH_KEY"
    fi
else
    echo "✓ Direct SSH configuration (no ProxyCommand)"
    echo "  User: $USER"
    echo "  Hostname: $HOSTNAME"
    echo "  Port: $PORT"
    FINAL_HOST=$HOSTNAME
    FINAL_PORT=$PORT
fi

echo ""

# Test connection
echo "Testing SSH connection to $SSH_ALIAS..."
if ssh "$SSH_ALIAS" -o ConnectTimeout=5 "echo 'Connection successful!'" &>/dev/null; then
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

SSH_HOST=$USER@$FINAL_HOST
SSH_PORT=$FINAL_PORT
EOF

# Add SSH key if detected
if [ -n "$SSH_KEY" ]; then
    cat >> "$ENV_FILE" <<EOF

# SSH key for cluster
SSH_KEY_PATH=$SSH_KEY
EOF
else
    cat >> "$ENV_FILE" <<EOF

# Optional: Uncomment and set if using a non-default SSH key
# SSH_KEY_PATH=~/.ssh/sky-key
EOF
fi

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

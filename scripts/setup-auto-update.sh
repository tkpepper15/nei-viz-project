#!/bin/bash

# Setup script for git-based auto-updates on Unraid
# Run this once on your Unraid server

set -e

# Configuration
PROJECT_DIR="/mnt/user/appdata/spideyplot-build"
CRON_USER="root"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
info() { echo -e "${BLUE}[SETUP] $1${NC}"; }

info "Setting up SpideyPlot auto-update system..."

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    warn "Project directory not found: $PROJECT_DIR"
    info "You need to first deploy the project to Unraid. Run:"
    echo "  1. Copy source code to Unraid"
    echo "  2. Build container locally"
    echo "  3. Then run this setup script"
    exit 1
fi

cd "$PROJECT_DIR"

# Make scripts executable
info "Making scripts executable..."
chmod +x scripts/*.sh

# Test the auto-update script
info "Testing auto-update script..."
if ./scripts/auto-update.sh; then
    log "✓ Auto-update script works correctly"
else
    warn "Auto-update script test failed"
fi

# Setup cron job for periodic updates (every 30 minutes)
info "Setting up cron job for automatic updates..."

# Remove existing cron job if it exists
crontab -l 2>/dev/null | grep -v "spideyplot-auto-update" | crontab -

# Add new cron job
(crontab -l 2>/dev/null; echo "# SpideyPlot auto-update - checks for git changes every 30 minutes") | crontab -
(crontab -l 2>/dev/null; echo "*/30 * * * * $PROJECT_DIR/scripts/auto-update.sh >> /var/log/spideyplot-update.log 2>&1 # spideyplot-auto-update") | crontab -

log "✓ Cron job added - will check for updates every 30 minutes"

# Create log directory
mkdir -p /var/log
touch /var/log/spideyplot-update.log

info "Setup complete!"
echo
log "Auto-update system is now active:"
echo "  • Checks for git changes every 30 minutes"
echo "  • Automatically rebuilds and restarts container when changes are found"
echo "  • Logs to: /var/log/spideyplot-update.log"
echo
info "Manual commands:"
echo "  • Manual update:    $PROJECT_DIR/scripts/auto-update.sh"
echo "  • View cron jobs:   crontab -l"
echo "  • View update log:  tail -f /var/log/spideyplot-update.log"
echo "  • Remove auto-update: crontab -l | grep -v spideyplot-auto-update | crontab -"
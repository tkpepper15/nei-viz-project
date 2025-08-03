#!/bin/bash

# Auto-update script for Unraid deployment
# This script pulls latest code and rebuilds the container

set -e

# Configuration
PROJECT_DIR="/mnt/user/appdata/spideyplot-build"
CONTAINER_NAME="spideyplot"
IMAGE_NAME="localhost/spideyplot:latest"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
warn() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }

log "Starting SpideyPlot auto-update process..."

# Check if project directory exists
if [ ! -d "$PROJECT_DIR" ]; then
    error "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"

# Check if it's a git repository
if [ ! -d ".git" ]; then
    error "Not a git repository: $PROJECT_DIR"
    exit 1
fi

# Pull latest code
log "Pulling latest code from git..."
git fetch origin
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse @{u})

if [ "$LOCAL" = "$REMOTE" ]; then
    log "Already up to date. No rebuild needed."
    exit 0
fi

log "New changes detected. Updating..."
git pull origin main

# Stop existing container
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    log "Stopping existing container..."
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
fi

# Remove old image
if docker images -q "$IMAGE_NAME" | grep -q .; then
    log "Removing old image..."
    docker rmi "$IMAGE_NAME"
fi

# Build new image
log "Building new Docker image..."
if ! ./scripts/docker-build.sh --latest; then
    error "Failed to build Docker image"
    exit 1
fi

# Start new container
log "Starting new container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /mnt/user/appdata/spideyplot/data:/app/data \
  -e NODE_ENV=production \
  -e NODE_OPTIONS="--max-old-space-size=8192" \
  "$IMAGE_NAME"

# Verify container is running
sleep 10
if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
    log "✓ SpideyPlot updated successfully!"
    log "✓ Container is running"
    log "✓ Access at: http://$(hostname -I | awk '{print $1}'):3000"
    
    # Test health endpoint
    if curl -f http://localhost:3000/api/health >/dev/null 2>&1; then
        log "✓ Health check passed"
    else
        warn "Health check failed - container may still be starting"
    fi
else
    error "Container failed to start"
    docker logs "$CONTAINER_NAME"
    exit 1
fi
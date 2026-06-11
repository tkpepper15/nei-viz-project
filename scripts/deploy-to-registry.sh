#!/bin/bash

# Universal deployment script for SpideyPlot
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"; }
success() { echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"; }
warning() { echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"; }
error() { echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"; }

# Default values
REGISTRY_TYPE=""
USERNAME=""
REPOSITORY="spideyplot"
VERSION="latest"
PUSH=false
BUILD=false

show_help() {
    echo "SpideyPlot Registry Deployment Script"
    echo
    echo "Usage: $0 --registry=TYPE [OPTIONS]"
    echo
    echo "Registry Types:"
    echo "  dockerhub    Deploy to Docker Hub"
    echo "  github       Deploy to GitHub Container Registry (ghcr.io)"
    echo "  local        Build locally only"
    echo
    echo "Options:"
    echo "  --username=USER    Your registry username"
    echo "  --repository=REPO  Repository name (default: spideyplot)"
    echo "  --version=VER      Image version (default: latest)"
    echo "  --build            Build image before pushing"
    echo "  --push             Push to registry after build"
    echo "  -h, --help         Show this help"
    echo
    echo "Examples:"
    echo "  $0 --registry=dockerhub --username=myuser --build --push"
    echo "  $0 --registry=github --username=myuser --build --push"
    echo "  $0 --registry=local --build"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --registry=*)
            REGISTRY_TYPE="${1#*=}"
            shift
            ;;
        --username=*)
            USERNAME="${1#*=}"
            shift
            ;;
        --repository=*)
            REPOSITORY="${1#*=}"
            shift
            ;;
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --push)
            PUSH=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate inputs
if [ -z "$REGISTRY_TYPE" ]; then
    error "Registry type is required. Use --registry=dockerhub|github|local"
    show_help
    exit 1
fi

if [ "$REGISTRY_TYPE" != "local" ] && [ -z "$USERNAME" ]; then
    error "Username is required for $REGISTRY_TYPE registry"
    exit 1
fi

# Set registry-specific variables
case $REGISTRY_TYPE in
    dockerhub)
        REGISTRY_URL="docker.io"
        IMAGE_NAME="$USERNAME/$REPOSITORY"
        FULL_IMAGE="$IMAGE_NAME:$VERSION"
        ;;
    github)
        REGISTRY_URL="ghcr.io"
        IMAGE_NAME="$USERNAME/$REPOSITORY"
        FULL_IMAGE="ghcr.io/$IMAGE_NAME:$VERSION"
        ;;
    local)
        REGISTRY_URL="localhost"
        IMAGE_NAME="$REPOSITORY"
        FULL_IMAGE="localhost/$IMAGE_NAME:$VERSION"
        ;;
    *)
        error "Unknown registry type: $REGISTRY_TYPE"
        exit 1
        ;;
esac

log "Deployment Configuration:"
log "Registry: $REGISTRY_TYPE ($REGISTRY_URL)"
log "Image: $FULL_IMAGE"
log "Build: $BUILD"
log "Push: $PUSH"
echo

# Check prerequisites
if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH"
    exit 1
fi

if [ ! -f "Dockerfile" ]; then
    error "Dockerfile not found. Please run from project root."
    exit 1
fi

# Build image if requested
if [ "$BUILD" = true ]; then
    log "Building Docker image: $FULL_IMAGE"
    
    if docker build -t "$FULL_IMAGE" .; then
        success "Built image: $FULL_IMAGE"
    else
        error "Failed to build image"
        exit 1
    fi
    
    # Also tag as latest if not already latest
    if [ "$VERSION" != "latest" ]; then
        LATEST_IMAGE="${FULL_IMAGE%:*}:latest"
        docker tag "$FULL_IMAGE" "$LATEST_IMAGE"
        log "Also tagged as: $LATEST_IMAGE"
    fi
fi

# Push to registry if requested
if [ "$PUSH" = true ]; then
    if [ "$REGISTRY_TYPE" = "local" ]; then
        warning "Cannot push to local registry. Image is available locally."
    else
        log "Logging in to $REGISTRY_TYPE..."
        
        case $REGISTRY_TYPE in
            dockerhub)
                if ! docker login; then
                    error "Failed to login to Docker Hub"
                    exit 1
                fi
                ;;
            github)
                if [ -z "$GITHUB_TOKEN" ]; then
                    error "GITHUB_TOKEN environment variable required for GitHub registry"
                    error "Generate token at: https://github.com/settings/tokens"
                    error "Then run: export GITHUB_TOKEN=your_token"
                    exit 1
                fi
                echo "$GITHUB_TOKEN" | docker login ghcr.io -u "$USERNAME" --password-stdin
                ;;
        esac
        
        log "Pushing image: $FULL_IMAGE"
        if docker push "$FULL_IMAGE"; then
            success "Pushed: $FULL_IMAGE"
        else
            error "Failed to push image"
            exit 1
        fi
        
        # Push latest tag if different
        if [ "$VERSION" != "latest" ]; then
            LATEST_IMAGE="${FULL_IMAGE%:*}:latest"
            if docker push "$LATEST_IMAGE"; then
                success "Pushed: $LATEST_IMAGE"
            fi
        fi
    fi
fi

# Show deployment instructions
echo
success "Deployment preparation complete!"
echo
log "Next steps for Unraid deployment:"
echo

case $REGISTRY_TYPE in
    dockerhub)
        echo "1. SSH into your Unraid server"
        echo "2. Run this command:"
        echo "   docker run -d \\"
        echo "     --name spideyplot \\"
        echo "     --restart unless-stopped \\"
        echo "     -p 3000:3000 \\"
        echo "     -v /mnt/user/appdata/spideyplot/data:/app/data \\"
        echo "     -e NODE_ENV=production \\"
        echo "     -e NODE_OPTIONS='--max-old-space-size=8192' \\"
        echo "     $FULL_IMAGE"
        ;;
    github)
        echo "1. SSH into your Unraid server"
        echo "2. Run this command:"
        echo "   docker run -d \\"
        echo "     --name spideyplot \\"
        echo "     --restart unless-stopped \\"
        echo "     -p 3000:3000 \\"
        echo "     -v /mnt/user/appdata/spideyplot/data:/app/data \\"
        echo "     -e NODE_ENV=production \\"
        echo "     -e NODE_OPTIONS='--max-old-space-size=8192' \\"
        echo "     $FULL_IMAGE"
        ;;
    local)
        echo "1. Transfer the image to your Unraid server:"
        echo "   docker save $FULL_IMAGE | gzip > spideyplot.tar.gz"
        echo "   scp spideyplot.tar.gz root@YOUR_UNRAID_IP:/tmp/"
        echo
        echo "2. SSH into Unraid and load the image:"
        echo "   ssh root@YOUR_UNRAID_IP"
        echo "   docker load < /tmp/spideyplot.tar.gz"
        echo
        echo "3. Run the container:"
        echo "   docker run -d \\"
        echo "     --name spideyplot \\"
        echo "     --restart unless-stopped \\"
        echo "     -p 3000:3000 \\"
        echo "     -v /mnt/user/appdata/spideyplot/data:/app/data \\"
        echo "     -e NODE_ENV=production \\"
        echo "     -e NODE_OPTIONS='--max-old-space-size=8192' \\"
        echo "     $FULL_IMAGE"
        ;;
esac

echo
echo "3. Access SpideyPlot at: http://YOUR_UNRAID_IP:3000"
echo "4. Check health: http://YOUR_UNRAID_IP:3000/api/health"
echo
log "For automated updates, set up Watchtower (see README-Docker.md)"
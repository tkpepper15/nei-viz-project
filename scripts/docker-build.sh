#!/bin/bash

# Docker build script for SpideyPlot application
set -e

# Configuration
APP_NAME="spideyplot"
VERSION=${VERSION:-$(date +%Y%m%d-%H%M%S)}
REGISTRY=${REGISTRY:-"localhost"}
BUILD_CONTEXT="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] ✓ $1${NC}"
}

warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] ⚠ $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ✗ $1${NC}"
}

# Parse command line arguments
PUSH=false
LATEST=false
PLATFORM="linux/amd64"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --latest)
            LATEST=true
            shift
            ;;
        --platform=*)
            PLATFORM="${1#*=}"
            shift
            ;;
        --registry=*)
            REGISTRY="${1#*=}"
            shift
            ;;
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --push              Push image to registry after build"
            echo "  --latest            Tag image as latest"
            echo "  --platform=PLATFORM Build platform (default: linux/amd64)"
            echo "  --registry=REGISTRY Registry URL (default: localhost)"
            echo "  --version=VERSION   Image version (default: timestamp)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Image tags
IMAGE_TAG="${REGISTRY}/${APP_NAME}:${VERSION}"
LATEST_TAG="${REGISTRY}/${APP_NAME}:latest"

log "Starting Docker build for SpideyPlot"
log "Image tag: ${IMAGE_TAG}"
log "Platform: ${PLATFORM}"

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    error "package.json not found. Please run this script from the project root."
    exit 1
fi

# Check if Dockerfile exists
if [ ! -f "Dockerfile" ]; then
    error "Dockerfile not found. Please ensure Dockerfile exists in the project root."
    exit 1
fi

# Build the Docker image
log "Building Docker image..."
if docker build \
    --platform "${PLATFORM}" \
    --tag "${IMAGE_TAG}" \
    --build-arg VERSION="${VERSION}" \
    --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    "${BUILD_CONTEXT}"; then
    success "Docker image built successfully: ${IMAGE_TAG}"
else
    error "Docker build failed"
    exit 1
fi

# Tag as latest if requested
if [ "$LATEST" = true ]; then
    log "Tagging as latest..."
    if docker tag "${IMAGE_TAG}" "${LATEST_TAG}"; then
        success "Tagged as latest: ${LATEST_TAG}"
    else
        error "Failed to tag as latest"
        exit 1
    fi
fi

# Push to registry if requested
if [ "$PUSH" = true ]; then
    log "Pushing to registry..."
    
    if docker push "${IMAGE_TAG}"; then
        success "Pushed ${IMAGE_TAG}"
    else
        error "Failed to push ${IMAGE_TAG}"
        exit 1
    fi
    
    if [ "$LATEST" = true ]; then
        if docker push "${LATEST_TAG}"; then
            success "Pushed ${LATEST_TAG}"
        else
            error "Failed to push ${LATEST_TAG}"
            exit 1
        fi
    fi
fi

# Show image information
log "Image information:"
docker images "${REGISTRY}/${APP_NAME}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

success "Build completed successfully!"

# Show next steps
echo
log "Next steps:"
echo "  1. Test the image locally:"
echo "     docker run --rm -p 3000:3000 ${IMAGE_TAG}"
echo
echo "  2. Deploy with docker-compose:"
echo "     docker-compose up -d"
echo
echo "  3. Check health status:"
echo "     curl http://localhost:3000/api/health"
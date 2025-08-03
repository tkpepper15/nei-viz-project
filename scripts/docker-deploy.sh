#!/bin/bash

# Docker deployment script for Unraid/Docker environments
set -e

# Configuration
APP_NAME="spideyplot"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env.local"

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
ACTION="up"
DETACHED=true
BUILD=false
FORCE_RECREATE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        up|down|restart|logs|status)
            ACTION="$1"
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        --force-recreate)
            FORCE_RECREATE=true
            shift
            ;;
        --no-detach)
            DETACHED=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [ACTION] [OPTIONS]"
            echo "Actions:"
            echo "  up        Start the application (default)"
            echo "  down      Stop the application"
            echo "  restart   Restart the application"
            echo "  logs      Show application logs"
            echo "  status    Show container status"
            echo "Options:"
            echo "  --build           Build images before starting"
            echo "  --force-recreate  Force recreate containers"
            echo "  --no-detach       Don't run in background (for logs)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    error "Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    error "docker-compose is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "${COMPOSE_FILE}" ]; then
    error "${COMPOSE_FILE} not found. Please run this script from the project root."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f "${ENV_FILE}" ]; then
    warning "${ENV_FILE} not found. Creating from template..."
    if [ -f ".env.docker" ]; then
        cp .env.docker "${ENV_FILE}"
        success "Created ${ENV_FILE} from template"
        warning "Please review and customize ${ENV_FILE} before deployment"
    else
        error "No environment template found. Please create ${ENV_FILE}"
        exit 1
    fi
fi

# Create data directory if it doesn't exist
DATA_DIR="${DATA_DIR:-./data}"
if [ ! -d "${DATA_DIR}" ]; then
    log "Creating data directory: ${DATA_DIR}"
    mkdir -p "${DATA_DIR}"
    success "Created data directory"
fi

# Function to handle different actions
case $ACTION in
    up)
        log "Starting SpideyPlot application..."
        
        DOCKER_ARGS=""
        if [ "$DETACHED" = true ]; then
            DOCKER_ARGS="$DOCKER_ARGS -d"
        fi
        
        if [ "$BUILD" = true ]; then
            DOCKER_ARGS="$DOCKER_ARGS --build"
        fi
        
        if [ "$FORCE_RECREATE" = true ]; then
            DOCKER_ARGS="$DOCKER_ARGS --force-recreate"
        fi
        
        if docker-compose --env-file "${ENV_FILE}" up $DOCKER_ARGS; then
            success "SpideyPlot started successfully"
            
            if [ "$DETACHED" = true ]; then
                echo
                log "Application is running at: http://localhost:${PORT:-3000}"
                log "Health check: http://localhost:${PORT:-3000}/api/health"
                echo
                log "To view logs: $0 logs"
                log "To stop: $0 down"
            fi
        else
            error "Failed to start SpideyPlot"
            exit 1
        fi
        ;;
        
    down)
        log "Stopping SpideyPlot application..."
        if docker-compose --env-file "${ENV_FILE}" down; then
            success "SpideyPlot stopped successfully"
        else
            error "Failed to stop SpideyPlot"
            exit 1
        fi
        ;;
        
    restart)
        log "Restarting SpideyPlot application..."
        if docker-compose --env-file "${ENV_FILE}" restart; then
            success "SpideyPlot restarted successfully"
        else
            error "Failed to restart SpideyPlot"
            exit 1
        fi
        ;;
        
    logs)
        log "Showing SpideyPlot logs..."
        docker-compose --env-file "${ENV_FILE}" logs -f
        ;;
        
    status)
        log "SpideyPlot container status:"
        docker-compose --env-file "${ENV_FILE}" ps
        echo
        log "Docker images:"
        docker images spideyplot* --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        echo
        log "Volume usage:"
        docker volume ls | grep spideyplot || echo "No SpideyPlot volumes found"
        ;;
        
    *)
        error "Unknown action: $ACTION"
        exit 1
        ;;
esac
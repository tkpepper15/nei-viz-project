# Docker Hub Deployment Guide

## Prerequisites

1. **Docker Hub Account**: Create account at https://hub.docker.com
2. **Docker Desktop**: Ensure Docker is running locally
3. **Repository**: Create repository `your-username/spideyplot` on Docker Hub

## Step 1: Build and Tag for Docker Hub

```bash
# Build with your Docker Hub username
./scripts/docker-build.sh --registry=your-username --latest

# Or build manually
docker build -t your-username/spideyplot:latest .
docker build -t your-username/spideyplot:v1.0.0 .
```

## Step 2: Login and Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Push images
docker push your-username/spideyplot:latest
docker push your-username/spideyplot:v1.0.0
```

## Step 3: Deploy from Docker Hub on Unraid

### Option A: Using Unraid Template

Update the Unraid template:

```xml
<?xml version="1.0"?>
<Container version="2">
  <Name>SpideyPlot</Name>
  <Repository>your-username/spideyplot:latest</Repository>
  <Registry>https://hub.docker.com/</Registry>
  <!-- rest of template... -->
</Container>
```

### Option B: Manual Container Creation

In Unraid WebUI:
1. Go to Docker tab
2. Click "Add Container"
3. **Repository**: `your-username/spideyplot:latest`
4. **Name**: `SpideyPlot`
5. **Network Type**: `bridge`
6. **Port**: `3000` → `3000`
7. **Path**: `/app/data` → `/mnt/user/appdata/spideyplot/data`
8. **Variable**: `NODE_OPTIONS` = `--max-old-space-size=8192`

### Option C: Command Line on Unraid

```bash
# SSH into Unraid
ssh root@YOUR_UNRAID_IP

# Pull and run
docker pull your-username/spideyplot:latest

docker run -d \
  --name spideyplot \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /mnt/user/appdata/spideyplot/data:/app/data \
  -e NODE_ENV=production \
  -e NODE_OPTIONS="--max-old-space-size=8192" \
  your-username/spideyplot:latest
```

## Step 4: Automated Updates

### Using Watchtower

```bash
# Install Watchtower on Unraid
docker run -d \
  --name watchtower \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --cleanup \
  --interval 3600 \
  spideyplot
```

### Manual Updates

```bash
# Pull latest
docker pull your-username/spideyplot:latest

# Stop and remove old container
docker stop spideyplot
docker rm spideyplot

# Start new container
docker run -d \
  --name spideyplot \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /mnt/user/appdata/spideyplot/data:/app/data \
  -e NODE_ENV=production \
  -e NODE_OPTIONS="--max-old-space-size=8192" \
  your-username/spideyplot:latest
```
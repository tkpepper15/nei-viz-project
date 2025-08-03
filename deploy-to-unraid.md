# Direct Deployment to Unraid Guide

## Method 1: Build Directly on Unraid (Easiest)

### Step 1: Transfer Files to Unraid

```bash
# On your Mac, compress the project
tar -czf spideyplot-source.tar.gz --exclude=node_modules --exclude=.git --exclude=.next .

# Copy to Unraid (replace with your Unraid IP)
scp spideyplot-source.tar.gz root@192.168.1.100:/mnt/user/appdata/

# SSH into Unraid
ssh root@192.168.1.100

# Extract on Unraid
cd /mnt/user/appdata/
tar -xzf spideyplot-source.tar.gz
mv nei-viz-project spideyplot-build
cd spideyplot-build
```

### Step 2: Build on Unraid

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Build the Docker image
./scripts/docker-build.sh --latest

# This creates: localhost/spideyplot:latest
```

### Step 3: Deploy on Unraid

```bash
# Set up environment
cp .env.docker .env.local

# Create data directory
mkdir -p /mnt/user/appdata/spideyplot/data

# Run the container
docker run -d \
  --name spideyplot \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /mnt/user/appdata/spideyplot/data:/app/data \
  -e NODE_ENV=production \
  -e NODE_OPTIONS="--max-old-space-size=8192" \
  localhost/spideyplot:latest
```

### Step 4: Verify Deployment

```bash
# Check if running
docker ps | grep spideyplot

# Check health
curl http://localhost:3000/api/health

# View logs
docker logs spideyplot -f
```

### Step 5: Add to Unraid UI (Optional)

Create a custom container in Unraid WebUI:
- **Name**: SpideyPlot
- **Repository**: localhost/spideyplot:latest
- **Network Type**: Bridge
- **Port**: 3000 → 3000
- **Path**: /app/data → /mnt/user/appdata/spideyplot/data

## Method 2: Using Docker Compose on Unraid

```bash
# On Unraid, in the project directory
cd /mnt/user/appdata/spideyplot-build

# Start with compose
./scripts/docker-deploy.sh up

# Or manually
docker-compose up -d
```
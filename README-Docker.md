# SpideyPlot Docker Deployment Guide

This guide explains how to deploy SpideyPlot using Docker on various platforms, including Unraid.

## Quick Start

### 1. Build and Run Locally

```bash
# Build the Docker image
./scripts/docker-build.sh

# Start the application
./scripts/docker-deploy.sh up

# View logs
./scripts/docker-deploy.sh logs

# Stop the application
./scripts/docker-deploy.sh down
```

### 2. Using Docker Compose

```bash
# Copy environment template
cp .env.docker .env.local

# Edit configuration (optional)
nano .env.local

# Start with docker-compose
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### 3. Simple Docker Run

```bash
# Basic run command
docker run -d \
  --name spideyplot \
  --restart unless-stopped \
  -p 3000:3000 \
  -v $(pwd)/data:/app/data \
  spideyplot:latest

# With custom settings
docker run -d \
  --name spideyplot \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /mnt/user/appdata/spideyplot/data:/app/data \
  -v /mnt/user/appdata/spideyplot/config:/app/config \
  -e NODE_ENV=production \
  -e LOG_LEVEL=info \
  spideyplot:latest
```

## Unraid Deployment

### Method 1: Using Unraid Template

1. **Add Template Repository**:
   - Go to Docker tab in Unraid
   - Click "Add Container"
   - Set Template repositories to include your repository URL

2. **Install from Template**:
   - Search for "SpideyPlot"
   - Click "Install"
   - Configure paths and ports as needed
   - Click "Apply"

### Method 2: Manual Configuration

1. **Create Container**:
   - Docker tab → Add Container
   - Name: `SpideyPlot`
   - Repository: `spideyplot:latest`
   - Network Type: `Bridge`

2. **Configure Ports**:
   - Container Port: `3000`
   - Host Port: `3000` (or your preferred port)
   - Connection Type: `TCP`

3. **Configure Volumes**:
   ```
   Container Path: /app/data
   Host Path: /mnt/user/appdata/spideyplot/data
   Access Mode: Read/Write
   
   Container Path: /app/config
   Host Path: /mnt/user/appdata/spideyplot/config
   Access Mode: Read/Write
   ```

4. **Environment Variables**:
   ```
   NODE_ENV=production
   NEXT_TELEMETRY_DISABLED=1
   NODE_OPTIONS=--max-old-space-size=8192
   LOG_LEVEL=info
   ```

5. **Apply Configuration**

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ENV` | `production` | Node.js environment |
| `PORT` | `3000` | Application port |
| `HOSTNAME` | `0.0.0.0` | Bind hostname |
| `DATA_PATH` | `/app/data` | Data storage path |
| `LOG_LEVEL` | `info` | Logging level |
| `NODE_OPTIONS` | `--max-old-space-size=8192` | Node.js memory settings |
| `NEXT_TELEMETRY_DISABLED` | `1` | Disable Next.js telemetry |

### Volume Mounts

| Container Path | Description | Required |
|----------------|-------------|----------|
| `/app/data` | Persistent data (profiles, results) | Yes |
| `/app/config` | Configuration files | No |
| `/app/logs` | Application logs | No |

### Ports

| Container Port | Description |
|----------------|-------------|
| `3000` | Web interface |

## Health Monitoring

### Health Check Endpoint

The container includes a built-in health check endpoint:

```bash
# Check application health
curl http://localhost:3000/api/health

# Example response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "version": "1.0.0",
  "uptime": 3600,
  "memory": {
    "used": 512,
    "total": 1024,
    "limit": 8192
  },
  "checks": {
    "server": "ok",
    "memory": true
  }
}
```

### Docker Health Check

The Dockerfile includes automatic health checking:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:3000/api/health || exit 1
```

### Monitoring Commands

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# View container logs
docker logs spideyplot -f

# Check resource usage
docker stats spideyplot

# Inspect container
docker inspect spideyplot
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```bash
   # Find what's using the port
   lsof -i :3000
   
   # Use different port
   docker run -p 3001:3000 spideyplot:latest
   ```

2. **Memory Issues**:
   ```bash
   # Increase memory limit
   docker run -e NODE_OPTIONS="--max-old-space-size=16384" spideyplot:latest
   ```

3. **Permission Issues**:
   ```bash
   # Fix data directory permissions
   sudo chown -R 1001:1001 /mnt/user/appdata/spideyplot/data
   ```

4. **Container Won't Start**:
   ```bash
   # Check logs
   docker logs spideyplot
   
   # Run in interactive mode
   docker run -it --rm spideyplot:latest sh
   ```

### Performance Tuning

1. **CPU Limits**:
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '4.0'
         memory: 8G
   ```

2. **Memory Settings**:
   ```bash
   # For large computations
   NODE_OPTIONS="--max-old-space-size=16384"
   ```

3. **Storage Performance**:
   - Use SSD storage for `/app/data`
   - Consider memory-mapped storage for large datasets

## Backup and Restore

### Backup Data

```bash
# Create backup
docker exec spideyplot tar -czf /tmp/backup.tar.gz /app/data
docker cp spideyplot:/tmp/backup.tar.gz ./spideyplot-backup-$(date +%Y%m%d).tar.gz

# Or backup host directory
tar -czf spideyplot-backup-$(date +%Y%m%d).tar.gz /mnt/user/appdata/spideyplot/
```

### Restore Data

```bash
# Stop container
docker stop spideyplot

# Restore data
tar -xzf spideyplot-backup-20240115.tar.gz -C /

# Start container
docker start spideyplot
```

## Updates

### Update Container

```bash
# Pull latest image
docker pull spideyplot:latest

# Stop and remove old container
docker stop spideyplot
docker rm spideyplot

# Start new container
./scripts/docker-deploy.sh up
```

### Using Watchtower (Unraid)

```yaml
# Add to docker-compose.yml
watchtower:
  image: containrrr/watchtower
  volumes:
    - /var/run/docker.sock:/var/run/docker.sock
  environment:
    - WATCHTOWER_CLEANUP=true
    - WATCHTOWER_POLL_INTERVAL=3600
```

## Security Considerations

1. **Network Security**:
   - Use reverse proxy (Nginx Proxy Manager, Traefik)
   - Enable HTTPS
   - Restrict access with firewall rules

2. **Data Security**:
   - Regular backups
   - Encrypted storage volumes
   - Access logging

3. **Container Security**:
   - Run as non-root user
   - Regular security updates
   - Resource limits

## Advanced Configuration

### Reverse Proxy Setup

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name spideyplot.local;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Custom Build

```bash
# Build with custom options
docker build \
  --build-arg VERSION=1.0.0 \
  --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
  -t spideyplot:custom \
  .
```

### Multi-stage Deployment

For production environments, consider a multi-container setup:

```yaml
# Future: Full production setup
version: '3.8'
services:
  web:
    image: spideyplot:latest
    depends_on: [redis, postgres]
  
  worker:
    image: spideyplot-worker:latest
    depends_on: [redis, postgres]
  
  redis:
    image: redis:7-alpine
  
  postgres:
    image: postgres:15-alpine
```

## Support

- **Documentation**: Check the main README.md
- **Issues**: Create GitHub issues for bugs/features
- **Health Check**: `http://localhost:3000/api/health`
- **Logs**: `docker logs spideyplot -f`
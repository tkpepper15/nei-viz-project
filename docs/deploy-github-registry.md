# GitHub Container Registry Deployment

## Advantages
- ✅ Free for public repositories
- ✅ Automatic builds on code changes
- ✅ Multi-architecture support (ARM64 + AMD64)
- ✅ Integrated with your GitHub repo
- ✅ Automatic testing

## Step 1: Enable GitHub Actions

1. **Push your code to GitHub** (if not already there)
2. **GitHub Actions will automatically build** when you push to main branch
3. **Images are available at**: `ghcr.io/your-username/nei-viz-project:latest`

## Step 2: Deploy on Unraid

### Using the built image from GitHub Registry

```bash
# SSH into Unraid
ssh root@YOUR_UNRAID_IP

# Pull from GitHub Registry (no login required for public repos)
docker pull ghcr.io/your-username/nei-viz-project:latest

# Run container
docker run -d \
  --name spideyplot \
  --restart unless-stopped \
  -p 3000:3000 \
  -v /mnt/user/appdata/spideyplot/data:/app/data \
  -e NODE_ENV=production \
  -e NODE_OPTIONS="--max-old-space-size=8192" \
  ghcr.io/your-username/nei-viz-project:latest
```

### Update Unraid Template

```xml
<Repository>ghcr.io/your-username/nei-viz-project:latest</Repository>
<Registry>https://ghcr.io/</Registry>
```

## Step 3: Automatic Updates

Every time you push code to GitHub:
1. **GitHub Actions builds** new image
2. **Watchtower pulls** latest image (if configured)
3. **Container restarts** with new code

### Set up Watchtower for auto-updates

```bash
docker run -d \
  --name watchtower \
  --restart unless-stopped \
  -v /var/run/docker.sock:/var/run/docker.sock \
  containrrr/watchtower \
  --cleanup \
  --interval 3600 \
  spideyplot
```
// Tile Rendering Web Worker
// Handles rendering of individual spider plot tiles for efficient parallel processing

class TileSpiderRenderer {
  constructor() {
    this.isRendering = false;
    this.cancelRequested = false;
  }

  // Main tile rendering function
  async renderTile(job, renderConfig) {
    const startTime = performance.now();
    const memoryBefore = this.getMemoryUsage();
    
    try {
      // Create offscreen canvas for this tile
      const canvas = new OffscreenCanvas(job.width, job.height);
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        throw new Error('Could not get canvas context');
      }

      // Set high-quality rendering
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Clear with dark background
      ctx.fillStyle = renderConfig.backgroundColor || '#0f172a';
      ctx.fillRect(0, 0, job.width, job.height);

      // Calculate scale factors for this tile
      const scaleX = job.width / job.viewBox.width;
      const scaleY = job.height / job.viewBox.height;

      // Transform coordinate system to match the viewBox
      ctx.save();
      ctx.scale(scaleX, scaleY);
      ctx.translate(-job.viewBox.x, -job.viewBox.y);

      // Render spider plot grid for this tile section
      this.renderSpiderGrid(ctx, job.viewBox, renderConfig);

      // Render models that intersect with this tile
      for (const model of job.models) {
        if (this.cancelRequested) break;
        this.renderSpiderPolygon(ctx, model, renderConfig);
      }

      ctx.restore();

      // Get image data
      const imageData = ctx.getImageData(0, 0, job.width, job.height);
      
      const renderTime = performance.now() - startTime;

      return {
        tileId: job.tileId,
        canvas,
        imageData,
        renderTime
      };

    } catch (error) {
      throw new Error(`Tile rendering failed: ${error.message}`);
    }
  }

  renderSpiderGrid(ctx, viewBox, config) {
    const centerX = config.spiderConfig.viewBoxSize / 2;
    const centerY = config.spiderConfig.viewBoxSize / 2;
    const maxRadius = config.spiderConfig.viewBoxSize / 2 - 50;

    // Only render grid elements that are visible in this viewBox
    const gridVisible = this.isGridVisible(viewBox, centerX, centerY, maxRadius);
    if (!gridVisible) return;

    ctx.strokeStyle = config.gridColor || '#4B5563';
    ctx.lineWidth = config.gridLineWidth || 0.5;

    // Draw concentric circles
    for (let i = 1; i <= 5; i++) {
      const radius = (maxRadius * i) / 5;
      if (this.isCircleInViewBox(centerX, centerY, radius, viewBox)) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }

    // Draw radial axes (5 axes for 5 parameters)
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    for (let i = 0; i < params.length; i++) {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      const endX = centerX + Math.cos(angle) * maxRadius;
      const endY = centerY + Math.sin(angle) * maxRadius;

      if (this.isLineInViewBox(centerX, centerY, endX, endY, viewBox)) {
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      }
    }
  }

  renderSpiderPolygon(ctx, model, config) {
    if (!model.parameters) return;

    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    const centerX = config.spiderConfig.viewBoxSize / 2;
    const centerY = config.spiderConfig.viewBoxSize / 2;
    const maxRadius = config.spiderConfig.viewBoxSize / 2 - 50;

    // Calculate polygon points
    const points = [];
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const value = model.parameters[param];
      
      if (typeof value !== 'number') continue;

      // Normalize parameter value
      let normalizedValue;
      if (param.includes('C')) {
        // Capacitance: 0.1µF to 50µF
        normalizedValue = Math.log10(value * 1e6 / 0.1) / Math.log10(50 / 0.1);
      } else {
        // Resistance: 10Ω to 10kΩ
        normalizedValue = Math.log10(value / 10) / Math.log10(10000 / 10);
      }

      normalizedValue = Math.max(0, Math.min(1, normalizedValue));
      const radius = normalizedValue * maxRadius;
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;

      points.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      });
    }

    if (points.length < 3) return; // Need at least 3 points for a polygon

    // Set rendering properties
    ctx.strokeStyle = model.color || config.defaultColor || '#3B82F6';
    ctx.lineWidth = model.strokeWidth || config.defaultStrokeWidth || 1;
    ctx.globalAlpha = model.opacity || config.defaultOpacity || 0.7;

    // Draw the polygon
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    // Optionally fill the polygon for reference models
    if (model.isReference) {
      ctx.fillStyle = model.color || config.defaultColor || '#3B82F6';
      ctx.globalAlpha = (model.opacity || 0.7) * 0.3; // Lighter fill
      ctx.fill();
    }

    ctx.globalAlpha = 1; // Reset alpha
  }

  // Utility functions for culling
  isGridVisible(viewBox, centerX, centerY, maxRadius) {
    // Check if the grid center or any part of the maximum radius circle is visible
    return this.isPointInViewBox(centerX, centerY, viewBox) ||
           this.isCircleInViewBox(centerX, centerY, maxRadius, viewBox);
  }

  isPointInViewBox(x, y, viewBox) {
    return x >= viewBox.x && x <= viewBox.x + viewBox.width &&
           y >= viewBox.y && y <= viewBox.y + viewBox.height;
  }

  isCircleInViewBox(centerX, centerY, radius, viewBox) {
    // Check if circle intersects with viewbox
    const distX = Math.abs(centerX - (viewBox.x + viewBox.width / 2));
    const distY = Math.abs(centerY - (viewBox.y + viewBox.height / 2));

    if (distX > (viewBox.width / 2 + radius)) return false;
    if (distY > (viewBox.height / 2 + radius)) return false;

    if (distX <= (viewBox.width / 2)) return true;
    if (distY <= (viewBox.height / 2)) return true;

    const cornerDistSq = Math.pow(distX - viewBox.width / 2, 2) + 
                        Math.pow(distY - viewBox.height / 2, 2);

    return cornerDistSq <= (radius * radius);
  }

  isLineInViewBox(x1, y1, x2, y2, viewBox) {
    // Simple line-rectangle intersection test
    // Check if either endpoint is in viewbox
    if (this.isPointInViewBox(x1, y1, viewBox) || 
        this.isPointInViewBox(x2, y2, viewBox)) {
      return true;
    }

    // Check if line intersects viewbox boundaries
    const left = viewBox.x;
    const right = viewBox.x + viewBox.width;
    const top = viewBox.y;
    const bottom = viewBox.y + viewBox.height;

    return this.lineIntersectsRect(x1, y1, x2, y2, left, top, right, bottom);
  }

  lineIntersectsRect(x1, y1, x2, y2, left, top, right, bottom) {
    // Check intersection with each edge of the rectangle
    return this.lineIntersectsLine(x1, y1, x2, y2, left, top, right, top) ||    // Top
           this.lineIntersectsLine(x1, y1, x2, y2, right, top, right, bottom) || // Right
           this.lineIntersectsLine(x1, y1, x2, y2, right, bottom, left, bottom) || // Bottom
           this.lineIntersectsLine(x1, y1, x2, y2, left, bottom, left, top);    // Left
  }

  lineIntersectsLine(x1, y1, x2, y2, x3, y3, x4, y4) {
    const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (Math.abs(denom) < 1e-10) return false; // Lines are parallel

    const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

    return t >= 0 && t <= 1 && u >= 0 && u <= 1;
  }

  cancel() {
    this.cancelRequested = true;
  }

  reset() {
    this.cancelRequested = false;
  }
}

// Global renderer instance
const renderer = new TileSpiderRenderer();

// Message handler
self.onmessage = async function(e) {
  const { type, data } = e.data;

  try {
    switch (type) {
      case 'RENDER_TILES': {
        const { jobs, renderConfig } = data;
        
        renderer.reset();
        
        // Process each tile job
        for (const job of jobs) {
          if (renderer.cancelRequested) break;
          
          try {
            const result = await renderer.renderTile(job, renderConfig);
            
            // Send completion message
            self.postMessage({
              type: 'TILE_COMPLETE',
              data: result
            });
            
          } catch (error) {
            self.postMessage({
              type: 'TILE_ERROR',
              data: {
                tileId: job.tileId,
                error: error.message
              }
            });
          }
        }
        break;
      }

      case 'CANCEL_RENDER': {
        renderer.cancel();
        break;
      }

      default:
        console.warn('Unknown message type:', type);
    }
    
  } catch (error) {
    self.postMessage({
      type: 'WORKER_ERROR',
      data: {
        error: error.message,
        stack: error.stack
      }
    });
  }
}; 
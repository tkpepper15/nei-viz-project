// Enhanced Tile Rendering Web Worker with Performance Monitoring
// Handles rendering with detailed logging and memory management

class EnhancedTileRenderer {
  constructor() {
    this.isRendering = false;
    this.cancelRequested = false;
    this.workerIndex = 0;
    this.renderCount = 0;
    this.totalPolygonsRendered = 0;
    this.memoryCheckInterval = null;
    this.lastMemoryUsage = 0;
    
    // Performance tracking
    this.performanceMetrics = {
      totalRenderTime: 0,
      totalTiles: 0,
      averageRenderTime: 0,
      slowestTile: null,
      fastestTile: null,
      memoryLeaks: []
    };
  }

  // Enhanced memory monitoring
  getMemoryUsage() {
    if (typeof performance !== 'undefined' && 'memory' in performance) {
      return {
        used: performance.memory.usedJSHeapSize,
        total: performance.memory.totalJSHeapSize,
        limit: performance.memory.jsHeapSizeLimit,
        percentage: (performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize) * 100
      };
    }
    return { used: 0, total: 0, limit: 0, percentage: 0 };
  }

  startMemoryMonitoring() {
    if (this.memoryCheckInterval) return;
    
    this.memoryCheckInterval = setInterval(() => {
      const memory = this.getMemoryUsage();
      
      // Detect memory leaks
      if (memory.used > this.lastMemoryUsage * 1.2) {
        this.performanceMetrics.memoryLeaks.push({
          timestamp: Date.now(),
          before: this.lastMemoryUsage,
          after: memory.used,
          increase: memory.used - this.lastMemoryUsage
        });
        
        console.warn(`⚠️ [WORKER-${this.workerIndex}] Memory increase detected: ${((memory.used - this.lastMemoryUsage) / 1024 / 1024).toFixed(1)}MB`);
      }
      
      // Critical memory warning
      if (memory.percentage > 90) {
        console.error(`🚨 [WORKER-${this.workerIndex}] Critical memory usage: ${memory.percentage.toFixed(1)}%`);
        this.requestGarbageCollection();
      }
      
      this.lastMemoryUsage = memory.used;
    }, 2000);
  }

  stopMemoryMonitoring() {
    if (this.memoryCheckInterval) {
      clearInterval(this.memoryCheckInterval);
      this.memoryCheckInterval = null;
    }
  }

  requestGarbageCollection() {
    if (typeof gc !== 'undefined') {
      gc();
      console.log(`🗑️ [WORKER-${this.workerIndex}] Garbage collection triggered`);
    }
  }

  // Main enhanced tile rendering function
  async renderTile(job, renderConfig) {
    const renderStart = performance.now();
    const memoryBefore = this.getMemoryUsage();
    
    console.log(`🔄 [WORKER-${this.workerIndex}] Starting tile ${job.tileId} | Models: ${job.models.length} | Memory: ${memoryBefore.percentage.toFixed(1)}%`);
    
    try {
      // Phase 1: Canvas Setup (logged)
      const canvasSetupStart = performance.now();
      const canvas = new OffscreenCanvas(job.width, job.height);
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        throw new Error('Could not get canvas context');
      }

      // High-quality rendering settings
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      
      const canvasSetupTime = performance.now() - canvasSetupStart;
      console.log(`📐 [WORKER-${this.workerIndex}] Canvas setup: ${canvasSetupTime.toFixed(1)}ms`);

      // Phase 2: Background rendering
      const backgroundStart = performance.now();
      ctx.fillStyle = renderConfig.backgroundColor || '#0f172a';
      ctx.fillRect(0, 0, job.width, job.height);
      const backgroundTime = performance.now() - backgroundStart;

      // Phase 3: Coordinate transformation
      const transformStart = performance.now();
      const scaleX = job.width / job.viewBox.width;
      const scaleY = job.height / job.viewBox.height;
      
      ctx.save();
      ctx.scale(scaleX, scaleY);
      ctx.translate(-job.viewBox.x, -job.viewBox.y);
      const transformTime = performance.now() - transformStart;

      // Phase 4: Grid rendering (logged)
      const gridStart = performance.now();
      const gridElementsRendered = this.renderSpiderGridWithLogging(ctx, job.viewBox, renderConfig);
      const gridTime = performance.now() - gridStart;

      // Phase 5: Polygon rendering (logged and batched)
      const polygonStart = performance.now();
      const polygonsRendered = await this.renderPolygonsBatched(ctx, job.models, renderConfig, job.tileId);
      const polygonTime = performance.now() - polygonStart;

      ctx.restore();

      // Phase 6: Final operations
      const finalizationStart = performance.now();
      const imageData = ctx.getImageData(0, 0, job.width, job.height);
      const finalizationTime = performance.now() - finalizationStart;

      // Calculate final metrics
      const totalRenderTime = performance.now() - renderStart;
      const memoryAfter = this.getMemoryUsage();
      const memoryDelta = memoryAfter.used - memoryBefore.used;

      // Update performance tracking
      this.updatePerformanceMetrics(job.tileId, totalRenderTime, job.models.length);
      this.renderCount++;
      this.totalPolygonsRendered += job.models.length;

      // Detailed logging
      const metrics = {
        tileId: job.tileId,
        renderTime: totalRenderTime,
        polygonCount: job.models.length,
        memoryBefore: memoryBefore.used,
        memoryAfter: memoryAfter.used,
        memoryDelta,
        canvasSize: job.width * job.height,
        workerIndex: this.workerIndex,
        breakdown: {
          canvasSetup: canvasSetupTime,
          background: backgroundTime,
          transform: transformTime,
          grid: gridTime,
          polygons: polygonTime,
          finalization: finalizationTime
        },
        gridElementsRendered,
        polygonsRendered
      };

      console.log(`✅ [WORKER-${this.workerIndex}] Tile ${job.tileId} complete:
├─ Total: ${totalRenderTime.toFixed(1)}ms
├─ Polygons: ${polygonsRendered}/${job.models.length}
├─ Memory Δ: ${(memoryDelta / 1024 / 1024).toFixed(1)}MB
├─ Efficiency: ${(polygonsRendered / totalRenderTime * 1000).toFixed(0)} polygons/sec
└─ Breakdown: Setup(${canvasSetupTime.toFixed(1)}ms) | Grid(${gridTime.toFixed(1)}ms) | Polygons(${polygonTime.toFixed(1)}ms)`);

      return {
        tileId: job.tileId,
        canvas,
        imageData,
        renderTime: totalRenderTime,
        metrics
      };

    } catch (error) {
      const errorTime = performance.now() - renderStart;
      console.error(`❌ [WORKER-${this.workerIndex}] Tile ${job.tileId} failed after ${errorTime.toFixed(1)}ms:`, error);
      throw new Error(`Tile rendering failed: ${error.message}`);
    }
  }

  renderSpiderGridWithLogging(ctx, viewBox, config) {
    const gridStart = performance.now();
    const centerX = config.spiderConfig.viewBoxSize / 2;
    const centerY = config.spiderConfig.viewBoxSize / 2;
    const maxRadius = config.spiderConfig.viewBoxSize / 2 - 50;

    let elementsRendered = 0;

    // Check if grid is visible
    const gridVisible = this.isGridVisible(viewBox, centerX, centerY, maxRadius);
    if (!gridVisible) {
      console.log(`🔍 [WORKER-${this.workerIndex}] Grid not visible in viewBox, skipping`);
      return 0;
    }

    ctx.strokeStyle = config.gridColor || '#4B5563';
    ctx.lineWidth = config.gridLineWidth || 0.5;

    // Render concentric circles with culling
    const circleStart = performance.now();
    for (let i = 1; i <= 5; i++) {
      const radius = (maxRadius * i) / 5;
      if (this.isCircleInViewBox(centerX, centerY, radius, viewBox)) {
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        ctx.stroke();
        elementsRendered++;
      }
    }
    const circleTime = performance.now() - circleStart;

    // Render radial axes with culling
    const axesStart = performance.now();
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
        elementsRendered++;
      }
    }
    const axesTime = performance.now() - axesStart;

    const totalGridTime = performance.now() - gridStart;
    console.log(`🕸️ [WORKER-${this.workerIndex}] Grid rendered: ${elementsRendered} elements in ${totalGridTime.toFixed(1)}ms (circles: ${circleTime.toFixed(1)}ms, axes: ${axesTime.toFixed(1)}ms)`);
    
    return elementsRendered;
  }

  async renderPolygonsBatched(ctx, models, config, tileId) {
    const totalStart = performance.now();
    let polygonsRendered = 0;
    let polygonsSkipped = 0;
    const batchSize = 100; // Process in batches to prevent blocking

    console.log(`🔺 [WORKER-${this.workerIndex}] Starting polygon rendering: ${models.length} models`);

    for (let i = 0; i < models.length; i += batchSize) {
      if (this.cancelRequested) {
        console.log(`🛑 [WORKER-${this.workerIndex}] Rendering cancelled at polygon ${i}`);
        break;
      }

      const batchStart = performance.now();
      const batch = models.slice(i, Math.min(i + batchSize, models.length));
      
      for (const model of batch) {
        const polygonStart = performance.now();
        const rendered = this.renderSpiderPolygonOptimized(ctx, model, config);
        const polygonTime = performance.now() - polygonStart;
        
        if (rendered) {
          polygonsRendered++;
          
          // Log slow polygons
          if (polygonTime > 10) {
            console.warn(`🐌 [WORKER-${this.workerIndex}] Slow polygon in ${tileId}: ${polygonTime.toFixed(1)}ms`);
          }
        } else {
          polygonsSkipped++;
        }
      }
      
      const batchTime = performance.now() - batchStart;
      const progress = ((i + batchSize) / models.length * 100).toFixed(0);
      
      console.log(`📊 [WORKER-${this.workerIndex}] Batch ${Math.floor(i/batchSize) + 1}: ${batch.length} polygons in ${batchTime.toFixed(1)}ms (${progress}%)`);

      // Yield to prevent blocking (critical for performance)
      if (batchTime > 50) {
        await new Promise(resolve => setTimeout(resolve, 1));
      }
    }

    const totalTime = performance.now() - totalStart;
    const efficiency = polygonsRendered / totalTime * 1000;
    
    console.log(`🎯 [WORKER-${this.workerIndex}] Polygon rendering complete:
├─ Rendered: ${polygonsRendered}
├─ Skipped: ${polygonsSkipped}
├─ Total time: ${totalTime.toFixed(1)}ms
└─ Efficiency: ${efficiency.toFixed(0)} polygons/sec`);

    return polygonsRendered;
  }

  renderSpiderPolygonOptimized(ctx, model, config) {
    if (!model.parameters) return false;

    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    const centerX = config.spiderConfig.viewBoxSize / 2;
    const centerY = config.spiderConfig.viewBoxSize / 2;
    const maxRadius = config.spiderConfig.viewBoxSize / 2 - 50;

    // Fast parameter validation
    const values = params.map(param => model.parameters[param]).filter(val => typeof val === 'number');
    if (values.length < 3) return false; // Need at least 3 points

    // Calculate polygon points with optimization
    const points = [];
    let validPoints = 0;
    
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const value = model.parameters[param];
      
      if (typeof value !== 'number') continue;

      // Optimized normalization
      let normalizedValue;
      if (param.includes('C')) {
        normalizedValue = Math.log10(value * 1e6 / 0.1) / Math.log10(500);
      } else {
        normalizedValue = Math.log10(value / 10) / Math.log10(1000);
      }

      normalizedValue = Math.max(0, Math.min(1, normalizedValue));
      const radius = normalizedValue * maxRadius;
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;

      points.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      });
      validPoints++;
    }

    if (validPoints < 3) return false;

    // Optimized rendering
    ctx.strokeStyle = model.color || config.defaultColor || '#3B82F6';
    ctx.lineWidth = model.strokeWidth || config.defaultStrokeWidth || 1;
    ctx.globalAlpha = model.opacity || config.defaultOpacity || 0.7;

    // Draw polygon
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    // Reference model handling
    if (model.isReference) {
      ctx.fillStyle = model.color || config.defaultColor || '#3B82F6';
      ctx.globalAlpha = (model.opacity || 0.7) * 0.3;
      ctx.fill();
    }

    ctx.globalAlpha = 1;
    return true;
  }

  updatePerformanceMetrics(tileId, renderTime, polygonCount) {
    this.performanceMetrics.totalRenderTime += renderTime;
    this.performanceMetrics.totalTiles++;
    this.performanceMetrics.averageRenderTime = this.performanceMetrics.totalRenderTime / this.performanceMetrics.totalTiles;

    const efficiency = polygonCount / renderTime * 1000;
    
    if (!this.performanceMetrics.fastestTile || renderTime < this.performanceMetrics.fastestTile.renderTime) {
      this.performanceMetrics.fastestTile = { tileId, renderTime, efficiency };
    }
    
    if (!this.performanceMetrics.slowestTile || renderTime > this.performanceMetrics.slowestTile.renderTime) {
      this.performanceMetrics.slowestTile = { tileId, renderTime, efficiency };
    }
  }

  // Utility functions for culling (optimized)
  isGridVisible(viewBox, centerX, centerY, maxRadius) {
    return this.isPointInViewBox(centerX, centerY, viewBox) ||
           this.isCircleInViewBox(centerX, centerY, maxRadius, viewBox);
  }

  isPointInViewBox(x, y, viewBox) {
    return x >= viewBox.x && x <= viewBox.x + viewBox.width &&
           y >= viewBox.y && y <= viewBox.y + viewBox.height;
  }

  isCircleInViewBox(centerX, centerY, radius, viewBox) {
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
    if (this.isPointInViewBox(x1, y1, viewBox) || 
        this.isPointInViewBox(x2, y2, viewBox)) {
      return true;
    }

    const left = viewBox.x;
    const right = viewBox.x + viewBox.width;
    const top = viewBox.y;
    const bottom = viewBox.y + viewBox.height;

    return this.lineIntersectsRect(x1, y1, x2, y2, left, top, right, bottom);
  }

  lineIntersectsRect(x1, y1, x2, y2, left, top, right, bottom) {
    return this.lineIntersectsLine(x1, y1, x2, y2, left, top, right, top) ||
           this.lineIntersectsLine(x1, y1, x2, y2, right, top, right, bottom) ||
           this.lineIntersectsLine(x1, y1, x2, y2, right, bottom, left, bottom) ||
           this.lineIntersectsLine(x1, y1, x2, y2, left, bottom, left, top);
  }

  lineIntersectsLine(x1, y1, x2, y2, x3, y3, x4, y4) {
    const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (Math.abs(denom) < 1e-10) return false;

    const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
    const u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom;

    return t >= 0 && t <= 1 && u >= 0 && u <= 1;
  }

  cancel() {
    this.cancelRequested = true;
    console.log(`🛑 [WORKER-${this.workerIndex}] Cancellation requested`);
  }

  reset() {
    this.cancelRequested = false;
    console.log(`🔄 [WORKER-${this.workerIndex}] Reset for new job`);
  }

  getStats() {
    return {
      ...this.performanceMetrics,
      renderCount: this.renderCount,
      totalPolygonsRendered: this.totalPolygonsRendered,
      workerIndex: this.workerIndex,
      memoryUsage: this.getMemoryUsage()
    };
  }

  cleanup() {
    this.stopMemoryMonitoring();
    this.requestGarbageCollection();
    console.log(`🧹 [WORKER-${this.workerIndex}] Cleanup completed`);
  }
}

// Global renderer instance
const renderer = new EnhancedTileRenderer();

// Enhanced message handler with detailed logging
self.onmessage = async function(e) {
  const { type, data } = e.data;
  const messageStart = performance.now();

  try {
    switch (type) {
      case 'RENDER_TILES': {
        const { jobs, renderConfig } = data;
        
        renderer.workerIndex = data.workerIndex || 0;
        renderer.reset();
        renderer.startMemoryMonitoring();
        
        console.log(`🚀 [WORKER-${renderer.workerIndex}] Starting ${jobs.length} tile jobs`);
        
        // Process each tile job with detailed logging
        for (let i = 0; i < jobs.length; i++) {
          const job = jobs[i];
          
          if (renderer.cancelRequested) {
            console.log(`🛑 [WORKER-${renderer.workerIndex}] Cancelled at job ${i}/${jobs.length}`);
            break;
          }
          
          try {
            const result = await renderer.renderTile(job, renderConfig);
            
            // Send completion message with metrics
            self.postMessage({
              type: 'TILE_COMPLETE',
              data: result
            });
            
          } catch (error) {
            console.error(`❌ [WORKER-${renderer.workerIndex}] Job ${i} failed:`, error);
            self.postMessage({
              type: 'TILE_ERROR',
              data: {
                tileId: job.tileId,
                error: error.message,
                workerIndex: renderer.workerIndex
              }
            });
          }
        }
        
        // Final statistics
        const totalTime = performance.now() - messageStart;
        const stats = renderer.getStats();
        
        console.log(`🏁 [WORKER-${renderer.workerIndex}] Batch complete:
├─ Jobs: ${jobs.length}
├─ Total time: ${totalTime.toFixed(0)}ms
├─ Avg tile time: ${stats.averageRenderTime.toFixed(1)}ms
├─ Total polygons: ${stats.totalPolygonsRendered.toLocaleString()}
├─ Memory leaks: ${stats.memoryLeaks.length}
└─ Final memory: ${stats.memoryUsage.percentage.toFixed(1)}%`);

        self.postMessage({
          type: 'WORKER_STATS',
          data: {
            workerIndex: renderer.workerIndex,
            stats,
            totalBatchTime: totalTime
          }
        });

        renderer.cleanup();
        break;
      }

      case 'CANCEL_RENDER': {
        renderer.cancel();
        break;
      }

      case 'GET_STATS': {
        self.postMessage({
          type: 'WORKER_STATS',
          data: {
            workerIndex: renderer.workerIndex,
            stats: renderer.getStats()
          }
        });
        break;
      }

      default:
        console.warn(`❓ [WORKER-${renderer.workerIndex}] Unknown message type: ${type}`);
    }
    
  } catch (error) {
    console.error(`💥 [WORKER-${renderer.workerIndex}] Worker error:`, error);
    self.postMessage({
      type: 'WORKER_ERROR',
      data: {
        error: error.message,
        stack: error.stack,
        workerIndex: renderer.workerIndex
      }
    });
  }
}; 
import { ModelSnapshot } from '../types';

export interface SpiderConfig {
  viewBoxSize: number;
  centerX?: number;
  centerY?: number;
  maxRadius?: number;
}

export interface RenderConfig {
  backgroundColor?: string;
  gridColor?: string;
  gridLineWidth?: number;
  defaultColor?: string;
  defaultStrokeWidth?: number;
  defaultOpacity?: number;
  spiderConfig: SpiderConfig;
}

export interface TileConfig {
  tileSize: number; // Size of each tile in pixels
  overlap: number; // Overlap between tiles to avoid seams
  totalWidth: number;
  totalHeight: number;
  tilesX: number;
  tilesY: number;
}

export interface TileRenderJob {
  tileId: string;
  x: number; // Starting X position
  y: number; // Starting Y position
  width: number;
  height: number;
  viewBox: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  models: ModelSnapshot[]; // Models that intersect with this tile
}

export interface TileRenderResult {
  tileId: string;
  canvas: HTMLCanvasElement;
  imageData: ImageData;
  renderTime: number;
}

export class TileRenderer {
  private workers: Worker[] = [];
  private renderQueue: TileRenderJob[] = [];
  private completedTiles: Map<string, TileRenderResult> = new Map();
  private onProgress?: (progress: number, completedTiles: number, totalTiles: number) => void;
  private onTileComplete?: (tile: TileRenderResult) => void;
  private onComplete?: (finalCanvas: HTMLCanvasElement) => void;

  constructor(workerCount: number = navigator.hardwareConcurrency || 4) {
    this.initializeWorkers(workerCount);
  }

  private initializeWorkers(count: number) {
    // Clean up existing workers
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];

    // Create new tile rendering workers
    for (let i = 0; i < count; i++) {
      const worker = new Worker('/tile-worker.js');
      worker.onmessage = this.handleWorkerMessage.bind(this);
      this.workers.push(worker);
    }
  }

  private handleWorkerMessage(event: MessageEvent) {
    const { type, data } = event.data;
    
    switch (type) {
      case 'TILE_COMPLETE':
        this.handleTileComplete(data);
        break;
      case 'TILE_ERROR':
        console.error('Tile rendering error:', data);
        break;
    }
  }

  private handleTileComplete(tileResult: TileRenderResult) {
    this.completedTiles.set(tileResult.tileId, tileResult);
    
    if (this.onTileComplete) {
      this.onTileComplete(tileResult);
    }

    const progress = this.completedTiles.size / this.renderQueue.length;
    if (this.onProgress) {
      this.onProgress(progress, this.completedTiles.size, this.renderQueue.length);
    }

    // Check if all tiles are complete
    if (this.completedTiles.size === this.renderQueue.length) {
      this.assembleFinalImage();
    }
  }

  public calculateTileConfiguration(
    canvasWidth: number, 
    canvasHeight: number, 
    preferredTileSize: number = 512
  ): TileConfig {
    // Calculate optimal tile configuration
    const tilesX = Math.ceil(canvasWidth / preferredTileSize);
    const tilesY = Math.ceil(canvasHeight / preferredTileSize);
    
    // Adjust tile size to fit evenly
    const actualTileWidth = Math.ceil(canvasWidth / tilesX);
    const actualTileHeight = Math.ceil(canvasHeight / tilesY);
    const tileSize = Math.max(actualTileWidth, actualTileHeight);
    
    // Add overlap to prevent seaming artifacts
    const overlap = Math.max(16, Math.floor(tileSize * 0.05));

    return {
      tileSize,
      overlap,
      totalWidth: canvasWidth,
      totalHeight: canvasHeight,
      tilesX,
      tilesY
    };
  }

  public generateTileJobs(
    config: TileConfig,
    models: ModelSnapshot[],
    spiderConfig: SpiderConfig
  ): TileRenderJob[] {
    const jobs: TileRenderJob[] = [];
    
    for (let y = 0; y < config.tilesY; y++) {
      for (let x = 0; x < config.tilesX; x++) {
        const startX = x * config.tileSize - (x > 0 ? config.overlap : 0);
        const startY = y * config.tileSize - (y > 0 ? config.overlap : 0);
        
        const endX = Math.min(
          (x + 1) * config.tileSize + (x < config.tilesX - 1 ? config.overlap : 0),
          config.totalWidth
        );
        const endY = Math.min(
          (y + 1) * config.tileSize + (y < config.tilesY - 1 ? config.overlap : 0),
          config.totalHeight
        );

        const tileWidth = endX - startX;
        const tileHeight = endY - startY;

        // Calculate view box for this tile (what portion of the spider plot to render)
        const viewBox = {
          x: (startX / config.totalWidth) * spiderConfig.viewBoxSize,
          y: (startY / config.totalHeight) * spiderConfig.viewBoxSize,
          width: (tileWidth / config.totalWidth) * spiderConfig.viewBoxSize,
          height: (tileHeight / config.totalHeight) * spiderConfig.viewBoxSize
        };

        // Filter models that are visible in this tile
        // For spider plots, we need to check if any polygon vertices fall within the tile bounds
        const visibleModels = this.filterModelsForTile(models, viewBox, spiderConfig);

        jobs.push({
          tileId: `tile_${x}_${y}`,
          x: startX,
          y: startY,
          width: tileWidth,
          height: tileHeight,
          viewBox,
          models: visibleModels
        });
      }
    }

    return jobs;
  }

  private filterModelsForTile(
    models: ModelSnapshot[],
    viewBox: { x: number; y: number; width: number; height: number },
    spiderConfig: SpiderConfig
  ): ModelSnapshot[] {
    // For spider plots, we need to check if any part of the polygon intersects the tile
    // Since this is complex geometry, we'll use a simpler approach:
    // Include models if their parameter values would place them within the viewbox
    
    return models.filter(model => {
      if (!model.parameters) return false;
      
      // Convert parameter values to spider plot coordinates
      const coords = this.modelToSpiderCoordinates(model, spiderConfig);
      
      // Check if any coordinate falls within the viewbox
      return coords.some(coord => 
        coord.x >= viewBox.x && coord.x <= viewBox.x + viewBox.width &&
        coord.y >= viewBox.y && coord.y <= viewBox.y + viewBox.height
      );
    });
  }

  private modelToSpiderCoordinates(model: ModelSnapshot, spiderConfig: SpiderConfig): Array<{x: number, y: number}> {
    // Convert model parameters to spider plot polygon coordinates
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    const centerX = spiderConfig.viewBoxSize / 2;
    const centerY = spiderConfig.viewBoxSize / 2;
    const maxRadius = spiderConfig.viewBoxSize / 2 - 50; // Leave margin
    
    return params.map((param, index) => {
      const value = model.parameters[param as keyof typeof model.parameters];
      if (typeof value !== 'number') return { x: centerX, y: centerY };
      
      // Normalize parameter value (this would need to match your existing logic)
      let normalizedValue: number;
      if (param.includes('C')) {
        // Capacitance normalization
        normalizedValue = Math.log10(value * 1e6 / 0.1) / Math.log10(50 / 0.1);
      } else {
        // Resistance normalization
        normalizedValue = Math.log10(value / 10) / Math.log10(10000 / 10);
      }
      
      normalizedValue = Math.max(0, Math.min(1, normalizedValue));
      const radius = normalizedValue * maxRadius;
      const angle = (index * 2 * Math.PI) / params.length - Math.PI / 2;
      
      return {
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      };
    });
  }

  public async renderTiles(
    jobs: TileRenderJob[],
    renderConfig: RenderConfig,
    onProgress?: (progress: number, completed: number, total: number) => void,
    onTileComplete?: (tile: TileRenderResult) => void,
    onComplete?: (finalCanvas: HTMLCanvasElement) => void
  ): Promise<HTMLCanvasElement> {
    this.renderQueue = jobs;
    this.completedTiles.clear();
    this.onProgress = onProgress;
    this.onTileComplete = onTileComplete;
    this.onComplete = onComplete;

    // Distribute jobs among workers
    const jobsPerWorker = Math.ceil(jobs.length / this.workers.length);
    
    for (let i = 0; i < this.workers.length; i++) {
      const workerJobs = jobs.slice(i * jobsPerWorker, (i + 1) * jobsPerWorker);
      if (workerJobs.length > 0) {
        this.workers[i].postMessage({
          type: 'RENDER_TILES',
          data: {
            jobs: workerJobs,
            renderConfig
          }
        });
      }
    }

    // Return a promise that resolves when all tiles are complete
    return new Promise((resolve) => {
      this.onComplete = resolve;
    });
  }

  private assembleFinalImage(): HTMLCanvasElement {
    if (this.renderQueue.length === 0) {
      throw new Error('No tiles to assemble');
    }

    // Get configuration from first tile
    const firstTile = this.completedTiles.values().next().value;
    if (!firstTile) {
      throw new Error('No completed tiles found');
    }

    // Calculate final canvas size
    const config = this.calculateTileConfiguration(
      firstTile.canvas.width * this.renderQueue[0].width,
      firstTile.canvas.height * this.renderQueue[0].height
    );

    // Create final canvas
    const finalCanvas = document.createElement('canvas');
    finalCanvas.width = config.totalWidth;
    finalCanvas.height = config.totalHeight;
    const ctx = finalCanvas.getContext('2d');
    
    if (!ctx) {
      throw new Error('Could not get canvas context');
    }

    // Set high-quality rendering
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';

    // Composite all tiles with proper blending
    for (const job of this.renderQueue) {
      const tile = this.completedTiles.get(job.tileId);
      if (!tile) continue;

      // Handle overlap blending
      if (job.x > 0 || job.y > 0) {
        // Use alpha blending for overlapping regions
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 0.8;
      } else {
        ctx.globalCompositeOperation = 'source-over';
        ctx.globalAlpha = 1.0;
      }

      ctx.drawImage(tile.canvas, job.x, job.y);
    }

    ctx.globalAlpha = 1.0;
    ctx.globalCompositeOperation = 'source-over';

    if (this.onComplete) {
      this.onComplete(finalCanvas);
    }

    return finalCanvas;
  }

  public cancelRendering() {
    this.workers.forEach(worker => {
      worker.postMessage({ type: 'CANCEL_RENDER' });
    });
    this.renderQueue = [];
    this.completedTiles.clear();
  }

  public destroy() {
    this.cancelRendering();
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
  }
}

// Export utility functions
export function createTileRenderingHook() {
  let renderer: TileRenderer | null = null;

  return {
    initRenderer: (workerCount?: number) => {
      if (renderer) renderer.destroy();
      renderer = new TileRenderer(workerCount);
      return renderer;
    },
    
    getRenderer: () => renderer,
    
    destroyRenderer: () => {
      if (renderer) {
        renderer.destroy();
        renderer = null;
      }
    }
  };
} 
"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ModelSnapshot } from '../types';
import { CircuitParameters } from '../types/parameters';

interface SpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string | null;
  opacityFactor: number;
  maxPolygons: number;
  onExportImage?: (canvas: HTMLCanvasElement) => void;
  visualizationMode?: 'color' | 'opacity';
  gridSize?: number;
  includeLabels?: boolean;
  backgroundColor?: 'transparent' | 'white' | 'black';
  groundTruthParams?: CircuitParameters;
  showGroundTruth?: boolean;
}

// Ultra-high performance Canvas-based spider plot renderer
class OptimizedSpiderRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationFrameId: number | null = null;
  private isRendering = false;
  private lastRenderTime = 0;
  private readonly THROTTLE_MS = 4; // ~240fps for ultra-smooth interaction
  private needsRender = false;
  private pendingModels: ModelSnapshot[] = [];
  private renderConfig: RenderConfig;
  private offscreenCanvas: HTMLCanvasElement | null = null;
  private offscreenCtx: CanvasRenderingContext2D | null = null;
  private cachedPaths = new Map<string, Path2D>();
  private viewport = { models: [] as ModelSnapshot[], hash: '' };

  constructor(canvas: HTMLCanvasElement, config: RenderConfig) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { 
      alpha: config.backgroundColor === 'transparent',
      willReadFrequently: false, // Enable GPU acceleration
      desynchronized: true // Reduce input latency
    })!;
    this.renderConfig = config;
    this.setupCanvas();
    this.setupOffscreenCanvas();
    this.startRenderLoop();
  }

  private setupCanvas() {
    this.ctx.imageSmoothingEnabled = false; // Disable for performance with large datasets
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
  }

  private setupOffscreenCanvas() {
    this.offscreenCanvas = document.createElement('canvas');
    this.offscreenCtx = this.offscreenCanvas.getContext('2d', {
      alpha: false,
      willReadFrequently: false
    })!;
  }

  private startRenderLoop() {
    const renderLoop = () => {
      if (this.needsRender && !this.isRendering) {
        this.performRender();
        this.needsRender = false;
      }
      this.animationFrameId = requestAnimationFrame(renderLoop);
    };
    renderLoop();
  }

  // Ultra-optimized render function with intelligent caching and virtualization
  public render(models: ModelSnapshot[]) {
    // Skip rendering if no models
    if (!models.length) {
      this.pendingModels = [];
      this.needsRender = false;
      this.clearCanvas();
      return;
    }

    // Generate hash for model set to detect changes
    const modelHash = this.generateModelHash(models);
    
    // Skip if same data is already rendered
    if (this.viewport.hash === modelHash) {
      return;
    }

    // Apply intelligent virtualization for large datasets
    const optimizedModels = this.virtualizeModels(models);

    const now = performance.now();
    if (now - this.lastRenderTime < this.THROTTLE_MS) {
      // Throttle updates to prevent excessive rendering
      this.pendingModels = optimizedModels;
      this.needsRender = true;
      return;
    }

    this.lastRenderTime = now;
    this.pendingModels = optimizedModels;
    this.viewport.hash = modelHash;
    this.needsRender = true;
  }

  // Intelligent model virtualization for performance
  private virtualizeModels(models: ModelSnapshot[]): ModelSnapshot[] {
    const maxRenderPoints = 100000; // Maximum points to render for performance
    
    if (models.length <= maxRenderPoints) {
      return models;
    }

    // For large datasets, use adaptive sampling
    // Prioritize models with better resnorm (lower values)
    const sortedModels = [...models].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
    
    // Take best 70% of max points, then sample remaining 30% for coverage
    const bestCount = Math.floor(maxRenderPoints * 0.7);
    const sampleCount = maxRenderPoints - bestCount;
    
    const bestModels = sortedModels.slice(0, bestCount);
    const remainingModels = sortedModels.slice(bestCount);
    
    // Sample remaining models uniformly
    const sampleStep = Math.max(1, Math.floor(remainingModels.length / sampleCount));
    const sampledModels = remainingModels.filter((_, index) => index % sampleStep === 0).slice(0, sampleCount);
    
    return [...bestModels, ...sampledModels];
  }

  // Generate hash for model set to enable caching
  private generateModelHash(models: ModelSnapshot[]): string {
    const key = `${models.length}_${models[0]?.id || ''}_${models[models.length - 1]?.id || ''}`;
    return key;
  }

  // Efficient canvas clearing
  private clearCanvas() {
    // Use clearRect for better performance than resetting width
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private performRender() {
    if (this.isRendering || !this.pendingModels.length) return;
    this.isRendering = true;

    try {
      // Use requestIdleCallback for non-blocking rendering when available
      const renderTask = () => {
        // Efficient canvas clearing
        this.clearCanvas();
        
        // Draw background
        this.drawBackground();

        // Draw grid (cached when possible)
        this.drawGrid();

        // Ultra-optimized batch drawing with Path2D caching
        this.ultraOptimizedBatchDraw(this.pendingModels);

        // Draw ground truth overlay if enabled
        if (this.renderConfig.showGroundTruth && this.renderConfig.groundTruthParams) {
          this.drawGroundTruthOverlay();
        }

        // Draw labels AFTER polygons to ensure highest z-index
        if (this.renderConfig.includeLabels) {
          this.drawLabels();
        }
      };

      // Use requestIdleCallback for better performance when available
      if ('requestIdleCallback' in window) {
        (window as Window & { requestIdleCallback: (callback: () => void, options?: { timeout: number }) => void }).requestIdleCallback(renderTask, { timeout: 16 });
      } else {
        renderTask();
      }

    } finally {
      this.isRendering = false;
    }
  }

  private drawBackground() {
    const { backgroundColor } = this.renderConfig;
    
    if (backgroundColor === 'white') {
      this.ctx.fillStyle = '#ffffff';
    } else if (backgroundColor === 'black') {
      this.ctx.fillStyle = '#000000';
    } else {
      // Default dark background like playground
      this.ctx.fillStyle = '#0f172a';
    }
    
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private drawGrid() {
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;
    // Increase radius for better space utilization - from 0.35 to 0.42
    const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.42;
    const gridSize = this.renderConfig.gridSize || 5;
    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb']; // Grouped: Rs (shunt), apical (Ra, Ca), basal (Rb, Cb)
    const angleStep = (2 * Math.PI) / params.length;

    this.ctx.strokeStyle = '#4B5563';
    this.ctx.lineWidth = 1;
    this.ctx.globalAlpha = 0.3;

    // Draw concentric grid circles/ticks only on axes (no connecting lines)
    for (let level = 1; level <= gridSize; level++) {
      const radius = (maxRadius * level) / gridSize;
      
      // Draw small tick marks on each axis instead of connecting pentagon
      for (let i = 0; i < params.length; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        // Draw small perpendicular tick mark
        const tickLength = 4;
        const perpAngle = angle + Math.PI / 2;
        const tickStartX = x - Math.cos(perpAngle) * tickLength / 2;
        const tickStartY = y - Math.sin(perpAngle) * tickLength / 2;
        const tickEndX = x + Math.cos(perpAngle) * tickLength / 2;
        const tickEndY = y + Math.sin(perpAngle) * tickLength / 2;
        
        this.ctx.beginPath();
        this.ctx.moveTo(tickStartX, tickStartY);
        this.ctx.lineTo(tickEndX, tickEndY);
        this.ctx.stroke();
      }
    }

    // Draw radial axes
    this.ctx.globalAlpha = 1;
    this.ctx.lineWidth = 0.5;
    for (let i = 0; i < params.length; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const endX = centerX + Math.cos(angle) * maxRadius;
      const endY = centerY + Math.sin(angle) * maxRadius;
      
      this.ctx.beginPath();
      this.ctx.moveTo(centerX, centerY);
      this.ctx.lineTo(endX, endY);
      this.ctx.stroke();
    }

    this.ctx.globalAlpha = 1;
  }

  private drawLabels() {
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;
    // Use same increased radius for consistency
    const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.42;
    const gridSize = this.renderConfig.gridSize || 5;
    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb']; // Grouped: Rs (shunt), apical (Ra, Ca), basal (Rb, Cb)
    const angleStep = (2 * Math.PI) / params.length;

    // Fixed parameter ranges exactly matching playground
    const paramRanges = {
      Rs: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };

    // Add grid value labels on each axis
    this.ctx.fillStyle = '#9CA3AF';
    this.ctx.font = '400 9px Inter, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    // Generate logarithmic grid values for each parameter
    for (let level = 1; level <= gridSize; level++) {
      const radius = (maxRadius * level) / gridSize;
      
      for (let i = 0; i < params.length; i++) {
        const param = params[i];
        const angle = i * angleStep - Math.PI / 2;
        const range = paramRanges[param as keyof typeof paramRanges];
        
        // Calculate logarithmic value at this grid level
        const logMin = Math.log10(range.min);
        const logMax = Math.log10(range.max);
        const logValue = logMin + (level / gridSize) * (logMax - logMin);
        const actualValue = Math.pow(10, logValue);
        
        // Format value based on parameter type (no units - they're in the parameter name)
        let displayValue: string;
        if (param.includes('C')) {
          const valueInMicroF = actualValue * 1e6;
          if (valueInMicroF >= 10) {
            displayValue = `${Math.round(valueInMicroF)}`;
          } else if (valueInMicroF >= 1) {
            displayValue = `${valueInMicroF.toFixed(1)}`;
          } else {
            displayValue = `${valueInMicroF.toFixed(2)}`;
          }
        } else {
          if (actualValue >= 1000) {
            const kValue = actualValue / 1000;
            displayValue = kValue >= 10 ? `${Math.round(kValue)}k` : `${kValue.toFixed(1)}k`;
          } else {
            displayValue = actualValue >= 100 ? `${Math.round(actualValue)}` : `${actualValue.toFixed(0)}`;
          }
        }
        
        // Position grid value label closer to tick marks for better readability
        const labelDistance = radius + 8;
        const labelX = centerX + Math.cos(angle) * labelDistance;
        const labelY = centerY + Math.sin(angle) * labelDistance;
        
        // Draw enhanced tick mark at exact position
        this.ctx.strokeStyle = '#6B7280';
        this.ctx.lineWidth = 1.2;
        this.ctx.beginPath();
        const tickStart = radius - 3;
        const tickEnd = radius + 3;
        this.ctx.moveTo(centerX + Math.cos(angle) * tickStart, centerY + Math.sin(angle) * tickStart);
        this.ctx.lineTo(centerX + Math.cos(angle) * tickEnd, centerY + Math.sin(angle) * tickEnd);
        this.ctx.stroke();
        
        // Draw value label with higher z-index styling
        this.ctx.save();
        this.ctx.shadowColor = 'rgba(0, 0, 0, 0.8)';
        this.ctx.shadowBlur = 3;
        this.ctx.shadowOffsetX = 1;
        this.ctx.shadowOffsetY = 1;
        this.ctx.fillStyle = '#E5E7EB';
        this.ctx.font = '600 10px Inter, sans-serif';
        this.ctx.fillText(displayValue, labelX, labelY);
        this.ctx.restore();
      }
    }

    // Parameter labels
    this.ctx.fillStyle = '#E5E7EB';
    this.ctx.font = '600 13px Inter, sans-serif';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const angle = i * angleStep - Math.PI / 2;
      const labelDistance = maxRadius + 32; // Reduced from 40 to save space
      const labelX = centerX + Math.cos(angle) * labelDistance;
      const labelY = centerY + Math.sin(angle) * labelDistance;
      
      // Draw parameter name with units in parentheses
      this.ctx.save();
      this.ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
      this.ctx.shadowBlur = 4;
      this.ctx.shadowOffsetX = 1;
      this.ctx.shadowOffsetY = 1;
      this.ctx.font = '700 14px Inter, sans-serif';
      this.ctx.fillStyle = '#F3F4F6';
      
      // Add units to parameter names
      const paramWithUnits = {
        Rs: 'Rs (Ω)',
        Ra: 'Ra (Ω)', 
        Rb: 'Rb (Ω)',
        Ca: 'Ca (µF)',
        Cb: 'Cb (µF)'
      };
      
      this.ctx.fillText(paramWithUnits[param as keyof typeof paramWithUnits] || param, labelX, labelY);
      this.ctx.restore();
    }
  }

  private drawGroundTruthOverlay() {
    if (!this.renderConfig.groundTruthParams) return;
    
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;
    const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.42;
    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'];
    const angleStep = (2 * Math.PI) / params.length;

    // Fixed parameter ranges exactly matching playground
    const paramRanges = {
      Rs: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };

    const groundTruthPath = new Path2D();
    
    // Calculate ground truth polygon points
    const groundTruthPoints: { x: number; y: number }[] = [];
    
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const angle = i * angleStep - Math.PI / 2;
      const range = paramRanges[param as keyof typeof paramRanges];
      
      // Get ground truth value for this parameter
      const groundTruthValue = this.renderConfig.groundTruthParams[param as keyof CircuitParameters];
      
      // Calculate logarithmic position
      const logMin = Math.log10(range.min);
      const logMax = Math.log10(range.max);
      const logValue = Math.log10(groundTruthValue as number);
      const normalizedValue = (logValue - logMin) / (logMax - logMin);
      
      // Clamp to valid range
      const clampedValue = Math.max(0, Math.min(1, normalizedValue));
      
      // Calculate position
      const radius = maxRadius * clampedValue;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      groundTruthPoints.push({ x, y });
      
      if (i === 0) {
        groundTruthPath.moveTo(x, y);
      } else {
        groundTruthPath.lineTo(x, y);
      }
    }
    
    // Close the path
    groundTruthPath.closePath();
    
    // Draw ground truth overlay with distinctive white color
    this.ctx.save();
    this.ctx.strokeStyle = '#FFFFFF';
    this.ctx.lineWidth = 1.5;
    this.ctx.setLineDash([6, 3]); // Dashed line for distinction
    this.ctx.globalAlpha = 0.9;
    this.ctx.stroke(groundTruthPath);
    
    this.ctx.restore();
    
    // Draw ground truth indicator points
    this.ctx.save();
    this.ctx.fillStyle = '#FFFFFF';
    this.ctx.strokeStyle = '#000000';
    this.ctx.lineWidth = 2;
    this.ctx.globalAlpha = 1;
    
    for (const point of groundTruthPoints) {
      this.ctx.beginPath();
      this.ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
      this.ctx.fill();
      this.ctx.stroke();
    }
    
    this.ctx.restore();
  }

  // Ultra-optimized batch drawing with Path2D caching and GPU acceleration
  private ultraOptimizedBatchDraw(models: ModelSnapshot[]) {
    if (!models.length) return;
    
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;
    const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.42;

    // Cached parameter ranges for performance
    const paramRanges = {
      Rs: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };

    // Ultra-efficient grouping with minimal object creation
    const renderGroups = new Map<string, {
      path: Path2D,
      color: string,
      opacity: number,
      count: number
    }>();
    
    // Single pass through models for grouping and path creation
    models.forEach(model => {
      const color = model.color || '#3B82F6';
      const opacity = Math.round((model.opacity || 0.7) * 10) / 10; // 0.1 steps for performance
      const groupKey = `${color}_${opacity}`;
      
      let group = renderGroups.get(groupKey);
      if (!group) {
        group = {
          path: new Path2D(),
          color,
          opacity,
          count: 0
        };
        renderGroups.set(groupKey, group);
      }
      
      // Calculate polygon points and add to path
      const points = this.calculatePolygonPoints(model, centerX, centerY, maxRadius, paramRanges);
      if (points.length >= 3) {
        group.path.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          group.path.lineTo(points[i].x, points[i].y);
        }
        group.path.closePath();
        group.count++;
      }
    });

    // Render all groups with minimal state changes
    let lastColor = '';
    let lastOpacity = -1;
    
    for (const group of renderGroups.values()) {
      if (group.count === 0) continue;
      
      // Only change canvas state when necessary
      if (group.color !== lastColor) {
        this.ctx.strokeStyle = group.color;
        lastColor = group.color;
      }
      
      const finalOpacity = group.opacity * (this.renderConfig.opacityFactor || 1);
      if (finalOpacity !== lastOpacity) {
        this.ctx.globalAlpha = finalOpacity;
        lastOpacity = finalOpacity;
      }
      
      // Single draw call per group using Path2D for GPU acceleration
      this.ctx.stroke(group.path);
    }

    // Reset canvas state
    this.ctx.globalAlpha = 1;
  }

  private calculatePolygonPoints(
    model: ModelSnapshot, 
    centerX: number, 
    centerY: number, 
    maxRadius: number,
    paramRanges: Record<string, { min: number; max: number }>
  ) {
    if (!model.parameters) return [];

    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb']; // Grouped: Rs (shunt), apical (Ra, Ca), basal (Rb, Cb)
    const points = [];
    const gridSize = this.renderConfig.gridSize || 5;

    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const value = model.parameters[param as keyof typeof model.parameters];
      
      if (typeof value !== 'number') continue;

      // Calculate which grid level this value should snap to
      const range = paramRanges[param as keyof typeof paramRanges];
      const logMin = Math.log10(range.min);
      const logMax = Math.log10(range.max);
      const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
      
      // Find the closest grid level (1 to gridSize)
      const normalizedPosition = (logValue - logMin) / (logMax - logMin);
      const gridLevel = Math.round(normalizedPosition * gridSize);
      const clampedGridLevel = Math.max(1, Math.min(gridSize, gridLevel));
      
      // Snap to exact grid level radius
      const radius = (maxRadius * clampedGridLevel) / gridSize;
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;

      points.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      });
    }

    return points;
  }

  public destroy() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
    }
  }
}

interface RenderConfig {
  backgroundColor?: 'transparent' | 'white' | 'black';
  gridSize?: number;
  includeLabels?: boolean;
  opacityFactor?: number;
  groundTruthParams?: CircuitParameters;
  showGroundTruth?: boolean;
}

const SpiderPlotComponent: React.FC<SpiderPlotProps> = ({
  meshItems,
  opacityFactor,
  maxPolygons,
  visualizationMode = 'color',
  gridSize = 5,
  includeLabels = true,
  backgroundColor = 'transparent',
  groundTruthParams,
  showGroundTruth = false
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const rendererRef = useRef<OptimizedSpiderRenderer | null>(null);
  const [isCanvasMode, setIsCanvasMode] = useState(true);
  const [renderStats, setRenderStats] = useState({
    modelCount: 0,
    renderTime: 0,
    memoryUsage: 0
  });

  // Memoized filtered models for performance
  const visibleModels = useMemo(() => {
    if (!meshItems?.length) return [];
    // Filter out models without valid parameters early to avoid processing
    const validModels = meshItems.filter(model => 
      model.parameters && 
      typeof model.parameters.Rs === 'number' &&
      typeof model.parameters.Ra === 'number' &&
      typeof model.parameters.Rb === 'number' &&
      typeof model.parameters.Ca === 'number' &&
      typeof model.parameters.Cb === 'number'
    );
    return validModels.slice(0, maxPolygons);
  }, [meshItems, maxPolygons]);

  // Initialize canvas renderer
  useEffect(() => {
    if (canvasRef.current && isCanvasMode) {
      const canvas = canvasRef.current;
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      canvas.style.width = canvas.offsetWidth + 'px';
      canvas.style.height = canvas.offsetHeight + 'px';
      
      rendererRef.current = new OptimizedSpiderRenderer(canvas, {
        backgroundColor,
        gridSize,
        includeLabels,
        opacityFactor: opacityFactor || 1,
        groundTruthParams,
        showGroundTruth
      });
    }

    return () => {
      if (rendererRef.current) {
        rendererRef.current.destroy();
        rendererRef.current = null;
      }
    };
  }, [isCanvasMode, backgroundColor, gridSize, includeLabels, opacityFactor, groundTruthParams, showGroundTruth]);

  // Render when models change
  useEffect(() => {
    if (rendererRef.current && visibleModels.length > 0) {
      const startTime = performance.now();
      
      rendererRef.current.render(visibleModels);

      const renderTime = performance.now() - startTime;
      const memoryUsage = ((performance as Performance & { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize || 0) / 1024 / 1024;
      
      setRenderStats({
        modelCount: visibleModels.length,
        renderTime,
        memoryUsage
      });
    }
  }, [visibleModels]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      if (canvasRef.current && rendererRef.current) {
        const canvas = canvasRef.current;
        canvas.width = canvas.offsetWidth * window.devicePixelRatio;
        canvas.height = canvas.offsetHeight * window.devicePixelRatio;
        canvas.style.width = canvas.offsetWidth + 'px';
        canvas.style.height = canvas.offsetHeight + 'px';
        
        // Re-render after resize
        if (visibleModels.length > 0) {
          rendererRef.current.render(visibleModels);
        }
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [visibleModels]);

  // Export function - available for future use
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const exportCanvas = useCallback(() => {
    if (canvasRef.current) {
      // Export functionality can be added here
      console.log('Export canvas functionality available');
    }
  }, []);

  // Performance controls
  const PerformanceControls = () => (
    <div className="absolute top-2 right-2 bg-neutral-800/90 backdrop-blur-sm border border-neutral-700 rounded-lg p-3 text-xs">
      <div className="flex items-center gap-3 mb-2">
        <span className="text-neutral-300 font-medium">Performance</span>
        <button
          onClick={() => setIsCanvasMode(!isCanvasMode)}
          className={`px-2 py-1 rounded text-xs ${
            isCanvasMode 
              ? 'bg-green-600 text-white' 
              : 'bg-neutral-600 text-neutral-300'
          }`}
        >
          {isCanvasMode ? 'Canvas' : 'SVG'}
        </button>
      </div>
      
      <div className="space-y-1 text-neutral-400">
        <div>Models: {renderStats.modelCount.toLocaleString()}</div>
        <div>Render: {renderStats.renderTime.toFixed(1)}ms</div>
        <div>Memory: {renderStats.memoryUsage.toFixed(1)}MB</div>
        <div>Mode: {visualizationMode}</div>
      </div>
    </div>
  );

  // SVG rendering for export compatibility
  const renderSVGContent = () => {
    const width = 800;
    const height = 800;
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) * 0.42;
    const params = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb']; // Grouped: Rs (shunt), apical (Ra, Ca), basal (Rb, Cb)
    const angleStep = (2 * Math.PI) / params.length;
    
    // Fixed parameter ranges
    const paramRanges = {
      Rs: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };

    return (
      <svg
        ref={svgRef}
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-full"
        style={{ background: backgroundColor === 'white' ? '#ffffff' : backgroundColor === 'black' ? '#000000' : '#0f172a' }}
      >
        {/* Background */}
        <rect
          width={width}
          height={height}
          fill={backgroundColor === 'white' ? '#ffffff' : backgroundColor === 'black' ? '#000000' : '#0f172a'}
        />
        
        {/* Grid */}
        <g stroke="#4B5563" strokeWidth="1" opacity="0.3" fill="none">
          {/* Grid tick marks on axes (no connecting lines) */}
          {Array.from({ length: gridSize }, (_, level) => {
            const radius = (maxRadius * (level + 1)) / gridSize;
            return params.map((_, i) => {
              const angle = i * angleStep - Math.PI / 2;
              const x = centerX + Math.cos(angle) * radius;
              const y = centerY + Math.sin(angle) * radius;
              
              // Create perpendicular tick mark
              const tickLength = 4;
              const perpAngle = angle + Math.PI / 2;
              const tickStartX = x - Math.cos(perpAngle) * tickLength / 2;
              const tickStartY = y - Math.sin(perpAngle) * tickLength / 2;
              const tickEndX = x + Math.cos(perpAngle) * tickLength / 2;
              const tickEndY = y + Math.sin(perpAngle) * tickLength / 2;
              
              return (
                <line
                  key={`${level}-${i}`}
                  x1={tickStartX}
                  y1={tickStartY}
                  x2={tickEndX}
                  y2={tickEndY}
                />
              );
            });
          }).flat()}
          
          {/* Radial axes */}
          {params.map((_, i) => {
            const angle = i * angleStep - Math.PI / 2;
            const endX = centerX + Math.cos(angle) * maxRadius;
            const endY = centerY + Math.sin(angle) * maxRadius;
            return (
              <line
                key={i}
                x1={centerX}
                y1={centerY}
                x2={endX}
                y2={endY}
                strokeWidth="0.5"
              />
            );
          })}
        </g>
        
        {/* Parameter labels */}
        {includeLabels && params.map((param, i) => {
          const angle = i * angleStep - Math.PI / 2;
          const labelDistance = maxRadius + 32;
          const labelX = centerX + Math.cos(angle) * labelDistance;
          const labelY = centerY + Math.sin(angle) * labelDistance;
          
          // Add units to parameter names
          const paramWithUnits = {
            Rs: 'Rs (Ω)',
            Ra: 'Ra (Ω)', 
            Rb: 'Rb (Ω)',
            Ca: 'Ca (µF)',
            Cb: 'Cb (µF)'
          };
          
          return (
            <g key={param}>
              <text
                x={labelX}
                y={labelY}
                textAnchor="middle"
                fill="#F3F4F6"
                fontSize="14"
                fontWeight="700"
                fontFamily="Inter, sans-serif"
              >
                {paramWithUnits[param as keyof typeof paramWithUnits] || param}
              </text>
            </g>
          );
        })}
        
        {/* Model polygons */}
        <g fill="none" strokeWidth="1">
          {visibleModels.slice(0, 1000).map((model, index) => { // Limit for SVG performance
            if (!model.parameters) return null;
            
            const points = params.map((param, i) => {
              const value = model.parameters[param as keyof typeof model.parameters];
              if (typeof value !== 'number') return null;
              
              // Snap to grid levels
              const range = paramRanges[param as keyof typeof paramRanges];
              const logMin = Math.log10(range.min);
              const logMax = Math.log10(range.max);
              const logValue = Math.log10(Math.max(range.min, Math.min(range.max, value)));
              
              const normalizedPosition = (logValue - logMin) / (logMax - logMin);
              const gridLevel = Math.round(normalizedPosition * gridSize);
              const clampedGridLevel = Math.max(1, Math.min(gridSize, gridLevel));
              
              const radius = (maxRadius * clampedGridLevel) / gridSize;
              const angle = i * angleStep - Math.PI / 2;
              const x = centerX + Math.cos(angle) * radius;
              const y = centerY + Math.sin(angle) * radius;
              
              return `${x},${y}`;
            }).filter(Boolean);
            
            if (points.length < 3) return null;
            
            return (
              <polygon
                key={`${model.id}-${index}`}
                points={points.join(' ')}
                stroke={model.color || '#3B82F6'}
                strokeOpacity={model.opacity || 0.7}
                fill="none"
              />
            );
          })}
        </g>
        
        {/* Ground Truth Overlay */}
        {showGroundTruth && groundTruthParams && (
          <g>
            {(() => {
              const groundTruthPoints = params.map((param, i) => {
                const angle = i * angleStep - Math.PI / 2;
                const range = paramRanges[param as keyof typeof paramRanges];
                
                // Get ground truth value for this parameter
                const groundTruthValue = groundTruthParams[param as keyof CircuitParameters];
                
                // Calculate logarithmic position
                const logMin = Math.log10(range.min);
                const logMax = Math.log10(range.max);
                const logValue = Math.log10(groundTruthValue as number);
                const normalizedValue = (logValue - logMin) / (logMax - logMin);
                
                // Clamp to valid range
                const clampedValue = Math.max(0, Math.min(1, normalizedValue));
                
                // Calculate position
                const radius = maxRadius * clampedValue;
                const x = centerX + Math.cos(angle) * radius;
                const y = centerY + Math.sin(angle) * radius;
                
                return `${x},${y}`;
              });
              
              return (
                <>
                  {/* Ground truth polygon */}
                  <polygon
                    points={groundTruthPoints.join(' ')}
                    stroke="#FFFFFF"
                    strokeWidth="1.5"
                    strokeDasharray="6 3"
                    fill="none"
                    strokeOpacity="0.9"
                  />
                  
                  {/* Ground truth indicator points */}
                  {groundTruthPoints.map((point, i) => {
                    const [x, y] = point.split(',').map(Number);
                    return (
                      <circle
                        key={`ground-truth-${i}`}
                        cx={x}
                        cy={y}
                        r="6"
                        fill="#FFFFFF"
                        stroke="#000000"
                        strokeWidth="2"
                      />
                    );
                  })}
                </>
              );
            })()}
          </g>
        )}
      </svg>
    );
  };

  // Render content based on mode
  const renderContent = () => {
    if (isCanvasMode) {
      return (
        <>
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ 
              display: 'block',
              background: backgroundColor === 'white' ? '#ffffff' : 
                          backgroundColor === 'black' ? '#000000' : '#0f172a'
            }}
          />
          {/* Hidden SVG for export */}
          <div className="hidden">
            {renderSVGContent()}
          </div>
        </>
      );
    } else {
      return renderSVGContent();
    }
  };

  return (
    <div className="relative w-full h-full min-h-[500px] max-h-[800px] aspect-square max-w-4xl mx-auto">
      <PerformanceControls />
      
      <div className="w-full h-full rounded-lg overflow-hidden border border-neutral-700/50 bg-gradient-to-br from-neutral-900/50 to-neutral-800/30">
        {renderContent()}
      </div>
      
      {/* Add gradient legend for resnorm colors */}
      <div className="absolute bottom-4 left-4 bg-neutral-800/90 backdrop-blur-sm border border-neutral-700 rounded-lg p-3">
        <div className="text-xs font-medium text-neutral-200 mb-2">Resnorm Scale</div>
        <div className="flex items-center gap-2">
          <div className="w-16 h-3 rounded bg-gradient-to-r from-green-500 via-yellow-500 to-red-500"></div>
          <div className="text-xs text-neutral-400">
            <span className="text-green-400">Low</span> → <span className="text-red-400">High</span>
          </div>
        </div>
        <div className="flex justify-between text-xs text-neutral-500 mt-1">
          <span>0.01</span>
          <span>10+</span>
        </div>
      </div>
    </div>
  );
};

// Memoized component with deep comparison for performance
export const SpiderPlot = React.memo(SpiderPlotComponent, (prevProps, nextProps) => {
  // Custom comparison for optimal performance
  return (
    prevProps.meshItems === nextProps.meshItems &&
    prevProps.opacityFactor === nextProps.opacityFactor &&
    prevProps.maxPolygons === nextProps.maxPolygons &&
    prevProps.visualizationMode === nextProps.visualizationMode &&
    prevProps.gridSize === nextProps.gridSize &&
    prevProps.includeLabels === nextProps.includeLabels &&
    prevProps.backgroundColor === nextProps.backgroundColor &&
    prevProps.meshItems.length === nextProps.meshItems.length
  );
});
"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ModelSnapshot } from '../types';

interface SpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string | null;
  opacityFactor: number;
  maxPolygons: number;
  onExportImage?: (canvas: HTMLCanvasElement) => void;
  visualizationMode?: 'color' | 'opacity';
  opacityIntensity?: number;
  gridSize?: number;
  includeLabels?: boolean;
  backgroundColor?: 'transparent' | 'white' | 'black';
}

// High-performance Canvas-based spider plot renderer
class OptimizedSpiderRenderer {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private animationFrameId: number | null = null;
  private isRendering = false;
  private lastRenderTime = 0;
  private readonly THROTTLE_MS = 16; // ~60fps
  private needsRender = false;
  private pendingModels: ModelSnapshot[] = [];
  private renderConfig: RenderConfig;

  constructor(canvas: HTMLCanvasElement, config: RenderConfig) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d')!;
    this.renderConfig = config;
    this.setupCanvas();
    this.startRenderLoop();
  }

  private setupCanvas() {
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
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

  // Throttled render function using RequestAnimationFrame
  public render(models: ModelSnapshot[]) {
    const now = performance.now();
    if (now - this.lastRenderTime < this.THROTTLE_MS) {
      // Throttle updates to prevent excessive rendering
      this.pendingModels = models;
      this.needsRender = true;
      return;
    }

    this.lastRenderTime = now;
    this.pendingModels = models;
    this.needsRender = true;
  }

  private performRender() {
    if (this.isRendering || !this.pendingModels.length) return;
    this.isRendering = true;

    try {
      // Clear canvas with background
      this.drawBackground();

      // Draw grid
      this.drawGrid();

      // Draw labels if enabled
      if (this.renderConfig.includeLabels) {
        this.drawLabels();
      }

      // Batch draw all polygons
      this.batchDrawPolygons(this.pendingModels);

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
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    const angleStep = (2 * Math.PI) / params.length;

    this.ctx.strokeStyle = '#4B5563';
    this.ctx.lineWidth = 1;
    this.ctx.globalAlpha = 0.3;

    // Draw concentric pentagon grids based on actual grid size
    for (let level = 1; level <= gridSize; level++) {
      const radius = (maxRadius * level) / gridSize;
      
      // Draw pentagon grid at this level
      this.ctx.beginPath();
      for (let i = 0; i <= params.length; i++) {
        const angle = i * angleStep - Math.PI / 2;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;
        
        if (i === 0) {
          this.ctx.moveTo(x, y);
        } else {
          this.ctx.lineTo(x, y);
        }
      }
      this.ctx.stroke();
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
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
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
        
        // Format value based on parameter type
        let displayValue: string;
        if (param.includes('C')) {
          const valueInMicroF = actualValue * 1e6;
          if (valueInMicroF >= 10) {
            displayValue = `${Math.round(valueInMicroF)}µF`;
          } else if (valueInMicroF >= 1) {
            displayValue = `${valueInMicroF.toFixed(1)}µF`;
          } else {
            displayValue = `${valueInMicroF.toFixed(2)}µF`;
          }
        } else {
          if (actualValue >= 1000) {
            const kValue = actualValue / 1000;
            displayValue = kValue >= 10 ? `${Math.round(kValue)}kΩ` : `${kValue.toFixed(1)}kΩ`;
          } else {
            displayValue = actualValue >= 100 ? `${Math.round(actualValue)}Ω` : `${actualValue.toFixed(0)}Ω`;
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
      
      // Enhanced axis labels with units and ranges
      const ranges = {
        Rs: '10Ω - 10kΩ',
        Ra: '10Ω - 10kΩ', 
        Rb: '10Ω - 10kΩ',
        Ca: '0.1 - 50µF',
        Cb: '0.1 - 50µF'
      };
      
      // Draw parameter name with enhanced visibility
      this.ctx.save();
      this.ctx.shadowColor = 'rgba(0, 0, 0, 0.9)';
      this.ctx.shadowBlur = 4;
      this.ctx.shadowOffsetX = 1;
      this.ctx.shadowOffsetY = 1;
      this.ctx.font = '700 14px Inter, sans-serif';
      this.ctx.fillStyle = '#F3F4F6';
      this.ctx.fillText(param, labelX, labelY - 8);
      
      // Draw range with enhanced visibility
      this.ctx.shadowBlur = 2;
      this.ctx.font = '500 10px Inter, sans-serif';
      this.ctx.fillStyle = '#D1D5DB';
      this.ctx.fillText(ranges[param as keyof typeof ranges], labelX, labelY + 8);
      this.ctx.restore();
    }
  }

  // Batch drawing for performance - group by color and draw in batches
  private batchDrawPolygons(models: ModelSnapshot[]) {
    const centerX = this.canvas.width / 2;
    const centerY = this.canvas.height / 2;
    // Use same increased radius for consistency
    const maxRadius = Math.min(this.canvas.width, this.canvas.height) * 0.42;

    // Fixed parameter ranges exactly matching playground
    const paramRanges = {
      Rs: { min: 10, max: 10000 },
      Ra: { min: 10, max: 10000 },
      Rb: { min: 10, max: 10000 },
      Ca: { min: 0.1e-6, max: 50e-6 },
      Cb: { min: 0.1e-6, max: 50e-6 }
    };

    // Group polygons by color for efficient batching
    const colorGroups = new Map<string, ModelSnapshot[]>();
    
    models.forEach(model => {
      const color = model.color || '#3B82F6';
      if (!colorGroups.has(color)) {
        colorGroups.set(color, []);
      }
      colorGroups.get(color)!.push(model);
    });

    // Draw each color group in batches
    colorGroups.forEach((groupModels, color) => {
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = 1;

      // Batch draw all polygons of the same color
      this.ctx.beginPath();
      
      groupModels.forEach(model => {
        const points = this.calculatePolygonPoints(model, centerX, centerY, maxRadius, paramRanges);
        if (points.length >= 3) {
          const opacity = model.opacity || 0.7;
          this.ctx.globalAlpha = opacity * (this.renderConfig.opacityFactor || 1);
          
          // Draw polygon path
          this.ctx.moveTo(points[0].x, points[0].y);
          for (let i = 1; i < points.length; i++) {
            this.ctx.lineTo(points[i].x, points[i].y);
          }
          this.ctx.closePath();
        }
      });
      
      this.ctx.stroke();
    });

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

    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
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
}

export const SpiderPlot: React.FC<SpiderPlotProps> = ({
  meshItems,
  opacityFactor,
  maxPolygons,
  visualizationMode = 'color',
  opacityIntensity = 1.0,
  gridSize = 5,
  includeLabels = true,
  backgroundColor = 'transparent'
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

    // Apply polygon limit
    const effectivePolygons = meshItems.slice(0, maxPolygons);
    
    return effectivePolygons.map(model => {
      const resnorm = model?.resnorm || 0;
      let logOpacity: number;
      
      if (visualizationMode === 'opacity') {
        // Use logarithmic scaling for opacity mode
        const allResnorms = effectivePolygons.map(i => i.resnorm).filter(r => r !== undefined) as number[];
        
        if (resnorm !== undefined && allResnorms.length > 1) {
          // Use iterative approach to avoid stack overflow
          let minResnorm = Infinity;
          let maxResnorm = -Infinity;
          for (const r of allResnorms) {
            if (r < minResnorm) minResnorm = r;
            if (r > maxResnorm) maxResnorm = r;
          }
          
          // Enhanced logarithmic calculation with aggressive contrast (matching orchestrator)
          const epsilon = 1e-12;
          const safeMin = Math.max(minResnorm, epsilon);
          const safeMax = Math.max(maxResnorm, safeMin * 100); // Wider range for better contrast
          
          // Calculate log-spaced normalization
          const logMin = Math.log10(safeMin);
          const logMax = Math.log10(safeMax);
          const logRange = Math.max(logMax - logMin, 1e-10);
          
          const safeResnorm = Math.max(resnorm, safeMin);
          const logResnorm = Math.log10(safeResnorm);
          const normalizedLog = (logResnorm - logMin) / logRange;
          
          // Invert so lower resnorm (better fit) = higher opacity
          const inverted = 1 - Math.max(0, Math.min(1, normalizedLog));
          
          // Apply aggressive intensity factor with gamma correction
          const gamma = 1 / Math.max(opacityIntensity, 0.1); // Prevent division by zero
          const intensified = Math.pow(inverted, gamma);
          
          // Map to more aggressive opacity range: 0.05 to 1.0 for maximum contrast
          logOpacity = Math.max(0.05, Math.min(1.0, 0.05 + intensified * 0.95));
        } else {
          logOpacity = 0.5;
        }
      } else {
        // Color groups mode
        logOpacity = calculateLogOpacity(resnorm);
      }

      return {
        ...model,
        opacity: logOpacity * (opacityFactor || 1),
        color: visualizationMode === 'opacity' ? '#3B82F6' : model.color
      };
    });
  }, [meshItems, maxPolygons, visualizationMode, opacityIntensity, opacityFactor]);

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
        opacityFactor: opacityFactor || 1
      });
    }

    return () => {
      if (rendererRef.current) {
        rendererRef.current.destroy();
        rendererRef.current = null;
      }
    };
  }, [isCanvasMode, backgroundColor, gridSize, includeLabels, opacityFactor]);

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
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
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
          {/* Concentric pentagons */}
          {Array.from({ length: gridSize }, (_, level) => {
            const radius = (maxRadius * (level + 1)) / gridSize;
            const points = params.map((_, i) => {
              const angle = i * angleStep - Math.PI / 2;
              const x = centerX + Math.cos(angle) * radius;
              const y = centerY + Math.sin(angle) * radius;
              return `${x},${y}`;
            }).join(' ');
            return <polygon key={level} points={points} />;
          })}
          
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
          
          const ranges = {
            Rs: '10Ω - 10kΩ',
            Ra: '10Ω - 10kΩ', 
            Rb: '10Ω - 10kΩ',
            Ca: '0.1 - 50µF',
            Cb: '0.1 - 50µF'
          };
          
          return (
            <g key={param}>
              <text
                x={labelX}
                y={labelY - 8}
                textAnchor="middle"
                fill="#F3F4F6"
                fontSize="14"
                fontWeight="700"
                fontFamily="Inter, sans-serif"
              >
                {param}
              </text>
              <text
                x={labelX}
                y={labelY + 8}
                textAnchor="middle"
                fill="#D1D5DB"
                fontSize="10"
                fontWeight="500"
                fontFamily="Inter, sans-serif"
              >
                {ranges[param as keyof typeof ranges]}
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

// Helper function for opacity calculation
function calculateLogOpacity(resnorm: number): number {
  if (resnorm <= 0) return 0.5;
  
  // Simple logarithmic mapping
  const logResnorm = Math.log10(resnorm);
  const normalized = Math.max(0, Math.min(1, (logResnorm + 5) / 10));
  return Math.pow(1 - normalized, 0.5);
}
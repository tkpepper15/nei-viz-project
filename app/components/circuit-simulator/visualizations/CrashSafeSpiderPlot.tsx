"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ModelSnapshot } from '../utils/types';
import { PerformanceMonitor } from '../utils/performanceMonitor';
import { PerformanceDashboard } from '../PerformanceDashboard';
import { PerformanceSettings } from '../controls/PerformanceControls';

interface CrashSafeSpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string | null;
  opacityFactor: number;
  maxPolygons: number;
  visualizationMode?: 'color' | 'opacity';
  opacityIntensity?: number;
  gridSize?: number;
  performanceSettings?: PerformanceSettings;
}

interface SafetyLimits {
  maxPolygons: number;
  maxMemoryMB: number;
  maxRenderTime: number;
  qualityLevel: 'low' | 'medium' | 'high';
}

interface RenderState {
  isRendering: boolean;
  currentStage: string;
  progress: number;
  processedPolygons: number;
  totalPolygons: number;
  memoryUsage: number;
  crashRisk: 'low' | 'medium' | 'high';
  renderStartTime: number;
  estimatedTimeRemaining: number;
}

export const CrashSafeSpiderPlot: React.FC<CrashSafeSpiderPlotProps> = ({
  meshItems,
  referenceId,
  opacityFactor,
  maxPolygons,
  visualizationMode = 'color',
  opacityIntensity = 1.0,
  gridSize = 5,
  performanceSettings
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [performanceMonitor] = useState(() => new PerformanceMonitor());
  const [showDashboard, setShowDashboard] = useState(false);
  const [finalImage, setFinalImage] = useState<HTMLCanvasElement | null>(null);
  const [isClient, setIsClient] = useState(false);
  
  const [renderState, setRenderState] = useState<RenderState>({
    isRendering: false,
    currentStage: 'idle',
    progress: 0,
    processedPolygons: 0,
    totalPolygons: 0,
    memoryUsage: 0,
    crashRisk: 'low',
    renderStartTime: 0,
    estimatedTimeRemaining: 0
  });

  const [emergencyStop, setEmergencyStop] = useState<boolean>(false);
  const [emergencyMessage, setEmergencyMessage] = useState<string>('');

  // Set client flag after hydration
  useEffect(() => {
    setIsClient(true);
  }, []);

  // Calculate safe limits based on performance settings or fallback to grid size
  const safetyLimits = useMemo((): SafetyLimits => {
    if (performanceSettings) {
      // Use custom performance settings
      console.log(`Using custom performance settings: ${performanceSettings.maxMemoryMB}MB limit, ${performanceSettings.chunkSize} chunk size`);
      return {
        maxPolygons: Math.min(performanceSettings.maxPolygons, maxPolygons),
        maxMemoryMB: performanceSettings.maxMemoryMB,
        maxRenderTime: performanceSettings.maxRenderTimeMs,
        qualityLevel: performanceSettings.maxPolygons > 5000 ? 'high' : 
                     performanceSettings.maxPolygons > 2000 ? 'medium' : 'low'
      };
    }
    
    // Fallback to grid-based auto calculation
    const totalGridPoints = Math.pow(gridSize, 5);
    const currentMemory = isClient ? getCurrentMemoryUsage() : 0;
    
          console.log(`Grid analysis: ${gridSize}^5 = ${totalGridPoints.toLocaleString()} potential points | Current memory: ${currentMemory.toFixed(0)}MB`);
    
    // Ultra-aggressive limits to prevent crashes - much lower memory limits
    if (gridSize >= 8 || totalGridPoints > 32768 || currentMemory > 50) {
      console.warn(`Very high grid size detected (${gridSize}). Applying ultra-conservative safety limits.`);
      return {
        maxPolygons: Math.min(500, maxPolygons),
        maxMemoryMB: 30,
        maxRenderTime: 3000,
        qualityLevel: 'low'
      };
    } else if (gridSize >= 6 || totalGridPoints > 7776 || currentMemory > 30) {
      console.warn(`High grid size detected (${gridSize}). Applying conservative safety limits.`);
      return {
        maxPolygons: Math.min(1000, maxPolygons),
        maxMemoryMB: 40,
        maxRenderTime: 5000,
        qualityLevel: 'low'
      };
    } else if (gridSize >= 4 || totalGridPoints > 1024) {
      return {
        maxPolygons: Math.min(2000, maxPolygons),
        maxMemoryMB: 60,
        maxRenderTime: 10000,
        qualityLevel: 'medium'
      };
    } else {
      return {
        maxPolygons: Math.min(5000, maxPolygons),
        maxMemoryMB: 80,
        maxRenderTime: 20000,
        qualityLevel: 'high'
      };
    }
  }, [gridSize, maxPolygons, isClient, performanceSettings]);

  // Safe polygon processing with chunking
  const processPolygonsInChunks = useCallback(async (
    polygons: ModelSnapshot[], 
    canvas: HTMLCanvasElement
  ): Promise<boolean> => {
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      console.error('Cannot get canvas context');
      return false;
    }

    // Configurable pre-emptive memory check before starting
    if (performanceSettings?.preRenderMemoryCheck !== false) {
      const initialMemory = getCurrentMemoryUsage();
      const preRenderThreshold = safetyLimits.maxMemoryMB * 0.5; // 50% limit
      if (initialMemory > preRenderThreshold) {
        const errorMsg = `Cannot start render: Initial memory ${initialMemory.toFixed(0)}MB > ${preRenderThreshold.toFixed(0)}MB (50% limit). Try refreshing the page.`;
        console.error(`${errorMsg}`);
        setEmergencyStop(true);
        setEmergencyMessage(errorMsg);
        return false;
      }
    }

    // Force GC before starting
    await requestGarbageCollection();
    
    const sessionId = `render-${Date.now()}`;
    performanceMonitor.startSession(sessionId, polygons.length);

    try {
      setRenderState(prev => ({
        ...prev,
        isRendering: true,
        currentStage: 'preparing',
        totalPolygons: polygons.length,
        renderStartTime: Date.now()
      }));

      // Setup canvas with configurable size
      const setupStart = Date.now();
      canvas.width = performanceSettings?.canvasWidth || 800;
      canvas.height = performanceSettings?.canvasHeight || 800;
      
      // Dark background
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Configurable quality rendering
      const enableHighQuality = performanceSettings?.enableHighQualityRendering !== false;
      ctx.imageSmoothingEnabled = enableHighQuality;
      if (enableHighQuality) {
        ctx.imageSmoothingQuality = 'high';
      }
      
      performanceMonitor.logStage('canvas_setup', Date.now() - setupStart);

      // Draw spider grid
      const gridStart = Date.now();
      drawSpiderGrid(ctx, canvas.width, canvas.height);
      performanceMonitor.logStage('grid_render', Date.now() - gridStart);

      setRenderState(prev => ({ ...prev, currentStage: 'rendering_polygons' }));

      // Process polygons in safe chunks with configurable memory management
      const chunkSize = performanceSettings ? performanceSettings.chunkSize : getOptimalChunkSize(polygons.length, safetyLimits);
      const gcFrequency = performanceSettings ? performanceSettings.gcFrequency : 2;
      let processedCount = 0;

      console.log(`Processing ${polygons.length.toLocaleString()} polygons in chunks of ${chunkSize} | Memory limit: ${safetyLimits.maxMemoryMB}MB`);

      for (let i = 0; i < polygons.length; i += chunkSize) {
        // Check memory and time limits more frequently
        const currentMemory = getCurrentMemoryUsage();
        const elapsedTime = Date.now() - renderState.renderStartTime;
        
        // Force garbage collection based on settings
        if (i > 0 && (i / chunkSize) % gcFrequency === 0) {
          await requestGarbageCollection();
          console.log(`Preventive GC at chunk ${Math.floor(i/chunkSize)} | Memory: ${getCurrentMemoryUsage().toFixed(0)}MB`);
        }
        
        // Configurable memory checking thresholds
        const warningThreshold = performanceSettings ? 
          (safetyLimits.maxMemoryMB * performanceSettings.memoryWarningThreshold / 100) :
          (safetyLimits.maxMemoryMB * 0.6);
        
        const emergencyThreshold = performanceSettings ?
          (safetyLimits.maxMemoryMB * performanceSettings.memoryEmergencyThreshold / 100) :
          (safetyLimits.maxMemoryMB * 0.75);

        if (currentMemory > warningThreshold) {
          console.warn(`Memory warning: ${currentMemory.toFixed(0)}MB > ${warningThreshold.toFixed(0)}MB (${performanceSettings?.memoryWarningThreshold || 60}% limit)`);
          await requestGarbageCollection();
          
          const memoryAfterGC = getCurrentMemoryUsage();
          if (memoryAfterGC > emergencyThreshold) {
            const errorMsg = `Memory critical: ${memoryAfterGC.toFixed(0)}MB > ${emergencyThreshold.toFixed(0)}MB. Stopped to prevent crash.`;
            console.error(`${errorMsg}`);
            setEmergencyStop(true);
            setEmergencyMessage(errorMsg);
            break;
          }
        }

        if (elapsedTime > safetyLimits.maxRenderTime) {
          console.warn(`Time limit exceeded: ${elapsedTime}ms > ${safetyLimits.maxRenderTime}ms`);
          break;
        }

        // Process chunk
        const chunkStart = Date.now();
        const chunk = polygons.slice(i, Math.min(i + chunkSize, polygons.length));
        
        let chunkProcessed = 0;
        for (const polygon of chunk) {
          if (renderSpiderPolygon(ctx, polygon, canvas.width, canvas.height)) {
            chunkProcessed++;
          }
          
          // Configurable mid-chunk memory checks
          if (performanceSettings?.midChunkMemoryChecks !== false) {
            const checkFrequency = performanceSettings?.memoryCheckFrequency || 3;
            if (chunkProcessed % checkFrequency === 0) {
              const midChunkMemory = getCurrentMemoryUsage();
              const midChunkThreshold = performanceSettings ?
                (safetyLimits.maxMemoryMB * 0.7) : // Default to 70% for mid-chunk
                (safetyLimits.maxMemoryMB * 0.7);
              
              if (midChunkMemory > midChunkThreshold) {
                console.warn(`Mid-chunk memory spike: ${midChunkMemory.toFixed(0)}MB`);
                await requestGarbageCollection();
                
                const memoryAfterMidGC = getCurrentMemoryUsage();
                const midChunkEmergencyThreshold = performanceSettings ?
                  (safetyLimits.maxMemoryMB * (performanceSettings.memoryEmergencyThreshold - 5) / 100) : // 5% below emergency
                  (safetyLimits.maxMemoryMB * 0.8);
                
                if (memoryAfterMidGC > midChunkEmergencyThreshold) {
                  const errorMsg = `Memory spike during render: ${memoryAfterMidGC.toFixed(0)}MB > ${midChunkEmergencyThreshold.toFixed(0)}MB. Emergency stop.`;
                  console.error(`${errorMsg}`);
                  setEmergencyStop(true);
                  setEmergencyMessage(errorMsg);
                  return false;
                }
              }
            }
          }
        }

        processedCount += chunkProcessed;
        const chunkTime = Date.now() - chunkStart;
        
        // Update progress
        const progress = processedCount / polygons.length;
        const avgTimePerPolygon = elapsedTime / processedCount;
        const estimatedRemaining = (polygons.length - processedCount) * avgTimePerPolygon;

        setRenderState(prev => ({
          ...prev,
          progress,
          processedPolygons: processedCount,
          memoryUsage: currentMemory,
          estimatedTimeRemaining: estimatedRemaining,
          crashRisk: calculateCrashRisk(currentMemory, elapsedTime, safetyLimits)
        }));

        console.log(`Chunk ${Math.floor(i/chunkSize) + 1}: ${chunkProcessed}/${chunk.length} polygons in ${chunkTime.toFixed(0)}ms | Memory: ${currentMemory.toFixed(0)}MB | Progress: ${(progress * 100).toFixed(1)}%`);

        // Configurable yield to browser after each chunk
        const yieldTime = performanceSettings?.yieldTimeMs || Math.min(20, chunkTime / 2);
        await new Promise(resolve => setTimeout(resolve, yieldTime));
        
        // Additional yield if memory is getting high (unless disabled)
        if (currentMemory > warningThreshold && yieldTime > 0) {
          await new Promise(resolve => setTimeout(resolve, Math.max(yieldTime, 50)));
        }

        performanceMonitor.logStage('polygon_chunk', chunkTime, {
          chunkSize: chunk.length,
          processedCount: chunkProcessed,
          memoryUsage: currentMemory
        });
      }

      setRenderState(prev => ({ ...prev, currentStage: 'finalizing' }));
      
      // Final drawing operations
      const finalizationStart = Date.now();
      if (referenceId) {
        highlightReferenceModel(ctx, polygons, referenceId, canvas.width, canvas.height);
      }
      
      performanceMonitor.logStage('finalization', Date.now() - finalizationStart);

      console.log(`Render complete: ${processedCount}/${polygons.length} polygons (${(processedCount/polygons.length*100).toFixed(1)}%)`);
      
      setFinalImage(canvas);
      return true;

    } catch (error) {
      console.error('Render failed:', error);
      return false;
    } finally {
      performanceMonitor.endSession();
      setRenderState(prev => ({ ...prev, isRendering: false, currentStage: 'complete' }));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [safetyLimits, referenceId, performanceMonitor]);

  // Get optimal chunk size based on memory and performance - ultra-small chunks
  function getOptimalChunkSize(totalPolygons: number, limits: SafetyLimits): number {
    // Use ultra-small chunks to prevent memory spikes
    if (limits.qualityLevel === 'low') return Math.min(5, Math.max(1, Math.floor(totalPolygons / 200)));
    if (limits.qualityLevel === 'medium') return Math.min(10, Math.max(1, Math.floor(totalPolygons / 100)));
    return Math.min(20, Math.max(1, Math.floor(totalPolygons / 50)));
  }

  // Get current memory usage in MB
  function getCurrentMemoryUsage(): number {
    if (typeof window !== 'undefined' && 'performance' in window && 'memory' in performance) {
      const memory = (performance as typeof performance & { memory: { usedJSHeapSize: number } }).memory;
      return memory.usedJSHeapSize / (1024 * 1024);
    }
    return 0;
  }

  // Request garbage collection
  async function requestGarbageCollection(): Promise<void> {
    if (typeof window !== 'undefined' && 'gc' in window) {
      const windowWithGc = window as typeof window & { gc: () => void };
      windowWithGc.gc();
      console.log('Forced garbage collection');
    }
    // Wait a moment for GC to complete
    await new Promise(resolve => setTimeout(resolve, 100));
  }

  // Calculate crash risk
  function calculateCrashRisk(memoryMB: number, elapsedTime: number, limits: SafetyLimits): 'low' | 'medium' | 'high' {
    const memoryRatio = memoryMB / limits.maxMemoryMB;
    const timeRatio = elapsedTime / limits.maxRenderTime;
    
    if (memoryRatio > 0.9 || timeRatio > 0.9) return 'high';
    if (memoryRatio > 0.7 || timeRatio > 0.7) return 'medium';
    return 'low';
  }

  // Draw spider plot grid
  function drawSpiderGrid(ctx: CanvasRenderingContext2D, width: number, height: number): void {
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 50;

    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 0.5;

    // Draw concentric circles
    for (let i = 1; i <= 5; i++) {
      const radius = (maxRadius * i) / 5;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
      ctx.stroke();
    }

    // Draw radial axes (5 axes for 5 parameters)
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    ctx.font = '12px Inter';
    ctx.fillStyle = '#E5E7EB';

    for (let i = 0; i < params.length; i++) {
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;
      const endX = centerX + Math.cos(angle) * maxRadius;
      const endY = centerY + Math.sin(angle) * maxRadius;

      // Draw axis line
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();

      // Draw parameter label
      const labelX = centerX + Math.cos(angle) * (maxRadius + 20);
      const labelY = centerY + Math.sin(angle) * (maxRadius + 20);
      ctx.fillText(params[i], labelX - 10, labelY + 5);
    }
  }

  // Render a single spider polygon
  function renderSpiderPolygon(
    ctx: CanvasRenderingContext2D, 
    model: ModelSnapshot, 
    width: number, 
    height: number
  ): boolean {
    if (!model.parameters) return false;

    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) / 2 - 50;

    // Calculate polygon points
    const points = [];
    for (let i = 0; i < params.length; i++) {
      const param = params[i];
      const value = model.parameters[param as keyof typeof model.parameters];
      
      if (typeof value !== 'number') continue;

      // Normalize parameter value
      let normalizedValue: number;
      if (param.includes('C')) {
        // Capacitance: 0.1µF to 50µF
        normalizedValue = Math.log10(value * 1e6 / 0.1) / Math.log10(500);
      } else {
        // Resistance: 10Ω to 10kΩ
        normalizedValue = Math.log10(value / 10) / Math.log10(1000);
      }

      normalizedValue = Math.max(0, Math.min(1, normalizedValue));
      const radius = normalizedValue * maxRadius;
      const angle = (i * 2 * Math.PI) / params.length - Math.PI / 2;

      points.push({
        x: centerX + Math.cos(angle) * radius,
        y: centerY + Math.sin(angle) * radius
      });
    }

    if (points.length < 3) return false;

    // Apply visualization mode
    let color: string;
    let opacity: number;

    if (visualizationMode === 'opacity') {
      color = '#3B82F6';
      opacity = calculateOpacity(model.resnorm || 0) * opacityFactor;
    } else {
      color = model.color || '#3B82F6';
      opacity = (model.opacity || 0.7) * opacityFactor;
    }

    // Draw polygon
    ctx.strokeStyle = color;
    ctx.lineWidth = model.id === referenceId ? 2 : 1;
    ctx.globalAlpha = opacity;

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.closePath();
    ctx.stroke();

    ctx.globalAlpha = 1;
    return true;
  }

  // Calculate opacity based on resnorm
  function calculateOpacity(resnorm: number): number {
    if (resnorm <= 0) return 0.5;
    
    // Simple logarithmic mapping
    const logResnorm = Math.log10(resnorm);
    const normalized = Math.max(0, Math.min(1, (logResnorm + 5) / 10)); // Adjust range as needed
    return Math.pow(1 - normalized, 1 / opacityIntensity);
  }

  // Highlight reference model
  function highlightReferenceModel(
    ctx: CanvasRenderingContext2D,
    polygons: ModelSnapshot[],
    refId: string,
    width: number,
    height: number
  ): void {
    const refModel = polygons.find(p => p.id === refId);
    if (refModel) {
      ctx.save();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 3;
      ctx.globalAlpha = 1;
      renderSpiderPolygon(ctx, refModel, width, height);
      ctx.restore();
    }
  }

  // Filter and prepare polygons for safe rendering
  const preparedPolygons = useMemo(() => {
    if (!meshItems?.length) return [];

    // Apply safety limits
    const limitedItems = meshItems
      .filter(item => item?.parameters && item.resnorm !== undefined)
      .slice(0, safetyLimits.maxPolygons);

    console.log(`Polygon preparation: ${meshItems.length.toLocaleString()} → ${limitedItems.length.toLocaleString()} (${safetyLimits.qualityLevel} quality)`);

    return limitedItems;
  }, [meshItems, safetyLimits]);

  // Start rendering when polygons change, but only if safe
  useEffect(() => {
    if (emergencyStop) return; // Don't render if in emergency state
    
    if (preparedPolygons.length > 0 && canvasRef.current && !renderState.isRendering) {
      // Ultra-conservative check: if grid size is very high, show message instead of rendering
      if (gridSize >= 10 || preparedPolygons.length > 5000) {
        const message = `Grid size too large (${gridSize}^5 = ${Math.pow(gridSize, 5).toLocaleString()} points). Use grid size ≤ 9 for visualization.`;
        console.warn(`${message}`);
        setEmergencyStop(true);
        setEmergencyMessage(message);
        return;
      }
      
      console.log(`Starting safe render: ${preparedPolygons.length.toLocaleString()} polygons | Quality: ${safetyLimits.qualityLevel} | Memory limit: ${safetyLimits.maxMemoryMB}MB`);
      
      const timer = setTimeout(() => {
        processPolygonsInChunks(preparedPolygons, canvasRef.current!);
      }, 100);

      return () => clearTimeout(timer);
    }
  }, [preparedPolygons, processPolygonsInChunks, renderState.isRendering, safetyLimits, emergencyStop, gridSize]);

  const formatTime = (ms: number) => {
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    const seconds = ms / 1000;
    return seconds < 60 ? `${seconds.toFixed(1)}s` : `${Math.floor(seconds / 60)}m ${(seconds % 60).toFixed(0)}s`;
  };

  const getCrashRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  // Show loading state during hydration
  if (!isClient) {
    return (
      <div className="relative w-full h-full bg-slate-900 rounded-lg overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-neutral-400 text-lg">Initializing...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-lg overflow-hidden">
      {/* Safety Status Panel */}
      <div className="absolute top-4 left-4 right-4 z-10 bg-neutral-800/95 backdrop-blur-sm border border-neutral-700 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="text-sm">
              <span className="text-neutral-400">Grid:</span>
              <span className="text-white ml-2">{gridSize}^5 = {Math.pow(gridSize, 5).toLocaleString()}</span>
            </div>
            <div className="text-sm">
              <span className="text-neutral-400">Polygons:</span>
              <span className="text-white ml-2">{preparedPolygons.length.toLocaleString()}</span>
            </div>
            <div className="text-sm">
              <span className="text-neutral-400">Quality:</span>
              <span className="text-white ml-2 capitalize">{safetyLimits.qualityLevel}</span>
            </div>
            <div className="text-sm">
              <span className="text-neutral-400">Crash Risk:</span>
              <span className={`ml-2 font-medium ${getCrashRiskColor(renderState.crashRisk)}`}>
                {renderState.crashRisk.toUpperCase()}
              </span>
            </div>
          </div>

          <button
            onClick={() => setShowDashboard(!showDashboard)}
            className="px-3 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded"
          >
            {showDashboard ? 'Hide' : 'Show'} Performance
          </button>
        </div>

        {/* Progress Bar */}
        {renderState.isRendering && (
          <div className="mt-3">
            <div className="flex items-center justify-between text-xs text-neutral-300 mb-1">
              <span className="capitalize">{renderState.currentStage.replace('_', ' ')}</span>
              <span>
                {renderState.processedPolygons.toLocaleString()}/{renderState.totalPolygons.toLocaleString()} 
                ({Math.round(renderState.progress * 100)}%)
              </span>
            </div>
            
            <div className="w-full bg-neutral-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-primary to-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${renderState.progress * 100}%` }}
              />
            </div>
            
            {renderState.estimatedTimeRemaining > 0 && (
              <div className="text-xs text-neutral-400 mt-1 flex justify-between">
                <span>Memory: {renderState.memoryUsage.toFixed(0)}MB</span>
                <span>~{formatTime(renderState.estimatedTimeRemaining)} remaining</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Canvas */}
      <div className="w-full h-full pt-24 flex items-center justify-center">
        <canvas
          ref={canvasRef}
          className="max-w-full max-h-full border border-neutral-700 rounded shadow-lg"
          style={{ 
            maxWidth: '90%',
            maxHeight: '90%'
          }}
        />
        
        {emergencyStop ? (
          <div className="absolute inset-0 flex items-center justify-center text-center text-red-400">
            <div className="bg-red-900/20 border border-red-600 rounded-lg p-6 max-w-md">
                              <div className="text-lg mb-2 font-semibold">Emergency Stop</div>
              <div className="text-sm mb-4">{emergencyMessage}</div>
              <button
                onClick={() => {
                  setEmergencyStop(false);
                  setEmergencyMessage('');
                  window.location.reload();
                }}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded text-sm"
              >
                Reload Page
              </button>
            </div>
          </div>
        ) : !renderState.isRendering && !finalImage ? (
          <div className="absolute inset-0 flex items-center justify-center text-center text-neutral-400">
            <div>
              <div className="text-lg mb-2">Ready to Render</div>
              <div className="text-sm">
                {preparedPolygons.length.toLocaleString()} polygons prepared for safe rendering
              </div>
            </div>
          </div>
        ) : null}
      </div>

      {/* Performance Dashboard */}
      <PerformanceDashboard
        performanceMonitor={performanceMonitor}
        isVisible={showDashboard}
        onToggle={() => setShowDashboard(!showDashboard)}
      />
    </div>
  );
}; 
"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ModelSnapshot } from '../utils/types';
import { createTileRenderingHook, RenderConfig, SpiderConfig } from '../utils/tileRenderer';

interface TiledSpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string | null;
  opacityFactor: number;
  maxPolygons: number;
  onExportImage?: (canvas: HTMLCanvasElement) => void;
  visualizationMode?: 'color' | 'opacity';
  opacityIntensity?: number;
  renderQuality?: 'draft' | 'preview' | 'final' | 'ultra';
}

interface RenderProgress {
  isRendering: boolean;
  progress: number;
  completedTiles: number;
  totalTiles: number;
  currentStage: 'preparing' | 'rendering' | 'assembling' | 'complete';
  estimatedTimeRemaining?: number;
  renderStartTime?: number;
}

interface TilePreview {
  tileId: string;
  canvas: HTMLCanvasElement;
  x: number;
  y: number;
  width: number;
  height: number;
}

export const TiledSpiderPlot: React.FC<TiledSpiderPlotProps> = ({
  meshItems,
  referenceId,
  opacityFactor,
  maxPolygons,
  onExportImage,
  visualizationMode = 'color',
  opacityIntensity = 1.0,
  renderQuality = 'preview'
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const finalCanvasRef = useRef<HTMLCanvasElement>(null);
  
  const [renderProgress, setRenderProgress] = useState<RenderProgress>({
    isRendering: false,
    progress: 0,
    completedTiles: 0,
    totalTiles: 0,
    currentStage: 'preparing'
  });

  const [tilePreviewsMode, setTilePreviewsMode] = useState(false);
  const [tilePreviewList, setTilePreviewList] = useState<TilePreview[]>([]);
  const [finalImage, setFinalImage] = useState<HTMLCanvasElement | null>(null);

  // Initialize tile renderer
  const { initRenderer, getRenderer, destroyRenderer } = createTileRenderingHook();

  // Quality settings that affect tile size and performance
  const qualitySettings = useMemo(() => ({
    draft: { 
      canvasSize: 1024, 
      tileSize: 512, 
      maxPolygons: 5000,
      workers: 2
    },
    preview: { 
      canvasSize: 2048, 
      tileSize: 512, 
      maxPolygons: 15000,
      workers: 4
    },
    final: { 
      canvasSize: 4096, 
      tileSize: 512, 
      maxPolygons: 50000,
      workers: 6
    },
    ultra: { 
      canvasSize: 8192, 
      tileSize: 1024, 
      maxPolygons: 100000,
      workers: 8
    }
  }), []);

  // Current render configuration
  const currentConfig = useMemo(() => qualitySettings[renderQuality], [renderQuality, qualitySettings]);

  // Spider plot configuration
  const spiderConfig: SpiderConfig = useMemo(() => ({
    viewBoxSize: currentConfig.canvasSize,
    centerX: currentConfig.canvasSize / 2,
    centerY: currentConfig.canvasSize / 2,
    maxRadius: currentConfig.canvasSize / 2 - 100
  }), [currentConfig]);

  // Render configuration
  const renderConfig: RenderConfig = useMemo(() => ({
    backgroundColor: '#0f172a',
    gridColor: '#4B5563',
    gridLineWidth: visualizationMode === 'opacity' ? 0.8 : 0.5,
    defaultColor: visualizationMode === 'opacity' ? '#3B82F6' : '#6B7280',
    defaultStrokeWidth: 1,
    defaultOpacity: 0.7,
    spiderConfig
  }), [spiderConfig, visualizationMode]);

  // Process and filter mesh items
  const processedMeshItems = useMemo(() => {
    if (!meshItems?.length) return [];

    // Apply polygon limit and filtering
    const validItems = meshItems
      .filter(item => item?.parameters && item.resnorm !== undefined)
      .slice(0, Math.min(maxPolygons, currentConfig.maxPolygons));

    // Calculate dynamic opacity if in opacity mode
    if (visualizationMode === 'opacity') {
      const resnorms = validItems.map(item => item.resnorm!).filter(r => isFinite(r));
      if (resnorms.length > 0) {
        const minResnorm = Math.min(...resnorms);
        const maxResnorm = Math.max(...resnorms);
        const logMin = Math.log10(Math.max(minResnorm, 1e-10));
        const logMax = Math.log10(maxResnorm);

        return validItems.map(item => {
          const logResnorm = Math.log10(Math.max(item.resnorm!, 1e-10));
          const normalized = 1 - ((logResnorm - logMin) / (logMax - logMin));
          const intensified = Math.pow(normalized, 1 / opacityIntensity);
          const opacity = Math.max(0.05, Math.min(1.0, intensified * opacityFactor));

          return {
            ...item,
            opacity,
            color: '#3B82F6', // Single color for opacity mode
            isReference: item.id === referenceId
          };
        });
      }
    }

    // Color mode - use existing colors with calculated opacity
    return validItems.map(item => ({
      ...item,
      opacity: (item.opacity || 0.7) * opacityFactor,
      isReference: item.id === referenceId
    }));
  }, [meshItems, maxPolygons, currentConfig.maxPolygons, visualizationMode, opacityIntensity, opacityFactor, referenceId]);

  // Start tile-based rendering
  const startTileRendering = useCallback(async () => {
    if (!processedMeshItems.length) return;

    setRenderProgress({
      isRendering: true,
      progress: 0,
      completedTiles: 0,
      totalTiles: 0,
      currentStage: 'preparing',
      renderStartTime: Date.now()
    });

    try {
      // Initialize renderer
      const renderer = initRenderer(currentConfig.workers);
      if (!renderer) throw new Error('Failed to initialize tile renderer');

      // Calculate tile configuration
      const tileConfig = renderer.calculateTileConfiguration(
        currentConfig.canvasSize,
        currentConfig.canvasSize,
        currentConfig.tileSize
      );

      console.log(`Rendering ${processedMeshItems.length} polygons in ${tileConfig.tilesX}x${tileConfig.tilesY} tiles`);

      // Generate tile jobs
      const tileJobs = renderer.generateTileJobs(
        tileConfig,
        processedMeshItems,
        spiderConfig
      );

      setRenderProgress(prev => ({
        ...prev,
        totalTiles: tileJobs.length,
        currentStage: 'rendering'
      }));

      // Clear previous previews
      setTilePreviewList([]);

      // Start rendering with progress callbacks
      await renderer.renderTiles(
        tileJobs,
        renderConfig,
        // Progress callback
        (progress, completed, total) => {
          const elapsed = Date.now() - (renderProgress.renderStartTime || Date.now());
          const estimatedTotal = elapsed / progress;
          const remaining = Math.max(0, estimatedTotal - elapsed);

          setRenderProgress(prev => ({
            ...prev,
            progress,
            completedTiles: completed,
            totalTiles: total,
            estimatedTimeRemaining: remaining
          }));
        },
        // Tile complete callback
        (tileResult) => {
          if (tilePreviewsMode) {
            // Find the corresponding job for positioning
            const job = tileJobs.find(j => j.tileId === tileResult.tileId);
            if (job) {
              setTilePreviewList(prev => [...prev, {
                tileId: tileResult.tileId,
                canvas: tileResult.canvas,
                x: job.x,
                y: job.y,
                width: job.width,
                height: job.height
              }]);
            }
          }
        },
        // Complete callback
        (canvas) => {
          setRenderProgress(prev => ({
            ...prev,
            currentStage: 'complete',
            isRendering: false,
            progress: 1
          }));

          setFinalImage(canvas);
          
          if (onExportImage) {
            onExportImage(canvas);
          }
        }
      );

    } catch (error) {
      console.error('Tile rendering failed:', error);
      setRenderProgress(prev => ({
        ...prev,
        isRendering: false,
        currentStage: 'complete'
      }));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [processedMeshItems, currentConfig, spiderConfig, renderConfig, tilePreviewsMode, onExportImage]);

  // Cancel rendering
  const cancelRendering = useCallback(() => {
    const renderer = getRenderer();
    if (renderer) {
      renderer.cancelRendering();
    }
    
    setRenderProgress(prev => ({
      ...prev,
      isRendering: false,
      currentStage: 'complete'
    }));
  }, [getRenderer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      destroyRenderer();
    };
  }, [destroyRenderer]);

  // Auto-start rendering when items change (debounced)
  useEffect(() => {
    const timer = setTimeout(() => {
      if (processedMeshItems.length > 0 && !renderProgress.isRendering) {
        startTileRendering();
      }
    }, 500);

    return () => clearTimeout(timer);
  }, [processedMeshItems, startTileRendering, renderProgress.isRendering]);

  // Render final image to canvas
  useEffect(() => {
    if (finalImage && finalCanvasRef.current) {
      const ctx = finalCanvasRef.current.getContext('2d');
      if (ctx) {
        finalCanvasRef.current.width = finalImage.width;
        finalCanvasRef.current.height = finalImage.height;
        ctx.drawImage(finalImage, 0, 0);
      }
    }
  }, [finalImage]);

  const formatTime = (ms: number) => {
    const seconds = Math.floor(ms / 1000);
    return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
  };

  return (
    <div className="relative w-full h-full bg-slate-900 rounded-lg overflow-hidden" ref={containerRef}>
      {/* Control Panel */}
      <div className="absolute top-4 left-4 right-4 z-10 bg-neutral-800/95 backdrop-blur-sm border border-neutral-700 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Quality Selector */}
            <div className="flex items-center gap-2">
              <label className="text-xs text-neutral-300">Quality:</label>
              <select 
                value={renderQuality}
                onChange={(e) => setRenderProgress(prev => ({ ...prev, renderQuality: e.target.value as 'draft' | 'preview' | 'final' | 'ultra' }))}
                className="px-2 py-1 text-xs bg-neutral-700 border border-neutral-600 rounded text-neutral-200"
                disabled={renderProgress.isRendering}
              >
                <option value="draft">Draft (1K)</option>
                <option value="preview">Preview (2K)</option>
                <option value="final">Final (4K)</option>
                <option value="ultra">Ultra (8K)</option>
              </select>
            </div>

            {/* Tile Preview Toggle */}
            <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="tilePreview"
                checked={tilePreviewsMode}
                onChange={(e) => setTilePreviewsMode(e.target.checked)}
                className="w-4 h-4"
                disabled={renderProgress.isRendering}
              />
              <label htmlFor="tilePreview" className="text-xs text-neutral-300">
                Show Tile Previews
              </label>
            </div>
          </div>

          {/* Stats and Controls */}
          <div className="flex items-center gap-4">
            <div className="text-xs text-neutral-400">
              {processedMeshItems.length.toLocaleString()} polygons
            </div>
            
            {renderProgress.isRendering ? (
              <button
                onClick={cancelRendering}
                className="px-3 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
              >
                Cancel
              </button>
            ) : (
              <button
                onClick={startTileRendering}
                disabled={processedMeshItems.length === 0}
                className="px-3 py-1 text-xs bg-primary hover:bg-primary-dark text-white rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Re-render
              </button>
            )}
          </div>
        </div>

        {/* Progress Bar */}
        {renderProgress.isRendering && (
          <div className="mt-3">
            <div className="flex items-center justify-between text-xs text-neutral-300 mb-1">
              <span className="capitalize">{renderProgress.currentStage}</span>
              <span>
                {renderProgress.completedTiles}/{renderProgress.totalTiles} tiles 
                ({Math.round(renderProgress.progress * 100)}%)
              </span>
            </div>
            
            <div className="w-full bg-neutral-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-primary to-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${renderProgress.progress * 100}%` }}
              />
            </div>
            
            {renderProgress.estimatedTimeRemaining && (
              <div className="text-xs text-neutral-400 mt-1">
                ~{formatTime(renderProgress.estimatedTimeRemaining)} remaining
              </div>
            )}
          </div>
        )}
      </div>

      {/* Rendering Area */}
      <div className="w-full h-full pt-20">
        {/* Tile Previews Mode */}
        {tilePreviewsMode && renderProgress.isRendering && (
          <div className="relative w-full h-full overflow-auto">
            <div 
              className="relative bg-slate-800"
              style={{ 
                width: currentConfig.canvasSize / 4, 
                height: currentConfig.canvasSize / 4,
                margin: '0 auto'
              }}
            >
              {tilePreviewList.map(tile => (
                <div
                  key={tile.tileId}
                  className="absolute border border-green-500/50"
                  style={{
                    left: tile.x / 4,
                    top: tile.y / 4,
                    width: tile.width / 4,
                    height: tile.height / 4
                  }}
                >
                  <canvas
                    width={tile.width / 4}
                    height={tile.height / 4}
                    className="w-full h-full"
                    ref={canvas => {
                      if (canvas) {
                        const ctx = canvas.getContext('2d');
                        if (ctx) {
                          ctx.drawImage(tile.canvas, 0, 0, tile.width / 4, tile.height / 4);
                        }
                      }
                    }}
                  />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Final Image Display */}
        {!tilePreviewsMode && (
          <div className="w-full h-full flex items-center justify-center">
            {finalImage ? (
              <canvas
                ref={finalCanvasRef}
                className="max-w-full max-h-full border border-neutral-700 rounded shadow-lg"
                style={{ 
                  imageRendering: 'pixelated',
                  maxWidth: '90%',
                  maxHeight: '90%'
                }}
              />
            ) : (
              <div className="text-center text-neutral-400">
                <div className="text-lg mb-2">
                  {renderProgress.isRendering ? 'Rendering...' : 'Ready to Render'}
                </div>
                <div className="text-sm">
                  {renderProgress.isRendering 
                    ? `Processing ${processedMeshItems.length.toLocaleString()} polygons`
                    : `${processedMeshItems.length.toLocaleString()} polygons ready`
                  }
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}; 
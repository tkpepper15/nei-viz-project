"use client";

import React, { useMemo, useEffect, useRef, useState, useCallback } from 'react';
import { ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';
import { ModelSnapshot, GridParameterArrays, RadarDataPoint } from '../utils/types';
import { generateGridValues } from '../utils/spider-utils';

interface SpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string | null;
  opacityFactor: number;
  maxPolygons: number;
  onGridValuesGenerated: (values: GridParameterArrays) => void;
  mode: 'interactive' | 'static';
  onExportSvg?: (svgString: string) => void;
  onExportPng?: (pngBlob: Blob) => void;
  visualizationMode?: 'color' | 'opacity';
  opacityIntensity?: number;
}

// Add new interfaces for chunked rendering
interface ChunkedRenderingState {
  isRendering: boolean;
  currentChunk: number;
  totalChunks: number;
  renderedPolygons: number;
  totalPolygons: number;
  renderStartTime: number;
  estimatedTimeRemaining: number;
}

interface RenderingControls {
  chunkSize: number;
  renderQuality: 'low' | 'medium' | 'high' | 'ultra';
  useCanvas: boolean;
  progressiveLoading: boolean;
  maxPolygonsPerFrame: number;
}

export const SpiderPlot: React.FC<SpiderPlotProps> = ({
  meshItems,
  referenceId,
  opacityFactor,
  maxPolygons,
  onGridValuesGenerated,
  mode,
  onExportSvg,
  onExportPng,
  visualizationMode = 'color',
  opacityIntensity = 1.0
}) => {
  const chartRef = useRef<HTMLDivElement>(null);

  // Generate default parameter structure for the spider plot axes
  const defaultSpiderData = useMemo(() => {
    const validParams = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'] as const;
    
    // Default parameter ranges for display
    const defaultRanges = {
      Rs: { min: 10, max: 10000, unit: 'Ω' },
      Ra: { min: 10, max: 10000, unit: 'Ω' },
      Rb: { min: 10, max: 10000, unit: 'Ω' },
      Ca: { min: 0.1, max: 50, unit: 'µF' },
      Cb: { min: 0.1, max: 50, unit: 'µF' }
    };

    return validParams.map(param => ({
      parameter: param,
      fullValue: defaultRanges[param].max,
      displayValue: `${defaultRanges[param].max}${defaultRanges[param].unit}`
    }));
  }, []);

  // Generate grid values only when meshItems change
  const gridValues = useMemo(() => {
    try {
      if (!meshItems || meshItems.length === 0) return null;
      
      // Ensure all required parameters are present
      const validItems = meshItems.filter(item => 
        item?.parameters && 
        typeof item.parameters.Rs === 'number' &&
        typeof item.parameters.Ra === 'number' &&
        typeof item.parameters.Ca === 'number' &&
        typeof item.parameters.Rb === 'number' &&
        typeof item.parameters.Cb === 'number'
      );
      
      if (validItems.length === 0) return null;
      return generateGridValues(validItems);
    } catch (error) {
      console.error('Error generating grid values:', error);
      return null;
    }
  }, [meshItems]);

  // Notify parent of grid values when meshItems change
  useEffect(() => {
    try {
      if (onGridValuesGenerated && gridValues) {
        onGridValuesGenerated(gridValues);
      }
    } catch (error) {
      console.error('Error in grid values callback:', error);
    }
  }, [gridValues, onGridValuesGenerated]);

  // Format parameter values with appropriate units and scale
  const formatParamValue = (param: string, value: number): string => {
    try {
      if (param.includes('C')) {
        // Convert capacitance from F to µF
        return `${(value * 1e6).toFixed(1)}µF`;
      } else {
        // Format resistance in Ω with appropriate scale
        if (value >= 1000) {
          return `${(value/1000).toFixed(1)}kΩ`;
        } else {
          return `${value.toFixed(1)}Ω`;
        }
      }
    } catch (error) {
      console.error('Error formatting param value:', error);
      return String(value);
    }
  };

  // Calculate more aggressive logarithmic opacity based on resnorm with intensity control
  const calculateLogOpacity = useMemo(() => {
    try {
      if (!meshItems?.length) return () => 0.7;
      
      // Get all valid resnorm values
      const resnorms = meshItems
        .map(item => item?.resnorm)
        .filter(r => typeof r === 'number' && r > 0 && isFinite(r)) as number[];
      
      if (resnorms.length === 0) return () => 0.7;
      
      // Use iterative approach to avoid stack overflow with large arrays
      let minResnorm = Infinity;
      let maxResnorm = -Infinity;
      for (const resnorm of resnorms) {
        if (resnorm < minResnorm) minResnorm = resnorm;
        if (resnorm > maxResnorm) maxResnorm = resnorm;
      }
      
      // Avoid log of zero and handle edge cases
      const safeMin = Math.max(minResnorm, 1e-10);
      const safeMax = Math.max(maxResnorm, safeMin * 10);
      const logMin = Math.log10(safeMin);
      const logMax = Math.log10(safeMax);
      const logRange = Math.max(logMax - logMin, 1e-10);
      
      return (resnorm: number) => {
        try {
          if (!resnorm || resnorm <= 0 || !isFinite(resnorm)) return 0.3;
          
          const safeResnorm = Math.max(resnorm, safeMin);
          const logResnorm = Math.log10(safeResnorm);
          const normalizedLog = (logResnorm - logMin) / logRange;
          
          // Invert so lower resnorm (better fit) = higher opacity
          const inverted = 1 - Math.max(0, Math.min(1, normalizedLog));
          
          // Apply intensity factor to make the curve more aggressive
          const intensified = Math.pow(inverted, 1 / opacityIntensity);
          
          // Map to opacity range 0.05 to 1.0 for better contrast
          return Math.max(0.05, Math.min(1.0, 0.05 + intensified * 0.95));
        } catch (error) {
          console.error('Error calculating opacity for resnorm:', resnorm, error);
          return 0.5;
        }
      };
    } catch (error) {
      console.error('Error in calculateLogOpacity:', error);
      return () => 0.7;
    }
  }, [meshItems, opacityIntensity]);

  // Generate spider plot data with normalized values
  const spiderData = useMemo(() => {
    try {
      // Always return the default structure for axes
      if (!meshItems?.length) return defaultSpiderData;

      const validParams = ['Rs', 'Ra', 'Ca', 'Rb', 'Cb'] as const;
      type ValidParam = typeof validParams[number];

      // Find min/max values for each parameter
      const paramRanges = validParams.reduce((acc, param) => {
        try {
          const values = meshItems
            .filter(item => item?.parameters && typeof item.parameters[param as ValidParam] === 'number')
            .map(item => item.parameters[param as ValidParam]);
          
          if (values.length > 0) {
            // Use iterative approach to avoid stack overflow
            let min = Infinity;
            let max = -Infinity;
            for (const value of values) {
              if (value < min) min = value;
              if (value > max) max = value;
            }
            acc[param] = { min, max };
          } else {
            // Use default ranges if no data
            const defaults = {
              Rs: { min: 10, max: 10000 },
              Ra: { min: 10, max: 10000 },
              Rb: { min: 10, max: 10000 },
              Ca: { min: 0.1e-6, max: 50e-6 },
              Cb: { min: 0.1e-6, max: 50e-6 }
            };
            acc[param] = defaults[param] || { min: 0, max: 1 };
          }
        } catch (error) {
          console.error(`Error processing parameter ${param}:`, error);
          acc[param] = { min: 0, max: 1 };
        }
        return acc;
      }, {} as Record<string, { min: number; max: number }>);

      // Create one data point per parameter
      const data: RadarDataPoint[] = validParams.map(param => {
        try {
          const range = paramRanges[param];
          const point: RadarDataPoint = {
            parameter: param,
            fullValue: range.max,
            displayValue: formatParamValue(param, range.max)
          };

          // Process all models to show complete grid exploration results
          const modelsToProcess = meshItems;
          modelsToProcess.forEach(item => {
            try {
              if (item?.parameters && typeof item.parameters[param as ValidParam] === 'number') {
                const value = item.parameters[param as ValidParam];
                const normalizedValue = (value - range.min) / Math.max(range.max - range.min, 1e-10);
                point[item.id] = Math.max(0, Math.min(1, normalizedValue));
              }
            } catch (error) {
              console.error(`Error processing item ${item?.id} for param ${param}:`, error);
            }
          });

          return point;
        } catch (error) {
          console.error(`Error creating data point for parameter ${param}:`, error);
          return {
            parameter: param,
            fullValue: 0,
            displayValue: param
          };
        }
      });

      return data;
    } catch (error) {
      console.error('Error generating spider data:', error);
      return defaultSpiderData;
    }
  }, [meshItems, defaultSpiderData]);

  // Filter models for visualization with logarithmic opacity
  const visibleModels = useMemo(() => {
    try {
      if (!meshItems?.length) return [];
      
      // Show all models to display complete parameter space exploration
      const modelsToShow = meshItems;
      
      const models = modelsToShow.map(item => {
        try {
          const resnorm = item?.resnorm || 0;
          let logOpacity: number;
          
          if (visualizationMode === 'opacity') {
            // Monochrome mode: use consistent logarithmic scaling with intensity factor
            const allResnorms = modelsToShow.map(i => i.resnorm).filter(r => r !== undefined) as number[];
            
            if (resnorm !== undefined && allResnorms.length > 1) {
              // Use iterative approach to avoid stack overflow
              let minResnorm = Infinity;
              let maxResnorm = -Infinity;
              for (const r of allResnorms) {
                if (r < minResnorm) minResnorm = r;
                if (r > maxResnorm) maxResnorm = r;
              }
              
              // Use logarithmic scaling with intensity factor for consistent calculation
              const logMin = Math.log10(Math.max(minResnorm, 1e-10));
              const logMax = Math.log10(maxResnorm);
              const logResnorm = Math.log10(Math.max(resnorm, 1e-10));
              
              // Normalize to 0-1 range (inverted so lower resnorm = higher opacity)
              const normalized = 1 - ((logResnorm - logMin) / (logMax - logMin));
              
              // Apply intensity factor to make the curve more aggressive
              const intensified = Math.pow(normalized, 1 / opacityIntensity);
              
              // Map to 0.05-1.0 range for better visibility with more contrast
              logOpacity = Math.max(0.05, Math.min(1.0, intensified));
            } else {
              logOpacity = 0.5;
            }
          } else {
            // Color groups mode: use original calculateLogOpacity function
            logOpacity = calculateLogOpacity(resnorm);
          }
          
          const finalOpacity = Math.max(0.05, Math.min(1.0, logOpacity * opacityFactor));
          
          // Apply visualization mode
          let color: string;
          if (visualizationMode === 'opacity') {
            // Monochrome mode: use single blue color
            color = '#3B82F6';
          } else {
            // Color groups mode: use original group colors
            color = item?.color || '#3B82F6';
          }
          
          return {
            id: item?.id || 'unknown',
            name: item?.name || 'Unknown',
            color: color,
            opacity: finalOpacity,
            strokeWidth: 1
          };
        } catch (error) {
          console.error('Error processing model for visibility:', item?.id, error);
          return {
            id: item?.id || 'unknown',
            name: item?.name || 'Unknown',
            color: '#3B82F6',
            opacity: 0.5,
            strokeWidth: 1
          };
        }
      });

      // Ensure reference model is included and properly styled
      if (referenceId) {
        try {
          const refModel = meshItems.find(item => item?.id === referenceId);
          if (refModel) {
            // Remove existing reference model if it's in the list
            const filteredModels = models.filter(m => m.id !== referenceId);
            // Add reference model at the end to ensure it's rendered on top
            filteredModels.push({
              id: refModel.id,
              name: refModel.name || 'Reference',
              color: '#FFFFFF', // White color for reference
              opacity: 1.0,
              strokeWidth: 2
            });
            return filteredModels;
          }
        } catch (error) {
          console.error('Error processing reference model:', error);
        }
      }

      return models;
    } catch (error) {
      console.error('Error creating visible models:', error);
      return [];
    }
  }, [meshItems, referenceId, opacityFactor, calculateLogOpacity, visualizationMode, opacityIntensity]);

  // Export functions with error handling - enhanced for high quality
  const exportSvg = (quality: 'standard' | 'high' | 'ultra' = 'high') => {
    try {
      if (!chartRef.current) return;
      const svgElement = chartRef.current.querySelector('svg');
      if (!svgElement) return;

      // Clone the SVG to modify it for high-quality export
      const svgClone = svgElement.cloneNode(true) as SVGElement;
      
      // Set high-resolution dimensions for export based on quality
      const qualitySettings = {
        standard: { width: 800, height: 800 },
        high: { width: 1600, height: 1600 },
        ultra: { width: 3200, height: 3200 }
      };
      
      const settings = qualitySettings[quality];
      svgClone.setAttribute('width', settings.width.toString());
      svgClone.setAttribute('height', settings.height.toString());
      svgClone.setAttribute('viewBox', '0 0 800 600');
      
      // Enhance quality settings
      svgClone.style.shapeRendering = 'geometricPrecision';
      svgClone.style.textRendering = 'geometricPrecision';
      
      // Convert to string
      const serializer = new XMLSerializer();
      const svgString = serializer.serializeToString(svgClone);
      
      if (onExportSvg) {
        onExportSvg(svgString);
      }
    } catch (error) {
      console.error('Error exporting SVG:', error);
    }
  };

  const exportPng = async (quality: 'standard' | 'high' | 'ultra' = 'high') => {
    try {
      if (!chartRef.current) return;
      const svgElement = chartRef.current.querySelector('svg');
      if (!svgElement) return;

      // Quality settings for PNG export
      const qualitySettings = {
        standard: { width: 800, height: 800, scale: 1 },
        high: { width: 1600, height: 1600, scale: 2 },
        ultra: { width: 3200, height: 3200, scale: 4 }
      };

      const settings = qualitySettings[quality];
      
      // Create a high-resolution canvas
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Set high-resolution dimensions
      canvas.width = settings.width;
      canvas.height = settings.height;
      
      // Enable high-quality rendering
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';

      // Create an image from the SVG
      const svgBlob = new Blob([svgElement.outerHTML], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);
      const img = new Image();

      img.onload = () => {
        try {
          // Draw high-quality background
          ctx.fillStyle = '#0f172a'; // Dark background
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          
          // Draw the image with antialiasing
          ctx.scale(settings.scale, settings.scale);
          ctx.drawImage(img, 0, 0, canvas.width / settings.scale, canvas.height / settings.scale);

          // Convert to high-quality PNG
          canvas.toBlob((blob) => {
            if (blob && onExportPng) {
              onExportPng(blob);
            }
            URL.revokeObjectURL(url);
          }, 'image/png', 1.0); // Maximum quality
        } catch (error) {
          console.error('Error in PNG export onload:', error);
          URL.revokeObjectURL(url);
        }
      };

      img.onerror = () => {
        console.error('Error loading SVG for PNG export');
        URL.revokeObjectURL(url);
      };

      img.src = url;
    } catch (error) {
      console.error('Error exporting PNG:', error);
    }
  };

  // Effect to handle export in static mode
  useEffect(() => {
    try {
      if (mode === 'static') {
        // Wait for chart to render
        const timer = setTimeout(() => {
          exportSvg();
          exportPng();
        }, 100);
        return () => clearTimeout(timer);
      }
    } catch (error) {
      console.error('Error in static mode export effect:', error);
    }
  }, [mode, spiderData, visibleModels]);

  // Add event listeners for export triggers
  useEffect(() => {
    const handleTriggerSvgExport = (event: Event) => {
      const customEvent = event as CustomEvent;
      const quality = customEvent.detail?.quality || 'high';
      exportSvg(quality);
    };

    const handleTriggerPngExport = (event: Event) => {
      const customEvent = event as CustomEvent;
      const quality = customEvent.detail?.quality || 'high';
      exportPng(quality);
    };

    window.addEventListener('triggerSvgExport', handleTriggerSvgExport);
    window.addEventListener('triggerPngExport', handleTriggerPngExport);

    return () => {
      window.removeEventListener('triggerSvgExport', handleTriggerSvgExport);
      window.removeEventListener('triggerPngExport', handleTriggerPngExport);
    };
  }, []);

    // Add state for chunked rendering
  const [renderingState, setRenderingState] = useState<ChunkedRenderingState>({
    isRendering: false,
    currentChunk: 0,
    totalChunks: 0,
    renderedPolygons: 0,
    totalPolygons: 0,
    renderStartTime: 0,
    estimatedTimeRemaining: 0
  });

  const [renderingControls, setRenderingControls] = useState<RenderingControls>({
    chunkSize: 500, // Polygons per chunk
    renderQuality: 'medium',
    useCanvas: false,
    progressiveLoading: true,
    maxPolygonsPerFrame: 100
  });

  const [renderedChunks, setRenderedChunks] = useState<Set<number>>(new Set());
  const [visiblePolygons, setVisiblePolygons] = useState<ModelSnapshot[]>([]);

  // Quality settings that affect performance
  const qualitySettings = {
    low: { chunkSize: 1000, maxPolygons: 5000, strokeWidth: 0.5 },
    medium: { chunkSize: 500, maxPolygons: 10000, strokeWidth: 1 },
    high: { chunkSize: 250, maxPolygons: 20000, strokeWidth: 1.5 },
    ultra: { chunkSize: 100, maxPolygons: 50000, strokeWidth: 2 }
  };

  // Chunked rendering function
  const renderPolygonsInChunks = useCallback(async (polygons: ModelSnapshot[]) => {
    const quality = qualitySettings[renderingControls.renderQuality];
    const actualChunkSize = renderingControls.chunkSize || quality.chunkSize;
    const maxPolygonsLimit = Math.min(polygons.length, quality.maxPolygons);
    const polygonsToRender = polygons.slice(0, maxPolygonsLimit);
    
    const totalChunks = Math.ceil(polygonsToRender.length / actualChunkSize);
    
    setRenderingState({
      isRendering: true,
      currentChunk: 0,
      totalChunks,
      renderedPolygons: 0,
      totalPolygons: polygonsToRender.length,
      renderStartTime: Date.now(),
      estimatedTimeRemaining: 0
    });

    setVisiblePolygons([]);
    setRenderedChunks(new Set());

    // Process chunks with delays to prevent UI blocking
    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const startIdx = chunkIndex * actualChunkSize;
      const endIdx = Math.min(startIdx + actualChunkSize, polygonsToRender.length);
      const chunk = polygonsToRender.slice(startIdx, endIdx);

      // Add chunk to visible polygons
      setVisiblePolygons(prev => [...prev, ...chunk]);
      setRenderedChunks(prev => new Set([...prev, chunkIndex]));

      // Update progress
      const renderedCount = endIdx;
      const progress = (renderedCount / polygonsToRender.length) * 100;
      const elapsed = Date.now() - renderingState.renderStartTime;
      const estimatedTotal = elapsed / (progress / 100);
      const remaining = Math.max(0, estimatedTotal - elapsed);

      setRenderingState(prev => ({
        ...prev,
        currentChunk: chunkIndex + 1,
        renderedPolygons: renderedCount,
        estimatedTimeRemaining: remaining
      }));

      // Add delay based on chunk size to keep UI responsive
      const delay = actualChunkSize > 250 ? 50 : actualChunkSize > 100 ? 25 : 10;
      await new Promise(resolve => setTimeout(resolve, delay));

      // Check if rendering was cancelled
      if (!renderingState.isRendering) {
        break;
      }
    }

    setRenderingState(prev => ({ ...prev, isRendering: false }));
  }, [renderingControls, renderingState.renderStartTime, renderingState.isRendering]);

  // Cancel rendering function
  const cancelRendering = useCallback(() => {
    setRenderingState(prev => ({ ...prev, isRendering: false }));
  }, []);

  // Update rendering when meshItems change
  useEffect(() => {
    if (meshItems.length > 1000 && renderingControls.progressiveLoading) {
      renderPolygonsInChunks(meshItems);
    } else {
      setVisiblePolygons(meshItems);
      setRenderingState(prev => ({ ...prev, isRendering: false }));
    }
  }, [meshItems, renderingControls.progressiveLoading, renderPolygonsInChunks]);

  // Modify existing visibleModels to use visiblePolygons instead of meshItems
  const chunkedVisibleModels = useMemo(() => {
    try {
      if (!visiblePolygons?.length) return [];

      // Apply polygon limit
      const effectivePolygons = visiblePolygons.slice(0, maxPolygons);
      
      return effectivePolygons.map(model => {
        try {
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
              
              // Use logarithmic scaling with intensity factor
              const logMin = Math.log10(Math.max(minResnorm, 1e-10));
              const logMax = Math.log10(maxResnorm);
              const logResnorm = Math.log10(Math.max(resnorm, 1e-10));
              
              // Normalize to 0-1 range (inverted so lower resnorm = higher opacity)
              const normalized = 1 - ((logResnorm - logMin) / (logMax - logMin));
              
              // Apply intensity factor
              const intensified = Math.pow(normalized, 1 / opacityIntensity);
              
              // Map to 0.05-1.0 range for better visibility
              logOpacity = Math.max(0.05, Math.min(1.0, intensified));
            } else {
              logOpacity = 0.5;
            }
          } else {
            // Color groups mode: use original calculateLogOpacity function
            logOpacity = calculateLogOpacity(resnorm);
          }
          
          const finalOpacity = Math.max(0.05, Math.min(1.0, logOpacity * opacityFactor));
          
          // Apply visualization mode
          let color: string;
          if (visualizationMode === 'opacity') {
            color = '#3B82F6';
          } else {
            color = model?.color || '#3B82F6';
          }
          
          return {
            id: model?.id || 'unknown',
            name: model?.name || 'Unknown',
            color: color,
            opacity: finalOpacity,
            strokeWidth: model.id === referenceId ? 2 : 1
          };
        } catch (error) {
          console.error('Error processing model for spider plot:', error, model);
          return {
            id: model?.id || 'unknown',
            name: model?.name || 'Unknown',
            color: '#3B82F6',
            opacity: 0.5,
            strokeWidth: 1
          };
        }
      }).filter(Boolean);
    } catch (error) {
      console.error('Error in chunkedVisibleModels calculation:', error);
      return [];
    }
  }, [visiblePolygons, maxPolygons, calculateLogOpacity, opacityFactor, visualizationMode, referenceId, opacityIntensity]);

  // Rendering controls component - converted to horizontal bar
  const RenderingControlsBar = () => (
    <div className="absolute top-4 left-4 right-4 bg-neutral-800/95 backdrop-blur-sm border border-neutral-700 rounded-lg p-3 z-10">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        {/* Left section - Rendering Mode & Quality */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4 text-primary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
            <span className="text-sm font-medium text-neutral-200">Render Controls</span>
          </div>

          {/* Render Mode Toggle */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-neutral-300">Mode:</label>
            <div className="flex bg-neutral-700 rounded-md overflow-hidden">
              <button
                onClick={() => setRenderingControls(prev => ({ ...prev, useCanvas: false }))}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  !renderingControls.useCanvas 
                    ? 'bg-primary text-white' 
                    : 'text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                Interactive
              </button>
              <button
                onClick={() => setRenderingControls(prev => ({ ...prev, useCanvas: true }))}
                className={`px-3 py-1 text-xs font-medium transition-colors ${
                  renderingControls.useCanvas 
                    ? 'bg-amber-600 text-white' 
                    : 'text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                Static Grid
              </button>
            </div>
          </div>

          {/* Quality Selector */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-neutral-300">Quality:</label>
            <select 
              value={renderingControls.renderQuality}
              onChange={(e) => setRenderingControls(prev => ({ 
                ...prev, 
                renderQuality: e.target.value as RenderingControls['renderQuality']
              }))}
              className="px-2 py-1 text-xs bg-neutral-700 border border-neutral-600 rounded text-neutral-200"
            >
              <option value="low">Low (5k)</option>
              <option value="medium">Medium (10k)</option>
              <option value="high">High (20k)</option>
              <option value="ultra">Ultra (50k)</option>
            </select>
          </div>
        </div>

        {/* Center section - Progressive Loading & Chunk Size */}
        <div className="flex items-center gap-4">
          {/* Progressive Loading Toggle */}
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={renderingControls.progressiveLoading}
              onChange={(e) => setRenderingControls(prev => ({ 
                ...prev, 
                progressiveLoading: e.target.checked
              }))}
              className="w-4 h-4"
            />
            <span className="text-xs text-neutral-300">Progressive</span>
          </label>

          {/* Chunk Size Slider */}
          <div className="flex items-center gap-2">
            <label className="text-xs text-neutral-300">Chunk:</label>
            <input
              type="range"
              min="50"
              max="2000"
              step="50"
              value={renderingControls.chunkSize}
              onChange={(e) => setRenderingControls(prev => ({ 
                ...prev, 
                chunkSize: parseInt(e.target.value)
              }))}
              className="w-20 h-2"
            />
            <span className="text-xs text-neutral-400 w-8">{renderingControls.chunkSize}</span>
          </div>
        </div>

        {/* Right section - Stats & Progress */}
        <div className="flex items-center gap-4">
          {/* Stats */}
          <div className="text-xs text-neutral-400 space-x-3">
            <span>Total: {meshItems.length.toLocaleString()}</span>
            <span>Visible: {visiblePolygons.length.toLocaleString()}</span>
            {renderingState.totalChunks > 0 && (
              <span>Chunks: {renderedChunks.size}/{renderingState.totalChunks}</span>
            )}
          </div>

          {/* Progress/Cancel Button */}
          {renderingState.isRendering ? (
            <div className="flex items-center gap-2">
              <div className="w-16 bg-neutral-600 rounded-full h-2">
                <div 
                  className="bg-primary h-2 rounded-full transition-all duration-300"
                  style={{ width: `${(renderingState.renderedPolygons / renderingState.totalPolygons) * 100}%` }}
                />
              </div>
              <span className="text-xs text-neutral-300">
                {Math.round((renderingState.renderedPolygons / renderingState.totalPolygons) * 100)}%
              </span>
              <button
                onClick={cancelRendering}
                className="text-xs px-2 py-1 bg-red-600 hover:bg-red-700 text-white rounded"
              >
                Cancel
              </button>
            </div>
          ) : (
            <div className="text-xs text-neutral-400">
              {renderingControls.useCanvas ? 'Static Mode' : 'Interactive Mode'}
            </div>
          )}
        </div>
      </div>

      {/* Performance Warning - compact */}
      {meshItems.length > 10000 && !renderingControls.progressiveLoading && (
        <div className="mt-2 px-3 py-1.5 bg-amber-600/20 border border-amber-500/30 rounded text-xs text-amber-200 flex items-center gap-2">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 15.5c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          <span>
            {meshItems.length.toLocaleString()} polygons detected. Enable Progressive Loading for better performance.
          </span>
        </div>
      )}
    </div>
  );

  // Add canvas ref for static rendering
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Static grid rendering function
  const generateStaticGrid = useCallback(async () => {
    if (!canvasRef.current || !visiblePolygons.length) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size for high quality
    const size = 800;
    canvas.width = size;
    canvas.height = size;
    
    // Clear canvas with dark background
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, size, size);
    
    // Grid configuration
    const gridSize = Math.ceil(Math.sqrt(visiblePolygons.length));
    const cellSize = size / gridSize;
    const miniPlotSize = cellSize * 0.8; // Leave some padding
    const centerOffset = cellSize * 0.1;
    
    setRenderingState(prev => ({
      ...prev,
      isRendering: true,
      totalPolygons: visiblePolygons.length,
      renderedPolygons: 0
    }));

    // Process polygons in chunks for performance
    const chunkSize = 50;
    for (let i = 0; i < visiblePolygons.length; i += chunkSize) {
      const chunk = visiblePolygons.slice(i, Math.min(i + chunkSize, visiblePolygons.length));
      
      for (let j = 0; j < chunk.length; j++) {
        const polygon = chunk[j];
        const index = i + j;
        
        // Calculate grid position
        const row = Math.floor(index / gridSize);
        const col = index % gridSize;
        const x = col * cellSize + centerOffset;
        const y = row * cellSize + centerOffset;
        
        // Draw mini spider plot for this polygon
        drawMiniSpiderPlot(ctx, x, y, miniPlotSize, polygon);
      }
      
      // Update progress
      setRenderingState(prev => ({
        ...prev,
        renderedPolygons: Math.min(i + chunkSize, visiblePolygons.length)
      }));
      
      // Yield to browser
      await new Promise(resolve => setTimeout(resolve, 5));
    }
    
    setRenderingState(prev => ({ ...prev, isRendering: false }));
  }, [visiblePolygons]);

  // Draw mini spider plot function
  const drawMiniSpiderPlot = (
    ctx: CanvasRenderingContext2D, 
    x: number, 
    y: number, 
    size: number, 
    model: ModelSnapshot
  ) => {
    const centerX = x + size / 2;
    const centerY = y + size / 2;
    const radius = size / 2 - 5;
    
    // Draw pentagon axes (5 parameters)
    const params = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'];
    const angleStep = (2 * Math.PI) / params.length;
    
    // Draw grid circles
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;
    for (let i = 1; i <= 3; i++) {
      ctx.beginPath();
      ctx.arc(centerX, centerY, (radius * i) / 3, 0, 2 * Math.PI);
      ctx.stroke();
    }
    
    // Draw axes
    ctx.strokeStyle = '#4B5563';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < params.length; i++) {
      const angle = i * angleStep - Math.PI / 2;
      const endX = centerX + Math.cos(angle) * radius;
      const endY = centerY + Math.sin(angle) * radius;
      
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
    }
    
    // Draw parameter polygon if model has parameters
    if (model.parameters) {
      const points: { x: number; y: number }[] = [];
      
      params.forEach((param, i) => {
        const value = model.parameters[param as keyof typeof model.parameters];
        if (typeof value === 'number') {
          // Normalize the value (0-1)
          let normalizedValue: number;
          if (param.includes('C')) {
            // Capacitance: 0.1µF to 50µF
            normalizedValue = Math.log10(value * 1e6 / 0.1) / Math.log10(50 / 0.1);
          } else {
            // Resistance: 10Ω to 10kΩ
            normalizedValue = Math.log10(value / 10) / Math.log10(10000 / 10);
          }
          
          normalizedValue = Math.max(0, Math.min(1, normalizedValue));
          const pointRadius = normalizedValue * radius;
          const angle = i * angleStep - Math.PI / 2;
          
          points.push({
            x: centerX + Math.cos(angle) * pointRadius,
            y: centerY + Math.sin(angle) * pointRadius
          });
        }
      });
      
      // Draw the polygon
      if (points.length === params.length) {
        ctx.strokeStyle = model.color || '#3B82F6';
        ctx.lineWidth = 1;
        ctx.globalAlpha = model.opacity || 0.7;
        
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
        ctx.stroke();
        
        ctx.globalAlpha = 1; // Reset alpha
      }
    }
    
    // Draw resnorm indicator (color-coded border)
    const resnorm = model.resnorm || 0;
    let borderColor = '#10B981'; // Green for good fit
    if (resnorm > 1e-3) borderColor = '#FBBF24'; // Yellow for moderate
    if (resnorm > 1e-1) borderColor = '#EF4444'; // Red for poor
    
    ctx.strokeStyle = borderColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, size, size);
  };

  // Trigger static grid generation when mode changes
  useEffect(() => {
    if (renderingControls.useCanvas && visiblePolygons.length > 0) {
      generateStaticGrid();
    }
  }, [renderingControls.useCanvas, generateStaticGrid]);

  // Render different content based on mode
  const renderContent = () => {
    if (renderingControls.useCanvas) {
      // Static grid mode
      return (
        <div className="w-full h-full flex items-center justify-center bg-slate-900">
          <canvas
            ref={canvasRef}
            className="max-w-full max-h-full border border-neutral-700 rounded"
            style={{ 
              imageRendering: 'pixelated',
              width: 'auto',
              height: 'auto'
            }}
          />
        </div>
      );
    } else {
      // Interactive mode - existing radar chart
      const chart = (
        <RadarChart data={spiderData} style={{ fontFamily: "'Inter', 'Segoe UI', Arial, sans-serif" }}>
          <PolarGrid 
            gridType="polygon"
            stroke="#4B5563"
            strokeOpacity={0.3}
            strokeWidth={0.5}
          />
          <PolarAngleAxis 
            dataKey="parameter"
            tick={{ 
              fill: '#E5E7EB', 
              fontSize: 13,
              fontWeight: 600
            }}
            tickLine={{ stroke: '#4B5563' }}
            axisLine={{ stroke: '#4B5563' }}
            tickFormatter={(value) => {
              try {
                const point = spiderData.find(p => p.parameter === value);
                if (!point) return String(value);
                
                // Enhanced axis labels with units and ranges
                const ranges = {
                  Rs: '10Ω - 10kΩ',
                  Ra: '10Ω - 10kΩ', 
                  Rb: '10Ω - 10kΩ',
                  Ca: '0.1 - 50µF',
                  Cb: '0.1 - 50µF'
                };
                
                return `${value}\n${ranges[value as keyof typeof ranges] || ''}`;
              } catch (error) {
                console.error('Error in tick formatter:', error);
                return String(value);
              }
            }}
          />
          <PolarRadiusAxis 
            tick={{ fill: '#E5E7EB', fontSize: 10 }}
            tickCount={5}
            stroke="#4B5563"
            axisLine={{ stroke: '#4B5563' }}
          />
          {/* Only render data polygons if we have actual models */}
          {chunkedVisibleModels.map(model => (
            <Radar
              key={model.id}
              name={model.name}
              dataKey={model.id}
              stroke={model.color}
              fill="none"
              fillOpacity={0}
              strokeOpacity={model.opacity}
              strokeWidth={model.strokeWidth}
              dot={model.id === referenceId}
              isAnimationActive={mode === 'interactive'}
            />
          ))}
        </RadarChart>
      );
      
      return (
        <div className="spider-plot-container" ref={chartRef}>
          <ResponsiveContainer width="100%" height="100%">
            {chart}
          </ResponsiveContainer>
        </div>
      );
    }
  };

  try {

    return (
      <div className="relative w-full h-96">
        {/* Rendering Controls Panel */}
        <RenderingControlsBar />
        
        <div className="spider-plot-container" ref={chartRef}>
          {renderContent()}
        </div>

        {/* Render Progress Overlay */}
        {renderingState.isRendering && (
          <div className="absolute inset-0 bg-neutral-900/20 backdrop-blur-[1px] flex items-center justify-center z-20">
            <div className="bg-neutral-800/95 backdrop-blur-sm border border-neutral-700 rounded-lg p-6 max-w-md">
              <div className="flex items-center mb-4">
                <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-primary mr-3"></div>
                <div>
                  <p className="text-sm font-medium text-neutral-200">Rendering Spider Plot</p>
                  <p className="text-xs text-neutral-400">
                    Processing {renderingState.totalPolygons.toLocaleString()} polygons in chunks...
                  </p>
                </div>
              </div>
              
              <div className="w-full bg-neutral-700 rounded-full h-3 mb-3">
                <div 
                  className="bg-gradient-to-r from-primary to-blue-400 h-3 rounded-full transition-all duration-500"
                  style={{ width: `${(renderingState.renderedPolygons / renderingState.totalPolygons) * 100}%` }}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4 text-xs text-neutral-400">
                <div>
                  <span className="font-medium">Progress:</span><br/>
                  {Math.round((renderingState.renderedPolygons / renderingState.totalPolygons) * 100)}%
                </div>
                <div>
                  <span className="font-medium">Chunk:</span><br/>
                  {renderingState.currentChunk} / {renderingState.totalChunks}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  } catch (error) {
    console.error('Error rendering SpiderPlot:', error);
    return (
      <div className="spider-plot-container flex items-center justify-center h-96">
        <div className="text-red-400 text-sm">
          Error rendering spider plot. Check console for details.
        </div>
      </div>
    );
  }
};
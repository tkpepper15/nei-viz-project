"use client";

import React, { useState, useMemo, useRef, useEffect, useCallback } from 'react';
import { ModelSnapshot } from '../types';
import LinearHistogramScrubber from '../controls/LinearHistogramScrubber';

interface ResnormDisplayProps {
  models: ModelSnapshot[]; // All available models for spectrum analysis
  onResnormRangeChange?: (minResnorm: number, maxResnorm: number) => void; // Callback for range selection
  currentResnorm?: number | null; // Current resnorm being navigated for highlighting
  onResnormSelect?: (resnorm: number) => void; // Callback for clicking on specific resnorm value
  taggedModels?: Map<string, string>; // Map of model id to tag name
  tagColors?: string[]; // Array of colors for tagged models
}

// Navigation state for granular model selection
interface NavigationState {
  enabled: boolean;
  currentSection: number;
  modelsPerSection: number;
  totalSections: number;
}

const ResnormDisplay: React.FC<ResnormDisplayProps> = ({ 
  models, 
  onResnormRangeChange,
  currentResnorm = null,
  onResnormSelect,
  taggedModels = new Map(),
  tagColors = ['#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8E8', '#F8C471', '#82E0AA', '#85C1E9']
}) => {

  // Spectrum interaction state
  const [isDragging, setIsDragging] = useState(false);
  const [dragMode, setDragMode] = useState<'start' | 'end' | 'range'>('range');
  const [selectedRange, setSelectedRange] = useState<{ min: number; max: number } | null>(null);
  const [hoveredResnorm, setHoveredResnorm] = useState<number | null>(null);
  const [cursorStyle, setCursorStyle] = useState<'default' | 'ew-resize' | 'crosshair'>('default');
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const lastCallbackTime = useRef<number>(0);

  // Navigation state for granular model selection
  const [navigation, setNavigation] = useState<NavigationState>({
    enabled: true, // Always enabled now
    currentSection: 0,
    modelsPerSection: 0, // 0 = All models (default)
    totalSections: 1
  });

  // Get current section models for resnorm distribution display
  const currentSectionModels = useMemo(() => {
    // If modelsPerSection is 0, show all models
    if (navigation.modelsPerSection === 0) {
      return models;
    }

    // Sort models by resnorm for consistent sectioning
    const sortedModels = [...models].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
    
    // Calculate section bounds
    const startIndex = navigation.currentSection * navigation.modelsPerSection;
    const endIndex = Math.min(startIndex + navigation.modelsPerSection, sortedModels.length);
    
    return sortedModels.slice(startIndex, endIndex);
  }, [models, navigation.currentSection, navigation.modelsPerSection]);

  // Calculate spectrum data from current section models
  const spectrumData = useMemo(() => {
    const dataModels = currentSectionModels;
    if (!dataModels.length) return null;
    
    const resnorms = dataModels.map(m => m.resnorm || 0).filter(r => r > 0).sort((a, b) => a - b);
    if (resnorms.length === 0) return null;
    
    const minResnorm = Math.min(...resnorms);
    const maxResnorm = Math.max(...resnorms);
    const resnormRange = maxResnorm - minResnorm;
    
    // Create detailed histogram (50 bins for smooth distribution)
    const histogramBins = 50;
    const distribution = new Array(histogramBins).fill(0);
    
    resnorms.forEach(r => {
      const binIndex = Math.floor(((r - minResnorm) / resnormRange) * (histogramBins - 1));
      distribution[Math.max(0, Math.min(histogramBins - 1, binIndex))]++;
    });
    
    return {
      resnorms,
      minResnorm,
      maxResnorm,
      resnormRange,
      distribution,
      histogramBins
    };
  }, [currentSectionModels]);

  // Initialize selected range based on spectrum data
  useEffect(() => {
    if (spectrumData && selectedRange === null) {
      setSelectedRange({
        min: spectrumData.minResnorm,
        max: spectrumData.maxResnorm
      });
    }
  }, [spectrumData, selectedRange]);

  // Throttled callback for real-time updates during dragging
  const throttledRangeChange = useCallback((min: number, max: number) => {
    const now = Date.now();
    if (onResnormRangeChange && (now - lastCallbackTime.current > 100)) { // 100ms throttle
      onResnormRangeChange(min, max);
      lastCallbackTime.current = now;
    }
  }, [onResnormRangeChange]);

  // Update total sections when models or section size changes
  useEffect(() => {
    if (navigation.modelsPerSection === 0) {
      setNavigation(prev => ({ ...prev, totalSections: 1, currentSection: 0 }));
    } else {
      const totalSections = Math.ceil(models.length / navigation.modelsPerSection);
      if (totalSections !== navigation.totalSections) {
        setNavigation(prev => ({ 
          ...prev, 
          totalSections,
          currentSection: Math.min(prev.currentSection, totalSections - 1)
        }));
      }
    }
  }, [models.length, navigation.modelsPerSection, navigation.totalSections]);

  // Apply navigation-based section filtering for granular model selection
  const applyNavigationFilter = useCallback(() => {
    if (!spectrumData || !currentSectionModels.length) return;

    if (navigation.modelsPerSection === 0) {
      // Show all models
      setSelectedRange({
        min: spectrumData.minResnorm,
        max: spectrumData.maxResnorm
      });
      
      if (onResnormRangeChange) {
        onResnormRangeChange(spectrumData.minResnorm, spectrumData.maxResnorm);
      }
      return;
    }

    // Set range to this section's bounds for precise selection
    const sectionMinResnorm = Math.min(...currentSectionModels.map(m => m.resnorm || 0));
    const sectionMaxResnorm = Math.max(...currentSectionModels.map(m => m.resnorm || 0));
    
    setSelectedRange({
      min: sectionMinResnorm,
      max: sectionMaxResnorm
    });
    
    // Notify parent of the new range
    if (onResnormRangeChange) {
      onResnormRangeChange(sectionMinResnorm, sectionMaxResnorm);
    }
  }, [spectrumData, navigation.modelsPerSection, currentSectionModels, onResnormRangeChange]);

  // Apply navigation filter when navigation state changes
  useEffect(() => {
    applyNavigationFilter();
  }, [applyNavigationFilter]);


  // Adobe Premiere Pro style range slider drawing
  const drawSpectrum = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !spectrumData || !selectedRange) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    const { width, height } = canvas;
    const margin = { top: 30, right: 20, bottom: 60, left: 20 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Set background
    ctx.fillStyle = '#1f2937';
    ctx.fillRect(0, 0, width, height);
    
    // Draw histogram bars (background visualization)
    const maxCount = Math.max(...spectrumData.distribution, 1);
    const barWidth = chartWidth / spectrumData.histogramBins;
    
    spectrumData.distribution.forEach((count, i) => {
      const barHeight = (count / maxCount) * chartHeight;
      const x = margin.left + i * barWidth;
      const y = margin.top + chartHeight - barHeight;
      
      // Dimmed background bars
      ctx.fillStyle = '#374151';
      ctx.fillRect(x, y, barWidth - 1, barHeight);
    });
    
    // Calculate crop positions
    const cropStartX = margin.left + ((selectedRange.min - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
    const cropEndX = margin.left + ((selectedRange.max - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
    const cropWidth = cropEndX - cropStartX;
    
    // Draw darkened overlay areas (outside of crop)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
    // Left overlay
    ctx.fillRect(margin.left, margin.top, cropStartX - margin.left, chartHeight);
    // Right overlay
    ctx.fillRect(cropEndX, margin.top, (margin.left + chartWidth) - cropEndX, chartHeight);
    
    // Redraw histogram bars in selected range with bright colors
    spectrumData.distribution.forEach((count, i) => {
      const barHeight = (count / maxCount) * chartHeight;
      const x = margin.left + i * barWidth;
      const y = margin.top + chartHeight - barHeight;
      
      const binResnorm = spectrumData.minResnorm + (i / (spectrumData.histogramBins - 1)) * spectrumData.resnormRange;
      const isInSelectedRange = binResnorm >= selectedRange.min && binResnorm <= selectedRange.max;
      
      if (isInSelectedRange) {
        ctx.fillStyle = '#3b82f6';
        ctx.fillRect(x, y, barWidth - 1, barHeight);
      }
    });
    
    // Draw crop handles (Adobe Premiere style) - same height as frame
    const handleWidth = 12;
    const handleHeight = chartHeight;
    
    // Left handle
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(cropStartX - handleWidth/2, margin.top, handleWidth, handleHeight);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(cropStartX - 1, margin.top + 2, 2, handleHeight - 4);
    
    // Right handle
    ctx.fillStyle = '#f59e0b';
    ctx.fillRect(cropEndX - handleWidth/2, margin.top, handleWidth, handleHeight);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(cropEndX - 1, margin.top + 2, 2, handleHeight - 4);
    
    // Draw crop border
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([]);
    ctx.strokeRect(cropStartX, margin.top, cropWidth, chartHeight);
    
    // Value display inside selected range (like Premiere's timecode)
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px monospace';
    ctx.textAlign = 'center';
    
    const centerX = cropStartX + cropWidth / 2;
    const centerY = margin.top + chartHeight / 2;
    
    // Background for text
    const rangeText = `${selectedRange.min.toExponential(2)} - ${selectedRange.max.toExponential(2)}`;
    const textMetrics = ctx.measureText(rangeText);
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(centerX - textMetrics.width/2 - 8, centerY - 8, textMetrics.width + 16, 16);
    
    // Range text
    ctx.fillStyle = '#ffffff';
    ctx.fillText(rangeText, centerX, centerY + 4);
    
    // Axis labels
    ctx.fillStyle = '#d1d5db';
    ctx.font = '10px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Min: ${spectrumData.minResnorm.toExponential(2)}`, margin.left, height - 35);
    ctx.textAlign = 'right';
    ctx.fillText(`Max: ${spectrumData.maxResnorm.toExponential(2)}`, margin.left + chartWidth, height - 35);
    
    // Models count
    ctx.textAlign = 'center';
    ctx.fillStyle = '#9ca3af';
    ctx.font = '11px Arial';
    const filteredCount = models.filter(m => {
      const resnorm = m.resnorm || 0;
      return resnorm >= selectedRange.min && resnorm <= selectedRange.max;
    }).length;
    ctx.fillText(`${filteredCount} / ${models.length} models selected`, margin.left + chartWidth/2, height - 20);
    
    // Hover feedback
    if (hoveredResnorm !== null) {
      const hoverX = margin.left + ((hoveredResnorm - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
      ctx.strokeStyle = '#60a5fa';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      ctx.beginPath();
      ctx.moveTo(hoverX, margin.top - 10);
      ctx.lineTo(hoverX, margin.top + chartHeight + 10);
      ctx.stroke();
      
      // Hover value
      ctx.fillStyle = '#60a5fa';
      ctx.font = '10px monospace';
      ctx.textAlign = 'center';
      ctx.fillText(hoveredResnorm.toExponential(2), hoverX, margin.top - 15);
    }
    
    // Current resnorm highlighting (from 3D navigation)
    if (currentResnorm !== null && currentResnorm >= spectrumData.minResnorm && currentResnorm <= spectrumData.maxResnorm) {
      const currentX = margin.left + ((currentResnorm - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
      
      // Draw gold vertical line for current resnorm
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = 3;
      ctx.setLineDash([]);
      
      ctx.beginPath();
      ctx.moveTo(currentX, margin.top - 5);
      ctx.lineTo(currentX, margin.top + chartHeight + 5);
      ctx.stroke();
      
      // Current resnorm value with background
      ctx.fillStyle = 'rgba(255, 215, 0, 0.9)';
      ctx.font = 'bold 11px monospace';
      ctx.textAlign = 'center';
      const currentText = currentResnorm.toExponential(2);
      const textMetrics = ctx.measureText(currentText);
      ctx.fillRect(currentX - textMetrics.width/2 - 4, margin.top - 25, textMetrics.width + 8, 16);
      
      ctx.fillStyle = '#000000';
      ctx.fillText(currentText, currentX, margin.top - 12);
      
      // Current position indicator dot
      ctx.fillStyle = '#FFD700';
      ctx.beginPath();
      ctx.arc(currentX, margin.top + chartHeight/2, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
    
    // Tagged models indicators
    if (taggedModels.size > 0 && spectrumData) {
      const taggedModelIds = Array.from(taggedModels.keys());
      taggedModelIds.forEach((modelId, index) => {
        const taggedModel = models.find(m => m.id === modelId);
        if (taggedModel && taggedModel.resnorm) {
          const tagResnorm = taggedModel.resnorm;
          if (tagResnorm >= spectrumData.minResnorm && tagResnorm <= spectrumData.maxResnorm) {
            const tagX = margin.left + ((tagResnorm - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
            const tagColor = tagColors[index % tagColors.length];
            
            // Draw vertical line for tagged model
            ctx.strokeStyle = tagColor;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.8;
            ctx.setLineDash([]);
            
            ctx.beginPath();
            ctx.moveTo(tagX, margin.top);
            ctx.lineTo(tagX, margin.top + chartHeight);
            ctx.stroke();
            
            // Draw tagged model indicator dot
            ctx.fillStyle = tagColor;
            ctx.globalAlpha = 1.0;
            ctx.beginPath();
            ctx.arc(tagX, margin.top + chartHeight + 15, 5, 0, 2 * Math.PI);
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1;
            ctx.stroke();
            
            // Draw tag name below
            const tagName = taggedModels.get(modelId) || '';
            ctx.fillStyle = tagColor;
            ctx.font = 'bold 9px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(tagName, tagX, margin.top + chartHeight + 30);
          }
        }
      });
    }
    
  }, [spectrumData, selectedRange, hoveredResnorm, models, currentResnorm, taggedModels, tagColors]);
  
  // Handle mouse interactions for crop-style handles
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!spectrumData || !canvasRef.current || !selectedRange) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const margin = { top: 30, right: 20, bottom: 60, left: 20 };
    
    // Use the actual rendered canvas dimensions for mouse coordinate calculations
    const chartWidth = rect.width - ((margin.left + margin.right) * rect.width / canvasRef.current.width);
    const marginLeft = margin.left * rect.width / canvasRef.current.width;
    
    if (x < marginLeft || x > marginLeft + chartWidth) return;
    
    const relativeX = Math.max(0, Math.min(chartWidth, x - marginLeft));
    const resnormValue = spectrumData.minResnorm + (relativeX / chartWidth) * spectrumData.resnormRange;

    // Handle double-click for resnorm selection (direct navigation)
    if (e.detail === 2 && onResnormSelect) {
      onResnormSelect(resnormValue);
      return;
    }
    
    // Calculate handle positions
    const startX = ((selectedRange.min - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
    const endX = ((selectedRange.max - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
    
    const distToStartHandle = Math.abs(relativeX - startX);
    const distToEndHandle = Math.abs(relativeX - endX);
    
    const handleTolerance = 15; // Larger tolerance for easier grabbing
    
    // Proximity-based selection: choose the closer handle, or if both are far, determine by which side of range we're closer to
    if (distToStartHandle < handleTolerance || distToEndHandle < handleTolerance) {
      // If close to a handle, pick the closest one
      if (distToStartHandle <= distToEndHandle) {
        setDragMode('start');
      } else {
        setDragMode('end');
      }
    } else {
      // Not close to any handle - determine which handle to move based on proximity to range endpoints
      const distToMin = Math.abs(resnormValue - selectedRange.min);
      const distToMax = Math.abs(resnormValue - selectedRange.max);
      
      if (distToMin < distToMax) {
        setDragMode('start'); // Move min handle
        setSelectedRange({ min: resnormValue, max: selectedRange.max });
      } else {
        setDragMode('end'); // Move max handle  
        setSelectedRange({ min: selectedRange.min, max: resnormValue });
      }
    }
    
    setIsDragging(true);
  };
  
  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!spectrumData || !canvasRef.current || !selectedRange) return;
    
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const margin = { top: 30, right: 20, bottom: 60, left: 20 };
    
    // Use the actual rendered canvas dimensions for mouse coordinate calculations
    const chartWidth = rect.width - ((margin.left + margin.right) * rect.width / canvasRef.current.width);
    const marginLeft = margin.left * rect.width / canvasRef.current.width;
    
    const relativeX = Math.max(0, Math.min(chartWidth, x - marginLeft));
    const resnormValue = spectrumData.minResnorm + (relativeX / chartWidth) * spectrumData.resnormRange;
    
    // Update hover state for visual feedback
    setHoveredResnorm(resnormValue);
    
    // Update cursor style based on hover position
    if (!isDragging) {
      const startX = ((selectedRange.min - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
      const endX = ((selectedRange.max - spectrumData.minResnorm) / spectrumData.resnormRange) * chartWidth;
      
      const distToStartHandle = Math.abs(relativeX - startX);
      const distToEndHandle = Math.abs(relativeX - endX);
      const handleTolerance = 15 * rect.width / canvasRef.current.width; // Scale tolerance with canvas
      
      if (distToStartHandle < handleTolerance || distToEndHandle < handleTolerance) {
        setCursorStyle('ew-resize'); // Horizontal resize cursor for handles
      } else {
        setCursorStyle('ew-resize'); // Always show resize cursor - proximity-based handle selection
      }
      return;
    }
    
    let newRange = selectedRange;
    
    // Constrain resnorm value to full spectrum bounds
    const clampedResnormValue = Math.max(spectrumData.minResnorm, Math.min(spectrumData.maxResnorm, resnormValue));
    
    if (dragMode === 'start') {
      // Left handle - can move to full spectrum min, but can't go past right handle
      newRange = { 
        min: Math.min(clampedResnormValue, selectedRange.max), 
        max: selectedRange.max 
      };
    } else if (dragMode === 'end') {
      // Right handle - can move to full spectrum max, but can't go past left handle
      newRange = { 
        min: selectedRange.min, 
        max: Math.max(clampedResnormValue, selectedRange.min) 
      };
    }
    
    setSelectedRange(newRange);
    throttledRangeChange(newRange.min, newRange.max);
  };
  
  const handleMouseUp = () => {
    setIsDragging(false);
    if (onResnormRangeChange && selectedRange) {
      onResnormRangeChange(selectedRange.min, selectedRange.max);
    }
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
    setHoveredResnorm(null);
    setCursorStyle('default');
    if (onResnormRangeChange && selectedRange) {
      onResnormRangeChange(selectedRange.min, selectedRange.max);
    }
  };
  
  // Redraw when data changes
  useEffect(() => {
    drawSpectrum();
  }, [drawSpectrum]);
  


  return (
    <div className="space-y-4">
      
      {/* Enhanced Resnorm Spectrum with Integrated Navigation */}
      {spectrumData && selectedRange && (
        <div className="bg-neutral-800/50 p-4 rounded border border-neutral-700">
          {/* Navigation Header Overlay */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <span className="text-xs font-medium text-neutral-200">Navigation:</span>
              <select
                value={navigation.modelsPerSection}
                onChange={(e) => setNavigation(prev => ({ 
                  ...prev, 
                  modelsPerSection: parseInt(e.target.value), 
                  currentSection: 0 
                }))}
                className="bg-neutral-700 text-neutral-200 text-xs rounded px-2 py-1 border border-neutral-600 focus:outline-none focus:ring-1 focus:ring-blue-500"
              >
                <option value={0}>All</option>
                <option value={50}>50</option>
                <option value={100}>100</option>
                <option value={200}>200</option>
                <option value={500}>500</option>
                <option value={1000}>1000</option>
              </select>
            </div>
            {navigation.modelsPerSection > 0 && (
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setNavigation(prev => ({ 
                    ...prev, 
                    currentSection: Math.max(0, prev.currentSection - 1) 
                  }))}
                  disabled={navigation.currentSection === 0 || navigation.totalSections <= 1}
                  className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:cursor-not-allowed rounded text-white text-xs transition-colors"
                >
                  ◀
                </button>
                <span className="text-xs text-neutral-400 min-w-[40px] text-center">
                  {navigation.currentSection + 1}/{navigation.totalSections}
                </span>
                <button
                  onClick={() => setNavigation(prev => ({ 
                    ...prev, 
                    currentSection: Math.min(prev.totalSections - 1, prev.currentSection + 1) 
                  }))}
                  disabled={navigation.currentSection >= navigation.totalSections - 1 || navigation.totalSections <= 1}
                  className="px-2 py-1 bg-neutral-700 hover:bg-neutral-600 disabled:bg-neutral-800 disabled:cursor-not-allowed rounded text-white text-xs transition-colors"
                >
                  ▶
                </button>
              </div>
            )}
          </div>

          <div ref={containerRef} className="relative">
            <canvas
              ref={canvasRef}
              width={360}
              height={200}
              className={`w-full bg-gray-800 rounded border border-neutral-600 ${
                cursorStyle === 'ew-resize' ? 'cursor-ew-resize' :
                cursorStyle === 'crosshair' ? 'cursor-crosshair' : 'cursor-default'
              }`}
              onMouseDown={handleMouseDown}
              onMouseMove={handleMouseMove}
              onMouseUp={handleMouseUp}
              onMouseLeave={handleMouseLeave}
            />
          </div>
          
          {/* Linear Histogram Scrubber */}
          {spectrumData && (
            <div className="mt-3 px-1">
              <LinearHistogramScrubber
                min={spectrumData.minResnorm}
                max={spectrumData.maxResnorm}
                value={currentResnorm}
                onChange={onResnormSelect || (() => {})}
                width={containerRef.current?.offsetWidth || 360}
                disabled={!onResnormSelect}
              />
            </div>
          )}
          
          <div className="mt-2 flex justify-between text-xs text-neutral-400">
            <span>Models: {models.length.toLocaleString()}</span>
            <span>Range: {(selectedRange.max - selectedRange.min).toExponential(2)}</span>
          </div>
          
        </div>
      )}

      
    </div>
  );
};

export default ResnormDisplay;

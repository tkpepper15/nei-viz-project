"use client";

import React, { useEffect, useState, useCallback } from 'react';
import { ModelSnapshot, GridParameterArrays } from './utils/types';

interface GridValue {
  level: number;
  Rs: number;
  Ra: number;
  Rb: number;
  Ca: number;
  Cb: number;
}

const CENTER_OFFSET = 50;
const AXIS_LENGTH = 200;
const AXES = ['Rs', 'Ra', 'Rb', 'Ca', 'Cb'] as const;

interface BaseSpiderPlotProps {
  meshItems: ModelSnapshot[];
  referenceId?: string;
  opacityFactor: number;
  gridSize: number;
  onGridValuesGenerated: (values: GridParameterArrays) => void;
  chromaEnabled?: boolean;
  opacityIntensity?: number;
  selectedOpacityGroups?: number[];
  resnormGroups?: Array<{
    range: [number, number];
    color: string;
    label: string;
    description: string;
    items: ModelSnapshot[];
  }>;
}

interface Corner {
  x: number;
  y: number;
}

export const BaseSpiderPlot: React.FC<BaseSpiderPlotProps> = ({ 
  meshItems, 
  referenceId, 
  opacityFactor,
  gridSize,
  onGridValuesGenerated,
  chromaEnabled = true,
  opacityIntensity = 1.0,
  selectedOpacityGroups = [],
  resnormGroups = []
}) => {
  const [gridValues, setGridValues] = useState<GridValue[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);
  const [zoomLevel, setZoomLevel] = useState(0.6);
  const [panX, setPanX] = useState(0);
  const [panY, setPanY] = useState(0);
  // Add state for labels toggle - on by default
  const [showLabels, setShowLabels] = useState(true);
  // Enhanced panning state for smoother interactions
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
  const [isInteracting, setIsInteracting] = useState(false);

  // Calculate more aggressive logarithmic opacity based on resnorm value with group-specific support
  const calculateLogOpacity = useCallback((resnorm: number): number => {
    // Get resnorm values based on selected groups
    let currentResnormValues: number[];
    
    if (selectedOpacityGroups.length === 0 || !resnormGroups.length) {
      // Use all mesh items if no specific groups are selected
      currentResnormValues = meshItems.map(item => item.resnorm).filter(r => r !== undefined) as number[];
    } else {
      // Use only the selected groups' resnorm values
      currentResnormValues = selectedOpacityGroups.flatMap(groupIndex => {
        const selectedGroup = resnormGroups[groupIndex];
        return selectedGroup ? selectedGroup.items.map(item => item.resnorm).filter(r => r !== undefined) as number[] : [];
      });
      
      // Fallback to all items if no valid groups found
      if (currentResnormValues.length === 0) {
        currentResnormValues = meshItems.map(item => item.resnorm).filter(r => r !== undefined) as number[];
      }
    }
    
    if (currentResnormValues.length === 0) return 0.5;
    
    // Use iterative approach to avoid stack overflow with large arrays
    let minResnorm = Infinity;
    let maxResnorm = -Infinity;
    for (const resnormValue of currentResnormValues) {
      if (resnormValue < minResnorm) minResnorm = resnormValue;
      if (resnormValue > maxResnorm) maxResnorm = resnormValue;
    }
    
    if (minResnorm === maxResnorm) return 0.5;
    
    // Use logarithmic scaling with intensity factor for more aggressive curves
    const logMin = Math.log10(Math.max(minResnorm, 1e-10));
    const logMax = Math.log10(maxResnorm);
    const logResnorm = Math.log10(Math.max(resnorm, 1e-10));
    
    // Normalize to 0-1 range (inverted so lower resnorm = higher opacity)
    const normalized = 1 - ((logResnorm - logMin) / (logMax - logMin));
    
    // Apply intensity factor to make the curve more aggressive
    const intensified = Math.pow(normalized, 1 / opacityIntensity);
    
    // Map to 0.05-1.0 range for better visibility with more contrast
    return Math.max(0.05, Math.min(1.0, intensified));
  }, [meshItems, opacityIntensity, selectedOpacityGroups, resnormGroups]);

  // Generate grid values based on the actual parameter space used in computation
  const generateGridValues = useCallback((): GridValue[] => {
    // Use EXACT same ranges as grid-worker.js to ensure alignment
    const rsRange = { min: 10, max: 10000 }; // Rs in Ohms  
    const raRange = { min: 10, max: 10000 }; // Ra in Ohms
    const rbRange = { min: 10, max: 10000 }; // Rb in Ohms
    const caRange = { min: 0.1e-6, max: 50e-6 }; // Ca in Farads (0.1 to 50 µF)
    const cbRange = { min: 0.1e-6, max: 50e-6 }; // Cb in Farads (0.1 to 50 µF)

    // Use EXACT same logarithmic spacing algorithm as grid-worker.js
    const generateLogSpace = (min: number, max: number, num: number): number[] => {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      const step = (logMax - logMin) / (num - 1);
      
      const result: number[] = [];
      for (let i = 0; i < num; i++) {
        result.push(Math.pow(10, logMin + i * step));
      }
      return result;
    };

    // Generate exact parameter values that match worker computation
    const rsValues = generateLogSpace(rsRange.min, rsRange.max, gridSize);
    const raValues = generateLogSpace(raRange.min, raRange.max, gridSize);
    const rbValues = generateLogSpace(rbRange.min, rbRange.max, gridSize);
    const caValues = generateLogSpace(caRange.min, caRange.max, gridSize);
    const cbValues = generateLogSpace(cbRange.min, cbRange.max, gridSize);

    // Create grid values with proper level mapping
    const newGridValues = Array.from({ length: gridSize }, (_, idx) => ({
      level: (idx + 1) / gridSize,
      Rs: rsValues[idx],
      Ra: raValues[idx],
      Rb: rbValues[idx],
      Ca: caValues[idx] * 1e6, // Convert Farads to µF for display
      Cb: cbValues[idx] * 1e6  // Convert Farads to µF for display
    }));

    // Log the generated values for debugging
    console.log('Spider plot grid values (matching worker computation):', newGridValues);

    return newGridValues;
  }, [gridSize]);

  // Helper function to find nearest grid value index using logarithmic distance
  const findNearestGridValueIndex = (value: number, gridValues: number[]): number => {
    let nearestIndex = 0;
    let minDiff = Infinity;
    
    gridValues.forEach((gridValue, index) => {
      const diff = Math.abs(Math.log10(value) - Math.log10(gridValue));
      if (diff < minDiff) {
        minDiff = diff;
        nearestIndex = index;
      }
    });
    
    return nearestIndex;
  };

  // Initialize grid values and handle grid size changes
  useEffect(() => {
    // Always generate grid values, regardless of meshItems
    const newGridValues = generateGridValues();
    setGridValues(newGridValues);
    setIsInitialized(true);

    // Notify parent component of new grid values if callback exists
    if (onGridValuesGenerated) {
      // Extract values ensuring correct units for computation (Farads for capacitance)
      const parameterArrays: GridParameterArrays = {
        Rs: newGridValues.map(v => v.Rs), // Ω
        Ra: newGridValues.map(v => v.Ra), // Ω
        Rb: newGridValues.map(v => v.Rb), // Ω
        Ca: newGridValues.map(v => v.Ca / 1e6), // Convert µF back to Farads for computation
        Cb: newGridValues.map(v => v.Cb / 1e6)  // Convert µF back to Farads for computation
      };

      // Log the values for debugging
      console.log('Generated grid parameter arrays (in computation units):', parameterArrays);
      
      onGridValuesGenerated(parameterArrays);
    }
  }, [gridSize, generateGridValues, onGridValuesGenerated]);

  // Initialize on mount if not already initialized
  useEffect(() => {
    if (!isInitialized) {
      const newGridValues = generateGridValues();
      setGridValues(newGridValues);
      setIsInitialized(true);
    }
  }, [isInitialized, generateGridValues]);

  // Debug effect to track opacity intensity changes
  useEffect(() => {
    console.log('BaseSpiderPlot: opacityIntensity changed to:', opacityIntensity);
    console.log('BaseSpiderPlot: chroma enabled:', chromaEnabled);
  }, [opacityIntensity, chromaEnabled]);

  // Separate reference model from other items
  const referenceModel = meshItems.find(item => item.id === referenceId);
  const nonReferenceItems = meshItems.filter(item => item.id !== referenceId);
  
  // Sort items by resnorm (worst first, best last) for proper opacity layering
  // Filtering is now handled upstream in VisualizerTab
  const sortedItems = [...nonReferenceItems].sort((a, b) => {
    if (!a.resnorm && !b.resnorm) return 0;
    if (!a.resnorm) return -1;
    if (!b.resnorm) return 1;
    return b.resnorm - a.resnorm;
  });
  
  const SVG_SIZE = 500;
  const CENTER_POINT = SVG_SIZE / 2;
  const LABEL_OFFSET = 30;
  const VALUE_LABEL_OFFSET = 15;
  
  // Helper functions for coordinate calculations with safety checks
  const toCartesian = (radius: number, angleDegrees: number): Corner => {
    // Ensure radius and angle are valid numbers
    const safeRadius = isNaN(radius) ? 0 : radius;
    const safeAngle = isNaN(angleDegrees) ? 0 : angleDegrees;
    const angleRadians = (safeAngle * Math.PI) / 180;
    return {
      x: safeRadius * Math.cos(angleRadians) + CENTER_POINT,
      y: safeRadius * Math.sin(angleRadians) + CENTER_POINT
    };
  };

  // Calculate label position with improved spacing to prevent overlaps
  const getLabelPosition = (angle: number, isValue: boolean = false): Corner => {
    const baseRadius = AXIS_LENGTH + (isValue ? VALUE_LABEL_OFFSET : LABEL_OFFSET);
    
    // Increase radius for specific angles to prevent overlaps
    let adjustedRadius = baseRadius;
    if (!isValue) {
      // Main parameter labels - increase spacing for better separation
      if (angle === -18 || angle === 54) { // Ra and Rb
        adjustedRadius = baseRadius + 20; // Increased spacing for right side labels
      } else if (angle === 126 || angle === 198) { // Ca and Cb  
        adjustedRadius = baseRadius + 18; // Increased spacing for left side labels
      } else if (angle === -90) { // Rs at top
        adjustedRadius = baseRadius + 12; // More spacing for top
      }
    }
    
    const pos = toCartesian(adjustedRadius, angle);
    
    // Fine-tune positions based on angle quadrant for better alignment and no overlaps
    const adjustedPos = {
      x: pos.x,
      y: pos.y
    };

    // More precise adjustments based on angle to prevent overlaps
    if (angle === -90) { // Rs - Top
      adjustedPos.y -= 18; // Move further up
    } else if (angle === -18) { // Ra - Top right
      adjustedPos.x += 12;
      adjustedPos.y -= 12; // Move up to separate from Rb
    } else if (angle === 54) { // Rb - Bottom right  
      adjustedPos.x += 12;
      adjustedPos.y += 8; // Move down to separate from Ra
    } else if (angle === 126) { // Ca - Bottom left
      adjustedPos.x -= 12;
      adjustedPos.y += 8; // Move down to separate from Cb
    } else if (angle === 198) { // Cb - Top left
      adjustedPos.x -= 12; 
      adjustedPos.y -= 12; // Move up to separate from Ca
    }

    return adjustedPos;
  };

  // Helper function to format parameter values for display
  const formatParamValue = (value: number, isCapacitance: boolean): string => {
    if (isCapacitance) {
      // For capacitance values (already in µF), show 2 decimal places
      return value.toFixed(2);
    }
    // For resistance values, show 1 decimal place for values < 1000, otherwise no decimals
    return value < 1000 ? value.toFixed(1) : value.toFixed(0);
  };

  // Function to generate pentagon points for an item
  const generatePointsForItem = (item: ModelSnapshot) => {
    if (!item?.parameters || !isInitialized) return null;
    
    try {
      // Extract grid value arrays (these are in display units: Ω for resistance, µF for capacitance)
      const rsValues = gridValues.map(v => v.Rs);
      const raValues = gridValues.map(v => v.Ra);
      const rbValues = gridValues.map(v => v.Rb);
      const caValues = gridValues.map(v => v.Ca); // These are in µF for display
      const cbValues = gridValues.map(v => v.Cb); // These are in µF for display
      
      // Model parameters come in Farads from computation, so convert to µF for comparison
      const modelCaInMicroF = item.parameters.Ca * 1e6; // Convert F to µF
      const modelCbInMicroF = item.parameters.Cb * 1e6; // Convert F to µF
      
      // Find nearest indices - now units match properly
      const rsIndex = findNearestGridValueIndex(item.parameters.Rs, rsValues);
      const raIndex = findNearestGridValueIndex(item.parameters.Ra, raValues);
      const rbIndex = findNearestGridValueIndex(item.parameters.Rb, rbValues);
      const caIndex = findNearestGridValueIndex(modelCaInMicroF, caValues); // Compare µF to µF
      const cbIndex = findNearestGridValueIndex(modelCbInMicroF, cbValues); // Compare µF to µF

      // Calculate levels based on grid indices
      const rsLevel = rsIndex / (gridValues.length - 1);
      const raLevel = raIndex / (gridValues.length - 1);
      const rbLevel = rbIndex / (gridValues.length - 1);
      const caLevel = caIndex / (gridValues.length - 1);
      const cbLevel = cbIndex / (gridValues.length - 1);
      
      // Convert to polar coordinates with fixed angles
      const corners = [
        toCartesian(CENTER_OFFSET + (AXIS_LENGTH - CENTER_OFFSET) * rsLevel, -90),
        toCartesian(CENTER_OFFSET + (AXIS_LENGTH - CENTER_OFFSET) * raLevel, -90 + 72),
        toCartesian(CENTER_OFFSET + (AXIS_LENGTH - CENTER_OFFSET) * rbLevel, -90 + 144),
        toCartesian(CENTER_OFFSET + (AXIS_LENGTH - CENTER_OFFSET) * caLevel, -90 + 216),
        toCartesian(CENTER_OFFSET + (AXIS_LENGTH - CENTER_OFFSET) * cbLevel, -90 + 288)
      ];

      return { corners };
    } catch (error) {
      console.error("Error generating points:", error);
      return null;
    }
  };

  // Smooth zoom and pan functions with reduced sensitivity
  const handleZoomIn = () => {
    setZoomLevel(prev => Math.min(prev * 1.15, 8.0)); // Reduced from 1.3 to 1.15 for smoother steps
  };

  const handleZoomOut = () => {
    setZoomLevel(prev => Math.max(prev / 1.15, 0.2)); // Reduced from 1.3 to 1.15 for smoother steps
  };

  const handleResetZoom = () => {
    setZoomLevel(1.0);
    setPanX(0);
    setPanY(0);
  };

  // Enhanced mouse interaction handlers for smooth Desmos-like navigation
  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsPanning(true);
    setIsInteracting(true);
    setLastPanPoint({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isPanning) {
      const deltaX = e.clientX - lastPanPoint.x;
      const deltaY = e.clientY - lastPanPoint.y;
      
      // Reduced sensitivity for smoother panning
      const sensitivity = 0.8 / zoomLevel; // Adaptive sensitivity based on zoom level
      setPanX(prev => prev - deltaX * sensitivity);
      setPanY(prev => prev - deltaY * sensitivity);
      setLastPanPoint({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setIsInteracting(false);
  };

  const handleMouseLeave = () => {
    setIsPanning(false);
    setIsInteracting(false);
  };

  // Smooth wheel zoom with reduced sensitivity
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    // Much more gradual zoom for smoother experience
    const zoomSensitivity = 0.002; // Reduced sensitivity
    const zoomDelta = -e.deltaY * zoomSensitivity;
    const newZoomLevel = Math.max(0.2, Math.min(8.0, zoomLevel * (1 + zoomDelta)));
    
    setZoomLevel(newZoomLevel);
  };

  // Calculate SVG viewBox based on zoom and pan with smooth transitions
  const calculateViewBox = () => {
    const baseSize = SVG_SIZE;
    const scaledSize = baseSize / zoomLevel;
    const offsetX = (baseSize - scaledSize) / 2 + panX;
    const offsetY = (baseSize - scaledSize) / 2 + panY;
    return `${offsetX} ${offsetY} ${scaledSize} ${scaledSize}`;
  };

  // Get dynamic cursor based on interaction state
  const getCursorStyle = () => {
    if (isPanning) return 'cursor-grabbing';
    if (zoomLevel > 1.0 || isInteracting) return 'cursor-grab';
    return 'cursor-default';
  };
  
  return (
    <div className="spider-plot-wrapper h-full w-full relative">
      {/* Zoom Controls - Outside scrollable content, positioned absolutely */}
      <div className="absolute top-4 right-4 z-[100] pointer-events-none">
        <div className="flex flex-col gap-0.5 bg-black/20 backdrop-blur-lg border border-white/10 rounded-xl p-2 shadow-2xl pointer-events-auto transition-all duration-200 hover:bg-black/30">
          <button
            onClick={handleZoomIn}
            className="w-9 h-9 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/15 rounded-lg transition-all duration-150 active:scale-95"
            title="Zoom In"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
            </svg>
          </button>
          <button
            onClick={handleZoomOut}
            className="w-9 h-9 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/15 rounded-lg transition-all duration-150 active:scale-95"
            title="Zoom Out"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M20 12H4" />
            </svg>
          </button>
          <button
            onClick={handleResetZoom}
            className="w-9 h-9 flex items-center justify-center text-white/70 hover:text-white hover:bg-white/15 rounded-lg transition-all duration-150 active:scale-95 text-[10px] font-mono font-bold"
            title="Reset Zoom (1:1)"
          >
            1:1
          </button>
          {/* Subtle divider */}
          <div className="w-full h-px bg-white/10 my-1"></div>
          {/* Labels Toggle */}
          <button
            onClick={() => setShowLabels(!showLabels)}
            className={`w-9 h-9 flex items-center justify-center rounded-lg transition-all duration-150 active:scale-95 ${
              showLabels 
                ? 'text-blue-300 bg-blue-500/25 hover:bg-blue-500/35' 
                : 'text-white/50 hover:text-white/70 hover:bg-white/15'
            }`}
            title={showLabels ? "Hide Labels" : "Show Labels"}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
            </svg>
          </button>
          
          {/* Zoom indicator */}
          {zoomLevel !== 1.0 && (
            <div className="mt-1 px-2 py-1 bg-black/40 rounded text-white/60 text-[9px] font-mono text-center border border-white/10">
              {(zoomLevel * 100).toFixed(0)}%
            </div>
          )}
        </div>
      </div>

      {/* Spider Plot Container - No scrolling, only pan/zoom navigation */}
      <div 
        className="spider-plot h-full w-full bg-surface border border-neutral-700 rounded"
        style={{ position: 'relative', overflow: 'hidden' }}
      >
        {/* Desmos-style Interactive Canvas */}
        <div 
          className={`h-full w-full select-none ${getCursorStyle()} transition-all duration-75`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onWheel={handleWheel}
          style={{ 
            touchAction: 'none', // Prevent default touch behaviors
            userSelect: 'none',   // Prevent text selection
            position: 'relative'  // Ensure proper positioning context
          }}
        >
          <svg 
            viewBox={calculateViewBox()}
            className="w-full h-full transition-transform duration-75 ease-out"
            style={{ 
              backgroundColor: 'transparent',
              transform: isInteracting ? 'scale(1.001)' : 'scale(1)', // Subtle feedback during interaction
              overflow: 'visible' // Allow SVG content to be visible beyond container if needed
            }}
            preserveAspectRatio="xMidYMid meet"
          >
              {/* Only show axes and grid if we have values */}
              {gridValues.length > 0 && (
                <g className="parameter-axes">
                  {/* Draw the main axes */}
                  {[0, 72, 144, 216, 288].map((angle, idx) => (
                    <line 
                      key={`axis-${idx}`}
                      x1={toCartesian(CENTER_OFFSET, -90 + angle).x} 
                      y1={toCartesian(CENTER_OFFSET, -90 + angle).y} 
                      x2={toCartesian(AXIS_LENGTH, -90 + angle).x} 
                      y2={toCartesian(AXIS_LENGTH, -90 + angle).y} 
                      stroke="#4b5563"
                      strokeWidth="0.5" 
                      opacity={0.6}
                    />
                  ))}

                  {/* Draw grid levels - only show labels if enabled */}
                  {gridValues.map((value, idx) => (
                    <g key={`grid-level-${idx}`} className="grid-level">
                      {/* Value labels and ticks - conditional rendering */}
                      {showLabels && AXES.map((axis, axisIdx) => {
                        const angle = (axisIdx * 72 - 90) * (Math.PI / 180);
                        const tickLength = 5;
                        const tickStart = CENTER_OFFSET + (AXIS_LENGTH - CENTER_OFFSET) * value.level;
                        const tickEnd = tickStart + tickLength;
                        const tickX = Math.cos(angle);
                        const tickY = Math.sin(angle);
                        
                        return (
                          <g key={`tick-${idx}-${axisIdx}`}>
                            <line
                              x1={CENTER_POINT + tickStart * tickX}
                              y1={CENTER_POINT + tickStart * tickY}
                              x2={CENTER_POINT + tickEnd * tickX}
                              y2={CENTER_POINT + tickEnd * tickY}
                              stroke="#6b7280"
                              strokeWidth="0.5"
                              opacity={0.7}
                            />
                            <text
                              x={CENTER_POINT + tickEnd * tickX * 1.1}
                              y={CENTER_POINT + tickEnd * tickY * 1.1}
                              fill="#9ca3af"
                              fontSize="10px"
                              textAnchor="middle"
                              className="font-mono"
                            >
                              {formatParamValue(value[axis], axis === 'Ca' || axis === 'Cb')}
                            </text>
                          </g>
                        );
                      })}
                    </g>
                  ))}
                </g>
              )}

              {/* Draw model polygons - Lower z-index */}
              {sortedItems.map((item, index) => {
                const pointsData = generatePointsForItem(item);
                if (!pointsData) return null;
                
                let logOpacity: number;
                let strokeColor: string;
                
                // Since we already filtered to only show selected groups, all items should receive opacity treatment
                if (chromaEnabled) {
                  // Chroma mode: use original group colors with logarithmic opacity
                  strokeColor = item.color || '#3B82F6';
                  logOpacity = calculateLogOpacity(item.resnorm || 0);
                } else {
                  // Monochrome mode: use single color with group-specific opacity calculation
                  strokeColor = '#3B82F6'; // Fixed blue color
                  logOpacity = calculateLogOpacity(item.resnorm || 0);
                }
                
                const finalOpacity = logOpacity * opacityFactor;

                return (
                  <g key={`model-${index}`}>
                    <path
                      d={`
                        M ${pointsData.corners[0].x} ${pointsData.corners[0].y}
                        L ${pointsData.corners[1].x} ${pointsData.corners[1].y}
                        L ${pointsData.corners[2].x} ${pointsData.corners[2].y}
                        L ${pointsData.corners[3].x} ${pointsData.corners[3].y}
                        L ${pointsData.corners[4].x} ${pointsData.corners[4].y}
                        Z
                      `}
                      stroke={strokeColor}
                      fill="none"
                      strokeWidth={1}
                      strokeOpacity={finalOpacity}
                    />
                  </g>
                );
              })}
              
              {/* Reference model - Lower z-index than labels */}
              {referenceModel && (() => {
                const pointsData = generatePointsForItem(referenceModel);
                if (!pointsData) return null;
                
                return (
                  <g>
                    <path
                      d={`
                        M ${pointsData.corners[0].x} ${pointsData.corners[0].y}
                        L ${pointsData.corners[1].x} ${pointsData.corners[1].y}
                        L ${pointsData.corners[2].x} ${pointsData.corners[2].y}
                        L ${pointsData.corners[3].x} ${pointsData.corners[3].y}
                        L ${pointsData.corners[4].x} ${pointsData.corners[4].y}
                        Z
                      `}
                      stroke="#6b7280"
                      fill="none"
                      strokeWidth={1}
                      strokeDasharray="5,3"
                      opacity={0.8}
                    />
                    {pointsData.corners.map((corner: Corner, idx: number) => (
                      <circle
                        key={`ref-point-${idx}`}
                        cx={corner.x}
                        cy={corner.y}
                        r={1.5}
                        fill="#6b7280"
                        stroke="none"
                        opacity={0.9}
                      />
                    ))}
                  </g>
                );
              })()}

              {/* Clean Parameter labels - No range indicators */}
              {showLabels && gridValues.length > 0 && (
                <g className="parameter-labels">
                  {[
                    { angle: -90, param: "Rs", label: "Rs (Ω)", anchor: "middle" },
                    { angle: -18, param: "Ra", label: "Ra (Ω)", anchor: "start" },
                    { angle: 54, param: "Rb", label: "Rb (Ω)", anchor: "start" },
                    { angle: 126, param: "Ca", label: "Ca (µF)", anchor: "end" },
                    { angle: 198, param: "Cb", label: "Cb (µF)", anchor: "end" }
                  ].map((param, idx) => {
                    const pos = getLabelPosition(param.angle);
                    return (
                      <g key={`param-label-group-${idx}`}>
                        {/* Main parameter label only - clean styling */}
                        <text
                          x={pos.x}
                          y={pos.y}
                          fill="#ffffff"
                          fontSize="14px"
                          fontWeight="700"
                          textAnchor={param.anchor}
                          className="font-bold"
                          style={{ 
                            filter: 'drop-shadow(0 2px 4px rgba(0, 0, 0, 0.8))',
                            paintOrder: 'stroke fill'
                          }}
                          stroke="rgba(0, 0, 0, 0.9)"
                          strokeWidth="0.8"
                        >
                          {param.label}
                        </text>
                      </g>
                    );
                  })}
                </g>
              )}
          </svg>
        </div>
      </div>
    </div>
  );
}; 
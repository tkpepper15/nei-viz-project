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
  visualizationMode?: 'color' | 'opacity';
  opacityIntensity?: number;
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
  visualizationMode = 'color',
  opacityIntensity = 1.0
}) => {
  const [gridValues, setGridValues] = useState<GridValue[]>([]);
  const [isInitialized, setIsInitialized] = useState(false);

  // Calculate more aggressive logarithmic opacity based on resnorm value  
  const calculateLogOpacity = useCallback((resnorm: number): number => {
    // Get fresh resnorm values each time to ensure dynamic updates
    const currentResnormValues = meshItems.map(item => item.resnorm).filter(r => r !== undefined) as number[];
    
    if (currentResnormValues.length === 0) return 0.5;
    
    // Use iterative approach to avoid stack overflow with large arrays
    let minResnorm = Infinity;
    let maxResnorm = -Infinity;
    for (const resnorm of currentResnormValues) {
      if (resnorm < minResnorm) minResnorm = resnorm;
      if (resnorm > maxResnorm) maxResnorm = resnorm;
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
  }, [meshItems, opacityIntensity]);

  // Move generateGridValues outside useEffect
  const generateGridValues = useCallback(() => {
    // Define parameter ranges
    const rsRange = { min: 10, max: 10000 }; // Rs in Ohms
    const raRange = { min: 10, max: 10000 }; // Ra in Ohms
    const rbRange = { min: 10, max: 10000 }; // Rb in Ohms
    const caRange = { min: 0.1, max: 50.0 }; // Ca in µF
    const cbRange = { min: 0.1, max: 50.0 }; // Cb in µF

    // Generate logarithmically spaced values
    const generateLogSpacedValues = (min: number, max: number, count: number): number[] => {
      const values: number[] = [];
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      const step = (logMax - logMin) / (count - 1);
      
      for (let i = 0; i < count; i++) {
        const value = Math.pow(10, logMin + i * step);
        values.push(Number(value.toFixed(value < 1 ? 2 : 1)));
      }
      return values;
    };

    let newGridValues: GridValue[];

    if (gridSize === 3) {
      // Use exact values for gridSize = 3
      newGridValues = [
        {
          level: 1/3,
          Rs: 10.0,
          Ra: 10.0,
          Rb: 10.0,
          Ca: 0.10, // µF
          Cb: 0.10  // µF
        },
        {
          level: 2/3,
          Rs: 2350.0,
          Ra: 7790.0,
          Rb: 7920.0,
          Ca: 37.20, // µF
          Cb: 37.20  // µF
        },
        {
          level: 1,
          Rs: 10000.0,
          Ra: 10000.0,
          Rb: 10000.0,
          Ca: 50.00, // µF
          Cb: 50.00  // µF
        }
      ];
    } else {
      // Generate raw logarithmic values for all parameters for consistency
      const rsValues = generateLogSpacedValues(rsRange.min, rsRange.max, gridSize);
      const raValues = generateLogSpacedValues(raRange.min, raRange.max, gridSize);
      const rbValues = generateLogSpacedValues(rbRange.min, rbRange.max, gridSize);
      const caValues = generateLogSpacedValues(caRange.min, caRange.max, gridSize);
      const cbValues = generateLogSpacedValues(cbRange.min, cbRange.max, gridSize);

      // Create grid values with proper formatting
      newGridValues = Array.from({ length: gridSize }, (_, idx) => ({
        level: (idx + 1) / gridSize,
        Rs: rsValues[idx],
        Ra: raValues[idx],
        Rb: rbValues[idx],
        Ca: Number(caValues[idx].toFixed(2)), // µF with 2 decimal places
        Cb: Number(cbValues[idx].toFixed(2))  // µF with 2 decimal places
      }));
    }

    // Log the generated values for debugging
    console.log('Generated grid values:', newGridValues);

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
      // Extract values and ensure proper units
      const parameterArrays: GridParameterArrays = {
        Rs: newGridValues.map(v => v.Rs),
        Ra: newGridValues.map(v => v.Ra),
        Rb: newGridValues.map(v => v.Rb),
        Ca: newGridValues.map(v => v.Ca), // Already in µF
        Cb: newGridValues.map(v => v.Cb)  // Already in µF
      };

      // Log the values for debugging
      console.log('Generated grid parameter arrays:', parameterArrays);
      
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
    console.log('BaseSpiderPlot: visualization mode is:', visualizationMode);
  }, [opacityIntensity, visualizationMode]);

  // Separate reference model from other items
  const referenceModel = meshItems.find(item => item.id === referenceId);
  
  // Show all grid points - remove arbitrary limits to display the complete parameter space exploration
  const nonReferenceItems = meshItems.filter(item => item.id !== referenceId);
  
  // Sort items by resnorm (worst first, best last) for proper opacity layering
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

  // Calculate label position with offset
  const getLabelPosition = (angle: number, isValue: boolean = false): Corner => {
    const radius = AXIS_LENGTH + (isValue ? VALUE_LABEL_OFFSET : LABEL_OFFSET);
    const pos = toCartesian(radius, angle);
    
    // Adjust based on angle quadrant for better alignment
    const adjustedPos = {
      x: pos.x,
      y: pos.y
    };

    // Fine-tune positions based on angle
    if (angle === -90) { // Top
      adjustedPos.y -= 10;
    } else if (angle > -90 && angle < 90) { // Right side
      adjustedPos.x += 5;
    } else if (angle > 90 && angle < 270) { // Left side
      adjustedPos.x -= 5;
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
      // Extract grid value arrays
      const rsValues = gridValues.map(v => v.Rs);
      const raValues = gridValues.map(v => v.Ra);
      const rbValues = gridValues.map(v => v.Rb);
      const caValues = gridValues.map(v => v.Ca); // These are in µF
      const cbValues = gridValues.map(v => v.Cb); // These are in µF
      
      // Convert parameters to appropriate units and find nearest indices
      const rsIndex = findNearestGridValueIndex(item.parameters.Rs, rsValues);
      const raIndex = findNearestGridValueIndex(item.parameters.Ra, raValues);
      const rbIndex = findNearestGridValueIndex(item.parameters.Rb, rbValues);
      const caIndex = findNearestGridValueIndex(item.parameters.Ca * 1e6, caValues); // Convert F to µF for comparison
      const cbIndex = findNearestGridValueIndex(item.parameters.Cb * 1e6, cbValues); // Convert F to µF for comparison

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
  
  return (
    <div className="spider-plot h-full w-full" style={{ border: '1px solid rgba(100, 116, 139, 0.2)', padding: '1rem' }}>
      <div className="h-full w-full flex items-center justify-center">
        <div className="spider-visualization h-full w-full relative">
          <svg 
            key={`spider-plot-${opacityIntensity}-${visualizationMode}`}
            viewBox={`0 0 ${SVG_SIZE} ${SVG_SIZE}`}
            className="spider-svg w-full h-full"
            style={{ 
              backgroundColor: '#0f172a',
              border: '1px solid rgba(100, 116, 139, 0.2)',
              borderRadius: '0.5rem',
              padding: '1rem'
            }}
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
                    stroke="rgba(100, 116, 139, 0.4)"
                    strokeWidth="1" 
                  />
                ))}

                {/* Draw grid levels */}
                {gridValues.map((value, idx) => (
                  <g key={`grid-level-${idx}`} className="grid-level">
                    {/* Value labels and ticks */}
                    {AXES.map((axis, axisIdx) => {
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
                            stroke="#94a3b8"
                            strokeWidth="1"
                          />
                          <text
                            x={CENTER_POINT + tickEnd * tickX * 1.1}
                            y={CENTER_POINT + tickEnd * tickY * 1.1}
                            fill="#94a3b8"
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

                {/* Enhanced Parameter labels with range indicators */}
                <g className="parameter-labels">
                  {[
                    { angle: -90, param: "Rs", label: "Rs (Ω)", anchor: "middle", range: "10 - 10k" },
                    { angle: -18, param: "Ra", label: "Ra (Ω)", anchor: "start", range: "10 - 10k" },
                    { angle: 54, param: "Rb", label: "Rb (Ω)", anchor: "start", range: "10 - 10k" },
                    { angle: 126, param: "Ca", label: "Ca (µF)", anchor: "end", range: "0.1 - 50" },
                    { angle: 198, param: "Cb", label: "Cb (µF)", anchor: "end", range: "0.1 - 50" }
                  ].map((param, idx) => {
                    const pos = getLabelPosition(param.angle);
                    const rangeLabelPos = getLabelPosition(param.angle, true);
                    return (
                      <g key={`param-label-group-${idx}`}>
                        {/* Main parameter label */}
                        <text
                          x={pos.x}
                          y={pos.y}
                          fill="#e2e8f0"
                          fontSize="13px"
                          textAnchor={param.anchor}
                          className="font-semibold"
                          style={{ filter: 'drop-shadow(0 1px 2px rgba(0, 0, 0, 0.8))' }}
                        >
                          {param.label}
                        </text>
                        {/* Range indicator */}
                        <text
                          x={rangeLabelPos.x}
                          y={rangeLabelPos.y + 12}
                          fill="#94a3b8"
                          fontSize="9px"
                          textAnchor={param.anchor}
                          className="font-mono"
                          style={{ opacity: 0.8 }}
                        >
                          {param.range}
                        </text>
                      </g>
                    );
                  })}
                </g>
              </g>
            )}

            {/* Draw model polygons */}
            {sortedItems.map((item, index) => {
              const pointsData = generatePointsForItem(item);
              if (!pointsData) return null;
              
              // Get all resnorm values for relative calculation
              const allResnorms = sortedItems.map(i => i.resnorm).filter(r => r !== undefined) as number[];
              
              let logOpacity: number;
              let strokeColor: string;
              
              if (visualizationMode === 'opacity') {
                // In opacity mode, use single color with relative opacity based on resnorm distribution
                strokeColor = '#3B82F6'; // Fixed blue color
                
                if (item.resnorm !== undefined && allResnorms.length > 1) {
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
                  const logResnorm = Math.log10(Math.max(item.resnorm, 1e-10));
                  
                  // Normalize to 0-1 range (inverted so lower resnorm = higher opacity)
                  const normalized = 1 - ((logResnorm - logMin) / (logMax - logMin));
                  
                  // Apply intensity factor to make the curve more aggressive
                  const intensified = Math.pow(normalized, 1 / opacityIntensity);
                  
                  // Map to 0.05-1.0 range for better visibility with more contrast
                  logOpacity = Math.max(0.05, Math.min(1.0, intensified));
                  
                  // Debug logging for first few items only to avoid spam
                  if (index < 3) {
                    console.log(`BaseSpiderPlot opacity calc: resnorm=${item.resnorm}, normalized=${normalized.toFixed(3)}, intensity=${opacityIntensity}, intensified=${intensified.toFixed(3)}, finalOpacity=${logOpacity.toFixed(3)}`);
                  }
                } else {
                  logOpacity = 0.5;
                }
              } else {
                // In color mode, use original group colors with logarithmic opacity
                strokeColor = item.color || '#3B82F6';
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
                    strokeWidth={1.5}
                    strokeOpacity={finalOpacity}
                  />
                </g>
              );
            })}
            
            {/* Reference model */}
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
                    stroke="white"
                    fill="none"
                    strokeWidth={2}
                    strokeDasharray="5,3"
                  />
                  {pointsData.corners.map((corner: Corner, idx: number) => (
                    <circle
                      key={`ref-point-${idx}`}
                      cx={corner.x}
                      cy={corner.y}
                      r={3}
                      fill="white"
                      stroke="none"
                    />
                  ))}
                </g>
              );
            })()}
          </svg>
        </div>
      </div>
    </div>
  );
}; 
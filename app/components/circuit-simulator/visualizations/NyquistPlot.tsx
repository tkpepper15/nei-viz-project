"use client";

import React, { useMemo, useRef, useEffect, useState } from 'react';
import { ModelSnapshot, ResnormGroup, ImpedancePoint } from '../types';
import { CircuitParameters } from '../types/parameters';
import { calculateTimeConstants } from '../math/utils';

interface NyquistPlotProps {
  groundTruthParams?: CircuitParameters | null;
  resnormGroups: ResnormGroup[];
  visibleModels: ModelSnapshot[];
  width?: number;
  height?: number;
  showGroundTruth?: boolean;
  chromaEnabled?: boolean;
  numPoints?: number;
  hiddenGroups?: number[];
  selectedOpacityGroups?: number[];
  groupPortion?: number;
}

// Calculate impedance for given parameters
const calculateImpedance = (
  params: CircuitParameters,
  frequencies: number[]
): ImpedancePoint[] => {
  const impedancePoints: ImpedancePoint[] = [];
  
  // Calculate time constants once outside the loop for efficiency
  const { tauA, tauB } = calculateTimeConstants(params);
  
  for (const freq of frequencies) {
    const omega = 2 * Math.PI * freq;
    
    // Calculate individual membrane impedances: Z = R/(1 + jωRC)
    // Za (apical membrane)
    const denominatorA = 1 + Math.pow(omega * tauA, 2);
    const realA = params.Ra / denominatorA;
    const imagA = -params.Ra * omega * tauA / denominatorA;
    
    // Zb (basal membrane)  
    const denominatorB = 1 + Math.pow(omega * tauB, 2);
    const realB = params.Rb / denominatorB;
    const imagB = -params.Rb * omega * tauB / denominatorB;
    
    // Za + Zb (series combination of membranes)
    const realSeriesMembranes = realA + realB;
    const imagSeriesMembranes = imagA + imagB;
    
    // Total impedance: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb) (parallel with shunt)
    const numeratorReal = params.Rsh * realSeriesMembranes;
    const numeratorImag = params.Rsh * imagSeriesMembranes;
    
    const denominatorReal = params.Rsh + realSeriesMembranes;
    const denominatorImag = imagSeriesMembranes;
    const denominatorMagSquared = denominatorReal * denominatorReal + denominatorImag * denominatorImag;
    
    const realTotal = (numeratorReal * denominatorReal + numeratorImag * denominatorImag) / denominatorMagSquared;
    const imagTotal = (numeratorImag * denominatorReal - numeratorReal * denominatorImag) / denominatorMagSquared;
    
    const magnitude = Math.sqrt(realTotal * realTotal + imagTotal * imagTotal);
    const phase = Math.atan2(imagTotal, realTotal) * (180 / Math.PI);
    
    impedancePoints.push({
      frequency: freq,
      real: realTotal,
      imaginary: imagTotal,
      magnitude,
      phase
    });
  }
  
  return impedancePoints;
};


// Generate consistent logarithmic frequency range for proper Nyquist parametric representation
const generateStandardFrequencies = (minFreq?: number, maxFreq?: number, numPoints?: number): number[] => {
  const logMin = Math.log10(minFreq || 0.1);
  const logMax = Math.log10(maxFreq || 10000);
  const pointsToUse = numPoints || 200; // Use provided numPoints or default to 200
  const logStep = (logMax - logMin) / (pointsToUse - 1);
  
  const freqs: number[] = [];
  for (let i = 0; i < pointsToUse; i++) {
    const logValue = logMin + i * logStep;
    freqs.push(Math.pow(10, logValue));
  }
  return freqs;
};

// Calculate statistics based on impedance magnitude at each frequency point
const calculateGroupStatisticsByImpedance = (group: ResnormGroup, standardFreqs?: number[]): {
  median: ImpedancePoint[];
  min: ImpedancePoint[];
  max: ImpedancePoint[];
} => {
  if (group.items.length === 0) {
    return { median: [], min: [], max: [] };
  }
  
  // Use consistent logarithmic frequency spacing
  const frequencies = standardFreqs || generateStandardFrequencies();
  const median: ImpedancePoint[] = [];
  const min: ImpedancePoint[] = [];
  const max: ImpedancePoint[] = [];
  
  for (const freq of frequencies) {
    // Calculate impedance for all models at this exact frequency
    const impedancePoints: ImpedancePoint[] = [];
    
    for (const model of group.items) {
      if (model.parameters) {
        const impedancePoint = calculateImpedanceAtFrequency(model.parameters, freq);
        impedancePoints.push(impedancePoint);
      }
    }
    
    if (impedancePoints.length > 0) {
      // Sort by magnitude to find coherent min/median/max impedance values
      // This preserves the real/imaginary relationship for each impedance point
      impedancePoints.sort((a, b) => a.magnitude - b.magnitude);
      
      const minIndex = 0;
      const medianIndex = Math.floor(impedancePoints.length / 2);
      const maxIndex = impedancePoints.length - 1;
      
      // Use the actual impedance points (preserves complex number relationships)
      min.push({
        frequency: freq,
        real: impedancePoints[minIndex].real,
        imaginary: impedancePoints[minIndex].imaginary,
        magnitude: impedancePoints[minIndex].magnitude,
        phase: impedancePoints[minIndex].phase
      });
      
      median.push({
        frequency: freq,
        real: impedancePoints[medianIndex].real,
        imaginary: impedancePoints[medianIndex].imaginary,
        magnitude: impedancePoints[medianIndex].magnitude,
        phase: impedancePoints[medianIndex].phase
      });
      
      max.push({
        frequency: freq,
        real: impedancePoints[maxIndex].real,
        imaginary: impedancePoints[maxIndex].imaginary,
        magnitude: impedancePoints[maxIndex].magnitude,
        phase: impedancePoints[maxIndex].phase
      });
    }
  }
  
  return { median, min, max };
};

// Calculate statistics based on resnorm ranking (best/median/worst fitting models)
const calculateGroupStatisticsByResnorm = (group: ResnormGroup, standardFreqs?: number[]): {
  median: ImpedancePoint[];
  min: ImpedancePoint[];
  max: ImpedancePoint[];
} => {
  if (group.items.length === 0) {
    return { median: [], min: [], max: [] };
  }
  
  // Use consistent logarithmic frequency spacing
  const frequencies = standardFreqs || generateStandardFrequencies();
  
  // Sort models by resnorm (best fit = lowest resnorm)
  const sortedModels = [...group.items].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
  
  const minIndex = 0; // Best fit (lowest resnorm)
  const medianIndex = Math.floor(sortedModels.length / 2); // Median fit
  const maxIndex = sortedModels.length - 1; // Worst fit (highest resnorm)
  
  // Calculate impedance curves for the selected models
  const minModel = sortedModels[minIndex];
  const medianModel = sortedModels[medianIndex];
  const maxModel = sortedModels[maxIndex];
  
  const min: ImpedancePoint[] = [];
  const median: ImpedancePoint[] = [];
  const max: ImpedancePoint[] = [];
  
  for (const freq of frequencies) {
    if (minModel.parameters) {
      min.push(calculateImpedanceAtFrequency(minModel.parameters, freq));
    }
    if (medianModel.parameters) {
      median.push(calculateImpedanceAtFrequency(medianModel.parameters, freq));
    }
    if (maxModel.parameters) {
      max.push(calculateImpedanceAtFrequency(maxModel.parameters, freq));
    }
  }
  
  return { median, min, max };
};

// Wrapper function to choose calculation method
const calculateGroupStatistics = (
  group: ResnormGroup, 
  standardFreqs: number[] | undefined, 
  byResnorm: boolean
): {
  median: ImpedancePoint[];
  min: ImpedancePoint[];
  max: ImpedancePoint[];
} => {
  return byResnorm 
    ? calculateGroupStatisticsByResnorm(group, standardFreqs)
    : calculateGroupStatisticsByImpedance(group, standardFreqs);
};


// Calculate impedance at a single frequency for consistent spacing
const calculateImpedanceAtFrequency = (params: CircuitParameters, freq: number): ImpedancePoint => {
  const omega = 2 * Math.PI * freq;
  
  // Calculate time constants
  const { tauA, tauB } = calculateTimeConstants(params);
  
  // Calculate individual membrane impedances: Z = R/(1 + jωRC)
  // Za (apical membrane)
  const denominatorA = 1 + Math.pow(omega * tauA, 2);
  const realA = params.Ra / denominatorA;
  const imagA = -params.Ra * omega * tauA / denominatorA;
  
  // Zb (basal membrane)  
  const denominatorB = 1 + Math.pow(omega * tauB, 2);
  const realB = params.Rb / denominatorB;
  const imagB = -params.Rb * omega * tauB / denominatorB;
  
  // Za + Zb (series combination of membranes)
  const realSeriesMembranes = realA + realB;
  const imagSeriesMembranes = imagA + imagB;
  
  // Total impedance: Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb) (parallel with shunt)
  const numeratorReal = params.Rsh * realSeriesMembranes;
  const numeratorImag = params.Rsh * imagSeriesMembranes;
  
  const denominatorReal = params.Rsh + realSeriesMembranes;
  const denominatorImag = imagSeriesMembranes;
  const denominatorMagSquared = denominatorReal * denominatorReal + denominatorImag * denominatorImag;
  
  const realTotal = (numeratorReal * denominatorReal + numeratorImag * denominatorImag) / denominatorMagSquared;
  const imagTotal = (numeratorImag * denominatorReal - numeratorReal * denominatorImag) / denominatorMagSquared;
  
  const magnitude = Math.sqrt(realTotal * realTotal + imagTotal * imagTotal);
  const phase = Math.atan2(imagTotal, realTotal) * (180 / Math.PI);
  
  return {
    frequency: freq,
    real: realTotal,
    imaginary: imagTotal,
    magnitude,
    phase
  };
};

export const NyquistPlot: React.FC<NyquistPlotProps> = ({
  groundTruthParams,
  resnormGroups,
  visibleModels: _visibleModels, // eslint-disable-line @typescript-eslint/no-unused-vars
  width = 600,
  height = 450,
  showGroundTruth = true,
  chromaEnabled = true,
  numPoints = 200,
  hiddenGroups = [],
  selectedOpacityGroups = [0],
  groupPortion = 1.0
  // Equal step size enforced - no logarithmic scaling options
}) => {
  // Toggle between impedance-based and resnorm-based median calculation
  const [medianByResnorm, setMedianByResnorm] = useState(false);
  
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Filter groups using both selectedOpacityGroups, hiddenGroups, and groupPortion
  const visibleResnormGroups = useMemo(() => {
    // If no groups are selected for opacity, show nothing
    if (selectedOpacityGroups.length === 0) {
      return [];
    }
    
    // Apply filtering and groupPortion (same logic as VisualizerTab)
    return resnormGroups
      .filter((_, index) => 
        selectedOpacityGroups.includes(index) && !hiddenGroups.includes(index)
      )
      .map(group => {
        // Apply groupPortion to each selected group
        const keepCount = Math.max(1, Math.floor(group.items.length * groupPortion));
        // Sort by resnorm (ascending = best fits first) and take the top portion
        const sortedItems = [...group.items].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
        const filteredItems = sortedItems.slice(0, keepCount);
        
        return {
          ...group,
          items: filteredItems
        };
      });
  }, [resnormGroups, selectedOpacityGroups, hiddenGroups, groupPortion]);
  
  // Generate consistent frequency range for proper parametric representation
  const frequencies = useMemo(() => {
    if (!groundTruthParams?.frequency_range) {
      return generateStandardFrequencies(100, 1000000, numPoints); // Default: 0.1 kHz to 1000 kHz (100 Hz to 1 MHz)
    }
    
    const [minFreq, maxFreq] = groundTruthParams.frequency_range;
    return generateStandardFrequencies(minFreq, maxFreq, numPoints);
  }, [groundTruthParams, numPoints]);
  
  // Calculate ground truth impedance (represents current parameters)
  const groundTruthImpedance = useMemo(() => {
    if (!groundTruthParams || !showGroundTruth) return [];
    return calculateImpedance(groundTruthParams, frequencies);
  }, [groundTruthParams, frequencies, showGroundTruth]);
  
  // Calculate group statistics (median, min, max) with consistent frequency spacing (only for visible groups)
  const groupStatistics = useMemo(() => {
    return visibleResnormGroups.map(group => ({
      group,
      ...calculateGroupStatistics(group, frequencies, medianByResnorm)
    }));
  }, [visibleResnormGroups, frequencies, medianByResnorm]);
  
  // Find plot bounds with better scaling logic
  const plotBounds = useMemo(() => {
    let minReal = Infinity, maxReal = -Infinity;
    let minImag = Infinity, maxImag = -Infinity;
    
    // Include ground truth bounds
    if (groundTruthImpedance.length > 0) {
      groundTruthImpedance.forEach(point => {
        minReal = Math.min(minReal, point.real);
        maxReal = Math.max(maxReal, point.real);
        minImag = Math.min(minImag, point.imaginary);
        maxImag = Math.max(maxImag, point.imaginary);
      });
    }
    
    
    // Include group bounds (median, min, max)
    groupStatistics.forEach(({ median, min, max }) => {
      [...median, ...min, ...max].forEach(point => {
        minReal = Math.min(minReal, point.real);
        maxReal = Math.max(maxReal, point.real);
        minImag = Math.min(minImag, point.imaginary);
        maxImag = Math.max(maxImag, point.imaginary);
      });
    });
    
    // Default bounds if no data
    if (!isFinite(minReal)) {
      minReal = 0;
      maxReal = 1000;
      minImag = -500;
      maxImag = 0;
    }
    
    // Ensure we show negative imaginary impedance (classic Nyquist plot style)
    if (maxImag > 0) maxImag = 0;
    
    // Adjust bounds to align with integer grid system
    // This ensures clean cropping and consistent axis stepping
    const realRange = maxReal - minReal;
    const imagRange = Math.abs(maxImag - minImag);
    const maxRange = Math.max(realRange, imagRange);
    
    // Calculate base step size that will be used for grid
    let baseStep = 1;
    if (maxRange > 100) {
      baseStep = Math.ceil(maxRange / 10);
      // Round to nice numbers
      if (baseStep <= 5) baseStep = 5;
      else if (baseStep <= 10) baseStep = 10;
      else if (baseStep <= 25) baseStep = 25;
      else if (baseStep <= 50) baseStep = 50;
      else if (baseStep <= 100) baseStep = 100;
      else if (baseStep <= 250) baseStep = 250;
      else if (baseStep <= 500) baseStep = 500;
      else baseStep = Math.ceil(baseStep / 1000) * 1000;
    } else if (maxRange > 50) {
      baseStep = 5;
    } else if (maxRange > 20) {
      baseStep = 2;
    }
    
    // Align bounds to grid steps for clean axis labels
    minReal = Math.floor(minReal / baseStep) * baseStep;
    maxReal = Math.ceil(maxReal / baseStep) * baseStep;
    minImag = Math.floor(minImag / baseStep) * baseStep;
    maxImag = Math.ceil(maxImag / baseStep) * baseStep;
    
    // Ensure imaginary axis shows negative values (classic Nyquist convention)
    if (maxImag > 0) maxImag = 0;
    
    // Maintain aspect ratio by expanding the smaller range if needed
    const adjustedRealRange = maxReal - minReal;
    const adjustedImagRange = Math.abs(maxImag - minImag);
    const targetRange = Math.max(adjustedRealRange, adjustedImagRange);
    
    // Expand to square aspect ratio for consistent grid visualization
    if (adjustedRealRange < targetRange) {
      const center = (minReal + maxReal) / 2;
      const halfRange = targetRange / 2;
      minReal = center - halfRange;
      maxReal = center + halfRange;
      // Keep real axis non-negative for impedance plots
      if (minReal < 0) {
        maxReal += Math.abs(minReal);
        minReal = 0;
      }
    }
    
    if (adjustedImagRange < targetRange) {
      const center = (minImag + maxImag) / 2;
      const halfRange = targetRange / 2;
      minImag = center - halfRange;
      maxImag = center + halfRange;
      // Keep imaginary axis non-positive for impedance plots
      if (maxImag > 0) {
        minImag -= maxImag;
        maxImag = 0;
      }
    }
    
    return {
      minReal,
      maxReal,
      minImag,
      maxImag,
    };
  }, [groundTruthImpedance, groupStatistics]);
  
  // Plotting area dimensions with better spacing
  const plotMargin = useMemo(() => ({ top: 50, right: 50, bottom: 70, left: 80 }), []);
  const plotWidth = width - plotMargin.left - plotMargin.right;
  const plotHeight = height - plotMargin.top - plotMargin.bottom;
  
  // Render the plot
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Enhanced coordinate transformation functions with smart scaling
    const toCanvasX = (real: number) => {
      const { minReal, maxReal } = plotBounds;
      return plotMargin.left + ((real - minReal) / (maxReal - minReal)) * plotWidth;
    };
    
    const toCanvasY = (imag: number) => {
      const { minImag, maxImag } = plotBounds;
      return plotMargin.top + (1 - (imag - minImag) / (maxImag - minImag)) * plotHeight;
    };
    
    // Clear canvas with dark background for dark mode
    ctx.fillStyle = '#1f2937'; // neutral-800
    ctx.fillRect(0, 0, width, height);
    
    // Draw plot border (light for dark mode)
    ctx.strokeStyle = '#6b7280'; // neutral-500
    ctx.lineWidth = 2;
    ctx.strokeRect(plotMargin.left, plotMargin.top, plotWidth, plotHeight);
    
    // Draw grid lines (darker for dark mode)
    ctx.strokeStyle = '#374151'; // neutral-700
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    const { minReal, maxReal, minImag, maxImag } = plotBounds;
    
    // Generate standardized integer grid system with consistent stepping
    // Determine appropriate base step size based on data range
    const realRange = maxReal - minReal;
    const imagRange = maxImag - minImag;
    const maxRange = Math.max(realRange, Math.abs(imagRange));
    
    // Calculate clean integer step size
    let baseStep = 1;
    if (maxRange > 100) {
      baseStep = Math.ceil(maxRange / 10); // e.g., for range 0-150, use step=15
      // Round to nice numbers
      if (baseStep <= 5) baseStep = 5;
      else if (baseStep <= 10) baseStep = 10;
      else if (baseStep <= 25) baseStep = 25;
      else if (baseStep <= 50) baseStep = 50;
      else if (baseStep <= 100) baseStep = 100;
      else if (baseStep <= 250) baseStep = 250;
      else if (baseStep <= 500) baseStep = 500;
      else baseStep = Math.ceil(baseStep / 1000) * 1000;
    } else if (maxRange > 50) {
      baseStep = 5;
    } else if (maxRange > 20) {
      baseStep = 2;
    } else {
      baseStep = 1;
    }
    
    // Generate clean integer grid values for both axes
    const realGridValues = [];
    const imagGridValues = [];
    
    // Real axis with integer steps: ..., -2, -1, 0, 1, 2, 3, 4, 5, ...
    const realStart = Math.floor(minReal / baseStep) * baseStep;
    const realEnd = Math.ceil(maxReal / baseStep) * baseStep;
    for (let value = realStart; value <= realEnd; value += baseStep) {
      realGridValues.push(value);
    }
    
    // Imaginary axis with identical integer steps: ..., -3, -2, -1, 0, 1, 2, 3, ...
    const imagStart = Math.floor(minImag / baseStep) * baseStep;
    const imagEnd = Math.ceil(maxImag / baseStep) * baseStep;
    for (let value = imagStart; value <= imagEnd; value += baseStep) {
      imagGridValues.push(value);
    }
    
    // Draw vertical grid lines (real axis) with unified step size
    realGridValues.forEach(real => {
      const x = toCanvasX(real);
      
      if (x >= plotMargin.left && x <= plotMargin.left + plotWidth) {
        ctx.beginPath();
        ctx.moveTo(x, plotMargin.top);
        ctx.lineTo(x, plotMargin.top + plotHeight);
        
        // Simplified grid styling - all major lines
        ctx.strokeStyle = '#4A5568';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
    
    // Draw horizontal grid lines (imaginary axis) with unified step size
    imagGridValues.forEach(imag => {
      const y = toCanvasY(imag);
      
      if (y >= plotMargin.top && y <= plotMargin.top + plotHeight) {
        ctx.beginPath();
        ctx.moveTo(plotMargin.left, y);
        ctx.lineTo(plotMargin.left + plotWidth, y);
        
        // Match vertical grid line styling exactly
        ctx.strokeStyle = '#4A5568';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
    
    // Reset grid styling
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    
    // Draw resnorm group ribbon plots (financial modeling style)
    if (chromaEnabled) {
      groupStatistics.forEach(({ group, median, min, max }) => {
        if (median.length > 0 && min.length > 0 && max.length > 0) {
          // Parse color and create semi-transparent fill
          const color = group.color;
          const rgbMatch = color.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
          let fillColor = color + '40'; // Add alpha for transparency
          
          if (rgbMatch) {
            const [, r, g, b] = rgbMatch;
            fillColor = `rgba(${r}, ${g}, ${b}, 0.25)`; // 25% opacity for ribbon fill
          }
          
          // Draw smooth ribbon fill between min and max
          if (min.length > 1 && max.length > 1) {
            ctx.fillStyle = fillColor;
            ctx.beginPath();
            
            // Draw smooth upper bound (max) using quadratic curves
            const maxX = toCanvasX(max[0].real);
            const maxY = toCanvasY(max[0].imaginary);
            ctx.moveTo(maxX, maxY);
            
            for (let i = 1; i < max.length; i++) {
              const x = toCanvasX(max[i].real);
              const y = toCanvasY(max[i].imaginary);
              
              if (i < max.length - 1) {
                // Use quadratic curve for smoothness
                const nextX = toCanvasX(max[i + 1].real);
                const nextY = toCanvasY(max[i + 1].imaginary);
                const cpX = x;
                const cpY = y;
                const endX = (x + nextX) / 2;
                const endY = (y + nextY) / 2;
                ctx.quadraticCurveTo(cpX, cpY, endX, endY);
              } else {
                // Last point
                ctx.lineTo(x, y);
              }
            }
            
            // Draw smooth lower bound (min) in reverse order
            for (let i = min.length - 1; i >= 0; i--) {
              const x = toCanvasX(min[i].real);
              const y = toCanvasY(min[i].imaginary);
              
              if (i > 0) {
                // Use quadratic curve for smoothness
                const prevX = toCanvasX(min[i - 1].real);
                const prevY = toCanvasY(min[i - 1].imaginary);
                const cpX = x;
                const cpY = y;
                const endX = (x + prevX) / 2;
                const endY = (y + prevY) / 2;
                ctx.quadraticCurveTo(cpX, cpY, endX, endY);
              } else {
                // Last point (first in min array)
                ctx.lineTo(x, y);
              }
            }
            
            ctx.closePath();
            ctx.fill();
          }
          
          // Draw smooth min line (dotted)
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([2, 3]); // Dotted line for min
          ctx.beginPath();
          if (min.length > 0) {
            const startX = toCanvasX(min[0].real);
            const startY = toCanvasY(min[0].imaginary);
            ctx.moveTo(startX, startY);
            
            for (let i = 1; i < min.length; i++) {
              const x = toCanvasX(min[i].real);
              const y = toCanvasY(min[i].imaginary);
              
              if (i < min.length - 1) {
                // Use quadratic curve for smoothness
                const nextX = toCanvasX(min[i + 1].real);
                const nextY = toCanvasY(min[i + 1].imaginary);
                const cpX = x;
                const cpY = y;
                const endX = (x + nextX) / 2;
                const endY = (y + nextY) / 2;
                ctx.quadraticCurveTo(cpX, cpY, endX, endY);
              } else {
                // Last point
                ctx.lineTo(x, y);
              }
            }
          }
          ctx.stroke();
          
          // Draw smooth max line (solid)
          ctx.setLineDash([]); // Solid line for max
          ctx.beginPath();
          if (max.length > 0) {
            const startX = toCanvasX(max[0].real);
            const startY = toCanvasY(max[0].imaginary);
            ctx.moveTo(startX, startY);
            
            for (let i = 1; i < max.length; i++) {
              const x = toCanvasX(max[i].real);
              const y = toCanvasY(max[i].imaginary);
              
              if (i < max.length - 1) {
                // Use quadratic curve for smoothness
                const nextX = toCanvasX(max[i + 1].real);
                const nextY = toCanvasY(max[i + 1].imaginary);
                const cpX = x;
                const cpY = y;
                const endX = (x + nextX) / 2;
                const endY = (y + nextY) / 2;
                ctx.quadraticCurveTo(cpX, cpY, endX, endY);
              } else {
                // Last point
                ctx.lineTo(x, y);
              }
            }
          }
          ctx.stroke();
          
          // Reset line dash
          ctx.setLineDash([]);
        }
      });
      
      // Draw median dots on top of all ribbons for better visibility
      groupStatistics.forEach(({ group, median }) => {
        if (median.length > 0) {
          // Draw median dots with white outline for visibility
          median.forEach((point) => {
            const x = toCanvasX(point.real);
            const y = toCanvasY(point.imaginary);
            
            // White outline
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = '#ffffff';
            ctx.fill();
            
            // Colored center
            ctx.beginPath();
            ctx.arc(x, y, 3.5, 0, 2 * Math.PI);
            ctx.fillStyle = group.color;
            ctx.fill();
          });
        }
      });
    }
    
    // Draw ground truth impedance as white dots (represents current parameters)
    if (showGroundTruth && groundTruthImpedance.length > 0) {
      ctx.fillStyle = '#ffffff'; // White for dark mode
      
      groundTruthImpedance.forEach((point) => {
        const x = toCanvasX(point.real);
        const y = toCanvasY(point.imaginary);
        
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, 2 * Math.PI); // 5px radius dots (slightly larger than group dots)
        ctx.fill();
      });
    }
    
    // Draw axes (light for dark mode)
    ctx.strokeStyle = '#e5e7eb'; // neutral-200
    ctx.lineWidth = 2;
    
    // X-axis
    ctx.beginPath();
    ctx.moveTo(plotMargin.left, plotMargin.top + plotHeight);
    ctx.lineTo(plotMargin.left + plotWidth, plotMargin.top + plotHeight);
    ctx.stroke();
    
    // Y-axis
    ctx.beginPath();
    ctx.moveTo(plotMargin.left, plotMargin.top);
    ctx.lineTo(plotMargin.left, plotMargin.top + plotHeight);
    ctx.stroke();
    
    // Draw axis labels and tick marks (light for dark mode)
    ctx.fillStyle = '#e5e7eb'; // neutral-200
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    
    // Create consistent axis labels with clean integer formatting
    const formatValue = (value: number) => {
      // Handle floating point precision issues
      const rounded = Math.round(value * 100) / 100;
      const absValue = Math.abs(rounded);
      
      if (absValue >= 1000) {
        const kValue = rounded / 1000;
        // For k-values, show decimal only if needed
        if (Math.abs(kValue % 1) < 0.01) {
          return Math.round(kValue) + 'k';
        }
        return kValue.toFixed(1) + 'k';
      }
      
      // For regular values, show as integer if it's a whole number
      if (Math.abs(rounded % 1) < 0.01) {
        return Math.round(rounded).toString();
      }
      
      // Show up to 1 decimal place for non-integers
      return rounded.toFixed(1);
    };
    
    // Draw X-axis labels with unified formatting
    realGridValues.forEach(real => {
      const x = toCanvasX(real);
      if (x >= plotMargin.left - 20 && x <= plotMargin.left + plotWidth + 20) {
        // Tick mark
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, plotMargin.top + plotHeight);
        ctx.lineTo(x, plotMargin.top + plotHeight + 8);
        ctx.stroke();
        
        // Unified label formatting
        const label = formatValue(real);
        
        ctx.fillStyle = '#e5e7eb';
        ctx.fillText(label, x, plotMargin.top + plotHeight + 25);
      }
    });
    
    // Draw Y-axis labels with matching unified formatting
    ctx.textAlign = 'right';
    imagGridValues.forEach(imag => {
      const y = toCanvasY(imag);
      if (y >= plotMargin.top - 10 && y <= plotMargin.top + plotHeight + 10) {
        // Tick mark
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(plotMargin.left - 8, y);
        ctx.lineTo(plotMargin.left, y);
        ctx.stroke();
        
        // Unified label formatting (same as X-axis)
        const label = formatValue(imag);
        
        ctx.fillStyle = '#e5e7eb';
        ctx.fillText(label, plotMargin.left - 12, y + 5);
      }
    });
    
    // Axis titles (light for dark mode)
    ctx.fillStyle = '#f9fafb'; // neutral-50
    ctx.font = 'bold 16px Arial';
    ctx.textAlign = 'center';
    
    // X-axis title
    ctx.fillText('Real (Ω)', width / 2, height - 20);
    
    // Y-axis title
    ctx.save();
    ctx.translate(25, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Imaginary (Ω)', 0, 0);
    ctx.restore();
    
    // Plot title
    ctx.font = 'bold 18px Arial';
    ctx.fillText('Nyquist Plot', width / 2, 30);
    
  }, [width, height, groundTruthImpedance, groupStatistics, plotBounds, showGroundTruth, chromaEnabled, plotWidth, plotHeight, plotMargin]);
  
  return (
    <div className="bg-neutral-800 rounded border border-neutral-600 p-4">
      {/* Median Calculation Toggle */}
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-lg font-medium text-neutral-200">Nyquist Plot</h3>
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-2 text-sm text-neutral-300">
            <input
              type="checkbox"
              checked={medianByResnorm}
              onChange={(e) => setMedianByResnorm(e.target.checked)}
              className="rounded text-blue-600 focus:ring-blue-500 bg-neutral-700 border-neutral-600"
            />
            <span>Median by Resnorm</span>
          </label>
          <div className="text-xs text-neutral-400 max-w-48">
            {medianByResnorm 
              ? "Shows best/median/worst fitting models as complete curves"
              : "Shows min/median/max impedance magnitude at each frequency"
            }
          </div>
        </div>
      </div>
      
      <canvas 
        ref={canvasRef}
        width={width}
        height={height}
        className="border border-neutral-500"
      />
      
      {/* Legend */}
      <div className="mt-4 flex flex-wrap gap-6 text-sm justify-center">
        {showGroundTruth && (
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-white rounded-full"></div>
            <span className="text-neutral-200 font-medium">Current Parameters</span>
          </div>
        )}
        
        {chromaEnabled && visibleResnormGroups.map((group, index) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: group.color }}
            ></div>
            <span className="text-neutral-200">
              {group.label} ({medianByResnorm ? 'resnorm-based' : 'impedance-based'})
            </span>
          </div>
        ))}
      </div>
      
      {visibleResnormGroups.length === 0 && !showGroundTruth && (
        <div className="mt-4 text-center text-neutral-400 text-sm">
          {resnormGroups.length === 0 
            ? "No data available. Run a computation to see impedance analysis."
            : "No groups selected. Select performance groups or unhide layers to see impedance analysis."
          }
        </div>
      )}
    </div>
  );
};
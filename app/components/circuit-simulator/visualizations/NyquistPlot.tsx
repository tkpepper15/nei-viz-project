"use client";

import React, { useMemo, useRef, useEffect } from 'react';
import { ModelSnapshot, ResnormGroup, ImpedancePoint } from '../types';
import { CircuitParameters } from '../types/parameters';

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
}

// Calculate impedance for given parameters
const calculateImpedance = (
  params: CircuitParameters,
  frequencies: number[]
): ImpedancePoint[] => {
  const impedancePoints: ImpedancePoint[] = [];
  
  for (const freq of frequencies) {
    const omega = 2 * Math.PI * freq;
    
    // Calculate individual membrane impedances: Z = R/(1 + jωRC)
    // Za (apical membrane)
    const tauA = params.Ra * params.Ca;
    const denominatorA = 1 + Math.pow(omega * tauA, 2);
    const realA = params.Ra / denominatorA;
    const imagA = -params.Ra * omega * tauA / denominatorA;
    
    // Zb (basal membrane)  
    const tauB = params.Rb * params.Cb;
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

// Calculate median impedance for resnorm groups using consistent frequency spacing
const calculateGroupMedian = (group: ResnormGroup, standardFreqs?: number[]): ImpedancePoint[] => {
  if (group.items.length === 0) {
    return [];
  }
  
  // Use consistent logarithmic frequency spacing
  const frequencies = standardFreqs || generateStandardFrequencies();
  const median: ImpedancePoint[] = [];
  
  for (const freq of frequencies) {
    // Recalculate impedance for each model at this exact frequency for consistency
    const realValues: number[] = [];
    const imagValues: number[] = [];
    
    for (const model of group.items) {
      // Directly calculate impedance for this frequency using model parameters
      if (model.parameters) {
        const impedancePoint = calculateImpedanceAtFrequency(model.parameters, freq);
        realValues.push(impedancePoint.real);
        imagValues.push(impedancePoint.imaginary);
      }
    }
    
    if (realValues.length > 0) {
      realValues.sort((a, b) => a - b);
      imagValues.sort((a, b) => a - b);
      
      const realMedian = realValues[Math.floor(realValues.length / 2)];
      const imagMedian = imagValues[Math.floor(imagValues.length / 2)];
      
      median.push({
        frequency: freq,
        real: realMedian,
        imaginary: imagMedian,
        magnitude: Math.sqrt(realMedian * realMedian + imagMedian * imagMedian),
        phase: Math.atan2(imagMedian, realMedian) * (180 / Math.PI)
      });
    }
  }
  
  return median;
};

// Calculate impedance at a single frequency for consistent spacing
const calculateImpedanceAtFrequency = (params: CircuitParameters, freq: number): ImpedancePoint => {
  const omega = 2 * Math.PI * freq;
  
  // Calculate individual membrane impedances: Z = R/(1 + jωRC)
  // Za (apical membrane)
  const tauA = params.Ra * params.Ca;
  const denominatorA = 1 + Math.pow(omega * tauA, 2);
  const realA = params.Ra / denominatorA;
  const imagA = -params.Ra * omega * tauA / denominatorA;
  
  // Zb (basal membrane)  
  const tauB = params.Rb * params.Cb;
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
  selectedOpacityGroups = [0]
  // Equal step size enforced - no logarithmic scaling options
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Filter groups using both selectedOpacityGroups and hiddenGroups - same logic as spider plots
  const visibleResnormGroups = useMemo(() => {
    // If no groups are selected for opacity, show nothing
    if (selectedOpacityGroups.length === 0) {
      return [];
    }
    
    // Apply both selectedOpacityGroups (performance tiles) and hiddenGroups (layer control) filtering
    return resnormGroups.filter((_, index) => 
      selectedOpacityGroups.includes(index) && !hiddenGroups.includes(index)
    );
  }, [resnormGroups, selectedOpacityGroups, hiddenGroups]);
  
  // Generate consistent frequency range for proper parametric representation
  const frequencies = useMemo(() => {
    if (!groundTruthParams?.frequency_range) {
      return generateStandardFrequencies(100, 1000000, numPoints); // Default: 0.1 kHz to 1000 kHz (100 Hz to 1 MHz)
    }
    
    const [minFreq, maxFreq] = groundTruthParams.frequency_range;
    return generateStandardFrequencies(minFreq, maxFreq, numPoints);
  }, [groundTruthParams, numPoints]);
  
  // Calculate ground truth impedance
  const groundTruthImpedance = useMemo(() => {
    if (!groundTruthParams || !showGroundTruth) return [];
    return calculateImpedance(groundTruthParams, frequencies);
  }, [groundTruthParams, frequencies, showGroundTruth]);
  
  // Calculate group medians with consistent frequency spacing (only for visible groups)
  const groupMedians = useMemo(() => {
    return visibleResnormGroups.map(group => ({
      group,
      median: calculateGroupMedian(group, frequencies)
    }));
  }, [visibleResnormGroups, frequencies]);
  
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
    
    // Include group median bounds
    groupMedians.forEach(({ median }) => {
      median.forEach(point => {
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
    
    // Force equal step sizes for proper data interpretation
    const realRange = maxReal - minReal;
    const imagRange = maxImag - minImag;
    const maxRange = Math.max(realRange, Math.abs(imagRange));
    
    // Expand smaller axis to match larger axis range (crop if needed)
    if (realRange < maxRange) {
      const realCenter = (minReal + maxReal) / 2;
      minReal = realCenter - maxRange / 2;
      maxReal = realCenter + maxRange / 2;
      minReal = Math.max(0, minReal); // Keep real axis non-negative
    }
    
    if (Math.abs(imagRange) < maxRange) {
      const imagCenter = (minImag + maxImag) / 2;
      minImag = imagCenter - maxRange / 2;
      maxImag = imagCenter + maxRange / 2;
      maxImag = Math.min(0, maxImag); // Keep imaginary axis non-positive
    }
    
    // No additional scaling - equal step sizes enforced above
    // Remove all logarithmic scaling features as requested
    
    return {
      minReal,
      maxReal,
      minImag,
      maxImag,
    };
  }, [groundTruthImpedance, groupMedians]); // Removed logScale dependencies
  
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
    
    // Generate orthogonal coordinate plane grid with identical step sizes
    const realRange = maxReal - minReal;
    const imagRange = maxImag - minImag;
    const maxRange = Math.max(realRange, Math.abs(imagRange));
    
    // Calculate unified step size for both axes to ensure orthogonal grid
    const targetSteps = 10;
    const unifiedStepSize = maxRange / targetSteps;
    
    // Generate grid values with identical step sizes
    const realGridValues = [];
    const imagGridValues = [];
    
    // Real axis grid with unified step size
    const realStart = Math.floor(minReal / unifiedStepSize) * unifiedStepSize;
    for (let step = realStart; step <= maxReal + unifiedStepSize; step += unifiedStepSize) {
      if (step >= minReal && step <= maxReal) {
        realGridValues.push(step);
      }
    }
    
    // Imaginary axis grid with identical step size
    const imagStart = Math.floor(minImag / unifiedStepSize) * unifiedStepSize;
    for (let step = imagStart; step <= maxImag + unifiedStepSize; step += unifiedStepSize) {
      if (step >= minImag && step <= maxImag) {
        imagGridValues.push(step);
      }
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
    
    // Draw resnorm group median dots
    if (chromaEnabled) {
      groupMedians.forEach(({ group, median }) => {
        if (median.length > 0) {
          // Draw median dots
          ctx.fillStyle = group.color;
          
          median.forEach((point) => {
            const x = toCanvasX(point.real);
            const y = toCanvasY(point.imaginary);
            
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, 2 * Math.PI); // 4px radius dots
            ctx.fill();
          });
        }
      });
    }
    
    // Draw ground truth impedance as white dots
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
    
    // Create consistent axis labels using unified step size
    const formatValue = (value: number) => {
      const absValue = Math.abs(value);
      if (absValue >= 1000) {
        const kValue = value / 1000;
        return kValue.toFixed(kValue % 1 === 0 ? 0 : 1) + 'k';
      }
      return value.toFixed(value % 1 === 0 ? 0 : 1);
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
    
  }, [width, height, groundTruthImpedance, groupMedians, plotBounds, showGroundTruth, chromaEnabled, plotWidth, plotHeight, plotMargin]);
  
  return (
    <div className="bg-neutral-800 rounded border border-neutral-600 p-4">
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
            <span className="text-neutral-200 font-medium">Ground Truth</span>
          </div>
        )}
        
        {chromaEnabled && visibleResnormGroups.map((group, index) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: group.color }}
            ></div>
            <span className="text-neutral-200">{group.label} (median)</span>
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
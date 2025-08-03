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
  logScaleReal?: boolean;
  logScaleImaginary?: boolean;
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
    
    // Total impedance: Z_total = Rs + Za + Zb (series combination)
    const realTotal = params.Rs + realA + realB;
    const imagTotal = imagA + imagB;
    
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

// Generate consistent coordinate plane grid values
const generateGridValues = (min: number, max: number, isLogScale: boolean, targetCount: number = 10) => {
  const gridValues = [];
  
  if (isLogScale) {
    const logMin = Math.log10(Math.max(1e-10, Math.abs(min)));
    const logMax = Math.log10(Math.abs(max));
    const logStep = (logMax - logMin) / targetCount;
    
    // Major grid lines
    for (let i = 0; i <= targetCount; i++) {
      const logValue = logMin + i * logStep;
      const value = Math.pow(10, logValue);
      gridValues.push(min < 0 ? -value : value);
    }
    
    // Minor grid lines for better granularity
    if ((logMax - logMin) > 2) {
      for (let i = 0; i < targetCount; i++) {
        const logValue = logMin + i * logStep + logStep / 2;
        const value = Math.pow(10, logValue);
        gridValues.push({ value: min < 0 ? -value : value, isMinor: true });
      }
    }
  } else {
    // Linear grid
    const step = (max - min) / targetCount;
    for (let i = 0; i <= targetCount; i++) {
      const value = min + i * step;
      gridValues.push(value);
      
      // Add minor grid lines
      if (i < targetCount) {
        gridValues.push({ value: min + i * step + step / 2, isMinor: true });
      }
    }
  }
  
  return gridValues;
};

// Calculate median impedance for resnorm groups
const calculateGroupMedian = (group: ResnormGroup): ImpedancePoint[] => {
  if (group.items.length === 0) {
    return [];
  }
  
  // Get all unique frequencies from the group models
  const frequencies = group.items[0]?.data?.map(point => point.frequency) || [];
  const median: ImpedancePoint[] = [];
  
  for (const freq of frequencies) {
    // Collect all real and imaginary values at this frequency
    const realValues: number[] = [];
    const imagValues: number[] = [];
    
    for (const model of group.items) {
      const point = model.data?.find(p => Math.abs(p.frequency - freq) < freq * 0.01);
      if (point) {
        realValues.push(point.real);
        imagValues.push(point.imaginary);
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

export const NyquistPlot: React.FC<NyquistPlotProps> = ({
  groundTruthParams,
  resnormGroups,
  visibleModels: _visibleModels, // eslint-disable-line @typescript-eslint/no-unused-vars
  width = 600,
  height = 450,
  showGroundTruth = true,
  chromaEnabled = true,
  logScaleReal = false,
  logScaleImaginary = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Generate frequency range for ground truth calculation
  const frequencies = useMemo(() => {
    if (!groundTruthParams?.frequency_range) {
      // Default frequency range: 0.1 Hz to 10 kHz, 100 points for smooth curves
      const logMin = Math.log10(0.1);
      const logMax = Math.log10(10000);
      const numPoints = 100;
      const logStep = (logMax - logMin) / (numPoints - 1);
      
      const freqs: number[] = [];
      for (let i = 0; i < numPoints; i++) {
        const logValue = logMin + i * logStep;
        freqs.push(Math.pow(10, logValue));
      }
      return freqs;
    }
    
    const [minFreq, maxFreq] = groundTruthParams.frequency_range;
    const logMin = Math.log10(minFreq);
    const logMax = Math.log10(maxFreq);
    const numPoints = 100;
    const logStep = (logMax - logMin) / (numPoints - 1);
    
    const freqs: number[] = [];
    for (let i = 0; i < numPoints; i++) {
      const logValue = logMin + i * logStep;
      freqs.push(Math.pow(10, logValue));
    }
    return freqs;
  }, [groundTruthParams]);
  
  // Calculate ground truth impedance
  const groundTruthImpedance = useMemo(() => {
    if (!groundTruthParams || !showGroundTruth) return [];
    return calculateImpedance(groundTruthParams, frequencies);
  }, [groundTruthParams, frequencies, showGroundTruth]);
  
  // Calculate group medians
  const groupMedians = useMemo(() => {
    return resnormGroups.map(group => ({
      group,
      median: calculateGroupMedian(group)
    }));
  }, [resnormGroups]);
  
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
    
    // Smart logarithmic scale bounds calculation
    if (logScaleReal) {
      // Enhanced log scale for real axis with better distribution
      const positiveMinReal = Math.max(0.01, minReal); // Allow smaller minimum for better range
      const logMin = Math.log10(positiveMinReal);
      const logMax = Math.log10(maxReal);
      const logRange = logMax - logMin;
      
      // Intelligent padding based on data distribution
      let logPadding;
      if (logRange < 1) {
        // Narrow range: use larger padding for better visibility
        logPadding = 0.3;
      } else if (logRange < 3) {
        // Medium range: moderate padding
        logPadding = 0.2;
      } else {
        // Wide range: smaller padding to preserve detail
        logPadding = 0.1;
      }
      
      // Apply smart bounds with decimal-friendly limits
      const newLogMin = logMin - logPadding;
      const newLogMax = logMax + logPadding;
      
      // Round to nice log boundaries (powers of 10 fractions)
      minReal = Math.pow(10, Math.floor(newLogMin * 2) / 2); // Round to 0.5 log units
      maxReal = Math.pow(10, Math.ceil(newLogMax * 2) / 2);
    } else {
      // Linear scale padding
      const realRange = maxReal - minReal;
      const realPadding = realRange * 0.1;
      minReal = Math.max(0, minReal - realPadding);
      maxReal = maxReal + realPadding;
    }
    
    if (logScaleImaginary) {
      // Enhanced log scale for imaginary axis (negative values)
      const absMinImag = Math.abs(maxImag); // Remember: maxImag is closest to 0 (least negative)
      const absMaxImag = Math.abs(minImag); // minImag is most negative
      const positiveAbsMinImag = Math.max(0.01, absMinImag);
      
      const logMin = Math.log10(positiveAbsMinImag);
      const logMax = Math.log10(absMaxImag);
      const logRange = logMax - logMin;
      
      // Intelligent padding for imaginary axis
      let logPadding;
      if (logRange < 1) {
        logPadding = 0.3;
      } else if (logRange < 3) {
        logPadding = 0.2;
      } else {
        logPadding = 0.1;
      }
      
      const newLogMin = logMin - logPadding;
      const newLogMax = logMax + logPadding;
      
      // Round to nice log boundaries and convert back to negative
      const newAbsMinImag = Math.pow(10, Math.floor(newLogMin * 2) / 2);
      const newAbsMaxImag = Math.pow(10, Math.ceil(newLogMax * 2) / 2);
      
      minImag = -newAbsMaxImag;
      maxImag = Math.min(0, -newAbsMinImag);
    } else {
      // Linear scale padding
      const imagRange = maxImag - minImag;
      const imagPadding = imagRange * 0.1;
      minImag = minImag - imagPadding;
      maxImag = Math.min(0, maxImag + imagPadding);
    }
    
    return {
      minReal,
      maxReal,
      minImag,
      maxImag,
    };
  }, [groundTruthImpedance, groupMedians, logScaleReal, logScaleImaginary]);
  
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
      if (logScaleReal) {
        // Clamp to bounds to prevent off-grid values
        const clampedReal = Math.max(minReal * 0.1, Math.min(maxReal * 10, real));
        const logValue = Math.log10(Math.max(1e-10, clampedReal));
        const logMin = Math.log10(minReal);
        const logMax = Math.log10(maxReal);
        const normalizedPosition = (logValue - logMin) / (logMax - logMin);
        
        // Apply slight sigmoid transformation for better distribution at extremes
        const enhancedPosition = 0.5 + 0.45 * Math.tanh(2 * (normalizedPosition - 0.5));
        return plotMargin.left + enhancedPosition * plotWidth;
      }
      return plotMargin.left + ((real - minReal) / (maxReal - minReal)) * plotWidth;
    };
    
    const toCanvasY = (imag: number) => {
      const { minImag, maxImag } = plotBounds;
      if (logScaleImaginary) {
        // Handle negative imaginary values with smart clamping
        const absImag = Math.abs(imag);
        const absMinImag = Math.abs(maxImag);
        const absMaxImag = Math.abs(minImag);
        
        // Clamp to bounds
        const clampedAbsImag = Math.max(absMinImag * 0.1, Math.min(absMaxImag * 10, absImag));
        const logValue = Math.log10(Math.max(1e-10, clampedAbsImag));
        const logMin = Math.log10(Math.max(1e-10, absMinImag));
        const logMax = Math.log10(absMaxImag);
        const normalizedPosition = (logValue - logMin) / (logMax - logMin);
        
        // Apply sigmoid transformation for better distribution
        const enhancedPosition = 0.5 + 0.45 * Math.tanh(2 * (normalizedPosition - 0.5));
        return plotMargin.top + enhancedPosition * plotHeight;
      }
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
    
    // Generate consistent coordinate plane grid
    const realGridValues = generateGridValues(minReal, maxReal, logScaleReal, 10);
    const imagGridValues = generateGridValues(minImag, maxImag, logScaleImaginary, 10);
    
    // Draw vertical grid lines (real axis) with consistent coordinate plane styling
    realGridValues.forEach(gridItem => {
      const real = typeof gridItem === 'number' ? gridItem : gridItem.value;
      const isMinor = typeof gridItem === 'object' && gridItem.isMinor;
      const x = toCanvasX(real);
      
      if (x >= plotMargin.left && x <= plotMargin.left + plotWidth) {
        ctx.beginPath();
        ctx.moveTo(x, plotMargin.top);
        ctx.lineTo(x, plotMargin.top + plotHeight);
        
        // Consistent coordinate plane styling
        ctx.strokeStyle = isMinor ? '#2D3748' : '#4A5568'; // Minor: darker, Major: lighter
        ctx.lineWidth = isMinor ? 0.5 : 1;
        ctx.setLineDash(isMinor ? [2, 2] : []); // Dashed for minor lines
        ctx.stroke();
        ctx.setLineDash([]); // Reset dash
      }
    });
    
    // Draw horizontal grid lines (imaginary axis) with matching coordinate plane styling
    imagGridValues.forEach(gridItem => {
      const imag = typeof gridItem === 'number' ? gridItem : gridItem.value;
      const isMinor = typeof gridItem === 'object' && gridItem.isMinor;
      const y = toCanvasY(imag);
      
      if (y >= plotMargin.top && y <= plotMargin.top + plotHeight) {
        ctx.beginPath();
        ctx.moveTo(plotMargin.left, y);
        ctx.lineTo(plotMargin.left + plotWidth, y);
        
        // Match vertical grid line styling exactly
        ctx.strokeStyle = isMinor ? '#2D3748' : '#4A5568';
        ctx.lineWidth = isMinor ? 0.5 : 1;
        ctx.setLineDash(isMinor ? [2, 2] : []);
        ctx.stroke();
        ctx.setLineDash([]);
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
    
    // Generate coordinate plane axis labels (major grid lines only)
    const realLabelValues = realGridValues
      .filter(item => typeof item === 'number') // Only major grid lines
      .filter((_, index) => index % 2 === 0); // Every other major line for readability
    
    // Draw X-axis labels with consistent formatting
    realLabelValues.forEach(real => {
      const x = toCanvasX(real);
      if (x >= plotMargin.left - 20 && x <= plotMargin.left + plotWidth + 20) {
        // Tick mark
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, plotMargin.top + plotHeight);
        ctx.lineTo(x, plotMargin.top + plotHeight + 8);
        ctx.stroke();
        
        // Consistent label formatting
        let label;
        if (logScaleReal) {
          if (real >= 1000000) {
            label = (real / 1000000).toFixed(real >= 10000000 ? 0 : 1) + 'M';
          } else if (real >= 1000) {
            label = (real / 1000).toFixed(real >= 10000 ? 0 : 1) + 'k';
          } else if (real >= 1) {
            label = real.toFixed(real >= 10 ? 0 : 1);
          } else if (real >= 0.001) {
            label = (real * 1000).toFixed(1) + 'm';
          } else {
            label = real.toExponential(0);
          }
        } else {
          label = real >= 1000 ? (real / 1000).toFixed(1) + 'k' : real.toFixed(0);
        }
        
        ctx.fillStyle = '#e5e7eb';
        ctx.fillText(label, x, plotMargin.top + plotHeight + 25);
      }
    });
    
    // Generate Y-axis labels matching X-axis density
    ctx.textAlign = 'right';
    const imagLabelValues = imagGridValues
      .filter(item => typeof item === 'number') // Only major grid lines
      .filter((_, index) => index % 2 === 0); // Every other major line for readability
    
    // Draw Y-axis labels with consistent formatting
    imagLabelValues.forEach(imag => {
      const y = toCanvasY(imag);
      if (y >= plotMargin.top - 10 && y <= plotMargin.top + plotHeight + 10) {
        // Tick mark
        ctx.strokeStyle = '#e5e7eb';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(plotMargin.left - 8, y);
        ctx.lineTo(plotMargin.left, y);
        ctx.stroke();
        
        // Consistent label formatting (matching X-axis style)
        const absImag = Math.abs(imag);
        let label;
        if (logScaleImaginary) {
          if (absImag >= 1000000) {
            label = '-' + (absImag / 1000000).toFixed(absImag >= 10000000 ? 0 : 1) + 'M';
          } else if (absImag >= 1000) {
            label = '-' + (absImag / 1000).toFixed(absImag >= 10000 ? 0 : 1) + 'k';
          } else if (absImag >= 1) {
            label = imag.toFixed(absImag >= 10 ? 0 : 1);
          } else if (absImag >= 0.001) {
            label = '-' + (absImag * 1000).toFixed(1) + 'm';
          } else {
            label = imag.toExponential(0);
          }
        } else {
          label = absImag >= 1000 ? '-' + (absImag / 1000).toFixed(1) + 'k' : imag.toFixed(0);
        }
        
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
    
  }, [width, height, groundTruthImpedance, groupMedians, plotBounds, showGroundTruth, chromaEnabled, plotWidth, plotHeight, plotMargin, logScaleReal, logScaleImaginary]);
  
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
        
        {chromaEnabled && resnormGroups.map((group, index) => (
          <div key={index} className="flex items-center gap-2">
            <div 
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: group.color }}
            ></div>
            <span className="text-neutral-200">{group.label} (median)</span>
          </div>
        ))}
      </div>
      
      {resnormGroups.length === 0 && !showGroundTruth && (
        <div className="mt-4 text-center text-neutral-400 text-sm">
          No data available. Run a computation to see impedance analysis.
        </div>
      )}
    </div>
  );
};
"use client";

import React, { useMemo, useRef, useEffect, useState, useCallback } from 'react';
import { SerializedComputationManager } from '../utils/serializedComputationManager';

interface ResidualPlotProps {
  manager: SerializedComputationManager | null;
  className?: string;
}

interface GridResidualData {
  parameterName: string;
  resnormValues: number[];
  gridResiduals: number[];
  frequencies: number[];
  colors: string[];
}

interface FrequencyHistogramData {
  bins: { min: number; max: number; count: number; color: string; label?: string }[];
  maxCount: number;
}

export const ResidualPlot: React.FC<ResidualPlotProps> = ({
  manager,
  className = ''
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedParameter, setSelectedParameter] = useState<string>('Rsh');

  // Export function for all residual data using efficient SRD format
  const exportAllResidualData = useCallback(() => {
    if (!manager) return;

    const allSnapshots = manager.generateModelSnapshots(); // Get ALL data
    console.log(`📤 Exporting ${allSnapshots.length.toLocaleString()} data points to efficient SRD format`);

    // Use the efficient SRD export format (ultra-compressed)
    const timestamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
    const filename = `residual-analysis-${timestamp}.json`;

    manager.exportToSRD(filename, {
      title: `Residual Analysis Export`,
      description: `Complete dataset export for residual plot analysis with ${allSnapshots.length.toLocaleString()} models`,
      author: 'SpideyPlot ResidualPlot Component',
      tags: ['residual-analysis', 'complete-dataset', 'grid-residuals']
    });

    console.log(`✅ Exported ${allSnapshots.length.toLocaleString()} data points to ultra-efficient SRD format (98% compression)`);
  }, [manager]);


  // Process data for grid residual analysis visualization
  const gridResidualData = useMemo(() => {
    if (!manager) return null;

    // Get results from SerializedComputationManager - get ALL results for comprehensive analysis
    const modelSnapshots = manager.generateModelSnapshots(); // Get all available results for complete analysis

    if (!modelSnapshots || modelSnapshots.length === 0) return null;

    console.log(`🔍 ResidualPlot: Processing ${modelSnapshots.length.toLocaleString()} model snapshots for grid residual analysis`);

    const parameterNames = ['Rsh', 'Ra', 'Ca', 'Rb', 'Cb'];
    const gridDataMap: Record<string, GridResidualData> = {};

    // Collect all unique frequency points from spectrum data
    const allFrequencyPoints = new Set<number>();
    modelSnapshots.forEach(snapshot => {
      if (snapshot.data && snapshot.data.length > 0) {
        snapshot.data.forEach(point => {
          allFrequencyPoints.add(point.frequency);
        });
      }
    });

    const uniqueFrequencies = Array.from(allFrequencyPoints).sort((a, b) => a - b);

    // If no frequency data found, use default frequency range for fallback
    if (uniqueFrequencies.length === 0) {
      console.log(`⚠️ ResidualPlot: No spectrum frequency data found, using default frequency range`);
      // Generate default frequencies for percentile grouping
      const defaultFreqs = [];
      for (let i = 0; i < 20; i++) {
        const freq = Math.pow(10, 0.1 + (i / 19) * 5); // Log scale from 1.26 Hz to 100 kHz
        defaultFreqs.push(freq);
      }
      uniqueFrequencies.push(...defaultFreqs);
      uniqueFrequencies.sort((a, b) => a - b);
    }

    console.log(`🌈 ResidualPlot: Found ${uniqueFrequencies.length} unique frequency points from ${uniqueFrequencies[0].toFixed(2)} to ${uniqueFrequencies[uniqueFrequencies.length - 1].toFixed(2)} Hz`);

    // Create frequency percentile groups
    const frequencyPercentileGroups = [
      { range: [0, 20], label: '0-20th percentile', color: '#ef4444' }, // Red - Low frequencies
      { range: [20, 40], label: '20-40th percentile', color: '#f97316' }, // Orange
      { range: [40, 60], label: '40-60th percentile', color: '#eab308' }, // Yellow
      { range: [60, 80], label: '60-80th percentile', color: '#22c55e' }, // Green
      { range: [80, 100], label: '80-100th percentile', color: '#3b82f6' }, // Blue - High frequencies
    ];

    // Function to get percentile group for a frequency
    const getFrequencyPercentileGroup = (frequency: number) => {
      const index = uniqueFrequencies.indexOf(frequency);
      if (index === -1) return frequencyPercentileGroups[2]; // Default to middle group if not found

      const percentile = (index / (uniqueFrequencies.length - 1)) * 100;

      for (const group of frequencyPercentileGroups) {
        if (percentile >= group.range[0] && percentile <= group.range[1]) {
          return group;
        }
      }
      return frequencyPercentileGroups[2]; // Default to middle group
    };

    // Determine ground truth values (use best resnorm as reference)
    const sortedByResnorm = [...modelSnapshots].sort((a, b) => (a.resnorm || Infinity) - (b.resnorm || Infinity));
    const groundTruth = sortedByResnorm[0]?.parameters; // Best fit as ground truth reference

    if (!groundTruth) return null;

    parameterNames.forEach(paramName => {
      const resnormValues: number[] = [];
      const gridResiduals: number[] = [];
      const frequencies: number[] = [];
      const colors: string[] = [];

      // Extract data for this parameter from ModelSnapshots
      // For display optimization: sample large datasets while keeping all data for export
      const maxDisplayPoints = 50000; // Display limit for performance
      const sampleStep = Math.max(1, Math.floor(modelSnapshots.length / maxDisplayPoints));
      const displaySnapshots = modelSnapshots.filter((_, index) => index % sampleStep === 0);

      console.log(`📊 ${paramName}: Processing ${modelSnapshots.length.toLocaleString()} total, displaying ${displaySnapshots.length.toLocaleString()}`);

      displaySnapshots.forEach((snapshot) => {
        if (snapshot.parameters && snapshot.resnorm !== undefined) {
          const paramValue = snapshot.parameters[paramName as keyof typeof snapshot.parameters];
          const groundTruthValue = groundTruth[paramName as keyof typeof groundTruth];

          if (paramValue !== undefined && typeof paramValue === 'number' &&
              groundTruthValue !== undefined && typeof groundTruthValue === 'number') {

            resnormValues.push(snapshot.resnorm);

            // Calculate grid residual: how far off this parameter is from ground truth
            // Expressed as percentage deviation from ground truth
            let gridResidual: number;
            if (groundTruthValue !== 0) {
              gridResidual = ((paramValue - groundTruthValue) / groundTruthValue) * 100; // Percentage deviation
            } else {
              gridResidual = paramValue * 100; // If ground truth is 0, show absolute scaled value
            }

            gridResiduals.push(gridResidual);

            // Use actual frequency points from impedance spectrum grouped by percentile
            if (snapshot.data && snapshot.data.length > 0) {
              // Get all frequency points from this spectrum
              const spectrumFrequencies = snapshot.data.map(point => point.frequency);

              // Use the median frequency point for this snapshot's representation
              const sortedFreqs = [...spectrumFrequencies].sort((a, b) => a - b);
              const medianFreq = sortedFreqs[Math.floor(sortedFreqs.length / 2)];

              // Get the percentile group for this frequency
              const percentileGroup = getFrequencyPercentileGroup(medianFreq);

              frequencies.push(medianFreq);
              colors.push(percentileGroup.color);
            } else {
              // Fallback to middle frequency range if no spectrum data
              const middleFreq = uniqueFrequencies.length > 0 ?
                uniqueFrequencies[Math.floor(uniqueFrequencies.length / 2)] :
                1000; // 1kHz fallback

              const percentileGroup = getFrequencyPercentileGroup(middleFreq);
              frequencies.push(middleFreq);
              colors.push(percentileGroup.color);
            }
          }
        }
      });

      gridDataMap[paramName] = {
        parameterName: paramName,
        resnormValues,
        gridResiduals,
        frequencies,
        colors
      };

      console.log(`📊 ResidualPlot: ${paramName} parameter has ${resnormValues.length.toLocaleString()} data points`);
    });

    return gridDataMap;
  }, [manager]);

  // Generate frequency histogram data based on percentile groups
  const histogramData = useMemo((): FrequencyHistogramData | null => {
    if (!gridResidualData || !gridResidualData[selectedParameter]) return null;

    const data = gridResidualData[selectedParameter];
    if (data.frequencies.length === 0) return null;

    // Count frequencies by their percentile groups
    const percentileGroups = [
      { range: [0, 20], label: '0-20th percentile', color: '#ef4444' }, // Red - Low frequencies
      { range: [20, 40], label: '20-40th percentile', color: '#f97316' }, // Orange
      { range: [40, 60], label: '40-60th percentile', color: '#eab308' }, // Yellow
      { range: [60, 80], label: '60-80th percentile', color: '#22c55e' }, // Green
      { range: [80, 100], label: '80-100th percentile', color: '#3b82f6' }, // Blue - High frequencies
    ];

    const colorCounts = new Map<string, { count: number; minFreq: number; maxFreq: number; label: string }>();

    // Count occurrences of each color (percentile group)
    data.frequencies.forEach((freq, index) => {
      const color = data.colors[index];
      const group = percentileGroups.find(g => g.color === color);
      const label = group?.label || 'Unknown';

      if (!colorCounts.has(color)) {
        colorCounts.set(color, { count: 0, minFreq: freq, maxFreq: freq, label });
      }

      const existing = colorCounts.get(color)!;
      existing.count++;
      existing.minFreq = Math.min(existing.minFreq, freq);
      existing.maxFreq = Math.max(existing.maxFreq, freq);
    });

    // Convert to histogram bins format
    const bins = Array.from(colorCounts.entries()).map(([color, data]) => ({
      min: data.minFreq,
      max: data.maxFreq,
      count: data.count,
      color: color,
      label: data.label
    })).sort((a, b) => a.min - b.min); // Sort by frequency

    const maxCount = Math.max(...bins.map(bin => bin.count));

    return { bins, maxCount };
  }, [gridResidualData, selectedParameter]);

  // Helper functions
  const getParameterUnit = useCallback((param: string): string => {
    switch (param) {
      case 'Rsh':
      case 'Ra':
      case 'Rb':
        return '(Ω)';
      case 'Ca':
      case 'Cb':
        return '(µF)';
      default:
        return '';
    }
  }, []);

  const formatTickValue = useCallback((value: number): string => {
    if (Math.abs(value) < 0.001) return value.toExponential(1);
    if (Math.abs(value) < 1) return value.toFixed(3);
    if (Math.abs(value) < 1000) return value.toFixed(1);
    return value.toExponential(1);
  }, []);

  // Render the main plot
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !gridResidualData || !gridResidualData[selectedParameter]) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { width, height } = canvas;
    const data = gridResidualData[selectedParameter];

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Set up plot margins
    const margin = { top: 40, right: 60, bottom: 80, left: 100 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    if (data.resnormValues.length === 0) return;

    // Calculate scales - X: resnorm, Y: grid residual
    const xMin = Math.min(...data.resnormValues);
    const xMax = Math.max(...data.resnormValues);
    const yMin = Math.min(...data.gridResiduals);
    const yMax = Math.max(...data.gridResiduals);

    // Add padding to Y-axis to show positive and negative deviations clearly
    const yRange = yMax - yMin;
    const yPadding = yRange * 0.1;
    const yMinPadded = yMin - yPadding;
    const yMaxPadded = yMax + yPadding;

    const xScale = (x: number) => margin.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
    const yScale = (y: number) => margin.top + plotHeight - ((y - yMinPadded) / (yMaxPadded - yMinPadded)) * plotHeight;

    // Draw axes
    ctx.strokeStyle = '#6b7280';
    ctx.lineWidth = 1;

    // X-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + plotHeight);
    ctx.lineTo(margin.left + plotWidth, margin.top + plotHeight);
    ctx.stroke();

    // Y-axis
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top);
    ctx.lineTo(margin.left, margin.top + plotHeight);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 0.5;

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = margin.left + (i / 10) * plotWidth;
      ctx.beginPath();
      ctx.moveTo(x, margin.top);
      ctx.lineTo(x, margin.top + plotHeight);
      ctx.stroke();
    }

    // Horizontal grid lines
    for (let i = 0; i <= 10; i++) {
      const y = margin.top + (i / 10) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(margin.left, y);
      ctx.lineTo(margin.left + plotWidth, y);
      ctx.stroke();
    }

    // Draw zero line for grid residual reference
    const zeroY = yScale(0);
    if (zeroY >= margin.top && zeroY <= margin.top + plotHeight) {
      ctx.strokeStyle = '#ef4444'; // Red line for zero reference
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(margin.left, zeroY);
      ctx.lineTo(margin.left + plotWidth, zeroY);
      ctx.stroke();
      ctx.setLineDash([]); // Reset line dash
    }

    // Draw data points - no outlines, just color
    data.resnormValues.forEach((resnormValue, i) => {
      const x = xScale(resnormValue);
      const y = yScale(data.gridResiduals[i]);

      ctx.fillStyle = data.colors[i];
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI); // Slightly larger dots without outline
      ctx.fill();
    });

    // Draw axis labels
    ctx.fillStyle = '#e5e7eb';
    ctx.font = '14px Inter, sans-serif';
    ctx.textAlign = 'center';

    // X-axis label
    ctx.fillText(
      'Resnorm Value',
      margin.left + plotWidth / 2,
      height - 20
    );

    // Y-axis label
    ctx.save();
    ctx.translate(20, margin.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(`Grid Residual: ${selectedParameter} (% deviation)`, 0, 0);
    ctx.restore();

    // Draw axis tick labels
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#9ca3af';

    // X-axis ticks (resnorm values)
    for (let i = 0; i <= 5; i++) {
      const value = xMin + (i / 5) * (xMax - xMin);
      const x = margin.left + (i / 5) * plotWidth;
      ctx.textAlign = 'center';
      ctx.fillText(formatTickValue(value), x, margin.top + plotHeight + 20);
    }

    // Y-axis ticks (grid residuals as percentages)
    for (let i = 0; i <= 5; i++) {
      const value = yMinPadded + (i / 5) * (yMaxPadded - yMinPadded);
      const y = margin.top + plotHeight - (i / 5) * plotHeight;
      ctx.textAlign = 'right';
      ctx.fillText(formatTickValue(value) + '%', margin.left - 10, y + 4);
    }

    // Draw title
    ctx.fillStyle = '#f3f4f6';
    ctx.font = 'bold 16px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(
      `Grid Fidelity Analysis: ${selectedParameter} Parameter`,
      width / 2,
      25
    );

  }, [gridResidualData, selectedParameter, getParameterUnit, formatTickValue]);

  if (!manager || !gridResidualData) {
    return (
      <div className={`flex items-center justify-center h-96 bg-neutral-900 rounded-lg ${className}`}>
        <div className="text-center text-neutral-400">
          <svg className="w-16 h-16 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4m0 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="text-lg font-medium">No Data Available</p>
          <p className="text-sm mt-1">Run a computation or load data to view grid fidelity analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-neutral-900 rounded-lg p-4 ${className}`}>
      {/* Parameter Selection */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-4">
            <label className="text-sm font-medium text-neutral-200">Parameter:</label>
            <select
              value={selectedParameter}
              onChange={(e) => setSelectedParameter(e.target.value)}
              className="bg-neutral-800 border border-neutral-600 rounded px-3 py-1 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="Rsh">Rsh (Shunt Resistance)</option>
              <option value="Ra">Ra (Resistance A)</option>
              <option value="Ca">Ca (Capacitance A)</option>
              <option value="Rb">Rb (Resistance B)</option>
              <option value="Cb">Cb (Capacitance B)</option>
            </select>
          </div>

          <button
            onClick={exportAllResidualData}
            className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md transition-colors flex items-center gap-2"
            title="Export all residual data to efficient SRD format"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-4-4m4 4l4-4M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
            </svg>
            Export SRD File
          </button>
        </div>

        <div className="text-xs text-neutral-400">
          Showing grid fidelity analysis: resnorm vs parameter deviation from ground truth.
          <span className="text-blue-400"> Colors represent frequency percentile groups from actual impedance spectrum data.</span>
        </div>
      </div>

      <div className="flex gap-4">
        {/* Main Plot */}
        <div className="flex-1">
          <canvas
            ref={canvasRef}
            width={600}
            height={400}
            className="w-full border border-neutral-700 rounded bg-neutral-800"
          />
        </div>

        {/* Frequency Histogram Sidebar */}
        <div className="w-48 bg-neutral-800 rounded border border-neutral-700 p-3">
          <h4 className="text-sm font-medium text-neutral-200 mb-3 flex items-center gap-2">
            <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v4m0 0a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Frequency Distribution
          </h4>

          {histogramData ? (
            <div className="space-y-2">
              {histogramData.bins.map((bin, i) => (
                <div key={i} className="space-y-1">
                  <div className="flex items-center gap-2 text-xs">
                    <div
                      className="w-3 h-3 rounded-sm border border-neutral-600"
                      style={{ backgroundColor: bin.color }}
                    />
                    <div className="flex-1">
                      <div className="text-neutral-300 font-medium">
                        {bin.label || `Group ${i + 1}`}
                      </div>
                      <div className="flex justify-between text-neutral-400 text-xs">
                        <span>{formatTickValue(bin.min)} - {formatTickValue(bin.max)} Hz</span>
                        <span className="text-neutral-500">{bin.count}</span>
                      </div>
                      <div className="w-full bg-neutral-700 rounded-full h-1.5 mt-1">
                        <div
                          className="h-1.5 rounded-full"
                          style={{
                            width: `${(bin.count / histogramData.maxCount) * 100}%`,
                            backgroundColor: bin.color
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-neutral-500">No frequency data</div>
          )}

          {/* Enhanced Color Legend */}
          <div className="mt-4 pt-3 border-t border-neutral-700">
            <div className="text-xs text-neutral-400 mb-2">Frequency Percentile Groups</div>
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-2 bg-red-500 rounded-sm" />
                <span className="text-neutral-300">0-20%</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-2 bg-orange-500 rounded-sm" />
                <span className="text-neutral-300">20-40%</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-2 bg-yellow-500 rounded-sm" />
                <span className="text-neutral-300">40-60%</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-2 bg-green-500 rounded-sm" />
                <span className="text-neutral-300">60-80%</span>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <div className="w-3 h-2 bg-blue-500 rounded-sm" />
                <span className="text-neutral-300">80-100%</span>
              </div>
            </div>
            <div className="text-xs text-neutral-500 mt-2">
              {histogramData && gridResidualData && gridResidualData[selectedParameter] ? (
                <>
                  Range: {formatTickValue(Math.min(...gridResidualData[selectedParameter].frequencies))} -
                  {formatTickValue(Math.max(...gridResidualData[selectedParameter].frequencies))} Hz
                </>
              ) : (
                'Based on actual impedance spectrum frequencies'
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Data Summary */}
      {gridResidualData && gridResidualData[selectedParameter] && gridResidualData[selectedParameter].resnormValues.length > 0 && (
        <div className="mt-4 p-3 bg-neutral-800 rounded border border-neutral-700">
          <div className="text-xs text-neutral-400 grid grid-cols-3 gap-4">
            <div>
              <span className="font-medium text-neutral-300">Data Points:</span> {gridResidualData[selectedParameter].resnormValues.length}
            </div>
            <div>
              <span className="font-medium text-neutral-300">Min Resnorm:</span> {formatTickValue(Math.min(...gridResidualData[selectedParameter].resnormValues))}
            </div>
            <div>
              <span className="font-medium text-neutral-300">Max Deviation:</span> {formatTickValue(Math.max(...gridResidualData[selectedParameter].gridResiduals.map(Math.abs)))}%
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
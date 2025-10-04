/**
 * TER/TEC Analysis Tab Component
 * ===============================
 *
 * Filters circuit models by constant TER or TEC values and displays IQR analysis
 * for the underlying 5 circuit parameters. This helps identify parameter patterns
 * when only TER/TEC measurements are known (equilibrium state).
 */

'use client';

import React, { useMemo, useState } from 'react';
import { ModelSnapshot } from '../../types';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';
import { calculateTER, calculateTEC, formatTER, formatTEC, getTERRange, getTECRange } from '../../utils/terTecCalculations';

interface FiveNumberSummary {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  iqr: number;
  count: number;
}

interface TERTECGroup {
  value: number;
  type: 'TER' | 'TEC';
  models: ModelSnapshot[];
  parameterSummaries: {
    Rsh: FiveNumberSummary;
    Ra: FiveNumberSummary;
    Rb: FiveNumberSummary;
    Ca: FiveNumberSummary;
    Cb: FiveNumberSummary;
  };
}

// Calculate five-number summary with IQR
const calculateFiveNumberSummary = (values: number[]): FiveNumberSummary => {
  if (values.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0, iqr: 0, count: 0 };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const count = sorted.length;

  const min = sorted[0];
  const max = sorted[count - 1];

  const medianIndex = Math.floor(count / 2);
  const median = count % 2 === 0
    ? (sorted[medianIndex - 1] + sorted[medianIndex]) / 2
    : sorted[medianIndex];

  const lowerHalf = sorted.slice(0, medianIndex);
  const q1Index = Math.floor(lowerHalf.length / 2);
  const q1 = lowerHalf.length % 2 === 0 && lowerHalf.length > 1
    ? (lowerHalf[q1Index - 1] + lowerHalf[q1Index]) / 2
    : lowerHalf[q1Index] || min;

  const upperHalf = count % 2 === 0
    ? sorted.slice(medianIndex)
    : sorted.slice(medianIndex + 1);
  const q3Index = Math.floor(upperHalf.length / 2);
  const q3 = upperHalf.length % 2 === 0 && upperHalf.length > 1
    ? (upperHalf[q3Index - 1] + upperHalf[q3Index]) / 2
    : upperHalf[q3Index] || max;

  const iqr = q3 - q1;

  return { min, q1, median, q3, max, iqr, count };
};

// Format parameter value for display
const formatParameterValue = (param: 'Rsh' | 'Ra' | 'Rb' | 'Ca' | 'Cb', value: number): string => {
  if (param === 'Ca' || param === 'Cb') {
    return `${(value * 1e6).toFixed(2)} µF`;
  }
  return `${value.toFixed(1)} Ω`;
};

// Distribution Plot Component
interface TERTECDistributionPlotProps {
  models: Array<ModelSnapshot & { ter: number; tec: number }>;
  filterType: 'TER' | 'TEC';
  selectedValue: number;
  tolerance: number;
}

const TERTECDistributionPlot: React.FC<TERTECDistributionPlotProps> = ({
  models,
  filterType,
  selectedValue,
  tolerance
}) => {
  const canvasRef = React.useRef<HTMLCanvasElement>(null);

  React.useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || models.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Canvas dimensions
    const width = canvas.width;
    const height = canvas.height;
    const padding = 50;
    const plotWidth = width - 2 * padding;
    const plotHeight = height - 2 * padding;

    // Clear canvas
    ctx.fillStyle = '#171717'; // neutral-900
    ctx.fillRect(0, 0, width, height);

    // Extract data
    const dataPoints = models.map(m => ({
      x: filterType === 'TER' ? m.ter : m.tec,
      y: m.resnorm,
      model: m
    })).filter(p => p.x > 0 && p.y !== undefined && p.y > 0);

    if (dataPoints.length === 0) return;

    // Find ranges
    const xValues = dataPoints.map(p => p.x);
    const yValues = dataPoints.map(p => p.y).filter((y): y is number => y !== undefined);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);

    // Add 10% padding to ranges
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;
    const xMinPadded = xMin - xRange * 0.1;
    const xMaxPadded = xMax + xRange * 0.1;
    const yMinPadded = yMin - yRange * 0.1;
    const yMaxPadded = yMax + yRange * 0.1;

    // Scale functions
    const scaleX = (x: number) => padding + ((x - xMinPadded) / (xMaxPadded - xMinPadded)) * plotWidth;
    const scaleY = (y: number) => height - padding - ((y - yMinPadded) / (yMaxPadded - yMinPadded)) * plotHeight;

    // Draw axes
    ctx.strokeStyle = '#525252'; // neutral-600
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.stroke();

    // Draw grid lines
    ctx.strokeStyle = '#262626'; // neutral-800
    ctx.lineWidth = 1;
    const gridLines = 5;
    for (let i = 0; i <= gridLines; i++) {
      const x = padding + (plotWidth / gridLines) * i;
      const y = height - padding - (plotHeight / gridLines) * i;

      // Vertical grid lines
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, height - padding);
      ctx.stroke();

      // Horizontal grid lines
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding, y);
      ctx.stroke();
    }

    // Calculate selected range
    const toleranceFraction = tolerance / 100;
    const selectedMin = selectedValue * (1 - toleranceFraction);
    const selectedMax = selectedValue * (1 + toleranceFraction);

    // Draw points
    dataPoints.forEach(point => {
      if (point.y === undefined) return; // Type guard

      const x = scaleX(point.x);
      const y = scaleY(point.y);

      const isSelected = selectedValue > 0 && point.x >= selectedMin && point.x <= selectedMax;

      ctx.beginPath();
      ctx.arc(x, y, isSelected ? 4 : 2, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? '#3b82f6' : '#737373'; // blue-500 or neutral-500
      ctx.fill();
    });

    // Draw selected range highlight
    if (selectedValue > 0) {
      ctx.fillStyle = 'rgba(59, 130, 246, 0.1)'; // blue-500 with alpha
      ctx.fillRect(
        scaleX(selectedMin),
        padding,
        scaleX(selectedMax) - scaleX(selectedMin),
        plotHeight
      );
    }

    // Draw axis labels
    ctx.fillStyle = '#a3a3a3'; // neutral-400
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';

    // X-axis label
    ctx.fillText(
      filterType,
      width / 2,
      height - 10
    );

    // Y-axis label
    ctx.save();
    ctx.translate(15, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('Resnorm', 0, 0);
    ctx.restore();

    // Draw tick labels
    ctx.font = '10px monospace';
    for (let i = 0; i <= gridLines; i++) {
      const xValue = xMinPadded + (xMaxPadded - xMinPadded) * (i / gridLines);
      const yValue = yMinPadded + (yMaxPadded - yMinPadded) * (i / gridLines);

      // X-axis ticks
      ctx.textAlign = 'center';
      ctx.fillText(
        xValue.toExponential(1),
        padding + (plotWidth / gridLines) * i,
        height - padding + 20
      );

      // Y-axis ticks
      ctx.textAlign = 'right';
      ctx.fillText(
        yValue.toExponential(1),
        padding - 10,
        height - padding - (plotHeight / gridLines) * i + 4
      );
    }

  }, [models, filterType, selectedValue, tolerance]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      className="w-full h-auto bg-neutral-900 rounded"
    />
  );
};

export const TERTECAnalysisTab: React.FC<BottomPanelTabProps> = ({
  gridResults,
  onTERTECFilterChange
}) => {
  const [filterType, setFilterType] = useState<'TER' | 'TEC'>('TER');
  const [targetValue, setTargetValue] = useState<number>(0);
  const [tolerance, setTolerance] = useState<number>(5); // Percentage
  const [selectedGroup, setSelectedGroup] = useState<TERTECGroup | null>(null);
  const [visualizationApplied, setVisualizationApplied] = useState<boolean>(false);

  // Calculate TER and TEC for all models
  const modelsWithMetrics = useMemo(() => {
    return gridResults.map(model => {
      const ter = calculateTER(model.parameters);
      const tec = calculateTEC(model.parameters);
      return {
        ...model,
        ter,
        tec
      };
    });
  }, [gridResults]);

  // Get unique TER/TEC values for selection
  const uniqueValues = useMemo(() => {
    const values = modelsWithMetrics.map(m => filterType === 'TER' ? m.ter : m.tec);
    const roundedValues = values.map(v => {
      if (filterType === 'TEC') {
        // Round to 3 significant figures for TEC (small values)
        const magnitude = Math.floor(Math.log10(Math.abs(v)));
        const scale = Math.pow(10, magnitude - 2);
        return Math.round(v / scale) * scale;
      }
      return Math.round(v * 1000) / 1000; // Round to 3 decimals for TER
    });
    const uniqueSet = new Set(roundedValues.filter(v => v > 0));
    return Array.from(uniqueSet).sort((a, b) => a - b);
  }, [modelsWithMetrics, filterType]);

  // Filter models by selected TER/TEC value
  const filteredGroup = useMemo(() => {
    if (targetValue === 0 || uniqueValues.length === 0) return null;

    const toleranceFraction = tolerance / 100;
    const range = filterType === 'TER'
      ? getTERRange(targetValue, toleranceFraction)
      : getTECRange(targetValue, toleranceFraction);

    const matchingModels = modelsWithMetrics.filter(m => {
      const value = filterType === 'TER' ? m.ter : m.tec;
      return value >= range.min && value <= range.max;
    });

    if (matchingModels.length === 0) return null;

    // Calculate parameter summaries
    const parameterSummaries = {
      Rsh: calculateFiveNumberSummary(matchingModels.map(m => m.parameters.Rsh)),
      Ra: calculateFiveNumberSummary(matchingModels.map(m => m.parameters.Ra)),
      Rb: calculateFiveNumberSummary(matchingModels.map(m => m.parameters.Rb)),
      Ca: calculateFiveNumberSummary(matchingModels.map(m => m.parameters.Ca)),
      Cb: calculateFiveNumberSummary(matchingModels.map(m => m.parameters.Cb))
    };

    return {
      value: targetValue,
      type: filterType,
      models: matchingModels,
      parameterSummaries
    };
  }, [targetValue, tolerance, filterType, modelsWithMetrics, uniqueValues]);

  // Update selected group when filter changes
  React.useEffect(() => {
    setSelectedGroup(filteredGroup);
    setVisualizationApplied(false); // Reset when filter changes
  }, [filteredGroup]);

  // Reset visualization applied when filter type changes
  React.useEffect(() => {
    setVisualizationApplied(false);
    setTargetValue(0); // Reset target value when switching type
  }, [filterType]);

  // Initialize target value to first available value
  React.useEffect(() => {
    if (targetValue === 0 && uniqueValues.length > 0) {
      setTargetValue(uniqueValues[Math.floor(uniqueValues.length / 2)]);
    }
  }, [uniqueValues, targetValue]);

  if (gridResults.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-neutral-400">
        <div className="text-center">
          <p className="text-lg mb-2">No data available for TER/TEC analysis</p>
          <p className="text-sm">Run a parameter grid computation to generate analysis data</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      {/* Header and Controls */}
      <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h2 className="text-lg font-semibold text-neutral-200">TER/TEC Parameter Analysis</h2>
            <p className="text-sm text-neutral-400">
              Filter models by constant TER or TEC values to identify underlying parameter patterns
            </p>
          </div>
          <div className="text-right">
            <div className="text-xs text-neutral-500">Total Models</div>
            <div className="text-2xl font-bold text-neutral-200">{gridResults.length.toLocaleString()}</div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Filter Type Selection */}
          <div>
            <label className="block text-xs font-medium text-neutral-400 mb-2">
              Metric Type
            </label>
            <div className="flex gap-2">
              <button
                onClick={() => setFilterType('TER')}
                className={`flex-1 px-4 py-2 rounded transition-colors ${
                  filterType === 'TER'
                    ? 'bg-blue-600 text-white'
                    : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                TER
              </button>
              <button
                onClick={() => setFilterType('TEC')}
                className={`flex-1 px-4 py-2 rounded transition-colors ${
                  filterType === 'TEC'
                    ? 'bg-blue-600 text-white'
                    : 'bg-neutral-700 text-neutral-300 hover:bg-neutral-600'
                }`}
              >
                TEC
              </button>
            </div>
          </div>

          {/* Target Value Selection - Dropdown */}
          <div>
            <label className="block text-xs font-medium text-neutral-400 mb-2">
              Target {filterType} Value
            </label>
            <select
              value={targetValue}
              onChange={(e) => {
                setTargetValue(parseFloat(e.target.value) || 0);
                setVisualizationApplied(false); // Reset when value changes
              }}
              className="w-full px-3 py-2 bg-neutral-700 border border-neutral-600 rounded text-neutral-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value={0}>Select {filterType} value...</option>
              {uniqueValues.map((value, index) => (
                <option key={index} value={value}>
                  {filterType === 'TER' ? formatTER(value) : formatTEC(value)}
                </option>
              ))}
            </select>
            <div className="text-xs text-neutral-500 mt-1">
              {uniqueValues.length} unique values available
            </div>
          </div>

          {/* Tolerance Slider */}
          <div>
            <label className="block text-xs font-medium text-neutral-400 mb-2">
              Tolerance: ±{tolerance}%
            </label>
            <input
              type="range"
              min="1"
              max="20"
              step="1"
              value={tolerance}
              onChange={(e) => setTolerance(parseInt(e.target.value))}
              className="w-full h-2 bg-neutral-700 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-neutral-500 mt-1">
              <span>1%</span>
              <span>20%</span>
            </div>
          </div>
        </div>

        {/* Active Filter Summary with Apply Button */}
        {selectedGroup && (
          <div className="mt-4 pt-4 border-t border-neutral-700">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-4">
                <div>
                  <div className="text-xs text-neutral-500">Filter Range</div>
                  <div className="text-sm font-mono text-cyan-400">
                    {filterType === 'TER'
                      ? formatTER(getTERRange(targetValue, tolerance / 100).min)
                      : formatTEC(getTECRange(targetValue, tolerance / 100).min)}
                    {' → '}
                    {filterType === 'TER'
                      ? formatTER(getTERRange(targetValue, tolerance / 100).max)
                      : formatTEC(getTECRange(targetValue, tolerance / 100).max)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-neutral-500">Matching Models</div>
                  <div className="text-sm font-semibold text-green-400">
                    {selectedGroup.models.length.toLocaleString()} models
                  </div>
                </div>
              </div>
            </div>

            {/* Apply to Visualization Button */}
            <button
              onClick={() => {
                if (onTERTECFilterChange && selectedGroup) {
                  const filteredIds = selectedGroup.models.map(m => m.id);
                  onTERTECFilterChange(filteredIds);
                  setVisualizationApplied(true);
                }
              }}
              disabled={selectedGroup.models.length === 0}
              className={`w-full px-4 py-2.5 rounded-lg font-medium transition-all ${
                visualizationApplied
                  ? 'bg-green-600 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white disabled:bg-neutral-700 disabled:text-neutral-500'
              }`}
            >
              {visualizationApplied ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Applied to Visualizations
                </span>
              ) : (
                'Apply Filter to Visualizations'
              )}
            </button>
            {visualizationApplied && (
              <p className="text-xs text-neutral-400 mt-2 text-center">
                Filter active in Spider Plot, Pentagon, and all visualizations
              </p>
            )}
          </div>
        )}
      </div>

      {/* Parameter IQR Analysis */}
      {selectedGroup && (
        <div className="bg-neutral-800 rounded-lg border border-neutral-700 overflow-hidden">
          <div className="px-4 py-3 bg-neutral-900 border-b border-neutral-700">
            <h3 className="font-medium text-neutral-200">
              Parameter Distribution Analysis (IQR)
            </h3>
            <p className="text-xs text-neutral-400 mt-1">
              Statistical summary of circuit parameters for models with {filterType} ≈{' '}
              {filterType === 'TER' ? formatTER(targetValue) : formatTEC(targetValue)}
            </p>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-neutral-900 border-b border-neutral-700">
                  <th className="px-4 py-2 text-left text-neutral-400 font-medium">Parameter</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">Min</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">Q1</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">Median</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">Q3</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">Max</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">IQR</th>
                  <th className="px-4 py-2 text-right text-neutral-400 font-medium">CV</th>
                </tr>
              </thead>
              <tbody>
                {(['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'] as const).map((param) => {
                  const summary = selectedGroup.parameterSummaries[param];
                  const coefficientOfVariation = summary.median !== 0
                    ? ((summary.iqr / 2) / summary.median) * 100
                    : 0;

                  return (
                    <tr key={param} className="border-b border-neutral-700 hover:bg-neutral-750">
                      <td className="px-4 py-2 text-neutral-200 font-mono font-semibold">{param}</td>
                      <td className="px-4 py-2 text-right text-neutral-300 font-mono">
                        {formatParameterValue(param, summary.min)}
                      </td>
                      <td className="px-4 py-2 text-right text-neutral-300 font-mono">
                        {formatParameterValue(param, summary.q1)}
                      </td>
                      <td className="px-4 py-2 text-right text-cyan-400 font-mono font-medium">
                        {formatParameterValue(param, summary.median)}
                      </td>
                      <td className="px-4 py-2 text-right text-neutral-300 font-mono">
                        {formatParameterValue(param, summary.q3)}
                      </td>
                      <td className="px-4 py-2 text-right text-neutral-300 font-mono">
                        {formatParameterValue(param, summary.max)}
                      </td>
                      <td className="px-4 py-2 text-right text-purple-400 font-mono font-semibold">
                        {formatParameterValue(param, summary.iqr)}
                      </td>
                      <td className="px-4 py-2 text-right text-orange-400 font-mono">
                        {coefficientOfVariation.toFixed(1)}%
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TER/TEC vs Resnorm Distribution Plot */}
      {modelsWithMetrics.length > 0 && (
        <div className="bg-neutral-800 rounded-lg border border-neutral-700 overflow-hidden">
          <div className="px-4 py-3 bg-neutral-900 border-b border-neutral-700">
            <h3 className="font-medium text-neutral-200">
              {filterType} vs Resnorm Distribution
            </h3>
            <p className="text-xs text-neutral-400 mt-1">
              Scatter plot showing relationship between {filterType} values and model fit quality
            </p>
          </div>

          <div className="p-4">
            <TERTECDistributionPlot
              models={modelsWithMetrics}
              filterType={filterType}
              selectedValue={targetValue}
              tolerance={tolerance}
            />
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4">
        <h3 className="font-medium text-neutral-200 mb-2">Analysis Guide</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-neutral-400">
          <div className="space-y-1">
            <p><strong className="text-neutral-300">TER:</strong> Transepithelial Resistance = Rsh ‖ (Ra + Rb)</p>
            <p><strong className="text-neutral-300">TEC:</strong> Transepithelial Capacitance from impedance integral</p>
            <p><strong className="text-cyan-400">Median:</strong> Middle value (50th percentile)</p>
            <p><strong className="text-purple-400">IQR:</strong> Interquartile range (Q3 - Q1), indicates spread</p>
          </div>
          <div className="space-y-1">
            <p><strong className="text-orange-400">CV:</strong> Coefficient of Variation = (IQR/2) / Median × 100%</p>
            <p className="text-neutral-500">Lower CV indicates tighter parameter clustering for constant TER/TEC</p>
            <p className="text-neutral-500">Use this to identify which parameters vary most at equilibrium</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TERTECAnalysisTab;

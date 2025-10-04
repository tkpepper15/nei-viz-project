/**
 * Pentagon Box-Whisker Visualization Component
 * ==========================================
 *
 * Displays five-number summary (min, Q1, median, Q3, max) for each parameter
 * as connected banded box-and-whisker plots on a pentagon framework.
 * Paginated by quartile groups.
 */

'use client';

import React, { useCallback, useRef, useEffect, useState } from 'react';
import { CircuitParameters } from '../types/parameters';
import { ModelSnapshot } from '../types';
import { ChevronLeftIcon, ChevronRightIcon } from '@heroicons/react/24/outline';

interface FiveNumberSummary {
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  count: number;
}

interface QuartileGroup {
  quartile: 'Q1' | 'Q2' | 'Q3' | 'Q4';
  label: string;
  resnormRange: [number, number];
  models: ModelSnapshot[];
  parameterSummaries: {
    Rsh: FiveNumberSummary;
    Ra: FiveNumberSummary;
    Rb: FiveNumberSummary;
    Ca: FiveNumberSummary;
    Cb: FiveNumberSummary;
  };
}

interface PentagonBoxWhiskerProps {
  models: ModelSnapshot[];
  currentParameters?: CircuitParameters;
  gridSize?: number; // Number of grid points per parameter dimension
}

// Calculate five-number summary for an array of values
const calculateFiveNumberSummary = (values: number[]): FiveNumberSummary => {
  if (values.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0, count: 0 };
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

  return { min, q1, median, q3, max, count };
};

// Group models by resnorm quartiles
const groupByQuartiles = (models: ModelSnapshot[]): QuartileGroup[] => {
  const validModels = models.filter(m => m.resnorm !== undefined && m.resnorm !== null);
  if (validModels.length === 0) return [];

  const sortedModels = [...validModels].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));
  const count = sortedModels.length;
  const q1Cutoff = Math.floor(count * 0.25);
  const q2Cutoff = Math.floor(count * 0.50);
  const q3Cutoff = Math.floor(count * 0.75);

  const q1Models = sortedModels.slice(0, q1Cutoff);
  const q2Models = sortedModels.slice(q1Cutoff, q2Cutoff);
  const q3Models = sortedModels.slice(q2Cutoff, q3Cutoff);
  const q4Models = sortedModels.slice(q3Cutoff);

  const calculateGroupSummaries = (groupModels: ModelSnapshot[]) => ({
    Rsh: calculateFiveNumberSummary(groupModels.map(m => m.parameters.Rsh)),
    Ra: calculateFiveNumberSummary(groupModels.map(m => m.parameters.Ra)),
    Rb: calculateFiveNumberSummary(groupModels.map(m => m.parameters.Rb)),
    Ca: calculateFiveNumberSummary(groupModels.map(m => m.parameters.Ca)),
    Cb: calculateFiveNumberSummary(groupModels.map(m => m.parameters.Cb))
  });

  const groups: QuartileGroup[] = [];

  if (q1Models.length > 0) {
    groups.push({
      quartile: 'Q1',
      label: 'Best Fit (Q1)',
      resnormRange: [q1Models[0].resnorm || 0, q1Models[q1Models.length - 1].resnorm || 0],
      models: q1Models,
      parameterSummaries: calculateGroupSummaries(q1Models)
    });
  }

  if (q2Models.length > 0) {
    groups.push({
      quartile: 'Q2',
      label: 'Good Fit (Q2)',
      resnormRange: [q2Models[0].resnorm || 0, q2Models[q2Models.length - 1].resnorm || 0],
      models: q2Models,
      parameterSummaries: calculateGroupSummaries(q2Models)
    });
  }

  if (q3Models.length > 0) {
    groups.push({
      quartile: 'Q3',
      label: 'Moderate Fit (Q3)',
      resnormRange: [q3Models[0].resnorm || 0, q3Models[q3Models.length - 1].resnorm || 0],
      models: q3Models,
      parameterSummaries: calculateGroupSummaries(q3Models)
    });
  }

  if (q4Models.length > 0) {
    groups.push({
      quartile: 'Q4',
      label: 'Poor Fit (Q4)',
      resnormRange: [q4Models[0].resnorm || 0, q4Models[q4Models.length - 1].resnorm || 0],
      models: q4Models,
      parameterSummaries: calculateGroupSummaries(q4Models)
    });
  }

  return groups;
};

// Type for circuit parameters (excluding frequency_range)
type CircuitParamKey = 'Rsh' | 'Ra' | 'Rb' | 'Ca' | 'Cb';

// Normalize parameter value to [0, 1] range using logarithmic scaling
const normalizeParameter = (param: CircuitParamKey, value: number): number => {
  switch (param) {
    case 'Rsh':
      return (Math.log10(Math.max(value, 100)) - Math.log10(100)) / (Math.log10(10000) - Math.log10(100));
    case 'Ra':
      return (Math.log10(Math.max(value, 500)) - Math.log10(500)) / (Math.log10(10000) - Math.log10(500));
    case 'Rb':
      return (Math.log10(Math.max(value, 1000)) - Math.log10(1000)) / (Math.log10(15000) - Math.log10(1000));
    case 'Ca':
      return (Math.log10(Math.max(value, 0.1e-6)) - Math.log10(0.1e-6)) / (Math.log10(10e-6) - Math.log10(0.1e-6));
    case 'Cb':
      return (Math.log10(Math.max(value, 0.1e-6)) - Math.log10(0.1e-6)) / (Math.log10(10e-6) - Math.log10(0.1e-6));
    default:
      return 0.5;
  }
};

// Get quartile color
const getQuartileColor = (quartile: string): string => {
  switch (quartile) {
    case 'Q1': return '#10B981'; // Green
    case 'Q2': return '#3B82F6'; // Blue
    case 'Q3': return '#F59E0B'; // Amber
    case 'Q4': return '#EF4444'; // Red
    default: return '#6B7280';
  }
};

export const PentagonBoxWhisker: React.FC<PentagonBoxWhiskerProps> = ({
  models,
  currentParameters,
  gridSize = 5
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [currentQuartileIndex, setCurrentQuartileIndex] = useState(0);
  const [quartileGroups, setQuartileGroups] = useState<QuartileGroup[]>([]);

  // Resnorm spectrum slider state (reserved for future functionality)
  const [useCustomRange] = useState(false);
  const [customResnormRange] = useState<[number, number]>([0, 100]);
  const [customRangeModels, setCustomRangeModels] = useState<ModelSnapshot[]>([]);

  // Calculate quartile groups
  useEffect(() => {
    const groups = groupByQuartiles(models);
    setQuartileGroups(groups);
    if (currentQuartileIndex >= groups.length) {
      setCurrentQuartileIndex(0);
    }
  }, [models, currentQuartileIndex]);

  // Calculate resnorm range for slider
  const resnormStats = React.useMemo(() => {
    const validModels = models.filter(m => m.resnorm !== undefined && m.resnorm !== null);
    if (validModels.length === 0) return { min: 0, max: 1 };

    const resnorms = validModels.map(m => m.resnorm!).sort((a, b) => a - b);
    return {
      min: resnorms[0],
      max: resnorms[resnorms.length - 1]
    };
  }, [models]);

  // Filter models by custom resnorm range
  useEffect(() => {
    if (!useCustomRange) {
      setCustomRangeModels([]);
      return;
    }

    const { min, max } = resnormStats;
    const rangeMin = min + (customResnormRange[0] / 100) * (max - min);
    const rangeMax = min + (customResnormRange[1] / 100) * (max - min);

    const filtered = models.filter(m => {
      const resnorm = m.resnorm;
      return resnorm !== undefined && resnorm !== null && resnorm >= rangeMin && resnorm <= rangeMax;
    });

    setCustomRangeModels(filtered);
  }, [useCustomRange, customResnormRange, models, resnormStats]);

  // Calculate custom range summaries
  const customRangeSummaries = React.useMemo(() => {
    if (!useCustomRange || customRangeModels.length === 0) return null;

    return {
      Rsh: calculateFiveNumberSummary(customRangeModels.map(m => m.parameters.Rsh)),
      Ra: calculateFiveNumberSummary(customRangeModels.map(m => m.parameters.Ra)),
      Rb: calculateFiveNumberSummary(customRangeModels.map(m => m.parameters.Rb)),
      Ca: calculateFiveNumberSummary(customRangeModels.map(m => m.parameters.Ca)),
      Cb: calculateFiveNumberSummary(customRangeModels.map(m => m.parameters.Cb))
    };
  }, [useCustomRange, customRangeModels]);

  // Render pentagon box-whisker plot
  const renderPentagonBoxWhisker = useCallback(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Use custom range data if enabled, otherwise use quartile data
    let parameterSummaries: {
      Rsh: FiveNumberSummary;
      Ra: FiveNumberSummary;
      Rb: FiveNumberSummary;
      Ca: FiveNumberSummary;
      Cb: FiveNumberSummary;
    } | undefined;
    let displayColor: string | undefined;
    let hasData = false;

    if (useCustomRange && customRangeSummaries) {
      parameterSummaries = customRangeSummaries;
      displayColor = '#8B5CF6'; // Purple for custom range
      hasData = true;
    } else if (quartileGroups.length > 0) {
      const currentGroup = quartileGroups[currentQuartileIndex];
      if (!currentGroup) return;
      parameterSummaries = currentGroup.parameterSummaries;
      displayColor = getQuartileColor(currentGroup.quartile);
      hasData = true;
    }

    if (!hasData || !parameterSummaries || !displayColor) return;

    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const maxRadius = Math.min(width, height) * 0.35;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw parameter labels and axes
    const params: CircuitParamKey[] = ['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'];
    const angles = params.map((_, i) => (i * 2 * Math.PI / 5) - Math.PI / 2);

    // Draw radial grid lines and labels
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '12px monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    params.forEach((param, i) => {
      const angle = angles[i];
      const labelRadius = maxRadius + 30;
      const labelX = centerX + Math.cos(angle) * labelRadius;
      const labelY = centerY + Math.sin(angle) * labelRadius;

      // Draw axis line
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(centerX + Math.cos(angle) * maxRadius, centerY + Math.sin(angle) * maxRadius);
      ctx.stroke();

      // Draw label
      ctx.fillText(param, labelX, labelY);
    });

    // Draw concentric reference circles based on gridSize
    // Generate grid lines matching the parameter space divisions
    const numGridLines = Math.max(2, gridSize); // At least 2 grid lines
    const gridScales = Array.from({ length: numGridLines }, (_, i) => (i + 1) / numGridLines);

    gridScales.forEach((scale) => {
      ctx.strokeStyle = scale === 1.0 ? '#374151' : '#1F2937'; // Outer circle slightly more visible
      ctx.lineWidth = scale === 1.0 ? 1.5 : 1;
      ctx.beginPath();
      const scaledRadius = maxRadius * scale;
      params.forEach((_, i) => {
        const angle = angles[i];
        const x = centerX + Math.cos(angle) * scaledRadius;
        const y = centerY + Math.sin(angle) * scaledRadius;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.closePath();
      ctx.stroke();
    });

    // Draw box-whisker plots for each parameter
    params.forEach((param, i) => {
      const summary = parameterSummaries[param];
      const angle = angles[i];

      // Normalize values to [0, 1]
      const normalizedMin = normalizeParameter(param, summary.min);
      const normalizedQ1 = normalizeParameter(param, summary.q1);
      const normalizedMedian = normalizeParameter(param, summary.median);
      const normalizedQ3 = normalizeParameter(param, summary.q3);
      const normalizedMax = normalizeParameter(param, summary.max);

      // Calculate positions
      const minRadius = normalizedMin * maxRadius;
      const q1Radius = normalizedQ1 * maxRadius;
      const medianRadius = normalizedMedian * maxRadius;
      const q3Radius = normalizedQ3 * maxRadius;
      const maxRadius_val = normalizedMax * maxRadius;

      const x = (r: number) => centerX + Math.cos(angle) * r;
      const y = (r: number) => centerY + Math.sin(angle) * r;

      // Draw whisker (min to max)
      ctx.strokeStyle = displayColor;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x(minRadius), y(minRadius));
      ctx.lineTo(x(maxRadius_val), y(maxRadius_val));
      ctx.stroke();

      // Draw box (Q1 to Q3) - create a band
      const perpAngle = angle + Math.PI / 2;
      const bandWidth = 8;

      ctx.fillStyle = displayColor + '40'; // 25% opacity
      ctx.strokeStyle = displayColor;
      ctx.lineWidth = 2;

      // Draw filled box
      ctx.beginPath();
      ctx.moveTo(x(q1Radius) + Math.cos(perpAngle) * bandWidth, y(q1Radius) + Math.sin(perpAngle) * bandWidth);
      ctx.lineTo(x(q3Radius) + Math.cos(perpAngle) * bandWidth, y(q3Radius) + Math.sin(perpAngle) * bandWidth);
      ctx.lineTo(x(q3Radius) - Math.cos(perpAngle) * bandWidth, y(q3Radius) - Math.sin(perpAngle) * bandWidth);
      ctx.lineTo(x(q1Radius) - Math.cos(perpAngle) * bandWidth, y(q1Radius) - Math.sin(perpAngle) * bandWidth);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();

      // Draw median line
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x(medianRadius) + Math.cos(perpAngle) * bandWidth, y(medianRadius) + Math.sin(perpAngle) * bandWidth);
      ctx.lineTo(x(medianRadius) - Math.cos(perpAngle) * bandWidth, y(medianRadius) - Math.sin(perpAngle) * bandWidth);
      ctx.stroke();

      // Draw min/max caps
      ctx.strokeStyle = displayColor;
      ctx.lineWidth = 2;
      const capWidth = 6;

      // Min cap
      ctx.beginPath();
      ctx.moveTo(x(minRadius) + Math.cos(perpAngle) * capWidth, y(minRadius) + Math.sin(perpAngle) * capWidth);
      ctx.lineTo(x(minRadius) - Math.cos(perpAngle) * capWidth, y(minRadius) - Math.sin(perpAngle) * capWidth);
      ctx.stroke();

      // Max cap
      ctx.beginPath();
      ctx.moveTo(x(maxRadius_val) + Math.cos(perpAngle) * capWidth, y(maxRadius_val) + Math.sin(perpAngle) * capWidth);
      ctx.lineTo(x(maxRadius_val) - Math.cos(perpAngle) * capWidth, y(maxRadius_val) - Math.sin(perpAngle) * capWidth);
      ctx.stroke();
    });

    // Draw current parameters as reference (if provided)
    if (currentParameters) {
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();

      params.forEach((param, i) => {
        const angle = angles[i];
        const value = currentParameters[param];
        const normalizedValue = normalizeParameter(param, value as number);
        const radius = normalizedValue * maxRadius;
        const x = centerX + Math.cos(angle) * radius;
        const y = centerY + Math.sin(angle) * radius;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });

      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }

  }, [quartileGroups, currentQuartileIndex, currentParameters, useCustomRange, customRangeSummaries, gridSize]);

  // Update visualization when dependencies change
  useEffect(() => {
    renderPentagonBoxWhisker();
  }, [renderPentagonBoxWhisker]);

  if (quartileGroups.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-neutral-400">
        <div className="text-center">
          <p className="text-lg mb-2">No data available</p>
          <p className="text-sm">Run a parameter grid computation to generate visualization data</p>
        </div>
      </div>
    );
  }

  const currentGroup = quartileGroups[currentQuartileIndex];

  return (
    <div className="w-full h-full flex flex-col bg-neutral-950">
      {/* Header with quartile info and pagination */}
      <div className="flex items-center justify-between px-6 py-4 bg-neutral-900 border-b border-neutral-800">
        <div className="flex items-center gap-4">
          <div
            className="w-4 h-4 rounded-full"
            style={{ backgroundColor: getQuartileColor(currentGroup.quartile) }}
          />
          <div>
            <h3 className="text-lg font-semibold text-neutral-200">{currentGroup.label}</h3>
            <p className="text-xs text-neutral-400">
              Resnorm: {currentGroup.resnormRange[0].toFixed(4)} - {currentGroup.resnormRange[1].toFixed(4)}
              {' • '}
              {currentGroup.models.length.toLocaleString()} models
            </p>
          </div>
        </div>

        {/* Pagination controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentQuartileIndex(Math.max(0, currentQuartileIndex - 1))}
            disabled={currentQuartileIndex === 0}
            className="p-2 bg-neutral-800 hover:bg-neutral-700 disabled:bg-neutral-800 disabled:opacity-30 rounded transition-colors"
            title="Previous quartile"
          >
            <ChevronLeftIcon className="w-4 h-4 text-neutral-400" />
          </button>
          <span className="text-sm text-neutral-400 font-mono">
            {currentQuartileIndex + 1} / {quartileGroups.length}
          </span>
          <button
            onClick={() => setCurrentQuartileIndex(Math.min(quartileGroups.length - 1, currentQuartileIndex + 1))}
            disabled={currentQuartileIndex === quartileGroups.length - 1}
            className="p-2 bg-neutral-800 hover:bg-neutral-700 disabled:bg-neutral-800 disabled:opacity-30 rounded transition-colors"
            title="Next quartile"
          >
            <ChevronRightIcon className="w-4 h-4 text-neutral-400" />
          </button>
        </div>
      </div>

      {/* Canvas visualization */}
      <div className="flex-1 flex items-center justify-center p-4">
        <canvas
          ref={canvasRef}
          width={600}
          height={600}
          className="max-w-full max-h-full"
        />
      </div>

      {/* Legend */}
      <div className="px-6 py-4 bg-neutral-900 border-t border-neutral-800">
        <div className="grid grid-cols-2 gap-4 text-xs text-neutral-400">
          <div className="space-y-1">
            <p><span className="text-neutral-300">◼</span> Box: Q1 to Q3 (IQR)</p>
            <p><span className="text-white">━</span> White line: Median</p>
            <p><span className="text-neutral-300">│</span> Whiskers: Min to Max</p>
          </div>
          <div className="space-y-1">
            <p><span className="text-white">- - -</span> Current parameters (reference)</p>
            <p className="text-neutral-500">Values normalized logarithmically</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PentagonBoxWhisker;

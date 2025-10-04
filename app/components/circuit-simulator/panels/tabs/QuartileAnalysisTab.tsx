/**
 * Quartile Analysis Tab Component
 * ================================
 *
 * Groups resnorm values across the entire spectrum into quartiles and analyzes
 * the five-number summary (min, Q1, median, Q3, max) for each of the 5 circuit
 * parameters within each quartile group.
 */

'use client';

import React, { useMemo } from 'react';
import { ModelSnapshot } from '../../types';
import { CircuitParameters } from '../../types/parameters';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';

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

// Calculate five-number summary for an array of values
const calculateFiveNumberSummary = (values: number[]): FiveNumberSummary => {
  if (values.length === 0) {
    return { min: 0, q1: 0, median: 0, q3: 0, max: 0, count: 0 };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const count = sorted.length;

  const min = sorted[0];
  const max = sorted[count - 1];

  // Calculate median
  const medianIndex = Math.floor(count / 2);
  const median = count % 2 === 0
    ? (sorted[medianIndex - 1] + sorted[medianIndex]) / 2
    : sorted[medianIndex];

  // Calculate Q1 (median of lower half)
  const lowerHalf = sorted.slice(0, medianIndex);
  const q1Index = Math.floor(lowerHalf.length / 2);
  const q1 = lowerHalf.length % 2 === 0 && lowerHalf.length > 1
    ? (lowerHalf[q1Index - 1] + lowerHalf[q1Index]) / 2
    : lowerHalf[q1Index] || min;

  // Calculate Q3 (median of upper half)
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
  // Filter models with valid resnorm values
  const validModels = models.filter(m => m.resnorm !== undefined && m.resnorm !== null);

  if (validModels.length === 0) {
    return [];
  }

  // Sort by resnorm
  const sortedModels = [...validModels].sort((a, b) => (a.resnorm || 0) - (b.resnorm || 0));

  // Calculate quartile boundaries
  const count = sortedModels.length;
  const q1Cutoff = Math.floor(count * 0.25);
  const q2Cutoff = Math.floor(count * 0.50);
  const q3Cutoff = Math.floor(count * 0.75);

  // Split into quartiles
  const q1Models = sortedModels.slice(0, q1Cutoff);
  const q2Models = sortedModels.slice(q1Cutoff, q2Cutoff);
  const q3Models = sortedModels.slice(q2Cutoff, q3Cutoff);
  const q4Models = sortedModels.slice(q3Cutoff);

  // Helper to calculate parameter summaries for a group
  const calculateGroupSummaries = (groupModels: ModelSnapshot[]) => {
    const rshValues = groupModels.map(m => m.parameters.Rsh);
    const raValues = groupModels.map(m => m.parameters.Ra);
    const rbValues = groupModels.map(m => m.parameters.Rb);
    const caValues = groupModels.map(m => m.parameters.Ca);
    const cbValues = groupModels.map(m => m.parameters.Cb);

    return {
      Rsh: calculateFiveNumberSummary(rshValues),
      Ra: calculateFiveNumberSummary(raValues),
      Rb: calculateFiveNumberSummary(rbValues),
      Ca: calculateFiveNumberSummary(caValues),
      Cb: calculateFiveNumberSummary(cbValues)
    };
  };

  // Create quartile groups
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

// Format parameter value for display
const formatParameterValue = (param: keyof CircuitParameters, value: number): string => {
  if (param === 'Ca' || param === 'Cb') {
    return `${(value * 1e6).toFixed(2)} µF`;
  }
  return `${value.toFixed(1)} Ω`;
};

// Get quartile color
const getQuartileColor = (quartile: string): string => {
  switch (quartile) {
    case 'Q1': return '#10B981'; // Green - best fit
    case 'Q2': return '#3B82F6'; // Blue - good fit
    case 'Q3': return '#F59E0B'; // Amber - moderate fit
    case 'Q4': return '#EF4444'; // Red - poor fit
    default: return '#6B7280'; // Gray
  }
};

export const QuartileAnalysisTab: React.FC<BottomPanelTabProps> = ({
  gridResults
}) => {
  // Calculate quartile groups
  const quartileGroups = useMemo(() => {
    return groupByQuartiles(gridResults);
  }, [gridResults]);

  if (quartileGroups.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-neutral-400">
        <div className="text-center">
          <p className="text-lg mb-2">No data available for quartile analysis</p>
          <p className="text-sm">Run a parameter grid computation to generate analysis data</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-neutral-200">Quartile Analysis</h2>
          <p className="text-sm text-neutral-400">
            Parameter distribution analysis across resnorm quartiles ({gridResults.length.toLocaleString()} total models)
          </p>
        </div>
      </div>

      {/* Quartile Groups */}
      <div className="space-y-4">
        {quartileGroups.map((group) => (
          <div
            key={group.quartile}
            className="bg-neutral-800 rounded-lg border border-neutral-700 overflow-hidden"
          >
            {/* Group Header */}
            <div
              className="px-4 py-3 border-b border-neutral-700 flex items-center justify-between"
              style={{ backgroundColor: `${getQuartileColor(group.quartile)}15` }}
            >
              <div className="flex items-center gap-3">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: getQuartileColor(group.quartile) }}
                />
                <div>
                  <h3 className="font-medium text-neutral-200">{group.label}</h3>
                  <p className="text-xs text-neutral-400">
                    Resnorm: {group.resnormRange[0].toFixed(4)} - {group.resnormRange[1].toFixed(4)}
                    {' • '}
                    {group.models.length.toLocaleString()} models
                  </p>
                </div>
              </div>
            </div>

            {/* Parameter Summaries Table */}
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
                  </tr>
                </thead>
                <tbody>
                  {(['Rsh', 'Ra', 'Rb', 'Ca', 'Cb'] as const).map((param) => {
                    const summary = group.parameterSummaries[param];
                    const iqr = summary.q3 - summary.q1;
                    const isCapacitance = param === 'Ca' || param === 'Cb';

                    return (
                      <tr key={param} className="border-b border-neutral-700 hover:bg-neutral-750">
                        <td className="px-4 py-2 text-neutral-200 font-mono">{param}</td>
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
                        <td className="px-4 py-2 text-right text-purple-400 font-mono">
                          {isCapacitance
                            ? `${(iqr * 1e6).toFixed(2)} µF`
                            : `${iqr.toFixed(1)} Ω`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ))}
      </div>

      {/* Summary Statistics */}
      <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4">
        <h3 className="font-medium text-neutral-200 mb-3">Summary Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {quartileGroups.map((group) => (
            <div key={group.quartile} className="text-center">
              <div
                className="w-2 h-2 rounded-full mx-auto mb-1"
                style={{ backgroundColor: getQuartileColor(group.quartile) }}
              />
              <div className="text-xs text-neutral-400">{group.quartile}</div>
              <div className="text-lg font-semibold text-neutral-200">
                {group.models.length.toLocaleString()}
              </div>
              <div className="text-xs text-neutral-500">models</div>
            </div>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="bg-neutral-800 rounded-lg border border-neutral-700 p-4">
        <h3 className="font-medium text-neutral-200 mb-2">Legend</h3>
        <div className="text-xs text-neutral-400 space-y-1">
          <p><strong className="text-neutral-300">Min/Max:</strong> Minimum and maximum values in the quartile</p>
          <p><strong className="text-neutral-300">Q1/Q3:</strong> First and third quartiles (25th and 75th percentiles)</p>
          <p><strong className="text-cyan-400">Median:</strong> Middle value (50th percentile)</p>
          <p><strong className="text-purple-400">IQR:</strong> Interquartile range (Q3 - Q1), indicates spread</p>
        </div>
      </div>
    </div>
  );
};

export default QuartileAnalysisTab;

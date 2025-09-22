/**
 * Enhanced Impedance Comparison Tab - Demo Implementation
 * ======================================================
 *
 * This is a comprehensive enhancement of the original impedance comparison tab
 * with advanced features for circuit impedance analysis and data export.
 *
 * Key Features Added:
 * - Advanced frequency configuration (linear/logarithmic scales)
 * - CSV and JSON export functionality
 * - Enhanced data formatting with phase calculations
 * - Interactive detailed view toggle
 * - Custom frequency range controls
 * - Professional UI with PyCharm-style design
 *
 * Usage Example:
 * ```tsx
 * <ImpedanceComparisonTab
 *   topConfigurations={resnormGroups}
 *   currentParameters={groundTruthParams}
 *   selectedConfigIndex={0}
 *   onConfigurationSelect={() => {}}
 *   isVisible={true}
 *   enableExport={true}
 *   frequencyPoints={[1, 10, 100, 1000]}
 * />
 * ```
 */

import React, { useMemo, useState, useCallback } from 'react';
import { ArrowDownTrayIcon, AdjustmentsHorizontalIcon, EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';

// Type definitions for enhanced functionality
interface ImpedanceDataPoint {
  frequency: number;
  referenceImpedance: { real: number; imag: number; magnitude: number };
  selectedImpedance: { real: number; imag: number; magnitude: number };
  resnorm: number;
  percentError: number;
}

interface FrequencyRange {
  min: number;
  max: number;
  points: number;
  scale: 'linear' | 'logarithmic';
}

// Enhanced props interface
interface EnhancedImpedanceTableProps {
  topConfigurations: any[]; // ResnormGroup[]
  currentParameters: any;   // CircuitParameters
  selectedConfigIndex: number;
  onConfigurationSelect: (index: number) => void;
  isVisible: boolean;
  enableExport?: boolean;
  frequencyPoints?: number[];
}

export const EnhancedImpedanceComparisonTab: React.FC<EnhancedImpedanceTableProps> = ({
  topConfigurations,
  currentParameters,
  selectedConfigIndex,
  onConfigurationSelect,
  isVisible,
  enableExport = true,
  frequencyPoints
}) => {
  // Enhanced state management
  const [sortColumn, setSortColumn] = useState<'frequency' | 'resnorm' | 'error'>('frequency');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [showDetailsFor, setShowDetailsFor] = useState<number | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [customFrequencyRange, setCustomFrequencyRange] = useState<FrequencyRange>({
    min: 0.1,
    max: 100000,
    points: 50,
    scale: 'logarithmic'
  });
  const [useCustomFrequencies, setUseCustomFrequencies] = useState(false);

  // Generate frequency points based on range and scale
  const generateFrequencyPoints = useCallback((range: FrequencyRange): number[] => {
    const { min, max, points, scale } = range;
    const frequencies: number[] = [];

    if (scale === 'logarithmic') {
      const logMin = Math.log10(min);
      const logMax = Math.log10(max);
      const step = (logMax - logMin) / (points - 1);

      for (let i = 0; i < points; i++) {
        frequencies.push(Math.pow(10, logMin + i * step));
      }
    } else {
      const step = (max - min) / (points - 1);
      for (let i = 0; i < points; i++) {
        frequencies.push(min + i * step);
      }
    }

    return frequencies;
  }, []);

  // Helper function for parallel impedance calculation
  const parallelImpedance = (z1: {real: number, imag: number}, z2: {real: number, imag: number}) => {
    const numeratorReal = z1.real * z2.real - z1.imag * z2.imag;
    const numeratorImag = z1.real * z2.imag + z1.imag * z2.real;
    const denominatorReal = z1.real + z2.real;
    const denominatorImag = z1.imag + z2.imag;
    const denominatorMag = denominatorReal * denominatorReal + denominatorImag * denominatorImag;

    return {
      real: (numeratorReal * denominatorReal + numeratorImag * denominatorImag) / denominatorMag,
      imag: (numeratorImag * denominatorReal - numeratorReal * denominatorImag) / denominatorMag
    };
  };

  // Calculate impedance for given frequency and parameters using Randles circuit model
  const calculateImpedance = useCallback((frequency: number, params: { Ra: number; Rb: number; Ca: number; Cb: number; Rsh: number }) => {
    const omega = 2 * Math.PI * frequency;

    // Randles circuit model: Rs + (Ra || (1/jωCa)) + (Rb || (1/jωCb))
    const invJωCa = { real: 0, imag: -1 / (omega * params.Ca) };
    const raParallel = parallelImpedance({ real: params.Ra, imag: 0 }, invJωCa);

    const invJωCb = { real: 0, imag: -1 / (omega * params.Cb) };
    const rbParallel = parallelImpedance({ real: params.Rb, imag: 0 }, invJωCb);

    const totalReal = params.Rsh + raParallel.real + rbParallel.real;
    const totalImag = raParallel.imag + rbParallel.imag;
    const magnitude = Math.sqrt(totalReal * totalReal + totalImag * totalImag);

    return { real: totalReal, imag: totalImag, magnitude };
  }, []);

  // Main impedance data calculation with enhanced frequency support
  const impedanceData = useMemo(() => {
    if (!isVisible || topConfigurations.length === 0 || selectedConfigIndex >= topConfigurations.length) {
      return [];
    }

    const selectedConfig = topConfigurations[selectedConfigIndex];
    if (!selectedConfig?.items || selectedConfig.items.length === 0) {
      return [];
    }

    // Use custom frequencies, predefined points, or default values
    const frequencies = useCustomFrequencies
      ? generateFrequencyPoints(customFrequencyRange)
      : frequencyPoints || [1, 2, 5, 7, 10, 100, 1000];

    // Get reference parameters
    const referenceParams = {
      Rsh: currentParameters.Rsh,
      Ra: currentParameters.Ra,
      Ca: currentParameters.Ca,
      Rb: currentParameters.Rb,
      Cb: currentParameters.Cb
    };

    const selectedModel = selectedConfig.items[0];
    const selectedParams = selectedModel?.parameters || referenceParams;

    const data: ImpedanceDataPoint[] = frequencies.map((freq, index) => {
      const refImpedance = calculateImpedance(freq, referenceParams);

      let selectedImpedance;
      if (selectedModel?.data && selectedModel.data.length > index) {
        const impedancePoint = selectedModel.data[index];
        selectedImpedance = {
          real: impedancePoint.real,
          imag: impedancePoint.imaginary,
          magnitude: impedancePoint.magnitude
        };
      } else {
        selectedImpedance = calculateImpedance(freq, selectedParams);
      }

      const resnorm = Math.abs(selectedImpedance.real - refImpedance.real) +
                      Math.abs(selectedImpedance.imag - refImpedance.imag);
      const percentError = (resnorm / refImpedance.magnitude) * 100;

      return {
        frequency: freq,
        referenceImpedance: refImpedance,
        selectedImpedance: selectedImpedance,
        resnorm,
        percentError
      };
    });

    return data;
  }, [isVisible, topConfigurations, selectedConfigIndex, currentParameters, frequencyPoints, calculateImpedance, useCustomFrequencies, customFrequencyRange, generateFrequencyPoints]);

  // Enhanced export functionality
  const exportToCSV = useCallback(() => {
    if (impedanceData.length === 0) return;

    const headers = [
      'Frequency (Hz)',
      'Reference Real (Ω)',
      'Reference Imag (Ω)',
      'Reference Magnitude (Ω)',
      'Reference Phase (°)',
      'Selected Real (Ω)',
      'Selected Imag (Ω)',
      'Selected Magnitude (Ω)',
      'Selected Phase (°)',
      'Resnorm',
      'Error (%)'
    ];

    const csvContent = [
      headers.join(','),
      ...impedanceData.map(point => [
        point.frequency.toFixed(3),
        point.referenceImpedance.real.toFixed(6),
        point.referenceImpedance.imag.toFixed(6),
        point.referenceImpedance.magnitude.toFixed(6),
        (Math.atan2(point.referenceImpedance.imag, point.referenceImpedance.real) * 180 / Math.PI).toFixed(3),
        point.selectedImpedance.real.toFixed(6),
        point.selectedImpedance.imag.toFixed(6),
        point.selectedImpedance.magnitude.toFixed(6),
        (Math.atan2(point.selectedImpedance.imag, point.selectedImpedance.real) * 180 / Math.PI).toFixed(3),
        point.resnorm.toFixed(6),
        point.percentError.toFixed(3)
      ].join(','))
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `enhanced_impedance_comparison_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [impedanceData]);

  const exportToJSON = useCallback(() => {
    if (impedanceData.length === 0) return;

    const exportData = {
      metadata: {
        exportDate: new Date().toISOString(),
        version: "Enhanced Impedance Comparison v2.0",
        referenceParameters: currentParameters,
        selectedConfiguration: topConfigurations[selectedConfigIndex]?.label || 'Unknown',
        frequencyConfiguration: useCustomFrequencies ? customFrequencyRange : 'Default',
        totalPoints: impedanceData.length,
        sortConfiguration: { column: sortColumn, direction: sortDirection }
      },
      statistics: {
        averageResnorm: impedanceData.reduce((sum, p) => sum + p.resnorm, 0) / impedanceData.length,
        maxError: Math.max(...impedanceData.map(p => p.percentError)),
        minError: Math.min(...impedanceData.map(p => p.percentError)),
        frequencyRange: {
          min: Math.min(...impedanceData.map(p => p.frequency)),
          max: Math.max(...impedanceData.map(p => p.frequency))
        }
      },
      data: impedanceData
    };

    const jsonContent = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `enhanced_impedance_comparison_${new Date().toISOString().split('T')[0]}.json`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [impedanceData, currentParameters, topConfigurations, selectedConfigIndex, useCustomFrequencies, customFrequencyRange, sortColumn, sortDirection]);

  // Formatting function with enhanced precision
  const formatNumber = (value: number, decimals: number = 2) => {
    if (Math.abs(value) < 0.001) return '0';
    if (Math.abs(value) >= 1000000) return value.toExponential(2);
    if (Math.abs(value) >= 1000) return (value / 1000).toFixed(1) + 'k';
    return value.toFixed(decimals);
  };

  // Return JSX with enhanced features
  return (
    <div className="h-full flex flex-col bg-neutral-900 text-white">
      {/* Enhanced Header */}
      <div className="p-4 border-b border-neutral-700 bg-neutral-800/50">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-neutral-200">Enhanced Impedance Comparison</h3>
          <div className="flex items-center space-x-2">
            {/* Advanced controls */}
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-1 text-xs bg-neutral-700 hover:bg-neutral-600 px-2 py-1 rounded transition-colors"
            >
              <AdjustmentsHorizontalIcon className="w-3 h-3" />
              Advanced
            </button>

            <button
              onClick={() => setShowDetailsFor(showDetailsFor ? null : 0)}
              className="flex items-center gap-1 text-xs bg-neutral-700 hover:bg-neutral-600 px-2 py-1 rounded transition-colors"
            >
              {showDetailsFor !== null ? <EyeSlashIcon className="w-3 h-3" /> : <EyeIcon className="w-3 h-3" />}
              Details
            </button>

            {/* Export functionality */}
            {enableExport && impedanceData.length > 0 && (
              <div className="relative group">
                <button className="flex items-center gap-1 text-xs bg-blue-600 hover:bg-blue-500 px-2 py-1 rounded transition-colors">
                  <ArrowDownTrayIcon className="w-3 h-3" />
                  Export
                </button>
                <div className="absolute right-0 top-full mt-1 bg-neutral-800 border border-neutral-600 rounded shadow-lg z-10 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={exportToCSV}
                    className="block w-full text-left px-3 py-2 text-xs hover:bg-neutral-700 transition-colors"
                  >
                    Export as CSV
                  </button>
                  <button
                    onClick={exportToJSON}
                    className="block w-full text-left px-3 py-2 text-xs hover:bg-neutral-700 transition-colors"
                  >
                    Export as JSON
                  </button>
                </div>
              </div>
            )}

            <span className="text-xs text-neutral-500">
              {impedanceData.length} frequency points
            </span>
          </div>
        </div>

        {/* Configuration selector */}
        <div className="bg-neutral-800 rounded-lg p-3 mb-4">
          <h4 className="text-sm font-medium text-neutral-200 mb-2">Circuit Selection</h4>
          <select
            value={selectedConfigIndex}
            onChange={(e) => onConfigurationSelect(Number(e.target.value))}
            className="w-full bg-neutral-700 border border-neutral-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            {topConfigurations.map((config, index) => (
              <option key={index} value={index}>
                #{index + 1} - {config.label} - Range: {config.range?.[0]?.toFixed(2)}-{config.range?.[1]?.toFixed(2)}
                {config.items && ` (${config.items.length} models)`}
              </option>
            ))}
          </select>
        </div>

        {/* Advanced frequency configuration */}
        {showAdvanced && (
          <div className="bg-neutral-800 rounded-lg p-3 mt-4">
            <h4 className="text-sm font-medium text-neutral-200 mb-3 flex items-center gap-2">
              <AdjustmentsHorizontalIcon className="w-4 h-4" />
              Frequency Configuration
            </h4>
            <div className="space-y-3">
              <label className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={useCustomFrequencies}
                  onChange={(e) => setUseCustomFrequencies(e.target.checked)}
                  className="rounded bg-neutral-700 border-neutral-600"
                />
                <span className="text-sm text-neutral-300">Use custom frequency range</span>
              </label>

              {useCustomFrequencies && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Min Frequency (Hz)</label>
                    <input
                      type="number"
                      value={customFrequencyRange.min}
                      onChange={(e) => setCustomFrequencyRange(prev => ({ ...prev, min: Number(e.target.value) }))}
                      className="w-full bg-neutral-700 border border-neutral-600 rounded px-2 py-1 text-xs text-white"
                      min="0.001"
                      step="0.001"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Max Frequency (Hz)</label>
                    <input
                      type="number"
                      value={customFrequencyRange.max}
                      onChange={(e) => setCustomFrequencyRange(prev => ({ ...prev, max: Number(e.target.value) }))}
                      className="w-full bg-neutral-700 border border-neutral-600 rounded px-2 py-1 text-xs text-white"
                      min="1"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Points</label>
                    <input
                      type="number"
                      value={customFrequencyRange.points}
                      onChange={(e) => setCustomFrequencyRange(prev => ({ ...prev, points: Number(e.target.value) }))}
                      className="w-full bg-neutral-700 border border-neutral-600 rounded px-2 py-1 text-xs text-white"
                      min="2"
                      max="1000"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-neutral-400 mb-1">Scale</label>
                    <select
                      value={customFrequencyRange.scale}
                      onChange={(e) => setCustomFrequencyRange(prev => ({ ...prev, scale: e.target.value as 'linear' | 'logarithmic' }))}
                      className="w-full bg-neutral-700 border border-neutral-600 rounded px-2 py-1 text-xs text-white"
                    >
                      <option value="logarithmic">Logarithmic</option>
                      <option value="linear">Linear</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Enhanced table with better formatting */}
      <div className="flex-1 overflow-auto">
        {impedanceData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-neutral-500">
            <div className="text-center">
              <div className="text-lg mb-2">No Impedance Data Available</div>
              <div className="text-sm">Configure parameters and run computation to generate data</div>
            </div>
          </div>
        ) : (
          <div className="min-w-full">
            <table className="w-full text-sm">
              <thead className="bg-neutral-800 sticky top-0">
                <tr>
                  <th className="px-4 py-3 text-left">Frequency</th>
                  <th className="px-4 py-3 text-left">Reference Impedance</th>
                  <th className="px-4 py-3 text-left">Selected Impedance</th>
                  <th className="px-4 py-3 text-left">Resnorm</th>
                  <th className="px-4 py-3 text-left">Error %</th>
                </tr>
              </thead>
              <tbody>
                {impedanceData.map((point, index) => (
                  <tr
                    key={point.frequency}
                    className={`border-b border-neutral-700 hover:bg-neutral-800/50 transition-colors ${
                      index % 2 === 0 ? 'bg-neutral-900' : 'bg-neutral-800/20'
                    }`}
                    onClick={() => setShowDetailsFor(showDetailsFor === index ? null : index)}
                  >
                    <td className="px-4 py-3 font-mono">
                      {formatNumber(point.frequency, point.frequency < 1 ? 3 : 1)} Hz
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-mono text-sm">
                        <div>{formatNumber(point.referenceImpedance.magnitude)} Ω</div>
                        {(showDetailsFor === index || showAdvanced) && (
                          <div className="text-xs text-neutral-400">
                            <div>Real: {formatNumber(point.referenceImpedance.real)} Ω</div>
                            <div>Imag: {formatNumber(point.referenceImpedance.imag)} Ω</div>
                            <div>Phase: {formatNumber(Math.atan2(point.referenceImpedance.imag, point.referenceImpedance.real) * 180 / Math.PI, 1)}°</div>
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-mono text-sm">
                        <div>{formatNumber(point.selectedImpedance.magnitude)} Ω</div>
                        {(showDetailsFor === index || showAdvanced) && (
                          <div className="text-xs text-neutral-400">
                            <div>Real: {formatNumber(point.selectedImpedance.real)} Ω</div>
                            <div>Imag: {formatNumber(point.selectedImpedance.imag)} Ω</div>
                            <div>Phase: {formatNumber(Math.atan2(point.selectedImpedance.imag, point.selectedImpedance.real) * 180 / Math.PI, 1)}°</div>
                          </div>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`font-mono ${
                        point.resnorm < 10 ? 'text-green-400' :
                        point.resnorm < 50 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {formatNumber(point.resnorm)}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`font-mono ${
                        point.percentError < 5 ? 'text-green-400' :
                        point.percentError < 20 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {formatNumber(point.percentError, 1)}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Enhanced summary footer */}
      {impedanceData.length > 0 && (
        <div className="p-3 bg-neutral-800 border-t border-neutral-700 text-xs text-neutral-400">
          <div className="flex justify-between items-center">
            <span>
              Avg Resnorm: <span className="text-neutral-200 font-mono">
                {formatNumber(impedanceData.reduce((sum, p) => sum + p.resnorm, 0) / impedanceData.length)}
              </span>
            </span>
            <span>
              Max Error: <span className="text-neutral-200 font-mono">
                {formatNumber(Math.max(...impedanceData.map(p => p.percentError)))}%
              </span>
            </span>
            <span>
              Frequency Range: <span className="text-neutral-200 font-mono">
                {formatNumber(Math.min(...impedanceData.map(p => p.frequency)))} - {formatNumber(Math.max(...impedanceData.map(p => p.frequency)))} Hz
              </span>
            </span>
            <span>Click rows for details • Hover Export for options</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default EnhancedImpedanceComparisonTab;

/*
 * INSTALLATION NOTES:
 *
 * 1. This component requires @heroicons/react for icons:
 *    npm install @heroicons/react
 *
 * 2. The component uses TailwindCSS for styling
 *
 * 3. Export functionality creates downloadable CSV and JSON files
 *
 * 4. Advanced features include:
 *    - Custom frequency range generation (linear/logarithmic)
 *    - Enhanced impedance calculations with phase information
 *    - Professional data export with metadata
 *    - Interactive UI with PyCharm-style design
 *
 * 5. Integration example:
 *    Replace the existing ImpedanceComparisonTab import with this enhanced version
 */
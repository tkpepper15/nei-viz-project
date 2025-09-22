/**
 * Enhanced Impedance Comparison Tab
 * ================================
 *
 * Features:
 * - Individual real and imaginary impedance columns
 * - Add new frequency row functionality
 * - Auto-selection from 3D plot hover/selection
 * - Dropdown with search and five number summary
 * - Streamlined UI without details/advanced buttons
 */

import React, { useMemo, useState, useCallback } from 'react';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';
import { PlusIcon } from '@heroicons/react/24/outline';

interface ImpedanceDataPoint {
  frequency: number;
  referenceReal: number;
  referenceImag: number;
  selectedReal: number;
  selectedImag: number;
  referenceMagnitude: number;
  selectedMagnitude: number;
  resnorm: number;
  isCustomFrequency?: boolean; // Flag for user-added frequencies
  uniqueKey?: string; // Unique key for React rendering
}

export const ImpedanceComparisonTab: React.FC<BottomPanelTabProps> = ({
  topConfigurations,
  currentParameters,
  selectedConfigIndex,
  isVisible,
  highlightedModelId
}) => {
  // State for table functionality
  const [sortColumn, setSortColumn] = useState<'frequency' | 'resnorm'>('frequency');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [customFrequencies, setCustomFrequencies] = useState<number[]>([]);
  const [newFrequency, setNewFrequency] = useState('');
  const [showAddFrequency, setShowAddFrequency] = useState(false);

  // Get selected model from right bar configuration
  const selectedModel = useMemo(() => {
    if (!topConfigurations || topConfigurations.length === 0) return null;
    if (selectedConfigIndex >= topConfigurations.length) return null;

    const config = topConfigurations[selectedConfigIndex];
    return config?.items?.[0] || null; // Use first item in the configuration
  }, [topConfigurations, selectedConfigIndex]);

  // Default frequency points (5 log-spaced values for minimal yet powerful display)
  const defaultFrequencies = useMemo(() => {
    // Logarithmic spacing: 0.1 Hz, 1 Hz, 10 Hz, 100 Hz, 1 kHz
    return [0.1, 1, 10, 100, 1000];
  }, []);

  // Combined frequency list (default + custom)
  const allFrequencies = useMemo(() => {
    return [...defaultFrequencies, ...customFrequencies].sort((a, b) => a - b);
  }, [defaultFrequencies, customFrequencies]);

  // Calculate impedance for given frequency and parameters (Randles model)
  const calculateImpedance = useCallback((frequency: number, params: { Ra: number; Rb: number; Ca: number; Cb: number; Rsh: number }) => {
    const omega = 2 * Math.PI * frequency;

    // Za = Ra/(1+jωRaCa)
    const za_denom_real = 1;
    const za_denom_imag = omega * params.Ra * params.Ca;
    const za_denom_mag_sq = za_denom_real * za_denom_real + za_denom_imag * za_denom_imag;
    const za_real = params.Ra * za_denom_real / za_denom_mag_sq;
    const za_imag = -params.Ra * za_denom_imag / za_denom_mag_sq;

    // Zb = Rb/(1+jωRbCb)
    const zb_denom_real = 1;
    const zb_denom_imag = omega * params.Rb * params.Cb;
    const zb_denom_mag_sq = zb_denom_real * zb_denom_real + zb_denom_imag * zb_denom_imag;
    const zb_real = params.Rb * zb_denom_real / zb_denom_mag_sq;
    const zb_imag = -params.Rb * zb_denom_imag / zb_denom_mag_sq;

    // Z_total = (Rsh * (Za + Zb)) / (Rsh + Za + Zb)
    const zaZb_real = za_real + zb_real;
    const zaZb_imag = za_imag + zb_imag;

    const num_real = params.Rsh * zaZb_real;
    const num_imag = params.Rsh * zaZb_imag;

    const denom_real = params.Rsh + zaZb_real;
    const denom_imag = zaZb_imag;
    const denom_mag_sq = denom_real * denom_real + denom_imag * denom_imag;

    const real = (num_real * denom_real + num_imag * denom_imag) / denom_mag_sq;
    const imag = (num_imag * denom_real - num_real * denom_imag) / denom_mag_sq;
    const magnitude = Math.sqrt(real * real + imag * imag);

    return { real, imag, magnitude };
  }, []);

  // Reference model for comparison
  const referenceModel = currentParameters;

  // Handler for adding new frequency
  const handleAddFrequency = useCallback(() => {
    const freq = parseFloat(newFrequency);
    if (!isNaN(freq) && freq > 0 && !allFrequencies.includes(freq)) {
      setCustomFrequencies(prev => [...prev, freq].sort((a, b) => a - b));
      setNewFrequency('');
      setShowAddFrequency(false);
    }
  }, [newFrequency, allFrequencies]);

  // Calculate impedance data for table
  const impedanceData = useMemo<ImpedanceDataPoint[]>(() => {
    if (!selectedModel || !referenceModel) return [];

    // Generate session timestamp for unique keys
    const sessionId = Date.now().toString(36);

    return allFrequencies.map((frequency, freqIndex) => {
      const refImpedance = calculateImpedance(frequency, referenceModel);
      const selImpedance = calculateImpedance(frequency, selectedModel.parameters);

      // Calculate resnorm for this frequency point
      const realDiff = refImpedance.real - selImpedance.real;
      const imagDiff = refImpedance.imag - selImpedance.imag;
      const resnorm = Math.sqrt(realDiff * realDiff + imagDiff * imagDiff);

      return {
        frequency,
        referenceReal: refImpedance.real,
        referenceImag: refImpedance.imag,
        selectedReal: selImpedance.real,
        selectedImag: selImpedance.imag,
        referenceMagnitude: refImpedance.magnitude,
        selectedMagnitude: selImpedance.magnitude,
        resnorm,
        isCustomFrequency: customFrequencies.includes(frequency),
        // Add unique key for React rendering
        uniqueKey: `${selectedModel.id || 'model'}-freq-${frequency}-${sessionId}-${freqIndex}`
      };
    });
  }, [selectedModel, referenceModel, allFrequencies, customFrequencies, calculateImpedance]);

  // Sorted impedance data
  const sortedData = useMemo(() => {
    if (!impedanceData.length) return [];

    const sorted = [...impedanceData].sort((a, b) => {
      const aVal = sortColumn === 'frequency' ? a.frequency : a.resnorm;
      const bVal = sortColumn === 'frequency' ? b.frequency : b.resnorm;
      return sortDirection === 'asc' ? aVal - bVal : bVal - aVal;
    });

    return sorted;
  }, [impedanceData, sortColumn, sortDirection]);

  // Get min/max values for color coding
  const getMinMaxValues = useCallback((data: ImpedanceDataPoint[]) => {
    if (data.length === 0) return null;

    const realValues = [...data.map(d => d.referenceReal), ...data.map(d => d.selectedReal)];
    const imagValues = [...data.map(d => d.referenceImag), ...data.map(d => d.selectedImag)];
    const magnitudeValues = [...data.map(d => d.referenceMagnitude), ...data.map(d => d.selectedMagnitude)];
    const resnormValues = data.map(d => d.resnorm);

    return {
      realMin: Math.min(...realValues),
      realMax: Math.max(...realValues),
      imagMin: Math.min(...imagValues),
      imagMax: Math.max(...imagValues),
      magnitudeMin: Math.min(...magnitudeValues),
      magnitudeMax: Math.max(...magnitudeValues),
      resnormMin: Math.min(...resnormValues),
      resnormMax: Math.max(...resnormValues)
    };
  }, []);

  // Calculate min/max for color coding
  const minMaxValues = useMemo(() => getMinMaxValues(sortedData), [sortedData, getMinMaxValues]);

  // Get color class for value based on min/max
  const getValueColorClass = useCallback((value: number, min: number, max: number, isResnorm = false) => {
    if (min === max) return '';

    const isMin = Math.abs(value - min) < Math.abs(value - max);
    if (isResnorm) {
      // For resnorm, lower is better (green for min, red for max)
      return isMin ? 'bg-green-900/30 text-green-200' : 'bg-red-900/30 text-red-200';
    } else {
      // For impedance values, just highlight extremes
      return isMin ? 'bg-blue-900/30 text-blue-200' : 'bg-orange-900/30 text-orange-200';
    }
  }, []);

  if (!isVisible) return null;

  return (
    <div className="flex-1 overflow-y-auto min-h-0 bg-neutral-900">
      <div className="p-4 space-y-4">
        {/* Circuit Selection Integration Notice */}
        {selectedModel && (
          <div className="bg-blue-900/20 rounded-lg p-3 border border-blue-700/30">
            <div className="text-xs text-blue-200">
              <span className="font-medium">Selected from right panel:</span> {selectedModel.name}
              {selectedModel.resnorm && (
                <span className="ml-2">• Resnorm: {selectedModel.resnorm.toExponential(3)}</span>
              )}
              {highlightedModelId === selectedModel.id && (
                <span className="ml-2 px-2 py-0.5 bg-blue-600 text-blue-100 rounded text-xs">Highlighted in 3D</span>
              )}
            </div>
          </div>
        )}

        {/* Impedance Comparison Table */}
        {selectedModel && (
          <div className="bg-neutral-800 rounded-lg overflow-hidden">
            <div className="p-4 border-b border-neutral-700">
              <h3 className="text-sm font-medium text-neutral-200">Impedance Comparison Table</h3>
              <p className="text-xs text-neutral-400 mt-1">
                Reference vs {selectedModel.name} • {sortedData.length} frequency points
              </p>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-neutral-700 text-neutral-300">
                  <tr>
                    <th
                      className="px-3 py-2 text-left cursor-pointer hover:bg-neutral-600"
                      onClick={() => {
                        if (sortColumn === 'frequency') {
                          setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                        } else {
                          setSortColumn('frequency');
                          setSortDirection('asc');
                        }
                      }}
                    >
                      Frequency (Hz) {sortColumn === 'frequency' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                    <th className="px-3 py-2 text-left">Ref Real (Ω)</th>
                    <th className="px-3 py-2 text-left">Ref Imag (Ω)</th>
                    <th className="px-3 py-2 text-left">Ref |Z| (Ω)</th>
                    <th className="px-3 py-2 text-left">Sel Real (Ω)</th>
                    <th className="px-3 py-2 text-left">Sel Imag (Ω)</th>
                    <th className="px-3 py-2 text-left">Sel |Z| (Ω)</th>
                    <th
                      className="px-3 py-2 text-left cursor-pointer hover:bg-neutral-600"
                      onClick={() => {
                        if (sortColumn === 'resnorm') {
                          setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
                        } else {
                          setSortColumn('resnorm');
                          setSortDirection('asc');
                        }
                      }}
                    >
                      Resnorm {sortColumn === 'resnorm' && (sortDirection === 'asc' ? '↑' : '↓')}
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedData.map((point, index) => (
                    <tr
                      key={point.uniqueKey || `fallback-${selectedModel?.id || 'ref'}-${point.frequency}-${index}`}
                      className={`border-t border-neutral-700 hover:bg-neutral-700/50 ${
                        point.isCustomFrequency ? 'bg-blue-900/20' : ''
                      }`}
                    >
                      <td className="px-3 py-2 font-mono">
                        {point.frequency < 1 ? point.frequency.toFixed(3) :
                         point.frequency < 1000 ? point.frequency.toFixed(1) :
                         (point.frequency / 1000).toFixed(1) + 'k'}
                        {point.isCustomFrequency && <span className="ml-1 text-blue-400">*</span>}
                      </td>
                      <td className={`px-3 py-2 font-mono ${minMaxValues ? getValueColorClass(point.referenceReal, minMaxValues.realMin, minMaxValues.realMax) : ''}`}>
                        {point.referenceReal.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono ${minMaxValues ? getValueColorClass(point.referenceImag, minMaxValues.imagMin, minMaxValues.imagMax) : ''}`}>
                        {point.referenceImag.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono ${minMaxValues ? getValueColorClass(point.referenceMagnitude, minMaxValues.magnitudeMin, minMaxValues.magnitudeMax) : ''}`}>
                        {point.referenceMagnitude.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono ${minMaxValues ? getValueColorClass(point.selectedReal, minMaxValues.realMin, minMaxValues.realMax) : ''}`}>
                        {point.selectedReal.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono ${minMaxValues ? getValueColorClass(point.selectedImag, minMaxValues.imagMin, minMaxValues.imagMax) : ''}`}>
                        {point.selectedImag.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono ${minMaxValues ? getValueColorClass(point.selectedMagnitude, minMaxValues.magnitudeMin, minMaxValues.magnitudeMax) : ''}`}>
                        {point.selectedMagnitude.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono text-xs ${minMaxValues ? getValueColorClass(point.resnorm, minMaxValues.resnormMin, minMaxValues.resnormMax, true) : ''}`}>
                        {point.resnorm.toExponential(2)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Inline Add Frequency Row */}
            <div className="border-t border-neutral-700">
              {!showAddFrequency ? (
                <button
                  onClick={() => setShowAddFrequency(true)}
                  className="w-full p-3 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50 transition-colors flex items-center justify-center gap-2 text-sm"
                >
                  <PlusIcon className="w-4 h-4" />
                  Add frequency point
                </button>
              ) : (
                <div className="p-3 flex gap-2">
                  <input
                    type="number"
                    placeholder="Frequency (Hz)"
                    value={newFrequency}
                    onChange={(e) => setNewFrequency(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddFrequency();
                      if (e.key === 'Escape') {
                        setShowAddFrequency(false);
                        setNewFrequency('');
                      }
                    }}
                    className="flex-1 px-3 py-2 bg-neutral-700 border border-neutral-600 rounded-md text-neutral-200 placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                    autoFocus
                  />
                  <button
                    onClick={handleAddFrequency}
                    disabled={!newFrequency || isNaN(parseFloat(newFrequency))}
                    className="px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-neutral-600 disabled:cursor-not-allowed text-white rounded-md transition-colors text-sm"
                  >
                    Add
                  </button>
                  <button
                    onClick={() => {
                      setShowAddFrequency(false);
                      setNewFrequency('');
                    }}
                    className="px-3 py-2 bg-neutral-600 hover:bg-neutral-500 text-white rounded-md transition-colors text-sm"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>

            {customFrequencies.length > 0 && (
              <div className="p-3 bg-neutral-700/50 border-t border-neutral-700 text-xs text-blue-300">
                * Custom frequency points: {customFrequencies.map(f => f.toFixed(2)).join(', ')} Hz
              </div>
            )}
          </div>
        )}

        {!selectedModel && (
          <div className="text-center py-8 text-neutral-400">
            <p>Select a circuit model to view impedance comparison</p>
          </div>
        )}
      </div>
    </div>
  );
};

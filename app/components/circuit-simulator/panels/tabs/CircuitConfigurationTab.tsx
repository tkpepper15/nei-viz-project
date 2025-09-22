/**
 * Circuit Configuration Tab
 * =========================
 *
 * Displays the current circuit parameter configuration matrix and grid information.
 * Provides a detailed view of the parameter space and current selections.
 */

import React, { useMemo, useState } from 'react';
import { BottomPanelTabProps } from '../CollapsibleBottomPanel';
import { ConfigSerializer } from '../../utils/configSerializer';

export const CircuitConfigurationTab: React.FC<BottomPanelTabProps> = ({
  gridResults,
  topConfigurations,
  currentParameters,
  selectedConfigIndex,
  onConfigurationSelect,
  isVisible,
  gridSize = 9  // Add gridSize prop with default value
  // highlightedModelId - Not used in this tab
}) => {
  const [activeSection, setActiveSection] = useState<'current' | 'grid' | 'selected'>('current');

  // Get configuration serializer for current grid size
  const configSerializer = useMemo(() => {
    // Use the passed gridSize prop instead of calculating from results
    return new ConfigSerializer(gridSize || 9);
  }, [gridSize]);

  // Get all parameter values from the grid
  const allParameterValues = useMemo(() => {
    return configSerializer.getAllParameterValues();
  }, [configSerializer]);

  // Get selected configuration details
  const selectedConfiguration = useMemo(() => {
    if (selectedConfigIndex >= 0 && selectedConfigIndex < topConfigurations.length) {
      const config = topConfigurations[selectedConfigIndex];
      return config.items?.[0] || null;
    }
    return null;
  }, [topConfigurations, selectedConfigIndex]);

  // Format parameter values for display
  const formatParameter = (param: string, value: number) => {
    switch (param) {
      case 'ca':
      case 'cb':
        return `${(value * 1e6).toFixed(2)} μF`;
      case 'rsh':
      case 'ra':
      case 'rb':
        return `${value.toFixed(1)} Ω`;
      default:
        return value.toFixed(3);
    }
  };

  // Calculate grid statistics
  const gridStats = useMemo(() => {
    const totalConfigs = configSerializer.getTotalConfigurations();
    const computedConfigs = gridResults.length;
    const completionPercentage = (computedConfigs / totalConfigs) * 100;

    return {
      totalConfigs,
      computedConfigs,
      completionPercentage,
      gridSize: configSerializer.gridSize
    };
  }, [configSerializer, gridResults]);

  if (!isVisible) {
    return <div className="h-full flex items-center justify-center text-neutral-500">
      Loading configuration data...
    </div>;
  }

  return (
    <div className="flex flex-col bg-neutral-900 text-white">
      {/* Section tabs */}
      <div className="flex bg-neutral-800 border-b border-neutral-700">
        {[
          { id: 'current', label: 'Current Parameters' },
          { id: 'grid', label: 'Parameter Grid' },
          { id: 'selected', label: 'Selected Config' }
        ].map(section => (
          <button
            key={section.id}
            onClick={() => setActiveSection(section.id as 'current' | 'grid' | 'selected')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeSection === section.id
                ? 'border-blue-500 text-blue-400 bg-neutral-700'
                : 'border-transparent text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700/50'
            }`}
          >
            {section.label}
          </button>
        ))}
      </div>

      {/* Content area - Enhanced scrollable container */}
      <div className="p-4">
        {activeSection === 'current' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-neutral-200">Current Circuit Parameters</h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Parameters table */}
              <div className="bg-neutral-800 rounded-lg p-4">
                <h4 className="text-sm font-medium text-neutral-300 mb-3">Circuit Values</h4>
                <div className="space-y-2">
                  {[
                    { key: 'Rsh', label: 'Shunt Resistance', value: currentParameters.Rsh, unit: 'Ω' },
                    { key: 'Ra', label: 'Apical Resistance', value: currentParameters.Ra, unit: 'Ω' },
                    { key: 'Ca', label: 'Apical Capacitance', value: currentParameters.Ca * 1e6, unit: 'μF' },
                    { key: 'Rb', label: 'Basal Resistance', value: currentParameters.Rb, unit: 'Ω' },
                    { key: 'Cb', label: 'Basal Capacitance', value: currentParameters.Cb * 1e6, unit: 'μF' }
                  ].map(param => (
                    <div key={param.key} className="flex justify-between items-center">
                      <span className="text-neutral-400 text-sm">{param.label}</span>
                      <span className="font-mono text-neutral-200">
                        {param.value.toFixed(param.unit === 'μF' ? 2 : 1)} {param.unit}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Frequency configuration */}
              <div className="bg-neutral-800 rounded-lg p-4">
                <h4 className="text-sm font-medium text-neutral-300 mb-3">Frequency Configuration</h4>
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-neutral-400 text-sm">Min Frequency</span>
                    <span className="font-mono text-neutral-200">
                      {currentParameters.frequency_range?.[0] || 0.1} Hz
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-neutral-400 text-sm">Max Frequency</span>
                    <span className="font-mono text-neutral-200">
                      {currentParameters.frequency_range?.[1] || 100000} Hz
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-neutral-400 text-sm">Frequency Span</span>
                    <span className="font-mono text-neutral-200">
                      {((currentParameters.frequency_range?.[1] || 100000) / (currentParameters.frequency_range?.[0] || 0.1)).toFixed(0)}x
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeSection === 'grid' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="text-lg font-semibold text-neutral-200">Parameter Grid Configuration</h3>
              <div className="text-sm text-neutral-400">
                Grid Size: {gridStats.gridSize}×{gridStats.gridSize}×{gridStats.gridSize}×{gridStats.gridSize}×{gridStats.gridSize}
              </div>
            </div>

            {/* Grid statistics */}
            <div className="bg-neutral-800 rounded-lg p-4">
              <h4 className="text-sm font-medium text-neutral-300 mb-3">Grid Statistics</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-mono text-blue-400">{gridStats.totalConfigs.toLocaleString()}</div>
                  <div className="text-xs text-neutral-400">Total Configs</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-mono text-green-400">{gridStats.computedConfigs.toLocaleString()}</div>
                  <div className="text-xs text-neutral-400">Computed</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-mono text-yellow-400">{gridStats.completionPercentage.toFixed(1)}%</div>
                  <div className="text-xs text-neutral-400">Complete</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-mono text-purple-400">{gridStats.gridSize}</div>
                  <div className="text-xs text-neutral-400">Grid Size</div>
                </div>
              </div>
            </div>

            {/* Parameter ranges */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {Object.entries(allParameterValues).map(([paramKey, values]) => (
                <div key={paramKey} className="bg-neutral-800 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3 capitalize">
                    {paramKey} Range ({values.length} points)
                  </h4>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {values.map((point, index) => (
                      <div key={index} className="flex justify-between items-center text-xs">
                        <span className="text-neutral-400">#{point.index}</span>
                        <span className="font-mono text-neutral-200">
                          {formatParameter(paramKey, point.value)}
                        </span>
                      </div>
                    ))}
                  </div>
                  <div className="mt-2 pt-2 border-t border-neutral-700 text-xs text-neutral-500">
                    Range: {formatParameter(paramKey, values[0].value)} → {formatParameter(paramKey, values[values.length - 1].value)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeSection === 'selected' && (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold text-neutral-200">Selected Configuration Details</h3>

            {selectedConfiguration ? (
              <div className="space-y-4">
                {/* Configuration selector */}
                <div className="bg-neutral-800 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Configuration Selection</h4>
                  <select
                    value={selectedConfigIndex}
                    onChange={(e) => onConfigurationSelect(Number(e.target.value))}
                    className="w-full bg-neutral-700 border border-neutral-600 rounded px-3 py-2 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                  >
                    {topConfigurations.map((config, index) => (
                      <option key={index} value={index}>
                        #{index + 1} - Range: {config.range[0].toFixed(4)} - {config.range[1].toFixed(4)}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Selected parameters */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="bg-neutral-800 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-neutral-300 mb-3">Selected Parameters</h4>
                    <div className="space-y-2">
                      {selectedConfiguration.parameters && Object.entries(selectedConfiguration.parameters).map(([key, value]) => {
                        if (key === 'frequency_range') return null;
                        return (
                          <div key={key} className="flex justify-between items-center">
                            <span className="text-neutral-400 text-sm capitalize">{key}</span>
                            <span className="font-mono text-neutral-200">
                              {formatParameter(key.toLowerCase(), value as number)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                  {/* Performance metrics */}
                  <div className="bg-neutral-800 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-neutral-300 mb-3">Performance Metrics</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-neutral-400 text-sm">Resnorm</span>
                        <span className="font-mono text-green-400">
                          {(selectedConfiguration.resnorm || 0).toFixed(6)}
                        </span>
                      </div>
                      {selectedConfiguration.data && (
                        <div className="flex justify-between items-center">
                          <span className="text-neutral-400 text-sm">Data Points</span>
                          <span className="font-mono text-neutral-200">
                            {selectedConfiguration.data.length}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Configuration comparison */}
                <div className="bg-neutral-800 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-neutral-300 mb-3">Comparison with Current</h4>
                  <div className="space-y-2">
                    {selectedConfiguration.parameters && Object.entries(selectedConfiguration.parameters).map(([key, value]) => {
                      if (key === 'frequency_range') return null;
                      const currentValue = currentParameters[key as keyof typeof currentParameters] as number;
                      const difference = ((value as number) - currentValue) / currentValue * 100;

                      return (
                        <div key={key} className="flex justify-between items-center">
                          <span className="text-neutral-400 text-sm capitalize">{key}</span>
                          <div className="flex items-center space-x-2">
                            <span className="font-mono text-neutral-200 text-xs">
                              {formatParameter(key.toLowerCase(), value as number)}
                            </span>
                            <span className={`font-mono text-xs px-2 py-1 rounded ${
                              Math.abs(difference) < 5 ? 'bg-green-900 text-green-300' :
                              Math.abs(difference) < 20 ? 'bg-yellow-900 text-yellow-300' :
                              'bg-red-900 text-red-300'
                            }`}>
                              {difference > 0 ? '+' : ''}{difference.toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-neutral-500 py-8">
                No configuration selected. Choose a configuration from the dropdown above.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CircuitConfigurationTab;